// FastText: train, predict, quantize, autotune, save/load, word/sentence vectors

use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;

use crate::args::{Args, LossName, ModelName};
use crate::dictionary::{Dictionary, EntryType, EOS};
use crate::error::{FastTextError, Result};
use crate::loss::{HierarchicalSoftmaxLoss, Loss, NegativeSamplingLoss, OneVsAllLoss, SoftmaxLoss};
use crate::matrix::{DenseMatrix, Matrix};
use crate::model::{Model, State};
use crate::utils;

/// Magic number identifying a valid fastText binary model file.
pub const FASTTEXT_FILEFORMAT_MAGIC_INT32: i32 = 793712314;
/// Current binary format version.
pub const FASTTEXT_VERSION: i32 = 12;

/// A single prediction result.
///
/// Contains the predicted label string and its probability (in `[0, 1]`).
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Probability of this label (exponentiated from log-probability).
    pub prob: f32,
    /// The label string (e.g. `"__label__baking"`).
    pub label: String,
}

/// Read a boolean (1 byte) from a reader.
fn read_bool<R: Read>(reader: &mut R) -> Result<bool> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
}

/// Write a boolean (1 byte) to a writer.
fn write_bool<W: Write>(writer: &mut W, value: bool) -> Result<()> {
    writer.write_all(&[value as u8])?;
    Ok(())
}

/// Build the appropriate loss function based on `args.loss()`.
///
/// - `SoftmaxLoss` — full softmax (used for supervised models with softmax loss)
/// - `NegativeSamplingLoss` — negative sampling (skipgram/CBOW)
/// - `HierarchicalSoftmaxLoss` — Huffman-tree hierarchical softmax
/// - `OneVsAllLoss` — one-vs-all binary logistic
fn build_loss(args: &Args, wo: Arc<DenseMatrix>, target_counts: &[i64]) -> Box<dyn Loss> {
    match args.loss() {
        LossName::HS => Box::new(HierarchicalSoftmaxLoss::new(wo, target_counts)),
        LossName::NS => Box::new(NegativeSamplingLoss::new(wo, args.neg(), target_counts)),
        LossName::OVA => Box::new(OneVsAllLoss::new(wo)),
        LossName::SOFTMAX => Box::new(SoftmaxLoss::new(wo)),
    }
}

/// A loaded fastText model.
///
/// Contains the model arguments, dictionary, input matrix, output matrix,
/// and a pre-built `Model` for efficient inference.
#[derive(Debug)]
pub struct FastText {
    /// Model hyperparameters.
    args: Arc<Args>,
    /// The vocabulary dictionary.
    dict: Dictionary,
    /// Input embedding matrix (word + subword vectors), shared via Arc for Model.
    input: Arc<DenseMatrix>,
    /// Output matrix (label/word vectors), shared via Arc for Model.
    output: Arc<DenseMatrix>,
    /// Whether the model uses quantized (QuantMatrix) input.
    quant: bool,
    /// Pre-built inference model.
    model: Model,
}

impl FastText {
    /// Load a model from a binary reader.
    ///
    /// Reads in this order:
    /// 1. Magic number (i32) – must be 793712314
    /// 2. Version (i32) – must be ≤ 12
    /// 3. Args block (56 bytes)
    /// 4. Dictionary block
    /// 5. quant_input (bool)
    /// 6. Input matrix (DenseMatrix if not quantized)
    /// 7. qout (bool → stored in args.qout)
    /// 8. Output matrix
    pub fn load<R: Read>(reader: &mut R) -> Result<Self> {
        // 1. Read and validate magic number
        let magic = utils::read_i32(reader)?;
        if magic != FASTTEXT_FILEFORMAT_MAGIC_INT32 {
            return Err(FastTextError::InvalidModel(format!(
                "Invalid magic number: {} (expected {})",
                magic, FASTTEXT_FILEFORMAT_MAGIC_INT32
            )));
        }

        // 2. Read and validate version
        let version = utils::read_i32(reader)?;
        if version > FASTTEXT_VERSION {
            return Err(FastTextError::InvalidModel(format!(
                "Unsupported version: {} (maximum supported: {})",
                version, FASTTEXT_VERSION
            )));
        }

        // 3. Read Args block
        let mut args = Args::default();
        args.load(reader)?;

        // Version 11 backward compatibility:
        // Old supervised models do not use character n-grams.
        if version == 11 && args.model() == ModelName::SUP {
            args.set_maxn(0);
        }

        // 4. Read Dictionary block
        let args_arc = Arc::new(args.clone());
        let dict = Dictionary::load_from_reader(reader, args_arc)?;

        // 5. Read quant_input flag
        let quant_input = read_bool(reader)?;

        // 6. Load input matrix
        let input = if !quant_input {
            DenseMatrix::load(reader)?
        } else {
            // QuantMatrix is deferred to a future feature.
            return Err(FastTextError::InvalidModel(
                "Quantized input matrix (.ftz) is not yet supported in this version".to_string(),
            ));
        };

        // C++ check: if not quantized but dict is pruned, reject
        if !quant_input && dict.is_pruned() {
            return Err(FastTextError::InvalidModel(
                "Invalid model file. Please download the updated model. \
                 See issue #332 on Github for more information."
                    .to_string(),
            ));
        }

        // 7. Read qout flag (stored in args)
        let qout = read_bool(reader)?;
        args.set_qout(qout);

        // 8. Load output matrix
        let output = if quant_input && qout {
            // QuantMatrix output – deferred
            return Err(FastTextError::InvalidModel(
                "Quantized output matrix (.ftz) is not yet supported".to_string(),
            ));
        } else {
            DenseMatrix::load(reader)?
        };

        // 9. Build the inference model.
        let input_arc = Arc::new(input);
        let output_arc = Arc::new(output);
        let label_counts = dict.get_counts(EntryType::Label);
        let word_counts = dict.get_counts(EntryType::Word);
        let target_counts = if args.model() == ModelName::SUP {
            label_counts
        } else {
            word_counts
        };
        let loss = build_loss(&args, Arc::clone(&output_arc), &target_counts);
        let normalize_gradient = args.model() == ModelName::SUP;
        let model = Model::new(Arc::clone(&input_arc), loss, normalize_gradient);

        Ok(FastText {
            args: Arc::new(args),
            dict,
            input: input_arc,
            output: output_arc,
            quant: quant_input,
            model,
        })
    }

    /// Load a model from a file path.
    pub fn load_model(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(FastTextError::IoError)?;
        let mut reader = BufReader::new(file);
        Self::load(&mut reader)
    }

    /// Save the model to a binary writer.
    ///
    /// Writes in the same order as `load()`:
    /// 1. Magic number (i32)
    /// 2. Version (i32)
    /// 3. Args block (56 bytes)
    /// 4. Dictionary block
    /// 5. quant_input (bool)
    /// 6. Input matrix
    /// 7. qout (bool)
    /// 8. Output matrix
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        // 1. Magic number
        utils::write_i32(writer, FASTTEXT_FILEFORMAT_MAGIC_INT32)?;
        // 2. Version
        utils::write_i32(writer, FASTTEXT_VERSION)?;
        // 3. Args block
        self.args.save(writer)?;
        // 4. Dictionary block
        self.dict.save(writer)?;
        // 5. quant_input
        write_bool(writer, self.quant)?;
        // 6. Input matrix
        self.input.save(writer)?;
        // 7. qout
        write_bool(writer, self.args.qout())?;
        // 8. Output matrix
        self.output.save(writer)?;
        Ok(())
    }

    /// Save the model to a file path.
    pub fn save_model(&self, path: &str) -> Result<()> {
        let file = std::fs::File::create(path).map_err(FastTextError::IoError)?;
        let mut writer = BufWriter::new(file);
        self.save(&mut writer)?;
        // Explicitly flush before dropping so buffered-write errors are caught.
        writer.flush().map_err(FastTextError::IoError)?;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Getters / accessors
    // -------------------------------------------------------------------------

    /// Get a reference to the model arguments.
    pub fn args(&self) -> &Args {
        &self.args
    }

    /// Get a reference to the dictionary.
    pub fn dict(&self) -> &Dictionary {
        &self.dict
    }

    /// Get a reference to the input matrix.
    pub fn input_matrix(&self) -> &DenseMatrix {
        &self.input
    }

    /// Get a reference to the output matrix.
    pub fn output_matrix(&self) -> &DenseMatrix {
        &self.output
    }

    /// Return whether the model is quantized.
    pub fn is_quant(&self) -> bool {
        self.quant
    }

    /// Return the word vector for a given word.
    ///
    /// For in-vocabulary words the vector is the average of all stored subword
    /// IDs (which for `minn=0 / maxn=0` is just the word's own row).  For
    /// OOV words the subwords are computed on-the-fly; if there are no subwords
    /// (e.g. `bucket=0`) a zero vector is returned.
    pub fn get_word_vector(&self, word: &str) -> Vec<f32> {
        let dim = self.args.dim() as usize;
        let mut result = vec![0.0f32; dim];
        let ids = self.dict.get_subwords_for_string(word);
        if ids.is_empty() {
            return result;
        }
        let scale = 1.0 / ids.len() as f32;
        for &id in &ids {
            let row = self.input.row(id as i64);
            for (r, &v) in result.iter_mut().zip(row.iter()) {
                *r += v * scale;
            }
        }
        result
    }

    /// Compute the sentence vector for the given text.
    ///
    /// For supervised models: tokenizes with the dictionary (including subword IDs),
    /// sums all input rows, and averages (raw, no L2 normalization).
    ///
    /// For unsupervised models (CBOW/SG): splits on whitespace, gets word vector
    /// for each word, L2-normalizes each (if norm > 0), sums and averages.
    ///
    /// Empty input returns a zero vector.
    pub fn get_sentence_vector(&self, sentence: &str) -> Vec<f32> {
        let dim = self.args.dim() as usize;
        let mut result = vec![0.0f32; dim];

        if self.args.model() == ModelName::SUP {
            // Supervised: use dictionary tokenization (subwords included).
            // Append EOS to match C++ getSentenceVector behavior: the C++
            // stream-based getLine includes an EOS token produced by the
            // trailing newline, so we explicitly append it here.
            let mut words: Vec<i32> = Vec::new();
            let mut labels: Vec<i32> = Vec::new();
            self.dict.get_line_from_str(sentence, &mut words, &mut labels);

            // Return zero vector for empty/whitespace-only input.
            if words.is_empty() {
                return result;
            }

            // Append EOS to match C++ stream-based getLine behavior (trailing newline → EOS).
            let eos_id = self.dict.get_id(EOS);
            if eos_id >= 0 {
                words.push(eos_id);
            }

            let count = words.len() as f32;
            for &id in &words {
                let row = self.input.row(id as i64);
                for (r, &v) in result.iter_mut().zip(row.iter()) {
                    *r += v;
                }
            }
            for r in &mut result {
                *r /= count;
            }
        } else {
            // Unsupervised: split whitespace, get word vector, L2-normalize, average
            let mut count = 0i32;
            for word in sentence.split_whitespace() {
                let vec = self.get_word_vector(word);
                let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for (r, &v) in result.iter_mut().zip(vec.iter()) {
                        *r += v / norm;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                let scale = 1.0 / count as f32;
                for r in &mut result {
                    *r *= scale;
                }
            }
        }
        result
    }

    /// Tokenize text by splitting on whitespace.
    ///
    /// Returns an empty vec for empty/whitespace-only input.
    /// Unicode tokens are preserved intact.
    /// Multiple consecutive whitespace characters are collapsed.
    /// Leading/trailing whitespace is ignored.
    pub fn tokenize(text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_string()).collect()
    }

    /// Return the vocabulary words and their frequencies.
    ///
    /// Returns all word entries (not labels) in dictionary order (by frequency rank).
    /// The cooking model returns 14543 words with the first entry being `</s>` with freq 12404.
    pub fn get_vocab(&self) -> (Vec<String>, Vec<i64>) {
        let nwords = self.dict.nwords() as usize;
        let words = self.dict.words();
        let mut vocab_words = Vec::with_capacity(nwords);
        let mut vocab_freqs = Vec::with_capacity(nwords);
        for entry in &words[..nwords] {
            vocab_words.push(entry.word.clone());
            vocab_freqs.push(entry.count);
        }
        (vocab_words, vocab_freqs)
    }

    /// Return the labels and their frequencies.
    ///
    /// Returns all label entries in dictionary order (by frequency rank).
    /// The cooking model returns 735 labels with the first label being
    /// `__label__baking` with freq 1156.
    pub fn get_labels(&self) -> (Vec<String>, Vec<i64>) {
        let nwords = self.dict.nwords() as usize;
        let nlabels = self.dict.nlabels() as usize;
        let words = self.dict.words();
        let mut label_words = Vec::with_capacity(nlabels);
        let mut label_freqs = Vec::with_capacity(nlabels);
        for entry in &words[nwords..nwords + nlabels] {
            label_words.push(entry.word.clone());
            label_freqs.push(entry.count);
        }
        (label_words, label_freqs)
    }

    /// Return the model dimensionality (the `dim` hyperparameter).
    pub fn get_dimension(&self) -> i32 {
        self.args.dim()
    }

    /// Return the word ID for the given word, or `-1` if not in vocabulary.
    pub fn get_word_id(&self, word: &str) -> i32 {
        self.dict.get_id(word)
    }

    /// Predict the top-`k` labels for `text`.
    ///
    /// Tokenizes `text` via the dictionary, appends the EOS token (matching
    /// C++ `predictLine` behavior where a stream newline triggers EOS), computes
    /// the hidden representation, runs the appropriate loss function to get
    /// predictions, and converts log-probabilities to probabilities via `exp`.
    ///
    /// Returns a list of `Prediction` pairs sorted by descending probability.
    /// Only predictions whose probability is ≥ `threshold` are returned.
    /// A negative `threshold` is treated as 0.
    ///
    /// Returns an empty vec for empty / whitespace-only input, `k = 0`, or
    /// models with no labels.
    pub fn predict(&self, text: &str, k: usize, threshold: f32) -> Vec<Prediction> {
        if k == 0 {
            return Vec::new();
        }
        // Negative threshold treated as 0
        let effective_threshold = if threshold < 0.0 { 0.0 } else { threshold };

        // Tokenise into word (subword) IDs (no EOS from get_line_from_str).
        let mut words: Vec<i32> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        self.dict.get_line_from_str(text, &mut words, &mut labels);
        if words.is_empty() {
            return Vec::new();
        }

        // Append EOS token to match C++ predictLine behavior: when reading
        // from a stream the newline character produces an EOS token that is
        // included in the hidden-representation average.
        let eos_id = self.dict.get_id(EOS);
        if eos_id >= 0 {
            words.push(eos_id);
        }

        self.predict_words_internal(&words, k, effective_threshold)
    }

    /// Predict the top-`k` labels from pre-tokenized word IDs.
    ///
    /// `word_ids` must be valid input-matrix row indices (as produced by
    /// `Dictionary::get_line_from_str` or equivalent tokenization).
    ///
    /// # EOS handling
    ///
    /// Unlike [`predict`], this method does **not** automatically append the
    /// EOS token (`</s>`).  If you want results identical to `predict(text, …)`,
    /// you must append the EOS token ID yourself before calling this method:
    ///
    /// ```text
    /// let eos_id = model.dict().get_id("</s>");
    /// if eos_id >= 0 { words.push(eos_id); }
    /// let preds = model.predict_on_words(&words, k, threshold);
    /// ```
    ///
    /// This design follows C++ fastText's `FastText::predictLine`, where the
    /// EOS token is injected by the stream tokenizer when a newline is read.
    /// The higher-level [`predict`] method replicates that behavior automatically
    /// (appending EOS after tokenizing the input string), while this lower-level
    /// method gives callers full control over the token sequence.
    pub fn predict_on_words(&self, word_ids: &[i32], k: usize, threshold: f32) -> Vec<Prediction> {
        if k == 0 || word_ids.is_empty() {
            return Vec::new();
        }
        let effective_threshold = if threshold < 0.0 { 0.0 } else { threshold };
        self.predict_words_internal(word_ids, k, effective_threshold)
    }

    /// Internal helper: run model.predict on pre-validated word IDs.
    fn predict_words_internal(&self, word_ids: &[i32], k: usize, threshold: f32) -> Vec<Prediction> {
        let nlabels = self.dict.nlabels() as usize;
        if nlabels == 0 {
            return Vec::new();
        }

        let dim = self.args.dim() as usize;
        let mut state = State::new(dim, nlabels, 0);

        // Clamp k to at most the number of labels.
        let k_eff = k.min(nlabels) as i32;

        let raw = self.model.predict(word_ids, k_eff, threshold, &mut state);

        raw.into_iter()
            .map(|(log_prob, label_idx)| {
                let label = self
                    .dict
                    .get_label(label_idx)
                    .unwrap_or("__unknown__")
                    .to_string();
                Prediction {
                    prob: log_prob.exp(),
                    label,
                }
            })
            .collect()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use crate::args::{Args, LossName, ModelName};
    use crate::dictionary::EOS;

    /// Path to the cooking reference model fixture.
    const COOKING_MODEL: &str = "tests/fixtures/cooking.model.bin";
    /// Path to the invalid model fixture.
    const INVALID_MODEL: &str = "tests/fixtures/invalid.model.bin";

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Build a minimal valid model binary for testing.
    ///
    /// Creates a small valid .bin file:
    /// - magic, version
    /// - args block (supervised, dim=2)
    /// - dict block (minimal, with </s> entry)
    /// - quant_input=false, 2x2 DenseMatrix (input)
    /// - qout=false, 1x2 DenseMatrix (output, 1 label)
    fn make_minimal_model_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic + version
        utils::write_i32(&mut buf, FASTTEXT_FILEFORMAT_MAGIC_INT32).unwrap();
        utils::write_i32(&mut buf, FASTTEXT_VERSION).unwrap();

        // Args: dim=2, ws=1, epoch=1, minCount=1, neg=5, wordNgrams=1,
        //        loss=SOFTMAX(3), model=SUP(3), bucket=0, minn=0, maxn=0,
        //        lrUpdateRate=100, t=0.0001
        let mut args = Args::default();
        args.set_dim(2);
        args.set_ws(1);
        args.set_epoch(1);
        args.set_min_count(1);
        args.set_neg(5);
        args.set_word_ngrams(1);
        args.set_loss(LossName::SOFTMAX);
        args.set_model(ModelName::SUP);
        args.set_bucket(0);
        args.set_minn(0);
        args.set_maxn(0);
        args.set_lr_update_rate(100);
        args.set_t(0.0001);
        args.save(&mut buf).unwrap();

        // Dictionary: 2 entries (</s> + one label), nwords=1, nlabels=1
        // size=2, nwords=1, nlabels=1, ntokens=10, pruneidx_size=-1
        utils::write_i32(&mut buf, 2).unwrap(); // size
        utils::write_i32(&mut buf, 1).unwrap(); // nwords
        utils::write_i32(&mut buf, 1).unwrap(); // nlabels
        utils::write_i64(&mut buf, 10).unwrap(); // ntokens
        utils::write_i64(&mut buf, -1i64).unwrap(); // pruneidx_size = -1 (not pruned)
                                                    // Entry 0: </s>, count=5, type=0 (word)
        buf.extend_from_slice(b"</s>\0");
        utils::write_i64(&mut buf, 5).unwrap();
        buf.push(0u8); // EntryType::Word
                       // Entry 1: __label__test, count=5, type=1 (label)
        buf.extend_from_slice(b"__label__test\0");
        utils::write_i64(&mut buf, 5).unwrap();
        buf.push(1u8); // EntryType::Label

        // quant_input = false
        buf.push(0u8);

        // Input matrix: 1 word × 2 dims (only 1 actual word entry contributes)
        // The input matrix has (nwords + bucket) rows and dim cols.
        // With nwords=1, bucket=0, dim=2: 1×2 matrix
        utils::write_i64(&mut buf, 1).unwrap(); // m = 1
        utils::write_i64(&mut buf, 2).unwrap(); // n = 2
        utils::write_f32(&mut buf, 0.1).unwrap();
        utils::write_f32(&mut buf, 0.2).unwrap();

        // qout = false
        buf.push(0u8);

        // Output matrix: 1 label × 2 dims
        utils::write_i64(&mut buf, 1).unwrap(); // m = 1
        utils::write_i64(&mut buf, 2).unwrap(); // n = 2
        utils::write_f32(&mut buf, 0.5).unwrap();
        utils::write_f32(&mut buf, -0.5).unwrap();

        buf
    }

    // =========================================================================
    // VAL-DICT-009: Magic number and version validation
    // =========================================================================

    #[test]
    fn test_binary_io_magic_version_valid() {
        // Valid magic + version should load successfully
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(&buf);
        let result = FastText::load(&mut cursor);
        assert!(
            result.is_ok(),
            "Valid model should load: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_binary_io_wrong_magic() {
        let mut buf = make_minimal_model_bytes();
        // Corrupt the magic number (first 4 bytes)
        buf[0] = 0xFF;
        buf[1] = 0xFF;
        buf[2] = 0xFF;
        buf[3] = 0xFF;

        let mut cursor = Cursor::new(&buf);
        let result = FastText::load(&mut cursor);
        assert!(result.is_err(), "Wrong magic should be rejected");
        match result.unwrap_err() {
            FastTextError::InvalidModel(msg) => {
                assert!(
                    msg.contains("magic") || msg.contains("Invalid"),
                    "Error should mention magic: {}",
                    msg
                );
            }
            e => panic!("Expected InvalidModel, got: {:?}", e),
        }
    }

    #[test]
    fn test_binary_io_wrong_version_too_high() {
        let mut buf = make_minimal_model_bytes();
        // Set version to 100 (too high) at bytes [4..8]
        let version_bytes: [u8; 4] = 100i32.to_le_bytes();
        buf[4] = version_bytes[0];
        buf[5] = version_bytes[1];
        buf[6] = version_bytes[2];
        buf[7] = version_bytes[3];

        let mut cursor = Cursor::new(&buf);
        let result = FastText::load(&mut cursor);
        assert!(result.is_err(), "Version 100 should be rejected");
        match result.unwrap_err() {
            FastTextError::InvalidModel(msg) => {
                assert!(
                    msg.contains("version") || msg.contains("Unsupported"),
                    "Error should mention version: {}",
                    msg
                );
            }
            e => panic!("Expected InvalidModel, got: {:?}", e),
        }
    }

    #[test]
    fn test_binary_io_version_12_accepted() {
        // Version 12 should be accepted
        let buf = make_minimal_model_bytes();
        assert_eq!(
            i32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            12,
            "Test data should have version 12"
        );
        let mut cursor = Cursor::new(&buf);
        let result = FastText::load(&mut cursor);
        assert!(result.is_ok(), "Version 12 should be accepted");
    }

    #[test]
    fn test_invalid_model_file() {
        // Load the invalid.model.bin fixture
        let result = FastText::load_model(INVALID_MODEL);
        assert!(result.is_err(), "Invalid model should be rejected");
    }

    // =========================================================================
    // VAL-DICT-010: Args block layout
    // =========================================================================

    #[test]
    fn test_binary_io_args_block() {
        // Load model and verify args are correct
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(&buf);
        let model = FastText::load(&mut cursor).unwrap();

        assert_eq!(model.args().dim(), 2);
        assert_eq!(model.args().ws(), 1);
        assert_eq!(model.args().epoch(), 1);
        assert_eq!(model.args().min_count(), 1);
        assert_eq!(model.args().neg(), 5);
        assert_eq!(model.args().word_ngrams(), 1);
        assert_eq!(model.args().loss(), LossName::SOFTMAX);
        assert_eq!(model.args().model(), ModelName::SUP);
        assert_eq!(model.args().bucket(), 0);
        assert_eq!(model.args().minn(), 0);
        assert_eq!(model.args().maxn(), 0);
        assert_eq!(model.args().lr_update_rate(), 100);
        assert!((model.args().t() - 0.0001).abs() < f64::EPSILON);
    }

    // =========================================================================
    // VAL-DICT-011: Dictionary block layout
    // =========================================================================

    #[test]
    fn test_binary_io_dict_block() {
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(&buf);
        let model = FastText::load(&mut cursor).unwrap();

        let dict = model.dict();
        assert_eq!(dict.size(), 2);
        assert_eq!(dict.nwords(), 1);
        assert_eq!(dict.nlabels(), 1);
        assert_eq!(dict.ntokens(), 10);

        // Check word entries
        let words = dict.words();
        assert_eq!(words[0].word, "</s>");
        assert_eq!(words[0].count, 5);
        assert_eq!(words[0].entry_type, crate::dictionary::EntryType::Word);

        assert_eq!(words[1].word, "__label__test");
        assert_eq!(words[1].count, 5);
        assert_eq!(words[1].entry_type, crate::dictionary::EntryType::Label);

        // Verify word lookup works
        assert_eq!(dict.get_id("</s>"), 0);
        assert_eq!(dict.get_id("__label__test"), 1);
        assert_eq!(dict.get_id("unknown"), -1);
    }

    // =========================================================================
    // VAL-DICT-012: Matrix blocks
    // =========================================================================

    #[test]
    fn test_binary_io_dense_matrix() {
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(&buf);
        let model = FastText::load(&mut cursor).unwrap();

        let input = model.input_matrix();
        assert_eq!(input.rows(), 1);
        assert_eq!(input.cols(), 2);
        assert!(
            (input.at(0, 0) - 0.1).abs() < 1e-6,
            "input[0,0] = {}",
            input.at(0, 0)
        );
        assert!(
            (input.at(0, 1) - 0.2).abs() < 1e-6,
            "input[0,1] = {}",
            input.at(0, 1)
        );

        let output = model.output_matrix();
        assert_eq!(output.rows(), 1);
        assert_eq!(output.cols(), 2);
        assert!(
            (output.at(0, 0) - 0.5).abs() < 1e-6,
            "output[0,0] = {}",
            output.at(0, 0)
        );
        assert!(
            (output.at(0, 1) - (-0.5)).abs() < 1e-6,
            "output[0,1] = {}",
            output.at(0, 1)
        );
    }

    #[test]
    fn test_binary_io_dense_matrix_roundtrip() {
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(buf);
        let model = FastText::load(&mut cursor).unwrap();

        // Save and reload
        let mut saved = Vec::new();
        model.save(&mut saved).unwrap();

        let mut cursor2 = Cursor::new(saved);
        let model2 = FastText::load(&mut cursor2).unwrap();

        // Verify matrices match
        let input1 = model.input_matrix();
        let input2 = model2.input_matrix();
        assert_eq!(input1.rows(), input2.rows());
        assert_eq!(input1.cols(), input2.cols());
        for i in 0..input1.rows() {
            for j in 0..input1.cols() {
                assert_eq!(
                    input1.at(i, j),
                    input2.at(i, j),
                    "Input matrix mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }

        let output1 = model.output_matrix();
        let output2 = model2.output_matrix();
        for i in 0..output1.rows() {
            for j in 0..output1.cols() {
                assert_eq!(
                    output1.at(i, j),
                    output2.at(i, j),
                    "Output matrix mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    // =========================================================================
    // VAL-DICT-013: Loading cooking.model.bin reference model
    // =========================================================================

    #[test]
    fn test_load_cooking_model_args() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        let args = model.args();
        // From C++ `dump args` reference output
        assert_eq!(args.dim(), 100, "dim should be 100");
        assert_eq!(args.ws(), 5, "ws should be 5");
        assert_eq!(args.epoch(), 50, "epoch should be 50");
        assert_eq!(args.min_count(), 1, "minCount should be 1");
        assert_eq!(args.neg(), 5, "neg should be 5");
        assert_eq!(args.word_ngrams(), 1, "wordNgrams should be 1");
        assert_eq!(args.loss(), LossName::SOFTMAX, "loss should be softmax");
        assert_eq!(args.model(), ModelName::SUP, "model should be SUP");
        assert_eq!(args.bucket(), 0, "bucket should be 0");
        assert_eq!(args.minn(), 0, "minn should be 0");
        assert_eq!(args.maxn(), 0, "maxn should be 0");
        assert_eq!(args.lr_update_rate(), 100, "lrUpdateRate should be 100");
        assert!(
            (args.t() - 0.0001).abs() < 1e-10,
            "t should be 0.0001, got {}",
            args.t()
        );
    }

    #[test]
    fn test_load_cooking_model_vocab() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        let dict = model.dict();
        assert_eq!(
            dict.nwords(),
            14543,
            "Should have 14543 words, got {}",
            dict.nwords()
        );
        assert_eq!(
            dict.nlabels(),
            735,
            "Should have 735 labels, got {}",
            dict.nlabels()
        );
        assert_eq!(dict.size(), 14543 + 735, "Total size should be 15278");
    }

    #[test]
    fn test_load_cooking_model_first_entry() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        let dict = model.dict();
        let words = dict.words();

        // First entry should be </s> with freq 12404
        assert_eq!(words[0].word, "</s>", "First word should be </s>");
        assert_eq!(
            words[0].count, 12404,
            "First word frequency should be 12404, got {}",
            words[0].count
        );
        assert_eq!(words[0].entry_type, crate::dictionary::EntryType::Word);
    }

    #[test]
    fn test_load_cooking_model_first_label() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        let dict = model.dict();
        // First label should be __label__baking with freq 1156
        let first_label = dict.get_label(0).expect("Should have at least one label");
        assert_eq!(
            first_label, "__label__baking",
            "First label should be __label__baking, got {}",
            first_label
        );

        let words = dict.words();
        let nwords = dict.nwords();
        let first_label_entry = &words[nwords as usize];
        assert_eq!(
            first_label_entry.count, 1156,
            "First label frequency should be 1156, got {}",
            first_label_entry.count
        );
    }

    #[test]
    fn test_load_cooking_model_matrices() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        // Input matrix: nwords × dim (no subwords since maxn=0, bucket=0)
        let input = model.input_matrix();
        assert_eq!(input.rows(), 14543, "Input rows should be nwords=14543");
        assert_eq!(input.cols(), 100, "Input cols should be dim=100");

        // Output matrix: nlabels × dim
        let output = model.output_matrix();
        assert_eq!(output.rows(), 735, "Output rows should be nlabels=735");
        assert_eq!(output.cols(), 100, "Output cols should be dim=100");

        // Matrices should not be quantized
        assert!(
            !model.is_quant(),
            "cooking.model.bin should not be quantized"
        );
    }

    #[test]
    fn test_load_cooking_model_word_lookup() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        let dict = model.dict();

        // EOS should be at index 0
        assert_eq!(dict.get_id("</s>"), 0, "EOS should be at index 0");

        // Known words should be findable
        assert!(
            dict.get_id("baking") >= 0,
            "'baking' should be in vocabulary"
        );
        assert!(
            dict.get_id("banana") >= 0,
            "'banana' should be in vocabulary"
        );

        // Unknown words should return -1
        assert_eq!(
            dict.get_id("xyzzy_definitely_not_a_word"),
            -1,
            "Unknown word should return -1"
        );
    }

    // =========================================================================
    // VAL-DICT-017: Invalid model rejection and backward compatibility
    // =========================================================================

    #[test]
    fn test_invalid_model_rejection() {
        // The invalid.model.bin fixture has a corrupt header
        let result = FastText::load_model(INVALID_MODEL);
        assert!(result.is_err(), "Invalid model should be rejected");
        match result {
            Err(FastTextError::InvalidModel(_)) | Err(FastTextError::IoError(_)) => {
                // Either is acceptable
            }
            Err(e) => panic!("Expected InvalidModel or IoError, got: {:?}", e),
            Ok(_) => panic!("Expected error for invalid model"),
        }
    }

    #[test]
    fn test_zero_bytes_rejected() {
        // Empty file should be rejected
        let result = FastText::load(&mut Cursor::new(vec![]));
        assert!(result.is_err(), "Empty data should be rejected");
    }

    #[test]
    fn test_truncated_model_rejected() {
        // Truncated model (only magic + partial version) should be rejected
        let mut buf = Vec::new();
        utils::write_i32(&mut buf, FASTTEXT_FILEFORMAT_MAGIC_INT32).unwrap();
        buf.push(0x01); // Incomplete version (only 1 byte instead of 4)
        let result = FastText::load(&mut Cursor::new(buf));
        assert!(result.is_err(), "Truncated model should be rejected");
    }

    #[test]
    fn test_backward_compat_v11_supervised() {
        // Version 11 + supervised model → maxn should be forced to 0
        let mut buf = make_minimal_model_bytes();
        // Change version from 12 to 11 at bytes [4..8]
        let version_bytes: [u8; 4] = 11i32.to_le_bytes();
        buf[4] = version_bytes[0];
        buf[5] = version_bytes[1];
        buf[6] = version_bytes[2];
        buf[7] = version_bytes[3];
        // The model in make_minimal_model_bytes has model=SUP and maxn=0 already,
        // but let's set maxn to 6 (default) in the args block to test the override.
        // Args start at byte 8. Field order: dim(0), ws(4), epoch(8), minCount(12),
        // neg(16), wordNgrams(20), loss(24), model(28), bucket(32), minn(36), maxn(40), ...
        // maxn is at offset 8 + 40 = 48
        let maxn_bytes: [u8; 4] = 6i32.to_le_bytes(); // set maxn=6
        buf[8 + 40] = maxn_bytes[0];
        buf[8 + 41] = maxn_bytes[1];
        buf[8 + 42] = maxn_bytes[2];
        buf[8 + 43] = maxn_bytes[3];

        let mut cursor = Cursor::new(buf);
        let model = FastText::load(&mut cursor).unwrap();

        // Version 11 with SUP model should force maxn=0
        assert_eq!(
            model.args().maxn(),
            0,
            "Version 11 supervised model should have maxn=0"
        );
    }

    #[test]
    fn test_backward_compat_v11_unsupervised() {
        // Version 11 with non-supervised model → maxn should NOT be forced to 0
        // Create a version 11 SG model
        let mut buf = Vec::new();
        utils::write_i32(&mut buf, FASTTEXT_FILEFORMAT_MAGIC_INT32).unwrap();
        utils::write_i32(&mut buf, 11).unwrap(); // version 11

        let mut args = Args::default();
        args.set_dim(2);
        args.set_ws(1);
        args.set_epoch(1);
        args.set_min_count(1);
        args.set_neg(5);
        args.set_word_ngrams(1);
        args.set_loss(LossName::NS);
        args.set_model(ModelName::SG); // NOT supervised
        args.set_bucket(100);
        args.set_minn(3);
        args.set_maxn(6); // should remain 6
        args.set_lr_update_rate(100);
        args.set_t(0.0001);
        args.save(&mut buf).unwrap();

        // Minimal dictionary (1 word, 0 labels)
        utils::write_i32(&mut buf, 1).unwrap(); // size
        utils::write_i32(&mut buf, 1).unwrap(); // nwords
        utils::write_i32(&mut buf, 0).unwrap(); // nlabels
        utils::write_i64(&mut buf, 5).unwrap(); // ntokens
        utils::write_i64(&mut buf, -1i64).unwrap(); // pruneidx_size
        buf.extend_from_slice(b"</s>\0");
        utils::write_i64(&mut buf, 5).unwrap();
        buf.push(0u8);

        // quant_input = false
        buf.push(0u8);

        // Input matrix: 1 + 100 = 101 rows (nwords=1, bucket=100) × 2 cols
        utils::write_i64(&mut buf, 101).unwrap();
        utils::write_i64(&mut buf, 2).unwrap();
        for _ in 0..(101 * 2) {
            utils::write_f32(&mut buf, 0.0).unwrap();
        }

        // qout = false
        buf.push(0u8);

        // Output matrix: 1 × 2
        utils::write_i64(&mut buf, 1).unwrap();
        utils::write_i64(&mut buf, 2).unwrap();
        utils::write_f32(&mut buf, 0.0).unwrap();
        utils::write_f32(&mut buf, 0.0).unwrap();

        let mut cursor = Cursor::new(buf);
        let model = FastText::load(&mut cursor).unwrap();

        // Non-supervised v11 model should keep maxn=6
        assert_eq!(
            model.args().maxn(),
            6,
            "Version 11 non-supervised model should keep maxn=6"
        );
    }

    // =========================================================================
    // Model save/load round-trip
    // =========================================================================

    #[test]
    fn test_model_save_load_roundtrip_minimal() {
        // Build minimal model, save, reload, verify args and dict match
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(buf);
        let model1 = FastText::load(&mut cursor).unwrap();

        // Save to buffer
        let mut saved = Vec::new();
        model1.save(&mut saved).unwrap();

        // Reload from saved buffer
        let mut cursor2 = Cursor::new(saved);
        let model2 = FastText::load(&mut cursor2).unwrap();

        // Args should match
        assert_eq!(model1.args().dim(), model2.args().dim());
        assert_eq!(model1.args().epoch(), model2.args().epoch());
        assert_eq!(model1.args().model(), model2.args().model());
        assert_eq!(model1.args().loss(), model2.args().loss());

        // Dict should match
        assert_eq!(model1.dict().size(), model2.dict().size());
        assert_eq!(model1.dict().nwords(), model2.dict().nwords());
        assert_eq!(model1.dict().nlabels(), model2.dict().nlabels());

        let w1 = model1.dict().words();
        let w2 = model2.dict().words();
        for i in 0..w1.len() {
            assert_eq!(w1[i].word, w2[i].word, "Word {} mismatch", i);
            assert_eq!(w1[i].count, w2[i].count, "Count {} mismatch", i);
            assert_eq!(w1[i].entry_type, w2[i].entry_type, "Type {} mismatch", i);
        }
    }

    #[test]
    fn test_model_is_quant_false_for_bin() {
        let model = FastText::load_model(COOKING_MODEL).unwrap();
        assert!(
            !model.is_quant(),
            "cooking.model.bin should not be quantized"
        );
    }

    // =========================================================================
    // VAL-DICT-009 specifically: test with constructed wrong-magic binary
    // =========================================================================

    #[test]
    fn test_binary_io_magic_version() {
        // Valid magic and version 12
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(&buf);
        let result = FastText::load(&mut cursor);
        assert!(
            result.is_ok(),
            "Valid magic+v12 should load: {:?}",
            result.err()
        );

        // Wrong magic
        let mut bad_magic = buf.clone();
        bad_magic[0] = 0x00;
        bad_magic[1] = 0x00;
        bad_magic[2] = 0x00;
        bad_magic[3] = 0x00;
        let result = FastText::load(&mut Cursor::new(bad_magic));
        assert!(result.is_err(), "Wrong magic should be rejected");

        // Version too high
        let mut bad_version = buf.clone();
        let v = 13i32.to_le_bytes();
        bad_version[4] = v[0];
        bad_version[5] = v[1];
        bad_version[6] = v[2];
        bad_version[7] = v[3];
        let result = FastText::load(&mut Cursor::new(bad_version));
        assert!(result.is_err(), "Version 13 should be rejected");

        // Version 11 (should be accepted)
        let mut v11 = buf.clone();
        let v = 11i32.to_le_bytes();
        v11[4] = v[0];
        v11[5] = v[1];
        v11[6] = v[2];
        v11[7] = v[3];
        let result = FastText::load(&mut Cursor::new(v11));
        assert!(
            result.is_ok(),
            "Version 11 should be accepted: {:?}",
            result.err()
        );
    }

    // =========================================================================
    // VAL-DICT-014: Full model save/load round-trip (cooking model)
    // =========================================================================

    /// Verify that saving the cooking model to a file and reloading it produces
    /// a model whose args, vocabulary, word vectors, and predictions are
    /// **bit-for-bit identical** to the original.
    #[test]
    fn test_model_save_load_roundtrip() {
        // ------------------------------------------------------------------
        // 1. Load the reference model.
        // ------------------------------------------------------------------
        let model1 = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        let test_input = "Which baking dish is best to bake a banana bread ?";

        // ------------------------------------------------------------------
        // 2. Capture baseline: predictions and word vector BEFORE save.
        // ------------------------------------------------------------------
        let preds_before = model1.predict(test_input, 5, 0.0);
        assert!(
            !preds_before.is_empty(),
            "Model should produce predictions before save"
        );

        let vec_before = model1.get_word_vector("banana");
        assert_eq!(
            vec_before.len(),
            model1.args().dim() as usize,
            "Word vector should have dim={} elements",
            model1.args().dim()
        );

        // ------------------------------------------------------------------
        // 3. Save to a temp file.
        // ------------------------------------------------------------------
        let tmp_path = std::env::temp_dir().join("fasttext_roundtrip_cooking.bin");
        let tmp_str = tmp_path.to_str().unwrap();
        model1.save_model(tmp_str).expect("Should save model");

        // ------------------------------------------------------------------
        // 4. Reload from the temp file.
        // ------------------------------------------------------------------
        let model2 = FastText::load_model(tmp_str).expect("Should reload model");
        // Clean up the temp file (ignore errors).
        std::fs::remove_file(tmp_str).ok();

        // ------------------------------------------------------------------
        // 5. Verify all args match.
        // ------------------------------------------------------------------
        assert_eq!(
            model1.args().dim(),
            model2.args().dim(),
            "dim should match after round-trip"
        );
        assert_eq!(
            model1.args().ws(),
            model2.args().ws(),
            "ws should match after round-trip"
        );
        assert_eq!(
            model1.args().epoch(),
            model2.args().epoch(),
            "epoch should match after round-trip"
        );
        assert_eq!(
            model1.args().min_count(),
            model2.args().min_count(),
            "minCount should match after round-trip"
        );
        assert_eq!(
            model1.args().neg(),
            model2.args().neg(),
            "neg should match after round-trip"
        );
        assert_eq!(
            model1.args().word_ngrams(),
            model2.args().word_ngrams(),
            "wordNgrams should match after round-trip"
        );
        assert_eq!(
            model1.args().loss(),
            model2.args().loss(),
            "loss should match after round-trip"
        );
        assert_eq!(
            model1.args().model(),
            model2.args().model(),
            "model should match after round-trip"
        );
        assert_eq!(
            model1.args().bucket(),
            model2.args().bucket(),
            "bucket should match after round-trip"
        );
        assert_eq!(
            model1.args().minn(),
            model2.args().minn(),
            "minn should match after round-trip"
        );
        assert_eq!(
            model1.args().maxn(),
            model2.args().maxn(),
            "maxn should match after round-trip"
        );
        assert_eq!(
            model1.args().lr_update_rate(),
            model2.args().lr_update_rate(),
            "lrUpdateRate should match after round-trip"
        );
        assert!(
            (model1.args().t() - model2.args().t()).abs() < f64::EPSILON,
            "t should match after round-trip: {} vs {}",
            model1.args().t(),
            model2.args().t()
        );

        // ------------------------------------------------------------------
        // 6. Verify vocabulary and labels match (count, words, frequencies).
        // ------------------------------------------------------------------
        assert_eq!(
            model1.dict().nwords(),
            model2.dict().nwords(),
            "nwords should match after round-trip"
        );
        assert_eq!(
            model1.dict().nlabels(),
            model2.dict().nlabels(),
            "nlabels should match after round-trip"
        );
        assert_eq!(
            model1.dict().ntokens(),
            model2.dict().ntokens(),
            "ntokens should match after round-trip"
        );
        assert_eq!(
            model1.dict().size(),
            model2.dict().size(),
            "dict size should match after round-trip"
        );

        // Verify all vocabulary entries (word string, frequency, entry type).
        let words1 = model1.dict().words();
        let words2 = model2.dict().words();
        assert_eq!(words1.len(), words2.len(), "words vec length should match");
        for (i, (w1, w2)) in words1.iter().zip(words2.iter()).enumerate() {
            assert_eq!(
                w1.word, w2.word,
                "word[{}] string should match: {} vs {}",
                i, w1.word, w2.word
            );
            assert_eq!(
                w1.count, w2.count,
                "word[{}] '{}' count should match: {} vs {}",
                i, w1.word, w1.count, w2.count
            );
            assert_eq!(
                w1.entry_type, w2.entry_type,
                "word[{}] '{}' type should match",
                i, w1.word
            );
        }

        // ------------------------------------------------------------------
        // 7. Verify word vectors are bitwise identical.
        // ------------------------------------------------------------------
        let vec_after = model2.get_word_vector("banana");
        assert_eq!(
            vec_before.len(),
            vec_after.len(),
            "Word vector length should match"
        );
        for (j, (v1, v2)) in vec_before.iter().zip(vec_after.iter()).enumerate() {
            assert_eq!(
                v1.to_bits(),
                v2.to_bits(),
                "Word vector element [{}] should be bitwise equal: {} vs {}",
                j,
                v1,
                v2
            );
        }

        // ------------------------------------------------------------------
        // 8. Verify predictions are bitwise identical.
        // ------------------------------------------------------------------
        let preds_after = model2.predict(test_input, 5, 0.0);
        assert_eq!(
            preds_before.len(),
            preds_after.len(),
            "Number of predictions should match after round-trip"
        );
        for (idx, (p1, p2)) in
            preds_before.iter().zip(preds_after.iter()).enumerate()
        {
            assert_eq!(
                p1.label, p2.label,
                "Prediction[{}] label should match: {} vs {}",
                idx, p1.label, p2.label
            );
            assert_eq!(
                p1.prob.to_bits(),
                p2.prob.to_bits(),
                "Prediction[{}] label='{}' probability should be bitwise equal: {} vs {}",
                idx,
                p1.label,
                p1.prob,
                p2.prob
            );
        }

        // Also verify the top label is baking-related (sanity check).
        assert!(
            preds_before[0].label.contains("baking") || preds_before[0].label.contains("bread"),
            "Top prediction for banana bread question should be baking or bread related, got: {}",
            preds_before[0].label
        );
    }

    /// Verify word vector round-trip for multiple words.
    #[test]
    fn test_word_vectors_roundtrip() {
        let model1 = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        // Save and reload.
        let tmp_path = std::env::temp_dir().join("fasttext_wordvec_roundtrip.bin");
        let tmp_str = tmp_path.to_str().unwrap();
        model1.save_model(tmp_str).expect("Should save model");
        let model2 = FastText::load_model(tmp_str).expect("Should reload model");
        std::fs::remove_file(tmp_str).ok();

        // Check several words from the cooking vocabulary.
        let test_words = ["banana", "baking", "bread", "chicken", "salt"];
        for word in &test_words {
            let v1 = model1.get_word_vector(word);
            let v2 = model2.get_word_vector(word);
            assert_eq!(v1.len(), v2.len(), "Vector length mismatch for '{}'", word);
            for (j, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "Vector[{}] for '{}' should be bitwise equal: {} vs {}",
                    j,
                    word,
                    a,
                    b
                );
            }
        }
    }

    // =========================================================================
    // Fix: save_model explicitly flushes BufWriter
    // =========================================================================

    /// Verify that save_model flushes the writer before returning, so all
    /// buffered data is written to disk.
    #[test]
    fn test_save_model_flushes_bufwriter() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        let tmp_path = std::env::temp_dir().join("fasttext_flush_test.bin");
        let tmp_str = tmp_path.to_str().unwrap();

        // save_model must succeed (implicit flush).
        model.save_model(tmp_str).expect("save_model should succeed");

        // Reload and verify the model is fully intact — proves the flush worked.
        let model2 = FastText::load_model(tmp_str).expect("Reloaded model should be valid");
        std::fs::remove_file(tmp_str).ok();

        // Sanity-check that predictions from reloaded model are valid.
        let preds = model2.predict("How to bake a banana bread?", 1, 0.0);
        assert!(!preds.is_empty(), "Reloaded model should produce predictions");
    }

    // =========================================================================
    // Fix: predict() does not panic for non-supervised models
    // =========================================================================

    /// Verify that predict() returns an empty vec (not a panic) on a model
    /// with no labels, confirming the doc comment is no longer misleading.
    #[test]
    fn test_predict_non_supervised_model_no_panic() {
        // Build a minimal model with no labels (SG model).
        let mut buf = Vec::new();
        utils::write_i32(&mut buf, FASTTEXT_FILEFORMAT_MAGIC_INT32).unwrap();
        utils::write_i32(&mut buf, FASTTEXT_VERSION).unwrap();

        let mut args = Args::default();
        args.set_dim(2);
        args.set_ws(1);
        args.set_epoch(1);
        args.set_min_count(1);
        args.set_neg(5);
        args.set_word_ngrams(1);
        args.set_loss(LossName::NS);
        args.set_model(ModelName::SG);
        args.set_bucket(0);
        args.set_minn(0);
        args.set_maxn(0);
        args.set_lr_update_rate(100);
        args.set_t(0.0001);
        args.save(&mut buf).unwrap();

        // Dictionary: 1 word entry (</s>), 0 labels.
        utils::write_i32(&mut buf, 1).unwrap(); // size
        utils::write_i32(&mut buf, 1).unwrap(); // nwords
        utils::write_i32(&mut buf, 0).unwrap(); // nlabels
        utils::write_i64(&mut buf, 5).unwrap(); // ntokens
        utils::write_i64(&mut buf, -1i64).unwrap(); // pruneidx_size
        buf.extend_from_slice(b"hello\0");
        utils::write_i64(&mut buf, 5).unwrap();
        buf.push(0u8); // EntryType::Word

        // quant_input = false
        buf.push(0u8);

        // Input matrix: 1 row × 2 cols
        utils::write_i64(&mut buf, 1).unwrap();
        utils::write_i64(&mut buf, 2).unwrap();
        utils::write_f32(&mut buf, 0.5).unwrap();
        utils::write_f32(&mut buf, 0.5).unwrap();

        // qout = false
        buf.push(0u8);

        // Output matrix: 1 row × 2 cols
        utils::write_i64(&mut buf, 1).unwrap();
        utils::write_i64(&mut buf, 2).unwrap();
        utils::write_f32(&mut buf, 0.3).unwrap();
        utils::write_f32(&mut buf, 0.3).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let model = FastText::load(&mut cursor).expect("Model should load");

        // predict() should return empty (no labels) without panicking.
        let preds = model.predict("hello", 5, 0.0);
        assert!(
            preds.is_empty(),
            "Model with no labels should return empty predictions, not panic"
        );
    }

    /// Verify that predictions are deterministic across multiple calls after round-trip.
    #[test]
    fn test_predictions_identical_after_roundtrip() {
        let model1 = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        let tmp_path = std::env::temp_dir().join("fasttext_pred_roundtrip.bin");
        let tmp_str = tmp_path.to_str().unwrap();
        model1.save_model(tmp_str).expect("Should save model");
        let model2 = FastText::load_model(tmp_str).expect("Should reload model");
        std::fs::remove_file(tmp_str).ok();

        let inputs = [
            "how to make pasta",
            "best knife for cutting vegetables",
            "what temperature to bake chicken",
        ];

        for input in &inputs {
            let p1 = model1.predict(input, 3, 0.0);
            let p2 = model2.predict(input, 3, 0.0);
            assert_eq!(
                p1.len(),
                p2.len(),
                "Prediction count should match for: {}",
                input
            );
            for (i, (pred1, pred2)) in p1.iter().zip(p2.iter()).enumerate() {
                assert_eq!(pred1.label, pred2.label, "Label[{}] should match for: {}", i, input);
                assert_eq!(
                    pred1.prob.to_bits(),
                    pred2.prob.to_bits(),
                    "Prob[{}] should be bitwise equal for: {}",
                    i,
                    input
                );
            }
        }
    }

    // =========================================================================
    // VAL-INF-007: predict() cooking model top-2 reference
    // =========================================================================

    /// Verify predict() returns __label__baking and __label__bread as top-2
    /// for the canonical cooking test query.
    #[test]
    fn test_predict_cooking_top2() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let input = "Which baking dish is best to bake a banana bread ?";
        let preds = model.predict(input, 2, 0.0);

        assert_eq!(preds.len(), 2, "Should return exactly 2 predictions, got {:?}",
            preds.iter().map(|p| &p.label).collect::<Vec<_>>());
        assert_eq!(
            preds[0].label, "__label__baking",
            "Top-1 should be __label__baking, got '{}'", preds[0].label
        );
        assert_eq!(
            preds[1].label, "__label__bread",
            "Top-2 should be __label__bread, got '{}'", preds[1].label
        );
        // Top-1 probability > top-2 probability
        assert!(
            preds[0].prob > preds[1].prob,
            "Top-1 prob ({}) should be > top-2 prob ({})",
            preds[0].prob, preds[1].prob
        );
    }

    // =========================================================================
    // VAL-INF-008: predict() probability values match C++
    // =========================================================================

    /// Verify predicted probabilities match C++ output within 1e-4 absolute tolerance.
    ///
    /// C++ reference (predict-prob cooking.model.bin 5):
    ///   __label__baking    0.706095
    ///   __label__bread     0.137935
    ///   __label__equipment 0.0167011
    ///   __label__muffins   0.0107388
    ///   __label__oven      0.0095826
    #[test]
    fn test_predict_cooking_probabilities() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let input = "Which baking dish is best to bake a banana bread ?";
        let preds = model.predict(input, 5, 0.0);

        assert!(preds.len() >= 2, "Should return at least 2 predictions");

        // Expected values from C++ reference (exp of log-prob)
        let expected = [
            ("__label__baking",    0.706095_f32),
            ("__label__bread",     0.137935_f32),
            ("__label__equipment", 0.0167011_f32),
            ("__label__muffins",   0.0107388_f32),
            ("__label__oven",      0.0095826_f32),
        ];

        // Verify at least the first 2 predictions match
        for (i, &(label, cpp_prob)) in expected.iter().take(2).enumerate() {
            assert_eq!(
                preds[i].label, label,
                "Prediction[{}] label mismatch: expected '{}', got '{}'",
                i, label, preds[i].label
            );
            assert!(
                (preds[i].prob - cpp_prob).abs() < 1e-4,
                "Prediction[{}] '{}': prob={} expected={} diff={}",
                i, label, preds[i].prob, cpp_prob,
                (preds[i].prob - cpp_prob).abs()
            );
        }

        // If we have 5 predictions, check all 5
        if preds.len() >= 5 {
            for (i, &(label, cpp_prob)) in expected.iter().enumerate() {
                assert_eq!(
                    preds[i].label, label,
                    "Prediction[{}] label mismatch", i
                );
                assert!(
                    (preds[i].prob - cpp_prob).abs() < 1e-4,
                    "Prediction[{}] '{}': prob={} expected={} diff={}",
                    i, label, preds[i].prob, cpp_prob,
                    (preds[i].prob - cpp_prob).abs()
                );
            }
        }
    }

    // =========================================================================
    // VAL-INF-009: predict() threshold filtering
    // =========================================================================

    /// Verify that only predictions with probability >= threshold are returned.
    #[test]
    fn test_predict_threshold_filtering() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let input = "Which baking dish is best to bake a banana bread ?";

        // threshold=0.0: all predictions returned (up to k)
        let preds_all = model.predict(input, 10, 0.0);
        assert!(!preds_all.is_empty(), "threshold=0.0 should return predictions");
        for p in &preds_all {
            assert!(p.prob >= 0.0, "All probs should be >= 0.0");
        }

        // threshold=0.5: only high-confidence predictions
        let preds_half = model.predict(input, 10, 0.5);
        for p in &preds_half {
            assert!(
                p.prob >= 0.5,
                "All probs should be >= 0.5 when threshold=0.5, got {}",
                p.prob
            );
        }
        // The top prediction (baking, ~0.706) should be above 0.5
        assert!(
            !preds_half.is_empty(),
            "At least one prediction should have prob >= 0.5"
        );
        assert_eq!(
            preds_half[0].label, "__label__baking",
            "Top prediction above 0.5 threshold should be __label__baking"
        );

        // threshold=1.0: no predictions should be returned (softmax prob < 1 generally)
        let preds_max = model.predict(input, 10, 1.0);
        for p in &preds_max {
            assert!(
                p.prob >= 1.0,
                "All probs should be >= 1.0 when threshold=1.0, got {}",
                p.prob
            );
        }

        // Verify threshold filtering: preds with high threshold is subset of low threshold
        let preds_low = model.predict(input, 100, 0.01);
        let preds_high = model.predict(input, 100, 0.1);
        assert!(
            preds_high.len() <= preds_low.len(),
            "Higher threshold should return fewer or equal predictions"
        );
        // All preds_high labels should appear in preds_low
        for p in &preds_high {
            assert!(
                p.prob >= 0.1,
                "High threshold result should have prob >= 0.1, got {}",
                p.prob
            );
        }
    }

    // =========================================================================
    // VAL-INF-010: predict() edge cases
    // =========================================================================

    /// Empty string input returns empty predictions without panic.
    #[test]
    fn test_predict_empty_input() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let preds = model.predict("", 5, 0.0);
        assert!(preds.is_empty(), "Empty input should return empty predictions");
    }

    /// Whitespace-only input returns empty predictions without panic.
    #[test]
    fn test_predict_whitespace_only() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let preds = model.predict("   \t  \n  ", 5, 0.0);
        assert!(preds.is_empty(), "Whitespace-only input should return empty predictions");
    }

    /// k=0 returns empty predictions without panic.
    #[test]
    fn test_predict_k_zero() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let preds = model.predict("Which baking dish is best?", 0, 0.0);
        assert!(preds.is_empty(), "k=0 should return empty predictions");
    }

    /// k > nlabels returns at most nlabels predictions.
    #[test]
    fn test_predict_k_large() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let nlabels = model.dict().nlabels() as usize;
        let very_large_k = nlabels + 10000;
        let preds = model.predict("Which baking dish is best?", very_large_k, 0.0);
        assert!(
            preds.len() <= nlabels,
            "k > nlabels should return at most nlabels={} predictions, got {}",
            nlabels, preds.len()
        );
    }

    /// Negative threshold is treated as 0 (no threshold filtering).
    #[test]
    fn test_predict_negative_threshold() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let input = "Which baking dish is best to bake a banana bread ?";
        let preds_zero = model.predict(input, 5, 0.0);
        let preds_neg = model.predict(input, 5, -1.0);

        // Negative threshold should return same as 0 threshold
        assert_eq!(
            preds_zero.len(),
            preds_neg.len(),
            "Negative threshold should return same count as threshold=0"
        );
        for (p1, p2) in preds_zero.iter().zip(preds_neg.iter()) {
            assert_eq!(p1.label, p2.label, "Labels should match");
            assert_eq!(p1.prob.to_bits(), p2.prob.to_bits(), "Probs should match");
        }
    }

    // =========================================================================
    // VAL-INF-011: predict_on_words() matches predict()
    // =========================================================================

    /// Verify predict_on_words produces identical results to predict for same input.
    #[test]
    fn test_predict_on_words_matches_predict() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let input = "Which baking dish is best to bake a banana bread ?";

        // Get word IDs via tokenization (same as what predict() uses internally,
        // including EOS to match C++ predictLine behavior).
        let mut words: Vec<i32> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        model.dict().get_line_from_str(input, &mut words, &mut labels);
        assert!(!words.is_empty(), "Should produce word IDs");
        // Add EOS just like predict() does
        let eos_id = model.dict().get_id(EOS);
        if eos_id >= 0 {
            words.push(eos_id);
        }

        let preds_text = model.predict(input, 5, 0.0);
        let preds_words = model.predict_on_words(&words, 5, 0.0);

        assert_eq!(
            preds_text.len(),
            preds_words.len(),
            "predict and predict_on_words should return same number of predictions"
        );
        for (p1, p2) in preds_text.iter().zip(preds_words.iter()) {
            assert_eq!(p1.label, p2.label, "Labels should match exactly");
            assert_eq!(
                p1.prob.to_bits(),
                p2.prob.to_bits(),
                "Probabilities should be bitwise equal"
            );
        }
    }

    /// predict_on_words with empty slice returns empty.
    #[test]
    fn test_predict_on_words_empty() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let preds = model.predict_on_words(&[], 5, 0.0);
        assert!(preds.is_empty(), "Empty word IDs should return empty predictions");
    }

    /// predict_on_words with k=0 returns empty.
    #[test]
    fn test_predict_on_words_k_zero() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let preds = model.predict_on_words(&[0, 1, 2], 0, 0.0);
        assert!(preds.is_empty(), "k=0 should return empty predictions");
    }

    // =========================================================================
    // VAL-INF-018: Prediction determinism
    // =========================================================================

    /// Verify 10 identical calls to predict() return bit-identical results.
    #[test]
    fn test_predict_determinism() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let input = "Which baking dish is best to bake a banana bread ?";

        let first = model.predict(input, 5, 0.0);
        assert!(!first.is_empty(), "Should produce predictions");

        for i in 1..10 {
            let preds = model.predict(input, 5, 0.0);
            assert_eq!(
                preds.len(), first.len(),
                "Call {} prediction count should match", i
            );
            for (j, (p1, p2)) in first.iter().zip(preds.iter()).enumerate() {
                assert_eq!(
                    p1.label, p2.label,
                    "Call {} prediction[{}] label should be identical", i, j
                );
                assert_eq!(
                    p1.prob.to_bits(), p2.prob.to_bits(),
                    "Call {} prediction[{}] prob should be bit-identical: {} vs {}",
                    i, j, p1.prob, p2.prob
                );
            }
        }
    }

    // =========================================================================
    // VAL-INF-019: Thread safety for concurrent prediction
    // =========================================================================

    /// Verify that concurrent predict() calls via Arc<FastText> work correctly.
    ///
    /// Spawns 8 threads, each calling predict() 10 times. All results must
    /// match the single-threaded reference result.
    #[test]
    fn test_predict_thread_safety() {
        use std::thread;

        let model = Arc::new(
            FastText::load_model(COOKING_MODEL).expect("Should load cooking model")
        );
        let input = "Which baking dish is best to bake a banana bread ?";

        // Get reference result single-threaded
        let reference = model.predict(input, 5, 0.0);
        assert!(!reference.is_empty(), "Reference should have predictions");

        let reference = Arc::new(reference);
        let n_threads = 8;
        let n_calls = 10;
        let mut handles = Vec::new();

        for thread_id in 0..n_threads {
            let model = Arc::clone(&model);
            let reference = Arc::clone(&reference);
            let input = input.to_string();
            let handle = thread::spawn(move || {
                for call in 0..n_calls {
                    let preds = model.predict(&input, 5, 0.0);
                    assert_eq!(
                        preds.len(), reference.len(),
                        "Thread {} call {}: prediction count mismatch",
                        thread_id, call
                    );
                    for (j, (p, r)) in preds.iter().zip(reference.iter()).enumerate() {
                        assert_eq!(
                            p.label, r.label,
                            "Thread {} call {}: prediction[{}] label mismatch: '{}' vs '{}'",
                            thread_id, call, j, p.label, r.label
                        );
                        assert_eq!(
                            p.prob.to_bits(), r.prob.to_bits(),
                            "Thread {} call {}: prediction[{}] prob mismatch: {} vs {}",
                            thread_id, call, j, p.prob, r.prob
                        );
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }
    }

    // =========================================================================
    // VAL-INF-020: Prediction probabilities validity
    // =========================================================================

    /// Verify all predicted probabilities are in [0.0, 1.0] and not NaN/Inf.
    #[test]
    fn test_predict_probability_validity() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        let test_inputs = [
            "Which baking dish is best to bake a banana bread ?",
            "how to make pasta at home",
            "best knife for cutting vegetables",
            "what temperature to bake chicken",
            "how long to boil eggs",
            "substitute for buttermilk",
        ];

        for input in &test_inputs {
            let preds = model.predict(input, 10, 0.0);
            assert!(!preds.is_empty(), "Should have predictions for: {}", input);
            for p in &preds {
                assert!(
                    p.prob.is_finite(),
                    "Probability should be finite for '{}': got {}",
                    input, p.prob
                );
                assert!(
                    p.prob >= 0.0,
                    "Probability should be >= 0.0 for '{}': got {}",
                    input, p.prob
                );
                assert!(
                    p.prob <= 1.0 + 1e-5,
                    "Probability should be <= 1.0 for '{}': got {}",
                    input, p.prob
                );
            }
            // Probabilities should be sorted descending
            for i in 1..preds.len() {
                assert!(
                    preds[i-1].prob >= preds[i].prob,
                    "Predictions should be sorted descending by prob for '{}': {} < {}",
                    input, preds[i-1].prob, preds[i].prob
                );
            }
        }
    }

    // =========================================================================
    // VAL-INF-012: get_word_vector() banana reference
    // =========================================================================

    /// Verify get_word_vector("banana") matches C++ reference within 1e-3.
    ///
    /// C++ reference (print-word-vectors cooking.model.bin):
    /// banana 0.48844 -0.14683 0.3119 -0.36661 0.22843 -0.07035 ...
    #[test]
    fn test_get_word_vector_banana() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let vec = model.get_word_vector("banana");

        // Check dimension
        assert_eq!(vec.len(), 100, "Word vector should have 100 dimensions");

        // C++ reference values for all 100 dimensions
        let expected: [f32; 100] = [
            0.48844, -0.14683, 0.3119, -0.36661, 0.22843, -0.07035, -0.26473, -0.35418,
            -0.19428, -0.10533, 0.22645, 0.13888, -0.40894, 0.17568, -0.31359, 0.38722,
            0.12278, -0.1001, 0.04358, -0.23915, 0.24731, 0.43714, -0.14672, 0.26647,
            0.4463, 0.4347, 0.034649, 0.064306, -0.6327, 0.17736, -0.26013, -0.25258,
            -0.03388, -0.27005, -0.12958, 0.44716, -0.32228, 0.24188, -0.31526, 0.33497,
            -0.20352, -0.21103, 0.50374, 0.077682, 0.66139, -0.5584, 0.10622, -0.07879,
            -0.17618, -0.21429, -0.31943, -0.026991, -0.32334, 0.44703, 0.19859, 0.17837,
            0.37342, -0.19418, -0.3752, -0.19296, -0.18952, 0.34282, -0.33506, 0.27638,
            -0.065614, 0.28327, 0.0028778, -0.11029, -0.24301, 0.50804, -0.14128, 0.44562,
            -0.15644, -0.49472, 0.074092, -0.61279, 0.029795, -0.26603, -0.51902, 0.11931,
            0.25819, 0.15659, 0.18606, 0.080266, 0.099765, 0.056123, -0.46964, 0.11671,
            0.32503, 0.10737, 0.086726, -0.13546, 0.10999, -0.22411, -0.26554, -0.010061,
            -0.37875, -0.083359, 0.57227, -0.69741,
        ];

        let tolerance = 1e-3_f32;
        for (i, (&got, &exp)) in vec.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < tolerance,
                "banana vector[{}]: got={}, expected={}, diff={}",
                i, got, exp, (got - exp).abs()
            );
        }
    }

    // =========================================================================
    // VAL-INF-013: get_word_vector() unknown word behavior
    // =========================================================================

    /// Unknown word with maxn=0 returns zero vector.
    #[test]
    fn test_get_word_vector_unknown_zero() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        // cooking model has maxn=0 (no subword computation)
        assert_eq!(model.args().maxn(), 0, "cooking model should have maxn=0");

        let vec = model.get_word_vector("xyzzy_definitely_not_in_vocabulary_42");
        assert_eq!(vec.len(), 100, "Vector should have 100 dimensions");

        for &v in &vec {
            assert_eq!(v, 0.0, "Unknown word with maxn=0 should return zero vector, got {}", v);
        }
    }

    /// Known word returns non-zero vector.
    #[test]
    fn test_get_word_vector_known_nonzero() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let vec = model.get_word_vector("banana");
        assert_eq!(vec.len(), 100, "Vector should have 100 dimensions");

        let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            norm > 0.0,
            "Known word 'banana' should have non-zero vector, norm={}", norm
        );
    }

    /// get_word_vector returns correct dimension.
    #[test]
    fn test_get_word_vector_dimension() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let vec = model.get_word_vector("baking");
        assert_eq!(
            vec.len(),
            model.get_dimension() as usize,
            "Word vector length should equal model dimension"
        );
    }

    // =========================================================================
    // VAL-INF-014: get_sentence_vector() behavior
    // =========================================================================

    /// Supervised model sentence vector: no L2 normalization, raw average.
    #[test]
    fn test_get_sentence_vector_supervised() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let sentence = "How to bake a banana bread";
        let svec = model.get_sentence_vector(sentence);
        assert_eq!(svec.len(), 100, "Sentence vector should have 100 dims");

        // Should be non-zero for a sentence with known words
        let norm: f32 = svec.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "Sentence vector should be non-zero for known words");
    }

    /// Empty input returns zero vector.
    #[test]
    fn test_get_sentence_vector_empty() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let svec = model.get_sentence_vector("");
        assert_eq!(svec.len(), 100, "Sentence vector should have 100 dims");
        for &v in &svec {
            assert_eq!(v, 0.0, "Empty sentence should return zero vector, got {}", v);
        }
    }

    /// Whitespace-only input returns zero vector.
    #[test]
    fn test_get_sentence_vector_whitespace_only() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let svec = model.get_sentence_vector("   \t  ");
        assert_eq!(svec.len(), 100, "Sentence vector should have 100 dims");
        for &v in &svec {
            assert_eq!(v, 0.0, "Whitespace-only should return zero vector, got {}", v);
        }
    }

    /// Sentence vector averaging is correct: multi-word sentence differs from single-word.
    /// The sentence vector is NOT just the word vector - it includes EOS in the average.
    #[test]
    fn test_get_sentence_vector_averaging() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        // Single-word sentence vector should be non-zero (baking is in vocab)
        let sent_vec = model.get_sentence_vector("baking");
        assert_eq!(sent_vec.len(), 100, "Sentence vector should have 100 dims");
        let norm: f32 = sent_vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "Sentence vector for 'baking' should be non-zero");

        // Two-word sentence should be different from single-word sentence
        let sent_vec2 = model.get_sentence_vector("baking bread");
        let norm2: f32 = sent_vec2.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(norm2 > 0.0, "Sentence vector for 'baking bread' should be non-zero");

        // Longer sentence should produce different result than shorter
        let sent_vec3 = model.get_sentence_vector("baking banana bread cake");
        assert_ne!(
            sent_vec, sent_vec3,
            "Different sentences should produce different vectors"
        );
    }

    /// Sentence vector matches C++ reference output.
    /// C++ print-sentence-vectors reference for "How to bake a banana bread"
    /// starts with: 0.073472 -0.027573 0.10399 -0.47752 0.031626 ...
    #[test]
    fn test_get_sentence_vector_reference() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let svec = model.get_sentence_vector("How to bake a banana bread");
        assert_eq!(svec.len(), 100, "Should have 100 dims");

        // First few C++ reference values
        let expected_first_5 = [0.073472_f32, -0.027573, 0.10399, -0.47752, 0.031626];
        let tolerance = 1e-3_f32;
        for (i, (&got, &exp)) in svec.iter().zip(expected_first_5.iter()).enumerate() {
            assert!(
                (got - exp).abs() < tolerance,
                "sentence_vector[{}]: got={}, expected={}, diff={}",
                i, got, exp, (got - exp).abs()
            );
        }
    }

    // =========================================================================
    // VAL-INF-015: tokenize() correctness
    // =========================================================================

    /// Basic whitespace splitting.
    #[test]
    fn test_tokenize_basic() {
        let tokens = FastText::tokenize("hello world foo");
        assert_eq!(tokens, vec!["hello", "world", "foo"]);
    }

    /// Unicode tokens preserved intact.
    #[test]
    fn test_tokenize_unicode() {
        let tokens = FastText::tokenize("日本語 café résumé");
        assert_eq!(tokens, vec!["日本語", "café", "résumé"]);
    }

    /// Empty string returns empty vec.
    #[test]
    fn test_tokenize_empty() {
        let tokens = FastText::tokenize("");
        assert!(tokens.is_empty(), "Empty string should return empty vec");
    }

    /// Multiple consecutive whitespace characters collapsed.
    #[test]
    fn test_tokenize_multi_whitespace() {
        let tokens = FastText::tokenize("hello   world\t\tfoo");
        assert_eq!(tokens, vec!["hello", "world", "foo"]);
    }

    /// Leading/trailing whitespace ignored.
    #[test]
    fn test_tokenize_leading_trailing_whitespace() {
        let tokens = FastText::tokenize("  hello world  ");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    /// Whitespace-only input returns empty vec.
    #[test]
    fn test_tokenize_whitespace_only() {
        let tokens = FastText::tokenize("   \t  \n  ");
        assert!(tokens.is_empty(), "Whitespace-only should return empty vec");
    }

    /// Single word returns single-element vec.
    #[test]
    fn test_tokenize_single_word() {
        let tokens = FastText::tokenize("hello");
        assert_eq!(tokens, vec!["hello"]);
    }

    // =========================================================================
    // VAL-INF-016: get_vocab() and get_labels() cooking model reference
    // =========================================================================

    /// get_vocab() returns 14543 words, first entry is </s> with freq 12404.
    #[test]
    fn test_get_vocab_cooking() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let (words, freqs) = model.get_vocab();

        assert_eq!(
            words.len(), 14543,
            "Should have 14543 words, got {}", words.len()
        );
        assert_eq!(
            freqs.len(), 14543,
            "Freqs length should match words length"
        );

        // First word should be </s> with freq 12404
        assert_eq!(
            words[0], "</s>",
            "First word should be </s>, got '{}'", words[0]
        );
        assert_eq!(
            freqs[0], 12404,
            "First word freq should be 12404, got {}", freqs[0]
        );

        // All words should be non-empty
        for (i, word) in words.iter().enumerate() {
            assert!(!word.is_empty(), "Word[{}] should not be empty", i);
        }

        // All freqs should be positive
        for (i, &freq) in freqs.iter().enumerate() {
            assert!(freq > 0, "Freq[{}] should be positive, got {}", i, freq);
        }
    }

    /// get_labels() returns 735 labels, first label is __label__baking with freq 1156.
    #[test]
    fn test_get_labels_cooking() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let (labels, freqs) = model.get_labels();

        assert_eq!(
            labels.len(), 735,
            "Should have 735 labels, got {}", labels.len()
        );
        assert_eq!(
            freqs.len(), 735,
            "Freqs length should match labels length"
        );

        // First label should be __label__baking with freq 1156
        assert_eq!(
            labels[0], "__label__baking",
            "First label should be __label__baking, got '{}'", labels[0]
        );
        assert_eq!(
            freqs[0], 1156,
            "First label freq should be 1156, got {}", freqs[0]
        );

        // All labels should start with __label__
        for (i, label) in labels.iter().enumerate() {
            assert!(
                label.starts_with("__label__"),
                "Label[{}] '{}' should start with '__label__'", i, label
            );
        }
    }

    // =========================================================================
    // VAL-INF-017: Metadata accessors
    // =========================================================================

    /// get_dimension() returns the correct value for the cooking model.
    #[test]
    fn test_get_dimension() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        assert_eq!(
            model.get_dimension(), 100,
            "Cooking model dimension should be 100"
        );
    }

    /// get_word_id() returns correct ID for known words and -1 for unknown.
    #[test]
    fn test_get_word_id_known() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

        // EOS should be at index 0
        let eos_id = model.get_word_id("</s>");
        assert_eq!(eos_id, 0, "EOS should be at index 0, got {}", eos_id);

        // Known words should be in vocabulary
        let banana_id = model.get_word_id("banana");
        assert!(
            banana_id >= 0,
            "'banana' should be in vocabulary, got id={}", banana_id
        );

        let baking_id = model.get_word_id("baking");
        assert!(
            baking_id >= 0,
            "'baking' should be in vocabulary, got id={}", baking_id
        );
    }

    /// get_word_id() returns -1 for unknown words.
    #[test]
    fn test_get_word_id_unknown() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let id = model.get_word_id("xyzzy_definitely_not_in_vocabulary_42");
        assert_eq!(id, -1, "Unknown word should return -1, got {}", id);
    }

    /// is_quant() returns false for .bin models.
    #[test]
    fn test_is_quant_false_for_bin_model() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        assert!(!model.is_quant(), "cooking.model.bin should not be quantized");
    }

    /// get_dimension() matches args.dim().
    #[test]
    fn test_get_dimension_matches_args() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        assert_eq!(
            model.get_dimension(),
            model.args().dim(),
            "get_dimension() should equal args().dim()"
        );
    }

    /// get_vocab() words are not labels (none should start with __label__).
    #[test]
    fn test_get_vocab_not_labels() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let (words, _) = model.get_vocab();
        for (i, word) in words.iter().enumerate() {
            assert!(
                !word.starts_with("__label__"),
                "Vocab word[{}] '{}' should not be a label", i, word
            );
        }
    }

    /// get_labels() returns proper label format.
    #[test]
    fn test_get_labels_format() {
        let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
        let (labels, freqs) = model.get_labels();

        assert_eq!(labels.len(), freqs.len(), "Labels and freqs should have same length");

        // All labels should be non-empty
        for (i, label) in labels.iter().enumerate() {
            assert!(!label.is_empty(), "Label[{}] should not be empty", i);
        }

        // Frequencies should be positive
        for (i, &freq) in freqs.iter().enumerate() {
            assert!(freq > 0, "Label freq[{}] should be > 0, got {}", i, freq);
        }
    }
}

