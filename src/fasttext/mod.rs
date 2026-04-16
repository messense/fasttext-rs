// FastText: save/load, accessors, word/sentence vectors

mod nn;
mod predict;
mod quantize;
mod train;

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::args::{Args, LossName, ModelName};
use crate::dictionary::{Dictionary, EntryType, EOS};
use crate::error::{FastTextError, Result};
use crate::loss::{
    HierarchicalSoftmaxLoss, Loss, LossTables, NegativeSamplingLoss, OneVsAllLoss, SoftmaxLoss,
};
use crate::matrix::{DenseMatrix, Matrix};
use crate::model::Model;
use crate::quant_matrix::QuantMatrix;
use crate::utils;
use crate::vector::Vector;

/// Magic number identifying a valid fastText binary model file.
pub const FASTTEXT_FILEFORMAT_MAGIC_INT32: i32 = 793712314;
/// Current binary format version.
pub const FASTTEXT_VERSION: i32 = 12;

/// A handle to an in-flight training run spawned by [`FastText::spawn_training`].
///
/// Allows aborting the training from the calling thread via [`TrainingHandle::abort`],
/// and retrieving the resulting model via [`TrainingHandle::join`].
pub struct TrainingHandle {
    abort_flag: Arc<AtomicBool>,
    join_handle: std::thread::JoinHandle<Result<FastText>>,
}

impl TrainingHandle {
    /// Signal the background training thread to stop early.
    ///
    /// This is idempotent: calling it multiple times has no adverse effect.
    pub fn abort(&self) {
        self.abort_flag.store(true, Ordering::Relaxed);
    }

    /// Wait for the background training thread to finish and return the model.
    ///
    /// The returned `FastText` is valid for inference even if training was
    /// aborted early (it will be under-trained but not corrupted).
    ///
    /// # Errors
    /// Returns `Err` if the training thread panicked (`std::thread::Result::Err`).
    /// Propagates `FastTextError` from the training run as the inner `Result`.
    pub fn join(self) -> std::thread::Result<Result<FastText>> {
        self.join_handle.join()
    }
}

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

/// Build the appropriate loss function based on `args.loss`.
///
/// - `SoftmaxLoss` — full softmax (used for supervised models with softmax loss)
/// - `NegativeSamplingLoss` — negative sampling (skipgram/CBOW)
/// - `HierarchicalSoftmaxLoss` — Huffman-tree hierarchical softmax
/// - `OneVsAllLoss` — one-vs-all binary logistic
pub(super) fn build_loss(
    args: &Args,
    wo: Arc<DenseMatrix>,
    target_counts: &[i64],
) -> Box<dyn Loss> {
    match args.loss {
        LossName::HS => Box::new(HierarchicalSoftmaxLoss::new(wo, target_counts)),
        LossName::NS => Box::new(NegativeSamplingLoss::new(wo, args.neg, target_counts)),
        LossName::OVA => Box::new(OneVsAllLoss::new(wo)),
        LossName::SOFTMAX => Box::new(SoftmaxLoss::new(wo)),
    }
}

/// Shared context passed to each training thread.
///
/// Bundles the per-run read-only state so that `train_thread_inner` stays
/// within clippy's argument-count limit.
pub(super) struct TrainThreadCtx<'a> {
    pub(super) args: &'a Args,
    pub(super) dict: &'a Dictionary,
    pub(super) model: &'a Model,
    pub(super) output_size: usize,
    pub(super) token_count: &'a AtomicI64,
    pub(super) abort_flag: &'a AtomicBool,
    /// Shared atomic accumulator for total training loss (f64 bits) across all threads.
    ///
    /// Each thread atomically adds its local loss sum to this counter at the end
    /// of training.  After all threads finish, the total loss can be read via
    /// `f64::from_bits(shared_loss.load(Ordering::Relaxed))`.
    pub(super) shared_loss: &'a AtomicU64,
    /// Optional per-epoch loss tracker.
    ///
    /// When `Some`, records the average loss after each completed epoch.
    /// Best used with `thread=1` for accurate epoch boundaries.
    pub(super) epoch_loss_tracker: Option<Arc<Mutex<Vec<f32>>>>,
}

/// A loaded fastText model.
///
/// Contains the model arguments, dictionary, input matrix, output matrix,
/// and a pre-built `Model` for efficient inference.
#[derive(Debug)]
pub struct FastText {
    /// Model hyperparameters.
    pub(super) args: Arc<Args>,
    /// The vocabulary dictionary.
    pub(super) dict: Dictionary,
    /// Input embedding matrix (word + subword vectors), shared via Arc for Model.
    /// For quantized models this is a zero-size placeholder; use `quant_input` instead.
    pub(super) input: Arc<DenseMatrix>,
    /// Output matrix (label/word vectors), shared via Arc for Model.
    /// For quantized output, use `quant_output` instead.
    pub(super) output: Arc<DenseMatrix>,
    /// Whether the model uses quantized (QuantMatrix) input.
    pub(super) quant: bool,
    /// Quantized input matrix. Present when `quant=true`.
    pub(super) quant_input: Option<QuantMatrix>,
    /// Quantized output matrix. Present when `quant=true` and `args.qout=true`.
    pub(super) quant_output: Option<QuantMatrix>,
    /// Pre-built inference model (uses dense matrices; bypassed when quant=true).
    pub(super) model: Model,
    /// Atomic flag for aborting an in-progress training run.
    ///
    /// Set via [`FastText::abort()`]; checked in the training loop.
    pub(super) abort_flag: Arc<AtomicBool>,
    /// Average training loss from the last training run.
    ///
    /// Set by [`FastText::train`] and related functions.  `0.0` for models
    /// loaded from disk (no training history available).
    pub(super) last_train_loss: f64,
    /// Cached sigmoid/log lookup tables for quantized prediction.
    pub(super) loss_tables: LossTables,
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
        if version == 11 && args.model == ModelName::SUP {
            args.maxn = 0;
        }

        // 4. Read Dictionary block
        let args_arc = Arc::new(args.clone());
        let dict = Dictionary::load_from_reader(reader, args_arc)?;

        // 5. Read quant_input flag
        let quant_input = utils::read_bool(reader)?;

        // 6. Load input matrix
        let (input_dense, input_quant) = if !quant_input {
            let dense = DenseMatrix::load(reader)?;
            (dense, None)
        } else {
            let qm = QuantMatrix::load(reader)?;
            // Use a zero-size placeholder for the dense input
            (DenseMatrix::new(0, 0), Some(qm))
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
        let qout = utils::read_bool(reader)?;
        args.qout = qout;

        // 8. Load output matrix
        let (output_dense, output_quant) = if quant_input && qout {
            let qm = QuantMatrix::load(reader)?;
            (DenseMatrix::new(0, 0), Some(qm))
        } else {
            let dense = DenseMatrix::load(reader)?;
            (dense, None)
        };

        // 9. Build the inference model.
        let input_arc = Arc::new(input_dense);
        let output_arc = Arc::new(output_dense);
        let label_counts = dict.get_counts(EntryType::Label);
        let word_counts = dict.get_counts(EntryType::Word);
        let target_counts = if args.model == ModelName::SUP {
            label_counts
        } else {
            word_counts
        };
        let loss = build_loss(&args, Arc::clone(&output_arc), &target_counts);
        let normalize_gradient = args.model == ModelName::SUP;
        let model = Model::new(Arc::clone(&input_arc), loss, normalize_gradient);

        Ok(FastText {
            args: Arc::new(args),
            dict,
            input: input_arc,
            output: output_arc,
            quant: quant_input,
            quant_input: input_quant,
            quant_output: output_quant,
            model,
            abort_flag: Arc::new(AtomicBool::new(false)),
            last_train_loss: 0.0,
            loss_tables: LossTables::new(),
        })
    }

    /// Load a model from a file path.
    pub fn load_model(path: impl AsRef<Path>) -> Result<Self> {
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
    /// 6. Input matrix (DenseMatrix or QuantMatrix)
    /// 7. qout (bool)
    /// 8. Output matrix (DenseMatrix or QuantMatrix)
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
        utils::write_bool(writer, self.quant)?;
        // 6. Input matrix
        if self.quant {
            if let Some(ref qm) = self.quant_input {
                qm.save(writer)?;
            } else {
                return Err(FastTextError::InvalidModel(
                    "quant=true but quant_input is None".to_string(),
                ));
            }
        } else {
            self.input.save(writer)?;
        }
        // 7. qout
        utils::write_bool(writer, self.args.qout)?;
        // 8. Output matrix
        if self.quant && self.args.qout {
            if let Some(ref qm) = self.quant_output {
                qm.save(writer)?;
            } else {
                return Err(FastTextError::InvalidModel(
                    "qout=true but quant_output is None".to_string(),
                ));
            }
        } else {
            self.output.save(writer)?;
        }
        Ok(())
    }

    /// Save the model to a file path.
    pub fn save_model(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = std::fs::File::create(path).map_err(FastTextError::IoError)?;
        let mut writer = BufWriter::new(file);
        self.save(&mut writer)?;
        // Explicitly flush before dropping so buffered-write errors are caught.
        writer.flush().map_err(FastTextError::IoError)?;
        Ok(())
    }

    // Getters / accessors

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

    /// Get a reference to the quantized input matrix, if present.
    pub fn quant_input(&self) -> Option<&QuantMatrix> {
        self.quant_input.as_ref()
    }

    /// Get a reference to the quantized output matrix, if present.
    pub fn quant_output(&self) -> Option<&QuantMatrix> {
        self.quant_output.as_ref()
    }

    /// Return the word vector for a given word.
    ///
    /// For in-vocabulary words the vector is the average of all stored subword
    /// IDs (which for `minn=0 / maxn=0` is just the word's own row).  For
    /// OOV words the subwords are computed on-the-fly; if there are no subwords
    /// (e.g. `bucket=0`) a zero vector is returned.
    pub fn get_word_vector(&self, word: &str) -> Vec<f32> {
        let dim = self.args.dim as usize;
        let mut result = vec![0.0f32; dim];
        self.get_word_vector_into(word, &mut result);
        result
    }

    /// Write the word vector for `word` into `out`.
    ///
    /// Like [`get_word_vector`] but avoids allocation by writing into a
    /// caller-provided buffer. `out` must have length equal to `self.get_dimension()`.
    ///
    /// # Panics
    /// Panics if `out.len() != self.get_dimension() as usize`.
    pub fn get_word_vector_into(&self, word: &str, out: &mut [f32]) {
        let dim = self.args.dim as usize;
        assert_eq!(
            out.len(),
            dim,
            "output buffer length must equal model dimension"
        );
        out.fill(0.0);
        let ids = self.dict.get_subwords_for_string(word);
        if ids.is_empty() {
            return;
        }
        let scale = 1.0 / ids.len() as f32;
        if self.quant {
            if let Some(ref qi) = self.quant_input {
                let mut vec = Vector::new(dim);
                for &id in &ids {
                    qi.add_row_to_vector(&mut vec, id, scale);
                }
                out.copy_from_slice(vec.data());
            }
        } else {
            for &id in &ids {
                let row = self.input.row(id as i64);
                for (r, &v) in out.iter_mut().zip(row.iter()) {
                    *r += v * scale;
                }
            }
        }
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
        let dim = self.args.dim as usize;
        let mut result = vec![0.0f32; dim];
        self.get_sentence_vector_into(sentence, &mut result);
        result
    }

    /// Write the sentence vector for `sentence` into `out`.
    ///
    /// Like [`get_sentence_vector`] but avoids allocation by writing into a
    /// caller-provided buffer. `out` must have length equal to `self.get_dimension()`.
    ///
    /// # Panics
    /// Panics if `out.len() != self.get_dimension() as usize`.
    pub fn get_sentence_vector_into(&self, sentence: &str, out: &mut [f32]) {
        let dim = self.args.dim as usize;
        assert_eq!(
            out.len(),
            dim,
            "output buffer length must equal model dimension"
        );
        out.fill(0.0);

        if self.args.model == ModelName::SUP {
            let mut words: Vec<i32> = Vec::new();
            let mut labels: Vec<i32> = Vec::new();
            self.dict
                .get_line_from_str(sentence, &mut words, &mut labels);

            if words.is_empty() {
                return;
            }

            let eos_id = self.dict.get_id(EOS);
            if let Some(eos_id) = eos_id {
                words.push(eos_id);
            }

            let count = words.len() as f32;
            if self.quant {
                if let Some(ref qi) = self.quant_input {
                    let mut vec = Vector::new(dim);
                    for &id in &words {
                        qi.add_row_to_vector(&mut vec, id, 1.0);
                    }
                    for (r, &v) in out.iter_mut().zip(vec.data().iter()) {
                        *r = v / count;
                    }
                }
            } else {
                for &id in &words {
                    let row = self.input.row(id as i64);
                    for (r, &v) in out.iter_mut().zip(row.iter()) {
                        *r += v;
                    }
                }
                for r in out.iter_mut() {
                    *r /= count;
                }
            }
        } else {
            let mut word_buf = vec![0.0f32; dim];
            let mut count = 0i32;
            for word in sentence.split_whitespace() {
                self.get_word_vector_into(word, &mut word_buf);
                let norm: f32 = word_buf.iter().map(|&v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for (r, &v) in out.iter_mut().zip(word_buf.iter()) {
                        *r += v / norm;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                let scale = 1.0 / count as f32;
                for r in out.iter_mut() {
                    *r *= scale;
                }
            }
        }
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
        self.args.dim
    }

    /// Return the word ID for the given word, or `None` if not in vocabulary.
    pub fn get_word_id(&self, word: &str) -> Option<i32> {
        self.dict.get_id(word)
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use crate::args::{Args, LossName, ModelName};

    // Helpers

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
        args.dim = 2;
        args.ws = 1;
        args.epoch = 1;
        args.min_count = 1;
        args.neg = 5;
        args.word_ngrams = 1;
        args.loss = LossName::SOFTMAX;
        args.model = ModelName::SUP;
        args.bucket = 0;
        args.minn = 0;
        args.maxn = 0;
        args.lr_update_rate = 100;
        args.t = 0.0001;
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

    // VAL-DICT-009: Magic number and version validation

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

    // VAL-DICT-010: Args block layout

    #[test]
    fn test_binary_io_args_block() {
        // Load model and verify args are correct
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(&buf);
        let model = FastText::load(&mut cursor).unwrap();

        assert_eq!(model.args().dim, 2);
        assert_eq!(model.args().ws, 1);
        assert_eq!(model.args().epoch, 1);
        assert_eq!(model.args().min_count, 1);
        assert_eq!(model.args().neg, 5);
        assert_eq!(model.args().word_ngrams, 1);
        assert_eq!(model.args().loss, LossName::SOFTMAX);
        assert_eq!(model.args().model, ModelName::SUP);
        assert_eq!(model.args().bucket, 0);
        assert_eq!(model.args().minn, 0);
        assert_eq!(model.args().maxn, 0);
        assert_eq!(model.args().lr_update_rate, 100);
        assert!((model.args().t - 0.0001).abs() < f64::EPSILON);
    }

    // VAL-DICT-011: Dictionary block layout

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
        assert_eq!(dict.get_id("</s>"), Some(0));
        assert_eq!(dict.get_id("__label__test"), Some(1));
        assert_eq!(dict.get_id("unknown"), None);
    }

    // VAL-DICT-012: Matrix blocks

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

    // VAL-DICT-013: Loading cooking.model.bin reference model

    // VAL-DICT-017: Invalid model rejection and backward compatibility

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
            model.args().maxn,
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
        args.dim = 2;
        args.ws = 1;
        args.epoch = 1;
        args.min_count = 1;
        args.neg = 5;
        args.word_ngrams = 1;
        args.loss = LossName::NS;
        args.model = ModelName::SG; // NOT supervised
        args.bucket = 100;
        args.minn = 3;
        args.maxn = 6; // should remain 6
        args.lr_update_rate = 100;
        args.t = 0.0001;
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
            model.args().maxn,
            6,
            "Version 11 non-supervised model should keep maxn=6"
        );
    }

    // Model save/load round-trip

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
        assert_eq!(model1.args().dim, model2.args().dim);
        assert_eq!(model1.args().epoch, model2.args().epoch);
        assert_eq!(model1.args().model, model2.args().model);
        assert_eq!(model1.args().loss, model2.args().loss);

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

    // VAL-DICT-009 specifically: test with constructed wrong-magic binary

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

    // VAL-DICT-014: Full model save/load round-trip (cooking model)

    // Fix: save_model explicitly flushes BufWriter

    // Fix: predict() does not panic for non-supervised models

    /// Verify that predict() returns an empty vec (not a panic) on a model
    /// with no labels, confirming the doc comment is no longer misleading.
    #[test]
    fn test_predict_non_supervised_model_no_panic() {
        // Build a minimal model with no labels (SG model).
        let mut buf = Vec::new();
        utils::write_i32(&mut buf, FASTTEXT_FILEFORMAT_MAGIC_INT32).unwrap();
        utils::write_i32(&mut buf, FASTTEXT_VERSION).unwrap();

        let mut args = Args::default();
        args.dim = 2;
        args.ws = 1;
        args.epoch = 1;
        args.min_count = 1;
        args.neg = 5;
        args.word_ngrams = 1;
        args.loss = LossName::NS;
        args.model = ModelName::SG;
        args.bucket = 0;
        args.minn = 0;
        args.maxn = 0;
        args.lr_update_rate = 100;
        args.t = 0.0001;
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

    // VAL-INF-007: predict() cooking model top-2 reference

    // VAL-INF-008: predict() probability values match C++

    // VAL-INF-009: predict() threshold filtering

    // VAL-INF-010: predict() edge cases

    // VAL-INF-011: predict_on_words() matches predict()

    // VAL-INF-018: Prediction determinism

    // VAL-INF-019: Thread safety for concurrent prediction

    // VAL-INF-020: Prediction probabilities validity

    // VAL-INF-012: get_word_vector() banana reference

    // VAL-INF-013: get_word_vector() unknown word behavior

    /// VAL-INF-013: Unknown word with maxn>0 returns non-zero vector via subword aggregation.
    ///
    /// Creates a small skip-gram model with minn=2, maxn=3, bucket=100 where all
    /// subword bucket rows in the input matrix are set to 1.0.  An OOV word must
    /// return a non-zero vector because its character n-grams map to those non-zero rows.
    #[test]
    fn test_get_word_vector_unknown_with_subwords() {
        let dim = 4usize;
        let bucket = 100i32;
        let minn = 2i32;
        let maxn = 3i32;
        let nwords = 1i32; // just </s>

        let mut buf = Vec::<u8>::new();

        // Magic + Version
        utils::write_i32(&mut buf, FASTTEXT_FILEFORMAT_MAGIC_INT32).unwrap();
        utils::write_i32(&mut buf, FASTTEXT_VERSION).unwrap();

        // Args: SG model with subwords enabled
        let mut args = Args::default();
        args.dim = dim as i32;
        args.ws = 1;
        args.epoch = 1;
        args.min_count = 1;
        args.neg = 1;
        args.word_ngrams = 1;
        args.loss = LossName::NS;
        args.model = ModelName::SG;
        args.bucket = bucket;
        args.minn = minn;
        args.maxn = maxn;
        args.lr_update_rate = 100;
        args.t = 0.0001;
        args.save(&mut buf).unwrap();

        // Dictionary: 1 word (</s>), 0 labels
        utils::write_i32(&mut buf, nwords).unwrap(); // size
        utils::write_i32(&mut buf, nwords).unwrap(); // nwords
        utils::write_i32(&mut buf, 0i32).unwrap(); // nlabels
        utils::write_i64(&mut buf, 10i64).unwrap(); // ntokens
        utils::write_i64(&mut buf, -1i64).unwrap(); // pruneidx_size = -1 (not pruned)
                                                    // Word 0: </s>
        buf.extend_from_slice(b"</s>\0");
        utils::write_i64(&mut buf, 10i64).unwrap(); // count
        buf.push(0u8); // EntryType::Word

        // quant_input = false
        buf.push(0u8);

        // Input matrix: (nwords + bucket) × dim = (1 + 100) × 4 = 101 × 4
        // Row 0 (</s>): zeros; rows 1..101 (subword buckets): all 1.0f32
        let n_rows = (nwords + bucket) as i64;
        let n_cols = dim as i64;
        utils::write_i64(&mut buf, n_rows).unwrap();
        utils::write_i64(&mut buf, n_cols).unwrap();
        for row in 0..n_rows {
            for _ in 0..n_cols {
                // Row 0 is the </s> word row (zero); rows 1+ are subword buckets (non-zero)
                let val = if row >= nwords as i64 { 1.0f32 } else { 0.0f32 };
                utils::write_f32(&mut buf, val).unwrap();
            }
        }

        // qout = false
        buf.push(0u8);

        // Output matrix: nwords × dim = 1 × 4
        utils::write_i64(&mut buf, nwords as i64).unwrap();
        utils::write_i64(&mut buf, n_cols).unwrap();
        for _ in 0..(nwords as i64 * n_cols) {
            utils::write_f32(&mut buf, 0.0f32).unwrap();
        }

        let mut cursor = Cursor::new(buf);
        let model = FastText::load(&mut cursor).expect("Subword model should load");

        // Verify model has subwords enabled
        assert_eq!(model.args().maxn, maxn, "Model should have maxn={}", maxn);
        assert_eq!(model.args().minn, minn, "Model should have minn={}", minn);

        // Test: OOV word "zyx" → character n-grams map to bucket rows [1..101]
        // which are all 1.0 → averaged result is non-zero
        let vec = model.get_word_vector("zyx");
        assert_eq!(vec.len(), dim, "Word vector should have {} dimensions", dim);
        let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            norm > 0.0,
            "OOV word 'zyx' with maxn={} should return non-zero vector \
             (subword aggregation from character n-grams), got norm={}",
            maxn,
            norm
        );

        // Also verify: for maxn=0 (the cooking model), an OOV word returns zero
        // (already tested in test_get_word_vector_unknown_zero, but confirmed here
        // by checking that our model DOES have maxn>0 and returns non-zero)
        assert!(
            model.args().maxn > 0,
            "Test model should have maxn>0 to exercise subword path"
        );
    }

    // VAL-INF-014: get_sentence_vector() behavior

    // VAL-INF-014 (supplementary): unsupervised model sentence vector normalization

    /// VAL-INF-014: Unsupervised (SG) model sentence vector is L2-normalized per word.
    ///
    /// For a model where model != SUP, get_sentence_vector normalizes each word
    /// vector by its L2 norm before adding to the running sum.  This means a
    /// single-word sentence returns a unit vector (norm ≈ 1.0).
    ///
    /// Creates a minimal SG model with one in-vocab word "hello" whose input
    /// vector is [2, 0, 0, 0] (norm=2, not a unit vector).  After normalization
    /// the sentence vector should be [1, 0, 0, 0] (norm=1.0).
    #[test]
    fn test_sentence_vector_unsupervised_normalization() {
        let dim = 4usize;
        let nwords = 2i32; // </s> and "hello"

        let mut buf = Vec::<u8>::new();

        // Magic + Version
        utils::write_i32(&mut buf, FASTTEXT_FILEFORMAT_MAGIC_INT32).unwrap();
        utils::write_i32(&mut buf, FASTTEXT_VERSION).unwrap();

        // Args: skip-gram model, no subwords (maxn=0), no buckets
        let mut args = Args::default();
        args.dim = dim as i32;
        args.ws = 1;
        args.epoch = 1;
        args.min_count = 1;
        args.neg = 1;
        args.word_ngrams = 1;
        args.loss = LossName::NS;
        args.model = ModelName::SG;
        args.bucket = 0;
        args.minn = 0;
        args.maxn = 0;
        args.lr_update_rate = 100;
        args.t = 0.0001;
        args.save(&mut buf).unwrap();

        // Dictionary: 2 words (</s> at index 0, "hello" at index 1), 0 labels
        utils::write_i32(&mut buf, nwords).unwrap(); // size
        utils::write_i32(&mut buf, nwords).unwrap(); // nwords
        utils::write_i32(&mut buf, 0i32).unwrap(); // nlabels
        utils::write_i64(&mut buf, 10i64).unwrap(); // ntokens
        utils::write_i64(&mut buf, -1i64).unwrap(); // pruneidx_size
                                                    // Word 0: </s>
        buf.extend_from_slice(b"</s>\0");
        utils::write_i64(&mut buf, 5i64).unwrap();
        buf.push(0u8); // EntryType::Word
                       // Word 1: "hello"
        buf.extend_from_slice(b"hello\0");
        utils::write_i64(&mut buf, 5i64).unwrap();
        buf.push(0u8); // EntryType::Word

        // quant_input = false
        buf.push(0u8);

        // Input matrix: 2 × 4 (nwords=2, bucket=0 → no extra rows)
        // Row 0 (</s>):  [0, 0, 0, 0]
        // Row 1 ("hello"): [2, 0, 0, 0]  ← non-unit vector (norm=2)
        utils::write_i64(&mut buf, 2i64).unwrap(); // m = 2
        utils::write_i64(&mut buf, 4i64).unwrap(); // n = 4
        for _ in 0..4 {
            utils::write_f32(&mut buf, 0.0f32).unwrap(); // row 0 (</s>)
        }
        utils::write_f32(&mut buf, 2.0f32).unwrap(); // row 1 [0]
        utils::write_f32(&mut buf, 0.0f32).unwrap(); // row 1 [1]
        utils::write_f32(&mut buf, 0.0f32).unwrap(); // row 1 [2]
        utils::write_f32(&mut buf, 0.0f32).unwrap(); // row 1 [3]

        // qout = false
        buf.push(0u8);

        // Output matrix: 2 × 4 (zeros)
        utils::write_i64(&mut buf, 2i64).unwrap();
        utils::write_i64(&mut buf, 4i64).unwrap();
        for _ in 0..8 {
            utils::write_f32(&mut buf, 0.0f32).unwrap();
        }

        let mut cursor = Cursor::new(buf);
        let model = FastText::load(&mut cursor).expect("SG model should load");

        // Confirm model type is unsupervised
        assert_eq!(
            model.args().model,
            ModelName::SG,
            "Model should be SG (unsupervised)"
        );

        // Verify the word vector of "hello" is [2, 0, 0, 0] (norm=2, not unit)
        let wv = model.get_word_vector("hello");
        assert_eq!(wv.len(), dim, "Word vector should have {} dims", dim);
        let wv_norm: f32 = wv.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (wv_norm - 2.0).abs() < 1e-4,
            "Word vector norm should be 2.0 (not normalized), got {}",
            wv_norm
        );

        // get_sentence_vector("hello") on unsupervised model:
        //   normalizes each word vector before averaging
        //   vec = [2,0,0,0] → normalized = [1,0,0,0]
        //   count = 1 → result = [1,0,0,0] * (1/1) = [1,0,0,0]
        //   norm = 1.0
        let sv = model.get_sentence_vector("hello");
        assert_eq!(sv.len(), dim, "Sentence vector should have {} dims", dim);

        let sv_norm: f32 = sv.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (sv_norm - 1.0).abs() < 1e-4,
            "Unsupervised model sentence vector for single-word input should be \
             L2-normalized (norm≈1.0), got norm={}",
            sv_norm
        );

        // Direction should be [1, 0, 0, 0]
        assert!(
            (sv[0] - 1.0).abs() < 1e-4,
            "sv[0] should be ≈1.0 (normalized direction), got {}",
            sv[0]
        );
        for i in 1..dim {
            assert!(
                sv[i].abs() < 1e-4,
                "sv[{}] should be ≈0.0, got {}",
                i,
                sv[i]
            );
        }

        // Contrast: the raw word vector has norm=2, proving normalization happened
        assert!(
            (wv_norm - sv_norm).abs() > 0.5,
            "Word vector norm ({}) should differ from sentence vector norm ({}) \
             confirming normalization was applied",
            wv_norm,
            sv_norm
        );
    }

    // VAL-INF-015: tokenize() correctness

    // VAL-INF-016: get_vocab() and get_labels() cooking model reference

    // VAL-INF-017: Metadata accessors

    // Training tests (VAL-TRAIN-001 through VAL-TRAIN-007)

    /// Small supervised training dataset: 2 classes, 30 examples.
    fn supervised_train_data() -> String {
        let mut data = String::new();
        for _ in 0..15 {
            data.push_str("__label__sports basketball player sport game team score win\n");
        }
        for _ in 0..15 {
            data.push_str("__label__food apple orange banana fruit eat cook recipe\n");
        }
        data
    }

    /// VAL-TRAIN-007: SGD update correctness.
    ///
    /// Verifies that supervised_fn performs a proper SGD update:
    /// the input weights change after one gradient step.
    #[test]
    fn test_train_sgd_update() {
        use crate::loss::SoftmaxLoss;
        use crate::matrix::DenseMatrix;
        use crate::model::{Model, State};

        let mut wi = DenseMatrix::new(2, 3);
        *wi.at_mut(0, 0) = 1.0;
        *wi.at_mut(1, 1) = 1.0;
        let wi = Arc::new(wi);

        let mut wo = DenseMatrix::new(2, 3);
        *wo.at_mut(0, 0) = 1.0;
        *wo.at_mut(1, 1) = 1.0;
        let wo = Arc::new(wo);

        let loss = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model = Model::new(Arc::clone(&wi), loss, true);
        let mut state = State::new(3, 2, 0);

        let wi_before: Vec<f32> = model.wi.data().to_vec();

        let line = vec![0i32, 1];
        let labels = vec![0i32];

        FastText::supervised_fn(&model, &mut state, 0.1, &line, &labels, false);

        let wi_after: Vec<f32> = model.wi.data().to_vec();

        let changed = wi_before
            .iter()
            .zip(wi_after.iter())
            .any(|(b, a)| (b - a).abs() > 1e-9);
        assert!(
            changed,
            "Input weights should be updated by SGD: before={:?} after={:?}",
            wi_before, wi_after
        );
    }

    /// Test supervised_fn with empty line or labels is a no-op.
    #[test]
    fn test_train_supervised_fn_empty_noop() {
        use crate::loss::SoftmaxLoss;
        use crate::matrix::DenseMatrix;
        use crate::model::{Model, State};

        let wi = Arc::new(DenseMatrix::new(2, 3));
        let wo = Arc::new(DenseMatrix::new(2, 3));
        let loss = Box::new(SoftmaxLoss::new(Arc::clone(&wo)));
        let model = Model::new(Arc::clone(&wi), loss, true);
        let mut state = State::new(3, 2, 0);

        let wi_before: Vec<f32> = model.wi.data().to_vec();

        FastText::supervised_fn(&model, &mut state, 0.1, &[], &[0], false);
        assert_eq!(model.wi.data(), wi_before.as_slice(), "No-op on empty line");

        FastText::supervised_fn(&model, &mut state, 0.1, &[0], &[], false);
        assert_eq!(
            model.wi.data(),
            wi_before.as_slice(),
            "No-op on empty labels"
        );
    }

    // Parallel training tests (training-parallel feature)

    // VAL-TRAIN-009: Meter metrics computation (test command integration)
    // VAL-TRAIN-008: Training save/load round-trip

    // VAL-TRAIN-011: Training edge cases

    // VAL-TRAIN-012: min_count filtering

    // VAL-TRAIN-013: Training loss decreases over epochs

    // VAL-TRAIN-014: Pretrained vectors loading

    /// Create a uniquely-named temp text file for tests that need isolation.
    fn write_unique_temp_file(content: &str, tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static UNIQUE_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = UNIQUE_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "fasttext_{}_{}_{}.txt",
            tag,
            std::process::id(),
            id
        ));
        std::fs::write(&path, content).expect("Failed to write unique temp file");
        path
    }

    // Integration test matching the "train_integration" verification step

    // VAL-QUANT-001 through VAL-QUANT-008: Quantization tests

    /// Helper: train a small supervised model for quantization tests.
    fn train_small_supervised(dim: i32, epoch: i32, bucket: i32) -> (FastText, std::path::PathBuf) {
        let data = supervised_train_data();
        let path = write_unique_temp_file(&data, "quant_train");
        let path_str = path.to_str().unwrap().to_string();

        let mut args = Args::default();
        args.input = path_str;
        args.output = "/dev/null".to_string();
        args.apply_supervised_defaults();
        args.dim = dim;
        args.epoch = epoch;
        args.min_count = 1;
        args.lr = 0.1;
        args.bucket = bucket;
        args.thread = 1;
        args.seed = 42;

        let model = FastText::train(args).expect("Training should succeed");
        (model, path)
    }

    /// VAL-QUANT-005: cutoff > 0 prunes vocabulary; cutoff = 0 preserves it.
    #[test]
    fn test_quantize_cutoff() {
        let (mut model_cutoff, train_path) = train_small_supervised(16, 5, 0);
        let nwords_before = model_cutoff.dict().nwords();
        std::fs::remove_file(&train_path).ok();

        // cutoff = 0: zero cutoff skips pruning
        let mut model_zero = model_cutoff.clone_for_test();
        {
            let mut qargs = Args::default();
            qargs.dsub = 2;
            qargs.cutoff = 0;
            model_zero
                .quantize(&qargs)
                .expect("zero cutoff quantize should succeed");
        }
        // Zero cutoff: vocabulary size should be unchanged
        assert_eq!(
            model_zero.dict().nwords(),
            nwords_before,
            "cutoff=0 should not prune vocabulary"
        );

        // cutoff = positive smaller than nwords: vocabulary should be pruned
        let cutoff = (nwords_before as usize / 2).max(1);
        {
            let mut qargs = Args::default();
            qargs.dsub = 2;
            qargs.cutoff = cutoff;
            model_cutoff
                .quantize(&qargs)
                .expect("positive cutoff quantize should succeed");
        }
        let nwords_after = model_cutoff.dict().nwords();
        assert!(
            nwords_after < nwords_before,
            "cutoff={} should prune vocabulary: before={}, after={}",
            cutoff,
            nwords_before,
            nwords_after
        );
    }

    /// VAL-QUANT-006 (part 1): qnorm flag works correctly.
    #[test]
    fn test_quantize_qnorm() {
        let (mut model, train_path) = train_small_supervised(16, 5, 0);
        std::fs::remove_file(&train_path).ok();

        let mut qargs = Args::default();
        qargs.dsub = 2;
        qargs.qnorm = true;

        model
            .quantize(&qargs)
            .expect("qnorm quantize should succeed");
        assert!(
            model.is_quant(),
            "is_quant() should be true with qnorm=true"
        );

        // quant_input should have qnorm=true
        assert!(
            model.quant_input.as_ref().unwrap().qnorm,
            "quant_input should have qnorm=true"
        );

        // Should still produce valid predictions
        let preds = model.predict("basketball player sport game", 1, 0.0);
        assert!(!preds.is_empty(), "qnorm model should produce predictions");
        assert!(
            preds[0].prob.is_finite() && preds[0].prob > 0.0,
            "qnorm prediction prob should be valid"
        );
    }

    /// VAL-QUANT-006 (part 2): qout flag works correctly.
    #[test]
    fn test_quantize_qout() {
        let (mut model, train_path) = train_small_supervised(16, 5, 0);
        std::fs::remove_file(&train_path).ok();

        let mut qargs = Args::default();
        qargs.dsub = 2;
        qargs.qout = true;

        model
            .quantize(&qargs)
            .expect("qout quantize should succeed");
        assert!(model.is_quant(), "is_quant() should be true with qout=true");
        assert!(model.args().qout, "args.qout should be true");

        // quant_output should be set
        assert!(
            model.quant_output.is_some(),
            "quant_output should be set when qout=true"
        );

        // Should still produce valid predictions
        let preds = model.predict("basketball player sport game", 1, 0.0);
        assert!(!preds.is_empty(), "qout model should produce predictions");
        assert!(
            preds[0].prob.is_finite() && preds[0].prob > 0.0,
            "qout prediction prob should be valid"
        );
    }

    // Fix tests: cutoff pruning row alignment

    // Fix tests: qout respects qnorm flag

    /// Verify that qout=true + qnorm=true sets qnorm on the output QuantMatrix.
    #[test]
    fn test_quantize_qout_respects_qnorm() {
        let (mut model, train_path) = train_small_supervised(16, 5, 0);
        std::fs::remove_file(&train_path).ok();

        let mut qargs = Args::default();
        qargs.dsub = 2;
        qargs.qout = true;
        qargs.qnorm = true;

        model
            .quantize(&qargs)
            .expect("qout+qnorm quantize should succeed");
        assert!(model.is_quant(), "is_quant() should be true");
        assert!(model.args().qout, "args.qout should be true");

        // Both input and output QuantMatrix should have qnorm=true.
        assert!(
            model.quant_input.as_ref().unwrap().qnorm,
            "quant_input should have qnorm=true when qnorm flag is set"
        );
        assert!(
            model.quant_output.as_ref().unwrap().qnorm,
            "quant_output should have qnorm=true when qout+qnorm flags are set"
        );

        // Should still produce valid predictions.
        let preds = model.predict("basketball player sport game", 1, 0.0);
        assert!(
            !preds.is_empty(),
            "qout+qnorm model should produce predictions"
        );
        assert!(
            preds[0].prob.is_finite() && preds[0].prob >= 0.0 && preds[0].prob <= 1.0,
            "qout+qnorm prediction prob should be in [0, 1]"
        );
    }

    /// Verify that qout=true + qnorm=false keeps qnorm=false on the output QuantMatrix.
    #[test]
    fn test_quantize_qout_false_qnorm_not_set() {
        let (mut model, train_path) = train_small_supervised(16, 5, 0);
        std::fs::remove_file(&train_path).ok();

        let mut qargs = Args::default();
        qargs.dsub = 2;
        qargs.qout = true;
        // qnorm is false by default

        model
            .quantize(&qargs)
            .expect("qout quantize should succeed");
        assert!(model.is_quant(), "is_quant() should be true");

        // quant_output should have qnorm=false (the default).
        assert!(
            !model.quant_output.as_ref().unwrap().qnorm,
            "quant_output should have qnorm=false when qnorm flag is not set"
        );
    }
}

// Helper trait for tests: clone-like for FastText (creates a copy via save/load)
impl FastText {
    #[cfg(test)]
    pub fn clone_for_test(&self) -> Self {
        let mut buf = Vec::new();
        self.save(&mut buf).expect("save for clone");
        let mut cursor = std::io::Cursor::new(buf);
        FastText::load(&mut cursor).expect("load for clone")
    }
}
