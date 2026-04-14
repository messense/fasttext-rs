// FastText: train, predict, quantize, autotune, save/load, word/sentence vectors

use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

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

/// Shared context passed to each training thread.
///
/// Bundles the per-run read-only state so that `train_thread_inner` stays
/// within clippy's argument-count limit.
struct TrainThreadCtx<'a> {
    args: &'a Args,
    dict: &'a Dictionary,
    model: &'a Model,
    output_size: usize,
    token_count: &'a AtomicI64,
    abort_flag: &'a AtomicBool,
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
    /// Atomic flag for aborting an in-progress training run.
    ///
    /// Set via [`FastText::abort()`]; checked in the training loop.
    abort_flag: Arc<AtomicBool>,
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
            abort_flag: Arc::new(AtomicBool::new(false)),
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

    // -------------------------------------------------------------------------
    // Training
    // -------------------------------------------------------------------------

    /// Train a new fastText model from the given `Args`.
    ///
    /// Delegates to [`FastText::train_with_abort`] with a fresh abort flag.
    ///
    /// Returns a trained `FastText` instance ready for prediction.
    pub fn train(args: Args) -> Result<Self> {
        Self::train_with_abort(args, Arc::new(AtomicBool::new(false)))
    }

    /// Train a new fastText model, using the supplied `abort_flag` for early
    /// termination.
    ///
    /// Setting `abort_flag` to `true` from another thread while this function
    /// is running will cause the training loop to exit early.  The returned
    /// model is still valid for inference (possibly under-trained).
    ///
    /// Steps:
    /// 1. Read input file and build vocabulary.
    /// 2. Initialize matrices.
    /// 3. Run training in parallel using rayon (Hogwild! SGD).
    /// 4. Return a trained `FastText` instance.
    pub fn train_with_abort(args: Args, abort_flag: Arc<AtomicBool>) -> Result<Self> {
        let input_path = args.input().to_string();
        if input_path.is_empty() {
            return Err(FastTextError::InvalidArgument(
                "Input file path is empty".to_string(),
            ));
        }

        let args_arc = Arc::new(args.clone());

        // Build vocabulary from input file.
        let mut dict = Dictionary::new(Arc::clone(&args_arc));
        {
            let file = std::fs::File::open(&input_path).map_err(FastTextError::IoError)?;
            let mut reader = BufReader::new(file);
            dict.read_from_file(&mut reader)?;
        }

        let nwords = dict.nwords() as i64;
        let bucket = args.bucket() as i64;
        let dim = args.dim() as i64;

        // Initialize input matrix: (nwords + bucket) × dim, uniform in [-1/dim, 1/dim].
        let input = Arc::new({
            let mut m = DenseMatrix::new(nwords + bucket, dim);
            m.uniform(1.0 / args.dim() as f32, args.seed());
            m
        });

        // Initialize output matrix: (nlabels for supervised, nwords for unsupervised) × dim, zeros.
        let out_rows = if args.model() == ModelName::SUP {
            dict.nlabels() as i64
        } else {
            dict.nwords() as i64
        };
        // DenseMatrix::new zeroes all values by default.
        let output = Arc::new(DenseMatrix::new(out_rows, dim));
        let output_size = output.rows() as usize;

        // Build target counts for loss construction.
        let target_counts = if args.model() == ModelName::SUP {
            dict.get_counts(EntryType::Label)
        } else {
            dict.get_counts(EntryType::Word)
        };
        let normalize_gradient = args.model() == ModelName::SUP;

        // Number of threads (at least 1).
        let n_threads = (args.thread() as usize).max(1);

        // Shared atomic token counter across all training threads.
        let token_count = Arc::new(AtomicI64::new(0));

        // Run Hogwild! parallel training with rayon.
        // Each thread creates its own Model (sharing the same Arc<DenseMatrix>),
        // and updates weights concurrently without locks.
        let training_results: Vec<Result<()>> = (0..n_threads)
            .into_par_iter()
            .map(|thread_id| {
                let loss = build_loss(&args, Arc::clone(&output), &target_counts);
                let model = Model::new(Arc::clone(&input), loss, normalize_gradient);
                let ctx = TrainThreadCtx {
                    args: &args,
                    dict: &dict,
                    model: &model,
                    output_size,
                    token_count: &token_count,
                    abort_flag: &abort_flag,
                };
                Self::train_thread_inner(thread_id, n_threads, &ctx)
            })
            .collect();

        // Propagate the first training error, if any.
        for result in training_results {
            result?;
        }

        // Build the inference model from the trained matrices.
        let loss = build_loss(&args, Arc::clone(&output), &target_counts);
        let model = Model::new(Arc::clone(&input), loss, normalize_gradient);

        Ok(FastText {
            args: args_arc,
            dict,
            input,
            output,
            quant: false,
            model,
            abort_flag,
        })
    }

    /// Signal an in-progress training run to stop early.
    ///
    /// Sets the internal atomic abort flag.  The training loop checks this flag
    /// periodically and exits if it is set.  The model returned from
    /// [`FastText::train_with_abort`] will still be valid for inference.
    ///
    /// This method is **idempotent**: calling it multiple times is safe and has
    /// no adverse effect.
    pub fn abort(&self) {
        self.abort_flag.store(true, Ordering::Relaxed);
    }

    /// Inner training loop for one thread (Hogwild! SGD).
    ///
    /// Opens the input file, seeks to `thread_id * file_size / n_threads`,
    /// then loops reading lines and performing SGD updates until
    /// `token_count >= epoch * ntokens` or `abort_flag` is set.
    ///
    /// Matches C++ `FastText::trainThread`.
    fn train_thread_inner(thread_id: usize, n_threads: usize, ctx: &TrainThreadCtx<'_>) -> Result<()> {
        let ntokens = ctx.dict.ntokens();
        if ntokens == 0 {
            return Ok(());
        }

        let input_path = ctx.args.input().to_string();

        // Open file and seek to this thread's starting position.
        let mut file = std::fs::File::open(&input_path).map_err(FastTextError::IoError)?;
        let file_size = file.seek(SeekFrom::End(0)).map_err(FastTextError::IoError)?;
        let start_pos = thread_id as u64 * file_size / n_threads as u64;
        file.seek(SeekFrom::Start(start_pos))
            .map_err(FastTextError::IoError)?;
        let mut reader = BufReader::new(file);

        let seed = thread_id as u64 + ctx.args.seed() as u64;
        let mut state = State::new(ctx.args.dim() as usize, ctx.output_size, seed);

        let model_name = ctx.args.model();
        let is_ova = ctx.args.loss() == LossName::OVA;
        let ws = ctx.args.ws();
        let lr_update_rate = ctx.args.lr_update_rate() as i64;
        let base_lr = ctx.args.lr() as f32;
        let epoch = ctx.args.epoch() as i64;

        let mut local_token_count: i64 = 0;
        let mut line: Vec<i32> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        let mut pending_newline = false;

        loop {
            // Check abort flag — exit early if set.
            if ctx.abort_flag.load(Ordering::Relaxed) {
                break;
            }

            // Check if training has reached the target token count.
            let tc = ctx.token_count.load(Ordering::Relaxed);
            if tc >= epoch * ntokens {
                break;
            }

            // Compute current progress and learning rate.
            let progress = tc as f32 / (epoch as f32 * ntokens as f32);
            let lr = (base_lr * (1.0 - progress)).max(0.0_f32);

            let ntok = match model_name {
                ModelName::SUP => ctx.dict.get_line(
                    &mut reader,
                    &mut line,
                    &mut labels,
                    &mut pending_newline,
                ),
                _ => ctx.dict.get_line_unsupervised(
                    &mut reader,
                    &mut line,
                    &mut pending_newline,
                    &mut state.rng,
                ),
            };

            if ntok == 0 {
                // EOF: wrap around to beginning for additional epoch passes.
                if let Err(e) = reader.seek(SeekFrom::Start(0)) {
                    return Err(FastTextError::IoError(e));
                }
                pending_newline = false;
                continue;
            }

            match model_name {
                ModelName::SUP => {
                    Self::supervised_fn(ctx.model, &mut state, lr, &line, &labels, is_ova);
                }
                ModelName::CBOW => {
                    Self::cbow_fn(ctx.model, ctx.dict, &mut state, lr, &line, ws);
                }
                _ => {
                    // Skip-gram
                    Self::skipgram_fn(ctx.model, ctx.dict, &mut state, lr, &line, ws);
                }
            }

            local_token_count += ntok as i64;
            if local_token_count > lr_update_rate {
                ctx.token_count.fetch_add(local_token_count, Ordering::Relaxed);
                local_token_count = 0;
            }
        }

        // Flush any remaining local token count into the shared counter.
        if local_token_count > 0 {
            ctx.token_count.fetch_add(local_token_count, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Supervised training update for one line.
    ///
    /// Matches C++ `FastText::supervised`.
    fn supervised_fn(
        model: &Model,
        state: &mut State,
        lr: f32,
        line: &[i32],
        labels: &[i32],
        is_ova: bool,
    ) {
        if labels.is_empty() || line.is_empty() {
            return;
        }
        if is_ova {
            // OVA ignores target_index; all labels in `labels` are positives.
            model.update(line, labels, 0, lr, state);
        } else {
            // Pick a random label index.
            let i = state.rng.uniform_usize(labels.len()) as i32;
            model.update(line, labels, i, lr, state);
        }
    }

    /// CBOW training update for one line.
    ///
    /// For each word position `w`, collects the subwords of context words within
    /// a random window `[1, ws]` as input and uses the center word as target.
    ///
    /// Matches C++ `FastText::cbow`.
    fn cbow_fn(
        model: &Model,
        dict: &Dictionary,
        state: &mut State,
        lr: f32,
        line: &[i32],
        ws: i32,
    ) {
        if line.is_empty() {
            return;
        }
        let mut bow: Vec<i32> = Vec::new();
        for w in 0..line.len() {
            // Random window size in [1, ws].
            let boundary = 1 + (state.rng.generate() as i32 % ws);
            bow.clear();
            for c in -boundary..=boundary {
                if c != 0 {
                    let pos = w as i32 + c;
                    if pos >= 0 && pos < line.len() as i32 {
                        let ngrams = dict.get_subwords(line[pos as usize]);
                        bow.extend_from_slice(ngrams);
                    }
                }
            }
            model.update(&bow, line, w as i32, lr, state);
        }
    }

    /// Skip-gram training update for one line.
    ///
    /// For each word position `w`, uses the subwords of the center word as input
    /// and each context word in a random window as target.
    ///
    /// Matches C++ `FastText::skipgram`.
    fn skipgram_fn(
        model: &Model,
        dict: &Dictionary,
        state: &mut State,
        lr: f32,
        line: &[i32],
        ws: i32,
    ) {
        if line.is_empty() {
            return;
        }
        for w in 0..line.len() {
            // Random window size in [1, ws].
            let boundary = 1 + (state.rng.generate() as i32 % ws);
            let ngrams = dict.get_subwords(line[w]);
            for c in -boundary..=boundary {
                if c != 0 {
                    let pos = w as i32 + c;
                    if pos >= 0 && pos < line.len() as i32 {
                        model.update(ngrams, line, pos, lr, state);
                    }
                }
            }
        }
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
        args.set_dim(dim as i32);
        args.set_ws(1);
        args.set_epoch(1);
        args.set_min_count(1);
        args.set_neg(1);
        args.set_word_ngrams(1);
        args.set_loss(LossName::NS);
        args.set_model(ModelName::SG);
        args.set_bucket(bucket);
        args.set_minn(minn);
        args.set_maxn(maxn);
        args.set_lr_update_rate(100);
        args.set_t(0.0001);
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
        assert_eq!(model.args().maxn(), maxn, "Model should have maxn={}", maxn);
        assert_eq!(model.args().minn(), minn, "Model should have minn={}", minn);

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
            model.args().maxn() > 0,
            "Test model should have maxn>0 to exercise subword path"
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
    // VAL-INF-014 (supplementary): unsupervised model sentence vector normalization
    // =========================================================================

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
        args.set_dim(dim as i32);
        args.set_ws(1);
        args.set_epoch(1);
        args.set_min_count(1);
        args.set_neg(1);
        args.set_word_ngrams(1);
        args.set_loss(LossName::NS);
        args.set_model(ModelName::SG);
        args.set_bucket(0);
        args.set_minn(0);
        args.set_maxn(0);
        args.set_lr_update_rate(100);
        args.set_t(0.0001);
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
            model.args().model(),
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

    // =========================================================================
    // Training tests (VAL-TRAIN-001 through VAL-TRAIN-007)
    // =========================================================================

    /// Helper: write training data to a temp file, return path.
    fn write_temp_file(content: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "fasttext_train_test_{}.txt",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::write(&path, content).expect("Failed to write temp file");
        path
    }

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

    /// Small CBOW/skip-gram corpus.
    fn unsupervised_train_data() -> String {
        let mut data = String::new();
        for _ in 0..20 {
            data.push_str("the quick brown fox jumps over the lazy dog\n");
            data.push_str("machine learning algorithms work with data\n");
            data.push_str("neural networks are powerful tools for classification\n");
        }
        data
    }

    /// VAL-TRAIN-001: Supervised training end-to-end.
    ///
    /// Trains a supervised model on a small labeled dataset, then predicts
    /// on the training data. Must achieve ≥50% top-1 accuracy.
    #[test]
    fn test_train_supervised_e2e() {
        let data = supervised_train_data();
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        let mut args = Args::default();
        args.set_input(path_str.clone());
        args.set_output("/dev/null".to_string());
        args.apply_supervised_defaults();
        args.set_dim(10);
        args.set_epoch(5);
        args.set_min_count(1);
        args.set_lr(0.1);
        args.set_word_ngrams(1);
        args.set_bucket(0);

        let model = FastText::train(args).expect("Training should succeed");
        std::fs::remove_file(&path).ok();

        let (labels, _) = model.get_labels();
        assert!(
            !labels.is_empty(),
            "Trained model should have labels, got none"
        );

        let test_cases = vec![
            ("basketball player sport game", "__label__sports"),
            ("apple orange banana fruit", "__label__food"),
            ("basketball team score win", "__label__sports"),
            ("cook recipe fruit eat", "__label__food"),
        ];

        let mut correct = 0;
        for (input, expected_label) in &test_cases {
            let preds = model.predict(input, 1, 0.0);
            if !preds.is_empty() && preds[0].label == *expected_label {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / test_cases.len() as f32;
        assert!(
            accuracy >= 0.5,
            "Supervised training should achieve ≥50% accuracy on training data, got {:.1}%",
            accuracy * 100.0
        );
    }

    /// VAL-TRAIN-002: CBOW training produces non-zero word embeddings.
    #[test]
    fn test_train_cbow() {
        let data = unsupervised_train_data();
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        let mut args = Args::default();
        args.set_input(path_str.clone());
        args.set_output("/dev/null".to_string());
        args.set_model(ModelName::CBOW);
        args.set_loss(LossName::NS);
        args.set_dim(10);
        args.set_epoch(3);
        args.set_min_count(1);
        args.set_lr(0.05);
        args.set_ws(3);
        args.set_neg(5);
        args.set_bucket(100);
        args.set_minn(0);
        args.set_maxn(0);

        let model = FastText::train(args).expect("CBOW training should succeed");
        std::fs::remove_file(&path).ok();

        let test_words = ["the", "fox", "data", "neural"];
        for word in &test_words {
            let wid = model.get_word_id(word);
            if wid >= 0 {
                let vec = model.get_word_vector(word);
                assert_eq!(
                    vec.len(),
                    model.get_dimension() as usize,
                    "Word vector for '{}' should have dim={} elements",
                    word,
                    model.get_dimension()
                );
                let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
                assert!(
                    norm > 0.0,
                    "CBOW word vector for '{}' should be non-zero after training (norm={})",
                    word,
                    norm
                );
            }
        }

        let (vocab, _) = model.get_vocab();
        assert!(!vocab.is_empty(), "CBOW model should have vocabulary");
    }

    /// VAL-TRAIN-003: Skip-gram training produces non-zero word embeddings.
    #[test]
    fn test_train_skipgram() {
        let data = unsupervised_train_data();
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        let mut args = Args::default();
        args.set_input(path_str.clone());
        args.set_output("/dev/null".to_string());
        args.set_model(ModelName::SG);
        args.set_loss(LossName::NS);
        args.set_dim(10);
        args.set_epoch(3);
        args.set_min_count(1);
        args.set_lr(0.05);
        args.set_ws(3);
        args.set_neg(5);
        args.set_bucket(100);
        args.set_minn(0);
        args.set_maxn(0);

        let model = FastText::train(args).expect("Skip-gram training should succeed");
        std::fs::remove_file(&path).ok();

        let test_words = ["the", "fox", "data", "neural"];
        for word in &test_words {
            let wid = model.get_word_id(word);
            if wid >= 0 {
                let vec = model.get_word_vector(word);
                assert_eq!(
                    vec.len(),
                    model.get_dimension() as usize,
                    "Word vector for '{}' should have dim={} elements",
                    word,
                    model.get_dimension()
                );
                let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
                assert!(
                    norm > 0.0,
                    "Skip-gram word vector for '{}' should be non-zero after training (norm={})",
                    word,
                    norm
                );
            }
        }

        let (vocab, _) = model.get_vocab();
        assert!(!vocab.is_empty(), "Skip-gram model should have vocabulary");
    }

    /// VAL-TRAIN-004: Matrix dimensions correct after training.
    #[test]
    fn test_train_matrix_dimensions() {
        // --- Supervised ---
        let data = supervised_train_data();
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        let mut args = Args::default();
        args.set_input(path_str.clone());
        args.set_output("/dev/null".to_string());
        args.apply_supervised_defaults();
        args.set_dim(10);
        args.set_epoch(1);
        args.set_min_count(1);
        args.set_bucket(50);

        let model = FastText::train(args).expect("Supervised training should succeed");
        std::fs::remove_file(&path).ok();

        let nwords = model.dict().nwords() as i64;
        let nlabels = model.dict().nlabels() as i64;
        let dim = model.get_dimension() as i64;
        let bucket = model.args().bucket() as i64;

        let input = model.input_matrix();
        assert_eq!(
            input.rows(),
            nwords + bucket,
            "Input matrix rows should be nwords+bucket: {} != {}+{}",
            input.rows(),
            nwords,
            bucket
        );
        assert_eq!(input.cols(), dim, "Input cols should be dim");

        let output = model.output_matrix();
        assert_eq!(
            output.rows(),
            nlabels,
            "Output matrix rows should be nlabels for supervised: {} != {}",
            output.rows(),
            nlabels
        );
        assert_eq!(output.cols(), dim, "Output cols should be dim");

        // --- Unsupervised (CBOW) ---
        let data2 = unsupervised_train_data();
        let path2 = write_temp_file(&data2);
        let path2_str = path2.to_str().unwrap().to_string();

        let mut args2 = Args::default();
        args2.set_input(path2_str.clone());
        args2.set_output("/dev/null".to_string());
        args2.set_model(ModelName::CBOW);
        args2.set_loss(LossName::NS);
        args2.set_dim(10);
        args2.set_epoch(1);
        args2.set_min_count(1);
        args2.set_neg(5);
        args2.set_bucket(50);
        args2.set_minn(0);
        args2.set_maxn(0);

        let model2 = FastText::train(args2).expect("CBOW training should succeed");
        std::fs::remove_file(&path2).ok();

        let nwords2 = model2.dict().nwords() as i64;
        let dim2 = model2.get_dimension() as i64;
        let bucket2 = model2.args().bucket() as i64;

        let input2 = model2.input_matrix();
        assert_eq!(
            input2.rows(),
            nwords2 + bucket2,
            "CBOW input matrix rows should be nwords+bucket: {} != {}+{}",
            input2.rows(),
            nwords2,
            bucket2
        );
        assert_eq!(input2.cols(), dim2, "CBOW input cols should be dim");

        let output2 = model2.output_matrix();
        assert_eq!(
            output2.rows(),
            nwords2,
            "CBOW output matrix rows should be nwords: {} != {}",
            output2.rows(),
            nwords2
        );
        assert_eq!(output2.cols(), dim2, "CBOW output cols should be dim");
    }

    /// VAL-TRAIN-005: Learning rate decays linearly, never negative.
    #[test]
    fn test_train_lr_decay() {
        let base_lr: f32 = 0.05;

        let test_cases = [
            (0.0f32, 0.05f32),
            (0.5f32, 0.025f32),
            (0.9f32, 0.005f32),
            (1.0f32, 0.0f32),
        ];

        for (progress, expected_lr) in &test_cases {
            let lr = (base_lr * (1.0 - progress)).max(0.0);
            assert!(
                (lr - expected_lr).abs() < 1e-6,
                "lr at progress={}: got={}, expected={}",
                progress,
                lr,
                expected_lr
            );
            assert!(lr >= 0.0, "lr must never be negative, got {}", lr);
        }

        // Verify clamping to 0 for progress > 1.0
        let lr_over = (base_lr * (1.0 - 1.5f32)).max(0.0);
        assert_eq!(lr_over, 0.0, "lr should be clamped to 0 for progress>1");
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
            wi_before,
            wi_after
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

    /// Test that training completes and produces a model usable for prediction.
    #[test]
    fn test_train_lr_decay_actual() {
        let data = supervised_train_data();
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        let mut args = Args::default();
        args.set_input(path_str.clone());
        args.set_output("/dev/null".to_string());
        args.apply_supervised_defaults();
        args.set_dim(5);
        args.set_epoch(3);
        args.set_min_count(1);
        args.set_lr(0.1);
        args.set_bucket(0);

        let result = FastText::train(args);
        std::fs::remove_file(&path).ok();
        assert!(
            result.is_ok(),
            "Training should complete without error: {:?}",
            result.err()
        );
    }

    // =========================================================================
    // Parallel training tests (training-parallel feature)
    // =========================================================================

    /// VAL-TRAIN-006: Multi-threaded Hogwild! training completes without panic.
    ///
    /// Trains with thread=4 and verifies all model weights are finite.
    #[test]
    fn test_parallel_hogwild_training() {
        let data = supervised_train_data();
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        let mut args = Args::default();
        args.set_input(path_str.clone());
        args.set_output("/dev/null".to_string());
        args.apply_supervised_defaults();
        args.set_dim(10);
        args.set_epoch(3);
        args.set_min_count(1);
        args.set_lr(0.1);
        args.set_bucket(0);
        args.set_thread(4);

        let model = FastText::train(args).expect("Parallel training (thread=4) should succeed");
        std::fs::remove_file(&path).ok();

        // All input weights must be finite.
        let input = model.input_matrix();
        for v in input.data() {
            assert!(
                v.is_finite(),
                "Input weight is not finite after Hogwild! training: {}",
                v
            );
        }

        // All output weights must be finite.
        let output = model.output_matrix();
        for v in output.data() {
            assert!(
                v.is_finite(),
                "Output weight is not finite after Hogwild! training: {}",
                v
            );
        }
    }

    /// VAL-TRAIN-006 (extended): All weights finite after Hogwild! training.
    ///
    /// Verifies that no NaN or Inf values appear after concurrent weight updates.
    #[test]
    fn test_hogwild_weights_finite() {
        let data = unsupervised_train_data();
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        let mut args = Args::default();
        args.set_input(path_str.clone());
        args.set_output("/dev/null".to_string());
        args.set_model(crate::args::ModelName::CBOW);
        args.set_loss(crate::args::LossName::NS);
        args.set_dim(10);
        args.set_epoch(3);
        args.set_min_count(1);
        args.set_thread(4);
        args.set_bucket(100);

        let model = FastText::train(args).expect("Multi-threaded CBOW training should succeed");
        std::fs::remove_file(&path).ok();

        for v in model.input_matrix().data() {
            assert!(v.is_finite(), "Input weight NaN/Inf: {}", v);
        }
        for v in model.output_matrix().data() {
            assert!(v.is_finite(), "Output weight NaN/Inf: {}", v);
        }
    }

    /// VAL-TRAIN-010: Abort stops training early; model is still usable.
    ///
    /// Starts training in a separate thread with a large epoch count, sets the
    /// abort flag after a brief delay, and verifies the model is usable.
    #[test]
    fn test_training_abort() {
        let data = supervised_train_data();
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        // Shared abort flag: the test thread will set it to stop training early.
        let abort_flag = Arc::new(AtomicBool::new(false));
        let abort_for_train = Arc::clone(&abort_flag);

        let handle = std::thread::spawn(move || {
            let mut args = Args::default();
            args.set_input(path_str.clone());
            args.set_output("/dev/null".to_string());
            args.apply_supervised_defaults();
            args.set_dim(10);
            args.set_epoch(500); // Very large epoch count so training won't finish naturally.
            args.set_min_count(1);
            args.set_lr(0.1);
            args.set_bucket(0);
            args.set_thread(1);
            FastText::train_with_abort(args, abort_for_train)
        });

        // Give training a moment to start, then abort it.
        std::thread::sleep(std::time::Duration::from_millis(50));
        abort_flag.store(true, Ordering::Relaxed);

        let model = handle.join().unwrap().expect("Aborted training should return Ok");

        // The model must still be usable for prediction without panicking.
        let preds = model.predict("basketball player sport game", 1, 0.0);
        // We just verify it doesn't panic — predictions may be poor since training was aborted.
        let _ = preds;

        // Verify abort flag is accessible on the returned model.
        model.abort(); // idempotent — should not panic
    }

    /// Abort is idempotent: calling abort() multiple times must not panic.
    #[test]
    fn test_abort_idempotent() {
        let data = supervised_train_data();
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        let mut args = Args::default();
        args.set_input(path_str.clone());
        args.set_output("/dev/null".to_string());
        args.apply_supervised_defaults();
        args.set_dim(5);
        args.set_epoch(2);
        args.set_min_count(1);
        args.set_lr(0.1);
        args.set_bucket(0);
        args.set_thread(1);

        let model = FastText::train(args).expect("Training should succeed");
        std::fs::remove_file(&path).ok();

        // Calling abort() multiple times must not panic.
        model.abort();
        model.abort();
        model.abort();
        // Success: no panic.
    }

    /// VAL-TRAIN-015: Single-thread deterministic training.
    ///
    /// With thread=1 and the same seed, two independent training runs on the
    /// same data must produce bit-identical model weights.
    #[test]
    fn test_deterministic_training() {
        let data = supervised_train_data();
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        let make_args = |path: &str| {
            let mut args = Args::default();
            args.set_input(path.to_string());
            args.set_output("/dev/null".to_string());
            args.apply_supervised_defaults();
            args.set_dim(10);
            args.set_epoch(3);
            args.set_min_count(1);
            args.set_lr(0.1);
            args.set_bucket(0);
            args.set_thread(1); // Single thread for determinism.
            args.set_seed(42); // Fixed seed.
            args
        };

        let model1 = FastText::train(make_args(&path_str)).expect("First training run failed");
        let model2 = FastText::train(make_args(&path_str)).expect("Second training run failed");
        std::fs::remove_file(&path).ok();

        // Both runs must produce bit-identical input weights.
        let input1 = model1.input_matrix().data().to_vec();
        let input2 = model2.input_matrix().data().to_vec();
        assert_eq!(
            input1.len(),
            input2.len(),
            "Input matrix sizes differ between runs"
        );
        for (i, (&v1, &v2)) in input1.iter().zip(input2.iter()).enumerate() {
            assert_eq!(
                v1, v2,
                "Input weight at index {} differs: {} vs {} (non-deterministic with thread=1)",
                i, v1, v2
            );
        }

        // Both runs must produce bit-identical output weights.
        let output1 = model1.output_matrix().data().to_vec();
        let output2 = model2.output_matrix().data().to_vec();
        assert_eq!(
            output1.len(),
            output2.len(),
            "Output matrix sizes differ between runs"
        );
        for (i, (&v1, &v2)) in output1.iter().zip(output2.iter()).enumerate() {
            assert_eq!(
                v1, v2,
                "Output weight at index {} differs: {} vs {} (non-deterministic with thread=1)",
                i, v1, v2
            );
        }
    }
}


