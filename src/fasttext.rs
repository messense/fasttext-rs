// FastText: train, predict, quantize, autotune, save/load, word/sentence vectors

use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::args::{Args, LossName, ModelName};
use crate::dictionary::{Dictionary, EntryType, EOS};
use crate::error::{FastTextError, Result};
use crate::loss::{find_k_best, HierarchicalSoftmaxLoss, Loss, LossTables, NegativeSamplingLoss, OneVsAllLoss, SoftmaxLoss};
use crate::matrix::{DenseMatrix, Matrix};
use crate::meter::Meter;
use crate::model::{Model, Predictions, State};
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
fn build_loss(args: &Args, wo: Arc<DenseMatrix>, target_counts: &[i64]) -> Box<dyn Loss> {
    match args.loss {
        LossName::HS => Box::new(HierarchicalSoftmaxLoss::new(wo, target_counts)),
        LossName::NS => Box::new(NegativeSamplingLoss::new(wo, args.neg, target_counts)),
        LossName::OVA => Box::new(OneVsAllLoss::new(wo)),
        LossName::SOFTMAX => Box::new(SoftmaxLoss::new(wo)),
    }
}

/// Atomically add a f64 value to an AtomicU64 (which stores f64 bits).
///
/// Uses a compare-exchange (CAS) loop since there is no native atomic f64 add.
fn atomic_f64_add(target: &AtomicU64, delta: f64) {
    let mut current = target.load(Ordering::Relaxed);
    loop {
        let new_bits = (f64::from_bits(current) + delta).to_bits();
        match target.compare_exchange_weak(current, new_bits, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
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
    /// Shared atomic accumulator for total training loss (f64 bits) across all threads.
    ///
    /// Each thread atomically adds its local loss sum to this counter at the end
    /// of training.  After all threads finish, the total loss can be read via
    /// `f64::from_bits(shared_loss.load(Ordering::Relaxed))`.
    shared_loss: &'a AtomicU64,
    /// Optional per-epoch loss tracker.
    ///
    /// When `Some`, records the average loss after each completed epoch.
    /// Best used with `thread=1` for accurate epoch boundaries.
    epoch_loss_tracker: Option<Arc<Mutex<Vec<f32>>>>,
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
    /// For quantized models this is a zero-size placeholder; use `quant_input` instead.
    input: Arc<DenseMatrix>,
    /// Output matrix (label/word vectors), shared via Arc for Model.
    /// For quantized output, use `quant_output` instead.
    output: Arc<DenseMatrix>,
    /// Whether the model uses quantized (QuantMatrix) input.
    quant: bool,
    /// Quantized input matrix. Present when `quant=true`.
    pub quant_input: Option<QuantMatrix>,
    /// Quantized output matrix. Present when `quant=true` and `args.qout=true`.
    pub quant_output: Option<QuantMatrix>,
    /// Pre-built inference model (uses dense matrices; bypassed when quant=true).
    model: Model,
    /// Atomic flag for aborting an in-progress training run.
    ///
    /// Set via [`FastText::abort()`]; checked in the training loop.
    abort_flag: Arc<AtomicBool>,
    /// Average training loss from the last training run.
    ///
    /// Set by [`FastText::train`] and related functions.  `0.0` for models
    /// loaded from disk (no training history available).
    last_train_loss: f64,
    /// Cached sigmoid/log lookup tables for quantized prediction.
    loss_tables: LossTables,
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
    pub fn save_model(&self, path: &str) -> Result<()> {
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

    /// Return the word vector for a given word.
    ///
    /// For in-vocabulary words the vector is the average of all stored subword
    /// IDs (which for `minn=0 / maxn=0` is just the word's own row).  For
    /// OOV words the subwords are computed on-the-fly; if there are no subwords
    /// (e.g. `bucket=0`) a zero vector is returned.
    pub fn get_word_vector(&self, word: &str) -> Vec<f32> {
        let dim = self.args.dim as usize;
        let mut result = vec![0.0f32; dim];
        let ids = self.dict.get_subwords_for_string(word);
        if ids.is_empty() {
            return result;
        }
        let scale = 1.0 / ids.len() as f32;
        if self.quant {
            // Quantized path: use QuantMatrix reconstruction.
            if let Some(ref qi) = self.quant_input {
                let mut vec = Vector::new(dim);
                for &id in &ids {
                    qi.add_row_to_vector(&mut vec, id, scale);
                }
                result.copy_from_slice(vec.data());
            }
        } else {
            for &id in &ids {
                let row = self.input.row(id as i64);
                for (r, &v) in result.iter_mut().zip(row.iter()) {
                    *r += v * scale;
                }
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
        let dim = self.args.dim as usize;
        let mut result = vec![0.0f32; dim];

        if self.args.model == ModelName::SUP {
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
            if self.quant {
                // Quantized path: use QuantMatrix reconstruction.
                if let Some(ref qi) = self.quant_input {
                    let mut vec = Vector::new(dim);
                    for &id in &words {
                        qi.add_row_to_vector(&mut vec, id, 1.0);
                    }
                    for (r, &v) in result.iter_mut().zip(vec.data().iter()) {
                        *r = v / count;
                    }
                }
            } else {
                for &id in &words {
                    let row = self.input.row(id as i64);
                    for (r, &v) in result.iter_mut().zip(row.iter()) {
                        *r += v;
                    }
                }
                for r in &mut result {
                    *r /= count;
                }
            }
        } else {
            // Unsupervised: split whitespace, get word vector, L2-normalize, average.
            // get_word_vector() already handles the quant path internally.
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
        self.args.dim
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

        let dim = self.args.dim as usize;
        let mut state = State::new(dim, nlabels, 0);

        // Clamp k to at most the number of labels.
        let k_eff = k.min(nlabels) as i32;

        let raw = if self.quant {
            self.predict_raw_quantized(word_ids, k_eff, threshold, &mut state)
        } else {
            self.model.predict(word_ids, k_eff, threshold, &mut state)
        };

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

    /// Predict using quantized matrices (quant=true path).
    ///
    /// Computes hidden from `quant_input`, then computes output scores using
    /// `quant_output` (if present) or the dense `output` matrix.  Applies
    /// the appropriate normalization based on the configured loss function
    /// (softmax, OVA/sigmoid, or HS) and returns top-k predictions as
    /// (log_probability, label_index) pairs.
    fn predict_raw_quantized(
        &self,
        word_ids: &[i32],
        k: i32,
        threshold: f32,
        state: &mut State,
    ) -> Predictions {
        let quant_input = match self.quant_input.as_ref() {
            Some(qi) => qi,
            None => return Predictions::new(),
        };
        let nlabels = self.dict.nlabels() as usize;
        if nlabels == 0 || k <= 0 {
            return Predictions::new();
        }

        // Compute hidden representation using quantized input matrix.
        quant_input.average_rows_to_vector(&mut state.hidden, word_ids);

        // Compute raw output scores.
        let osz = nlabels;
        match &self.quant_output {
            Some(qout) => {
                for i in 0..osz {
                    let dot = qout.dot_row(&state.hidden, i as i64).unwrap_or(0.0);
                    state.output[i] = dot;
                }
            }
            None => {
                // Use dense output matrix (qout=false).
                for i in 0..osz {
                    let dot = self.output.dot_row(&state.hidden, i as i64).unwrap_or(0.0);
                    state.output[i] = dot;
                }
            }
        }

        // Apply the appropriate normalization based on loss type.
        match self.args.loss {
            LossName::OVA => {
                // One-vs-all: independent sigmoid per class.
                let tables = &self.loss_tables;
                for i in 0..osz {
                    state.output[i] = tables.sigmoid(state.output[i]);
                }
            }
            _ => {
                // Softmax (and NS, which also uses softmax for prediction).
                let max = state.output.data()[..osz]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut z = 0.0f32;
                for i in 0..osz {
                    state.output[i] = (state.output[i] - max).exp();
                    z += state.output[i];
                }
                if z > 0.0 {
                    for i in 0..osz {
                        state.output[i] /= z;
                    }
                }
            }
        }

        // Find top-k predictions above threshold.
        let mut heap = Predictions::new();
        find_k_best(k as usize, threshold, &mut heap, &state.output);
        heap
    }

    // Evaluation (test command)

    /// Evaluate the model on labeled test data and return a `Meter` with metrics.
    ///
    /// Reads each line from `reader`, extracts word IDs and gold label IDs via
    /// the dictionary, runs `k`-best prediction, and accumulates the results in
    /// a [`Meter`].  Lines with no labels or no words are skipped (matching C++
    /// `FastText::test`).
    ///
    /// # Arguments
    /// - `reader`: source of labeled test data (one labeled example per line).
    /// - `k`: number of top predictions to request per example.
    /// - `threshold`: minimum probability threshold (predictions below this
    ///   are excluded from the prediction set passed to the meter).
    ///
    /// Returns a [`Meter`] containing accumulated precision, recall, and F1
    /// statistics for all examples in `reader`.
    pub fn test_model<R: Read + Seek>(&self, reader: &mut R, k: usize, threshold: f32) -> Result<Meter> {
        let nlabels = self.dict.nlabels() as usize;
        let dim = self.args.dim as usize;
        let k_eff = if nlabels == 0 { 0i32 } else { k.min(nlabels) as i32 };
        let effective_threshold = if threshold < 0.0 { 0.0 } else { threshold };

        // Rewind to the beginning, matching C++ `in.seekg(0, beg)`.
        reader.seek(SeekFrom::Start(0)).map_err(FastTextError::IoError)?;
        let mut buf_reader = BufReader::new(reader);

        let mut meter = Meter::new();
        let mut words: Vec<i32> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        let mut pending_newline = false;

        loop {
            words.clear();
            labels.clear();
            let ntokens = self.dict.get_line(&mut buf_reader, &mut words, &mut labels, &mut pending_newline);
            if ntokens == 0 && words.is_empty() && labels.is_empty() {
                // EOF: read_word_from_reader returned false with no tokens
                // Check if we actually hit EOF by trying to read again.
                // The dictionary returns 0 tokens on EOF.
                break;
            }

            if !labels.is_empty() && !words.is_empty() && k_eff > 0 {
                let mut state = State::new(dim, nlabels, 0);
                let raw = if self.quant {
                    self.predict_raw_quantized(&words, k_eff, effective_threshold, &mut state)
                } else {
                    self.model.predict(&words, k_eff, effective_threshold, &mut state)
                };

                // Convert log-probs to probabilities (matching C++ `min(exp(score), 1.0)`).
                let predictions: Vec<(f32, i32)> = raw
                    .into_iter()
                    .map(|(log_prob, label_idx)| (log_prob.exp().min(1.0), label_idx))
                    .collect();

                meter.add(&predictions, &labels, k_eff as usize);
            }
        }

        Ok(meter)
    }

    // N-gram vectors, nearest neighbors, and analogies

    /// Return the character n-gram vectors for a word.
    ///
    /// For each subword of the word (including the word itself if in vocab),
    /// returns the n-gram string and its corresponding vector from the input matrix.
    ///
    /// This mirrors the C++ `FastText::getNgramVectors`.
    pub fn get_ngram_vectors(&self, word: &str) -> Vec<(String, Vec<f32>)> {
        let dim = self.args.dim as usize;
        let entries = self.dict.get_ngram_strings(word);
        entries
            .into_iter()
            .map(|(id, s)| {
                let mut vec = vec![0.0f32; dim];
                if id >= 0 {
                    if self.quant {
                        // Quantized path: use QuantMatrix reconstruction.
                        if let Some(ref qi) = self.quant_input {
                            let mut v = Vector::new(dim);
                            qi.add_row_to_vector(&mut v, id, 1.0);
                            vec.copy_from_slice(v.data());
                        }
                    } else {
                        let row = self.input.row(id as i64);
                        vec.copy_from_slice(row);
                    }
                }
                (s, vec)
            })
            .collect()
    }

    /// Precompute L2-normalized word vectors for all vocabulary words.
    ///
    /// Returns a `DenseMatrix` where row `i` is the L2-normalized word vector
    /// for word `i`. Words with zero-norm vectors have a zero row.
    ///
    /// Used as a precomputation step for nearest-neighbor and analogy queries.
    pub fn precompute_word_vectors(&self) -> DenseMatrix {
        let nwords = self.dict.nwords() as usize;
        let dim = self.args.dim as i64;
        let mut word_vectors = DenseMatrix::new(nwords as i64, dim);
        for i in 0..nwords {
            let word = self.dict.get_word(i as i32);
            let vec = self.get_word_vector(word);
            let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                let row = word_vectors.row_mut(i as i64);
                for (r, &v) in row.iter_mut().zip(vec.iter()) {
                    *r = v / norm;
                }
            }
        }
        word_vectors
    }

    /// Find the `k` nearest neighbors to `word` by cosine similarity.
    ///
    /// Precomputes normalized word vectors for all vocabulary words, then
    /// linearly scans for the top-k words (excluding the query word itself).
    ///
    /// Returns a vec of `(similarity, word)` pairs sorted by descending similarity.
    ///
    /// This mirrors the C++ `FastText::getNN`.
    pub fn get_nn(&self, word: &str, k: usize) -> Vec<(f32, String)> {
        let query = self.get_word_vector(word);
        let word_vectors = self.precompute_word_vectors();
        let ban_words = vec![word];
        self.nn_from_word_vectors(&word_vectors, &query, k, &ban_words)
    }

    /// Find the `k` nearest neighbors to `word_a - word_b + word_c`.
    ///
    /// Computes the query vector as:
    ///   `query = normalize(A) - normalize(B) + normalize(C)`
    /// then finds the top-k nearest words (excluding A, B, C).
    ///
    /// Returns a vec of `(similarity, word)` pairs sorted by descending similarity.
    ///
    /// This mirrors the C++ `FastText::getAnalogies`.
    pub fn get_analogies(
        &self,
        word_a: &str,
        word_b: &str,
        word_c: &str,
        k: usize,
    ) -> Vec<(f32, String)> {
        let dim = self.args.dim as usize;
        let mut query = vec![0.0f32; dim];

        let buf = self.get_word_vector(word_a);
        let norm = buf.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let norm = if norm < 1e-8 { 1e-8 } else { norm };
        for (q, &v) in query.iter_mut().zip(buf.iter()) {
            *q += v / norm;
        }

        let buf = self.get_word_vector(word_b);
        let norm = buf.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let norm = if norm < 1e-8 { 1e-8 } else { norm };
        for (q, &v) in query.iter_mut().zip(buf.iter()) {
            *q -= v / norm;
        }

        let buf = self.get_word_vector(word_c);
        let norm = buf.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let norm = if norm < 1e-8 { 1e-8 } else { norm };
        for (q, &v) in query.iter_mut().zip(buf.iter()) {
            *q += v / norm;
        }

        let word_vectors = self.precompute_word_vectors();
        let ban_words = vec![word_a, word_b, word_c];
        self.nn_from_word_vectors(&word_vectors, &query, k, &ban_words)
    }

    /// Internal: linear scan for top-k nearest neighbors given precomputed word vectors.
    ///
    /// `word_vectors` must be a matrix of L2-normalized word vectors (one per row).
    /// `query` is the raw (unnormalized) query vector.
    /// `ban_words` are excluded from the results.
    ///
    /// Returns a vec of `(similarity, word)` sorted by descending similarity.
    fn nn_from_word_vectors(
        &self,
        word_vectors: &DenseMatrix,
        query: &[f32],
        k: usize,
        ban_words: &[&str],
    ) -> Vec<(f32, String)> {
        let query_norm: f32 = query.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let query_norm = if query_norm.abs() < 1e-8 {
            1.0
        } else {
            query_norm
        };

        let nwords = self.dict.nwords() as usize;
        let mut similarities: Vec<(f32, usize)> = Vec::with_capacity(nwords);

        for i in 0..nwords {
            let word = self.dict.get_word(i as i32);
            if ban_words.contains(&word) {
                continue;
            }
            let row = word_vectors.row(i as i64);
            // row is already normalized; dot product = cosine similarity / query_norm
            let dp: f32 = query.iter().zip(row.iter()).map(|(&q, &r)| q * r).sum();
            let similarity = dp / query_norm;
            similarities.push((similarity, i));
        }

        // Sort by descending similarity.
        similarities.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        similarities.truncate(k);

        similarities
            .into_iter()
            .map(|(sim, i)| (sim, self.dict.get_word(i as i32).to_string()))
            .collect()
    }

    // Quantization

    /// Select the top `cutoff` embedding row indices by L2 norm.
    ///
    /// The EOS token is always ranked first (so it is always retained).
    /// Remaining rows are sorted by descending L2 norm, and the top `cutoff`
    /// are returned.
    ///
    /// Matches C++ `FastText::selectEmbeddings`.
    fn select_embeddings(&self, cutoff: usize) -> Vec<i32> {
        let nrows = self.input.rows() as usize;
        let norms: Vec<f32> = (0..nrows)
            .map(|i| self.input.l2_norm_row(i as i64).unwrap_or(0.0))
            .collect();

        let eos_id = self.dict.get_id(EOS);

        let mut idx: Vec<i32> = (0..nrows as i32).collect();
        idx.sort_unstable_by(|&i1, &i2| {
            // EOS always comes first.
            if i1 == eos_id && i2 == eos_id {
                return std::cmp::Ordering::Equal;
            }
            if i1 == eos_id {
                return std::cmp::Ordering::Less;
            }
            if i2 == eos_id {
                return std::cmp::Ordering::Greater;
            }
            norms[i2 as usize]
                .partial_cmp(&norms[i1 as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let cutoff = cutoff.min(nrows);
        idx.truncate(cutoff);
        idx
    }

    /// Quantize the model in-place.
    ///
    /// Only supervised models can be quantized; attempting to quantize a CBOW or
    /// skip-gram model returns [`FastTextError::InvalidArgument`].
    ///
    /// Steps (matching C++ `FastText::quantize`):
    /// 1. Reject non-supervised models.
    /// 2. If `qargs.cutoff > 0` and smaller than current input rows:
    ///    a. Select top embeddings by L2 norm.
    ///    b. Prune the dictionary to those embeddings.
    ///    c. Build a new pruned input `DenseMatrix`.
    ///    d. If `qargs.retrain`: retrain with pruned input.
    /// 3. Create a `QuantMatrix` from the (possibly pruned) input.
    /// 4. If `qargs.qout`: create a `QuantMatrix` from the output.
    /// 5. Set `quant=true`, update `args.qout`.
    pub fn quantize(&mut self, qargs: &Args) -> Result<()> {
        if self.args.model != ModelName::SUP {
            return Err(FastTextError::InvalidArgument(
                "For now we only support quantization of supervised models".to_string(),
            ));
        }

        // Copy input data to a mutable local buffer.
        let mut input_m = self.input.rows();
        let input_n = self.input.cols();
        let mut input_data: Vec<f32> = self.input.data().to_vec();

        let cutoff = qargs.cutoff;
        if cutoff > 0 && cutoff < input_m as usize {
            let idx = self.select_embeddings(cutoff);

            // Capture nwords before pruning to separate word rows from ngram rows.
            let nwords_before = self.dict.nwords();
            // Prune dictionary to the selected indices.
            // After prune(), words are kept in ascending original-ID order.
            self.dict.prune(&idx);

            // Separate word rows and ngram rows from `idx`.
            // Sort word rows ascending to match the pruned dictionary's word order.
            // Ngram rows keep their order from `idx` (which matches pruneidx mapping).
            let mut words_idx: Vec<i32> =
                idx.iter().copied().filter(|&i| i < nwords_before).collect();
            let ngrams_idx: Vec<i32> =
                idx.iter().copied().filter(|&i| i >= nwords_before).collect();
            words_idx.sort_unstable(); // match dict word order (ascending original ID)

            // ordered_idx: word rows first (sorted), then ngram rows.
            // Invariant: matrix row j == dictionary word j (for j < new_nwords).
            let ordered_idx: Vec<i32> =
                words_idx.iter().chain(ngrams_idx.iter()).copied().collect();

            // Build pruned input matrix (rows in ordered_idx order).
            let pruned_m = ordered_idx.len() as i64;
            let mut pruned_data = vec![0f32; (pruned_m as usize) * (input_n as usize)];
            for (i, &old_row) in ordered_idx.iter().enumerate() {
                let src_start = old_row as usize * input_n as usize;
                let dst_start = i * input_n as usize;
                pruned_data[dst_start..dst_start + input_n as usize]
                    .copy_from_slice(&input_data[src_start..src_start + input_n as usize]);
            }
            input_m = pruned_m;
            input_data = pruned_data;

            if qargs.retrain {
                // Rebuild model with pruned input and retrain.
                let pruned_dense = DenseMatrix::from_data(pruned_m, input_n, &input_data);
                let pruned_arc = Arc::new(pruned_dense);
                self.retrain_after_prune(Arc::clone(&pruned_arc), qargs)?;
                // After retrain, use the (updated) pruned input data.
                input_data = pruned_arc.data().to_vec();
                // Update self.input so get_word_vector still works on dense models.
                self.input = pruned_arc;
            }
        }

        // Quantize the input matrix.
        let dsub = qargs.dsub as i32;
        let qnorm = qargs.qnorm;
        let quant_in = QuantMatrix::from_dense(&input_data, input_m, input_n, dsub, qnorm);
        self.quant_input = Some(quant_in);

        // Optionally quantize the output matrix.
        if qargs.qout {
            let output_data = self.output.data().to_vec();
            let out_m = self.output.rows();
            let out_n = self.output.cols();
            // C++ uses dsub=2 and passes the qnorm flag for the output matrix.
            let quant_out = QuantMatrix::from_dense(&output_data, out_m, out_n, 2, qnorm);
            self.quant_output = Some(quant_out);
        }

        // Mark model as quantized and update args.
        self.quant = true;
        let mut new_args = (*self.args).clone();
        new_args.qout = qargs.qout;
        self.args = Arc::new(new_args);

        Ok(())
    }

    /// Retrain the model after vocabulary pruning.
    ///
    /// Uses the `pruned_input` DenseMatrix (already compacted to the pruned
    /// vocabulary) and the existing output matrix.  Training hyperparameters
    /// (`epoch`, `lr`, `thread`) are taken from `qargs`.
    ///
    /// The pruned_input weights are updated in-place via Hogwild! SGD.
    fn retrain_after_prune(
        &self,
        pruned_input: Arc<DenseMatrix>,
        qargs: &Args,
    ) -> Result<()> {
        let input_path = &qargs.input;
        if input_path.is_empty() {
            return Err(FastTextError::InvalidArgument(
                "retrain=true requires qargs.input to be set to the training data path"
                    .to_string(),
            ));
        }

        // Build retrain args from qargs values.
        let mut retrain_args = (*self.args).clone();
        retrain_args.input = input_path.to_string();
        retrain_args.epoch = qargs.epoch;
        retrain_args.lr = qargs.lr;
        retrain_args.thread = qargs.thread;

        let n_threads = (retrain_args.thread as usize).max(1);
        let output_size = self.output.rows() as usize;
        let output = Arc::clone(&self.output);
        let target_counts = self.dict.get_counts(EntryType::Label);
        let normalize_gradient = true; // supervised

        let token_count = Arc::new(AtomicI64::new(0));
        let shared_loss = Arc::new(AtomicU64::new(f64::to_bits(0.0)));
        let shared_loss_count = Arc::new(AtomicI64::new(0));
        let abort_flag = Arc::new(AtomicBool::new(false));

        let training_results: Vec<Result<()>> = (0..n_threads)
            .into_par_iter()
            .map(|thread_id| {
                let loss = build_loss(&retrain_args, Arc::clone(&output), &target_counts);
                let model = Model::new(Arc::clone(&pruned_input), loss, normalize_gradient);
                let ctx = TrainThreadCtx {
                    args: &retrain_args,
                    dict: &self.dict,
                    model: &model,
                    output_size,
                    token_count: &token_count,
                    abort_flag: &abort_flag,
                    shared_loss: &shared_loss,
                    epoch_loss_tracker: None,
                };
                Self::train_thread_inner(thread_id, n_threads, &ctx, &shared_loss_count)
            })
            .collect();

        for r in training_results {
            r?;
        }

        Ok(())
    }

    // Training

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
        Self::train_internal(args, abort_flag, None)
    }

    /// Spawn training in a background thread, returning a [`TrainingHandle`].
    ///
    /// The handle exposes an [`TrainingHandle::abort`] method that can be called
    /// from the main (or any other) thread to stop the in-flight training run
    /// early.  Use [`TrainingHandle::join`] to wait for training to finish and
    /// obtain the (possibly under-trained) model.
    ///
    /// This is the canonical way to abort in-flight training via the public API —
    /// it solves the chicken-and-egg problem of `abort(&self)` requiring an already-
    /// built `FastText` while training entry points only return after completion.
    pub fn spawn_training(args: Args) -> TrainingHandle {
        let abort_flag = Arc::new(AtomicBool::new(false));
        let abort_for_train = Arc::clone(&abort_flag);
        let join_handle = std::thread::spawn(move || Self::train_with_abort(args, abort_for_train));
        TrainingHandle { abort_flag, join_handle }
    }

    /// Internal training implementation with optional epoch loss tracking.
    ///
    /// Called by `train`, `train_with_abort`, and `train_tracking_epoch_losses`.
    /// When `epoch_loss_tracker` is `Some`, records the average loss after each
    /// completed epoch (most accurate with `thread=1`).
    fn train_internal(
        args: Args,
        abort_flag: Arc<AtomicBool>,
        epoch_loss_tracker: Option<Arc<Mutex<Vec<f32>>>>,
    ) -> Result<Self> {
        let input_path = args.input.to_string();
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

        // Guard: empty training file.
        if dict.ntokens() == 0 {
            return Err(FastTextError::InvalidArgument(
                "Training file is empty or contains no valid tokens".to_string(),
            ));
        }

        // Guard: supervised mode requires at least one label.
        if args.model == ModelName::SUP && dict.nlabels() == 0 {
            return Err(FastTextError::InvalidArgument(
                "Supervised training requires at least one label, but none were found. \
                 Labels must start with the label prefix (default: '__label__')."
                    .to_string(),
            ));
        }

        let nwords = dict.nwords() as i64;
        let bucket = args.bucket as i64;
        let dim = args.dim as i64;

        // Initialize input matrix: (nwords + bucket) × dim, uniform in [-1/dim, 1/dim].
        // If pretrained vectors are specified, load them after uniform initialization.
        let input = Arc::new({
            let mut m = DenseMatrix::new(nwords + bucket, dim);
            m.uniform(1.0 / args.dim as f32, args.seed);
            if !args.pretrained_vectors.is_empty() {
                Self::load_pretrained_vectors(&args.pretrained_vectors, &args, &dict, &mut m)?;
            }
            m
        });

        // Initialize output matrix: (nlabels for supervised, nwords for unsupervised) × dim, zeros.
        let out_rows = if args.model == ModelName::SUP {
            dict.nlabels() as i64
        } else {
            dict.nwords() as i64
        };
        // DenseMatrix::new zeroes all values by default.
        let output = Arc::new(DenseMatrix::new(out_rows, dim));
        let output_size = output.rows() as usize;

        // Build target counts for loss construction.
        let target_counts = if args.model == ModelName::SUP {
            dict.get_counts(EntryType::Label)
        } else {
            dict.get_counts(EntryType::Word)
        };
        let normalize_gradient = args.model == ModelName::SUP;

        // Number of threads (at least 1).
        let n_threads = (args.thread as usize).max(1);

        // Shared atomic token counter across all training threads.
        let token_count = Arc::new(AtomicI64::new(0));

        // Shared atomic loss accumulator (f64 bits) across all training threads.
        // Each thread atomically adds its local total loss at the end of training.
        let shared_loss = Arc::new(AtomicU64::new(f64::to_bits(0.0)));
        // Shared example count (for computing average loss).
        let shared_loss_count = Arc::new(AtomicI64::new(0));

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
                    shared_loss: &shared_loss,
                    epoch_loss_tracker: epoch_loss_tracker.clone(),
                };
                Self::train_thread_inner(thread_id, n_threads, &ctx, &shared_loss_count)
            })
            .collect();

        // Propagate the first training error, if any.
        for result in training_results {
            result?;
        }

        // Compute average training loss across all threads.
        let total_loss = f64::from_bits(shared_loss.load(Ordering::Relaxed));
        let total_examples = shared_loss_count.load(Ordering::Relaxed);
        let avg_loss = if total_examples > 0 {
            total_loss / total_examples as f64
        } else {
            0.0
        };

        // Build the inference model from the trained matrices.
        let loss = build_loss(&args, Arc::clone(&output), &target_counts);
        let model = Model::new(Arc::clone(&input), loss, normalize_gradient);

        Ok(FastText {
            args: args_arc,
            dict,
            input,
            output,
            quant: false,
            quant_input: None,
            quant_output: None,
            model,
            abort_flag,
            last_train_loss: avg_loss,
            loss_tables: LossTables::new(),
        })
    }

    /// Train a new fastText model and collect per-epoch average training losses.
    ///
    /// This is a variant of [`FastText::train`] that additionally records the
    /// average training loss after each completed epoch.  The returned `Vec<f32>`
    /// contains one entry per epoch (index 0 = epoch 1, etc.).
    ///
    /// For accurate per-epoch measurements, use `thread=1` in the `Args`
    /// (multi-threaded training may produce more or fewer entries depending on
    /// how token batches align with epoch boundaries).
    pub fn train_tracking_epoch_losses(args: Args) -> Result<(Self, Vec<f32>)> {
        let tracker = Arc::new(Mutex::new(Vec::<f32>::new()));
        let model = Self::train_internal(
            args,
            Arc::new(AtomicBool::new(false)),
            Some(Arc::clone(&tracker)),
        )?;
        let losses = Arc::try_unwrap(tracker)
            .map_err(|_| FastTextError::InvalidArgument("Epoch loss tracker still in use".to_string()))?
            .into_inner()
            .map_err(|_| FastTextError::InvalidArgument("Epoch loss tracker lock poisoned".to_string()))?;
        Ok((model, losses))
    }

    /// Load pretrained word vectors from a `.vec` file into the input matrix.
    ///
    /// The `.vec` format (same as C++ fastText output):
    /// - First line: `<n_words> <dim>` (header)
    /// - Each subsequent line: `<word> <val1> <val2> ... <val_dim>`
    ///
    /// For each word in the `.vec` file that is also in the vocabulary, its row
    /// in the input matrix is overwritten with the pretrained vector.  Words not
    /// in the `.vec` file retain their uniform random initialization.
    ///
    /// # Errors
    /// - `FastTextError::IoError` if the file cannot be opened.
    /// - `FastTextError::InvalidArgument` if the `.vec` dimension doesn't match `args.dim`.
    /// - `FastTextError::InvalidModel` if the file is malformed.
    fn load_pretrained_vectors(
        path: &str,
        args: &Args,
        dict: &Dictionary,
        input: &mut DenseMatrix,
    ) -> Result<()> {
        let file = std::fs::File::open(path).map_err(FastTextError::IoError)?;
        let reader = std::io::BufReader::new(file);
        let mut lines = reader.lines();

        // Read header: "<n_words> <dim>"
        let header = lines
            .next()
            .ok_or_else(|| FastTextError::InvalidModel("Empty pretrained vectors file".to_string()))?
            .map_err(FastTextError::IoError)?;

        let mut header_parts = header.split_whitespace();
        let n: i64 = header_parts
            .next()
            .ok_or_else(|| {
                FastTextError::InvalidModel("Missing word count in pretrained vectors header".to_string())
            })?
            .parse()
            .map_err(|_| {
                FastTextError::InvalidModel("Invalid word count in pretrained vectors header".to_string())
            })?;
        let vec_dim: i32 = header_parts
            .next()
            .ok_or_else(|| {
                FastTextError::InvalidModel("Missing dim in pretrained vectors header".to_string())
            })?
            .parse()
            .map_err(|_| {
                FastTextError::InvalidModel("Invalid dim in pretrained vectors header".to_string())
            })?;

        // Dimension must match the model dim.
        if vec_dim != args.dim {
            return Err(FastTextError::InvalidArgument(format!(
                "Dimension of pretrained vectors ({}) does not match model dimension ({})",
                vec_dim,
                args.dim
            )));
        }

        let dim = vec_dim as usize;
        let nwords = dict.nwords();

        // For each word in the pretrained file, update the input matrix if it
        // is also in the vocabulary.
        for _ in 0..n {
            let line = lines
                .next()
                .ok_or_else(|| {
                    FastTextError::InvalidModel("Pretrained vectors file is truncated".to_string())
                })?
                .map_err(FastTextError::IoError)?;

            let mut parts = line.split_whitespace();
            let word = parts
                .next()
                .ok_or_else(|| {
                    FastTextError::InvalidModel("Missing word in pretrained vectors line".to_string())
                })?;

            // Parse all dim float values.
            let mut vec = Vec::with_capacity(dim);
            for j in 0..dim {
                let val: f32 = parts
                    .next()
                    .ok_or_else(|| {
                        FastTextError::InvalidModel(format!(
                            "Missing value {} in pretrained vectors for word '{}'",
                            j, word
                        ))
                    })?
                    .parse()
                    .map_err(|_| {
                        FastTextError::InvalidModel(format!(
                            "Invalid float at position {} in pretrained vectors for word '{}'",
                            j, word
                        ))
                    })?;
                vec.push(val);
            }

            // If the word is in the vocabulary, overwrite its input row.
            let idx = dict.get_id(word);
            if idx >= 0 && idx < nwords {
                let row = input.row_mut(idx as i64);
                row.copy_from_slice(&vec);
            }
        }

        Ok(())
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

    /// Return the average training loss from the last training run.
    ///
    /// For models loaded from disk via [`FastText::load_model`], this returns `0.0`
    /// (no training history is stored in the binary format).
    ///
    /// For trained models, this is the global average loss across all threads,
    /// accumulated atomically during training.
    pub fn last_train_loss(&self) -> f64 {
        self.last_train_loss
    }

    /// Inner training loop for one thread (Hogwild! SGD).
    ///
    /// Opens the input file, seeks to `thread_id * file_size / n_threads`,
    /// then loops reading lines and performing SGD updates until
    /// `token_count >= epoch * ntokens` or `abort_flag` is set.
    ///
    /// Matches C++ `FastText::trainThread`.
    fn train_thread_inner(
        thread_id: usize,
        n_threads: usize,
        ctx: &TrainThreadCtx<'_>,
        shared_loss_count: &AtomicI64,
    ) -> Result<()> {
        let ntokens = ctx.dict.ntokens();
        if ntokens == 0 {
            return Ok(());
        }

        let input_path = ctx.args.input.to_string();

        // Open file and seek to this thread's starting position.
        let mut file = std::fs::File::open(&input_path).map_err(FastTextError::IoError)?;
        let file_size = file.seek(SeekFrom::End(0)).map_err(FastTextError::IoError)?;
        let start_pos = thread_id as u64 * file_size / n_threads as u64;
        file.seek(SeekFrom::Start(start_pos))
            .map_err(FastTextError::IoError)?;
        let mut reader = BufReader::new(file);

        let seed = thread_id as u64 + ctx.args.seed as u64;
        let mut state = State::new(ctx.args.dim as usize, ctx.output_size, seed);

        let model_name = ctx.args.model;
        let is_ova = ctx.args.loss == LossName::OVA;
        let ws = ctx.args.ws;
        let lr_update_rate = ctx.args.lr_update_rate as i64;
        let base_lr = ctx.args.lr as f32;
        let epoch = ctx.args.epoch as i64;

        let mut local_token_count: i64 = 0;
        let mut line: Vec<i32> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        let mut pending_newline = false;

        // For per-epoch loss tracking: track which epoch we last recorded.
        let mut last_recorded_epoch: i64 = 0;

        loop {
            // Check abort flag — exit early if set.
            if ctx.abort_flag.load(Ordering::Relaxed) {
                break;
            }

            // Load current global token count.
            let tc = ctx.token_count.load(Ordering::Relaxed);

            // Per-epoch loss recording: check if we have completed a new epoch.
            // Record the average loss for the completed epoch and reset state.
            // Placed before the break check so the final epoch's loss is captured.
            if let Some(ref tracker) = ctx.epoch_loss_tracker {
                let current_epoch = if ntokens > 0 { tc / ntokens } else { 0 };
                if current_epoch > last_recorded_epoch && state.nexamples() > 0 {
                    let epoch_loss = state.get_loss();
                    tracker.lock().unwrap().push(epoch_loss);
                    state.reset();
                    last_recorded_epoch = current_epoch;
                }
            }

            // Check if training has reached the target token count.
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

        // Record the final (partial) epoch's loss, if any examples remain unrecorded.
        if let Some(ref tracker) = ctx.epoch_loss_tracker {
            if state.nexamples() > 0 {
                tracker.lock().unwrap().push(state.get_loss());
            }
        }

        // Atomically flush this thread's total loss contribution to the shared counter.
        // Uses a CAS loop (via atomic_f64_add) since there is no native atomic f64 add.
        let examples = state.nexamples();
        if examples > 0 {
            let total_loss = state.get_loss() as f64 * examples as f64;
            atomic_f64_add(ctx.shared_loss, total_loss);
            shared_loss_count.fetch_add(examples, Ordering::Relaxed);
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
        assert_eq!(dict.get_id("</s>"), 0);
        assert_eq!(dict.get_id("__label__test"), 1);
        assert_eq!(dict.get_id("unknown"), -1);
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
            model_zero.quantize(&qargs).expect("zero cutoff quantize should succeed");
        }
        // Zero cutoff: vocabulary size should be unchanged
        assert_eq!(model_zero.dict().nwords(), nwords_before,
            "cutoff=0 should not prune vocabulary");

        // cutoff = positive smaller than nwords: vocabulary should be pruned
        let cutoff = (nwords_before as usize / 2).max(1);
        {
            let mut qargs = Args::default();
            qargs.dsub = 2;
            qargs.cutoff = cutoff;
            model_cutoff.quantize(&qargs).expect("positive cutoff quantize should succeed");
        }
        let nwords_after = model_cutoff.dict().nwords();
        assert!(nwords_after < nwords_before,
            "cutoff={} should prune vocabulary: before={}, after={}",
            cutoff, nwords_before, nwords_after);
    }

    /// VAL-QUANT-006 (part 1): qnorm flag works correctly.
    #[test]
    fn test_quantize_qnorm() {
        let (mut model, train_path) = train_small_supervised(16, 5, 0);
        std::fs::remove_file(&train_path).ok();

        let mut qargs = Args::default();
        qargs.dsub = 2;
        qargs.qnorm = true;

        model.quantize(&qargs).expect("qnorm quantize should succeed");
        assert!(model.is_quant(), "is_quant() should be true with qnorm=true");

        // quant_input should have qnorm=true
        assert!(model.quant_input.as_ref().unwrap().qnorm,
            "quant_input should have qnorm=true");

        // Should still produce valid predictions
        let preds = model.predict("basketball player sport game", 1, 0.0);
        assert!(!preds.is_empty(), "qnorm model should produce predictions");
        assert!(preds[0].prob.is_finite() && preds[0].prob > 0.0,
            "qnorm prediction prob should be valid");
    }

    /// VAL-QUANT-006 (part 2): qout flag works correctly.
    #[test]
    fn test_quantize_qout() {
        let (mut model, train_path) = train_small_supervised(16, 5, 0);
        std::fs::remove_file(&train_path).ok();

        let mut qargs = Args::default();
        qargs.dsub = 2;
        qargs.qout = true;

        model.quantize(&qargs).expect("qout quantize should succeed");
        assert!(model.is_quant(), "is_quant() should be true with qout=true");
        assert!(model.args().qout, "args.qout should be true");

        // quant_output should be set
        assert!(model.quant_output.is_some(), "quant_output should be set when qout=true");

        // Should still produce valid predictions
        let preds = model.predict("basketball player sport game", 1, 0.0);
        assert!(!preds.is_empty(), "qout model should produce predictions");
        assert!(preds[0].prob.is_finite() && preds[0].prob > 0.0,
            "qout prediction prob should be valid");
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

        model.quantize(&qargs).expect("qout+qnorm quantize should succeed");
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
        assert!(!preds.is_empty(), "qout+qnorm model should produce predictions");
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

        model.quantize(&qargs).expect("qout quantize should succeed");
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