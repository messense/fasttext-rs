use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::args::{Args, LossName, ModelName};
use crate::dictionary::{Dictionary, EntryType};
use crate::error::{FastTextError, Result};
use crate::loss::LossTables;
use crate::matrix::{DenseMatrix, Matrix};
use crate::model::{Model, State};

use super::{build_loss, FastText, TrainThreadCtx, TrainingHandle};

/// Atomically add a f64 value to an AtomicU64 (which stores f64 bits).
///
/// Uses a compare-exchange (CAS) loop since there is no native atomic f64 add.
fn atomic_f64_add(target: &AtomicU64, delta: f64) {
    let mut current = target.load(Ordering::Relaxed);
    loop {
        let new_bits = (f64::from_bits(current) + delta).to_bits();
        match target.compare_exchange_weak(current, new_bits, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

impl FastText {
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
        TrainingHandle {
            abort_flag,
            join_handle,
        }
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
            .map_err(|_| {
                FastTextError::InvalidArgument("Epoch loss tracker still in use".to_string())
            })?
            .into_inner()
            .map_err(|_| {
                FastTextError::InvalidArgument("Epoch loss tracker lock poisoned".to_string())
            })?;
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
            .ok_or_else(|| {
                FastTextError::InvalidModel("Empty pretrained vectors file".to_string())
            })?
            .map_err(FastTextError::IoError)?;

        let mut header_parts = header.split_whitespace();
        let n: i64 = header_parts
            .next()
            .ok_or_else(|| {
                FastTextError::InvalidModel(
                    "Missing word count in pretrained vectors header".to_string(),
                )
            })?
            .parse()
            .map_err(|_| {
                FastTextError::InvalidModel(
                    "Invalid word count in pretrained vectors header".to_string(),
                )
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
                vec_dim, args.dim
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
            let word = parts.next().ok_or_else(|| {
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
            if let Some(idx) = dict.get_id(word) {
                if idx < nwords {
                    let row = input.row_mut(idx as i64);
                    row.copy_from_slice(&vec);
                }
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
    pub(super) fn train_thread_inner(
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
        let file_size = file
            .seek(SeekFrom::End(0))
            .map_err(FastTextError::IoError)?;
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
        let mut word_hashes: Vec<i32> = Vec::new();
        let mut token = String::new();
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
                ModelName::SUP => ctx.dict.get_line_with_scratch(
                    &mut reader,
                    &mut line,
                    &mut labels,
                    &mut word_hashes,
                    &mut token,
                    &mut pending_newline,
                ),
                _ => ctx.dict.get_line_unsupervised_with_scratch(
                    &mut reader,
                    &mut line,
                    &mut token,
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
                ctx.token_count
                    .fetch_add(local_token_count, Ordering::Relaxed);
                local_token_count = 0;
            }
        }

        // Flush any remaining local token count into the shared counter.
        if local_token_count > 0 {
            ctx.token_count
                .fetch_add(local_token_count, Ordering::Relaxed);
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
    pub(super) fn supervised_fn(
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
