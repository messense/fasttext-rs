use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;
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

/// Scratch buffers reused across training iterations to avoid repeated allocation.
struct TrainScratch {
    line: Vec<i32>,
    labels: Vec<i32>,
    word_hashes: Vec<i32>,
    token: String,
    pending_newline: bool,
}

impl TrainScratch {
    fn new() -> Self {
        TrainScratch {
            line: Vec::new(),
            labels: Vec::new(),
            word_hashes: Vec::new(),
            token: String::new(),
            pending_newline: false,
        }
    }
}

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

    /// Build the vocabulary dictionary from the training input file.
    ///
    /// Validates that the file is non-empty and, for supervised models,
    /// that at least one label is present.
    fn build_vocabulary(args: &Args, args_arc: &Arc<Args>) -> Result<Dictionary> {
        let mut dict = Dictionary::new(Arc::clone(args_arc));
        {
            let file = std::fs::File::open(&args.input).map_err(FastTextError::IoError)?;
            let mut reader = BufReader::new(file);
            dict.read_from_file(&mut reader)?;
        }
        if dict.ntokens() == 0 {
            return Err(FastTextError::InvalidArgument(
                "Training file is empty or contains no valid tokens".to_string(),
            ));
        }
        if args.model == ModelName::Supervised && dict.nlabels() == 0 {
            return Err(FastTextError::InvalidArgument(
                "Supervised training requires at least one label, but none were found. \
                 Labels must start with the label prefix (default: '__label__')."
                    .to_string(),
            ));
        }
        Ok(dict)
    }

    /// Initialize input and output matrices for training.
    ///
    /// Input matrix is uniform random in `[-1/dim, 1/dim]`, optionally
    /// overwritten by pretrained vectors. Output matrix is zero-initialized.
    fn initialize_matrices(
        args: &Args,
        dict: &Dictionary,
    ) -> Result<(Arc<DenseMatrix>, Arc<DenseMatrix>)> {
        let nwords = dict.nwords() as i64;
        let bucket = args.bucket as i64;
        let dim = args.dim as i64;

        let input = Arc::new({
            let mut m = DenseMatrix::new(nwords + bucket, dim);
            m.uniform(1.0 / args.dim as f32, args.seed);
            if !args.pretrained_vectors.as_os_str().is_empty() {
                Self::load_pretrained_vectors(&args.pretrained_vectors, args, dict, &mut m)?;
            }
            m
        });

        let out_rows = if args.model == ModelName::Supervised {
            dict.nlabels() as i64
        } else {
            dict.nwords() as i64
        };
        let output = Arc::new(DenseMatrix::new(out_rows, dim));

        Ok((input, output))
    }

    /// Compute average training loss from shared atomic counters.
    fn compute_avg_loss(shared_loss: &AtomicU64, shared_loss_count: &AtomicI64) -> f64 {
        let total_loss = f64::from_bits(shared_loss.load(Ordering::Relaxed));
        let total_examples = shared_loss_count.load(Ordering::Relaxed);
        if total_examples > 0 {
            total_loss / total_examples as f64
        } else {
            0.0
        }
    }

    /// Run parallel Hogwild! SGD training across `n_threads`.
    ///
    /// Returns the average training loss.
    fn run_hogwild_training(
        args: &Args,
        dict: &Dictionary,
        input: &Arc<DenseMatrix>,
        output: &Arc<DenseMatrix>,
        loss_tables: &Arc<LossTables>,
        abort_flag: &Arc<AtomicBool>,
        epoch_loss_tracker: &Option<Arc<Mutex<Vec<f32>>>>,
    ) -> Result<f64> {
        let output_size = output.rows() as usize;
        let target_counts = if args.model == ModelName::Supervised {
            dict.get_counts(EntryType::Label)
        } else {
            dict.get_counts(EntryType::Word)
        };
        let normalize_gradient = args.model == ModelName::Supervised;
        let n_threads = (args.thread as usize).max(1);
        let token_count = AtomicI64::new(0);
        let shared_loss = AtomicU64::new(f64::to_bits(0.0));
        let shared_loss_count = AtomicI64::new(0);

        let training_results: Vec<Result<()>> = (0..n_threads)
            .into_par_iter()
            .map(|thread_id| {
                let loss = build_loss(
                    args,
                    Arc::clone(output),
                    &target_counts,
                    Arc::clone(loss_tables),
                );
                let model = Model::new(Arc::clone(input), loss, normalize_gradient);
                let ctx = TrainThreadCtx {
                    args,
                    dict,
                    model: &model,
                    output_size,
                    token_count: &token_count,
                    abort_flag,
                    shared_loss: &shared_loss,
                    epoch_loss_tracker: epoch_loss_tracker.clone(),
                };
                Self::train_thread_inner(thread_id, n_threads, &ctx, &shared_loss_count)
            })
            .collect();

        for result in training_results {
            result?;
        }

        Ok(Self::compute_avg_loss(&shared_loss, &shared_loss_count))
    }

    /// Internal training implementation with optional epoch loss tracking.
    fn train_internal(
        args: Args,
        abort_flag: Arc<AtomicBool>,
        epoch_loss_tracker: Option<Arc<Mutex<Vec<f32>>>>,
    ) -> Result<Self> {
        if args.input.as_os_str().is_empty() {
            return Err(FastTextError::InvalidArgument(
                "Input file path is empty".to_string(),
            ));
        }

        let args_arc = Arc::new(args.clone());
        let dict = Self::build_vocabulary(&args, &args_arc)?;
        let (input, output) = Self::initialize_matrices(&args, &dict)?;
        let loss_tables = Arc::new(LossTables::new());

        let avg_loss = Self::run_hogwild_training(
            &args,
            &dict,
            &input,
            &output,
            &loss_tables,
            &abort_flag,
            &epoch_loss_tracker,
        )?;

        let model = Self::build_inference_model(&args, &dict, &input, &output, &loss_tables);

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
            loss_tables,
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

    /// Parse the header of a `.vec` pretrained vectors file.
    ///
    /// Returns `(n_words, dim)` from the first line.
    fn parse_vec_header(header: &str, args_dim: i32) -> Result<(i64, i32)> {
        let mut parts = header.split_whitespace();
        let n: i64 = parts
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
        let vec_dim: i32 = parts
            .next()
            .ok_or_else(|| {
                FastTextError::InvalidModel("Missing dim in pretrained vectors header".to_string())
            })?
            .parse()
            .map_err(|_| {
                FastTextError::InvalidModel("Invalid dim in pretrained vectors header".to_string())
            })?;
        if vec_dim != args_dim {
            return Err(FastTextError::InvalidArgument(format!(
                "Dimension of pretrained vectors ({}) does not match model dimension ({})",
                vec_dim, args_dim
            )));
        }
        Ok((n, vec_dim))
    }

    /// Parse one line of a `.vec` file and write it into the input matrix.
    ///
    /// Returns `Ok(())` on success, propagating errors for malformed lines.
    fn apply_pretrained_line(
        line: &str,
        dim: usize,
        dict: &Dictionary,
        input: &mut DenseMatrix,
    ) -> Result<()> {
        let mut parts = line.split_whitespace();
        let word = parts.next().ok_or_else(|| {
            FastTextError::InvalidModel("Missing word in pretrained vectors line".to_string())
        })?;

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

        if let Some(idx) = dict.get_id(word) {
            if idx < dict.nwords() {
                let row = input.row_mut(idx as i64);
                row.copy_from_slice(&vec);
            }
        }
        Ok(())
    }

    /// Load pretrained word vectors from a `.vec` file into the input matrix.
    ///
    /// The `.vec` format (same as C++ fastText output):
    /// - First line: `<n_words> <dim>` (header)
    /// - Each subsequent line: `<word> <val1> <val2> ... <val_dim>`
    fn load_pretrained_vectors(
        path: &Path,
        args: &Args,
        dict: &Dictionary,
        input: &mut DenseMatrix,
    ) -> Result<()> {
        let file = std::fs::File::open(path).map_err(FastTextError::IoError)?;
        let reader = std::io::BufReader::new(file);
        let mut lines = reader.lines();

        let header = lines
            .next()
            .ok_or_else(|| {
                FastTextError::InvalidModel("Empty pretrained vectors file".to_string())
            })?
            .map_err(FastTextError::IoError)?;

        let (n, vec_dim) = Self::parse_vec_header(&header, args.dim)?;
        let dim = vec_dim as usize;

        for _ in 0..n {
            let line = lines
                .next()
                .ok_or_else(|| {
                    FastTextError::InvalidModel("Pretrained vectors file is truncated".to_string())
                })?
                .map_err(FastTextError::IoError)?;
            Self::apply_pretrained_line(&line, dim, dict, input)?;
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

    /// Compute the learning rate based on current progress.
    ///
    /// Linear decay from `base_lr` to 0 over the training run.
    #[inline]
    fn compute_lr(base_lr: f32, token_count: i64, epoch: i64, ntokens: i64) -> f32 {
        let progress = token_count as f32 / (epoch as f32 * ntokens as f32);
        (base_lr * (1.0 - progress)).max(0.0_f32)
    }

    /// Read one training line and apply the appropriate SGD update.
    ///
    /// Returns the number of tokens read (0 at EOF).
    fn process_training_line<R: Read + Seek>(
        reader: &mut BufReader<R>,
        ctx: &TrainThreadCtx<'_>,
        state: &mut State,
        lr: f32,
        scratch: &mut TrainScratch,
    ) -> i32 {
        let model_name = ctx.args.model;
        let ntok = match model_name {
            ModelName::Supervised => ctx.dict.get_line_with_scratch(
                reader,
                &mut scratch.line,
                &mut scratch.labels,
                &mut scratch.word_hashes,
                &mut scratch.token,
                &mut scratch.pending_newline,
            ),
            _ => ctx.dict.get_line_unsupervised_with_scratch(
                reader,
                &mut scratch.line,
                &mut scratch.token,
                &mut scratch.pending_newline,
                &mut state.rng,
            ),
        };

        if ntok > 0 {
            let is_ova = ctx.args.loss == LossName::OneVsAll;
            let ws = ctx.args.ws;
            match model_name {
                ModelName::Supervised => {
                    Self::supervised_fn(
                        ctx.model,
                        state,
                        lr,
                        &scratch.line,
                        &scratch.labels,
                        is_ova,
                    );
                }
                ModelName::Cbow => {
                    Self::cbow_fn(ctx.model, ctx.dict, state, lr, &scratch.line, ws);
                }
                _ => {
                    Self::skipgram_fn(ctx.model, ctx.dict, state, lr, &scratch.line, ws);
                }
            }
        }

        ntok
    }

    /// Flush accumulated loss from a thread into the shared atomic counters.
    fn flush_thread_loss(state: &State, shared_loss: &AtomicU64, shared_loss_count: &AtomicI64) {
        let examples = state.nexamples();
        if examples > 0 {
            let total_loss = state.get_loss() as f64 * examples as f64;
            atomic_f64_add(shared_loss, total_loss);
            shared_loss_count.fetch_add(examples, Ordering::Relaxed);
        }
    }

    /// Open the training input file and seek to this thread's starting position.
    fn open_thread_reader(
        input_path: &Path,
        thread_id: usize,
        n_threads: usize,
    ) -> Result<BufReader<std::fs::File>> {
        let mut file = std::fs::File::open(input_path).map_err(FastTextError::IoError)?;
        let file_size = file
            .seek(SeekFrom::End(0))
            .map_err(FastTextError::IoError)?;
        let start_pos = thread_id as u64 * file_size / n_threads as u64;
        file.seek(SeekFrom::Start(start_pos))
            .map_err(FastTextError::IoError)?;
        Ok(BufReader::new(file))
    }

    /// Record per-epoch loss if a new epoch has been completed.
    ///
    /// Returns the updated `last_recorded_epoch` value.
    fn maybe_record_epoch_loss(
        tracker: &Option<Arc<Mutex<Vec<f32>>>>,
        state: &mut State,
        tc: i64,
        ntokens: i64,
        last_recorded_epoch: i64,
    ) -> i64 {
        if let Some(ref tracker) = tracker {
            let current_epoch = if ntokens > 0 { tc / ntokens } else { 0 };
            if current_epoch > last_recorded_epoch && state.nexamples() > 0 {
                tracker.lock().unwrap().push(state.get_loss());
                state.reset();
                return current_epoch;
            }
        }
        last_recorded_epoch
    }

    /// Finalize a training thread: flush remaining token count, record final
    /// epoch loss, and flush accumulated loss to shared counters.
    fn finalize_training_thread(
        local_token_count: i64,
        ctx: &TrainThreadCtx<'_>,
        state: &State,
        shared_loss_count: &AtomicI64,
    ) {
        if local_token_count > 0 {
            ctx.token_count
                .fetch_add(local_token_count, Ordering::Relaxed);
        }
        if let Some(ref tracker) = ctx.epoch_loss_tracker {
            if state.nexamples() > 0 {
                tracker.lock().unwrap().push(state.get_loss());
            }
        }
        Self::flush_thread_loss(state, ctx.shared_loss, shared_loss_count);
    }

    /// Inner training loop for one thread (Hogwild! SGD).
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

        let mut reader = Self::open_thread_reader(&ctx.args.input, thread_id, n_threads)?;
        let seed = thread_id as u64 + ctx.args.seed as u64;
        let mut state = State::new(ctx.args.dim as usize, ctx.output_size, seed);
        let lr_update_rate = ctx.args.lr_update_rate as i64;
        let base_lr = ctx.args.lr as f32;
        let epoch = ctx.args.epoch as i64;

        let (mut local_token_count, mut last_recorded_epoch) = (0i64, 0i64);
        let mut scratch = TrainScratch::new();

        loop {
            if ctx.abort_flag.load(Ordering::Relaxed) {
                break;
            }
            let tc = ctx.token_count.load(Ordering::Relaxed);
            last_recorded_epoch = Self::maybe_record_epoch_loss(
                &ctx.epoch_loss_tracker,
                &mut state,
                tc,
                ntokens,
                last_recorded_epoch,
            );
            if tc >= epoch * ntokens {
                break;
            }

            let lr = Self::compute_lr(base_lr, tc, epoch, ntokens);
            let ntok = Self::process_training_line(&mut reader, ctx, &mut state, lr, &mut scratch);
            if ntok == 0 {
                reader
                    .seek(SeekFrom::Start(0))
                    .map_err(FastTextError::IoError)?;
                scratch.pending_newline = false;
                continue;
            }
            local_token_count += ntok as i64;
            if local_token_count > lr_update_rate {
                ctx.token_count
                    .fetch_add(local_token_count, Ordering::Relaxed);
                local_token_count = 0;
            }
        }

        Self::finalize_training_thread(local_token_count, ctx, &state, shared_loss_count);
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
