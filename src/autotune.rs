// Autotune and AutotuneStrategy: Gaussian perturbation, time-boxed hyperparameter search

use std::io::BufReader;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::args::{Args, LossName};
use crate::error::{FastTextError, Result};
use crate::fasttext::FastText;
use crate::model::MinstdRng;

/// Global counter for unique temp-file naming during model-size checks.
static SIZE_CHECK_COUNTER: AtomicU64 = AtomicU64::new(0);

// Normal distribution sampler (Box-Muller transform)

/// Generate a standard normal sample using the Box-Muller transform.
///
/// Uses two uniform samples from `rng` to produce one Gaussian variate
/// with mean=0, stddev=1.
fn normal_sample(rng: &mut MinstdRng) -> f64 {
    // Box-Muller transform: two uniform samples → one Gaussian
    let u1 = (rng.generate() as f64 / MinstdRng::M as f64).max(1e-15); // avoid ln(0)
    let u2 = rng.generate() as f64 / MinstdRng::M as f64;
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

// Gaussian perturbation helpers

/// Gaussian-perturb `val` with decaying sigma and optional linear/multiplicative mode.
///
/// Matches C++ `getArgGauss` / `updateArgGauss`:
/// - `stddev = startSigma - ((startSigma - endSigma) / 0.5) * clamp(t - 0.25, 0, 0.5)`
/// - Linear mode:         `result = val + N(0, stddev)`
/// - Multiplicative mode: `result = val * 2^N(0, stddev)`
/// - Result clamped to `[min, max]`.
#[allow(clippy::too_many_arguments)]
fn update_arg_gauss(
    val: f64,
    min: f64,
    max: f64,
    start_sigma: f64,
    end_sigma: f64,
    t: f64,
    linear: bool,
    rng: &mut MinstdRng,
) -> f64 {
    let stddev =
        start_sigma - ((start_sigma - end_sigma) / 0.5) * (0.5f64.min((t - 0.25).max(0.0)));
    let coeff = normal_sample(rng) * stddev;
    let result = if linear {
        val + coeff
    } else {
        val * 2.0f64.powf(coeff)
    };
    result.max(min).min(max)
}

/// Integer variant of [`update_arg_gauss`].
///
/// Applies Gaussian perturbation and truncates towards zero (matching C++ `static_cast<int>`).
#[allow(clippy::too_many_arguments)]
fn update_arg_gauss_i32(
    val: i32,
    min: i32,
    max: i32,
    start_sigma: f64,
    end_sigma: f64,
    t: f64,
    linear: bool,
    rng: &mut MinstdRng,
) -> i32 {
    let result = update_arg_gauss(
        val as f64,
        min as f64,
        max as f64,
        start_sigma,
        end_sigma,
        t,
        linear,
        rng,
    );
    // C++ uses static_cast<int> which truncates towards zero.
    result as i32
}

// AutotuneStrategy

/// Hyperparameter perturbation strategy for autotune.
///
/// Generates candidate [`Args`] sets by applying Gaussian perturbations to the
/// best-known hyperparameters. The perturbation magnitude (sigma) decreases over
/// time, favouring exploitation over exploration as the time budget is exhausted.
///
/// Matches C++ `fasttext::AutotuneStrategy`.
pub struct AutotuneStrategy {
    /// Best args found so far (initialized to the original args).
    best_args: Args,
    /// Total duration of the autotune run (seconds), for progress computation.
    max_duration: f64,
    /// Internal RNG (matches C++ `std::minstd_rand`).
    rng: MinstdRng,
    /// Number of trials requested so far (trial 1 always returns best_args unchanged).
    trials: u32,
    /// Index into `minn_choices` for the best-known `minn` value.
    best_minn_index: usize,
    /// `log2(dsub)` for the best-known `dsub` value.
    best_dsub_exponent: i32,
    /// Best-known non-zero bucket count (used when bucket was non-zero).
    best_nonzero_bucket: i32,
    /// Original `bucket` value from the initial args (preserved for the first trial).
    original_bucket: i32,
    /// Discrete choices for the `minn` parameter, matching C++ `{0, 2, 3}`.
    minn_choices: Vec<i32>,
}

impl AutotuneStrategy {
    /// Create a new strategy starting from `original_args`.
    ///
    /// `seed` is used to initialise the internal RNG (should equal `args.seed`).
    pub fn new(original_args: &Args, seed: u64) -> Self {
        let minn_choices = vec![0, 2, 3];
        let mut strategy = AutotuneStrategy {
            best_args: original_args.clone(),
            max_duration: original_args.autotune_duration as f64,
            rng: MinstdRng::new(seed),
            trials: 0,
            best_minn_index: 0,
            best_dsub_exponent: 1,
            best_nonzero_bucket: 2_000_000,
            original_bucket: original_args.bucket,
            minn_choices,
        };
        strategy.update_best(original_args);
        strategy
    }

    /// Return the next candidate [`Args`] set given that `elapsed` seconds have passed.
    ///
    /// - Trial 1 always returns `best_args` unchanged (baseline evaluation).
    /// - Subsequent trials perturb `best_args` using Gaussian noise.
    pub fn ask(&mut self, elapsed: f64) -> Args {
        let t = (elapsed / self.max_duration).min(1.0);
        self.trials += 1;

        // Trial 1: evaluate the baseline (original) args unchanged.
        if self.trials == 1 {
            return self.best_args.clone();
        }

        let mut args = self.best_args.clone();

        // epoch: multiplicative, range [1, 100], startSigma=2.8, endSigma=2.5
        let epoch = update_arg_gauss_i32(args.epoch, 1, 100, 2.8, 2.5, t, false, &mut self.rng);
        args.epoch = epoch;

        // lr: multiplicative, range [0.01, 5.0], startSigma=1.9, endSigma=1.0
        let lr = update_arg_gauss(args.lr, 0.01, 5.0, 1.9, 1.0, t, false, &mut self.rng);
        args.lr = lr;

        // dim: multiplicative, range [1, 1000], startSigma=1.4, endSigma=0.3
        let dim = update_arg_gauss_i32(args.dim, 1, 1000, 1.4, 0.3, t, false, &mut self.rng);
        args.dim = dim;

        // wordNgrams: additive (linear), range [1, 5], startSigma=4.3, endSigma=2.4
        let word_ngrams =
            update_arg_gauss_i32(args.word_ngrams, 1, 5, 4.3, 2.4, t, true, &mut self.rng);
        args.word_ngrams = word_ngrams;

        // dsub: perturb the exponent additively in [1, 4], then dsub = 2^exponent
        let dsub_exp = update_arg_gauss_i32(
            self.best_dsub_exponent,
            1,
            4,
            2.0,
            1.0,
            t,
            true,
            &mut self.rng,
        );
        args.dsub = 1usize << dsub_exp;

        // minn: perturb the index into minn_choices additively in [0, len-1]
        let minn_idx = update_arg_gauss_i32(
            self.best_minn_index as i32,
            0,
            (self.minn_choices.len() - 1) as i32,
            4.0,
            1.4,
            t,
            true,
            &mut self.rng,
        );
        let minn_idx_clamped = minn_idx.max(0) as usize;
        let minn = self.minn_choices[minn_idx_clamped.min(self.minn_choices.len() - 1)];
        args.minn = minn;

        // maxn: derived from minn (minn + 3, or 0 if minn == 0)
        if minn == 0 {
            args.maxn = 0;
        } else {
            args.maxn = minn + 3;
        }

        // bucket: multiplicative, range [10_000, 10_000_000], startSigma=2.0, endSigma=1.5
        let nonzero_bucket = update_arg_gauss_i32(
            self.best_nonzero_bucket,
            10_000,
            10_000_000,
            2.0,
            1.5,
            t,
            false,
            &mut self.rng,
        );
        if args.word_ngrams > 1 || minn != 0 {
            args.bucket = nonzero_bucket;
        } else {
            // No n-grams needed — restore original bucket
            args.bucket = self.original_bucket;
        }

        // If wordNgrams <= 1 and maxn == 0: no n-grams, set bucket to 0
        if args.word_ngrams <= 1 && args.maxn == 0 {
            args.bucket = 0;
        }

        // loss: always softmax for supervised classification
        args.loss = LossName::Softmax;

        args
    }

    /// Update the strategy's best-known args after a new best score was found.
    pub fn update_best(&mut self, args: &Args) {
        self.best_args = args.clone();
        self.best_minn_index = Self::find_index(args.minn, &self.minn_choices);
        let dsub = args.dsub as f64;
        self.best_dsub_exponent = if dsub > 0.0 {
            dsub.log2().round() as i32
        } else {
            1
        };
        if args.bucket != 0 {
            self.best_nonzero_bucket = args.bucket;
        }
    }

    /// Find the index of `val` in `choices`, defaulting to 0 if not found.
    fn find_index(val: i32, choices: &[i32]) -> usize {
        choices.iter().position(|&x| x == val).unwrap_or(0)
    }

    /// Return the number of trials requested so far.
    pub fn trials(&self) -> u32 {
        self.trials
    }
}

// Autotune

/// Sentinel value for "no score found yet".
const UNKNOWN_BEST_SCORE: f64 = f64::NEG_INFINITY;

/// Parse a human-readable byte-size string into a `u64` byte count.
///
/// Supported formats (case-insensitive suffixes):
/// - `"100"` or `"100B"` → 100 bytes
/// - `"100K"` or `"100KB"` → 100 × 1 024 bytes
/// - `"100M"` or `"100MB"` → 100 × 1 024² bytes
/// - `"100G"` or `"100GB"` → 100 × 1 024³ bytes
///
/// Fractional values are supported (e.g. `"1.5M"`).  Returns `None` for
/// unrecognised formats.
fn parse_size_to_bytes(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // Split at the first character that is not an ASCII digit or '.'.
    let split_pos = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(s.len());
    let num_str = &s[..split_pos];
    let unit = s[split_pos..].trim();

    let num: f64 = num_str.parse().ok()?;
    let multiplier: u64 = match unit.to_ascii_uppercase().as_str() {
        "" | "B" => 1,
        "K" | "KB" => 1_024,
        "M" | "MB" => 1_024 * 1_024,
        "G" | "GB" => 1_024 * 1_024 * 1_024,
        _ => return None,
    };
    Some((num * multiplier as f64) as u64)
}

/// Outcome of a single training trial spawned by [`Autotune::run`].
enum TrialOutcome {
    /// The time budget expired during training; the search loop should stop.
    TimedOut,
    /// Training completed but failed (NaN loss, panic, etc.); skip to the next trial.
    Failed,
    /// Training completed successfully.
    Success(Box<FastText>),
}

/// Time-boxed hyperparameter search for fastText supervised models.
///
/// Runs multiple training trials within the configured duration, evaluates
/// each trial on a validation file, tracks the best score, and retrains with
/// the best-found hyperparameters.
///
/// Matches C++ `fasttext::Autotune`.
pub struct Autotune;

impl Autotune {
    /// Run hyperparameter search, returning a model trained with the best-found args.
    ///
    /// The search proceeds as follows:
    /// 1. Ask `AutotuneStrategy` for the next candidate `Args`.
    /// 2. Train a model with those args (in a background thread).
    /// 3. If the time budget expires, abort the in-flight training and break.
    /// 4. Evaluate the trained model on the validation file.
    /// 5. If this score is the new best, update `strategy`.
    /// 6. Repeat until the time budget is exhausted.
    /// 7. Retrain from scratch with the best-found `Args`.
    ///
    /// # Errors
    ///
    /// - `FastTextError::InvalidArgument` — validation file not set or no trial succeeded.
    /// - `FastTextError::IoError` — validation file cannot be opened.
    pub fn run(autotune_args: Args) -> Result<FastText> {
        let val_path = autotune_args.autotune_validation_file.clone();
        if val_path.as_os_str().is_empty() {
            return Err(FastTextError::InvalidArgument(
                "autotune validation file is not set".to_string(),
            ));
        }
        let _ = std::fs::File::open(&val_path).map_err(FastTextError::IoError)?;

        let seed = autotune_args.seed as u64;
        let duration_secs = autotune_args.autotune_duration as f64;
        let k = autotune_args.autotune_predictions.max(1) as usize;
        let metric = autotune_args.autotune_metric.to_string();
        let model_size_bytes = parse_size_to_bytes(&autotune_args.autotune_model_size);

        let mut search_args = autotune_args.clone();
        search_args.verbose = 0;
        let mut strategy = AutotuneStrategy::new(&search_args, seed);
        let start = Instant::now();
        let mut best_args: Option<Args> = None;
        let mut best_score = UNKNOWN_BEST_SCORE;

        loop {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed >= duration_secs {
                break;
            }
            let trial_args = strategy.ask(elapsed);
            let model = match Self::train_trial(trial_args.clone(), &start, duration_secs) {
                TrialOutcome::TimedOut => break,
                TrialOutcome::Failed => continue,
                TrialOutcome::Success(m) => *m,
            };
            let model = if let Some(max_bytes) = model_size_bytes {
                match Self::check_model_size(model, max_bytes, &start, duration_secs) {
                    Some(m) => m,
                    None => continue,
                }
            } else {
                model
            };
            if let Ok(score) = Self::evaluate(&model, &val_path, k, &metric) {
                if best_args.is_none() || score > best_score {
                    best_score = score;
                    strategy.update_best(&trial_args);
                    best_args = Some(trial_args);
                }
            }
        }

        Self::finish_with_best(best_args, autotune_args.verbose)
    }

    /// Spawn a training trial in a background thread and wait for it to finish or time out.
    ///
    /// Returns [`TrialOutcome::TimedOut`] if the time budget expires while training,
    /// [`TrialOutcome::Failed`] if training diverged or panicked, or
    /// [`TrialOutcome::Success`] with the trained model on success.
    fn train_trial(trial_args: Args, start: &Instant, duration_secs: f64) -> TrialOutcome {
        let abort_flag = Arc::new(AtomicBool::new(false));
        let abort_clone = Arc::clone(&abort_flag);
        let handle =
            std::thread::spawn(move || FastText::train_with_abort(trial_args, abort_clone));

        // Poll until training finishes or the time budget expires.
        let timed_out = loop {
            if handle.is_finished() {
                break false;
            }
            if start.elapsed().as_secs_f64() >= duration_secs {
                abort_flag.store(true, Ordering::Relaxed);
                break true;
            }
            std::thread::sleep(Duration::from_millis(10));
        };

        let model_result = handle.join();
        if timed_out {
            return TrialOutcome::TimedOut;
        }
        match model_result {
            Ok(Ok(m)) => TrialOutcome::Success(Box::new(m)),
            Ok(Err(_)) | Err(_) => TrialOutcome::Failed,
        }
    }

    /// Quantize `model`, measure its serialized size, and return it if within `max_bytes`.
    ///
    /// Returns `None` if quantization fails, the model is too large, or the time
    /// budget expires before or after quantization.
    fn check_model_size(
        mut model: FastText,
        max_bytes: u64,
        start: &Instant,
        duration_secs: f64,
    ) -> Option<FastText> {
        if start.elapsed().as_secs_f64() >= duration_secs {
            return None;
        }
        let qargs = Args::default();
        if model.quantize(&qargs).is_err() {
            return None;
        }
        if start.elapsed().as_secs_f64() >= duration_secs {
            return None;
        }
        let tmp_path = std::env::temp_dir().join(format!(
            "fasttext_autotune_size_{}_{}.ftz",
            std::process::id(),
            SIZE_CHECK_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let tmp_str = tmp_path.to_string_lossy().to_string();
        if model.save_model(&tmp_str).is_err() {
            std::fs::remove_file(&tmp_path).ok();
            return None;
        }
        let ftz_size = std::fs::metadata(&tmp_path)
            .map(|m| m.len())
            .unwrap_or(u64::MAX);
        std::fs::remove_file(&tmp_path).ok();
        if ftz_size > max_bytes {
            None
        } else {
            Some(model)
        }
    }

    /// Retrain from scratch using the best-found args, or return an error if no
    /// trial completed successfully within the time budget.
    fn finish_with_best(best_args: Option<Args>, original_verbose: i32) -> Result<FastText> {
        match best_args {
            None => Err(FastTextError::InvalidArgument(
                "Autotune: no trial completed successfully within the time budget. \
                 Consider increasing autotune-duration."
                    .to_string(),
            )),
            Some(mut final_args) => {
                final_args.verbose = original_verbose;
                FastText::train(final_args)
            }
        }
    }

    /// Evaluate `model` on the validation file at `val_path`.
    ///
    /// Dispatches to the correct metric based on `metric`:
    /// - `"f1"` — macro F1 score (`meter.f1()`)
    /// - `"f1:LABEL"` — per-label F1 score for the named label
    /// - Other values — falls back to macro F1
    fn evaluate(model: &FastText, val_path: &Path, k: usize, metric: &str) -> Result<f64> {
        let file = std::fs::File::open(val_path).map_err(FastTextError::IoError)?;
        let mut reader = BufReader::new(file);
        let meter = model.test_model(&mut reader, k, 0.0)?;
        // Dispatch based on metric string.
        if metric == "f1" {
            Ok(meter.f1())
        } else if let Some(label_name) = metric.strip_prefix("f1:") {
            // Per-label F1: look up the label ID in the model's dictionary.
            let label_id = model.dict().get_id(label_name).unwrap_or(-1);
            Ok(meter.f1_for_label(label_id))
        } else {
            // Unknown or unsupported metric: fall back to macro F1.
            Ok(meter.f1())
        }
    }
}

// Tests

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]
    use super::*;
    use crate::args::Args;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    // Helpers

    /// Unique temp-file counter to avoid races between parallel tests.
    static FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Write content to a uniquely-named temp file. Returns the path.
    fn write_temp(content: &str, tag: &str) -> std::path::PathBuf {
        let id = FILE_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "fasttext_autotune_{}_{}_{}.txt",
            tag,
            std::process::id(),
            id
        ));
        std::fs::write(&path, content).expect("Failed to write temp file");
        path
    }

    /// Small supervised training dataset — 2 classes, 20 examples each.
    fn make_train_data() -> String {
        let mut data = String::new();
        for _ in 0..20 {
            data.push_str(
                "__label__sports basketball player sport game team score win tournament championship\n",
            );
        }
        for _ in 0..20 {
            data.push_str(
                "__label__food apple orange banana mango fruit eat cook recipe meal dessert\n",
            );
        }
        data
    }

    /// Small validation dataset — same classes, distinct sentences.
    fn make_val_data() -> String {
        let mut data = String::new();
        for _ in 0..10 {
            data.push_str("__label__sports sport player game win score\n");
        }
        for _ in 0..10 {
            data.push_str("__label__food banana fruit eat recipe cook\n");
        }
        data
    }

    /// Build Args configured for fast supervised training (for tests).
    fn make_fast_supervised_args(input: &std::path::Path) -> Args {
        let mut args = Args::default();
        args.input = input.to_path_buf();
        args.output = std::path::PathBuf::from("/dev/null");
        args.apply_supervised_defaults();
        args.dim = 10;
        args.epoch = 3;
        args.min_count = 1;
        args.lr = 0.1;
        args.bucket = 0;
        args.thread = 1;
        args.seed = 42;
        args
    }

    // VAL-AUTO-001: Autotune activation

    /// VAL-AUTO-001: Autotune is inactive (validation file empty) by default.
    #[test]
    fn test_autotune_activation_default_inactive() {
        let args = Args::default();
        assert!(
            !args.has_autotune(),
            "has_autotune() should be false when validation file is empty"
        );
        assert!(
            args.autotune_validation_file.as_os_str().is_empty(),
            "Default autotune_validation_file should be empty"
        );
    }

    /// VAL-AUTO-001: Setting a non-empty validation file activates autotune.
    #[test]
    fn test_autotune_activation_when_validation_file_set() {
        let mut args = Args::default();
        args.autotune_validation_file = std::path::PathBuf::from("some_validation_file.txt");
        assert!(
            args.has_autotune(),
            "has_autotune() should be true when validation file is set"
        );
    }

    // VAL-AUTO-002: Default duration and configurability

    /// VAL-AUTO-002: Default autotune duration is 300 seconds.
    #[test]
    fn test_autotune_duration_default() {
        let args = Args::default();
        assert_eq!(
            args.autotune_duration, 300,
            "Default autotune duration should be 300 seconds"
        );
    }

    /// VAL-AUTO-002: Autotune duration can be customized.
    #[test]
    fn test_autotune_duration_custom() {
        let mut args = Args::default();
        args.autotune_duration = 60;
        assert_eq!(
            args.autotune_duration, 60,
            "Autotune duration should reflect the custom value"
        );

        args.autotune_duration = 1;
        assert_eq!(
            args.autotune_duration, 1,
            "Should accept duration of 1 second"
        );

        args.autotune_duration = 3600;
        assert_eq!(
            args.autotune_duration, 3600,
            "Should accept large durations"
        );
    }

    // VAL-AUTO-003: Respects time budget

    /// VAL-AUTO-003: Autotune completes within a bounded time.
    ///
    /// Runs autotune with a short duration (3 seconds) and verifies that the
    /// total elapsed time (including the final retrain) is within a reasonable
    /// bound of `duration + 30 seconds` (feature-spec tolerance).
    #[test]
    fn test_autotune_time_budget() {
        let train_data = make_train_data();
        let train_path = write_temp(&train_data, "time_budget_train");
        let val_data = make_val_data();
        let val_path = write_temp(&val_data, "time_budget_val");

        let mut args = make_fast_supervised_args(&train_path);
        args.epoch = 1; // fast per-trial training
        args.dim = 5;
        args.autotune_validation_file = val_path.clone();
        args.autotune_duration = 3; // 3-second budget

        let start = Instant::now();
        let result = Autotune::run(args);
        let elapsed = start.elapsed();

        std::fs::remove_file(&train_path).ok();
        std::fs::remove_file(&val_path).ok();

        // Must complete (possibly with error if no trial succeeded, but must not hang).
        // Allow up to duration + 30 seconds as per feature spec.
        let max_allowed = Duration::from_secs(3 + 30);
        assert!(
            elapsed <= max_allowed,
            "Autotune ran for {:?} which exceeds 3 + 30 seconds",
            elapsed
        );

        // The result should be a valid model (or a recoverable error).
        match result {
            Ok(model) => {
                let preds = model.predict("basketball player sport", 1, 0.0);
                assert!(
                    !preds.is_empty(),
                    "Model returned by autotune should produce predictions"
                );
            }
            Err(_e) => {
                // If no trial succeeded (edge case), error is acceptable.
                // The time constraint above still must hold.
            }
        }
    }

    // VAL-AUTO-004: Hyperparameter tuning

    /// VAL-AUTO-004: Autotune explores and potentially modifies hyperparameters.
    ///
    /// Uses a setup where the initial epoch is clearly suboptimal (epoch=1) so
    /// that higher epoch values discovered by the strategy will achieve better F1.
    #[test]
    fn test_autotune_tunes_params() {
        let train_data = make_train_data();
        let train_path = write_temp(&train_data, "tunes_params_train");
        let val_data = make_val_data();
        let val_path = write_temp(&val_data, "tunes_params_val");

        // Start with epoch=1 (suboptimal) so higher epochs beat the baseline.
        let mut args = make_fast_supervised_args(&train_path);
        args.epoch = 1; // deliberately suboptimal
        args.dim = 5;
        args.autotune_validation_file = val_path.clone();
        args.autotune_duration = 4; // 4 seconds: enough for several trials

        let result = Autotune::run(args);
        std::fs::remove_file(&train_path).ok();
        std::fs::remove_file(&val_path).ok();

        let model = result.expect("Autotune should succeed within 4 seconds");
        let best_args = model.args();

        // Verify returned args are within valid ranges.
        assert!(
            best_args.epoch >= 1 && best_args.epoch <= 100,
            "Best epoch {} out of range [1, 100]",
            best_args.epoch
        );
        assert!(
            best_args.lr >= 0.01 && best_args.lr <= 5.0,
            "Best lr {} out of range [0.01, 5.0]",
            best_args.lr
        );
        assert!(
            best_args.dim >= 1 && best_args.dim <= 1000,
            "Best dim {} out of range [1, 1000]",
            best_args.dim
        );
        assert!(
            best_args.word_ngrams >= 1 && best_args.word_ngrams <= 5,
            "Best wordNgrams {} out of range [1, 5]",
            best_args.word_ngrams
        );

        // The returned model must be usable.
        let preds = model.predict("basketball player sport game", 1, 0.0);
        assert!(
            !preds.is_empty(),
            "Autotune model should produce predictions"
        );
    }

    /// Verify that AutotuneStrategy::ask produces different args for trials > 1.
    ///
    /// This is a unit-level check that the strategy DOES explore the parameter space,
    /// independent of whether any trial outperforms the baseline.
    #[test]
    fn test_autotune_strategy_explores_params() {
        let train_path = write_temp(&make_train_data(), "strategy_dummy");
        let mut args = make_fast_supervised_args(&train_path);
        args.epoch = 5;
        args.autotune_duration = 300;
        std::fs::remove_file(&train_path).ok();

        let mut strategy = AutotuneStrategy::new(&args, 42);

        // Trial 1: should return original args unchanged.
        let trial1 = strategy.ask(0.0);
        assert_eq!(
            trial1.epoch, args.epoch,
            "Trial 1 must return original epoch"
        );
        assert_eq!(trial1.dim, args.dim, "Trial 1 must return original dim");

        // Trial 2: should differ in at least one parameter.
        let trial2 = strategy.ask(1.0);
        let epoch_differs = trial2.epoch != args.epoch;
        let lr_differs = (trial2.lr - args.lr).abs() > 1e-9;
        let dim_differs = trial2.dim != args.dim;

        assert!(
            epoch_differs || lr_differs || dim_differs,
            "Trial 2 must differ from original in at least one parameter \
             (epoch={}, lr={:.4}, dim={})",
            trial2.epoch,
            trial2.lr,
            trial2.dim
        );
    }

    // VAL-AUTO-005: Returns best parameters and usable model

    /// VAL-AUTO-005: Autotune returns a model that can be used for prediction.
    #[test]
    fn test_autotune_returns_model() {
        let train_data = make_train_data();
        let train_path = write_temp(&train_data, "returns_model_train");
        let val_data = make_val_data();
        let val_path = write_temp(&val_data, "returns_model_val");

        let mut args = make_fast_supervised_args(&train_path);
        args.epoch = 2;
        args.dim = 5;
        args.autotune_validation_file = val_path.clone();
        args.autotune_duration = 3;

        let result = Autotune::run(args);
        std::fs::remove_file(&train_path).ok();
        std::fs::remove_file(&val_path).ok();

        let model = result.expect("Autotune should return a model");

        // The returned model must produce valid predictions.
        let preds = model.predict("basketball player sport game", 1, 0.0);
        assert!(!preds.is_empty(), "Predictions must be non-empty");

        // Probabilities must be in [0, 1] range.
        for p in &preds {
            assert!(
                p.prob >= 0.0 && p.prob <= 1.0,
                "Prediction probability {} out of range [0, 1]",
                p.prob
            );
        }

        // The args stored in the model must have valid values.
        let best_args = model.args();
        assert!(best_args.epoch >= 1, "Best epoch must be >= 1");
        assert!(best_args.lr > 0.0, "Best lr must be positive");
        assert!(best_args.dim >= 1, "Best dim must be >= 1");
    }

    // VAL-AUTO-006: Minimal duration produces valid model

    /// VAL-AUTO-006: A 1-second autotune duration still produces a valid model.
    ///
    /// With tiny data and small hyperparameters, the first trial (which uses the
    /// initial args unchanged) should complete well within 1 second. The model
    /// returned from that trial is used for the final retrain.
    #[test]
    fn test_autotune_minimal_duration() {
        let train_data = make_train_data();
        let train_path = write_temp(&train_data, "minimal_dur_train");
        let val_data = make_val_data();
        let val_path = write_temp(&val_data, "minimal_dur_val");

        // Minimal hyperparameters so training is very fast.
        let mut args = make_fast_supervised_args(&train_path);
        args.epoch = 1;
        args.dim = 5;
        args.autotune_validation_file = val_path.clone();
        args.autotune_duration = 1; // 1-second budget

        let result = Autotune::run(args);
        std::fs::remove_file(&train_path).ok();
        std::fs::remove_file(&val_path).ok();

        let model = result.expect("Autotune with 1-second duration should return a model");

        // The model must be usable for prediction.
        let preds = model.predict("sport game basketball player", 1, 0.0);
        assert!(!preds.is_empty(), "Predictions must be non-empty");
        assert!(
            !preds[0].label.is_empty(),
            "Prediction label must be non-empty"
        );
    }

    // Additional AutotuneStrategy unit tests

    /// Verify update_best tracks best_minn_index and best_dsub_exponent correctly.
    #[test]
    fn test_autotune_strategy_update_best() {
        let train_path = write_temp(&make_train_data(), "strategy_update");
        let mut args = make_fast_supervised_args(&train_path);
        args.epoch = 5;
        args.autotune_duration = 300;
        std::fs::remove_file(&train_path).ok();

        let mut strategy = AutotuneStrategy::new(&args, 1);

        // Update best with minn=2 (index 1 in minn_choices=[0,2,3])
        let mut new_best = args.clone();
        new_best.minn = 2;
        new_best.dsub = 4; // dsub=4 → exponent=2
        strategy.update_best(&new_best);

        assert_eq!(strategy.best_minn_index, 1, "minn=2 should map to index 1");
        assert_eq!(
            strategy.best_dsub_exponent, 2,
            "dsub=4 should give exponent 2"
        );
    }

    /// Verify normal_sample produces values in a plausible range.
    #[test]
    fn test_normal_sample_basic() {
        let mut rng = MinstdRng::new(42);
        let samples: Vec<f64> = (0..1000).map(|_| normal_sample(&mut rng)).collect();

        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let var: f64 = samples
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
            / samples.len() as f64;
        let stddev = var.sqrt();

        // Mean should be close to 0, stddev close to 1.
        assert!(
            mean.abs() < 0.15,
            "Normal sample mean should be near 0, got {}",
            mean
        );
        assert!(
            (stddev - 1.0).abs() < 0.2,
            "Normal sample stddev should be near 1, got {}",
            stddev
        );
    }

    /// Verify update_arg_gauss respects min/max bounds.
    #[test]
    fn test_update_arg_gauss_bounds() {
        let mut rng = MinstdRng::new(7);
        for _ in 0..100 {
            let v = update_arg_gauss(10.0, 5.0, 15.0, 3.0, 1.0, 0.0, false, &mut rng);
            assert!((5.0..=15.0).contains(&v), "Value {} out of [5, 15]", v);
            let vi = update_arg_gauss_i32(10, 5, 15, 3.0, 1.0, 0.0, false, &mut rng);
            assert!((5..=15).contains(&vi), "Value {} out of [5, 15]", vi);
        }
    }

    /// Verify sigma decreases over time in update_arg_gauss (higher t → smaller variance).
    #[test]
    fn test_update_arg_gauss_sigma_decreases() {
        // At t=0, sigma = startSigma = 3.0.
        // At t=1, sigma = endSigma = 1.0.
        // We verify that the variance of the output is smaller at t=1 vs t=0.
        let mut rng_early = MinstdRng::new(99);
        let mut rng_late = MinstdRng::new(99);

        let early_samples: Vec<f64> = (0..200)
            .map(|_| update_arg_gauss(10.0, 1.0, 1000.0, 3.0, 1.0, 0.0, false, &mut rng_early))
            .collect();
        let late_samples: Vec<f64> = (0..200)
            .map(|_| update_arg_gauss(10.0, 1.0, 1000.0, 3.0, 1.0, 1.0, false, &mut rng_late))
            .collect();

        let early_var: f64 = {
            let mean = early_samples.iter().sum::<f64>() / early_samples.len() as f64;
            early_samples
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<f64>()
                / early_samples.len() as f64
        };
        let late_var: f64 = {
            let mean = late_samples.iter().sum::<f64>() / late_samples.len() as f64;
            late_samples
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<f64>()
                / late_samples.len() as f64
        };

        assert!(
            early_var > late_var,
            "Early variance ({:.2}) should be greater than late variance ({:.2})",
            early_var,
            late_var
        );
    }

    /// Verify that Autotune::run returns an error when the validation file is not set.
    #[test]
    fn test_autotune_requires_validation_file() {
        let train_path = write_temp(&make_train_data(), "no_val_file");
        let mut args = make_fast_supervised_args(&train_path);
        args.autotune_duration = 1;
        // No validation file set!
        std::fs::remove_file(&train_path).ok();

        let result = Autotune::run(args);
        assert!(
            result.is_err(),
            "Autotune without validation file should return an error"
        );
    }

    /// Verify that Autotune::run returns an error for a missing validation file.
    #[test]
    fn test_autotune_missing_validation_file() {
        let train_path = write_temp(&make_train_data(), "missing_val");
        let mut args = make_fast_supervised_args(&train_path);
        args.autotune_duration = 1;
        args.autotune_validation_file = std::path::PathBuf::from("/nonexistent/path/validation.txt");
        std::fs::remove_file(&train_path).ok();

        let result = Autotune::run(args);
        assert!(
            result.is_err(),
            "Autotune with missing validation file should return an error"
        );
    }

    // Fix tests: parse_size_to_bytes

    #[test]
    fn test_parse_size_to_bytes() {
        assert_eq!(parse_size_to_bytes("100"), Some(100));
        assert_eq!(parse_size_to_bytes("100B"), Some(100));
        assert_eq!(parse_size_to_bytes("1K"), Some(1_024));
        assert_eq!(parse_size_to_bytes("1KB"), Some(1_024));
        assert_eq!(parse_size_to_bytes("2M"), Some(2 * 1_024 * 1_024));
        assert_eq!(parse_size_to_bytes("2MB"), Some(2 * 1_024 * 1_024));
        assert_eq!(parse_size_to_bytes("1G"), Some(1_024 * 1_024 * 1_024));
        assert_eq!(parse_size_to_bytes("1GB"), Some(1_024 * 1_024 * 1_024));
        // Fractional
        let expected = (1.5 * 1_024.0 * 1_024.0) as u64;
        assert_eq!(parse_size_to_bytes("1.5M"), Some(expected));
        // Empty / unknown
        assert_eq!(parse_size_to_bytes(""), None);
        assert_eq!(parse_size_to_bytes("100X"), None);
        // Case insensitive
        assert_eq!(parse_size_to_bytes("1m"), Some(1_024 * 1_024));
        assert_eq!(parse_size_to_bytes("1k"), Some(1_024));
    }

    // Fix tests: autotune evaluate() metric dispatch

    /// Verify autotune runs successfully with a label-specific F1 metric.
    #[test]
    fn test_autotune_label_f1_metric() {
        let train_data = make_train_data();
        let train_path = write_temp(&train_data, "label_f1_train");
        let val_data = make_val_data();
        let val_path = write_temp(&val_data, "label_f1_val");

        let mut args = make_fast_supervised_args(&train_path);
        args.epoch = 1;
        args.dim = 5;
        args.autotune_validation_file = val_path.clone();
        args.autotune_duration = 3;
        // Use per-label F1 metric for the __label__sports class.
        args.autotune_metric = "f1:__label__sports".to_string();

        let result = Autotune::run(args);
        std::fs::remove_file(&train_path).ok();
        std::fs::remove_file(&val_path).ok();

        // Should complete without error and produce a usable model.
        let model = result.expect("Autotune with label F1 metric should succeed");
        let preds = model.predict("basketball player sport", 1, 0.0);
        assert!(
            !preds.is_empty(),
            "Model from label-F1 autotune should produce predictions"
        );
    }

    /// Verify autotune with default F1 metric still works after the dispatch refactor.
    #[test]
    fn test_autotune_default_f1_metric() {
        let train_data = make_train_data();
        let train_path = write_temp(&train_data, "default_f1_train");
        let val_data = make_val_data();
        let val_path = write_temp(&val_data, "default_f1_val");

        let mut args = make_fast_supervised_args(&train_path);
        args.epoch = 1;
        args.dim = 5;
        args.autotune_validation_file = val_path.clone();
        args.autotune_duration = 3;
        args.autotune_metric = "f1".to_string(); // explicit default

        let result = Autotune::run(args);
        std::fs::remove_file(&train_path).ok();
        std::fs::remove_file(&val_path).ok();

        let model = result.expect("Autotune with default F1 metric should succeed");
        let preds = model.predict("banana fruit eat recipe", 1, 0.0);
        assert!(
            !preds.is_empty(),
            "Model from default-F1 autotune should produce predictions"
        );
    }

    // Fix tests: autotune model_size enforcement

    /// Verify the quantize-save-size-check mechanism works correctly.
    ///
    /// Trains a small model, quantizes it, and checks that its .ftz size
    /// is within expected bounds. This validates the core of model_size enforcement
    /// without relying on autotune timing behaviour.
    #[test]
    fn test_autotune_model_size_check_mechanism() {
        let train_data = make_train_data();
        let train_path = write_temp(&train_data, "size_check_mech");
        let mut args = make_fast_supervised_args(&train_path);
        args.epoch = 1;
        args.dim = 5;
        let mut model = FastText::train(args).expect("Training should succeed");
        std::fs::remove_file(&train_path).ok();

        // Quantize the model using default settings.
        let qargs = Args::default();
        model
            .quantize(&qargs)
            .expect("Quantize should succeed for size-check path");

        // Save and measure size.
        let tmp_path =
            std::env::temp_dir().join(format!("test_size_check_mech_{}.ftz", std::process::id()));
        model
            .save_model(tmp_path.to_str().unwrap())
            .expect("Save should succeed");
        let ftz_size = std::fs::metadata(&tmp_path).map(|m| m.len()).unwrap_or(0);
        std::fs::remove_file(&tmp_path).ok();

        // A real quantized model must be larger than 1 byte.
        assert!(
            ftz_size > 1,
            "Quantized model should be > 1 byte; got {} bytes",
            ftz_size
        );

        // Our tiny test model should be well under 10 MB.
        let ten_mb = 10u64 * 1024 * 1024;
        assert!(
            ftz_size < ten_mb,
            "Tiny test model should be < 10 MB; got {} bytes",
            ftz_size
        );
    }
}
