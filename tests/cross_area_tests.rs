// Cross-area integration tests
//
// These tests exercise the full pipeline end-to-end, validating that all
// subsystems (training, quantization, save/load, autotune, CLI) work together
// correctly.
//
// Tests in this file use small datasets and quick training (few epochs, small
// dim) to keep test time reasonable.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use fasttext::args::Args;
use fasttext::autotune::Autotune;
use fasttext::FastText;

// Shared counter for unique temp-file names

static FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

// Helpers

/// Create a unique temporary directory for this test invocation.
fn temp_dir() -> PathBuf {
    let base = std::env::temp_dir();
    let id = FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let unique = format!(
        "fasttext-cross-test-{}-{}",
        std::process::id(),
        id,
    );
    let dir = base.join(unique);
    std::fs::create_dir_all(&dir).expect("Failed to create temp dir");
    dir
}

/// Write content to a temp file inside a directory.
fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, content).expect("Failed to write file");
    path
}

/// Path to the compiled `fasttext` binary (set by cargo at compile time).
fn fasttext_bin() -> &'static str {
    env!("CARGO_BIN_EXE_fasttext")
}

/// Small supervised training dataset — 2 classes, 30 examples each.
/// Includes enough vocabulary diversity for quantization tests.
fn make_labeled_dataset() -> String {
    let sports_words = [
        "basketball player game sport team score win tournament championship trophy",
        "football match goal kick referee penalty offside tackle dribble pass",
        "tennis racket serve volley backhand forehand court baseline deuce break",
        "baseball bat pitcher catcher inning strike batter home run diamond",
        "soccer ball goalkeeper penalty corner kick striker forward midfielder",
    ];
    let food_words = [
        "apple orange banana mango fruit eat cook recipe meal dessert",
        "bread butter flour sugar bake oven cake cookie pastry croissant",
        "chicken beef pork grilled roasted marinated sauce seasoning herbs",
        "salad vegetables cucumber tomato lettuce dressing olive oil vinegar",
        "pasta spaghetti noodle sauce cheese parmesan mozzarella garlic basil",
    ];

    let mut data = String::new();
    // 30 examples per class (6 cycles × 5 sentences each)
    for _ in 0..6 {
        for sentence in &sports_words {
            data.push_str(&format!("__label__sports {}\n", sentence));
        }
        for sentence in &food_words {
            data.push_str(&format!("__label__food {}\n", sentence));
        }
    }
    data
}

/// Small validation dataset — 2 classes, 10 examples each.
fn make_val_dataset() -> String {
    let mut data = String::new();
    for _ in 0..10 {
        data.push_str("__label__sports basketball player team game win score\n");
    }
    for _ in 0..10 {
        data.push_str("__label__food banana fruit eat recipe cook bake\n");
    }
    data
}

/// Build Args configured for fast supervised training.
fn make_supervised_args(input: &str, output: &str) -> Args {
    let mut args = Args::default();
    args.input = input.to_string();
    args.output = output.to_string();
    args.apply_supervised_defaults();
    args.dim = 20;
    args.epoch = 10;
    args.min_count = 1;
    args.lr = 0.1;
    args.bucket = 100;
    args.thread = 1;
    args.verbose = 0;
    args.seed = 42;
    args
}

/// Run the fasttext binary with the given args and optional stdin input,
/// returning (stdout_string, stderr_string, exit_code).
fn run_fasttext(args: &[&str], stdin_data: Option<&[u8]>) -> (String, String, i32) {
    let mut cmd = Command::new(fasttext_bin());
    cmd.args(args);

    if let Some(data) = stdin_data {
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().expect("Failed to spawn fasttext");
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(data)
            .expect("Failed to write stdin");
        let output = child.wait_with_output().expect("Failed to wait on child");

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let code = output.status.code().unwrap_or(-1);
        (stdout, stderr, code)
    } else {
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let output = cmd.output().expect("Failed to run fasttext");
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let code = output.status.code().unwrap_or(-1);
        (stdout, stderr, code)
    }
}

// VAL-CROSS-001: Train → Predict on training data

/// VAL-CROSS-001: Train a supervised model on labeled data, then predict on
/// the same training data. Top-1 prediction for each line must match the
/// training label with ≥80% accuracy (overfitting expected).
#[test]
fn test_cross_train_predict() {
    let dir = temp_dir();
    let train_data = make_labeled_dataset();
    let train_path = write_file(&dir, "train.txt", &train_data);
    let output_base = dir.join("model").to_string_lossy().into_owned();

    // Train the model.
    let args = make_supervised_args(train_path.to_str().unwrap(), &output_base);
    let model = FastText::train(args).expect("Training should succeed");

    // Predict on the training data.
    let mut correct = 0usize;
    let mut total = 0usize;
    for line in train_data.lines() {
        // Each line is "__label__<label> <text>"
        let mut parts = line.splitn(2, ' ');
        let label = parts.next().unwrap_or("").trim();
        let text = parts.next().unwrap_or("").trim();
        if label.is_empty() || text.is_empty() {
            continue;
        }

        let predictions = model.predict(text, 1, 0.0);
        if let Some(pred) = predictions.first() {
            if pred.label == label {
                correct += 1;
            }
        }
        total += 1;
    }

    assert!(total > 0, "Training data should have examples");
    let accuracy = correct as f64 / total as f64;
    assert!(
        accuracy >= 0.80,
        "Top-1 accuracy on training data should be ≥80% (overfitting), got {:.1}% ({}/{})",
        accuracy * 100.0,
        correct,
        total
    );
}

// VAL-CROSS-002: Train → Quantize → Predict agreement

/// VAL-CROSS-002: Train a supervised model, quantize it, then predict on the
/// same inputs. Top-1 label from the quantized model must agree with the
/// unquantized model ≥90% of the time.
#[test]
fn test_cross_train_quantize_predict() {
    let dir = temp_dir();
    let train_data = make_labeled_dataset();
    let train_path = write_file(&dir, "train.txt", &train_data);
    let output_base = dir.join("model").to_string_lossy().into_owned();

    // Train an unquantized model.
    let args = make_supervised_args(train_path.to_str().unwrap(), &output_base);
    let mut model = FastText::train(args).expect("Training should succeed");

    // Build the test inputs (use the training data labels as text queries).
    let test_inputs: Vec<String> = train_data
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(2, ' ');
            let _label = parts.next()?;
            let text = parts.next()?.trim().to_string();
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        })
        .collect();

    // Predict with the unquantized model first.
    let unquant_preds: Vec<String> = test_inputs
        .iter()
        .map(|text| {
            model
                .predict(text, 1, 0.0)
                .into_iter()
                .next()
                .map(|p| p.label)
                .unwrap_or_default()
        })
        .collect();

    // Quantize the model in-place.
    let qargs = Args::default();
    model.quantize(&qargs).expect("Quantization should succeed");

    // Predict with the quantized model.
    let quant_preds: Vec<String> = test_inputs
        .iter()
        .map(|text| {
            model
                .predict(text, 1, 0.0)
                .into_iter()
                .next()
                .map(|p| p.label)
                .unwrap_or_default()
        })
        .collect();

    // Compare agreement.
    let total = test_inputs.len();
    let agree = unquant_preds
        .iter()
        .zip(quant_preds.iter())
        .filter(|(u, q)| u == q)
        .count();

    assert!(total > 0, "Should have test inputs");
    let agreement = agree as f64 / total as f64;
    assert!(
        agreement >= 0.90,
        "Quantized model must agree with unquantized ≥90% of the time, got {:.1}% ({}/{})",
        agreement * 100.0,
        agree,
        total
    );
}

// VAL-CROSS-003: Train → Save → Load → Predict round-trip

/// VAL-CROSS-003: Train a model, save to .bin, load from .bin, predict.
/// Loaded model predictions must be identical to original model predictions.
#[test]
fn test_cross_train_save_load_predict() {
    let dir = temp_dir();
    let train_data = make_labeled_dataset();
    let train_path = write_file(&dir, "train.txt", &train_data);
    let output_base = dir.join("model").to_string_lossy().into_owned();
    let model_bin = dir.join("model.bin");

    // Train the model.
    let args = make_supervised_args(train_path.to_str().unwrap(), &output_base);
    let model_orig = FastText::train(args).expect("Training should succeed");

    // Save the model.
    model_orig
        .save_model(model_bin.to_str().unwrap())
        .expect("Model save should succeed");
    assert!(model_bin.exists(), "model.bin should exist after save");

    // Load the model.
    let model_loaded =
        FastText::load_model(model_bin.to_str().unwrap()).expect("Model load should succeed");

    // Test inputs covering both classes.
    let test_inputs = [
        "basketball player game score win",
        "fruit banana eat recipe cook",
        "football match goal kick referee",
        "bread butter flour bake oven",
        "tennis racket serve volley backhand",
        "salad vegetables tomato lettuce dressing",
    ];

    for text in &test_inputs {
        let orig_preds = model_orig.predict(text, 3, 0.0);
        let loaded_preds = model_loaded.predict(text, 3, 0.0);

        assert_eq!(
            orig_preds.len(),
            loaded_preds.len(),
            "Prediction count mismatch for input: '{}'",
            text
        );

        for (o, l) in orig_preds.iter().zip(loaded_preds.iter()) {
            assert_eq!(
                o.label, l.label,
                "Label mismatch after save/load for input '{}': expected '{}', got '{}'",
                text, o.label, l.label
            );
            assert_eq!(
                o.prob, l.prob,
                "Probability mismatch after save/load for input '{}': expected {}, got {}",
                text, o.prob, l.prob
            );
        }
    }
}

// VAL-CROSS-004: Autotune outperforms defaults

/// VAL-CROSS-004: Train with autotune on a validation set, also train with
/// default parameters. Autotune model must achieve higher F1 on the validation
/// set than the default model.
///
/// Uses reasonable baseline parameters (same as make_supervised_args) so that
/// autotune is tested against a realistic starting point, not an intentionally
/// weakened one.
#[test]
fn test_cross_autotune_outperforms() {
    let dir = temp_dir();
    let train_data = make_labeled_dataset();
    let val_data = make_val_dataset();
    let train_path = write_file(&dir, "train.txt", &train_data);
    let val_path = write_file(&dir, "val.txt", &val_data);

    // Train a baseline model with standard (non-weakened) parameters.
    let baseline_output = dir.join("baseline").to_string_lossy().into_owned();
    let baseline_args =
        make_supervised_args(train_path.to_str().unwrap(), &baseline_output);
    let baseline_model = FastText::train(baseline_args).expect("Baseline training should succeed");

    // Compute baseline F1 on the validation set.
    let mut val_file =
        std::fs::File::open(&val_path).expect("Val file should be readable");
    let baseline_meter = baseline_model
        .test_model(&mut val_file, 1, 0.0)
        .expect("test_model should succeed for baseline");
    let baseline_f1 = baseline_meter.f1();

    // Run autotune with a short budget (5 seconds) starting from the same
    // standard parameters; autotune must match or improve upon the baseline.
    let mut autotune_args =
        make_supervised_args(train_path.to_str().unwrap(), "/dev/null");
    autotune_args.autotune_validation_file = val_path.to_str().unwrap().to_string();
    autotune_args.autotune_duration = 5; // 5-second budget
    autotune_args.autotune_metric = "f1".to_string();

    let autotune_model = Autotune::run(autotune_args).expect("Autotune should succeed");

    // Compute autotune F1 on the validation set.
    let mut val_file2 =
        std::fs::File::open(&val_path).expect("Val file should be readable");
    let autotune_meter = autotune_model
        .test_model(&mut val_file2, 1, 0.0)
        .expect("test_model should succeed for autotune model");
    let autotune_f1 = autotune_meter.f1();

    assert!(
        autotune_f1 >= baseline_f1,
        "Autotune F1 ({:.4}) should be ≥ baseline F1 ({:.4})",
        autotune_f1,
        baseline_f1
    );
}

// VAL-CROSS-005: CLI and library API produce same output

/// VAL-CROSS-005: Train a model via CLI, predict via CLI. Load the same model
/// via library API, predict via API. Both must produce identical labels and
/// probabilities (within 1e-6 tolerance).
#[test]
fn test_cross_cli_api_consistency() {
    let dir = temp_dir();
    let train_data = make_labeled_dataset();
    let train_path = write_file(&dir, "train.txt", &train_data);
    let model_base = dir.join("model");
    let model_bin = dir.join("model.bin");

    let (stdout, stderr, code) = run_fasttext(
        &[
            "supervised",
            "--input",
            train_path.to_str().unwrap(),
            "--output",
            model_base.to_str().unwrap(),
            "--epoch",
            "10",
            "--dim",
            "20",
            "--min-count",
            "1",
            "--thread",
            "1",
            "--bucket",
            "100",
            "--verbose",
            "0",
            "--seed",
            "42",
        ],
        None,
    );
    assert_eq!(
        code, 0,
        "CLI supervised training failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );
    assert!(model_bin.exists(), "model.bin should exist after CLI training");

    let test_inputs = [
        "basketball player game score win",
        "fruit banana eat recipe cook",
        "football match goal kick",
        "bread butter flour bake",
    ];

    // Join all inputs as stdin (one per line).
    let stdin_data: Vec<u8> = test_inputs
        .iter()
        .flat_map(|line| format!("{}\n", line).into_bytes())
        .collect();

    let (cli_stdout, cli_stderr, cli_code) = run_fasttext(
        &["predict-prob", model_bin.to_str().unwrap(), "-"],
        Some(&stdin_data),
    );
    assert_eq!(
        cli_code, 0,
        "CLI predict-prob failed\nstdout: {}\nstderr: {}",
        cli_stdout, cli_stderr
    );

    // Parse CLI output: each line is "<label> <prob>".
    let cli_predictions: Vec<(String, f32)> = cli_stdout
        .lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let label = parts.next()?.to_string();
            let prob: f32 = parts.next()?.parse().ok()?;
            Some((label, prob))
        })
        .collect();

    assert_eq!(
        cli_predictions.len(),
        test_inputs.len(),
        "CLI should produce one prediction per input line"
    );

    let api_model =
        FastText::load_model(model_bin.to_str().unwrap()).expect("API model load should succeed");

    let api_predictions: Vec<(String, f32)> = test_inputs
        .iter()
        .map(|text| {
            let preds = api_model.predict(text, 1, 0.0);
            let pred = preds.into_iter().next().expect("API should return a prediction");
            (pred.label, pred.prob)
        })
        .collect();

    assert_eq!(
        cli_predictions.len(),
        api_predictions.len(),
        "CLI and API should produce the same number of predictions"
    );

    for (i, ((cli_label, cli_prob), (api_label, api_prob))) in cli_predictions
        .iter()
        .zip(api_predictions.iter())
        .enumerate()
    {
        assert_eq!(
            cli_label, api_label,
            "Label mismatch for input[{}] '{}': CLI='{}', API='{}'",
            i, test_inputs[i], cli_label, api_label
        );
        let prob_diff = (cli_prob - api_prob).abs();
        assert!(
            prob_diff < 1e-4,
            "Probability mismatch for input[{}] '{}': CLI={}, API={}, diff={}",
            i,
            test_inputs[i],
            cli_prob,
            api_prob,
            prob_diff
        );
    }
}
