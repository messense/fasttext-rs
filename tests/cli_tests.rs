// CLI integration tests
//
// These tests build the `fasttext` binary via `env!("CARGO_BIN_EXE_fasttext")`
// and invoke it through `std::process::Command`, validating stdin/stdout
// behaviour for each subcommand.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Path to the compiled `fasttext` binary (set by cargo at compile time).
fn fasttext_bin() -> &'static str {
    env!("CARGO_BIN_EXE_fasttext")
}

/// Path to the cooking reference model.
fn cooking_model() -> &'static str {
    "tests/fixtures/cooking.model.bin"
}

/// Create a unique temporary directory for this test invocation.
fn temp_dir() -> PathBuf {
    let base = std::env::temp_dir();
    let unique = format!(
        "fasttext-cli-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dir = base.join(unique);
    std::fs::create_dir_all(&dir).expect("Failed to create temp dir");
    dir
}

/// Write a small labeled training corpus (2 categories: cat / dog).
fn write_train_data(path: &Path) {
    let data = "\
__label__cat the cat meows
__label__dog the dog barks
__label__cat cats are feline animals
__label__dog dogs are canine animals
__label__cat a cat loves to sleep
__label__dog a dog loves to run
__label__cat the kitten is a young cat
__label__dog the puppy is a young dog
__label__cat cats purr when happy
__label__dog dogs wag their tails
";
    std::fs::write(path, data).expect("Failed to write training data");
}

/// Write a small labeled test corpus (same 2 categories).
#[allow(dead_code)]
fn write_test_data(path: &Path) {
    let data = "\
__label__cat this cat sleeps a lot
__label__dog this dog runs fast
__label__cat the cat is purring
";
    std::fs::write(path, data).expect("Failed to write test data");
}

/// Run the fasttext binary with the given args and optional stdin, returning
/// (stdout_string, stderr_string, exit_code).
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

/// Train a small supervised model and return the model path.
/// Uses --epoch 3, --dim 10, --min-count 1, --thread 1.
fn train_small_supervised_model(dir: &Path) -> PathBuf {
    let train_file = dir.join("train.txt");
    let model_base = dir.join("model");

    write_train_data(&train_file);

    let (stdout, stderr, code) = run_fasttext(
        &[
            "supervised",
            "--input",
            train_file.to_str().unwrap(),
            "--output",
            model_base.to_str().unwrap(),
            "--epoch",
            "3",
            "--dim",
            "10",
            "--min-count",
            "1",
            "--thread",
            "1",
        ],
        None,
    );

    assert_eq!(
        code, 0,
        "supervised training failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    let model_bin = dir.join("model.bin");
    assert!(
        model_bin.exists(),
        "model.bin not created after supervised training"
    );
    model_bin
}

// ---------------------------------------------------------------------------
// VAL-CLI-001: Training commands produce correct model types
// ---------------------------------------------------------------------------

#[test]
fn test_cli_train_supervised() {
    let dir = temp_dir();
    let model_bin = train_small_supervised_model(&dir);

    // Load the model and verify it is a supervised model.
    let model = fasttext::FastText::load_model(model_bin.to_str().unwrap())
        .expect("Failed to load trained model");
    assert_eq!(
        model.args().model(),
        fasttext::args::ModelName::SUP,
        "supervised training must produce a supervised model"
    );
}

#[test]
fn test_cli_train_skipgram() {
    let dir = temp_dir();
    let train_file = dir.join("train.txt");
    let model_base = dir.join("model");

    // Write word2vec-style corpus (no labels).
    let corpus = "\
the cat meows at the dog
the dog barks at the cat
cats and dogs are both pets
";
    std::fs::write(&train_file, corpus).unwrap();

    let (stdout, stderr, code) = run_fasttext(
        &[
            "skipgram",
            "--input",
            train_file.to_str().unwrap(),
            "--output",
            model_base.to_str().unwrap(),
            "--epoch",
            "2",
            "--dim",
            "10",
            "--min-count",
            "1",
            "--thread",
            "1",
        ],
        None,
    );

    assert_eq!(
        code, 0,
        "skipgram training failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    let model_bin = dir.join("model.bin");
    assert!(model_bin.exists(), "model.bin not created after skipgram training");

    let model = fasttext::FastText::load_model(model_bin.to_str().unwrap())
        .expect("Failed to load skipgram model");
    assert_eq!(
        model.args().model(),
        fasttext::args::ModelName::SG,
        "skipgram training must produce a SG model"
    );
}

#[test]
fn test_cli_train_cbow() {
    let dir = temp_dir();
    let train_file = dir.join("train.txt");
    let model_base = dir.join("model");

    let corpus = "\
the cat meows at the dog
the dog barks at the cat
cats and dogs are both pets
";
    std::fs::write(&train_file, corpus).unwrap();

    let (stdout, stderr, code) = run_fasttext(
        &[
            "cbow",
            "--input",
            train_file.to_str().unwrap(),
            "--output",
            model_base.to_str().unwrap(),
            "--epoch",
            "2",
            "--dim",
            "10",
            "--min-count",
            "1",
            "--thread",
            "1",
        ],
        None,
    );

    assert_eq!(
        code, 0,
        "cbow training failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    let model_bin = dir.join("model.bin");
    assert!(model_bin.exists(), "model.bin not created after cbow training");

    let model = fasttext::FastText::load_model(model_bin.to_str().unwrap())
        .expect("Failed to load cbow model");
    assert_eq!(
        model.args().model(),
        fasttext::args::ModelName::CBOW,
        "cbow training must produce a CBOW model"
    );
}

// ---------------------------------------------------------------------------
// VAL-CLI-002: predict reads stdin and outputs labels
// ---------------------------------------------------------------------------

#[test]
fn test_cli_predict_stdin() {
    // Use cooking model: input line → one label per line starting with __label__
    let input = b"Which baking dish is best to bake a banana bread ?\n";

    let (stdout, stderr, code) = run_fasttext(
        &["predict", cooking_model(), "-"],
        Some(input),
    );

    assert_eq!(
        code, 0,
        "predict failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1, "One input line should produce one output line for k=1");
    assert!(
        lines[0].starts_with("__label__"),
        "Output should start with __label__, got: {}",
        lines[0]
    );
}

// ---------------------------------------------------------------------------
// VAL-CLI-003: predict-prob outputs labels with probabilities
// ---------------------------------------------------------------------------

#[test]
fn test_cli_predict_prob() {
    let input = b"Which baking dish is best to bake a banana bread ?\n";

    let (stdout, stderr, code) = run_fasttext(
        &["predict-prob", cooking_model(), "-"],
        Some(input),
    );

    assert_eq!(
        code, 0,
        "predict-prob failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1, "One input line should produce one output line for k=1");

    // Each line should have exactly two tokens: label and probability.
    let parts: Vec<&str> = lines[0].split_whitespace().collect();
    assert_eq!(
        parts.len(),
        2,
        "predict-prob output should be 'label prob', got: {}",
        lines[0]
    );
    assert!(
        parts[0].starts_with("__label__"),
        "First token should be a label, got: {}",
        parts[0]
    );
    let prob: f64 = parts[1]
        .parse()
        .expect("Second token should be a probability float");
    assert!(
        (0.0..=1.0).contains(&prob),
        "Probability must be in [0, 1], got: {}",
        prob
    );
}

// ---------------------------------------------------------------------------
// VAL-CLI-004: predict with k and threshold parameters
// ---------------------------------------------------------------------------

#[test]
fn test_cli_predict_k() {
    let input = b"Which baking dish is best to bake a banana bread ?\n";

    // k=2 → should produce 2 output lines for 1 input line.
    let (stdout, stderr, code) = run_fasttext(
        &["predict", cooking_model(), "-", "2"],
        Some(input),
    );

    assert_eq!(
        code, 0,
        "predict with k=2 failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines.len(),
        2,
        "One input line with k=2 should produce 2 output lines, got: {:?}",
        lines
    );
    for line in &lines {
        assert!(
            line.starts_with("__label__"),
            "Each output line should start with __label__, got: {}",
            line
        );
    }
}

#[test]
fn test_cli_predict_threshold() {
    let input = b"Which baking dish is best to bake a banana bread ?\n";

    // Threshold = 0.0 → at least one prediction returned.
    let (stdout, stderr, code) = run_fasttext(
        &["predict", cooking_model(), "-", "3", "0.0"],
        Some(input),
    );
    assert_eq!(
        code, 0,
        "predict with threshold=0.0 failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );
    let lines_low: Vec<&str> = stdout.lines().collect();
    assert!(
        !lines_low.is_empty(),
        "With threshold=0.0, at least one prediction should be returned"
    );

    // Threshold = 1.0 → very high threshold, should return ≤ lines_low.len() predictions.
    let (stdout_hi, stderr_hi, code_hi) = run_fasttext(
        &["predict", cooking_model(), "-", "3", "1.0"],
        Some(input),
    );
    assert_eq!(
        code_hi, 0,
        "predict with threshold=1.0 failed\nstdout: {}\nstderr: {}",
        stdout_hi, stderr_hi
    );
    let lines_hi: Vec<&str> = stdout_hi.lines().collect();
    assert!(
        lines_hi.len() <= lines_low.len(),
        "High threshold should return ≤ predictions vs low threshold. hi={}, low={}",
        lines_hi.len(),
        lines_low.len()
    );
}

// ---------------------------------------------------------------------------
// VAL-CLI-006: test prints metrics
// ---------------------------------------------------------------------------

#[test]
fn test_cli_test_metrics() {
    let dir = temp_dir();
    let test_file = dir.join("test.txt");

    // Create a small test file with cooking-related labels.
    let test_data = "\
__label__baking I love baking bread
__label__bread the bread is fresh
";
    std::fs::write(&test_file, test_data).unwrap();

    let (stdout, stderr, code) = run_fasttext(
        &[
            "test",
            cooking_model(),
            test_file.to_str().unwrap(),
        ],
        None,
    );

    assert_eq!(
        code, 0,
        "test command failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    // Output should contain N (example count), P@1, and R@1.
    assert!(
        stdout.contains("N\t"),
        "test output should contain 'N\\t...', got:\n{}",
        stdout
    );
    assert!(
        stdout.contains("P@1"),
        "test output should contain 'P@1', got:\n{}",
        stdout
    );
    assert!(
        stdout.contains("R@1"),
        "test output should contain 'R@1', got:\n{}",
        stdout
    );

    // Verify that N is a parseable integer.
    for line in stdout.lines() {
        if line.starts_with("N\t") {
            let n_str = line.trim_start_matches("N\t");
            let n: u64 = n_str.parse().expect("N value should be an integer");
            assert_eq!(n, 2, "N should equal the number of test examples");
        }
    }
}

// ---------------------------------------------------------------------------
// VAL-CLI-007: test-label prints per-label metrics
// ---------------------------------------------------------------------------

#[test]
fn test_cli_test_label() {
    let dir = temp_dir();
    let test_file = dir.join("test.txt");

    let test_data = "\
__label__baking I love baking bread
__label__bread the bread is fresh
";
    std::fs::write(&test_file, test_data).unwrap();

    let (stdout, stderr, code) = run_fasttext(
        &[
            "test-label",
            cooking_model(),
            test_file.to_str().unwrap(),
        ],
        None,
    );

    assert_eq!(
        code, 0,
        "test-label command failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    // Each output line should have 4 tab-separated fields:
    // label_name, precision, recall, f1
    assert!(
        !stdout.is_empty(),
        "test-label should produce non-empty output"
    );

    let mut found_label_row = false;
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() == 4 {
            assert!(
                parts[0].starts_with("__label__"),
                "First column should be a label, got: {}",
                parts[0]
            );
            // Parse the 3 numeric columns.
            for col in &parts[1..] {
                let _: f64 = col
                    .parse()
                    .unwrap_or_else(|_| panic!("Expected float in test-label output, got: {}", col));
            }
            found_label_row = true;
        }
    }
    assert!(found_label_row, "test-label should have at least one label row");
}

// ---------------------------------------------------------------------------
// VAL-CLI-012: Help, errors, and flag pass-through
// ---------------------------------------------------------------------------

#[test]
fn test_cli_help() {
    // Running with no args should show help (and exit non-zero due to arg_required_else_help).
    let (stdout, stderr, _) = run_fasttext(&[], None);

    let combined = format!("{}{}", stdout, stderr);
    // Help text should mention at least the main commands.
    assert!(
        combined.contains("supervised") || combined.contains("Usage") || combined.contains("fasttext"),
        "Help output should mention subcommands or usage, got:\n{}",
        combined
    );

    // --help should exit 0 and show usage.
    let (stdout_h, stderr_h, code_h) = run_fasttext(&["--help"], None);
    let combined_h = format!("{}{}", stdout_h, stderr_h);
    assert_eq!(
        code_h, 0,
        "--help should exit 0"
    );
    assert!(
        combined_h.contains("supervised") || combined_h.contains("predict"),
        "--help should mention subcommands, got:\n{}",
        combined_h
    );
}

#[test]
fn test_cli_unknown_command() {
    let (stdout, stderr, code) = run_fasttext(&["foobar_unknown_command"], None);
    assert_ne!(
        code, 0,
        "Unknown command should exit non-zero\nstdout: {}\nstderr: {}",
        stdout, stderr
    );
    // Error message should appear on stderr.
    let combined = format!("{}{}", stdout, stderr);
    assert!(
        !combined.is_empty(),
        "Some error output expected for unknown command"
    );
}

#[test]
fn test_cli_missing_model() {
    let (stdout, stderr, code) = run_fasttext(
        &["predict", "/tmp/nonexistent_model_xyz.bin", "-"],
        Some(b"hello world\n"),
    );
    assert_ne!(
        code, 0,
        "Missing model file should exit non-zero\nstdout: {}\nstderr: {}",
        stdout, stderr
    );
    assert!(
        stderr.contains("nonexistent_model_xyz") || stderr.contains("Error") || stderr.contains("error"),
        "Error message should mention the missing file or 'Error', got stderr: {}",
        stderr
    );
}

#[test]
fn test_cli_flag_passthrough() {
    // Train with custom --epoch and --dim; load model and verify the args were applied.
    let dir = temp_dir();
    let train_file = dir.join("train.txt");
    let model_base = dir.join("flagtest_model");

    write_train_data(&train_file);

    let (stdout, stderr, code) = run_fasttext(
        &[
            "supervised",
            "--input",
            train_file.to_str().unwrap(),
            "--output",
            model_base.to_str().unwrap(),
            "--epoch",
            "2",
            "--dim",
            "20",
            "--min-count",
            "1",
            "--thread",
            "1",
        ],
        None,
    );

    assert_eq!(
        code, 0,
        "flag_passthrough training failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    let model_bin = dir.join("flagtest_model.bin");
    let model = fasttext::FastText::load_model(model_bin.to_str().unwrap())
        .expect("Failed to load model for flag passthrough test");

    assert_eq!(
        model.args().epoch(),
        2,
        "epoch flag should be passed through to Args"
    );
    assert_eq!(
        model.args().dim(),
        20,
        "dim flag should be passed through to Args"
    );
}
