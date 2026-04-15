// fastText CLI — clap derive-based command-line interface
//
// Subcommands:
//   supervised   – train a supervised text classifier
//   skipgram     – train a skip-gram word vector model
//   cbow         – train a CBOW word vector model
//   predict      – predict top-k labels for each line of stdin / file
//   predict-prob – predict top-k labels + probabilities for each line
//   test         – evaluate model and print N, P@k, R@k
//   test-label   – evaluate model and print per-label P, R, F1

use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process;

use clap::{Args, Parser, Subcommand};

use fasttext::args::{Args as FTArgs, LossName, ModelName};
use fasttext::meter::Meter;
use fasttext::FastText;

// ---------------------------------------------------------------------------
// CLI structure
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "fasttext",
    about = "fastText text classification and representation learning",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a supervised text classifier
    Supervised(TrainArgs),
    /// Train a skip-gram word vector model
    Skipgram(TrainArgs),
    /// Train a CBOW word vector model
    Cbow(TrainArgs),
    /// Predict top-k labels for each input line
    Predict(PredictArgs),
    /// Predict top-k labels with probabilities for each input line
    #[command(name = "predict-prob")]
    PredictProb(PredictArgs),
    /// Evaluate model on test data: prints N, P@k, R@k
    Test(TestEvalArgs),
    /// Evaluate model on test data: prints per-label P, R, F1
    #[command(name = "test-label")]
    TestLabel(TestEvalArgs),
}

// ---------------------------------------------------------------------------
// Argument structs
// ---------------------------------------------------------------------------

/// Training arguments shared by supervised / skipgram / cbow.
#[derive(Args, Debug)]
struct TrainArgs {
    /// Training data file path
    #[arg(long)]
    input: String,

    /// Output model path (without .bin extension)
    #[arg(long)]
    output: String,

    /// Learning rate
    #[arg(long)]
    lr: Option<f64>,

    /// Learning rate update rate (tokens between LR updates)
    #[arg(long)]
    lr_update_rate: Option<i32>,

    /// Size of word vectors
    #[arg(long)]
    dim: Option<i32>,

    /// Size of the context window
    #[arg(long)]
    ws: Option<i32>,

    /// Number of training epochs
    #[arg(long)]
    epoch: Option<i32>,

    /// Minimal number of word occurrences
    #[arg(long)]
    min_count: Option<i32>,

    /// Minimal number of label occurrences
    #[arg(long)]
    min_count_label: Option<i32>,

    /// Number of negatives sampled
    #[arg(long)]
    neg: Option<i32>,

    /// Max length of word n-gram
    #[arg(long)]
    word_ngrams: Option<i32>,

    /// Loss function: ns (negative sampling), hs (hierarchical softmax),
    /// softmax, or ova (one-vs-all)
    #[arg(long)]
    loss: Option<String>,

    /// Number of buckets
    #[arg(long)]
    bucket: Option<i32>,

    /// Minimum length of character n-gram
    #[arg(long)]
    minn: Option<i32>,

    /// Maximum length of character n-gram
    #[arg(long)]
    maxn: Option<i32>,

    /// Number of threads
    #[arg(long)]
    thread: Option<i32>,

    /// Sampling threshold (subsampling high-frequency words)
    #[arg(long)]
    t: Option<f64>,

    /// Label prefix used to identify labels
    #[arg(long)]
    label: Option<String>,

    /// Verbose level (0 = quiet, 1 = progress, 2 = verbose)
    #[arg(long)]
    verbose: Option<i32>,

    /// Path to pretrained word vectors (.vec file)
    #[arg(long)]
    pretrained_vectors: Option<String>,

    /// Save output matrix (output.bin) alongside the model
    #[arg(long)]
    save_output: bool,

    /// Random seed
    #[arg(long)]
    seed: Option<i32>,
}

/// Arguments for predict / predict-prob.
#[derive(Args, Debug)]
struct PredictArgs {
    /// Path to the trained model (.bin or .ftz file)
    model: String,

    /// Input file path ('-' for stdin)
    #[arg(default_value = "-")]
    input: String,

    /// Number of top predictions per input line
    #[arg(default_value_t = 1)]
    k: usize,

    /// Minimum probability threshold (predictions below this are omitted)
    #[arg(default_value_t = 0.0)]
    threshold: f32,
}

/// Arguments for test / test-label.
#[derive(Args, Debug)]
struct TestEvalArgs {
    /// Path to the trained model (.bin or .ftz file)
    model: String,

    /// Test data file path
    test_file: String,

    /// Number of top predictions per example
    #[arg(default_value_t = 1)]
    k: usize,

    /// Minimum probability threshold
    #[arg(default_value_t = 0.0)]
    threshold: f32,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Supervised(args) => run_train(args, ModelName::SUP),
        Commands::Skipgram(args) => run_train(args, ModelName::SG),
        Commands::Cbow(args) => run_train(args, ModelName::CBOW),
        Commands::Predict(args) => run_predict(args, false),
        Commands::PredictProb(args) => run_predict(args, true),
        Commands::Test(args) => run_test(args, false),
        Commands::TestLabel(args) => run_test(args, true),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a loss name string to `LossName`.
fn parse_loss(s: &str) -> Option<LossName> {
    match s.to_lowercase().as_str() {
        "ns" => Some(LossName::NS),
        "hs" => Some(LossName::HS),
        "softmax" => Some(LossName::SOFTMAX),
        "ova" | "one-vs-all" | "ovr" => Some(LossName::OVA),
        _ => None,
    }
}

/// Load a model from `path`, exiting with an error message if it fails.
fn load_model_or_exit(path: &str) -> FastText {
    if !std::path::Path::new(path).exists() {
        eprintln!("Error: model file '{}' does not exist", path);
        process::exit(1);
    }
    match FastText::load_model(path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error loading model '{}': {}", path, e);
            process::exit(1);
        }
    }
}

/// Build `FTArgs` from a `TrainArgs`, applying model-specific defaults first.
fn build_ft_args(train_args: TrainArgs, model_name: ModelName) -> FTArgs {
    let mut args = FTArgs::default();

    // Apply model-specific defaults before user overrides.
    if model_name == ModelName::SUP {
        args.apply_supervised_defaults();
    } else {
        args.set_model(model_name);
    }

    // Apply user-provided values (only when explicitly set).
    args.set_input(train_args.input);
    args.set_output(train_args.output.clone());

    if let Some(v) = train_args.lr {
        args.set_lr(v);
    }
    if let Some(v) = train_args.lr_update_rate {
        args.set_lr_update_rate(v);
    }
    if let Some(v) = train_args.dim {
        args.set_dim(v);
    }
    if let Some(v) = train_args.ws {
        args.set_ws(v);
    }
    if let Some(v) = train_args.epoch {
        args.set_epoch(v);
    }
    if let Some(v) = train_args.min_count {
        args.set_min_count(v);
    }
    if let Some(v) = train_args.min_count_label {
        args.set_min_count_label(v);
    }
    if let Some(v) = train_args.neg {
        args.set_neg(v);
    }
    if let Some(v) = train_args.word_ngrams {
        args.set_word_ngrams(v);
    }
    if let Some(ref loss_str) = train_args.loss {
        match parse_loss(loss_str) {
            Some(loss) => args.set_loss(loss),
            None => {
                eprintln!("Error: unknown loss function '{}'. Valid values: ns, hs, softmax, ova", loss_str);
                process::exit(1);
            }
        }
    }
    if let Some(v) = train_args.bucket {
        args.set_bucket(v);
    }
    if let Some(v) = train_args.minn {
        args.set_minn(v);
    }
    if let Some(v) = train_args.maxn {
        args.set_maxn(v);
    }
    if let Some(v) = train_args.thread {
        args.set_thread(v);
    }
    if let Some(v) = train_args.t {
        args.set_t(v);
    }
    if let Some(v) = train_args.label {
        args.set_label(v);
    }
    if let Some(v) = train_args.verbose {
        args.set_verbose(v);
    }
    if let Some(v) = train_args.pretrained_vectors {
        args.set_pretrained_vectors(v);
    }
    if train_args.save_output {
        args.set_save_output(true);
    }
    if let Some(v) = train_args.seed {
        args.set_seed(v);
    }

    args
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn run_train(train_args: TrainArgs, model_name: ModelName) {
    let output_base = train_args.output.clone();
    let args = build_ft_args(train_args, model_name);

    match FastText::train(args) {
        Ok(model) => {
            // C++ fastText appends ".bin" to the output path automatically.
            let model_path = format!("{}.bin", output_base);
            if let Err(e) = model.save_model(&model_path) {
                eprintln!("Error saving model to '{}': {}", model_path, e);
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Error training model: {}", e);
            process::exit(1);
        }
    }
}

fn run_predict(predict_args: PredictArgs, with_prob: bool) {
    let model = load_model_or_exit(&predict_args.model);
    let k = predict_args.k;
    let threshold = predict_args.threshold;

    // Buffered stdout for performance.
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let process_line = |line: &str, out: &mut dyn Write| {
        let predictions = model.predict(line, k, threshold);
        for pred in &predictions {
            if with_prob {
                writeln!(out, "{} {:.6}", pred.label, pred.prob)
                    .unwrap_or_else(|_| process::exit(1));
            } else {
                writeln!(out, "{}", pred.label)
                    .unwrap_or_else(|_| process::exit(1));
            }
        }
    };

    if predict_args.input == "-" {
        // Read from stdin.
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = line.unwrap_or_else(|e| {
                eprintln!("Error reading stdin: {}", e);
                process::exit(1);
            });
            process_line(&line, &mut out);
        }
    } else {
        // Read from file.
        let file = std::fs::File::open(&predict_args.input).unwrap_or_else(|e| {
            eprintln!("Error opening input file '{}': {}", predict_args.input, e);
            process::exit(1);
        });
        for line in BufReader::new(file).lines() {
            let line = line.unwrap_or_else(|e| {
                eprintln!("Error reading input file: {}", e);
                process::exit(1);
            });
            process_line(&line, &mut out);
        }
    }

    out.flush().unwrap_or_else(|_| process::exit(1));
}

fn run_test(test_args: TestEvalArgs, per_label: bool) {
    let model = load_model_or_exit(&test_args.model);
    let k = test_args.k;
    let threshold = test_args.threshold;

    let file = match std::fs::File::open(&test_args.test_file) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening test file '{}': {}", test_args.test_file, e);
            process::exit(1);
        }
    };

    let mut reader = BufReader::new(file);
    let meter: Meter = match model.test_model(&mut reader, k, threshold) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error evaluating model: {}", e);
            process::exit(1);
        }
    };

    if per_label {
        // Print per-label metrics: label, precision, recall, F1
        let nlabels = model.dict().nlabels();
        for lid in 0..nlabels {
            if let Ok(label_str) = model.dict().get_label(lid) {
                let p = meter.precision_for_label(lid);
                let r = meter.recall_for_label(lid);
                let f = meter.f1_for_label(lid);
                println!("{}\t{:.3}\t{:.3}\t{:.3}", label_str, p, r, f);
            }
        }
    } else {
        // Print aggregate metrics: N, P@k, R@k
        meter.write_general_metrics(k as i32);
    }
}
