// fastText CLI — clap derive-based command-line interface
//
// Subcommands:
//   supervised            – train a supervised text classifier
//   skipgram              – train a skip-gram word vector model
//   cbow                  – train a CBOW word vector model
//   predict               – predict top-k labels for each line of stdin / file
//   predict-prob          – predict top-k labels + probabilities for each line
//   test                  – evaluate model and print N, P@k, R@k
//   test-label            – evaluate model and print per-label P, R, F1
//   quantize              – quantize a model to reduce memory usage
//   print-word-vectors    – print word vectors for words read from stdin
//   print-sentence-vectors – print sentence vectors for sentences read from stdin
//   print-ngrams          – print n-gram vectors for a given word
//   nn                    – query for nearest neighbors
//   analogies             – query for analogies
//   dump                  – dump args/dict/input/output in text format

use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process;

use clap::{Args, Parser, Subcommand};

use fasttext::args::{Args as FTArgs, LossName, ModelName};
use fasttext::matrix::Matrix;
use fasttext::meter::Meter;
use fasttext::FastText;

// CLI structure

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
    /// Quantize a model to reduce memory usage
    Quantize(QuantizeArgs),
    /// Print word vectors for words read from stdin
    #[command(name = "print-word-vectors")]
    PrintWordVectors(ModelPathArgs),
    /// Print sentence vectors for sentences read from stdin
    #[command(name = "print-sentence-vectors")]
    PrintSentenceVectors(ModelPathArgs),
    /// Print character n-gram vectors for a word
    #[command(name = "print-ngrams")]
    PrintNgrams(PrintNgramsArgs),
    /// Query for k nearest neighbors (reads query words from stdin)
    Nn(NnArgs),
    /// Query for analogies: reads 3 words per line, returns nearest to A-B+C
    Analogies(AnalogiesArgs),
    /// Dump model information in text format
    Dump(DumpArgs),
}

// Argument structs

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

/// Arguments for quantize.
#[derive(Args, Debug)]
struct QuantizeArgs {
    /// Path to the model base (without .bin/.ftz extension).
    /// Loads <output>.bin, saves <output>.ftz.
    #[arg(long)]
    output: String,

    /// Training data file path (required for --retrain)
    #[arg(long, default_value = "")]
    input: String,

    /// Vocabulary size cutoff (0 = no cutoff)
    #[arg(long, default_value_t = 0)]
    cutoff: usize,

    /// Retrain model after quantization (requires --input)
    #[arg(long, default_value_t = false)]
    retrain: bool,

    /// Quantize norms of word vectors
    #[arg(long, default_value_t = false)]
    qnorm: bool,

    /// Quantize output matrix as well
    #[arg(long, default_value_t = false)]
    qout: bool,

    /// Size of each sub-vector for product quantization
    #[arg(long, default_value_t = 2)]
    dsub: usize,
}

/// Arguments for print-word-vectors and print-sentence-vectors.
#[derive(Args, Debug)]
struct ModelPathArgs {
    /// Path to the trained model (.bin or .ftz file)
    model: String,
}

/// Arguments for print-ngrams.
#[derive(Args, Debug)]
struct PrintNgramsArgs {
    /// Path to the trained model (.bin or .ftz file)
    model: String,

    /// Word to print n-grams for
    word: String,
}

/// Arguments for nn.
#[derive(Args, Debug)]
struct NnArgs {
    /// Path to the trained model (.bin or .ftz file)
    model: String,

    /// Number of nearest neighbors to return
    #[arg(default_value_t = 10)]
    k: usize,
}

/// Arguments for analogies.
#[derive(Args, Debug)]
struct AnalogiesArgs {
    /// Path to the trained model (.bin or .ftz file)
    model: String,

    /// Number of analogy results to return
    #[arg(default_value_t = 10)]
    k: usize,
}

/// Arguments for dump.
#[derive(Args, Debug)]
struct DumpArgs {
    /// Path to the trained model (.bin or .ftz file)
    model: String,

    /// What to dump: args, dict, input, or output
    option: String,
}

// Entry point

/// Preprocess command-line arguments to support C++ fastText-style single-dash
/// long flags (e.g. `-epoch 5`) in addition to the standard double-dash
/// (`--epoch 5`).
///
/// Maps known single-dash C++ flags to their clap double-dash equivalents so
/// that clap parsing works unchanged.
fn normalize_args(raw: impl Iterator<Item = String>) -> Vec<String> {
    // Mapping from single-dash C++ flag name → double-dash clap flag name.
    // C++ uses camelCase; clap uses kebab-case.
    const FLAG_MAP: &[(&str, &str)] = &[
        ("-epoch",           "--epoch"),
        ("-lr",              "--lr"),
        ("-lrUpdateRate",    "--lr-update-rate"),
        ("-dim",             "--dim"),
        ("-ws",              "--ws"),
        ("-minCount",        "--min-count"),
        ("-minCountLabel",   "--min-count-label"),
        ("-neg",             "--neg"),
        ("-wordNgrams",      "--word-ngrams"),
        ("-loss",            "--loss"),
        ("-bucket",          "--bucket"),
        ("-minn",            "--minn"),
        ("-maxn",            "--maxn"),
        ("-thread",          "--thread"),
        ("-t",               "--t"),
        ("-label",           "--label"),
        ("-verbose",         "--verbose"),
        ("-seed",            "--seed"),
        ("-input",           "--input"),
        ("-output",          "--output"),
        ("-pretrainedVectors", "--pretrained-vectors"),
        ("-saveOutput",      "--save-output"),
        ("-cutoff",          "--cutoff"),
        ("-retrain",         "--retrain"),
        ("-qnorm",           "--qnorm"),
        ("-qout",            "--qout"),
        ("-dsub",            "--dsub"),
        ("-autotuneValidationFile", "--autotune-validation-file"),
        ("-autotuneDuration", "--autotune-duration"),
        ("-autotuneModelSize", "--autotune-model-size"),
        ("-autotuneMetric",  "--autotune-metric"),
    ];

    raw.map(|arg| {
        for (single, double) in FLAG_MAP {
            if arg == *single {
                return double.to_string();
            }
        }
        arg
    })
    .collect()
}

fn main() {
    let raw_args = std::env::args();
    let normalized = normalize_args(raw_args);
    let cli = Cli::parse_from(normalized);

    match cli.command {
        Commands::Supervised(args) => run_train(args, ModelName::SUP),
        Commands::Skipgram(args) => run_train(args, ModelName::SG),
        Commands::Cbow(args) => run_train(args, ModelName::CBOW),
        Commands::Predict(args) => run_predict(args, false),
        Commands::PredictProb(args) => run_predict(args, true),
        Commands::Test(args) => run_test(args, false),
        Commands::TestLabel(args) => run_test(args, true),
        Commands::Quantize(args) => run_quantize(args),
        Commands::PrintWordVectors(args) => run_print_word_vectors(args),
        Commands::PrintSentenceVectors(args) => run_print_sentence_vectors(args),
        Commands::PrintNgrams(args) => run_print_ngrams(args),
        Commands::Nn(args) => run_nn(args),
        Commands::Analogies(args) => run_analogies(args),
        Commands::Dump(args) => run_dump(args),
    }
}

// Helpers

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

/// Format an f64 like C++ `std::cout` with `std::setprecision(n)` (default notation).
/// This matches C++ `printf("%.*g", n, val)` behavior.
fn cpp_default_format(val: f64, sig_digits: usize) -> String {
    let sci = format!("{:.prec$e}", val, prec = sig_digits.saturating_sub(1));
    let e_pos = sci.find('e').unwrap();
    let exp: i32 = sci[e_pos + 1..].parse().unwrap();

    // C++ default notation: use fixed if -4 <= exp < sig_digits, else scientific.
    if exp >= -4 && exp < sig_digits as i32 {
        let decimal_places = if exp >= 0 {
            sig_digits.saturating_sub(1 + exp as usize)
        } else {
            sig_digits + (-exp as usize) - 1
        };
        let fixed = format!("{:.prec$}", val, prec = decimal_places);
        if fixed.contains('.') {
            fixed.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            fixed
        }
    } else {
        // C++ uses zero-padded 2-digit exponents and strips trailing zeros from mantissa.
        let mantissa = sci[..e_pos].trim_end_matches('0').trim_end_matches('.');
        let exp_sign = if exp < 0 { "-" } else { "+" };
        let exp_abs = exp.unsigned_abs();
        if exp_abs < 10 {
            format!("{}e{}{:02}", mantissa, exp_sign, exp_abs)
        } else {
            format!("{}e{}{}", mantissa, exp_sign, exp_abs)
        }
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
        args.model = model_name;
    }

    // Apply user-provided values (only when explicitly set).
    args.input = train_args.input;
    args.output = train_args.output.clone();

    if let Some(v) = train_args.lr {
        args.lr = v;
    }
    if let Some(v) = train_args.lr_update_rate {
        args.lr_update_rate = v;
    }
    if let Some(v) = train_args.dim {
        args.dim = v;
    }
    if let Some(v) = train_args.ws {
        args.ws = v;
    }
    if let Some(v) = train_args.epoch {
        args.epoch = v;
    }
    if let Some(v) = train_args.min_count {
        args.min_count = v;
    }
    if let Some(v) = train_args.min_count_label {
        args.min_count_label = v;
    }
    if let Some(v) = train_args.neg {
        args.neg = v;
    }
    if let Some(v) = train_args.word_ngrams {
        args.word_ngrams = v;
    }
    if let Some(ref loss_str) = train_args.loss {
        match parse_loss(loss_str) {
            Some(loss) => args.loss = loss,
            None => {
                eprintln!("Error: unknown loss function '{}'. Valid values: ns, hs, softmax, ova", loss_str);
                process::exit(1);
            }
        }
    }
    if let Some(v) = train_args.bucket {
        args.bucket = v;
    }
    if let Some(v) = train_args.minn {
        args.minn = v;
    }
    if let Some(v) = train_args.maxn {
        args.maxn = v;
    }
    if let Some(v) = train_args.thread {
        args.thread = v;
    }
    if let Some(v) = train_args.t {
        args.t = v;
    }
    if let Some(v) = train_args.label {
        args.label = v;
    }
    if let Some(v) = train_args.verbose {
        args.verbose = v;
    }
    if let Some(v) = train_args.pretrained_vectors {
        args.pretrained_vectors = v;
    }
    if train_args.save_output {
        args.save_output = true;
    }
    if let Some(v) = train_args.seed {
        args.seed = v;
    }

    args
}

// Subcommand implementations

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
        // C++ printPredictions: all on one line, space-separated.
        let mut first = true;
        for pred in &predictions {
            if !first {
                write!(out, " ").unwrap_or_else(|_| process::exit(1));
            }
            first = false;
            write!(out, "{}", pred.label).unwrap_or_else(|_| process::exit(1));
            if with_prob {
                write!(out, " {}", cpp_default_format(pred.prob as f64, 6))
                    .unwrap_or_else(|_| process::exit(1));
            }
        }
        writeln!(out).unwrap_or_else(|_| process::exit(1));
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
        let nlabels = model.dict().nlabels();
        let fmt_metric = |name: &str, val: f64| -> String {
            if val.is_finite() {
                format!("{} : {:.6}", name, val)
            } else {
                format!("{} : --------", name)
            }
        };
        for lid in 0..nlabels {
            if let Ok(label_str) = model.dict().get_label(lid) {
                let f = meter.f1_for_label(lid);
                let p = meter.precision_for_label(lid);
                let r = meter.recall_for_label(lid);
                println!(
                    "{}  {}  {}   {}",
                    fmt_metric("F1-Score", f),
                    fmt_metric("Precision", p),
                    fmt_metric("Recall", r),
                    label_str
                );
            }
        }
    }
    if per_label {
        // C++ test-label sets std::fixed before writeGeneralMetrics, so
        // setprecision(3) becomes 3 fixed decimal places, not 3 sig digits.
        println!("N\t{}", meter.n_examples());
        println!("P@{}\t{:.3}", k, meter.precision());
        println!("R@{}\t{:.3}", k, meter.recall());
    } else {
        meter.write_general_metrics(k as i32);
    }
}

// Quantize

fn run_quantize(qargs: QuantizeArgs) {
    let model_bin = format!("{}.bin", qargs.output);
    let model_ftz = format!("{}.ftz", qargs.output);

    if !std::path::Path::new(&model_bin).exists() {
        eprintln!("Error: model file '{}' does not exist", model_bin);
        process::exit(1);
    }

    let mut model = match FastText::load_model(&model_bin) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model '{}': {}", model_bin, e);
            process::exit(1);
        }
    };

    // Build quantize args from the CLI options.
    let mut ft_qargs = model.args().clone();
    ft_qargs.cutoff = qargs.cutoff;
    ft_qargs.retrain = qargs.retrain;
    ft_qargs.qnorm = qargs.qnorm;
    ft_qargs.qout = qargs.qout;
    ft_qargs.dsub = qargs.dsub;
    if !qargs.input.is_empty() {
        ft_qargs.input = qargs.input;
    }

    if let Err(e) = model.quantize(&ft_qargs) {
        eprintln!("Error quantizing model: {}", e);
        process::exit(1);
    }

    if let Err(e) = model.save_model(&model_ftz) {
        eprintln!("Error saving quantized model to '{}': {}", model_ftz, e);
        process::exit(1);
    }
}

// Print word vectors

fn run_print_word_vectors(args: ModelPathArgs) {
    let model = load_model_or_exit(&args.model);

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap_or_else(|e| {
            eprintln!("Error reading stdin: {}", e);
            process::exit(1);
        });
        // Read one word per line (whitespace-split, use first token)
        for word in line.split_whitespace() {
            let vec = model.get_word_vector(word);
            write!(out, "{} ", word).unwrap_or_else(|_| process::exit(1));
            for &v in &vec {
                write!(out, "{} ", cpp_default_format(v as f64, 5))
                    .unwrap_or_else(|_| process::exit(1));
            }
            writeln!(out).unwrap_or_else(|_| process::exit(1));
        }
    }
    out.flush().unwrap_or_else(|_| process::exit(1));
}

// Print sentence vectors

fn run_print_sentence_vectors(args: ModelPathArgs) {
    let model = load_model_or_exit(&args.model);

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap_or_else(|e| {
            eprintln!("Error reading stdin: {}", e);
            process::exit(1);
        });
        let vec = model.get_sentence_vector(&line);
        for &v in &vec {
            write!(out, "{} ", cpp_default_format(v as f64, 5))
                .unwrap_or_else(|_| process::exit(1));
        }
        writeln!(out).unwrap_or_else(|_| process::exit(1));
    }
    out.flush().unwrap_or_else(|_| process::exit(1));
}

// Print ngrams

fn run_print_ngrams(args: PrintNgramsArgs) {
    let model = load_model_or_exit(&args.model);

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let ngram_vecs = model.get_ngram_vectors(&args.word);
    for (ngram_str, vec) in &ngram_vecs {
        write!(out, "{} ", ngram_str).unwrap_or_else(|_| process::exit(1));
        for &v in vec {
            write!(out, "{} ", cpp_default_format(v as f64, 5))
                .unwrap_or_else(|_| process::exit(1));
        }
        writeln!(out).unwrap_or_else(|_| process::exit(1));
    }
    out.flush().unwrap_or_else(|_| process::exit(1));
}

// Nearest neighbors

fn run_nn(args: NnArgs) {
    let model = load_model_or_exit(&args.model);
    let k = args.k;

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    write!(out, "Query word? ").unwrap_or_else(|_| process::exit(1));
    out.flush().unwrap_or_else(|_| process::exit(1));

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap_or_else(|e| {
            eprintln!("Error reading stdin: {}", e);
            process::exit(1);
        });
        let query = line.trim();
        if query.is_empty() {
            continue;
        }
        let neighbors = model.get_nn(query, k);
        for (similarity, word) in &neighbors {
            writeln!(out, "{} {}", word, cpp_default_format(*similarity as f64, 6))
                .unwrap_or_else(|_| process::exit(1));
        }
        write!(out, "Query word? ").unwrap_or_else(|_| process::exit(1));
        out.flush().unwrap_or_else(|_| process::exit(1));
    }
}

// Analogies

fn run_analogies(args: AnalogiesArgs) {
    let k = args.k;

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    writeln!(out, "Loading model {}", args.model).unwrap_or_else(|_| process::exit(1));
    out.flush().unwrap_or_else(|_| process::exit(1));

    let model = load_model_or_exit(&args.model);

    write!(out, "Query triplet (A - B + C)? ").unwrap_or_else(|_| process::exit(1));
    out.flush().unwrap_or_else(|_| process::exit(1));

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap_or_else(|e| {
            eprintln!("Error reading stdin: {}", e);
            process::exit(1);
        });
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.len() < 3 {
            continue;
        }
        let (word_a, word_b, word_c) = (words[0], words[1], words[2]);
        let results = model.get_analogies(word_a, word_b, word_c, k);
        for (similarity, word) in &results {
            writeln!(out, "{} {}", word, cpp_default_format(*similarity as f64, 6))
                .unwrap_or_else(|_| process::exit(1));
        }
        write!(out, "Query triplet (A - B + C)? ").unwrap_or_else(|_| process::exit(1));
        out.flush().unwrap_or_else(|_| process::exit(1));
    }
}

// Dump

fn run_dump(args: DumpArgs) {
    let model = load_model_or_exit(&args.model);

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    match args.option.as_str() {
        "args" => {
            let a = model.args();
            writeln!(out, "dim {}", a.dim).unwrap_or_else(|_| process::exit(1));
            writeln!(out, "ws {}", a.ws).unwrap_or_else(|_| process::exit(1));
            writeln!(out, "epoch {}", a.epoch).unwrap_or_else(|_| process::exit(1));
            writeln!(out, "minCount {}", a.min_count).unwrap_or_else(|_| process::exit(1));
            writeln!(out, "neg {}", a.neg).unwrap_or_else(|_| process::exit(1));
            writeln!(out, "wordNgrams {}", a.word_ngrams)
                .unwrap_or_else(|_| process::exit(1));
            writeln!(out, "loss {}", a.loss_to_string()).unwrap_or_else(|_| process::exit(1));
            writeln!(out, "model {}", a.model_to_string()).unwrap_or_else(|_| process::exit(1));
            writeln!(out, "bucket {}", a.bucket).unwrap_or_else(|_| process::exit(1));
            writeln!(out, "minn {}", a.minn).unwrap_or_else(|_| process::exit(1));
            writeln!(out, "maxn {}", a.maxn).unwrap_or_else(|_| process::exit(1));
            writeln!(out, "lrUpdateRate {}", a.lr_update_rate)
                .unwrap_or_else(|_| process::exit(1));
            writeln!(out, "t {}", a.t).unwrap_or_else(|_| process::exit(1));
        }
        "dict" => {
            let dict = model.dict();
            let words = dict.words();
            writeln!(out, "{}", words.len()).unwrap_or_else(|_| process::exit(1));
            for entry in words {
                let entry_type = match entry.entry_type {
                    fasttext::dictionary::EntryType::Word => "word",
                    fasttext::dictionary::EntryType::Label => "label",
                };
                writeln!(out, "{} {} {}", entry.word, entry.count, entry_type)
                    .unwrap_or_else(|_| process::exit(1));
            }
        }
        "input" => {
            if model.is_quant() {
                eprintln!("Not supported for quantized models.");
                process::exit(1);
            }
            let m = model.input_matrix();
            writeln!(out, "{} {}", m.rows(), m.cols())
                .unwrap_or_else(|_| process::exit(1));
            for i in 0..m.rows() {
                let row = m.row(i);
                let mut first = true;
                for &v in row {
                    if !first {
                        write!(out, " ").unwrap_or_else(|_| process::exit(1));
                    }
                    write!(out, "{}", cpp_default_format(v as f64, 6)).unwrap_or_else(|_| process::exit(1));
                    first = false;
                }
                writeln!(out).unwrap_or_else(|_| process::exit(1));
            }
        }
        "output" => {
            if model.is_quant() {
                eprintln!("Not supported for quantized models.");
                process::exit(1);
            }
            let m = model.output_matrix();
            writeln!(out, "{} {}", m.rows(), m.cols())
                .unwrap_or_else(|_| process::exit(1));
            for i in 0..m.rows() {
                let row = m.row(i);
                let mut first = true;
                for &v in row {
                    if !first {
                        write!(out, " ").unwrap_or_else(|_| process::exit(1));
                    }
                    write!(out, "{}", cpp_default_format(v as f64, 6)).unwrap_or_else(|_| process::exit(1));
                    first = false;
                }
                writeln!(out).unwrap_or_else(|_| process::exit(1));
            }
        }
        other => {
            eprintln!(
                "Error: unknown dump option '{}'. Valid options: args, dict, input, output",
                other
            );
            process::exit(1);
        }
    }

    out.flush().unwrap_or_else(|_| process::exit(1));
}
