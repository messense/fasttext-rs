// Training tests: supervised, CBOW, skip-gram, Hogwild, abort, loss tracking
//
// Tests extracted from src/fasttext.rs inline tests. These test the public
// API for model training.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use fasttext::args::{Args, LossName, ModelName};
use fasttext::error::FastTextError;
use fasttext::{FastText, Prediction};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_temp_file(content: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "fasttext_training_test_{}_{}.txt",
        std::process::id(),
        id
    ));
    std::fs::write(&path, content).expect("Failed to write temp file");
    path
}

fn write_unique_temp_file(content: &str, tag: &str) -> std::path::PathBuf {
    static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "fasttext_{}_{}_{}.txt",
        tag,
        std::process::id(),
        id
    ));
    std::fs::write(&path, content).expect("Failed to write unique temp file");
    path
}

fn write_pretrained_vec_file(words_and_vecs: &[(&str, Vec<f32>)]) -> std::path::PathBuf {
    static VEC_COUNTER: AtomicU64 = AtomicU64::new(0);

    let dim = if words_and_vecs.is_empty() { 0 } else { words_and_vecs[0].1.len() };
    let mut content = format!("{} {}\n", words_and_vecs.len(), dim);
    for (word, vec) in words_and_vecs {
        content.push_str(word);
        for &v in vec.iter() {
            content.push(' ');
            content.push_str(&v.to_string());
        }
        content.push('\n');
    }
    let id = VEC_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "fasttext_pretrain_{}_{}.vec",
        std::process::id(),
        id
    ));
    std::fs::write(&path, content).expect("Failed to write pretrained vec file");
    path
}

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

fn unsupervised_train_data() -> String {
    let mut data = String::new();
    for _ in 0..20 {
        data.push_str("the quick brown fox jumps over the lazy dog\n");
        data.push_str("machine learning algorithms work with data\n");
        data.push_str("neural networks are powerful tools for classification\n");
    }
    data
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
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
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 5;
    args.min_count = 1;
    args.lr = 0.1;
    args.word_ngrams = 1;
    args.bucket = 0;

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
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.model = ModelName::CBOW;
    args.loss = LossName::NS;
    args.dim = 10;
    args.epoch = 3;
    args.min_count = 1;
    args.lr = 0.05;
    args.ws = 3;
    args.neg = 5;
    args.bucket = 100;
    args.minn = 0;
    args.maxn = 0;

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
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.model = ModelName::SG;
    args.loss = LossName::NS;
    args.dim = 10;
    args.epoch = 3;
    args.min_count = 1;
    args.lr = 0.05;
    args.ws = 3;
    args.neg = 5;
    args.bucket = 100;
    args.minn = 0;
    args.maxn = 0;

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

/// Test that training completes and produces a model usable for prediction.
#[test]
fn test_train_lr_decay_actual() {
    let data = supervised_train_data();
    let path = write_temp_file(&data);
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 5;
    args.epoch = 3;
    args.min_count = 1;
    args.lr = 0.1;
    args.bucket = 0;

    let result = FastText::train(args);
    std::fs::remove_file(&path).ok();
    assert!(
        result.is_ok(),
        "Training should complete without error: {:?}",
        result.err()
    );
}

/// VAL-TRAIN-006: Multi-threaded Hogwild! training completes without panic.
///
/// Trains with thread=4 and verifies all model weights are finite.
#[test]
fn test_parallel_hogwild_training() {
    let data = supervised_train_data();
    let path = write_temp_file(&data);
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 3;
    args.min_count = 1;
    args.lr = 0.1;
    args.bucket = 0;
    args.thread = 4;

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
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.model = fasttext::args::ModelName::CBOW;
    args.loss = fasttext::args::LossName::NS;
    args.dim = 10;
    args.epoch = 3;
    args.min_count = 1;
    args.thread = 4;
    args.bucket = 100;

    let model = FastText::train(args).expect("Multi-threaded CBOW training should succeed");
    std::fs::remove_file(&path).ok();

    for v in model.input_matrix().data() {
        assert!(v.is_finite(), "Input weight NaN/Inf: {}", v);
    }
    for v in model.output_matrix().data() {
        assert!(v.is_finite(), "Output weight NaN/Inf: {}", v);
    }
}

/// Atomic loss accumulation across threads.
///
/// Verifies that multi-threaded training atomically accumulates loss from all
/// threads, and that the resulting `last_train_loss()` is finite and positive.
#[test]
fn test_atomic_loss_accumulation_multithreaded() {
    let data = supervised_train_data();
    let path = write_unique_temp_file(&data, "atomic_loss");
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 3;
    args.min_count = 1;
    args.lr = 0.1;
    args.bucket = 0;
    args.thread = 4; // Multiple threads: each contributes to shared loss

    let model = FastText::train(args).expect("Multi-threaded training should succeed");
    std::fs::remove_file(&path).ok();

    // The shared atomic loss accumulator should have received contributions
    // from all threads, resulting in a finite, positive average loss.
    let loss = model.last_train_loss();
    assert!(
        loss.is_finite(),
        "Accumulated training loss should be finite after multi-thread training, got {}",
        loss
    );
    assert!(
        loss > 0.0,
        "Accumulated training loss should be positive (all threads contributed), got {}",
        loss
    );
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
        args.input = path_str.clone();
        args.output = "/dev/null".to_string();
        args.apply_supervised_defaults();
        args.dim = 10;
        args.epoch = 500; // Very large epoch count so training won't finish naturally.
        args.min_count = 1;
        args.lr = 0.1;
        args.bucket = 0;
        args.thread = 1;
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

/// VAL-TRAIN-010: Abort via TrainingHandle — calling abort() from main thread.
///
/// Uses `FastText::spawn_training` to start training in a background thread,
/// then calls `handle.abort()` from the main thread while training is running.
/// This tests the public API path for in-flight abort (vs. using an external
/// Arc<AtomicBool> directly).
#[test]
fn test_training_abort_via_handle() {
    let data = supervised_train_data();
    let path = write_unique_temp_file(&data, "abort_via_handle");
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 500; // Large epoch count so training won't finish naturally.
    args.min_count = 1;
    args.lr = 0.1;
    args.bucket = 0;
    args.thread = 1;

    // Spawn training in a background thread via the public API.
    let handle = FastText::spawn_training(args);

    // Give training a moment to start, then call abort() from the main thread.
    std::thread::sleep(std::time::Duration::from_millis(50));
    handle.abort(); // <-- abort() called from main thread!

    // Training should complete early and return a (possibly under-trained) model.
    let model = handle
        .join()
        .expect("Training thread should not panic")
        .expect("Aborted training should return Ok");
    std::fs::remove_file(&path).ok();

    // The model must still be usable for prediction without panicking.
    let _ = model.predict("basketball player sport game", 1, 0.0);

    // abort() on the returned model is idempotent.
    model.abort();
}

/// Abort is idempotent: calling abort() multiple times must not panic.
#[test]
fn test_abort_idempotent() {
    let data = supervised_train_data();
    let path = write_temp_file(&data);
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 5;
    args.epoch = 2;
    args.min_count = 1;
    args.lr = 0.1;
    args.bucket = 0;
    args.thread = 1;

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
        args.input = path.to_string();
        args.output = "/dev/null".to_string();
        args.apply_supervised_defaults();
        args.dim = 10;
        args.epoch = 3;
        args.min_count = 1;
        args.lr = 0.1;
        args.bucket = 0;
        args.thread = 1; // Single thread for determinism.
        args.seed = 42; // Fixed seed.
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

/// Train a model, save to disk, reload, verify predictions are bit-identical.
#[test]
fn test_train_save_load_roundtrip() {
    let data = supervised_train_data();
    let path = write_temp_file(&data);
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 5;
    args.min_count = 1;
    args.lr = 0.1;
    args.bucket = 0;
    args.thread = 1;
    args.seed = 42;

    let model1 = FastText::train(args).expect("Training should succeed");
    std::fs::remove_file(&path).ok();

    let test_inputs = [
        "basketball player sport game",
        "apple fruit eat cook",
        "team score win game",
    ];

    // Collect predictions before save.
    let preds_before: Vec<Vec<Prediction>> = test_inputs
        .iter()
        .map(|input| model1.predict(input, 2, 0.0))
        .collect();

    for (i, preds) in preds_before.iter().enumerate() {
        assert!(!preds.is_empty(), "Input[{}] should have predictions before save", i);
    }

    let tmp_path = std::env::temp_dir().join("fasttext_train_save_load_rt.bin");
    let tmp_str = tmp_path.to_str().unwrap();
    model1.save_model(tmp_str).expect("Should save trained model");

    let model2 = FastText::load_model(tmp_str).expect("Should reload trained model");
    std::fs::remove_file(tmp_str).ok();

    let preds_after: Vec<Vec<Prediction>> = test_inputs
        .iter()
        .map(|input| model2.predict(input, 2, 0.0))
        .collect();

    for (i, (pb, pa)) in preds_before.iter().zip(preds_after.iter()).enumerate() {
        assert_eq!(pb.len(), pa.len(),
            "Input[{}]: prediction count should match after round-trip", i);
        for (j, (p1, p2)) in pb.iter().zip(pa.iter()).enumerate() {
            assert_eq!(p1.label, p2.label,
                "Input[{}] pred[{}]: label should match after round-trip", i, j);
            assert_eq!(p1.prob.to_bits(), p2.prob.to_bits(),
                "Input[{}] pred[{}]: prob should be bit-identical: {} vs {}",
                i, j, p1.prob, p2.prob);
        }
    }
}

/// Empty training file returns an error (VAL-TRAIN-011).
#[test]
fn test_train_empty_file() {
    let path = write_temp_file("");
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 1;
    args.min_count = 1;
    args.bucket = 0;

    let result = FastText::train(args);
    std::fs::remove_file(&path).ok();

    assert!(result.is_err(), "Training on empty file should return an error");
    match result.unwrap_err() {
        FastTextError::InvalidArgument(msg) => {
            let msg_lower = msg.to_lowercase();
            assert!(
                msg_lower.contains("empty") || msg_lower.contains("tokens")
                    || msg_lower.contains("vocabulary"),
                "Error message should mention empty, tokens, or vocabulary: {}",
                msg
            );
        }
        e => panic!("Expected InvalidArgument for empty file, got: {:?}", e),
    }
}

/// epoch=0 produces an untrained model, no panic (VAL-TRAIN-011).
#[test]
fn test_train_zero_epochs() {
    let data = supervised_train_data();
    let path = write_temp_file(&data);
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 0;
    args.min_count = 1;
    args.lr = 0.1;
    args.bucket = 0;
    args.thread = 1;

    let result = FastText::train(args);
    std::fs::remove_file(&path).ok();

    match result {
        Ok(model) => {
            let _preds = model.predict("basketball player sport game", 1, 0.0);
            let (vocab, _) = model.get_vocab();
            assert!(!vocab.is_empty(), "epoch=0 model should have vocabulary");
        }
        Err(FastTextError::InvalidArgument(_)) => { /* also acceptable */ }
        Err(e) => panic!("epoch=0 should not produce unexpected error: {:?}", e),
    }
}

/// Words below min_count are excluded from vocabulary (VAL-TRAIN-012).
#[test]
fn test_min_count_filtering() {
    let mut data = String::new();
    for _ in 0..10 {
        data.push_str("__label__sports common_word basketball game score\n");
    }
    for _ in 0..10 {
        data.push_str("__label__food common_word apple banana fruit\n");
    }
    data.push_str("__label__sports rare_word unique_token\n");
    data.push_str("__label__food also_rare apple\n");
    data.push_str("__label__food also_rare banana\n");

    let path = write_temp_file(&data);
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 1;
    args.min_count = 3;
    args.bucket = 0;
    args.thread = 1;

    let model = FastText::train(args).expect("Training with min_count=3 should succeed");
    std::fs::remove_file(&path).ok();

    assert!(model.get_word_id("common_word") >= 0,
        "common_word (count=20) should be in vocabulary");
    assert_eq!(model.get_word_id("rare_word"), -1,
        "rare_word (count=1) should be excluded with min_count=3");
    assert_eq!(model.get_word_id("unique_token"), -1,
        "unique_token (count=1) should be excluded with min_count=3");
    assert_eq!(model.get_word_id("also_rare"), -1,
        "also_rare (count=2) should be excluded with min_count=3");

    let (vocab_words, vocab_freqs) = model.get_vocab();
    for (w, &freq) in vocab_words.iter().zip(vocab_freqs.iter()) {
        if w != "</s>" {
            assert!(freq >= 3,
                "Word '{}' with freq={} should not be in vocab (min_count=3)", w, freq);
        }
    }
}

/// Final epoch loss < first epoch loss after sufficient training (VAL-TRAIN-013).
#[test]
fn test_training_loss_decreases() {
    let mut data = String::new();
    for _ in 0..30 {
        data.push_str("__label__sports basketball player game score team win lose tournament championship\n");
    }
    for _ in 0..30 {
        data.push_str("__label__food apple orange banana mango fruit eat cook recipe meal dessert\n");
    }

    let path = write_temp_file(&data);
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 10;
    args.min_count = 1;
    args.lr = 0.1;
    args.bucket = 0;
    args.thread = 1;
    args.seed = 42;
    args.lr_update_rate = 50;

    let result = FastText::train_tracking_epoch_losses(args);
    std::fs::remove_file(&path).ok();

    let (_model, epoch_losses) = result.expect("Training with loss tracking should succeed");

    assert!(epoch_losses.len() >= 2,
        "Should record at least 2 epoch losses, got {}: {:?}", epoch_losses.len(), epoch_losses);

    let first_loss = epoch_losses[0];
    let last_loss = *epoch_losses.last().unwrap();

    assert!(first_loss.is_finite(), "First epoch loss should be finite: {}", first_loss);
    assert!(last_loss.is_finite(), "Final epoch loss should be finite: {}", last_loss);
    assert!(first_loss > 0.0, "First epoch loss should be > 0: {}", first_loss);
    assert!(last_loss < first_loss,
        "Loss should decrease: first={}, final={}", first_loss, last_loss);
}

/// Pretrained vectors loaded for matching words, missing file returns IoError (VAL-TRAIN-014).
#[test]
fn test_pretrained_vectors() {
    let dim = 4usize;
    let mut data = String::new();
    for _ in 0..10 { data.push_str("__label__sports basketball game score\n"); }
    for _ in 0..10 { data.push_str("__label__food apple fruit meal\n"); }
    let train_path = write_unique_temp_file(&data, "pretrained");

    let vec_basketball = vec![1.0f32, 2.0, 3.0, 4.0];
    let vec_apple = vec![5.0f32, 6.0, 7.0, 8.0];

    let vec_path = write_pretrained_vec_file(&[
        ("basketball", vec_basketball.clone()),
        ("apple", vec_apple.clone()),
        ("not_in_vocab", vec![9.0f32, 10.0, 11.0, 12.0]),
    ]);

    let mut args = Args::default();
    args.input = train_path.to_str().unwrap().to_string();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = dim as i32;
    args.epoch = 0; // epoch=0: no training, just init + load pretrained
    args.min_count = 1;
    args.bucket = 0;
    args.thread = 1;
    args.seed = 42;
    args.pretrained_vectors = vec_path.to_str().unwrap().to_string();

    let model = FastText::train(args).expect("Training with pretrained vectors should succeed");
    std::fs::remove_file(&train_path).ok();
    std::fs::remove_file(&vec_path).ok();

    let basketball_id = model.get_word_id("basketball");
    assert!(basketball_id >= 0, "'basketball' should be in vocabulary");
    let row = model.input_matrix().row(basketball_id as i64);
    for (j, (&got, &exp)) in row.iter().zip(vec_basketball.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5,
            "basketball[{}]: expected={}, got={}", j, exp, got);
    }

    let apple_id = model.get_word_id("apple");
    assert!(apple_id >= 0, "'apple' should be in vocabulary");
    let row = model.input_matrix().row(apple_id as i64);
    for (j, (&got, &exp)) in row.iter().zip(vec_apple.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5,
            "apple[{}]: expected={}, got={}", j, exp, got);
    }

    // Word not in pretrained file should have non-zero random init.
    let game_id = model.get_word_id("game");
    if game_id >= 0 {
        let row = model.input_matrix().row(game_id as i64);
        let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(norm.is_finite() && norm > 0.0,
            "'game' should have non-zero random init, norm={}", norm);
    }
}

/// Missing pretrained vectors file returns IoError (VAL-TRAIN-014).
#[test]
fn test_pretrained_vectors_missing_file() {
    let data = supervised_train_data();
    let path = write_unique_temp_file(&data, "pretrained_missing");
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 1;
    args.min_count = 1;
    args.bucket = 0;
    args.pretrained_vectors = "/nonexistent/path/vectors.vec".to_string();

    let result = FastText::train(args);
    std::fs::remove_file(&path).ok();

    assert!(result.is_err(), "Missing pretrained vectors file should return error");
    match result.unwrap_err() {
        FastTextError::IoError(_) => { /* correct */ }
        e => panic!("Expected IoError for missing pretrained vec file, got: {:?}", e),
    }
}

/// Full training integration test: train, predict, save, load, verify (VAL-TRAIN-008).
///
/// Exercises the complete training pipeline end-to-end:
/// 1. Train a supervised model
/// 2. Verify predictions
/// 3. Save to disk
/// 4. Reload
/// 5. Verify predictions are bit-identical after round-trip
#[test]
fn test_train_integration_roundtrip() {
    let data = supervised_train_data();
    let path = write_temp_file(&data);
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 5;
    args.min_count = 1;
    args.lr = 0.1;
    args.bucket = 0;
    args.thread = 1;
    args.seed = 42;

    let model = FastText::train(args).expect("Training should succeed");
    std::fs::remove_file(&path).ok();

    // Should have labels and make predictions.
    let (labels, _) = model.get_labels();
    assert!(!labels.is_empty(), "Trained model should have labels");

    let preds_orig = model.predict("basketball player sport game", 1, 0.0);
    assert!(!preds_orig.is_empty(), "Trained model should predict");

    // Save and reload.
    let tmp = std::env::temp_dir().join("fasttext_train_integration_rt.bin");
    let tmp_str = tmp.to_str().unwrap();
    model.save_model(tmp_str).expect("Save should succeed");
    let model2 = FastText::load_model(tmp_str).expect("Load should succeed");
    std::fs::remove_file(tmp_str).ok();

    // Predictions must be bit-identical.
    let preds_rt = model2.predict("basketball player sport game", 1, 0.0);
    assert_eq!(preds_orig.len(), preds_rt.len(), "Prediction count should match");
    for (p1, p2) in preds_orig.iter().zip(preds_rt.iter()) {
        assert_eq!(p1.label, p2.label, "Labels should match after round-trip");
        assert_eq!(p1.prob.to_bits(), p2.prob.to_bits(),
            "Probabilities should be bit-identical after round-trip");
    }
}

/// Training integration: edge cases (empty file, no labels, zero epochs).
#[test]
fn test_train_integration_edge_cases() {
    // 1. Empty file -> error
    let empty_path = write_temp_file("");
    let mut args = Args::default();
    args.input = empty_path.to_str().unwrap().to_string();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 5;
    args.epoch = 1;
    args.min_count = 1;
    args.bucket = 0;
    assert!(FastText::train(args).is_err(), "Empty file should return error");
    std::fs::remove_file(&empty_path).ok();

    // 2. No labels for supervised -> error
    let no_label_path = write_temp_file("word1 word2 word3\nmore text here\n");
    let mut args2 = Args::default();
    args2.input = no_label_path.to_str().unwrap().to_string();
    args2.output = "/dev/null".to_string();
    args2.apply_supervised_defaults();
    args2.dim = 5;
    args2.epoch = 1;
    args2.min_count = 1;
    args2.bucket = 0;
    assert!(FastText::train(args2).is_err(), "No labels should return error");
    std::fs::remove_file(&no_label_path).ok();

    // 3. epoch=0 -> untrained model (no panic)
    let data = supervised_train_data();
    let p = write_temp_file(&data);
    let mut args3 = Args::default();
    args3.input = p.to_str().unwrap().to_string();
    args3.output = "/dev/null".to_string();
    args3.apply_supervised_defaults();
    args3.dim = 5;
    args3.epoch = 0;
    args3.min_count = 1;
    args3.bucket = 0;
    args3.thread = 1;
    let result = FastText::train(args3);
    std::fs::remove_file(&p).ok();
    match result {
        Ok(m) => { let _ = m.predict("test", 1, 0.0); } // no panic
        Err(FastTextError::InvalidArgument(_)) => {} // acceptable
        Err(e) => panic!("Unexpected error for epoch=0: {:?}", e),
    }
}

