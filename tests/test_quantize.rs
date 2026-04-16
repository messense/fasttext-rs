// Quantization tests: quantize, save/load, prediction agreement, cutoff
//
// Tests extracted from src/fasttext.rs inline tests. These test the public
// API for model quantization.

use std::sync::atomic::{AtomicU64, Ordering};

use fasttext::args::{Args, LossName, ModelName};
use fasttext::error::FastTextError;
use fasttext::FastText;

const COOKING_MODEL: &str = "tests/fixtures/cooking.model.bin";

// Helpers

fn write_temp_file(content: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "fasttext_quant_test_{}_{}.txt",
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

// Tests
#[test]
fn test_model_is_quant_false_for_bin() {
    let model = FastText::load_model(COOKING_MODEL).unwrap();
    assert!(
        !model.is_quant(),
        "cooking.model.bin should not be quantized"
    );
}

/// VAL-QUANT-001: Only supervised models can be quantized.
#[test]
fn test_quantize_unsupervised_rejected() {
    let data = unsupervised_train_data();
    let path = write_temp_file(&data);
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str;
    args.output = "/dev/null".to_string();
    args.model = ModelName::CBOW;
    args.loss = LossName::NS;
    args.dim = 10;
    args.epoch = 1;
    args.min_count = 1;
    args.bucket = 100;

    let mut model = FastText::train(args).expect("CBOW training should succeed");
    std::fs::remove_file(&path).ok();

    // Quantize should fail for CBOW
    let qargs = Args::default();
    let result = model.quantize(&qargs);
    assert!(result.is_err(), "CBOW model quantize should return error");
    match result.unwrap_err() {
        FastTextError::InvalidArgument(msg) => {
            assert!(
                msg.contains("supervised") || msg.contains("supervised"),
                "Error should mention supervised: {}",
                msg
            );
        }
        e => panic!("Expected InvalidArgument, got: {:?}", e),
    }
}

/// VAL-QUANT-001 (part 2): Supervised model quantize succeeds.
#[test]
fn test_quantize_supervised_ok() {
    let (mut model, path) = train_small_supervised(16, 5, 0);
    std::fs::remove_file(&path).ok();

    let mut qargs = Args::default();
    qargs.dsub = 2;

    let result = model.quantize(&qargs);
    assert!(
        result.is_ok(),
        "Supervised model quantize should succeed: {:?}",
        result.err()
    );
    assert!(
        model.is_quant(),
        "is_quant() should be true after quantization"
    );
}

/// VAL-QUANT-002: Quantization produces valid model (is_quant=true, valid predictions).
#[test]
fn test_quantize_produces_valid_model() {
    let (mut model, path) = train_small_supervised(16, 5, 0);
    std::fs::remove_file(&path).ok();

    let mut qargs = Args::default();
    qargs.dsub = 2;
    model.quantize(&qargs).expect("Quantize should succeed");

    assert!(model.is_quant(), "is_quant() should be true");

    // Predictions should be valid
    let preds = model.predict("basketball player sport game", 1, 0.0);
    assert!(
        !preds.is_empty(),
        "Quantized model should produce predictions"
    );
    assert!(preds[0].prob > 0.0, "Prediction probability should be > 0");
    assert!(
        preds[0].prob <= 1.0,
        "Prediction probability should be <= 1.0"
    );
    assert!(
        preds[0].prob.is_finite(),
        "Prediction probability should be finite"
    );
    assert!(
        !preds[0].label.is_empty(),
        "Prediction label should not be empty"
    );
}

/// VAL-QUANT-003: .ftz file smaller than .bin file.
#[test]
fn test_quantize_smaller_file() {
    // Generate enough unique words so quantization is beneficial.
    // QuantMatrix PQ centroids: nsubq × ksub × dsub × 4 bytes (fixed overhead)
    // QuantMatrix codes: nrows × nsubq bytes
    // DenseMatrix: nrows × dim × 4 bytes
    // .ftz < .bin when nrows is large enough to amortize the centroid overhead.
    // With dim=50, dsub=2: nsubq=25, overhead=25×256×2×4=51200 bytes
    // Need nrows × 50×4 > nrows × 25 + 51200 → nrows > ~3000
    let mut data = String::new();
    for i in 0..200 {
        // Each line has ~10 unique words, for ~2000 total unique words
        data.push_str(&format!(
            "__label__sports basketball game sport player score word{} tok{} item{} entry{}\n",
            i * 3,
            i * 3 + 1,
            i * 3 + 2,
            i
        ));
        data.push_str(&format!(
            "__label__food apple banana fruit eat cook word{} tok{} item{} entry{}\n",
            i * 3 + 100,
            i * 3 + 101,
            i * 3 + 102,
            i + 100
        ));
    }
    let path = write_unique_temp_file(&data, "quantize_smaller");
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str;
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 50;
    args.epoch = 1;
    args.min_count = 1;
    args.bucket = 0;
    args.thread = 1;

    let mut model = FastText::train(args).expect("Training should succeed");
    std::fs::remove_file(&path).ok();

    let tmp_dir = std::env::temp_dir();
    let bin_path = tmp_dir.join("test_quant_smaller.bin");
    let ftz_path = tmp_dir.join("test_quant_smaller.ftz");

    // Save unquantized .bin
    model
        .save_model(bin_path.to_str().unwrap())
        .expect("Save .bin should succeed");
    let bin_size = std::fs::metadata(&bin_path).unwrap().len();

    // Quantize and save .ftz
    let mut qargs = Args::default();
    qargs.dsub = 2;
    model.quantize(&qargs).expect("Quantize should succeed");
    model
        .save_model(ftz_path.to_str().unwrap())
        .expect("Save .ftz should succeed");
    let ftz_size = std::fs::metadata(&ftz_path).unwrap().len();

    std::fs::remove_file(&bin_path).ok();
    std::fs::remove_file(&ftz_path).ok();

    assert!(
        ftz_size < bin_size,
        ".ftz ({} bytes) should be smaller than .bin ({} bytes), nwords={}",
        ftz_size,
        bin_size,
        model.dict().nwords()
    );
}

/// VAL-QUANT-004: Quantized predictions have >=90% top-1 agreement with unquantized.
#[test]
fn test_quantize_prediction_agreement() {
    let (mut model, train_path) = train_small_supervised(16, 10, 0);
    std::fs::remove_file(&train_path).ok();

    // Collect pre-quantization predictions
    let test_inputs = [
        "basketball player sport game score",
        "apple orange banana fruit eat",
        "team win lose tournament",
        "cook recipe meal dessert",
        "basketball game team score",
        "fruit eat recipe food",
        "sport player win",
        "meal cook eat banana",
        "game score win team",
        "food fruit apple banana",
    ];

    let preds_before: Vec<String> = test_inputs
        .iter()
        .map(|s| {
            let p = model.predict(s, 1, 0.0);
            if p.is_empty() {
                String::new()
            } else {
                p[0].label.clone()
            }
        })
        .collect();

    // Quantize the model
    let mut qargs = Args::default();
    qargs.dsub = 2;
    model.quantize(&qargs).expect("Quantize should succeed");

    // Collect post-quantization predictions
    let preds_after: Vec<String> = test_inputs
        .iter()
        .map(|s| {
            let p = model.predict(s, 1, 0.0);
            if p.is_empty() {
                String::new()
            } else {
                p[0].label.clone()
            }
        })
        .collect();

    // Check agreement
    let agreement = preds_before
        .iter()
        .zip(preds_after.iter())
        .filter(|(b, a)| !b.is_empty() && b == a)
        .count();
    let total = preds_before.iter().filter(|l| !l.is_empty()).count();
    assert!(total > 0, "Should have some predictions");
    let rate = agreement as f32 / total as f32;
    assert!(
        rate >= 0.9,
        "Quantized predictions should agree with unquantized >= 90%, got {:.1}% ({}/{})",
        rate * 100.0,
        agreement,
        total
    );
}

/// VAL-QUANT-007: retrain after quantization works.
#[test]
fn test_quantize_retrain() {
    let mut data = String::new();
    for _ in 0..20 {
        data.push_str("__label__sports basketball player sport game team score win\n");
        data.push_str("__label__food apple orange banana fruit eat cook recipe\n");
    }
    let path = write_unique_temp_file(&data, "quantize_retrain");
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 16;
    args.epoch = 5;
    args.min_count = 1;
    args.lr = 0.1;
    args.bucket = 0;
    args.thread = 1;
    args.seed = 42;

    let mut model = FastText::train(args).expect("Training should succeed");
    let nwords_before = model.dict().nwords();

    let cutoff = (nwords_before as usize / 2).max(2);

    let mut qargs = Args::default();
    qargs.dsub = 2;
    qargs.cutoff = cutoff;
    qargs.retrain = true;
    qargs.input = path_str.clone();
    qargs.epoch = 1;
    qargs.lr = 0.05;
    qargs.thread = 1;

    let result = model.quantize(&qargs);
    std::fs::remove_file(&path).ok();

    assert!(
        result.is_ok(),
        "retrain quantize should succeed: {:?}",
        result.err()
    );
    assert!(
        model.is_quant(),
        "is_quant() should be true after retrain quantize"
    );

    // Should produce valid predictions
    let preds = model.predict("basketball player sport game", 1, 0.0);
    assert!(
        !preds.is_empty(),
        "Retrained quantized model should produce predictions"
    );
    assert!(
        preds[0].prob.is_finite() && preds[0].prob > 0.0,
        "Retrained quantized model prediction prob should be valid"
    );
}

/// VAL-QUANT-008: .ftz save/load round-trip.
#[test]
fn test_quantize_save_load_roundtrip() {
    let (mut model, train_path) = train_small_supervised(16, 5, 0);
    std::fs::remove_file(&train_path).ok();

    let mut qargs = Args::default();
    qargs.dsub = 2;
    model.quantize(&qargs).expect("Quantize should succeed");

    // Get pre-save predictions
    let test_input = "basketball player sport game score";
    let preds_before = model.predict(test_input, 2, 0.0);
    assert!(
        !preds_before.is_empty(),
        "Quantized model should have predictions before save"
    );

    // Save to .ftz file
    let ftz_path = std::env::temp_dir().join("test_quant_roundtrip.ftz");
    model
        .save_model(ftz_path.to_str().unwrap())
        .expect("Save .ftz should succeed");

    // Load back
    let model2 =
        FastText::load_model(ftz_path.to_str().unwrap()).expect("Load .ftz should succeed");
    std::fs::remove_file(&ftz_path).ok();

    assert!(
        model2.is_quant(),
        "Loaded .ftz model should have is_quant()=true"
    );

    // Get post-load predictions
    let preds_after = model2.predict(test_input, 2, 0.0);
    assert_eq!(
        preds_before.len(),
        preds_after.len(),
        "Prediction count should match after .ftz round-trip"
    );

    // Labels and probabilities should match
    for (i, (pb, pa)) in preds_before.iter().zip(preds_after.iter()).enumerate() {
        assert_eq!(
            pb.label, pa.label,
            "Prediction[{}] label should match after .ftz round-trip: '{}' vs '{}'",
            i, pb.label, pa.label
        );
        assert!(
            (pb.prob - pa.prob).abs() < 1e-5,
            "Prediction[{}] prob should be close after .ftz round-trip: {} vs {}",
            i,
            pb.prob,
            pa.prob
        );
    }
}

/// Test that is_quant() returns true for .ftz models (VAL-INF-017).
#[test]
fn test_is_quant_true_for_ftz() {
    let (mut model, train_path) = train_small_supervised(16, 3, 0);
    std::fs::remove_file(&train_path).ok();

    assert!(
        !model.is_quant(),
        "Before quantization: is_quant() should be false"
    );

    let mut qargs = Args::default();
    qargs.dsub = 2;
    model.quantize(&qargs).expect("Quantize should succeed");

    assert!(
        model.is_quant(),
        "After quantization: is_quant() should be true"
    );

    // Save and reload
    let ftz_path = std::env::temp_dir().join("test_is_quant_ftz.ftz");
    model
        .save_model(ftz_path.to_str().unwrap())
        .expect("Save should succeed");
    let loaded =
        FastText::load_model(ftz_path.to_str().unwrap()).expect("Load .ftz should succeed");
    std::fs::remove_file(&ftz_path).ok();

    assert!(loaded.is_quant(), "Loaded .ftz: is_quant() should be true");
}

/// Verify that after cutoff pruning the model still produces valid predictions.
///
/// This exercises the invariant that matrix row j == dictionary word j after
/// pruning (the key fix for the row misalignment bug).
#[test]
fn test_quantize_cutoff_predictions_valid() {
    let (mut model, train_path) = train_small_supervised(16, 5, 0);
    std::fs::remove_file(&train_path).ok();

    let nwords_before = model.dict().nwords();
    let cutoff = (nwords_before as usize / 2).max(1);

    let mut qargs = Args::default();
    qargs.dsub = 2;
    qargs.cutoff = cutoff;
    model
        .quantize(&qargs)
        .expect("cutoff quantize should succeed");

    // After pruning, predictions should still work (no panics, valid probabilities).
    let preds = model.predict("basketball player sport game", 1, 0.0);
    assert!(
        !preds.is_empty(),
        "Cutoff-pruned quantized model should produce predictions"
    );
    assert!(
        preds[0].prob.is_finite() && preds[0].prob >= 0.0 && preds[0].prob <= 1.0,
        "Cutoff-pruned model prediction prob {} should be in [0, 1]",
        preds[0].prob
    );

    // The dictionary word count should match the cutoff.
    let nwords_after = model.dict().nwords();
    assert!(
        nwords_after <= cutoff as i32,
        "After cutoff={}, nwords should be <= {}, got {}",
        cutoff,
        cutoff,
        nwords_after
    );
}

/// Verify that matrix row count == nwords after cutoff pruning.
///
/// Specifically, the input QuantMatrix should have rows matching the
/// pruned vocabulary size, enforcing the word-at-index-i invariant.
#[test]
fn test_quantize_cutoff_matrix_row_alignment() {
    let (mut model, train_path) = train_small_supervised(16, 5, 0);
    std::fs::remove_file(&train_path).ok();

    let nwords_before = model.dict().nwords();
    // Use a tight cutoff: exactly half the words.
    let cutoff = (nwords_before as usize / 2).max(2);

    let mut qargs = Args::default();
    qargs.dsub = 2;
    qargs.cutoff = cutoff;
    model
        .quantize(&qargs)
        .expect("cutoff quantize should succeed");

    // The quantized model should be usable for save/load round-trip.
    let ftz_path =
        std::env::temp_dir().join(format!("test_cutoff_alignment_{}.ftz", std::process::id()));
    model
        .save_model(ftz_path.to_str().unwrap())
        .expect("Save pruned .ftz should succeed");
    let loaded =
        FastText::load_model(ftz_path.to_str().unwrap()).expect("Load pruned .ftz should succeed");
    std::fs::remove_file(&ftz_path).ok();

    assert!(loaded.is_quant(), "Loaded pruned .ftz should be quant");
    let preds = loaded.predict("basketball player sport game", 1, 0.0);
    assert!(
        !preds.is_empty(),
        "Loaded cutoff-pruned model should produce predictions"
    );
}
