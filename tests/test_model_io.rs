
// Model I/O tests: load, save, round-trip, validation
//
// Tests extracted from src/fasttext.rs inline tests. These test the public
// API for model loading, saving, and round-trip correctness.

use expect_test::expect;
use fasttext::error::FastTextError;
use fasttext::matrix::Matrix;
use fasttext::FastText;

const COOKING_MODEL: &str = "tests/fixtures/cooking.model.bin";
const INVALID_MODEL: &str = "tests/fixtures/invalid.model.bin";

#[test]
fn test_invalid_model_file() {
    // Load the invalid.model.bin fixture
    let result = FastText::load_model(INVALID_MODEL);
    assert!(result.is_err(), "Invalid model should be rejected");
}

#[test]
fn test_load_cooking_model_args() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let args = model.args();
    let actual = format!(
        "\
dim {}
ws {}
epoch {}
minCount {}
neg {}
wordNgrams {}
loss {:?}
model {:?}
bucket {}
minn {}
maxn {}
lrUpdateRate {}
t {}",
        args.dim, args.ws, args.epoch, args.min_count, args.neg,
        args.word_ngrams, args.loss, args.model, args.bucket,
        args.minn, args.maxn, args.lr_update_rate, args.t
    );
    expect![[r#"
dim 10
ws 5
epoch 25
minCount 1
neg 5
wordNgrams 1
loss SOFTMAX
model SUP
bucket 0
minn 0
maxn 0
lrUpdateRate 100
t 0.0001"#]].assert_eq(&actual);
}

#[test]
fn test_load_cooking_model_first_entry() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let dict = model.dict();
    let words = dict.words();
    let actual = format!("{} {} {:?}", words[0].word, words[0].count, words[0].entry_type);
    expect![[r#"</s> 12404 Word"#]].assert_eq(&actual);
}

#[test]
fn test_load_cooking_model_first_label() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let dict = model.dict();
    let first_label = dict.get_label(0).expect("Should have at least one label");
    let words = dict.words();
    let nwords = dict.nwords();
    let first_label_entry = &words[nwords as usize];
    let actual = format!("{} {}", first_label, first_label_entry.count);
    expect![[r#"__label__baking 1156"#]].assert_eq(&actual);
}

#[test]
fn test_load_cooking_model_matrices() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let input = model.input_matrix();
    let output = model.output_matrix();
    let actual = format!(
        "input {}x{}\noutput {}x{}\nquant {}",
        input.rows(), input.cols(),
        output.rows(), output.cols(),
        model.is_quant()
    );
    expect![[r#"
input 8952x10
output 735x10
quant false"#]].assert_eq(&actual);
}

#[test]
fn test_load_cooking_model_word_lookup() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

    let dict = model.dict();

    // EOS should be at index 0
    assert_eq!(dict.get_id("</s>"), 0, "EOS should be at index 0");

    // Known words should be findable
    assert!(
        dict.get_id("baking") >= 0,
        "'baking' should be in vocabulary"
    );
    assert!(
        dict.get_id("banana") >= 0,
        "'banana' should be in vocabulary"
    );

    // Unknown words should return -1
    assert_eq!(
        dict.get_id("xyzzy_definitely_not_a_word"),
        -1,
        "Unknown word should return -1"
    );
}

#[test]
fn test_invalid_model_rejection() {
    // The invalid.model.bin fixture has a corrupt header
    let result = FastText::load_model(INVALID_MODEL);
    assert!(result.is_err(), "Invalid model should be rejected");
    match result {
        Err(FastTextError::InvalidModel(_)) | Err(FastTextError::IoError(_)) => {
            // Either is acceptable
        }
        Err(e) => panic!("Expected InvalidModel or IoError, got: {:?}", e),
        Ok(_) => panic!("Expected error for invalid model"),
    }
}

/// Verify that saving the cooking model to a file and reloading it produces
/// a model whose args, vocabulary, word vectors, and predictions are
/// **bit-for-bit identical** to the original.
#[test]
fn test_model_save_load_roundtrip() {
    // 1. Load the reference model.
    let model1 = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

    let test_input = "Which baking dish is best to bake a banana bread ?";

    // 2. Capture baseline: predictions and word vector BEFORE save.
    let preds_before = model1.predict(test_input, 5, 0.0);
    assert!(
        !preds_before.is_empty(),
        "Model should produce predictions before save"
    );

    let vec_before = model1.get_word_vector("banana");
    assert_eq!(
        vec_before.len(),
        model1.args().dim as usize,
        "Word vector should have dim={} elements",
        model1.args().dim
    );

    // 3. Save to a temp file.
    let tmp_path = std::env::temp_dir().join("fasttext_roundtrip_cooking.bin");
    let tmp_str = tmp_path.to_str().unwrap();
    model1.save_model(tmp_str).expect("Should save model");

    // 4. Reload from the temp file.
    let model2 = FastText::load_model(tmp_str).expect("Should reload model");
    // Clean up the temp file (ignore errors).
    std::fs::remove_file(tmp_str).ok();

    // 5. Verify all args match.
    assert_eq!(
        model1.args().dim,
        model2.args().dim,
        "dim should match after round-trip"
    );
    assert_eq!(
        model1.args().ws,
        model2.args().ws,
        "ws should match after round-trip"
    );
    assert_eq!(
        model1.args().epoch,
        model2.args().epoch,
        "epoch should match after round-trip"
    );
    assert_eq!(
        model1.args().min_count,
        model2.args().min_count,
        "minCount should match after round-trip"
    );
    assert_eq!(
        model1.args().neg,
        model2.args().neg,
        "neg should match after round-trip"
    );
    assert_eq!(
        model1.args().word_ngrams,
        model2.args().word_ngrams,
        "wordNgrams should match after round-trip"
    );
    assert_eq!(
        model1.args().loss,
        model2.args().loss,
        "loss should match after round-trip"
    );
    assert_eq!(
        model1.args().model,
        model2.args().model,
        "model should match after round-trip"
    );
    assert_eq!(
        model1.args().bucket,
        model2.args().bucket,
        "bucket should match after round-trip"
    );
    assert_eq!(
        model1.args().minn,
        model2.args().minn,
        "minn should match after round-trip"
    );
    assert_eq!(
        model1.args().maxn,
        model2.args().maxn,
        "maxn should match after round-trip"
    );
    assert_eq!(
        model1.args().lr_update_rate,
        model2.args().lr_update_rate,
        "lrUpdateRate should match after round-trip"
    );
    assert!(
        (model1.args().t - model2.args().t).abs() < f64::EPSILON,
        "t should match after round-trip: {} vs {}",
        model1.args().t,
        model2.args().t
    );

    // 6. Verify vocabulary and labels match (count, words, frequencies).
    assert_eq!(
        model1.dict().nwords(),
        model2.dict().nwords(),
        "nwords should match after round-trip"
    );
    assert_eq!(
        model1.dict().nlabels(),
        model2.dict().nlabels(),
        "nlabels should match after round-trip"
    );
    assert_eq!(
        model1.dict().ntokens(),
        model2.dict().ntokens(),
        "ntokens should match after round-trip"
    );
    assert_eq!(
        model1.dict().size(),
        model2.dict().size(),
        "dict size should match after round-trip"
    );

    // Verify all vocabulary entries (word string, frequency, entry type).
    let words1 = model1.dict().words();
    let words2 = model2.dict().words();
    assert_eq!(words1.len(), words2.len(), "words vec length should match");
    for (i, (w1, w2)) in words1.iter().zip(words2.iter()).enumerate() {
        assert_eq!(
            w1.word, w2.word,
            "word[{}] string should match: {} vs {}",
            i, w1.word, w2.word
        );
        assert_eq!(
            w1.count, w2.count,
            "word[{}] '{}' count should match: {} vs {}",
            i, w1.word, w1.count, w2.count
        );
        assert_eq!(
            w1.entry_type, w2.entry_type,
            "word[{}] '{}' type should match",
            i, w1.word
        );
    }

    // 7. Verify word vectors are bitwise identical.
    let vec_after = model2.get_word_vector("banana");
    assert_eq!(
        vec_before.len(),
        vec_after.len(),
        "Word vector length should match"
    );
    for (j, (v1, v2)) in vec_before.iter().zip(vec_after.iter()).enumerate() {
        assert_eq!(
            v1.to_bits(),
            v2.to_bits(),
            "Word vector element [{}] should be bitwise equal: {} vs {}",
            j,
            v1,
            v2
        );
    }

    // 8. Verify predictions are bitwise identical.
    let preds_after = model2.predict(test_input, 5, 0.0);
    assert_eq!(
        preds_before.len(),
        preds_after.len(),
        "Number of predictions should match after round-trip"
    );
    for (idx, (p1, p2)) in
        preds_before.iter().zip(preds_after.iter()).enumerate()
    {
        assert_eq!(
            p1.label, p2.label,
            "Prediction[{}] label should match: {} vs {}",
            idx, p1.label, p2.label
        );
        assert_eq!(
            p1.prob.to_bits(),
            p2.prob.to_bits(),
            "Prediction[{}] label='{}' probability should be bitwise equal: {} vs {}",
            idx,
            p1.label,
            p1.prob,
            p2.prob
        );
    }

    // Also verify the top label is baking-related (sanity check).
    assert!(
        preds_before[0].label.contains("baking") || preds_before[0].label.contains("bread"),
        "Top prediction for banana bread question should be baking or bread related, got: {}",
        preds_before[0].label
    );
}

/// Verify that save_model flushes the writer before returning, so all
/// buffered data is written to disk.
#[test]
fn test_save_model_flushes_bufwriter() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

    let tmp_path = std::env::temp_dir().join("fasttext_flush_test.bin");
    let tmp_str = tmp_path.to_str().unwrap();

    // save_model must succeed (implicit flush).
    model.save_model(tmp_str).expect("save_model should succeed");

    // Reload and verify the model is fully intact — proves the flush worked.
    let model2 = FastText::load_model(tmp_str).expect("Reloaded model should be valid");
    std::fs::remove_file(tmp_str).ok();

    // Sanity-check that predictions from reloaded model are valid.
    let preds = model2.predict("How to bake a banana bread?", 1, 0.0);
    assert!(!preds.is_empty(), "Reloaded model should produce predictions");
}
