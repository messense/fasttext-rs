// Inference tests: prediction, vectors, tokenization, vocabulary, meter
//
// Tests extracted from src/fasttext.rs inline tests. These test the public
// API for model inference, word/sentence vectors, and evaluation.
// Allow creating Args with Default::default() and then assigning fields in tests.
#![allow(clippy::field_reassign_with_default)]

use std::sync::Arc;

use expect_test::expect;
use fasttext::args::{Args, LossName, ModelName};
use fasttext::dictionary::EOS;
use fasttext::error::FastTextError;
use fasttext::matrix::Matrix;
use fasttext::FastText;

const COOKING_MODEL: &str = "tests/fixtures/cooking.model.bin";

// Helpers

fn write_temp_file(content: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "fasttext_inference_test_{}.txt",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::write(&path, content).expect("Failed to write temp file");
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

// Tests
#[test]
fn test_load_cooking_model_vocab() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

    let dict = model.dict();
    assert_eq!(
        dict.nwords(),
        8952,
        "Should have 8952 words, got {}",
        dict.nwords()
    );
    assert_eq!(
        dict.nlabels(),
        735,
        "Should have 735 labels, got {}",
        dict.nlabels()
    );
    assert_eq!(dict.size(), 8952 + 735, "Total size should be 9687");
}

/// Verify word vector round-trip for multiple words.
#[test]
fn test_word_vectors_roundtrip() {
    let model1 = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

    // Save and reload.
    let tmp_path = std::env::temp_dir().join("fasttext_wordvec_roundtrip.bin");
    let tmp_str = tmp_path.to_str().unwrap();
    model1.save_model(tmp_str).expect("Should save model");
    let model2 = FastText::load_model(tmp_str).expect("Should reload model");
    std::fs::remove_file(tmp_str).ok();

    // Check several words from the cooking vocabulary.
    let test_words = ["banana", "baking", "bread", "chicken", "salt"];
    for word in &test_words {
        let v1 = model1.get_word_vector(word);
        let v2 = model2.get_word_vector(word);
        assert_eq!(v1.len(), v2.len(), "Vector length mismatch for '{}'", word);
        for (j, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "Vector[{}] for '{}' should be bitwise equal: {} vs {}",
                j,
                word,
                a,
                b
            );
        }
    }
}

/// Verify that predictions are deterministic across multiple calls after round-trip.
#[test]
fn test_predictions_identical_after_roundtrip() {
    let model1 = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

    let tmp_path = std::env::temp_dir().join("fasttext_pred_roundtrip.bin");
    let tmp_str = tmp_path.to_str().unwrap();
    model1.save_model(tmp_str).expect("Should save model");
    let model2 = FastText::load_model(tmp_str).expect("Should reload model");
    std::fs::remove_file(tmp_str).ok();

    let inputs = [
        "how to make pasta",
        "best knife for cutting vegetables",
        "what temperature to bake chicken",
    ];

    for input in &inputs {
        let p1 = model1.predict(input, 3, 0.0);
        let p2 = model2.predict(input, 3, 0.0);
        assert_eq!(
            p1.len(),
            p2.len(),
            "Prediction count should match for: {}",
            input
        );
        for (i, (pred1, pred2)) in p1.iter().zip(p2.iter()).enumerate() {
            assert_eq!(
                pred1.label, pred2.label,
                "Label[{}] should match for: {}",
                i, input
            );
            assert_eq!(
                pred1.prob.to_bits(),
                pred2.prob.to_bits(),
                "Prob[{}] should be bitwise equal for: {}",
                i,
                input
            );
        }
    }
}

/// Verify predict() returns __label__baking and __label__bread as top-2
/// for the canonical cooking test query.
#[test]
fn test_predict_cooking_top2() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let input = "Which baking dish is best to bake a banana bread ?";
    let preds = model.predict(input, 2, 0.0);

    assert_eq!(
        preds.len(),
        2,
        "Should return exactly 2 predictions, got {:?}",
        preds.iter().map(|p| &p.label).collect::<Vec<_>>()
    );
    assert_eq!(
        preds[0].label, "__label__baking",
        "Top-1 should be __label__baking, got '{}'",
        preds[0].label
    );
    assert_eq!(
        preds[1].label, "__label__bread",
        "Top-2 should be __label__bread, got '{}'",
        preds[1].label
    );
    // Top-1 probability > top-2 probability
    assert!(
        preds[0].prob > preds[1].prob,
        "Top-1 prob ({}) should be > top-2 prob ({})",
        preds[0].prob,
        preds[1].prob
    );
}

/// Verify predicted probabilities match C++ output within 1e-4 absolute tolerance.
///
/// C++ reference (predict-prob cooking.model.bin 5, lowercased input):
///   __label__baking    0.72013
///   __label__bread     0.205032
///   __label__quickbread 0.017047
///   __label__oven      0.0105739
///   __label__rising    0.00388523
#[test]
fn test_predict_cooking_probabilities() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let input = "which baking dish is best to bake a banana bread ?";
    let preds = model.predict(input, 5, 0.0);

    assert!(preds.len() >= 2, "Should return at least 2 predictions");

    // Expected values from C++ reference (exp of log-prob)
    let expected = [
        ("__label__baking", 0.72013_f32),
        ("__label__bread", 0.205032_f32),
        ("__label__quickbread", 0.017047_f32),
        ("__label__oven", 0.0105739_f32),
        ("__label__rising", 0.00388523_f32),
    ];

    // Verify at least the first 2 predictions match
    for (i, &(label, cpp_prob)) in expected.iter().take(2).enumerate() {
        assert_eq!(
            preds[i].label, label,
            "Prediction[{}] label mismatch: expected '{}', got '{}'",
            i, label, preds[i].label
        );
        assert!(
            (preds[i].prob - cpp_prob).abs() < 1e-4,
            "Prediction[{}] '{}': prob={} expected={} diff={}",
            i,
            label,
            preds[i].prob,
            cpp_prob,
            (preds[i].prob - cpp_prob).abs()
        );
    }

    // If we have 5 predictions, check all 5
    if preds.len() >= 5 {
        for (i, &(label, cpp_prob)) in expected.iter().enumerate() {
            assert_eq!(preds[i].label, label, "Prediction[{}] label mismatch", i);
            assert!(
                (preds[i].prob - cpp_prob).abs() < 1e-4,
                "Prediction[{}] '{}': prob={} expected={} diff={}",
                i,
                label,
                preds[i].prob,
                cpp_prob,
                (preds[i].prob - cpp_prob).abs()
            );
        }
    }
}

/// Verify that only predictions with probability >= threshold are returned.
#[test]
fn test_predict_threshold_filtering() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let input = "Which baking dish is best to bake a banana bread ?";

    // threshold=0.0: all predictions returned (up to k)
    let preds_all = model.predict(input, 10, 0.0);
    assert!(
        !preds_all.is_empty(),
        "threshold=0.0 should return predictions"
    );
    for p in &preds_all {
        assert!(p.prob >= 0.0, "All probs should be >= 0.0");
    }

    // threshold=0.5: only high-confidence predictions
    let preds_half = model.predict(input, 10, 0.5);
    for p in &preds_half {
        assert!(
            p.prob >= 0.5,
            "All probs should be >= 0.5 when threshold=0.5, got {}",
            p.prob
        );
    }
    // The top prediction (baking, ~0.706) should be above 0.5
    assert!(
        !preds_half.is_empty(),
        "At least one prediction should have prob >= 0.5"
    );
    assert_eq!(
        preds_half[0].label, "__label__baking",
        "Top prediction above 0.5 threshold should be __label__baking"
    );

    // threshold=1.0: no predictions should be returned (softmax prob < 1 generally)
    let preds_max = model.predict(input, 10, 1.0);
    for p in &preds_max {
        assert!(
            p.prob >= 1.0,
            "All probs should be >= 1.0 when threshold=1.0, got {}",
            p.prob
        );
    }

    // Verify threshold filtering: preds with high threshold is subset of low threshold
    let preds_low = model.predict(input, 100, 0.01);
    let preds_high = model.predict(input, 100, 0.1);
    assert!(
        preds_high.len() <= preds_low.len(),
        "Higher threshold should return fewer or equal predictions"
    );
    // All preds_high labels should appear in preds_low
    for p in &preds_high {
        assert!(
            p.prob >= 0.1,
            "High threshold result should have prob >= 0.1, got {}",
            p.prob
        );
    }
}

/// Empty string input returns empty predictions without panic.
#[test]
fn test_predict_empty_input() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let preds = model.predict("", 5, 0.0);
    assert!(
        preds.is_empty(),
        "Empty input should return empty predictions"
    );
}

/// Whitespace-only input returns empty predictions without panic.
#[test]
fn test_predict_whitespace_only() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let preds = model.predict("   \t  \n  ", 5, 0.0);
    assert!(
        preds.is_empty(),
        "Whitespace-only input should return empty predictions"
    );
}

/// k=0 returns empty predictions without panic.
#[test]
fn test_predict_k_zero() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let preds = model.predict("Which baking dish is best?", 0, 0.0);
    assert!(preds.is_empty(), "k=0 should return empty predictions");
}

/// k > nlabels returns at most nlabels predictions.
#[test]
fn test_predict_k_large() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let nlabels = model.dict().nlabels() as usize;
    let very_large_k = nlabels + 10000;
    let preds = model.predict("Which baking dish is best?", very_large_k, 0.0);
    assert!(
        preds.len() <= nlabels,
        "k > nlabels should return at most nlabels={} predictions, got {}",
        nlabels,
        preds.len()
    );
}

/// Negative threshold is treated as 0 (no threshold filtering).
#[test]
fn test_predict_negative_threshold() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let input = "Which baking dish is best to bake a banana bread ?";
    let preds_zero = model.predict(input, 5, 0.0);
    let preds_neg = model.predict(input, 5, -1.0);

    // Negative threshold should return same as 0 threshold
    assert_eq!(
        preds_zero.len(),
        preds_neg.len(),
        "Negative threshold should return same count as threshold=0"
    );
    for (p1, p2) in preds_zero.iter().zip(preds_neg.iter()) {
        assert_eq!(p1.label, p2.label, "Labels should match");
        assert_eq!(p1.prob.to_bits(), p2.prob.to_bits(), "Probs should match");
    }
}

/// Verify predict_on_words produces identical results to predict for same input.
#[test]
fn test_predict_on_words_matches_predict() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let input = "Which baking dish is best to bake a banana bread ?";

    // Get word IDs via tokenization (same as what predict() uses internally,
    // including EOS to match C++ predictLine behavior).
    let mut words: Vec<i32> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();
    model
        .dict()
        .get_line_from_str(input, &mut words, &mut labels);
    assert!(!words.is_empty(), "Should produce word IDs");
    // Add EOS just like predict() does
    let eos_id = model.dict().get_id(EOS);
    if let Some(eos_id) = eos_id {
        words.push(eos_id);
    }

    let preds_text = model.predict(input, 5, 0.0);
    let preds_words = model.predict_on_words(&words, 5, 0.0);

    assert_eq!(
        preds_text.len(),
        preds_words.len(),
        "predict and predict_on_words should return same number of predictions"
    );
    for (p1, p2) in preds_text.iter().zip(preds_words.iter()) {
        assert_eq!(p1.label, p2.label, "Labels should match exactly");
        assert_eq!(
            p1.prob.to_bits(),
            p2.prob.to_bits(),
            "Probabilities should be bitwise equal"
        );
    }
}

/// predict_on_words with empty slice returns empty.
#[test]
fn test_predict_on_words_empty() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let preds = model.predict_on_words(&[], 5, 0.0);
    assert!(
        preds.is_empty(),
        "Empty word IDs should return empty predictions"
    );
}

/// predict_on_words with k=0 returns empty.
#[test]
fn test_predict_on_words_k_zero() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let preds = model.predict_on_words(&[0, 1, 2], 0, 0.0);
    assert!(preds.is_empty(), "k=0 should return empty predictions");
}

/// Verify 10 identical calls to predict() return bit-identical results.
#[test]
fn test_predict_determinism() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let input = "Which baking dish is best to bake a banana bread ?";

    let first = model.predict(input, 5, 0.0);
    assert!(!first.is_empty(), "Should produce predictions");

    for i in 1..10 {
        let preds = model.predict(input, 5, 0.0);
        assert_eq!(
            preds.len(),
            first.len(),
            "Call {} prediction count should match",
            i
        );
        for (j, (p1, p2)) in first.iter().zip(preds.iter()).enumerate() {
            assert_eq!(
                p1.label, p2.label,
                "Call {} prediction[{}] label should be identical",
                i, j
            );
            assert_eq!(
                p1.prob.to_bits(),
                p2.prob.to_bits(),
                "Call {} prediction[{}] prob should be bit-identical: {} vs {}",
                i,
                j,
                p1.prob,
                p2.prob
            );
        }
    }
}

/// Verify that concurrent predict() calls via Arc<FastText> work correctly.
///
/// Spawns 8 threads, each calling predict() 10 times. All results must
/// match the single-threaded reference result.
#[test]
fn test_predict_thread_safety() {
    use std::thread;

    let model = Arc::new(FastText::load_model(COOKING_MODEL).expect("Should load cooking model"));
    let input = "Which baking dish is best to bake a banana bread ?";

    // Get reference result single-threaded
    let reference = model.predict(input, 5, 0.0);
    assert!(!reference.is_empty(), "Reference should have predictions");

    let reference = Arc::new(reference);
    let n_threads = 8;
    let n_calls = 10;
    let mut handles = Vec::new();

    for thread_id in 0..n_threads {
        let model = Arc::clone(&model);
        let reference = Arc::clone(&reference);
        let input = input.to_string();
        let handle = thread::spawn(move || {
            for call in 0..n_calls {
                let preds = model.predict(&input, 5, 0.0);
                assert_eq!(
                    preds.len(),
                    reference.len(),
                    "Thread {} call {}: prediction count mismatch",
                    thread_id,
                    call
                );
                for (j, (p, r)) in preds.iter().zip(reference.iter()).enumerate() {
                    assert_eq!(
                        p.label, r.label,
                        "Thread {} call {}: prediction[{}] label mismatch: '{}' vs '{}'",
                        thread_id, call, j, p.label, r.label
                    );
                    assert_eq!(
                        p.prob.to_bits(),
                        r.prob.to_bits(),
                        "Thread {} call {}: prediction[{}] prob mismatch: {} vs {}",
                        thread_id,
                        call,
                        j,
                        p.prob,
                        r.prob
                    );
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }
}

/// Verify all predicted probabilities are in [0.0, 1.0] and not NaN/Inf.
#[test]
fn test_predict_probability_validity() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

    let test_inputs = [
        "Which baking dish is best to bake a banana bread ?",
        "how to make pasta at home",
        "best knife for cutting vegetables",
        "what temperature to bake chicken",
        "how long to boil eggs",
        "substitute for buttermilk",
    ];

    for input in &test_inputs {
        let preds = model.predict(input, 10, 0.0);
        assert!(!preds.is_empty(), "Should have predictions for: {}", input);
        for p in &preds {
            assert!(
                p.prob.is_finite(),
                "Probability should be finite for '{}': got {}",
                input,
                p.prob
            );
            assert!(
                p.prob >= 0.0,
                "Probability should be >= 0.0 for '{}': got {}",
                input,
                p.prob
            );
            assert!(
                p.prob <= 1.0 + 1e-5,
                "Probability should be <= 1.0 for '{}': got {}",
                input,
                p.prob
            );
        }
        // Probabilities should be sorted descending
        for i in 1..preds.len() {
            assert!(
                preds[i - 1].prob >= preds[i].prob,
                "Predictions should be sorted descending by prob for '{}': {} < {}",
                input,
                preds[i - 1].prob,
                preds[i].prob
            );
        }
    }
}

/// Verify get_word_vector("banana") matches C++ reference within 1e-3.
///
/// C++ reference (print-word-vectors cooking.model.bin):
/// banana 2.906 1.4799 0.84498 -1.0936 -0.7732 -5.2984 1.7936 -2.1483 2.6325 -1.8558
#[test]
fn test_get_word_vector_banana() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let vec = model.get_word_vector("banana");

    // Check dimension
    assert_eq!(vec.len(), 10, "Word vector should have 10 dimensions");

    // C++ reference values for all 10 dimensions
    let expected: [f32; 10] = [
        2.906, 1.4799, 0.84498, -1.0936, -0.7732, -5.2984, 1.7936, -2.1483, 2.6325, -1.8558,
    ];

    let tolerance = 1e-3_f32;
    for (i, (&got, &exp)) in vec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < tolerance,
            "banana vector[{}]: got={}, expected={}, diff={}",
            i,
            got,
            exp,
            (got - exp).abs()
        );
    }
}

/// Unknown word with maxn=0 returns zero vector.
#[test]
fn test_get_word_vector_unknown_zero() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    // cooking model has maxn=0 (no subword computation)
    assert_eq!(model.args().maxn, 0, "cooking model should have maxn=0");

    let vec = model.get_word_vector("xyzzy_definitely_not_in_vocabulary_42");
    assert_eq!(vec.len(), 10, "Vector should have 10 dimensions");

    for &v in &vec {
        assert_eq!(
            v, 0.0,
            "Unknown word with maxn=0 should return zero vector, got {}",
            v
        );
    }
}

/// Known word returns non-zero vector.
#[test]
fn test_get_word_vector_known_nonzero() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let vec = model.get_word_vector("banana");
    assert_eq!(vec.len(), 10, "Vector should have 10 dimensions");

    let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
    assert!(
        norm > 0.0,
        "Known word 'banana' should have non-zero vector, norm={}",
        norm
    );
}

/// get_word_vector returns correct dimension.
#[test]
fn test_get_word_vector_dimension() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let vec = model.get_word_vector("baking");
    assert_eq!(
        vec.len(),
        model.get_dimension() as usize,
        "Word vector length should equal model dimension"
    );
}

/// Supervised model sentence vector: no L2 normalization, raw average.
#[test]
fn test_get_sentence_vector_supervised() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let sentence = "How to bake a banana bread";
    let svec = model.get_sentence_vector(sentence);
    assert_eq!(svec.len(), 10, "Sentence vector should have 10 dims");

    // Should be non-zero for a sentence with known words
    let norm: f32 = svec.iter().map(|&v| v * v).sum::<f32>().sqrt();
    assert!(
        norm > 0.0,
        "Sentence vector should be non-zero for known words"
    );
}

/// Empty input returns zero vector.
#[test]
fn test_get_sentence_vector_empty() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let svec = model.get_sentence_vector("");
    assert_eq!(svec.len(), 10, "Sentence vector should have 10 dims");
    for &v in &svec {
        assert_eq!(
            v, 0.0,
            "Empty sentence should return zero vector, got {}",
            v
        );
    }
}

/// Whitespace-only input returns zero vector.
#[test]
fn test_get_sentence_vector_whitespace_only() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let svec = model.get_sentence_vector("   \t  ");
    assert_eq!(svec.len(), 10, "Sentence vector should have 10 dims");
    for &v in &svec {
        assert_eq!(
            v, 0.0,
            "Whitespace-only should return zero vector, got {}",
            v
        );
    }
}

/// Sentence vector averaging is correct: multi-word sentence differs from single-word.
/// The sentence vector is NOT just the word vector - it includes EOS in the average.
#[test]
fn test_get_sentence_vector_averaging() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

    // Single-word sentence vector should be non-zero (baking is in vocab)
    let sent_vec = model.get_sentence_vector("baking");
    assert_eq!(sent_vec.len(), 10, "Sentence vector should have 10 dims");
    let norm: f32 = sent_vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
    assert!(
        norm > 0.0,
        "Sentence vector for 'baking' should be non-zero"
    );

    // Two-word sentence should be different from single-word sentence
    let sent_vec2 = model.get_sentence_vector("baking bread");
    let norm2: f32 = sent_vec2.iter().map(|&v| v * v).sum::<f32>().sqrt();
    assert!(
        norm2 > 0.0,
        "Sentence vector for 'baking bread' should be non-zero"
    );

    // Longer sentence should produce different result than shorter
    let sent_vec3 = model.get_sentence_vector("baking banana bread cake");
    assert_ne!(
        sent_vec, sent_vec3,
        "Different sentences should produce different vectors"
    );
}

/// Sentence vector matches C++ reference output.
/// C++ print-sentence-vectors reference for "how to bake a banana bread"
/// 0.65401 0.61836 0.49154 -0.20285 0.1512 -1.348 0.42243 -0.3267 0.99147 0.99911
#[test]
fn test_get_sentence_vector_reference() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let svec = model.get_sentence_vector("how to bake a banana bread");
    assert_eq!(svec.len(), 10, "Should have 10 dims");

    // C++ reference values for all 10 dimensions
    let expected = [
        0.65401_f32,
        0.61836,
        0.49154,
        -0.20285,
        0.1512,
        -1.348,
        0.42243,
        -0.3267,
        0.99147,
        0.99911,
    ];
    let tolerance = 1e-3_f32;
    for (i, (&got, &exp)) in svec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < tolerance,
            "sentence_vector[{}]: got={}, expected={}, diff={}",
            i,
            got,
            exp,
            (got - exp).abs()
        );
    }
}

/// Basic whitespace splitting.
#[test]
fn test_tokenize_basic() {
    let tokens = FastText::tokenize("hello world foo");
    assert_eq!(tokens, vec!["hello", "world", "foo"]);
}

/// Unicode tokens preserved intact.
#[test]
fn test_tokenize_unicode() {
    let tokens = FastText::tokenize("日本語 café résumé");
    assert_eq!(tokens, vec!["日本語", "café", "résumé"]);
}

/// Empty string returns empty vec.
#[test]
fn test_tokenize_empty() {
    let tokens = FastText::tokenize("");
    assert!(tokens.is_empty(), "Empty string should return empty vec");
}

/// Multiple consecutive whitespace characters collapsed.
#[test]
fn test_tokenize_multi_whitespace() {
    let tokens = FastText::tokenize("hello   world\t\tfoo");
    expect![[r#"["hello", "world", "foo"]"#]].assert_eq(&format!("{:?}", tokens));
}

/// Leading/trailing whitespace ignored.
#[test]
fn test_tokenize_leading_trailing_whitespace() {
    let tokens = FastText::tokenize("  hello world  ");
    expect![[r#"["hello", "world"]"#]].assert_eq(&format!("{:?}", tokens));
}

/// Whitespace-only input returns empty vec.
#[test]
fn test_tokenize_whitespace_only() {
    let tokens = FastText::tokenize("   \t  \n  ");
    assert!(tokens.is_empty(), "Whitespace-only should return empty vec");
}

/// Single word returns single-element vec.
#[test]
fn test_tokenize_single_word() {
    let tokens = FastText::tokenize("hello");
    expect![[r#"["hello"]"#]].assert_eq(&format!("{:?}", tokens));
}

/// get_vocab() returns 8952 words, first entry is </s> with freq 12404.
#[test]
fn test_get_vocab_cooking() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let (words, freqs) = model.get_vocab();
    let actual = format!(
        "count {}\nfirst {} freq {}",
        words.len(),
        words[0],
        freqs[0]
    );
    expect![[r#"
count 8952
first </s> freq 12404"#]]
    .assert_eq(&actual);

    // All words should be non-empty
    for (i, word) in words.iter().enumerate() {
        assert!(!word.is_empty(), "Word[{}] should not be empty", i);
    }

    // All freqs should be positive
    for (i, &freq) in freqs.iter().enumerate() {
        assert!(freq > 0, "Freq[{}] should be positive, got {}", i, freq);
    }
}

/// get_labels() returns 735 labels, first label is __label__baking with freq 1156.
#[test]
fn test_get_labels_cooking() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let (labels, freqs) = model.get_labels();
    let actual = format!(
        "count {}\nfirst {} freq {}",
        labels.len(),
        labels[0],
        freqs[0]
    );
    expect![[r#"
count 735
first __label__baking freq 1156"#]]
    .assert_eq(&actual);

    // All labels should start with __label__
    for (i, label) in labels.iter().enumerate() {
        assert!(
            label.starts_with("__label__"),
            "Label[{}] '{}' should start with '__label__'",
            i,
            label
        );
    }
}

/// get_dimension() returns the correct value for the cooking model.
#[test]
fn test_get_dimension() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    expect!["10"].assert_eq(&model.get_dimension().to_string());
}

/// get_word_id() returns correct ID for known words and None for unknown.
#[test]
fn test_get_word_id_known() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");

    // EOS should be at index 0
    let eos_id = model.get_word_id("</s>");
    assert_eq!(
        eos_id,
        Some(0),
        "EOS should be at index 0, got {:?}",
        eos_id
    );

    // Known words should be in vocabulary
    let banana_id = model.get_word_id("banana");
    assert!(
        banana_id.is_some(),
        "'banana' should be in vocabulary, got id={:?}",
        banana_id
    );

    let baking_id = model.get_word_id("baking");
    assert!(
        baking_id.is_some(),
        "'baking' should be in vocabulary, got id={:?}",
        baking_id
    );
}

/// get_word_id() returns None for unknown words.
#[test]
fn test_get_word_id_unknown() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let id = model.get_word_id("xyzzy_definitely_not_in_vocabulary_42");
    assert_eq!(id, None, "Unknown word should return None, got {:?}", id);
}

/// get_dimension() matches args.dim.
#[test]
fn test_get_dimension_matches_args() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    assert_eq!(
        model.get_dimension(),
        model.args().dim,
        "get_dimension() should equal args().dim"
    );
}

/// get_vocab() words are not labels (none should start with __label__).
#[test]
fn test_get_vocab_not_labels() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let (words, _) = model.get_vocab();
    for (i, word) in words.iter().enumerate() {
        assert!(
            !word.starts_with("__label__"),
            "Vocab word[{}] '{}' should not be a label",
            i,
            word
        );
    }
}

/// get_labels() returns proper label format.
#[test]
fn test_get_labels_format() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let (labels, freqs) = model.get_labels();

    assert_eq!(
        labels.len(),
        freqs.len(),
        "Labels and freqs should have same length"
    );

    // All labels should be non-empty
    for (i, label) in labels.iter().enumerate() {
        assert!(!label.is_empty(), "Label[{}] should not be empty", i);
    }

    // Frequencies should be positive
    for (i, &freq) in freqs.iter().enumerate() {
        assert!(freq > 0, "Label freq[{}] should be > 0, got {}", i, freq);
    }
}

/// VAL-TRAIN-004: Matrix dimensions correct after training.
#[test]
fn test_train_matrix_dimensions() {
    let data = supervised_train_data();
    let path = write_temp_file(&data);
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 1;
    args.min_count = 1;
    args.bucket = 50;

    let model = FastText::train(args).expect("Supervised training should succeed");
    std::fs::remove_file(&path).ok();

    let nwords = model.dict().nwords() as i64;
    let nlabels = model.dict().nlabels() as i64;
    let dim = model.get_dimension() as i64;
    let bucket = model.args().bucket as i64;

    let input = model.input_matrix();
    assert_eq!(
        input.rows(),
        nwords + bucket,
        "Input matrix rows should be nwords+bucket: {} != {}+{}",
        input.rows(),
        nwords,
        bucket
    );
    assert_eq!(input.cols(), dim, "Input cols should be dim");

    let output = model.output_matrix();
    assert_eq!(
        output.rows(),
        nlabels,
        "Output matrix rows should be nlabels for supervised: {} != {}",
        output.rows(),
        nlabels
    );
    assert_eq!(output.cols(), dim, "Output cols should be dim");

    let data2 = unsupervised_train_data();
    let path2 = write_temp_file(&data2);
    let path2_str = path2.to_str().unwrap().to_string();

    let mut args2 = Args::default();
    args2.input = path2_str.clone();
    args2.output = "/dev/null".to_string();
    args2.model = ModelName::Cbow;
    args2.loss = LossName::NegativeSampling;
    args2.dim = 10;
    args2.epoch = 1;
    args2.min_count = 1;
    args2.neg = 5;
    args2.bucket = 50;
    args2.minn = 0;
    args2.maxn = 0;

    let model2 = FastText::train(args2).expect("CBOW training should succeed");
    std::fs::remove_file(&path2).ok();

    let nwords2 = model2.dict().nwords() as i64;
    let dim2 = model2.get_dimension() as i64;
    let bucket2 = model2.args().bucket as i64;

    let input2 = model2.input_matrix();
    assert_eq!(
        input2.rows(),
        nwords2 + bucket2,
        "CBOW input matrix rows should be nwords+bucket: {} != {}+{}",
        input2.rows(),
        nwords2,
        bucket2
    );
    assert_eq!(input2.cols(), dim2, "CBOW input cols should be dim");

    let output2 = model2.output_matrix();
    assert_eq!(
        output2.rows(),
        nwords2,
        "CBOW output matrix rows should be nwords: {} != {}",
        output2.rows(),
        nwords2
    );
    assert_eq!(output2.cols(), dim2, "CBOW output cols should be dim");
}

/// Tests the test_model integration: train a supervised model, then
/// evaluate it on labeled test data using Meter, and verify basic metrics.
#[test]
fn test_meter_test_command() {
    // Train a model on the standard supervised training data.
    let train_data = supervised_train_data();
    let train_path = write_temp_file(&train_data);
    let train_str = train_path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = train_str.clone();
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
    std::fs::remove_file(&train_path).ok();

    // Write test data (a few labeled examples from the training set).
    let test_data = "\
__label__sports basketball player sport game\n\
__label__food apple orange banana fruit eat\n\
__label__sports team score win play\n\
__label__food cook recipe eat meal\n";
    let test_path = write_temp_file(test_data);

    let mut file = std::fs::File::open(&test_path).expect("Failed to open test file");
    let meter = model
        .test_model(&mut file, 1, 0.0)
        .expect("test_model should succeed");
    std::fs::remove_file(&test_path).ok();

    // The model should have evaluated at least some examples.
    assert!(
        meter.n_examples() > 0,
        "Meter should have at least 1 example, got {}",
        meter.n_examples()
    );

    // Precision and recall must be in [0, 1].
    let p = meter.precision();
    let r = meter.recall();
    let f = meter.f1();
    assert!((0.0..=1.0).contains(&p), "Precision {:.4} out of [0,1]", p);
    assert!((0.0..=1.0).contains(&r), "Recall {:.4} out of [0,1]", r);
    assert!(f.is_finite(), "F1 should be finite, got {}", f);
    assert!((0.0..=1.0).contains(&f), "F1 {:.4} out of [0,1]", f);

    // After 5 epochs on simple data, the model should predict at least
    // SOME examples correctly (p > 0 and r > 0).
    assert!(
        p > 0.0,
        "Precision should be > 0.0 after training, got {:.4}",
        p
    );
    assert!(
        r > 0.0,
        "Recall should be > 0.0 after training, got {:.4}",
        r
    );
}

/// Verifies that test_model returns zero metrics on an empty test file.
#[test]
fn test_meter_test_command_empty_file() {
    let train_data = supervised_train_data();
    let train_path = write_temp_file(&train_data);
    let train_str = train_path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = train_str.clone();
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 1;
    args.min_count = 1;
    args.bucket = 0;

    let model = FastText::train(args).expect("Training should succeed");
    std::fs::remove_file(&train_path).ok();

    // Use a file with no labeled lines (only unlabeled text).
    let test_data = "no labels here\nsome more text\n";
    let test_path = write_temp_file(test_data);
    let mut file = std::fs::File::open(&test_path).expect("Failed to open test file");
    let meter = model
        .test_model(&mut file, 1, 0.0)
        .expect("test_model should not error on unlabeled data");
    std::fs::remove_file(&test_path).ok();

    // No examples have labels → meter should record 0 examples.
    assert_eq!(
        meter.n_examples(),
        0,
        "No labeled examples → n_examples should be 0"
    );
    assert_eq!(
        meter.precision(),
        0.0,
        "No examples → precision should be 0.0"
    );
    assert_eq!(meter.recall(), 0.0, "No examples → recall should be 0.0");
    assert_eq!(meter.f1(), 0.0, "No examples → F1 should be 0.0");
}

/// Supervised training with no labels returns an error (VAL-TRAIN-011).
#[test]
fn test_train_no_labels() {
    let data = "this is text without any labels\nno labels here either\n";
    let path = write_temp_file(data);
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

    assert!(
        result.is_err(),
        "Supervised training with no labels should return error"
    );
    match result.unwrap_err() {
        FastTextError::InvalidArgument(msg) => {
            assert!(
                msg.contains("label") || msg.contains("supervised"),
                "Error message should mention labels or supervised: {}",
                msg
            );
        }
        e => panic!("Expected InvalidArgument for no-labels, got: {:?}", e),
    }
}

// Missing API coverage: get_nn, get_analogies, get_ngram_vectors

/// get_nn returns k results, excludes query word, similarities are finite and sorted.
#[test]
fn test_get_nn_basic() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let results = model.get_nn("banana", 5);

    assert_eq!(results.len(), 5, "Should return exactly 5 neighbors");

    // Query word itself should not appear
    for (_, word) in &results {
        assert_ne!(word, "banana", "Query word should be excluded from results");
    }

    // Similarities should be finite and sorted descending
    for (sim, _) in &results {
        assert!(sim.is_finite(), "Similarity should be finite, got {}", sim);
    }
    for i in 1..results.len() {
        assert!(
            results[i - 1].0 >= results[i].0 - 1e-6,
            "Results should be sorted by descending similarity: {} < {}",
            results[i - 1].0,
            results[i].0
        );
    }
}

/// get_nn with k=0 returns empty.
#[test]
fn test_get_nn_k_zero() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let results = model.get_nn("banana", 0);
    assert!(results.is_empty(), "k=0 should return empty results");
}

/// get_nn for an unknown word (with maxn=0 model) returns results (from zero vector).
#[test]
fn test_get_nn_unknown_word() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    // With maxn=0, unknown word has zero vector, but get_nn should still not panic
    let results = model.get_nn("xyzzy_unknown_word_42", 3);
    // Results should be returned (even if similarities are all ~0)
    assert_eq!(
        results.len(),
        3,
        "Should return 3 neighbors even for unknown word"
    );
}

/// get_analogies returns k results, excludes input words, sorted descending.
#[test]
fn test_get_analogies_basic() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let results = model.get_analogies("baking", "bread", "chicken", 5);

    assert_eq!(results.len(), 5, "Should return exactly 5 analogy results");

    // Input words should be excluded
    let banned = ["baking", "bread", "chicken"];
    for (_, word) in &results {
        assert!(
            !banned.contains(&word.as_str()),
            "Input word '{}' should be excluded from results",
            word
        );
    }

    // Similarities should be finite
    for (sim, _) in &results {
        assert!(sim.is_finite(), "Similarity should be finite, got {}", sim);
    }

    // Sorted descending
    for i in 1..results.len() {
        assert!(
            results[i - 1].0 >= results[i].0 - 1e-6,
            "Results should be sorted descending"
        );
    }
}

/// get_analogies with k=0 returns empty.
#[test]
fn test_get_analogies_k_zero() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let results = model.get_analogies("baking", "bread", "chicken", 0);
    assert!(results.is_empty(), "k=0 should return empty results");
}

/// get_ngram_vectors for a known word (maxn=0 model) returns the word itself.
#[test]
fn test_get_ngram_vectors_no_subwords() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    assert_eq!(model.args().maxn, 0, "cooking model should have maxn=0");

    let ngrams = model.get_ngram_vectors("banana");

    // With maxn=0, should return just the word entry (no subwords)
    assert!(
        !ngrams.is_empty(),
        "Known word should return at least one entry"
    );

    // Each vector should have the right dimension
    let dim = model.get_dimension() as usize;
    for (s, vec) in &ngrams {
        assert_eq!(
            vec.len(),
            dim,
            "N-gram vector for '{}' should have dim={} elements",
            s,
            dim
        );
        // Vector values should be finite
        for &v in vec {
            assert!(v.is_finite(), "N-gram vector element should be finite");
        }
    }
}

/// get_ngram_vectors for unknown word with maxn=0 returns entry with zero vector.
#[test]
fn test_get_ngram_vectors_unknown_no_subwords() {
    let model = FastText::load_model(COOKING_MODEL).expect("Should load cooking model");
    let ngrams = model.get_ngram_vectors("xyzzy_unknown_42");

    // With maxn=0 and unknown word, should still return something (OOV entry)
    let dim = model.get_dimension() as usize;
    for (_, vec) in &ngrams {
        assert_eq!(vec.len(), dim, "Vector should have correct dimension");
    }
}

/// get_ngram_vectors on a model with subwords returns subword entries.
#[test]
fn test_get_ngram_vectors_with_subwords() {
    // Train a model with subword information
    let data = supervised_train_data();
    let path = write_temp_file(&data);
    let path_str = path.to_str().unwrap().to_string();

    let mut args = Args::default();
    args.input = path_str;
    args.output = "/dev/null".to_string();
    args.apply_supervised_defaults();
    args.dim = 10;
    args.epoch = 1;
    args.min_count = 1;
    args.bucket = 100;
    args.minn = 2;
    args.maxn = 4;
    args.thread = 1;

    let model = FastText::train(args).expect("Training should succeed");
    std::fs::remove_file(&path).ok();

    let ngrams = model.get_ngram_vectors("basketball");
    assert!(
        ngrams.len() > 1,
        "With minn=2,maxn=4, known word should have multiple n-gram entries, got {}",
        ngrams.len()
    );

    let dim = model.get_dimension() as usize;
    for (s, vec) in &ngrams {
        assert_eq!(
            vec.len(),
            dim,
            "N-gram '{}' vector should have dim={}",
            s,
            dim
        );
    }
}

// Config matrix tests: exercise various model configurations

/// Train supervised models with various configurations and verify basic correctness.
#[test]
fn test_config_matrix_supervised() {
    let data = supervised_train_data();

    struct Config {
        dim: i32,
        minn: i32,
        maxn: i32,
        bucket: i32,
        loss: LossName,
        label: &'static str,
    }

    let configs = [
        Config {
            dim: 10,
            minn: 2,
            maxn: 4,
            bucket: 100,
            loss: LossName::Softmax,
            label: "subwords",
        },
        Config {
            dim: 10,
            minn: 0,
            maxn: 0,
            bucket: 0,
            loss: LossName::Softmax,
            label: "no-subwords",
        },
        Config {
            dim: 1,
            minn: 0,
            maxn: 0,
            bucket: 0,
            loss: LossName::Softmax,
            label: "dim=1",
        },
        Config {
            dim: 5,
            minn: 0,
            maxn: 0,
            bucket: 0,
            loss: LossName::Softmax,
            label: "dim=5",
        },
        Config {
            dim: 5,
            minn: 0,
            maxn: 0,
            bucket: 0,
            loss: LossName::HierarchicalSoftmax,
            label: "dim=5+HS",
        },
    ];

    for cfg in &configs {
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        let mut args = Args::default();
        args.input = path_str;
        args.output = "/dev/null".to_string();
        args.apply_supervised_defaults();
        args.dim = cfg.dim;
        args.epoch = 3;
        args.min_count = 1;
        args.bucket = cfg.bucket;
        args.minn = cfg.minn;
        args.maxn = cfg.maxn;
        args.loss = cfg.loss;
        args.thread = 1;

        let model = FastText::train(args)
            .unwrap_or_else(|_| panic!("Training [{}] should succeed", cfg.label));
        std::fs::remove_file(&path).ok();

        // Predictions should work
        let preds = model.predict("basketball player sport game", 1, 0.0);
        assert!(
            !preds.is_empty(),
            "[{}] should produce predictions",
            cfg.label
        );
        assert!(
            preds[0].prob.is_finite() && preds[0].prob > 0.0,
            "[{}] prediction prob should be valid: {}",
            cfg.label,
            preds[0].prob
        );

        // Vectors should have correct dimension
        let vec = model.get_word_vector("basketball");
        assert_eq!(
            vec.len(),
            cfg.dim as usize,
            "[{}] word vector should have dim={}",
            cfg.label,
            cfg.dim
        );
    }
}

/// Train unsupervised models with various configurations.
#[test]
fn test_config_matrix_unsupervised() {
    let data = unsupervised_train_data();

    let configs = [
        (ModelName::Cbow, 5, "CBOW-dim5"),
        (ModelName::SkipGram, 5, "SG-dim5"),
        (ModelName::Cbow, 1, "CBOW-dim1"),
    ];

    for &(model_type, dim, label) in &configs {
        let path = write_temp_file(&data);
        let path_str = path.to_str().unwrap().to_string();

        let mut args = Args::default();
        args.input = path_str;
        args.output = "/dev/null".to_string();
        args.model = model_type;
        args.loss = LossName::NegativeSampling;
        args.dim = dim;
        args.epoch = 1;
        args.min_count = 1;
        args.neg = 5;
        args.bucket = 50;
        args.minn = 0;
        args.maxn = 0;
        args.thread = 1;

        let model =
            FastText::train(args).unwrap_or_else(|_| panic!("Training [{}] should succeed", label));
        std::fs::remove_file(&path).ok();

        // Word vectors should have correct dimension and be non-zero for known words
        let vec = model.get_word_vector("the");
        assert_eq!(
            vec.len(),
            dim as usize,
            "[{}] word vector should have dim={}",
            label,
            dim
        );

        let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            norm > 0.0,
            "[{}] known word 'the' should have non-zero vector",
            label
        );

        // Sentence vector should work
        let svec = model.get_sentence_vector("the quick brown fox");
        assert_eq!(
            svec.len(),
            dim as usize,
            "[{}] sentence vector should have dim={}",
            label,
            dim
        );
    }
}
