// C++ parity tests: verify Rust output matches C++ fastText golden references
//
// The cooking.model.bin fixture was trained with C++ fastText using:
//   ./fasttext supervised -input cooking.train -output cooking_ref \
//     -dim 10 -epoch 25 -lr 1.0 -wordNgrams 1 -minCount 1 -bucket 0 \
//     -minn 0 -maxn 0 -thread 1 -seed 42 -loss softmax
//
// Training data: cooking.stackexchange.txt preprocessed (lowercased, punctuation
// separated) and split into 12404 train / 3000 valid lines, per the official
// fastText tutorial: https://fasttext.cc/docs/en/supervised-tutorial.html
//
// Golden reference outputs were captured from the C++ binary.

use expect_test::expect;
use fasttext::FastText;

const COOKING_MODEL: &str = "tests/fixtures/cooking.model.bin";
const COOKING_VALID: &str = "tests/fixtures/cooking.valid";

// ---------------------------------------------------------------------------
// predict / predict-prob parity
// ---------------------------------------------------------------------------

/// C++ reference:
///   echo "which baking dish is best to bake a banana bread ?" | ./fasttext predict-prob model - 5
///   __label__baking 0.72013 __label__bread 0.205032 __label__quickbread 0.017047
///   __label__oven 0.0105739 __label__rising 0.00388523
#[test]
fn test_cpp_parity_predict_prob() {
    let model = FastText::load_model(COOKING_MODEL).unwrap();
    let input = "which baking dish is best to bake a banana bread ?";
    let preds = model.predict(input, 5, 0.0);

    let expected = [
        ("__label__baking", 0.72013_f32),
        ("__label__bread", 0.205032),
        ("__label__quickbread", 0.017047),
        ("__label__oven", 0.0105739),
        ("__label__rising", 0.00388523),
    ];

    assert_eq!(preds.len(), 5, "Should return 5 predictions");
    for (i, &(label, prob)) in expected.iter().enumerate() {
        assert_eq!(
            preds[i].label, label,
            "Prediction[{}] label mismatch: got '{}', expected '{}'",
            i, preds[i].label, label
        );
        assert!(
            (preds[i].prob - prob).abs() < 1e-4,
            "Prediction[{}] '{}' prob mismatch: got {}, expected {}, diff={}",
            i, label, preds[i].prob, prob, (preds[i].prob - prob).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// test (evaluation) parity
// ---------------------------------------------------------------------------

/// C++ reference:
///   ./fasttext test model cooking.valid 1
///   N  3000  P@1  0.482  R@1  0.209
#[test]
fn test_cpp_parity_test_k1() {
    let model = FastText::load_model(COOKING_MODEL).unwrap();
    let mut file = std::fs::File::open(COOKING_VALID).unwrap();
    let meter = model.test_model(&mut file, 1, 0.0).unwrap();

    assert_eq!(meter.n_examples(), 3000, "N should be 3000");

    let p = meter.precision();
    let r = meter.recall();
    assert!(
        (p - 0.482).abs() < 1e-3,
        "P@1 mismatch: got {:.4}, expected 0.482",
        p
    );
    assert!(
        (r - 0.209).abs() < 1e-3,
        "R@1 mismatch: got {:.4}, expected 0.209",
        r
    );
}

/// C++ reference:
///   ./fasttext test model cooking.valid 5
///   N  3000  P@5  0.21  R@5  0.455
#[test]
fn test_cpp_parity_test_k5() {
    let model = FastText::load_model(COOKING_MODEL).unwrap();
    let mut file = std::fs::File::open(COOKING_VALID).unwrap();
    let meter = model.test_model(&mut file, 5, 0.0).unwrap();

    assert_eq!(meter.n_examples(), 3000);

    let p = meter.precision();
    let r = meter.recall();
    assert!(
        (p - 0.21).abs() < 1e-3,
        "P@5 mismatch: got {:.4}, expected 0.21",
        p
    );
    assert!(
        (r - 0.455).abs() < 1e-3,
        "R@5 mismatch: got {:.4}, expected 0.455",
        r
    );
}

// ---------------------------------------------------------------------------
// word vector parity
// ---------------------------------------------------------------------------

/// C++ reference:
///   echo "banana" | ./fasttext print-word-vectors model
///   banana 2.906 1.4799 0.84498 -1.0936 -0.7732 -5.2984 1.7936 -2.1483 2.6325 -1.8558
#[test]
fn test_cpp_parity_word_vector() {
    let model = FastText::load_model(COOKING_MODEL).unwrap();
    let vec = model.get_word_vector("banana");

    let expected: [f32; 10] = [
        2.906, 1.4799, 0.84498, -1.0936, -0.7732, -5.2984, 1.7936, -2.1483, 2.6325, -1.8558,
    ];

    assert_eq!(vec.len(), 10);
    for (i, (&got, &exp)) in vec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "word_vector[{}]: got {}, expected {}, diff={}",
            i, got, exp, (got - exp).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// sentence vector parity
// ---------------------------------------------------------------------------

/// C++ reference:
///   echo "how to bake a banana bread" | ./fasttext print-sentence-vectors model
///   0.65401 0.61836 0.49154 -0.20285 0.1512 -1.348 0.42243 -0.3267 0.99147 0.99911
#[test]
fn test_cpp_parity_sentence_vector() {
    let model = FastText::load_model(COOKING_MODEL).unwrap();
    let vec = model.get_sentence_vector("how to bake a banana bread");

    let expected: [f32; 10] = [
        0.65401, 0.61836, 0.49154, -0.20285, 0.1512, -1.348, 0.42243, -0.3267, 0.99147, 0.99911,
    ];

    assert_eq!(vec.len(), 10);
    for (i, (&got, &exp)) in vec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "sentence_vector[{}]: got {}, expected {}, diff={}",
            i, got, exp, (got - exp).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// nearest neighbor parity
// ---------------------------------------------------------------------------

/// C++ reference:
///   echo "banana" | ./fasttext nn model 5
///   bananas 0.89783
///   unmixed 0.864038
///   blueberry 0.846918
///   pulp 0.831612
///   muffin 0.80902
#[test]
fn test_cpp_parity_nn() {
    let model = FastText::load_model(COOKING_MODEL).unwrap();
    let results = model.get_nn("banana", 5);

    let expected = [
        ("bananas", 0.89783_f32),
        ("unmixed", 0.864038),
        ("blueberry", 0.846918),
        ("pulp", 0.831612),
        ("muffin", 0.80902),
    ];

    assert_eq!(results.len(), 5, "Should return 5 neighbors");
    for (i, &(word, sim)) in expected.iter().enumerate() {
        assert_eq!(
            results[i].1, word,
            "nn[{}] word mismatch: got '{}', expected '{}'",
            i, results[i].1, word
        );
        assert!(
            (results[i].0 - sim).abs() < 1e-4,
            "nn[{}] '{}' similarity mismatch: got {}, expected {}, diff={}",
            i, word, results[i].0, sim, (results[i].0 - sim).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// analogies parity
// ---------------------------------------------------------------------------

/// C++ reference:
///   echo "baking bread chicken" | ./fasttext analogies model 5
///   215f 0.848404, immediately 0.847833, waffles 0.845236,
///   steaming 0.840774, surfaces 0.83015
#[test]
fn test_cpp_parity_analogies() {
    let model = FastText::load_model(COOKING_MODEL).unwrap();
    let results = model.get_analogies("baking", "bread", "chicken", 5);

    let expected = [
        ("215f", 0.848404_f32),
        ("immediately", 0.847833),
        ("waffles", 0.845236),
        ("steaming", 0.840774),
        ("surfaces", 0.83015),
    ];

    assert_eq!(results.len(), 5, "Should return 5 analogy results");
    for (i, &(word, sim)) in expected.iter().enumerate() {
        assert_eq!(
            results[i].1, word,
            "analogies[{}] word mismatch: got '{}', expected '{}'",
            i, results[i].1, word
        );
        assert!(
            (results[i].0 - sim).abs() < 1e-4,
            "analogies[{}] '{}' similarity mismatch: got {}, expected {}, diff={}",
            i, word, results[i].0, sim, (results[i].0 - sim).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// model args parity
// ---------------------------------------------------------------------------

/// Verify loaded model args match the training configuration exactly.
#[test]
fn test_cpp_parity_args() {
    let model = FastText::load_model(COOKING_MODEL).unwrap();
    let args = model.args();
    let actual = format!(
        "dim {}\nws {}\nepoch {}\nminCount {}\nneg {}\nwordNgrams {}\nbucket {}\nminn {}\nmaxn {}\nlrUpdateRate {}\nt {}",
        args.dim, args.ws, args.epoch, args.min_count, args.neg, args.word_ngrams,
        args.bucket, args.minn, args.maxn, args.lr_update_rate, args.t
    );
    expect![[r#"
dim 10
ws 5
epoch 25
minCount 1
neg 5
wordNgrams 1
bucket 0
minn 0
maxn 0
lrUpdateRate 100
t 0.0001"#]].assert_eq(&actual);
}

/// Verify dictionary sizes match C++ dump output.
#[test]
fn test_cpp_parity_dict() {
    let model = FastText::load_model(COOKING_MODEL).unwrap();
    let dict = model.dict();
    let actual = format!("nwords {}\nnlabels {}", dict.nwords(), dict.nlabels());
    expect![[r#"
nwords 8952
nlabels 735"#]].assert_eq(&actual);
}

// ---------------------------------------------------------------------------
// ngram vectors parity (maxn=0: just the word vector)
// ---------------------------------------------------------------------------

/// C++ reference:
///   ./fasttext print-ngrams model banana
///   banana 2.906 1.4799 0.84498 -1.0936 -0.7732 -5.2984 1.7936 -2.1483 2.6325 -1.8558
#[test]
fn test_cpp_parity_ngram_vectors() {
    let model = FastText::load_model(COOKING_MODEL).unwrap();
    let ngrams = model.get_ngram_vectors("banana");

    // With maxn=0, should return just the word entry
    assert!(!ngrams.is_empty(), "Should return at least one entry");

    let expected: [f32; 10] = [
        2.906, 1.4799, 0.84498, -1.0936, -0.7732, -5.2984, 1.7936, -2.1483, 2.6325, -1.8558,
    ];

    // Find the "banana" entry
    let banana_entry = ngrams.iter().find(|(s, _)| s == "banana");
    assert!(banana_entry.is_some(), "Should have 'banana' entry");
    let (_, vec) = banana_entry.unwrap();
    assert_eq!(vec.len(), 10);
    for (i, (&got, &exp)) in vec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "ngram_vector[{}]: got {}, expected {}",
            i, got, exp
        );
    }
}
