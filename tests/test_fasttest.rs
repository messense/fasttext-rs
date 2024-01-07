use fasttext::FastText;

#[test]
fn test_fasttext_load_model() {
    let mut fasttext = FastText::new();
    assert!(fasttext
        .load_model("tests/fixtures/cooking.model.bin")
        .is_ok());
}

#[test]
fn test_fasttext_load_invalid_model() {
    let mut fasttext = FastText::new();
    assert!(fasttext
        .load_model("tests/fixtures/invalid.model.bin")
        .is_err());
}

#[test]
fn test_fasttext_get_args() {
    let mut fasttext = FastText::new();
    fasttext
        .load_model("tests/fixtures/cooking.model.bin")
        .unwrap();
    let args = fasttext.get_args();
    assert_eq!(fasttext::ModelName::SUP, args.model());
}

#[test]
fn test_fasttext_is_quant() {
    let mut fasttext = FastText::new();
    fasttext
        .load_model("tests/fixtures/cooking.model.bin")
        .unwrap();
    assert!(!fasttext.is_quant());
}

#[test]
fn test_fasttext_predict() {
    let mut fasttext = FastText::new();
    fasttext
        .load_model("tests/fixtures/cooking.model.bin")
        .unwrap();
    let preds = fasttext
        .predict("Which baking\0 dish is best to bake a banana bread ?", 2, 0.0)
        .unwrap();
    assert_eq!(2, preds.len());
    assert_eq!("__label__baking", &preds[0].label);
    assert_eq!("__label__bread", &preds[1].label);
}

#[test]
fn test_fasttext_get_vocab() {
    let mut fasttext = FastText::new();
    fasttext
        .load_model("tests/fixtures/cooking.model.bin")
        .unwrap();
    let (words, freqs) = fasttext.get_vocab().unwrap();
    assert_eq!(14543, words.len());
    assert_eq!(14543, freqs.len());
    assert_eq!("</s>", &words[0]);
    assert_eq!(12404, freqs[0]);
}

#[test]
fn test_fasttext_get_labels() {
    let mut fasttext = FastText::new();
    fasttext
        .load_model("tests/fixtures/cooking.model.bin")
        .unwrap();
    let (labels, freqs) = fasttext.get_labels().unwrap();
    assert_eq!(735, labels.len());
    assert_eq!(735, freqs.len());
    assert_eq!("__label__baking", &labels[0]);
    assert_eq!(1156, freqs[0]);
}

#[test]
fn test_fasttext_predict_on_words() {
    let mut fasttext = FastText::new();
    fasttext
        .load_model("tests/fixtures/cooking.model.bin")
        .unwrap();
    let words = vec![
        "Which", "baking", "dish", "is", "best", "to", "bake", "a", "banana", "bread", "?",
    ];
    let words_ids: Vec<i32> = words
        .iter()
        .map(|&x| fasttext.get_word_id(x).unwrap() as i32)
        .collect();
    let preds = fasttext.predict_on_words(&words_ids, 2, 0.0).unwrap();
    assert_eq!(2, preds.len());
    assert_eq!("__label__baking", &preds[0].label);
    assert_eq!("__label__bread", &preds[1].label);
}
