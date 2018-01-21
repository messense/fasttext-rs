extern crate fasttext;

use fasttext::FastText;


#[test]
fn test_fasttext_load_model() {
    let mut fasttext = FastText::new();
    fasttext.load_model("tests/fixtures/cooking.model.bin");
}

#[test]
fn test_fasttext_is_quant() {
    let mut fasttext = FastText::new();
    fasttext.load_model("tests/fixtures/cooking.model.bin");
    assert!(!fasttext.is_quant());
}

#[test]
fn test_fasttext_predict() {
    let mut fasttext = FastText::new();
    fasttext.load_model("tests/fixtures/cooking.model.bin");
    let preds = fasttext.predict("Which baking dish is best to bake a banana bread ?", 1, 0.0);
    assert_eq!(1, preds.len());
    assert_eq!("__label__baking", &preds[0].label);
}
