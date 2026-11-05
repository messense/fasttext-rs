//! Synthetic hierarchical-softmax models: no downloaded fixtures or training.
#![allow(clippy::field_reassign_with_default)]

use std::io::Cursor;
use std::sync::Arc;

use fasttext::args::{Args, LossName};
use fasttext::dictionary::Dictionary;
use fasttext::error::FastTextError;
use fasttext::fasttext::{FASTTEXT_FILEFORMAT_MAGIC_INT32, FASTTEXT_VERSION};
use fasttext::matrix::{DenseMatrix, Matrix};
use fasttext::FastText;

fn small_model(loss: LossName) -> FastText {
    let mut args = Args::default();
    args.apply_supervised_defaults();
    args.loss = loss;
    args.dim = 2;
    args.bucket = 0;
    args.min_count = 1;
    args.minn = 0;
    args.maxn = 0;

    // Unequal counts fix the Huffman tree: root -> (gamma/beta, alpha).
    let data = "__label__alpha word\n".repeat(4)
        + &"__label__beta word\n".repeat(2)
        + "__label__gamma word\n";
    let mut dict = Dictionary::new_with_capacity(Arc::new(args.clone()), 32);
    dict.read_from_file(&mut Cursor::new(data)).unwrap();

    // Every input row (including EOS) is [1, 0]. The two internal-node
    // sigmoids are 3/4 and 4/5, giving leaf probabilities .05, .15, .80.
    let input = DenseMatrix::from_data(
        dict.nwords() as i64,
        2,
        &[1.0, 0.0].repeat(dict.nwords() as usize),
    );
    let output = DenseMatrix::from_data(3, 2, &[3.0_f32.ln(), 0.0, 4.0_f32.ln(), 0.0, 0.0, 0.0]);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&FASTTEXT_FILEFORMAT_MAGIC_INT32.to_le_bytes());
    bytes.extend_from_slice(&FASTTEXT_VERSION.to_le_bytes());
    args.save(&mut bytes).unwrap();
    dict.save(&mut bytes).unwrap();
    bytes.push(0); // Dense input.
    input.save(&mut bytes).unwrap();
    bytes.push(0); // Dense output.
    output.save(&mut bytes).unwrap();
    FastText::load(&mut Cursor::new(bytes)).unwrap()
}

fn assert_hs_predictions(model: &FastText) {
    for (k, threshold, expected_len) in [(3, 0.0, 3), (1, 0.0, 1), (3, 0.1, 2), (3, 0.9, 0)] {
        let predictions = model.predict("word", k, threshold);
        assert_eq!(predictions.len(), expected_len);
        for (prediction, (label, prob)) in predictions.iter().zip([
            ("__label__alpha", 0.80),
            ("__label__beta", 0.15),
            ("__label__gamma", 0.05),
        ]) {
            assert_eq!(prediction.label, label);
            // std_log adds 1e-5, so allow the resulting small probability offset.
            assert!((prediction.prob - prob).abs() < 3e-5, "{prediction:?}");
        }
    }
}

#[test]
fn quantized_hs_predictions_and_roundtrip() {
    let mut model = small_model(LossName::HierarchicalSoftmax);
    assert_hs_predictions(&model);

    let qargs = Args::default(); // qout=false, dsub=2; the input is encoded exactly.
    model.quantize(&qargs).unwrap();
    assert!(model.is_quant());
    assert!(model.quant_output().is_none());
    assert_hs_predictions(&model);

    let mut bytes = Vec::new();
    model.save(&mut bytes).unwrap();
    let loaded = FastText::load(&mut Cursor::new(bytes)).unwrap();
    assert!(loaded.is_quant());
    assert_hs_predictions(&loaded);
}

#[test]
fn quantized_hs_rejects_qout_before_mutation() {
    let mut model = small_model(LossName::HierarchicalSoftmax);
    let mut before = Vec::new();
    model.save(&mut before).unwrap();
    let qargs = Args {
        qout: true,
        cutoff: 1,
        ..Args::default()
    };
    let Err(error) = model.quantize(&qargs) else {
        panic!("hierarchical softmax with qout=true must be rejected");
    };
    assert!(
        matches!(&error, FastTextError::InvalidArgument(message)
        if message.contains("hierarchical softmax") && message.contains("qout")),
        "{error}"
    );
    let mut after = Vec::new();
    model.save(&mut after).unwrap();
    assert_eq!(before, after, "rejection must leave the model unchanged");
    assert_hs_predictions(&model);
}

#[test]
fn quantized_hs_rejects_loading_qout() {
    let mut model = small_model(LossName::Softmax);
    model
        .quantize(&Args {
            qout: true,
            ..Args::default()
        })
        .unwrap();
    let mut bytes = Vec::new();
    model.save(&mut bytes).unwrap();
    // First confirm that quantized output is still supported for softmax.
    FastText::load(&mut Cursor::new(&bytes)).unwrap();

    // Construct an HS + qout file independently of the public quantize guard.
    // The matrix format is identical; only the loss in Args changes, after
    // the two i32 header fields (magic and version).
    let mut args = model.args().clone();
    args.loss = LossName::HierarchicalSoftmax;
    args.save(&mut Cursor::new(&mut bytes[8..])).unwrap();
    let Err(error) = FastText::load(&mut Cursor::new(bytes)) else {
        panic!("loading hierarchical softmax with qout=true must be rejected");
    };
    assert!(
        matches!(&error, FastTextError::InvalidModel(message)
        if message.contains("hierarchical softmax") && message.contains("qout")),
        "{error}"
    );
}
