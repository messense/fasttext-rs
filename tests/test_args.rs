// Args tests: defaults, supervised overrides, serialization, enums
//
// Extracted from src/args.rs inline tests. These test the public
// API for Args hyperparameter configuration.

use std::io::Cursor;

use fasttext::args::{Args, LossName, MetricName, ModelName};

// ===== VAL-CORE-001: Args default values match C++ =====

#[test]
fn test_args_defaults() {
    let args = Args::default();

    // Basic parameters
    assert_eq!(args.input, "");
    assert_eq!(args.output, "");
    assert!((args.lr - 0.05).abs() < f64::EPSILON);
    assert_eq!(args.lr_update_rate, 100);
    assert_eq!(args.dim, 100);
    assert_eq!(args.ws, 5);
    assert_eq!(args.epoch, 5);
    assert_eq!(args.min_count, 5);
    assert_eq!(args.min_count_label, 0);
    assert_eq!(args.neg, 5);
    assert_eq!(args.word_ngrams, 1);
    assert_eq!(args.loss, LossName::NS);
    assert_eq!(args.model, ModelName::SG);
    assert_eq!(args.bucket, 2_000_000);
    assert_eq!(args.minn, 3);
    assert_eq!(args.maxn, 6);
    assert_eq!(args.thread, 12);
    assert!((args.t - 1e-4).abs() < f64::EPSILON);
    assert_eq!(args.label, "__label__");
    assert_eq!(args.verbose, 2);
    assert_eq!(args.pretrained_vectors, "");
    assert!(!args.save_output);
    assert_eq!(args.seed, 0);

    // Quantization parameters
    assert!(!args.qout);
    assert!(!args.retrain);
    assert!(!args.qnorm);
    assert_eq!(args.cutoff, 0);
    assert_eq!(args.dsub, 2);

    // Autotune parameters
    assert_eq!(args.autotune_validation_file, "");
    assert_eq!(args.autotune_metric, "f1");
    assert_eq!(args.autotune_predictions, 1);
    assert_eq!(args.autotune_duration, 300);
    assert_eq!(args.autotune_model_size, "");
}

// ===== VAL-CORE-002: Args supervised mode overrides =====

#[test]
fn test_args_supervised_overrides() {
    let mut args = Args::default();
    args.apply_supervised_defaults();

    assert_eq!(args.model, ModelName::SUP);
    assert_eq!(args.loss, LossName::SOFTMAX);
    assert_eq!(args.min_count, 1);
    assert_eq!(args.minn, 0);
    assert_eq!(args.maxn, 0);
    assert!((args.lr - 0.1).abs() < f64::EPSILON);
}

#[test]
fn test_args_supervised_bucket_zero() {
    // Default: wordNgrams=1, maxn=0 (after supervised overrides), no autotune
    // => bucket should be 0
    let mut args = Args::default();
    args.apply_supervised_defaults();
    assert_eq!(args.bucket, 0);
}

#[test]
fn test_args_supervised_bucket_nonzero_with_word_ngrams() {
    // wordNgrams > 1 => bucket should NOT be zeroed
    let mut args = Args::default();
    args.word_ngrams = 2;
    args.apply_supervised_defaults();
    assert_eq!(args.bucket, 2_000_000);
}

#[test]
fn test_args_supervised_bucket_nonzero_with_autotune() {
    // With autotune enabled => bucket should NOT be zeroed
    let mut args = Args::default();
    args.autotune_validation_file = "valid.txt".to_string();
    args.apply_supervised_defaults();
    // maxn is 0 and wordNgrams is 1, but autotune is enabled
    assert_eq!(args.bucket, 2_000_000);
}

// ===== has_autotune() =====

#[test]
fn test_has_autotune() {
    let args = Args::default();
    assert!(!args.has_autotune());

    let mut args = Args::default();
    args.autotune_validation_file = "validation.txt".to_string();
    assert!(args.has_autotune());
}

// ===== VAL-CORE-004: Args binary serialization layout =====

#[test]
fn test_args_binary_serialization_layout() {
    let args = Args::default();

    // Serialize
    let mut buf = Vec::new();
    args.save(&mut buf).unwrap();

    // Exactly 56 bytes: 12 * 4 (i32) + 1 * 8 (f64) = 48 + 8 = 56
    assert_eq!(buf.len(), 56, "Args binary block must be exactly 56 bytes");

    // Deserialize and verify round-trip
    let mut args2 = Args::default();
    // Set non-default values to make sure load overwrites them
    args2.dim = 999;
    args2.ws = 999;
    args2.epoch = 999;

    let mut cursor = Cursor::new(&buf);
    args2.load(&mut cursor).unwrap();

    assert_eq!(args2.dim, args.dim);
    assert_eq!(args2.ws, args.ws);
    assert_eq!(args2.epoch, args.epoch);
    assert_eq!(args2.min_count, args.min_count);
    assert_eq!(args2.neg, args.neg);
    assert_eq!(args2.word_ngrams, args.word_ngrams);
    assert_eq!(args2.loss, args.loss);
    assert_eq!(args2.model, args.model);
    assert_eq!(args2.bucket, args.bucket);
    assert_eq!(args2.minn, args.minn);
    assert_eq!(args2.maxn, args.maxn);
    assert_eq!(args2.lr_update_rate, args.lr_update_rate);
    assert!((args2.t - args.t).abs() < f64::EPSILON);
}

#[test]
fn test_args_binary_serialization_nondefault() {
    let mut args = Args::default();
    args.dim = 300;
    args.ws = 10;
    args.epoch = 25;
    args.min_count = 3;
    args.neg = 10;
    args.word_ngrams = 2;
    args.loss = LossName::SOFTMAX;
    args.model = ModelName::SUP;
    args.bucket = 500_000;
    args.minn = 2;
    args.maxn = 5;
    args.lr_update_rate = 50;
    args.t = 1e-3;

    let mut buf = Vec::new();
    args.save(&mut buf).unwrap();
    assert_eq!(buf.len(), 56);

    let mut args2 = Args::default();
    let mut cursor = Cursor::new(&buf);
    args2.load(&mut cursor).unwrap();

    assert_eq!(args2.dim, 300);
    assert_eq!(args2.ws, 10);
    assert_eq!(args2.epoch, 25);
    assert_eq!(args2.min_count, 3);
    assert_eq!(args2.neg, 10);
    assert_eq!(args2.word_ngrams, 2);
    assert_eq!(args2.loss, LossName::SOFTMAX);
    assert_eq!(args2.model, ModelName::SUP);
    assert_eq!(args2.bucket, 500_000);
    assert_eq!(args2.minn, 2);
    assert_eq!(args2.maxn, 5);
    assert_eq!(args2.lr_update_rate, 50);
    assert!((args2.t - 1e-3).abs() < f64::EPSILON);
}

#[test]
fn test_args_binary_serialization_field_order() {
    // Verify exact field order by manually checking written bytes
    let mut args = Args::default();
    args.dim = 1;
    args.ws = 2;
    args.epoch = 3;
    args.min_count = 4;
    args.neg = 5;
    args.word_ngrams = 6;
    args.loss = LossName::SOFTMAX; // = 3
    args.model = ModelName::SUP; // = 3
    args.bucket = 9;
    args.minn = 10;
    args.maxn = 11;
    args.lr_update_rate = 12;
    args.t = 0.5;

    let mut buf = Vec::new();
    args.save(&mut buf).unwrap();

    // Read back as raw i32 values
    let mut cursor = Cursor::new(&buf);
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 1); // dim
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 2); // ws
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 3); // epoch
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 4); // minCount
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 5); // neg
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 6); // wordNgrams
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 3); // loss (SOFTMAX=3)
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 3); // model (SUP=3)
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 9); // bucket
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 10); // minn
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 11); // maxn
    assert_eq!(fasttext::utils::read_i32(&mut cursor).unwrap(), 12); // lrUpdateRate
    let t_val = fasttext::utils::read_f64(&mut cursor).unwrap();
    assert!((t_val - 0.5).abs() < f64::EPSILON); // t
}

#[test]
fn test_args_binary_load_invalid_loss() {
    // Create a buffer with an invalid loss value
    let mut buf = Vec::new();
    fasttext::utils::write_i32(&mut buf, 100).unwrap(); // dim
    fasttext::utils::write_i32(&mut buf, 5).unwrap(); // ws
    fasttext::utils::write_i32(&mut buf, 5).unwrap(); // epoch
    fasttext::utils::write_i32(&mut buf, 5).unwrap(); // minCount
    fasttext::utils::write_i32(&mut buf, 5).unwrap(); // neg
    fasttext::utils::write_i32(&mut buf, 1).unwrap(); // wordNgrams
    fasttext::utils::write_i32(&mut buf, 99).unwrap(); // loss = INVALID
    fasttext::utils::write_i32(&mut buf, 1).unwrap(); // model
    fasttext::utils::write_i32(&mut buf, 2000000).unwrap(); // bucket
    fasttext::utils::write_i32(&mut buf, 3).unwrap(); // minn
    fasttext::utils::write_i32(&mut buf, 6).unwrap(); // maxn
    fasttext::utils::write_i32(&mut buf, 100).unwrap(); // lrUpdateRate
    fasttext::utils::write_f64(&mut buf, 1e-4).unwrap(); // t

    let mut args = Args::default();
    let mut cursor = Cursor::new(&buf);
    let result = args.load(&mut cursor);
    assert!(result.is_err());
}

#[test]
fn test_args_binary_load_invalid_model() {
    let mut buf = Vec::new();
    fasttext::utils::write_i32(&mut buf, 100).unwrap(); // dim
    fasttext::utils::write_i32(&mut buf, 5).unwrap(); // ws
    fasttext::utils::write_i32(&mut buf, 5).unwrap(); // epoch
    fasttext::utils::write_i32(&mut buf, 5).unwrap(); // minCount
    fasttext::utils::write_i32(&mut buf, 5).unwrap(); // neg
    fasttext::utils::write_i32(&mut buf, 1).unwrap(); // wordNgrams
    fasttext::utils::write_i32(&mut buf, 2).unwrap(); // loss = NS (valid)
    fasttext::utils::write_i32(&mut buf, 99).unwrap(); // model = INVALID
    fasttext::utils::write_i32(&mut buf, 2000000).unwrap(); // bucket
    fasttext::utils::write_i32(&mut buf, 3).unwrap(); // minn
    fasttext::utils::write_i32(&mut buf, 6).unwrap(); // maxn
    fasttext::utils::write_i32(&mut buf, 100).unwrap(); // lrUpdateRate
    fasttext::utils::write_f64(&mut buf, 1e-4).unwrap(); // t

    let mut args = Args::default();
    let mut cursor = Cursor::new(&buf);
    let result = args.load(&mut cursor);
    assert!(result.is_err());
}

#[test]
fn test_args_binary_load_truncated() {
    // Only 20 bytes, not enough for the full 56-byte block
    let buf = vec![0u8; 20];
    let mut args = Args::default();
    let mut cursor = Cursor::new(&buf);
    let result = args.load(&mut cursor);
    assert!(result.is_err());
}

// ===== Enum tests =====

#[test]
fn test_model_name_values() {
    assert_eq!(ModelName::CBOW as i32, 1);
    assert_eq!(ModelName::SG as i32, 2);
    assert_eq!(ModelName::SUP as i32, 3);
}

#[test]
fn test_loss_name_values() {
    assert_eq!(LossName::HS as i32, 1);
    assert_eq!(LossName::NS as i32, 2);
    assert_eq!(LossName::SOFTMAX as i32, 3);
    assert_eq!(LossName::OVA as i32, 4);
}

#[test]
fn test_model_name_from_i32() {
    assert_eq!(ModelName::from_i32(1), Some(ModelName::CBOW));
    assert_eq!(ModelName::from_i32(2), Some(ModelName::SG));
    assert_eq!(ModelName::from_i32(3), Some(ModelName::SUP));
    assert_eq!(ModelName::from_i32(0), None);
    assert_eq!(ModelName::from_i32(4), None);
    assert_eq!(ModelName::from_i32(-1), None);
}

#[test]
fn test_loss_name_from_i32() {
    assert_eq!(LossName::from_i32(1), Some(LossName::HS));
    assert_eq!(LossName::from_i32(2), Some(LossName::NS));
    assert_eq!(LossName::from_i32(3), Some(LossName::SOFTMAX));
    assert_eq!(LossName::from_i32(4), Some(LossName::OVA));
    assert_eq!(LossName::from_i32(0), None);
    assert_eq!(LossName::from_i32(5), None);
    assert_eq!(LossName::from_i32(-1), None);
}

// ===== String conversion tests =====

#[test]
fn test_loss_to_string() {
    let mut args = Args::default();

    args.loss = LossName::HS;
    assert_eq!(args.loss_to_string(), "hs");

    args.loss = LossName::NS;
    assert_eq!(args.loss_to_string(), "ns");

    args.loss = LossName::SOFTMAX;
    assert_eq!(args.loss_to_string(), "softmax");

    args.loss = LossName::OVA;
    assert_eq!(args.loss_to_string(), "one-vs-all");
}

#[test]
fn test_model_to_string() {
    let mut args = Args::default();

    args.model = ModelName::CBOW;
    assert_eq!(args.model_to_string(), "cbow");

    args.model = ModelName::SG;
    assert_eq!(args.model_to_string(), "sg");

    args.model = ModelName::SUP;
    assert_eq!(args.model_to_string(), "sup");
}

// ===== Autotune metric parsing =====

#[test]
fn test_autotune_metric_name_default() {
    let args = Args::default();
    assert_eq!(args.get_autotune_metric_name(), Some(MetricName::F1Score));
}

#[test]
fn test_autotune_metric_name_label_f1() {
    let mut args = Args::default();
    args.autotune_metric = "f1:cooking".to_string();
    assert_eq!(
        args.get_autotune_metric_name(),
        Some(MetricName::LabelF1Score)
    );
}

#[test]
fn test_autotune_metric_name_precision_at_recall() {
    let mut args = Args::default();
    args.autotune_metric = "precisionAtRecall:50".to_string();
    assert_eq!(
        args.get_autotune_metric_name(),
        Some(MetricName::PrecisionAtRecall)
    );
}

#[test]
fn test_autotune_metric_name_precision_at_recall_label() {
    let mut args = Args::default();
    args.autotune_metric = "precisionAtRecall:50:cooking".to_string();
    assert_eq!(
        args.get_autotune_metric_name(),
        Some(MetricName::PrecisionAtRecallLabel)
    );
}

#[test]
fn test_autotune_metric_name_recall_at_precision() {
    let mut args = Args::default();
    args.autotune_metric = "recallAtPrecision:50".to_string();
    assert_eq!(
        args.get_autotune_metric_name(),
        Some(MetricName::RecallAtPrecision)
    );
}

#[test]
fn test_autotune_metric_name_recall_at_precision_label() {
    let mut args = Args::default();
    args.autotune_metric = "recallAtPrecision:50:cooking".to_string();
    assert_eq!(
        args.get_autotune_metric_name(),
        Some(MetricName::RecallAtPrecisionLabel)
    );
}

#[test]
fn test_autotune_metric_name_unknown() {
    let mut args = Args::default();
    args.autotune_metric = "unknown_metric".to_string();
    assert_eq!(args.get_autotune_metric_name(), None);
}

// ===== Binary serialization: all enum values =====

#[test]
fn test_args_binary_all_loss_types() {
    for loss in &[LossName::HS, LossName::NS, LossName::SOFTMAX, LossName::OVA] {
        let mut args = Args::default();
        args.loss = *loss;

        let mut buf = Vec::new();
        args.save(&mut buf).unwrap();

        let mut args2 = Args::default();
        let mut cursor = Cursor::new(&buf);
        args2.load(&mut cursor).unwrap();

        assert_eq!(args2.loss, *loss);
    }
}

#[test]
fn test_args_binary_all_model_types() {
    for model in &[ModelName::CBOW, ModelName::SG, ModelName::SUP] {
        let mut args = Args::default();
        args.model = *model;

        let mut buf = Vec::new();
        args.save(&mut buf).unwrap();

        let mut args2 = Args::default();
        let mut cursor = Cursor::new(&buf);
        args2.load(&mut cursor).unwrap();

        assert_eq!(args2.model, *model);
    }
}

// ===== Binary serialization: fields not saved =====

#[test]
fn test_args_binary_does_not_save_non_serialized_fields() {
    // These fields are NOT part of the 56-byte binary block
    // They should retain their default values after load
    let mut args = Args::default();
    args.lr = 0.2; // not serialized
    args.verbose = 0; // not serialized
    args.label = "custom".to_string(); // not serialized

    let mut buf = Vec::new();
    args.save(&mut buf).unwrap();
    assert_eq!(buf.len(), 56); // still 56 bytes

    let mut args2 = Args::default();
    let mut cursor = Cursor::new(&buf);
    args2.load(&mut cursor).unwrap();

    // lr, verbose, label should still be at their Args::default() values in args2
    // since they are not part of the binary block
    assert!((args2.lr - 0.05).abs() < f64::EPSILON); // default, not 0.2
    assert_eq!(args2.verbose, 2); // default, not 0
    assert_eq!(args2.label, "__label__"); // default, not "custom"
}

// ===== Edge cases for binary serialization =====

#[test]
fn test_args_binary_extreme_values() {
    let mut args = Args::default();
    args.dim = i32::MAX;
    args.ws = i32::MIN;
    args.epoch = 0;
    args.min_count = -1;
    args.neg = 0;
    args.word_ngrams = 0;
    args.bucket = i32::MAX;
    args.minn = 0;
    args.maxn = 0;
    args.lr_update_rate = 0;
    args.t = f64::MIN_POSITIVE;

    let mut buf = Vec::new();
    args.save(&mut buf).unwrap();

    let mut args2 = Args::default();
    let mut cursor = Cursor::new(&buf);
    args2.load(&mut cursor).unwrap();

    assert_eq!(args2.dim, i32::MAX);
    assert_eq!(args2.ws, i32::MIN);
    assert_eq!(args2.epoch, 0);
    assert_eq!(args2.min_count, -1);
    assert_eq!(args2.neg, 0);
    assert_eq!(args2.word_ngrams, 0);
    assert_eq!(args2.bucket, i32::MAX);
    assert_eq!(args2.minn, 0);
    assert_eq!(args2.maxn, 0);
    assert_eq!(args2.lr_update_rate, 0);
    assert_eq!(args2.t, f64::MIN_POSITIVE);
}
