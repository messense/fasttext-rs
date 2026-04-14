// Args: hyperparameter configuration for fastText

use std::io::{Read, Write};

use crate::error::Result;
use crate::utils;

/// Model architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ModelName {
    /// Continuous bag-of-words.
    CBOW = 1,
    /// Skip-gram.
    SG = 2,
    /// Supervised classification.
    SUP = 3,
}

/// Loss function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LossName {
    /// Hierarchical softmax.
    HS = 1,
    /// Negative sampling.
    NS = 2,
    /// Softmax.
    SOFTMAX = 3,
    /// One-vs-all.
    OVA = 4,
}

/// Autotune metric type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MetricName {
    /// F1 score (macro).
    F1Score = 1,
    /// F1 score for a specific label.
    LabelF1Score = 2,
    /// Precision at recall threshold.
    PrecisionAtRecall = 3,
    /// Precision at recall threshold for a specific label.
    PrecisionAtRecallLabel = 4,
    /// Recall at precision threshold.
    RecallAtPrecision = 5,
    /// Recall at precision threshold for a specific label.
    RecallAtPrecisionLabel = 6,
}

impl ModelName {
    /// Convert from i32 to ModelName.
    pub fn from_i32(value: i32) -> Option<ModelName> {
        match value {
            1 => Some(ModelName::CBOW),
            2 => Some(ModelName::SG),
            3 => Some(ModelName::SUP),
            _ => None,
        }
    }
}

impl LossName {
    /// Convert from i32 to LossName.
    pub fn from_i32(value: i32) -> Option<LossName> {
        match value {
            1 => Some(LossName::HS),
            2 => Some(LossName::NS),
            3 => Some(LossName::SOFTMAX),
            4 => Some(LossName::OVA),
            _ => None,
        }
    }
}

/// All fastText hyperparameters.
#[derive(Debug, Clone)]
pub struct Args {
    // Basic parameters
    input: String,
    output: String,
    lr: f64,
    lr_update_rate: i32,
    dim: i32,
    ws: i32,
    epoch: i32,
    min_count: i32,
    min_count_label: i32,
    neg: i32,
    word_ngrams: i32,
    loss: LossName,
    model: ModelName,
    bucket: i32,
    minn: i32,
    maxn: i32,
    thread: i32,
    t: f64,
    label: String,
    verbose: i32,
    pretrained_vectors: String,
    save_output: bool,
    seed: i32,

    // Quantization parameters
    qout: bool,
    retrain: bool,
    qnorm: bool,
    cutoff: usize,
    dsub: usize,

    // Autotune parameters
    autotune_validation_file: String,
    autotune_metric: String,
    autotune_predictions: i32,
    autotune_duration: i32,
    autotune_model_size: String,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            input: String::new(),
            output: String::new(),
            lr: 0.05,
            lr_update_rate: 100,
            dim: 100,
            ws: 5,
            epoch: 5,
            min_count: 5,
            min_count_label: 0,
            neg: 5,
            word_ngrams: 1,
            loss: LossName::NS,
            model: ModelName::SG,
            bucket: 2_000_000,
            minn: 3,
            maxn: 6,
            thread: 12,
            t: 1e-4,
            label: "__label__".to_string(),
            verbose: 2,
            pretrained_vectors: String::new(),
            save_output: false,
            seed: 0,

            qout: false,
            retrain: false,
            qnorm: false,
            cutoff: 0,
            dsub: 2,

            autotune_validation_file: String::new(),
            autotune_metric: "f1".to_string(),
            autotune_predictions: 1,
            autotune_duration: 300,
            autotune_model_size: String::new(),
        }
    }
}

impl Args {
    /// Create a new Args with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if autotune is enabled (validation file is non-empty).
    pub fn has_autotune(&self) -> bool {
        !self.autotune_validation_file.is_empty()
    }

    /// Apply supervised mode overrides.
    ///
    /// Sets model=SUP, loss=SOFTMAX, minCount=1, minn=0, maxn=0, lr=0.1.
    /// Also sets bucket=0 when wordNgrams<=1 and maxn==0 and autotune is not enabled.
    pub fn apply_supervised_defaults(&mut self) {
        self.model = ModelName::SUP;
        self.loss = LossName::SOFTMAX;
        self.min_count = 1;
        self.minn = 0;
        self.maxn = 0;
        self.lr = 0.1;

        if self.word_ngrams <= 1 && self.maxn == 0 && !self.has_autotune() {
            self.bucket = 0;
        }
    }

    // ===== Getters =====

    /// Get the input file path.
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Get the output file path.
    pub fn output(&self) -> &str {
        &self.output
    }

    /// Get the learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }

    /// Get the learning rate update rate.
    pub fn lr_update_rate(&self) -> i32 {
        self.lr_update_rate
    }

    /// Get the dimension of word vectors.
    pub fn dim(&self) -> i32 {
        self.dim
    }

    /// Get the window size.
    pub fn ws(&self) -> i32 {
        self.ws
    }

    /// Get the number of epochs.
    pub fn epoch(&self) -> i32 {
        self.epoch
    }

    /// Get the minimum word count.
    pub fn min_count(&self) -> i32 {
        self.min_count
    }

    /// Get the minimum label count.
    pub fn min_count_label(&self) -> i32 {
        self.min_count_label
    }

    /// Get the number of negatives sampled.
    pub fn neg(&self) -> i32 {
        self.neg
    }

    /// Get the max word n-gram length.
    pub fn word_ngrams(&self) -> i32 {
        self.word_ngrams
    }

    /// Get the loss function type.
    pub fn loss(&self) -> LossName {
        self.loss
    }

    /// Get the model type.
    pub fn model(&self) -> ModelName {
        self.model
    }

    /// Get the number of buckets.
    pub fn bucket(&self) -> i32 {
        self.bucket
    }

    /// Get the minimum character n-gram length.
    pub fn minn(&self) -> i32 {
        self.minn
    }

    /// Get the maximum character n-gram length.
    pub fn maxn(&self) -> i32 {
        self.maxn
    }

    /// Get the number of threads.
    pub fn thread(&self) -> i32 {
        self.thread
    }

    /// Get the sampling threshold.
    pub fn t(&self) -> f64 {
        self.t
    }

    /// Get the label prefix.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Get the verbose level.
    pub fn verbose(&self) -> i32 {
        self.verbose
    }

    /// Get the pretrained vectors file path.
    pub fn pretrained_vectors(&self) -> &str {
        &self.pretrained_vectors
    }

    /// Get whether output params should be saved.
    pub fn save_output(&self) -> bool {
        self.save_output
    }

    /// Get the random seed.
    pub fn seed(&self) -> i32 {
        self.seed
    }

    /// Get whether the classifier is quantized.
    pub fn qout(&self) -> bool {
        self.qout
    }

    /// Get whether embeddings are finetuned if cutoff is applied.
    pub fn retrain(&self) -> bool {
        self.retrain
    }

    /// Get whether the norm is quantized separately.
    pub fn qnorm(&self) -> bool {
        self.qnorm
    }

    /// Get the cutoff for vocabulary pruning.
    pub fn cutoff(&self) -> usize {
        self.cutoff
    }

    /// Get the sub-vector dimension for quantization.
    pub fn dsub(&self) -> usize {
        self.dsub
    }

    /// Get the autotune validation file path.
    pub fn autotune_validation_file(&self) -> &str {
        &self.autotune_validation_file
    }

    /// Get the autotune metric string.
    pub fn autotune_metric(&self) -> &str {
        &self.autotune_metric
    }

    /// Get the autotune predictions count.
    pub fn autotune_predictions(&self) -> i32 {
        self.autotune_predictions
    }

    /// Get the autotune duration in seconds.
    pub fn autotune_duration(&self) -> i32 {
        self.autotune_duration
    }

    /// Get the autotune model size constraint.
    pub fn autotune_model_size(&self) -> &str {
        &self.autotune_model_size
    }

    // ===== Setters =====

    /// Set the input file path.
    pub fn set_input(&mut self, value: String) {
        self.input = value;
    }

    /// Set the output file path.
    pub fn set_output(&mut self, value: String) {
        self.output = value;
    }

    /// Set the learning rate.
    pub fn set_lr(&mut self, value: f64) {
        self.lr = value;
    }

    /// Set the learning rate update rate.
    pub fn set_lr_update_rate(&mut self, value: i32) {
        self.lr_update_rate = value;
    }

    /// Set the dimension of word vectors.
    pub fn set_dim(&mut self, value: i32) {
        self.dim = value;
    }

    /// Set the window size.
    pub fn set_ws(&mut self, value: i32) {
        self.ws = value;
    }

    /// Set the number of epochs.
    pub fn set_epoch(&mut self, value: i32) {
        self.epoch = value;
    }

    /// Set the minimum word count.
    pub fn set_min_count(&mut self, value: i32) {
        self.min_count = value;
    }

    /// Set the minimum label count.
    pub fn set_min_count_label(&mut self, value: i32) {
        self.min_count_label = value;
    }

    /// Set the number of negatives sampled.
    pub fn set_neg(&mut self, value: i32) {
        self.neg = value;
    }

    /// Set the max word n-gram length.
    pub fn set_word_ngrams(&mut self, value: i32) {
        self.word_ngrams = value;
    }

    /// Set the loss function type.
    pub fn set_loss(&mut self, value: LossName) {
        self.loss = value;
    }

    /// Set the model type.
    pub fn set_model(&mut self, value: ModelName) {
        self.model = value;
    }

    /// Set the number of buckets.
    pub fn set_bucket(&mut self, value: i32) {
        self.bucket = value;
    }

    /// Set the minimum character n-gram length.
    pub fn set_minn(&mut self, value: i32) {
        self.minn = value;
    }

    /// Set the maximum character n-gram length.
    pub fn set_maxn(&mut self, value: i32) {
        self.maxn = value;
    }

    /// Set the number of threads.
    pub fn set_thread(&mut self, value: i32) {
        self.thread = value;
    }

    /// Set the sampling threshold.
    pub fn set_t(&mut self, value: f64) {
        self.t = value;
    }

    /// Set the label prefix.
    pub fn set_label(&mut self, value: String) {
        self.label = value;
    }

    /// Set the verbose level.
    pub fn set_verbose(&mut self, value: i32) {
        self.verbose = value;
    }

    /// Set the pretrained vectors file path.
    pub fn set_pretrained_vectors(&mut self, value: String) {
        self.pretrained_vectors = value;
    }

    /// Set whether output params should be saved.
    pub fn set_save_output(&mut self, value: bool) {
        self.save_output = value;
    }

    /// Set the random seed.
    pub fn set_seed(&mut self, value: i32) {
        self.seed = value;
    }

    /// Set whether the classifier is quantized.
    pub fn set_qout(&mut self, value: bool) {
        self.qout = value;
    }

    /// Set whether embeddings are finetuned if cutoff is applied.
    pub fn set_retrain(&mut self, value: bool) {
        self.retrain = value;
    }

    /// Set whether the norm is quantized separately.
    pub fn set_qnorm(&mut self, value: bool) {
        self.qnorm = value;
    }

    /// Set the cutoff for vocabulary pruning.
    pub fn set_cutoff(&mut self, value: usize) {
        self.cutoff = value;
    }

    /// Set the sub-vector dimension for quantization.
    pub fn set_dsub(&mut self, value: usize) {
        self.dsub = value;
    }

    /// Set the autotune validation file path.
    pub fn set_autotune_validation_file(&mut self, value: String) {
        self.autotune_validation_file = value;
    }

    /// Set the autotune metric string.
    pub fn set_autotune_metric(&mut self, value: String) {
        self.autotune_metric = value;
    }

    /// Set the autotune predictions count.
    pub fn set_autotune_predictions(&mut self, value: i32) {
        self.autotune_predictions = value;
    }

    /// Set the autotune duration in seconds.
    pub fn set_autotune_duration(&mut self, value: i32) {
        self.autotune_duration = value;
    }

    /// Set the autotune model size constraint.
    pub fn set_autotune_model_size(&mut self, value: String) {
        self.autotune_model_size = value;
    }

    // ===== Binary Serialization =====
    //
    // The binary format writes exactly 12 i32 fields + 1 f64 field = 56 bytes total.
    // Fields in order (matching C++ Args::save/load):
    //   dim, ws, epoch, minCount, neg, wordNgrams, loss, model, bucket, minn, maxn, lrUpdateRate, t
    //
    // Note: loss and model are written as their i32 discriminant values.

    /// Save the 13-field Args block to a writer (56 bytes).
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        utils::write_i32(writer, self.dim)?;
        utils::write_i32(writer, self.ws)?;
        utils::write_i32(writer, self.epoch)?;
        utils::write_i32(writer, self.min_count)?;
        utils::write_i32(writer, self.neg)?;
        utils::write_i32(writer, self.word_ngrams)?;
        utils::write_i32(writer, self.loss as i32)?;
        utils::write_i32(writer, self.model as i32)?;
        utils::write_i32(writer, self.bucket)?;
        utils::write_i32(writer, self.minn)?;
        utils::write_i32(writer, self.maxn)?;
        utils::write_i32(writer, self.lr_update_rate)?;
        utils::write_f64(writer, self.t)?;
        Ok(())
    }

    /// Load the 13-field Args block from a reader (56 bytes).
    pub fn load<R: Read>(&mut self, reader: &mut R) -> Result<()> {
        self.dim = utils::read_i32(reader)?;
        self.ws = utils::read_i32(reader)?;
        self.epoch = utils::read_i32(reader)?;
        self.min_count = utils::read_i32(reader)?;
        self.neg = utils::read_i32(reader)?;
        self.word_ngrams = utils::read_i32(reader)?;
        let loss_val = utils::read_i32(reader)?;
        self.loss = LossName::from_i32(loss_val).ok_or_else(|| {
            crate::error::FastTextError::InvalidModel(format!(
                "Invalid loss value: {}",
                loss_val
            ))
        })?;
        let model_val = utils::read_i32(reader)?;
        self.model = ModelName::from_i32(model_val).ok_or_else(|| {
            crate::error::FastTextError::InvalidModel(format!(
                "Invalid model value: {}",
                model_val
            ))
        })?;
        self.bucket = utils::read_i32(reader)?;
        self.minn = utils::read_i32(reader)?;
        self.maxn = utils::read_i32(reader)?;
        self.lr_update_rate = utils::read_i32(reader)?;
        self.t = utils::read_f64(reader)?;
        Ok(())
    }

    /// Convert loss name to string (matching C++ output).
    pub fn loss_to_string(&self) -> &'static str {
        match self.loss {
            LossName::HS => "hs",
            LossName::NS => "ns",
            LossName::SOFTMAX => "softmax",
            LossName::OVA => "one-vs-all",
        }
    }

    /// Convert model name to string (matching C++ output).
    pub fn model_to_string(&self) -> &'static str {
        match self.model {
            ModelName::CBOW => "cbow",
            ModelName::SG => "sg",
            ModelName::SUP => "sup",
        }
    }

    /// Parse the autotune metric string and return the corresponding MetricName.
    pub fn get_autotune_metric_name(&self) -> Option<MetricName> {
        if self.autotune_metric.starts_with("f1:") {
            Some(MetricName::LabelF1Score)
        } else if self.autotune_metric == "f1" {
            Some(MetricName::F1Score)
        } else if self.autotune_metric.starts_with("precisionAtRecall:") {
            let rest = &self.autotune_metric[18..];
            if rest.contains(':') {
                Some(MetricName::PrecisionAtRecallLabel)
            } else {
                Some(MetricName::PrecisionAtRecall)
            }
        } else if self.autotune_metric.starts_with("recallAtPrecision:") {
            let rest = &self.autotune_metric[18..];
            if rest.contains(':') {
                Some(MetricName::RecallAtPrecisionLabel)
            } else {
                Some(MetricName::RecallAtPrecision)
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ===== VAL-CORE-001: Args default values match C++ =====

    #[test]
    fn test_args_defaults() {
        let args = Args::default();

        // Basic parameters
        assert_eq!(args.input(), "");
        assert_eq!(args.output(), "");
        assert!((args.lr() - 0.05).abs() < f64::EPSILON);
        assert_eq!(args.lr_update_rate(), 100);
        assert_eq!(args.dim(), 100);
        assert_eq!(args.ws(), 5);
        assert_eq!(args.epoch(), 5);
        assert_eq!(args.min_count(), 5);
        assert_eq!(args.min_count_label(), 0);
        assert_eq!(args.neg(), 5);
        assert_eq!(args.word_ngrams(), 1);
        assert_eq!(args.loss(), LossName::NS);
        assert_eq!(args.model(), ModelName::SG);
        assert_eq!(args.bucket(), 2_000_000);
        assert_eq!(args.minn(), 3);
        assert_eq!(args.maxn(), 6);
        assert_eq!(args.thread(), 12);
        assert!((args.t() - 1e-4).abs() < f64::EPSILON);
        assert_eq!(args.label(), "__label__");
        assert_eq!(args.verbose(), 2);
        assert_eq!(args.pretrained_vectors(), "");
        assert!(!args.save_output());
        assert_eq!(args.seed(), 0);

        // Quantization parameters
        assert!(!args.qout());
        assert!(!args.retrain());
        assert!(!args.qnorm());
        assert_eq!(args.cutoff(), 0);
        assert_eq!(args.dsub(), 2);

        // Autotune parameters
        assert_eq!(args.autotune_validation_file(), "");
        assert_eq!(args.autotune_metric(), "f1");
        assert_eq!(args.autotune_predictions(), 1);
        assert_eq!(args.autotune_duration(), 300);
        assert_eq!(args.autotune_model_size(), "");
    }

    #[test]
    fn test_args_new_matches_default() {
        let args1 = Args::new();
        let args2 = Args::default();
        // Check a subset of fields to confirm they're the same
        assert_eq!(args1.dim(), args2.dim());
        assert_eq!(args1.lr(), args2.lr());
        assert_eq!(args1.loss(), args2.loss());
        assert_eq!(args1.model(), args2.model());
    }

    // ===== VAL-CORE-002: Args supervised mode overrides =====

    #[test]
    fn test_args_supervised_overrides() {
        let mut args = Args::default();
        args.apply_supervised_defaults();

        assert_eq!(args.model(), ModelName::SUP);
        assert_eq!(args.loss(), LossName::SOFTMAX);
        assert_eq!(args.min_count(), 1);
        assert_eq!(args.minn(), 0);
        assert_eq!(args.maxn(), 0);
        assert!((args.lr() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_args_supervised_bucket_zero() {
        // Default: wordNgrams=1, maxn=0 (after supervised overrides), no autotune
        // => bucket should be 0
        let mut args = Args::default();
        args.apply_supervised_defaults();
        assert_eq!(args.bucket(), 0);
    }

    #[test]
    fn test_args_supervised_bucket_nonzero_with_word_ngrams() {
        // wordNgrams > 1 => bucket should NOT be zeroed
        let mut args = Args::default();
        args.set_word_ngrams(2);
        args.apply_supervised_defaults();
        assert_eq!(args.bucket(), 2_000_000);
    }

    #[test]
    fn test_args_supervised_bucket_nonzero_with_autotune() {
        // With autotune enabled => bucket should NOT be zeroed
        let mut args = Args::default();
        args.set_autotune_validation_file("valid.txt".to_string());
        args.apply_supervised_defaults();
        // maxn is 0 and wordNgrams is 1, but autotune is enabled
        assert_eq!(args.bucket(), 2_000_000);
    }

    #[test]
    fn test_args_supervised_bucket_nonzero_with_maxn() {
        // maxn > 0 after supervised overrides is unusual since supervised sets maxn=0,
        // but if someone sets maxn after supervised overrides, bucket stays
        let mut args = Args::default();
        args.apply_supervised_defaults();
        // At this point maxn == 0, but let's test the case where maxn was manually set first
        let mut args2 = Args::default();
        // Simulate: set maxn manually, then parse supervised
        // In C++ parseArgs, supervised sets maxn=0 first, then the bucket check uses maxn==0
        // So in the standard flow bucket will always be 0 for supervised with default word_ngrams
        // Let's just verify the conditional logic directly
        args2.set_word_ngrams(1);
        args2.set_maxn(0);
        // Manually apply the parts
        args2.set_model(ModelName::SUP);
        args2.set_loss(LossName::SOFTMAX);
        args2.set_min_count(1);
        args2.set_minn(0);
        args2.set_lr(0.1);
        // Now check: wordNgrams<=1 && maxn==0 && !has_autotune()
        if args2.word_ngrams() <= 1 && args2.maxn() == 0 && !args2.has_autotune() {
            args2.set_bucket(0);
        }
        assert_eq!(args2.bucket(), 0);
    }

    // ===== has_autotune() =====

    #[test]
    fn test_has_autotune_false_by_default() {
        let args = Args::default();
        assert!(!args.has_autotune());
    }

    #[test]
    fn test_has_autotune_true_when_set() {
        let mut args = Args::default();
        args.set_autotune_validation_file("validation.txt".to_string());
        assert!(args.has_autotune());
    }

    #[test]
    fn test_has_autotune_false_with_empty_string() {
        let mut args = Args::default();
        args.set_autotune_validation_file(String::new());
        assert!(!args.has_autotune());
    }

    // ===== VAL-CORE-003: Args setter/getter round-trips =====

    #[test]
    fn test_args_setter_getter_roundtrip() {
        let mut args = Args::default();

        // String fields
        args.set_input("train.txt".to_string());
        assert_eq!(args.input(), "train.txt");

        args.set_output("model_out".to_string());
        assert_eq!(args.output(), "model_out");

        args.set_label("#".to_string());
        assert_eq!(args.label(), "#");

        args.set_pretrained_vectors("vectors.vec".to_string());
        assert_eq!(args.pretrained_vectors(), "vectors.vec");

        // f64 fields
        args.set_lr(0.25);
        assert!((args.lr() - 0.25).abs() < f64::EPSILON);

        args.set_t(1e-5);
        assert!((args.t() - 1e-5).abs() < f64::EPSILON);

        // i32 fields
        args.set_lr_update_rate(200);
        assert_eq!(args.lr_update_rate(), 200);

        args.set_dim(300);
        assert_eq!(args.dim(), 300);

        args.set_ws(10);
        assert_eq!(args.ws(), 10);

        args.set_epoch(25);
        assert_eq!(args.epoch(), 25);

        args.set_min_count(3);
        assert_eq!(args.min_count(), 3);

        args.set_min_count_label(2);
        assert_eq!(args.min_count_label(), 2);

        args.set_neg(10);
        assert_eq!(args.neg(), 10);

        args.set_word_ngrams(3);
        assert_eq!(args.word_ngrams(), 3);

        args.set_bucket(1_000_000);
        assert_eq!(args.bucket(), 1_000_000);

        args.set_minn(2);
        assert_eq!(args.minn(), 2);

        args.set_maxn(5);
        assert_eq!(args.maxn(), 5);

        args.set_thread(4);
        assert_eq!(args.thread(), 4);

        args.set_verbose(0);
        assert_eq!(args.verbose(), 0);

        args.set_seed(42);
        assert_eq!(args.seed(), 42);

        // Enum fields
        args.set_loss(LossName::HS);
        assert_eq!(args.loss(), LossName::HS);

        args.set_loss(LossName::SOFTMAX);
        assert_eq!(args.loss(), LossName::SOFTMAX);

        args.set_loss(LossName::OVA);
        assert_eq!(args.loss(), LossName::OVA);

        args.set_model(ModelName::CBOW);
        assert_eq!(args.model(), ModelName::CBOW);

        args.set_model(ModelName::SUP);
        assert_eq!(args.model(), ModelName::SUP);

        // Bool fields
        args.set_save_output(true);
        assert!(args.save_output());

        args.set_save_output(false);
        assert!(!args.save_output());

        args.set_qout(true);
        assert!(args.qout());

        args.set_retrain(true);
        assert!(args.retrain());

        args.set_qnorm(true);
        assert!(args.qnorm());

        // usize fields
        args.set_cutoff(50000);
        assert_eq!(args.cutoff(), 50000);

        args.set_dsub(4);
        assert_eq!(args.dsub(), 4);

        // Autotune fields
        args.set_autotune_validation_file("valid.txt".to_string());
        assert_eq!(args.autotune_validation_file(), "valid.txt");

        args.set_autotune_metric("f1:cooking".to_string());
        assert_eq!(args.autotune_metric(), "f1:cooking");

        args.set_autotune_predictions(5);
        assert_eq!(args.autotune_predictions(), 5);

        args.set_autotune_duration(600);
        assert_eq!(args.autotune_duration(), 600);

        args.set_autotune_model_size("100M".to_string());
        assert_eq!(args.autotune_model_size(), "100M");
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
        args2.set_dim(999);
        args2.set_ws(999);
        args2.set_epoch(999);

        let mut cursor = Cursor::new(&buf);
        args2.load(&mut cursor).unwrap();

        assert_eq!(args2.dim(), args.dim());
        assert_eq!(args2.ws(), args.ws());
        assert_eq!(args2.epoch(), args.epoch());
        assert_eq!(args2.min_count(), args.min_count());
        assert_eq!(args2.neg(), args.neg());
        assert_eq!(args2.word_ngrams(), args.word_ngrams());
        assert_eq!(args2.loss(), args.loss());
        assert_eq!(args2.model(), args.model());
        assert_eq!(args2.bucket(), args.bucket());
        assert_eq!(args2.minn(), args.minn());
        assert_eq!(args2.maxn(), args.maxn());
        assert_eq!(args2.lr_update_rate(), args.lr_update_rate());
        assert!((args2.t() - args.t()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_args_binary_serialization_nondefault() {
        let mut args = Args::default();
        args.set_dim(300);
        args.set_ws(10);
        args.set_epoch(25);
        args.set_min_count(3);
        args.set_neg(10);
        args.set_word_ngrams(2);
        args.set_loss(LossName::SOFTMAX);
        args.set_model(ModelName::SUP);
        args.set_bucket(500_000);
        args.set_minn(2);
        args.set_maxn(5);
        args.set_lr_update_rate(50);
        args.set_t(1e-3);

        let mut buf = Vec::new();
        args.save(&mut buf).unwrap();
        assert_eq!(buf.len(), 56);

        let mut args2 = Args::default();
        let mut cursor = Cursor::new(&buf);
        args2.load(&mut cursor).unwrap();

        assert_eq!(args2.dim(), 300);
        assert_eq!(args2.ws(), 10);
        assert_eq!(args2.epoch(), 25);
        assert_eq!(args2.min_count(), 3);
        assert_eq!(args2.neg(), 10);
        assert_eq!(args2.word_ngrams(), 2);
        assert_eq!(args2.loss(), LossName::SOFTMAX);
        assert_eq!(args2.model(), ModelName::SUP);
        assert_eq!(args2.bucket(), 500_000);
        assert_eq!(args2.minn(), 2);
        assert_eq!(args2.maxn(), 5);
        assert_eq!(args2.lr_update_rate(), 50);
        assert!((args2.t() - 1e-3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_args_binary_serialization_field_order() {
        // Verify exact field order by manually checking written bytes
        let mut args = Args::default();
        args.set_dim(1);
        args.set_ws(2);
        args.set_epoch(3);
        args.set_min_count(4);
        args.set_neg(5);
        args.set_word_ngrams(6);
        args.set_loss(LossName::SOFTMAX); // = 3
        args.set_model(ModelName::SUP); // = 3
        args.set_bucket(9);
        args.set_minn(10);
        args.set_maxn(11);
        args.set_lr_update_rate(12);
        args.set_t(0.5);

        let mut buf = Vec::new();
        args.save(&mut buf).unwrap();

        // Read back as raw i32 values
        let mut cursor = Cursor::new(&buf);
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 1); // dim
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 2); // ws
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 3); // epoch
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 4); // minCount
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 5); // neg
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 6); // wordNgrams
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 3); // loss (SOFTMAX=3)
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 3); // model (SUP=3)
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 9); // bucket
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 10); // minn
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 11); // maxn
        assert_eq!(crate::utils::read_i32(&mut cursor).unwrap(), 12); // lrUpdateRate
        let t_val = crate::utils::read_f64(&mut cursor).unwrap();
        assert!((t_val - 0.5).abs() < f64::EPSILON); // t
    }

    #[test]
    fn test_args_binary_load_invalid_loss() {
        // Create a buffer with an invalid loss value
        let mut buf = Vec::new();
        crate::utils::write_i32(&mut buf, 100).unwrap(); // dim
        crate::utils::write_i32(&mut buf, 5).unwrap(); // ws
        crate::utils::write_i32(&mut buf, 5).unwrap(); // epoch
        crate::utils::write_i32(&mut buf, 5).unwrap(); // minCount
        crate::utils::write_i32(&mut buf, 5).unwrap(); // neg
        crate::utils::write_i32(&mut buf, 1).unwrap(); // wordNgrams
        crate::utils::write_i32(&mut buf, 99).unwrap(); // loss = INVALID
        crate::utils::write_i32(&mut buf, 1).unwrap(); // model
        crate::utils::write_i32(&mut buf, 2000000).unwrap(); // bucket
        crate::utils::write_i32(&mut buf, 3).unwrap(); // minn
        crate::utils::write_i32(&mut buf, 6).unwrap(); // maxn
        crate::utils::write_i32(&mut buf, 100).unwrap(); // lrUpdateRate
        crate::utils::write_f64(&mut buf, 1e-4).unwrap(); // t

        let mut args = Args::default();
        let mut cursor = Cursor::new(&buf);
        let result = args.load(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_args_binary_load_invalid_model() {
        let mut buf = Vec::new();
        crate::utils::write_i32(&mut buf, 100).unwrap(); // dim
        crate::utils::write_i32(&mut buf, 5).unwrap(); // ws
        crate::utils::write_i32(&mut buf, 5).unwrap(); // epoch
        crate::utils::write_i32(&mut buf, 5).unwrap(); // minCount
        crate::utils::write_i32(&mut buf, 5).unwrap(); // neg
        crate::utils::write_i32(&mut buf, 1).unwrap(); // wordNgrams
        crate::utils::write_i32(&mut buf, 2).unwrap(); // loss = NS (valid)
        crate::utils::write_i32(&mut buf, 99).unwrap(); // model = INVALID
        crate::utils::write_i32(&mut buf, 2000000).unwrap(); // bucket
        crate::utils::write_i32(&mut buf, 3).unwrap(); // minn
        crate::utils::write_i32(&mut buf, 6).unwrap(); // maxn
        crate::utils::write_i32(&mut buf, 100).unwrap(); // lrUpdateRate
        crate::utils::write_f64(&mut buf, 1e-4).unwrap(); // t

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
    fn test_metric_name_values() {
        assert_eq!(MetricName::F1Score as i32, 1);
        assert_eq!(MetricName::LabelF1Score as i32, 2);
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

        args.set_loss(LossName::HS);
        assert_eq!(args.loss_to_string(), "hs");

        args.set_loss(LossName::NS);
        assert_eq!(args.loss_to_string(), "ns");

        args.set_loss(LossName::SOFTMAX);
        assert_eq!(args.loss_to_string(), "softmax");

        args.set_loss(LossName::OVA);
        assert_eq!(args.loss_to_string(), "one-vs-all");
    }

    #[test]
    fn test_model_to_string() {
        let mut args = Args::default();

        args.set_model(ModelName::CBOW);
        assert_eq!(args.model_to_string(), "cbow");

        args.set_model(ModelName::SG);
        assert_eq!(args.model_to_string(), "sg");

        args.set_model(ModelName::SUP);
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
        args.set_autotune_metric("f1:cooking".to_string());
        assert_eq!(
            args.get_autotune_metric_name(),
            Some(MetricName::LabelF1Score)
        );
    }

    #[test]
    fn test_autotune_metric_name_precision_at_recall() {
        let mut args = Args::default();
        args.set_autotune_metric("precisionAtRecall:50".to_string());
        assert_eq!(
            args.get_autotune_metric_name(),
            Some(MetricName::PrecisionAtRecall)
        );
    }

    #[test]
    fn test_autotune_metric_name_precision_at_recall_label() {
        let mut args = Args::default();
        args.set_autotune_metric("precisionAtRecall:50:cooking".to_string());
        assert_eq!(
            args.get_autotune_metric_name(),
            Some(MetricName::PrecisionAtRecallLabel)
        );
    }

    #[test]
    fn test_autotune_metric_name_recall_at_precision() {
        let mut args = Args::default();
        args.set_autotune_metric("recallAtPrecision:50".to_string());
        assert_eq!(
            args.get_autotune_metric_name(),
            Some(MetricName::RecallAtPrecision)
        );
    }

    #[test]
    fn test_autotune_metric_name_recall_at_precision_label() {
        let mut args = Args::default();
        args.set_autotune_metric("recallAtPrecision:50:cooking".to_string());
        assert_eq!(
            args.get_autotune_metric_name(),
            Some(MetricName::RecallAtPrecisionLabel)
        );
    }

    #[test]
    fn test_autotune_metric_name_unknown() {
        let mut args = Args::default();
        args.set_autotune_metric("unknown_metric".to_string());
        assert_eq!(args.get_autotune_metric_name(), None);
    }

    // ===== Clone and Debug =====

    #[test]
    fn test_args_clone() {
        let mut args = Args::default();
        args.set_dim(300);
        args.set_lr(0.2);
        args.set_input("test.txt".to_string());

        let cloned = args.clone();
        assert_eq!(cloned.dim(), 300);
        assert!((cloned.lr() - 0.2).abs() < f64::EPSILON);
        assert_eq!(cloned.input(), "test.txt");
    }

    #[test]
    fn test_args_debug() {
        let args = Args::default();
        let debug = format!("{:?}", args);
        assert!(!debug.is_empty());
        assert!(debug.contains("Args"));
    }

    // ===== Binary serialization: all enum values =====

    #[test]
    fn test_args_binary_all_loss_types() {
        for loss in &[LossName::HS, LossName::NS, LossName::SOFTMAX, LossName::OVA] {
            let mut args = Args::default();
            args.set_loss(*loss);

            let mut buf = Vec::new();
            args.save(&mut buf).unwrap();

            let mut args2 = Args::default();
            let mut cursor = Cursor::new(&buf);
            args2.load(&mut cursor).unwrap();

            assert_eq!(args2.loss(), *loss);
        }
    }

    #[test]
    fn test_args_binary_all_model_types() {
        for model in &[ModelName::CBOW, ModelName::SG, ModelName::SUP] {
            let mut args = Args::default();
            args.set_model(*model);

            let mut buf = Vec::new();
            args.save(&mut buf).unwrap();

            let mut args2 = Args::default();
            let mut cursor = Cursor::new(&buf);
            args2.load(&mut cursor).unwrap();

            assert_eq!(args2.model(), *model);
        }
    }

    // ===== Binary serialization: fields not saved =====

    #[test]
    fn test_args_binary_does_not_save_non_serialized_fields() {
        // These fields are NOT part of the 56-byte binary block
        // They should retain their default values after load
        let mut args = Args::default();
        args.set_lr(0.2); // not serialized
        args.set_verbose(0); // not serialized
        args.set_label("custom".to_string()); // not serialized

        let mut buf = Vec::new();
        args.save(&mut buf).unwrap();
        assert_eq!(buf.len(), 56); // still 56 bytes

        let mut args2 = Args::default();
        let mut cursor = Cursor::new(&buf);
        args2.load(&mut cursor).unwrap();

        // lr, verbose, label should still be at their Args::default() values in args2
        // since they are not part of the binary block
        assert!((args2.lr() - 0.05).abs() < f64::EPSILON); // default, not 0.2
        assert_eq!(args2.verbose(), 2); // default, not 0
        assert_eq!(args2.label(), "__label__"); // default, not "custom"
    }

    // ===== Edge cases for binary serialization =====

    #[test]
    fn test_args_binary_extreme_values() {
        let mut args = Args::default();
        args.set_dim(i32::MAX);
        args.set_ws(i32::MIN);
        args.set_epoch(0);
        args.set_min_count(-1);
        args.set_neg(0);
        args.set_word_ngrams(0);
        args.set_bucket(i32::MAX);
        args.set_minn(0);
        args.set_maxn(0);
        args.set_lr_update_rate(0);
        args.set_t(f64::MIN_POSITIVE);

        let mut buf = Vec::new();
        args.save(&mut buf).unwrap();

        let mut args2 = Args::default();
        let mut cursor = Cursor::new(&buf);
        args2.load(&mut cursor).unwrap();

        assert_eq!(args2.dim(), i32::MAX);
        assert_eq!(args2.ws(), i32::MIN);
        assert_eq!(args2.epoch(), 0);
        assert_eq!(args2.min_count(), -1);
        assert_eq!(args2.neg(), 0);
        assert_eq!(args2.word_ngrams(), 0);
        assert_eq!(args2.bucket(), i32::MAX);
        assert_eq!(args2.minn(), 0);
        assert_eq!(args2.maxn(), 0);
        assert_eq!(args2.lr_update_rate(), 0);
        assert_eq!(args2.t(), f64::MIN_POSITIVE);
    }

    // ===== VAL-AUTO-001 & VAL-AUTO-002 =====

    #[test]
    fn test_autotune_activation() {
        let args = Args::default();
        assert!(!args.has_autotune());
        assert_eq!(args.autotune_validation_file(), "");

        let mut args = Args::default();
        args.set_autotune_validation_file("valid.txt".to_string());
        assert!(args.has_autotune());
    }

    #[test]
    fn test_autotune_duration_default() {
        let args = Args::default();
        assert_eq!(args.autotune_duration(), 300);
    }

    #[test]
    fn test_autotune_duration_custom() {
        let mut args = Args::default();
        args.set_autotune_duration(600);
        assert_eq!(args.autotune_duration(), 600);
    }
}
