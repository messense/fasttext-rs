// Args: hyperparameter configuration for fastText

use std::convert::TryFrom;
use std::fmt;
use std::io::{Read, Write};
use std::path::PathBuf;

use crate::error::Result;
use crate::utils;

/// Model architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ModelName {
    /// Continuous bag-of-words.
    Cbow = 1,
    /// Skip-gram.
    SkipGram = 2,
    /// Supervised classification.
    Supervised = 3,
}

/// Loss function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LossName {
    /// Hierarchical softmax.
    HierarchicalSoftmax = 1,
    /// Negative sampling.
    NegativeSampling = 2,
    /// Softmax.
    Softmax = 3,
    /// One-vs-all.
    OneVsAll = 4,
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

impl TryFrom<i32> for ModelName {
    type Error = i32;

    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            1 => Ok(ModelName::Cbow),
            2 => Ok(ModelName::SkipGram),
            3 => Ok(ModelName::Supervised),
            _ => Err(value),
        }
    }
}

impl fmt::Display for ModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelName::Cbow => write!(f, "cbow"),
            ModelName::SkipGram => write!(f, "sg"),
            ModelName::Supervised => write!(f, "sup"),
        }
    }
}

impl TryFrom<i32> for LossName {
    type Error = i32;

    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            1 => Ok(LossName::HierarchicalSoftmax),
            2 => Ok(LossName::NegativeSampling),
            3 => Ok(LossName::Softmax),
            4 => Ok(LossName::OneVsAll),
            _ => Err(value),
        }
    }
}

impl fmt::Display for LossName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LossName::HierarchicalSoftmax => write!(f, "hs"),
            LossName::NegativeSampling => write!(f, "ns"),
            LossName::Softmax => write!(f, "softmax"),
            LossName::OneVsAll => write!(f, "one-vs-all"),
        }
    }
}

/// All fastText hyperparameters.
#[derive(Debug, Clone)]
pub struct Args {
    pub input: PathBuf,
    pub output: PathBuf,
    pub lr: f64,
    pub lr_update_rate: i32,
    pub dim: i32,
    pub ws: i32,
    pub epoch: i32,
    pub min_count: i32,
    pub min_count_label: i32,
    pub neg: i32,
    pub word_ngrams: i32,
    pub loss: LossName,
    pub model: ModelName,
    pub bucket: i32,
    pub minn: i32,
    pub maxn: i32,
    pub thread: i32,
    pub t: f64,
    pub label: String,
    pub verbose: i32,
    pub pretrained_vectors: PathBuf,
    pub save_output: bool,
    pub seed: i32,
    pub qout: bool,
    pub retrain: bool,
    pub qnorm: bool,
    pub cutoff: usize,
    pub dsub: usize,
    pub autotune_validation_file: PathBuf,
    pub autotune_metric: String,
    pub autotune_predictions: i32,
    pub autotune_duration: i32,
    pub autotune_model_size: String,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            input: PathBuf::new(),
            output: PathBuf::new(),
            lr: 0.05,
            lr_update_rate: 100,
            dim: 100,
            ws: 5,
            epoch: 5,
            min_count: 5,
            min_count_label: 0,
            neg: 5,
            word_ngrams: 1,
            loss: LossName::NegativeSampling,
            model: ModelName::SkipGram,
            bucket: 2_000_000,
            minn: 3,
            maxn: 6,
            thread: 12,
            t: 1e-4,
            label: "__label__".to_string(),
            verbose: 2,
            pretrained_vectors: PathBuf::new(),
            save_output: false,
            seed: 0,

            qout: false,
            retrain: false,
            qnorm: false,
            cutoff: 0,
            dsub: 2,

            autotune_validation_file: PathBuf::new(),
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
        !self.autotune_validation_file.as_os_str().is_empty()
    }

    /// Apply supervised mode overrides.
    ///
    /// Sets model=Supervised, loss=Softmax, minCount=1, minn=0, maxn=0, lr=0.1.
    /// Also sets bucket=0 when wordNgrams<=1 and maxn==0 and autotune is not enabled.
    pub fn apply_supervised_defaults(&mut self) {
        self.model = ModelName::Supervised;
        self.loss = LossName::Softmax;
        self.min_count = 1;
        self.minn = 0;
        self.maxn = 0;
        self.lr = 0.1;

        if self.word_ngrams <= 1 && self.maxn == 0 && !self.has_autotune() {
            self.bucket = 0;
        }
    }

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
        self.loss = LossName::try_from(loss_val).map_err(|v| {
            crate::error::FastTextError::InvalidModel(format!("Invalid loss value: {}", v))
        })?;
        let model_val = utils::read_i32(reader)?;
        self.model = ModelName::try_from(model_val).map_err(|v| {
            crate::error::FastTextError::InvalidModel(format!("Invalid model value: {}", v))
        })?;
        self.bucket = utils::read_i32(reader)?;
        self.minn = utils::read_i32(reader)?;
        self.maxn = utils::read_i32(reader)?;
        self.lr_update_rate = utils::read_i32(reader)?;
        self.t = utils::read_f64(reader)?;
        Ok(())
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
