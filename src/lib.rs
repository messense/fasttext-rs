pub mod args;
pub mod autotune;
pub mod dictionary;
pub mod error;
pub mod fasttext;
pub(crate) mod loss;
pub mod matrix;
pub mod meter;
pub(crate) mod model;
pub(crate) mod product_quantizer;
pub(crate) mod quant_matrix;
pub mod utils;
pub(crate) mod vector;

pub use fasttext::{FastText, Prediction, TrainingHandle, FASTTEXT_FILEFORMAT_MAGIC_INT32, FASTTEXT_VERSION};
