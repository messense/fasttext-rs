pub mod args;
pub mod autotune;
pub mod dictionary;
pub mod error;
pub mod fasttext;
pub mod loss;
pub mod matrix;
pub mod meter;
pub mod model;
pub mod product_quantizer;
pub mod quant_matrix;
pub mod utils;
pub mod vector;

// Re-export the most commonly used public types at the crate root.
pub use fasttext::{FastText, Prediction, TrainingHandle, FASTTEXT_FILEFORMAT_MAGIC_INT32, FASTTEXT_VERSION};
