// FastText: train, predict, quantize, autotune, save/load, word/sentence vectors

use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;

use crate::args::{Args, ModelName};
use crate::dictionary::Dictionary;
use crate::error::{FastTextError, Result};
use crate::matrix::{DenseMatrix, Matrix};
use crate::utils;

/// Magic number identifying a valid fastText binary model file.
pub const FASTTEXT_FILEFORMAT_MAGIC_INT32: i32 = 793712314;
/// Current binary format version.
pub const FASTTEXT_VERSION: i32 = 12;

/// Read a boolean (1 byte) from a reader.
fn read_bool<R: Read>(reader: &mut R) -> Result<bool> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
}

/// Write a boolean (1 byte) to a writer.
fn write_bool<W: Write>(writer: &mut W, value: bool) -> Result<()> {
    writer.write_all(&[value as u8])?;
    Ok(())
}

/// A loaded fastText model.
///
/// Contains the model arguments, dictionary, input matrix, and output matrix.
/// Used for inference (prediction) and model inspection.
#[derive(Debug)]
pub struct FastText {
    /// Model hyperparameters.
    args: Arc<Args>,
    /// The vocabulary dictionary.
    dict: Dictionary,
    /// Input embedding matrix (word + subword vectors).
    input: DenseMatrix,
    /// Output matrix (label/word vectors).
    output: DenseMatrix,
    /// Whether the model uses quantized (QuantMatrix) input.
    quant: bool,
}

impl FastText {
    /// Load a model from a binary reader.
    ///
    /// Reads in this order:
    /// 1. Magic number (i32) – must be 793712314
    /// 2. Version (i32) – must be ≤ 12
    /// 3. Args block (56 bytes)
    /// 4. Dictionary block
    /// 5. quant_input (bool)
    /// 6. Input matrix (DenseMatrix if not quantized)
    /// 7. qout (bool → stored in args.qout)
    /// 8. Output matrix
    pub fn load<R: Read>(reader: &mut R) -> Result<Self> {
        // 1. Read and validate magic number
        let magic = utils::read_i32(reader)?;
        if magic != FASTTEXT_FILEFORMAT_MAGIC_INT32 {
            return Err(FastTextError::InvalidModel(format!(
                "Invalid magic number: {} (expected {})",
                magic, FASTTEXT_FILEFORMAT_MAGIC_INT32
            )));
        }

        // 2. Read and validate version
        let version = utils::read_i32(reader)?;
        if version > FASTTEXT_VERSION {
            return Err(FastTextError::InvalidModel(format!(
                "Unsupported version: {} (maximum supported: {})",
                version, FASTTEXT_VERSION
            )));
        }

        // 3. Read Args block
        let mut args = Args::default();
        args.load(reader)?;

        // Version 11 backward compatibility:
        // Old supervised models do not use character n-grams.
        if version == 11 && args.model() == ModelName::SUP {
            args.set_maxn(0);
        }

        // 4. Read Dictionary block
        let args_arc = Arc::new(args.clone());
        let dict = Dictionary::load_from_reader(reader, args_arc)?;

        // 5. Read quant_input flag
        let quant_input = read_bool(reader)?;

        // 6. Load input matrix
        let input = if !quant_input {
            DenseMatrix::load(reader)?
        } else {
            // QuantMatrix is deferred to a future feature.
            return Err(FastTextError::InvalidModel(
                "Quantized input matrix (.ftz) is not yet supported in this version".to_string(),
            ));
        };

        // C++ check: if not quantized but dict is pruned, reject
        if !quant_input && dict.is_pruned() {
            return Err(FastTextError::InvalidModel(
                "Invalid model file. Please download the updated model. \
                 See issue #332 on Github for more information."
                    .to_string(),
            ));
        }

        // 7. Read qout flag (stored in args)
        let qout = read_bool(reader)?;
        args.set_qout(qout);

        // 8. Load output matrix
        let output = if quant_input && qout {
            // QuantMatrix output – deferred
            return Err(FastTextError::InvalidModel(
                "Quantized output matrix (.ftz) is not yet supported".to_string(),
            ));
        } else {
            DenseMatrix::load(reader)?
        };

        Ok(FastText {
            args: Arc::new(args),
            dict,
            input,
            output,
            quant: quant_input,
        })
    }

    /// Load a model from a file path.
    pub fn load_model(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(FastTextError::IoError)?;
        let mut reader = BufReader::new(file);
        Self::load(&mut reader)
    }

    /// Save the model to a binary writer.
    ///
    /// Writes in the same order as `load()`:
    /// 1. Magic number (i32)
    /// 2. Version (i32)
    /// 3. Args block (56 bytes)
    /// 4. Dictionary block
    /// 5. quant_input (bool)
    /// 6. Input matrix
    /// 7. qout (bool)
    /// 8. Output matrix
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        // 1. Magic number
        utils::write_i32(writer, FASTTEXT_FILEFORMAT_MAGIC_INT32)?;
        // 2. Version
        utils::write_i32(writer, FASTTEXT_VERSION)?;
        // 3. Args block
        self.args.save(writer)?;
        // 4. Dictionary block
        self.dict.save(writer)?;
        // 5. quant_input
        write_bool(writer, self.quant)?;
        // 6. Input matrix
        self.input.save(writer)?;
        // 7. qout
        write_bool(writer, self.args.qout())?;
        // 8. Output matrix
        self.output.save(writer)?;
        Ok(())
    }

    /// Save the model to a file path.
    pub fn save_model(&self, path: &str) -> Result<()> {
        let file = std::fs::File::create(path).map_err(FastTextError::IoError)?;
        let mut writer = BufWriter::new(file);
        self.save(&mut writer)?;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Getters / accessors
    // -------------------------------------------------------------------------

    /// Get a reference to the model arguments.
    pub fn args(&self) -> &Args {
        &self.args
    }

    /// Get a reference to the dictionary.
    pub fn dict(&self) -> &Dictionary {
        &self.dict
    }

    /// Get a reference to the input matrix.
    pub fn input_matrix(&self) -> &DenseMatrix {
        &self.input
    }

    /// Get a reference to the output matrix.
    pub fn output_matrix(&self) -> &DenseMatrix {
        &self.output
    }

    /// Return whether the model is quantized.
    pub fn is_quant(&self) -> bool {
        self.quant
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use crate::args::{Args, LossName, ModelName};

    /// Path to the cooking reference model fixture.
    const COOKING_MODEL: &str = "tests/fixtures/cooking.model.bin";
    /// Path to the invalid model fixture.
    const INVALID_MODEL: &str = "tests/fixtures/invalid.model.bin";

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Build a minimal valid model binary for testing.
    ///
    /// Creates a small valid .bin file:
    /// - magic, version
    /// - args block (supervised, dim=2)
    /// - dict block (minimal, with </s> entry)
    /// - quant_input=false, 2x2 DenseMatrix (input)
    /// - qout=false, 1x2 DenseMatrix (output, 1 label)
    fn make_minimal_model_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic + version
        utils::write_i32(&mut buf, FASTTEXT_FILEFORMAT_MAGIC_INT32).unwrap();
        utils::write_i32(&mut buf, FASTTEXT_VERSION).unwrap();

        // Args: dim=2, ws=1, epoch=1, minCount=1, neg=5, wordNgrams=1,
        //        loss=SOFTMAX(3), model=SUP(3), bucket=0, minn=0, maxn=0,
        //        lrUpdateRate=100, t=0.0001
        let mut args = Args::default();
        args.set_dim(2);
        args.set_ws(1);
        args.set_epoch(1);
        args.set_min_count(1);
        args.set_neg(5);
        args.set_word_ngrams(1);
        args.set_loss(LossName::SOFTMAX);
        args.set_model(ModelName::SUP);
        args.set_bucket(0);
        args.set_minn(0);
        args.set_maxn(0);
        args.set_lr_update_rate(100);
        args.set_t(0.0001);
        args.save(&mut buf).unwrap();

        // Dictionary: 2 entries (</s> + one label), nwords=1, nlabels=1
        // size=2, nwords=1, nlabels=1, ntokens=10, pruneidx_size=-1
        utils::write_i32(&mut buf, 2).unwrap(); // size
        utils::write_i32(&mut buf, 1).unwrap(); // nwords
        utils::write_i32(&mut buf, 1).unwrap(); // nlabels
        utils::write_i64(&mut buf, 10).unwrap(); // ntokens
        utils::write_i64(&mut buf, -1i64).unwrap(); // pruneidx_size = -1 (not pruned)
        // Entry 0: </s>, count=5, type=0 (word)
        buf.extend_from_slice(b"</s>\0");
        utils::write_i64(&mut buf, 5).unwrap();
        buf.push(0u8); // EntryType::Word
        // Entry 1: __label__test, count=5, type=1 (label)
        buf.extend_from_slice(b"__label__test\0");
        utils::write_i64(&mut buf, 5).unwrap();
        buf.push(1u8); // EntryType::Label

        // quant_input = false
        buf.push(0u8);

        // Input matrix: 1 word × 2 dims (only 1 actual word entry contributes)
        // The input matrix has (nwords + bucket) rows and dim cols.
        // With nwords=1, bucket=0, dim=2: 1×2 matrix
        utils::write_i64(&mut buf, 1).unwrap(); // m = 1
        utils::write_i64(&mut buf, 2).unwrap(); // n = 2
        utils::write_f32(&mut buf, 0.1).unwrap();
        utils::write_f32(&mut buf, 0.2).unwrap();

        // qout = false
        buf.push(0u8);

        // Output matrix: 1 label × 2 dims
        utils::write_i64(&mut buf, 1).unwrap(); // m = 1
        utils::write_i64(&mut buf, 2).unwrap(); // n = 2
        utils::write_f32(&mut buf, 0.5).unwrap();
        utils::write_f32(&mut buf, -0.5).unwrap();

        buf
    }

    // =========================================================================
    // VAL-DICT-009: Magic number and version validation
    // =========================================================================

    #[test]
    fn test_binary_io_magic_version_valid() {
        // Valid magic + version should load successfully
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(&buf);
        let result = FastText::load(&mut cursor);
        assert!(result.is_ok(), "Valid model should load: {:?}", result.err());
    }

    #[test]
    fn test_binary_io_wrong_magic() {
        let mut buf = make_minimal_model_bytes();
        // Corrupt the magic number (first 4 bytes)
        buf[0] = 0xFF;
        buf[1] = 0xFF;
        buf[2] = 0xFF;
        buf[3] = 0xFF;

        let mut cursor = Cursor::new(&buf);
        let result = FastText::load(&mut cursor);
        assert!(result.is_err(), "Wrong magic should be rejected");
        match result.unwrap_err() {
            FastTextError::InvalidModel(msg) => {
                assert!(
                    msg.contains("magic") || msg.contains("Invalid"),
                    "Error should mention magic: {}",
                    msg
                );
            }
            e => panic!("Expected InvalidModel, got: {:?}", e),
        }
    }

    #[test]
    fn test_binary_io_wrong_version_too_high() {
        let mut buf = make_minimal_model_bytes();
        // Set version to 100 (too high) at bytes [4..8]
        let version_bytes: [u8; 4] = 100i32.to_le_bytes();
        buf[4] = version_bytes[0];
        buf[5] = version_bytes[1];
        buf[6] = version_bytes[2];
        buf[7] = version_bytes[3];

        let mut cursor = Cursor::new(&buf);
        let result = FastText::load(&mut cursor);
        assert!(result.is_err(), "Version 100 should be rejected");
        match result.unwrap_err() {
            FastTextError::InvalidModel(msg) => {
                assert!(
                    msg.contains("version") || msg.contains("Unsupported"),
                    "Error should mention version: {}",
                    msg
                );
            }
            e => panic!("Expected InvalidModel, got: {:?}", e),
        }
    }

    #[test]
    fn test_binary_io_version_12_accepted() {
        // Version 12 should be accepted
        let buf = make_minimal_model_bytes();
        assert_eq!(
            i32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            12,
            "Test data should have version 12"
        );
        let mut cursor = Cursor::new(&buf);
        let result = FastText::load(&mut cursor);
        assert!(result.is_ok(), "Version 12 should be accepted");
    }

    #[test]
    fn test_invalid_model_file() {
        // Load the invalid.model.bin fixture
        let result = FastText::load_model(INVALID_MODEL);
        assert!(result.is_err(), "Invalid model should be rejected");
    }

    // =========================================================================
    // VAL-DICT-010: Args block layout
    // =========================================================================

    #[test]
    fn test_binary_io_args_block() {
        // Load model and verify args are correct
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(&buf);
        let model = FastText::load(&mut cursor).unwrap();

        assert_eq!(model.args().dim(), 2);
        assert_eq!(model.args().ws(), 1);
        assert_eq!(model.args().epoch(), 1);
        assert_eq!(model.args().min_count(), 1);
        assert_eq!(model.args().neg(), 5);
        assert_eq!(model.args().word_ngrams(), 1);
        assert_eq!(model.args().loss(), LossName::SOFTMAX);
        assert_eq!(model.args().model(), ModelName::SUP);
        assert_eq!(model.args().bucket(), 0);
        assert_eq!(model.args().minn(), 0);
        assert_eq!(model.args().maxn(), 0);
        assert_eq!(model.args().lr_update_rate(), 100);
        assert!((model.args().t() - 0.0001).abs() < f64::EPSILON);
    }

    // =========================================================================
    // VAL-DICT-011: Dictionary block layout
    // =========================================================================

    #[test]
    fn test_binary_io_dict_block() {
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(&buf);
        let model = FastText::load(&mut cursor).unwrap();

        let dict = model.dict();
        assert_eq!(dict.size(), 2);
        assert_eq!(dict.nwords(), 1);
        assert_eq!(dict.nlabels(), 1);
        assert_eq!(dict.ntokens(), 10);

        // Check word entries
        let words = dict.words();
        assert_eq!(words[0].word, "</s>");
        assert_eq!(words[0].count, 5);
        assert_eq!(
            words[0].entry_type,
            crate::dictionary::EntryType::Word
        );

        assert_eq!(words[1].word, "__label__test");
        assert_eq!(words[1].count, 5);
        assert_eq!(
            words[1].entry_type,
            crate::dictionary::EntryType::Label
        );

        // Verify word lookup works
        assert_eq!(dict.get_id("</s>"), 0);
        assert_eq!(dict.get_id("__label__test"), 1);
        assert_eq!(dict.get_id("unknown"), -1);
    }

    // =========================================================================
    // VAL-DICT-012: Matrix blocks
    // =========================================================================

    #[test]
    fn test_binary_io_dense_matrix() {
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(&buf);
        let model = FastText::load(&mut cursor).unwrap();

        let input = model.input_matrix();
        assert_eq!(input.rows(), 1);
        assert_eq!(input.cols(), 2);
        assert!((input.at(0, 0) - 0.1).abs() < 1e-6, "input[0,0] = {}", input.at(0, 0));
        assert!((input.at(0, 1) - 0.2).abs() < 1e-6, "input[0,1] = {}", input.at(0, 1));

        let output = model.output_matrix();
        assert_eq!(output.rows(), 1);
        assert_eq!(output.cols(), 2);
        assert!((output.at(0, 0) - 0.5).abs() < 1e-6, "output[0,0] = {}", output.at(0, 0));
        assert!((output.at(0, 1) - (-0.5)).abs() < 1e-6, "output[0,1] = {}", output.at(0, 1));
    }

    #[test]
    fn test_binary_io_dense_matrix_roundtrip() {
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(buf);
        let model = FastText::load(&mut cursor).unwrap();

        // Save and reload
        let mut saved = Vec::new();
        model.save(&mut saved).unwrap();

        let mut cursor2 = Cursor::new(saved);
        let model2 = FastText::load(&mut cursor2).unwrap();

        // Verify matrices match
        let input1 = model.input_matrix();
        let input2 = model2.input_matrix();
        assert_eq!(input1.rows(), input2.rows());
        assert_eq!(input1.cols(), input2.cols());
        for i in 0..input1.rows() {
            for j in 0..input1.cols() {
                assert_eq!(
                    input1.at(i, j),
                    input2.at(i, j),
                    "Input matrix mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }

        let output1 = model.output_matrix();
        let output2 = model2.output_matrix();
        for i in 0..output1.rows() {
            for j in 0..output1.cols() {
                assert_eq!(
                    output1.at(i, j),
                    output2.at(i, j),
                    "Output matrix mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    // =========================================================================
    // VAL-DICT-013: Loading cooking.model.bin reference model
    // =========================================================================

    #[test]
    fn test_load_cooking_model_args() {
        let model = FastText::load_model(COOKING_MODEL)
            .expect("Should load cooking model");

        let args = model.args();
        // From C++ `dump args` reference output
        assert_eq!(args.dim(), 100, "dim should be 100");
        assert_eq!(args.ws(), 5, "ws should be 5");
        assert_eq!(args.epoch(), 50, "epoch should be 50");
        assert_eq!(args.min_count(), 1, "minCount should be 1");
        assert_eq!(args.neg(), 5, "neg should be 5");
        assert_eq!(args.word_ngrams(), 1, "wordNgrams should be 1");
        assert_eq!(args.loss(), LossName::SOFTMAX, "loss should be softmax");
        assert_eq!(args.model(), ModelName::SUP, "model should be SUP");
        assert_eq!(args.bucket(), 0, "bucket should be 0");
        assert_eq!(args.minn(), 0, "minn should be 0");
        assert_eq!(args.maxn(), 0, "maxn should be 0");
        assert_eq!(args.lr_update_rate(), 100, "lrUpdateRate should be 100");
        assert!(
            (args.t() - 0.0001).abs() < 1e-10,
            "t should be 0.0001, got {}",
            args.t()
        );
    }

    #[test]
    fn test_load_cooking_model_vocab() {
        let model = FastText::load_model(COOKING_MODEL)
            .expect("Should load cooking model");

        let dict = model.dict();
        assert_eq!(
            dict.nwords(),
            14543,
            "Should have 14543 words, got {}",
            dict.nwords()
        );
        assert_eq!(
            dict.nlabels(),
            735,
            "Should have 735 labels, got {}",
            dict.nlabels()
        );
        assert_eq!(
            dict.size(),
            14543 + 735,
            "Total size should be 15278"
        );
    }

    #[test]
    fn test_load_cooking_model_first_entry() {
        let model = FastText::load_model(COOKING_MODEL)
            .expect("Should load cooking model");

        let dict = model.dict();
        let words = dict.words();

        // First entry should be </s> with freq 12404
        assert_eq!(words[0].word, "</s>", "First word should be </s>");
        assert_eq!(
            words[0].count, 12404,
            "First word frequency should be 12404, got {}",
            words[0].count
        );
        assert_eq!(
            words[0].entry_type,
            crate::dictionary::EntryType::Word
        );
    }

    #[test]
    fn test_load_cooking_model_first_label() {
        let model = FastText::load_model(COOKING_MODEL)
            .expect("Should load cooking model");

        let dict = model.dict();
        // First label should be __label__baking with freq 1156
        let first_label = dict.get_label(0).expect("Should have at least one label");
        assert_eq!(
            first_label, "__label__baking",
            "First label should be __label__baking, got {}",
            first_label
        );

        let words = dict.words();
        let nwords = dict.nwords();
        let first_label_entry = &words[nwords as usize];
        assert_eq!(
            first_label_entry.count, 1156,
            "First label frequency should be 1156, got {}",
            first_label_entry.count
        );
    }

    #[test]
    fn test_load_cooking_model_matrices() {
        let model = FastText::load_model(COOKING_MODEL)
            .expect("Should load cooking model");

        // Input matrix: nwords × dim (no subwords since maxn=0, bucket=0)
        let input = model.input_matrix();
        assert_eq!(input.rows(), 14543, "Input rows should be nwords=14543");
        assert_eq!(input.cols(), 100, "Input cols should be dim=100");

        // Output matrix: nlabels × dim
        let output = model.output_matrix();
        assert_eq!(output.rows(), 735, "Output rows should be nlabels=735");
        assert_eq!(output.cols(), 100, "Output cols should be dim=100");

        // Matrices should not be quantized
        assert!(!model.is_quant(), "cooking.model.bin should not be quantized");
    }

    #[test]
    fn test_load_cooking_model_word_lookup() {
        let model = FastText::load_model(COOKING_MODEL)
            .expect("Should load cooking model");

        let dict = model.dict();

        // EOS should be at index 0
        assert_eq!(dict.get_id("</s>"), 0, "EOS should be at index 0");

        // Known words should be findable
        assert!(dict.get_id("baking") >= 0, "'baking' should be in vocabulary");
        assert!(dict.get_id("banana") >= 0, "'banana' should be in vocabulary");

        // Unknown words should return -1
        assert_eq!(
            dict.get_id("xyzzy_definitely_not_a_word"),
            -1,
            "Unknown word should return -1"
        );
    }

    // =========================================================================
    // VAL-DICT-017: Invalid model rejection and backward compatibility
    // =========================================================================

    #[test]
    fn test_invalid_model_rejection() {
        // The invalid.model.bin fixture has a corrupt header
        let result = FastText::load_model(INVALID_MODEL);
        assert!(result.is_err(), "Invalid model should be rejected");
        match result {
            Err(FastTextError::InvalidModel(_)) | Err(FastTextError::IoError(_)) => {
                // Either is acceptable
            }
            Err(e) => panic!("Expected InvalidModel or IoError, got: {:?}", e),
            Ok(_) => panic!("Expected error for invalid model"),
        }
    }

    #[test]
    fn test_zero_bytes_rejected() {
        // Empty file should be rejected
        let result = FastText::load(&mut Cursor::new(vec![]));
        assert!(result.is_err(), "Empty data should be rejected");
    }

    #[test]
    fn test_truncated_model_rejected() {
        // Truncated model (only magic + partial version) should be rejected
        let mut buf = Vec::new();
        utils::write_i32(&mut buf, FASTTEXT_FILEFORMAT_MAGIC_INT32).unwrap();
        buf.push(0x01); // Incomplete version (only 1 byte instead of 4)
        let result = FastText::load(&mut Cursor::new(buf));
        assert!(result.is_err(), "Truncated model should be rejected");
    }

    #[test]
    fn test_backward_compat_v11_supervised() {
        // Version 11 + supervised model → maxn should be forced to 0
        let mut buf = make_minimal_model_bytes();
        // Change version from 12 to 11 at bytes [4..8]
        let version_bytes: [u8; 4] = 11i32.to_le_bytes();
        buf[4] = version_bytes[0];
        buf[5] = version_bytes[1];
        buf[6] = version_bytes[2];
        buf[7] = version_bytes[3];
        // The model in make_minimal_model_bytes has model=SUP and maxn=0 already,
        // but let's set maxn to 6 (default) in the args block to test the override.
        // Args start at byte 8. Field order: dim(0), ws(4), epoch(8), minCount(12),
        // neg(16), wordNgrams(20), loss(24), model(28), bucket(32), minn(36), maxn(40), ...
        // maxn is at offset 8 + 40 = 48
        let maxn_bytes: [u8; 4] = 6i32.to_le_bytes(); // set maxn=6
        buf[8 + 40] = maxn_bytes[0];
        buf[8 + 41] = maxn_bytes[1];
        buf[8 + 42] = maxn_bytes[2];
        buf[8 + 43] = maxn_bytes[3];

        let mut cursor = Cursor::new(buf);
        let model = FastText::load(&mut cursor).unwrap();

        // Version 11 with SUP model should force maxn=0
        assert_eq!(
            model.args().maxn(),
            0,
            "Version 11 supervised model should have maxn=0"
        );
    }

    #[test]
    fn test_backward_compat_v11_unsupervised() {
        // Version 11 with non-supervised model → maxn should NOT be forced to 0
        // Create a version 11 SG model
        let mut buf = Vec::new();
        utils::write_i32(&mut buf, FASTTEXT_FILEFORMAT_MAGIC_INT32).unwrap();
        utils::write_i32(&mut buf, 11).unwrap(); // version 11

        let mut args = Args::default();
        args.set_dim(2);
        args.set_ws(1);
        args.set_epoch(1);
        args.set_min_count(1);
        args.set_neg(5);
        args.set_word_ngrams(1);
        args.set_loss(LossName::NS);
        args.set_model(ModelName::SG); // NOT supervised
        args.set_bucket(100);
        args.set_minn(3);
        args.set_maxn(6); // should remain 6
        args.set_lr_update_rate(100);
        args.set_t(0.0001);
        args.save(&mut buf).unwrap();

        // Minimal dictionary (1 word, 0 labels)
        utils::write_i32(&mut buf, 1).unwrap(); // size
        utils::write_i32(&mut buf, 1).unwrap(); // nwords
        utils::write_i32(&mut buf, 0).unwrap(); // nlabels
        utils::write_i64(&mut buf, 5).unwrap(); // ntokens
        utils::write_i64(&mut buf, -1i64).unwrap(); // pruneidx_size
        buf.extend_from_slice(b"</s>\0");
        utils::write_i64(&mut buf, 5).unwrap();
        buf.push(0u8);

        // quant_input = false
        buf.push(0u8);

        // Input matrix: 1 + 100 = 101 rows (nwords=1, bucket=100) × 2 cols
        utils::write_i64(&mut buf, 101).unwrap();
        utils::write_i64(&mut buf, 2).unwrap();
        for _ in 0..(101 * 2) {
            utils::write_f32(&mut buf, 0.0).unwrap();
        }

        // qout = false
        buf.push(0u8);

        // Output matrix: 1 × 2
        utils::write_i64(&mut buf, 1).unwrap();
        utils::write_i64(&mut buf, 2).unwrap();
        utils::write_f32(&mut buf, 0.0).unwrap();
        utils::write_f32(&mut buf, 0.0).unwrap();

        let mut cursor = Cursor::new(buf);
        let model = FastText::load(&mut cursor).unwrap();

        // Non-supervised v11 model should keep maxn=6
        assert_eq!(
            model.args().maxn(),
            6,
            "Version 11 non-supervised model should keep maxn=6"
        );
    }

    // =========================================================================
    // Model save/load round-trip
    // =========================================================================

    #[test]
    fn test_model_save_load_roundtrip_minimal() {
        // Build minimal model, save, reload, verify args and dict match
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(buf);
        let model1 = FastText::load(&mut cursor).unwrap();

        // Save to buffer
        let mut saved = Vec::new();
        model1.save(&mut saved).unwrap();

        // Reload from saved buffer
        let mut cursor2 = Cursor::new(saved);
        let model2 = FastText::load(&mut cursor2).unwrap();

        // Args should match
        assert_eq!(model1.args().dim(), model2.args().dim());
        assert_eq!(model1.args().epoch(), model2.args().epoch());
        assert_eq!(model1.args().model(), model2.args().model());
        assert_eq!(model1.args().loss(), model2.args().loss());

        // Dict should match
        assert_eq!(model1.dict().size(), model2.dict().size());
        assert_eq!(model1.dict().nwords(), model2.dict().nwords());
        assert_eq!(model1.dict().nlabels(), model2.dict().nlabels());

        let w1 = model1.dict().words();
        let w2 = model2.dict().words();
        for i in 0..w1.len() {
            assert_eq!(w1[i].word, w2[i].word, "Word {} mismatch", i);
            assert_eq!(w1[i].count, w2[i].count, "Count {} mismatch", i);
            assert_eq!(
                w1[i].entry_type, w2[i].entry_type,
                "Type {} mismatch",
                i
            );
        }
    }

    #[test]
    fn test_model_is_quant_false_for_bin() {
        let model = FastText::load_model(COOKING_MODEL).unwrap();
        assert!(!model.is_quant(), "cooking.model.bin should not be quantized");
    }

    // =========================================================================
    // VAL-DICT-009 specifically: test with constructed wrong-magic binary
    // =========================================================================

    #[test]
    fn test_binary_io_magic_version() {
        // Valid magic and version 12
        let buf = make_minimal_model_bytes();
        let mut cursor = Cursor::new(&buf);
        let result = FastText::load(&mut cursor);
        assert!(result.is_ok(), "Valid magic+v12 should load: {:?}", result.err());

        // Wrong magic
        let mut bad_magic = buf.clone();
        bad_magic[0] = 0x00;
        bad_magic[1] = 0x00;
        bad_magic[2] = 0x00;
        bad_magic[3] = 0x00;
        let result = FastText::load(&mut Cursor::new(bad_magic));
        assert!(result.is_err(), "Wrong magic should be rejected");

        // Version too high
        let mut bad_version = buf.clone();
        let v = 13i32.to_le_bytes();
        bad_version[4] = v[0];
        bad_version[5] = v[1];
        bad_version[6] = v[2];
        bad_version[7] = v[3];
        let result = FastText::load(&mut Cursor::new(bad_version));
        assert!(result.is_err(), "Version 13 should be rejected");

        // Version 11 (should be accepted)
        let mut v11 = buf.clone();
        let v = 11i32.to_le_bytes();
        v11[4] = v[0];
        v11[5] = v[1];
        v11[6] = v[2];
        v11[7] = v[3];
        let result = FastText::load(&mut Cursor::new(v11));
        assert!(result.is_ok(), "Version 11 should be accepted: {:?}", result.err());
    }
}
