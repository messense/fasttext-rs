// QuantMatrix: product-quantized matrix using codebook lookup
//
// Matches the C++ fastText `QuantMatrix` class with identical binary format
// and computation semantics.

use std::io::{Read, Write};

use crate::error::{FastTextError, Result};
use crate::matrix::Matrix;
use crate::product_quantizer::ProductQuantizer;
use crate::utils;
use crate::vector::Vector;

// ============================================================================
// Helper I/O functions
// ============================================================================

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

// ============================================================================
// QuantMatrix
// ============================================================================

/// A product-quantized matrix using codebook lookup for approximate inner products.
///
/// Implements the `Matrix` trait with read-only quantized access:
/// - `dot_row` computes approximate dot product via `pq.mulcode`, optionally norm-scaled.
/// - `add_row_to_vector` uses `pq.addcode`, optionally scaled by quantized norm.
/// - `add_vector_to_row` panics (quantized matrices are read-only).
///
/// Matches the C++ fastText `QuantMatrix` class exactly in binary format and
/// computation semantics.
#[derive(Debug, Clone)]
pub struct QuantMatrix {
    /// Whether row norms are separately quantized.
    pub qnorm: bool,
    /// Number of rows.
    pub m: i64,
    /// Number of columns (original embedding dimension).
    pub n: i64,
    /// Total number of PQ codes: `m * pq.nsubq`.
    pub codesize: i32,
    /// Packed PQ codes for all rows, row-major. Length `codesize`.
    pub codes: Vec<u8>,
    /// Main product quantizer for the embedding vectors.
    pub pq: ProductQuantizer,
    /// Optional quantized norm codes (one per row). Present iff `qnorm`.
    pub norm_codes: Option<Vec<u8>>,
    /// Optional norm product quantizer (1-dimensional, 1 sub-quantizer). Present iff `qnorm`.
    pub npq: Option<ProductQuantizer>,
}

impl QuantMatrix {
    /// Create a `QuantMatrix` by quantizing a dense matrix.
    ///
    /// - `data` must be a row-major `f32` slice of shape `m × n`.
    /// - `dsub` is the sub-dimension for the product quantizer.
    /// - `qnorm` controls whether row norms are separately quantized.
    ///
    /// Matches C++ `QuantMatrix::QuantMatrix(DenseMatrix&&, int32_t, bool)`.
    pub fn from_dense(
        data: &[f32],
        m: i64,
        n: i64,
        dsub: i32,
        qnorm: bool,
    ) -> Self {
        let nsubq = if n % dsub as i64 == 0 {
            n / dsub as i64
        } else {
            n / dsub as i64 + 1
        };
        let codesize = (m * nsubq) as i32;
        let mut codes = vec![0u8; codesize as usize];
        let mut pq = ProductQuantizer::new(n as i32, dsub);

        // If qnorm, normalize the data by L2 norms first.
        let (norm_codes, npq) = if qnorm {
            // Compute L2 norms for each row.
            let mut norms = Vector::new(m as usize);
            for i in 0..m as usize {
                let row = &data[i * n as usize..(i + 1) * n as usize];
                let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
                norms[i] = norm;
            }

            // Normalize data (make a copy since we need to modify it).
            let mut normalized = data.to_vec();
            for i in 0..m as usize {
                let norm = norms[i];
                if norm != 0.0 {
                    let row = &mut normalized[i * n as usize..(i + 1) * n as usize];
                    for v in row.iter_mut() {
                        *v /= norm;
                    }
                }
            }

            // Train and encode with the normalized data.
            pq.train(m as i32, &normalized);
            pq.compute_codes(&normalized, &mut codes, m as i32);

            // Quantize norms separately.
            let mut npq = ProductQuantizer::new(1, 1);
            let mut nc = vec![0u8; m as usize];
            npq.train(m as i32, norms.data());
            npq.compute_codes(norms.data(), &mut nc, m as i32);

            (Some(nc), Some(npq))
        } else {
            pq.train(m as i32, data);
            pq.compute_codes(data, &mut codes, m as i32);
            (None, None)
        };

        QuantMatrix {
            qnorm,
            m,
            n,
            codesize,
            codes,
            pq,
            norm_codes,
            npq,
        }
    }

    /// Get the quantized norm for row `i`.
    ///
    /// Returns 1.0 if `qnorm` is false, otherwise looks up the norm centroid.
    #[inline]
    fn norm_for_row(&self, i: usize) -> f32 {
        if self.qnorm {
            let norm_codes = self.norm_codes.as_ref().expect("norm_codes must exist when qnorm=true");
            let npq = self.npq.as_ref().expect("npq must exist when qnorm=true");
            npq.get_centroids(0, norm_codes[i])[0]
        } else {
            1.0
        }
    }
}

// ============================================================================
// Matrix trait implementation
// ============================================================================

impl Matrix for QuantMatrix {
    #[inline]
    fn rows(&self) -> i64 {
        self.m
    }

    #[inline]
    fn cols(&self) -> i64 {
        self.n
    }

    /// Compute approximate dot product of `vec` with quantized row `i`.
    ///
    /// When `qnorm=true`, the result is scaled by the quantized norm of the row.
    /// Matches C++ `QuantMatrix::dotRow`.
    fn dot_row(&self, vec: &Vector, i: i64) -> Result<f32> {
        assert!(i >= 0 && i < self.m, "Row index out of bounds: {}", i);
        assert_eq!(
            vec.len(),
            self.n as usize,
            "Vector size {} does not match matrix columns {}",
            vec.len(),
            self.n
        );
        let norm = self.norm_for_row(i as usize);
        let result = self.pq.mulcode(vec, &self.codes, i as i32, norm);
        if result.is_nan() {
            return Err(FastTextError::EncounteredNaN);
        }
        Ok(result)
    }

    /// **Not permitted on quantized matrices. Always panics.**
    ///
    /// Matches C++ which throws `std::runtime_error`.
    fn add_vector_to_row(&mut self, _vec: &Vector, _i: i64, _scale: f32) {
        panic!("Operation not permitted on quantized matrices.");
    }

    /// Add quantized reconstruction of row `i` to vector `x`, scaled by `scale`.
    ///
    /// When `qnorm=true`, the scale is further multiplied by the quantized norm.
    /// Matches C++ `QuantMatrix::addRowToVector(x, i, a)`.
    fn add_row_to_vector(&self, x: &mut Vector, i: i32, scale: f32) {
        assert!(i >= 0 && (i as i64) < self.m, "Row index out of bounds: {}", i);
        let norm = self.norm_for_row(i as usize);
        self.pq.addcode(x, &self.codes, i, scale * norm);
    }

    /// Average the specified rows into vector `x`.
    ///
    /// Zeros `x`, sums all row reconstructions with norm=1.0 scale, then divides by count.
    /// Matches C++ `QuantMatrix::averageRowsToVector`.
    fn average_rows_to_vector(&self, x: &mut Vector, rows: &[i32]) {
        x.zero();
        for &row in rows {
            self.add_row_to_vector(x, row, 1.0);
        }
        if !rows.is_empty() {
            x.mul(1.0 / rows.len() as f32);
        }
    }

    /// Save the `QuantMatrix` in C++ binary format.
    ///
    /// Format:
    /// ```text
    /// qnorm:     bool    (1 byte)
    /// m:         i64     (8 bytes)
    /// n:         i64     (8 bytes)
    /// codesize:  i32     (4 bytes)
    /// codes:     [u8; codesize]
    /// pq:        ProductQuantizer (variable)
    /// [if qnorm:]
    ///   norm_codes: [u8; m]
    ///   npq:        ProductQuantizer (variable)
    /// ```
    fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        write_bool(writer, self.qnorm)?;
        utils::write_i64(writer, self.m)?;
        utils::write_i64(writer, self.n)?;
        utils::write_i32(writer, self.codesize)?;
        writer.write_all(&self.codes)?;
        self.pq.save(writer)?;
        if self.qnorm {
            let norm_codes = self.norm_codes.as_ref().expect("norm_codes must exist when qnorm=true");
            writer.write_all(norm_codes)?;
            let npq = self.npq.as_ref().expect("npq must exist when qnorm=true");
            npq.save(writer)?;
        }
        Ok(())
    }

    /// Load a `QuantMatrix` from C++ binary format.
    ///
    /// Reads the same layout as `save`.
    fn load<R: Read>(reader: &mut R) -> Result<Self> {
        let qnorm = read_bool(reader)?;
        let m = utils::read_i64(reader)?;
        let n = utils::read_i64(reader)?;
        let codesize = utils::read_i32(reader)?;

        if m < 0 || n < 0 {
            return Err(FastTextError::InvalidModel(format!(
                "Invalid QuantMatrix dimensions: {}x{}",
                m, n
            )));
        }
        if codesize < 0 {
            return Err(FastTextError::InvalidModel(format!(
                "Invalid QuantMatrix codesize: {}",
                codesize
            )));
        }

        let mut codes = vec![0u8; codesize as usize];
        reader.read_exact(&mut codes)?;

        let pq = ProductQuantizer::load(reader)?;

        let (norm_codes, npq) = if qnorm {
            let mut nc = vec![0u8; m as usize];
            reader.read_exact(&mut nc)?;
            let npq = ProductQuantizer::load(reader)?;
            (Some(nc), Some(npq))
        } else {
            (None, None)
        };

        Ok(QuantMatrix {
            qnorm,
            m,
            n,
            codesize,
            codes,
            pq,
            norm_codes,
            npq,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // -----------------------------------------------------------------------
    // Helper: build a minimal QuantMatrix with known PQ for testing
    // -----------------------------------------------------------------------

    /// Build a QuantMatrix from a `make_known_pq()`-style PQ with 4-dim vectors.
    ///
    /// We set up:
    ///   - dim=4, dsub=2, nsubq=2, m=4 rows
    ///   - Row 0 codes: [0, 0]  → [c0_0, c1_0] = [1,0, 2,0]
    ///   - Row 1 codes: [0, 1]  → [c0_0, c1_1] = [1,0, 0,2]
    ///   - Row 2 codes: [1, 0]  → [c0_1, c1_0] = [0,1, 2,0]
    ///   - Row 3 codes: [1, 1]  → [c0_1, c1_1] = [0,1, 0,2]
    fn make_test_qm() -> QuantMatrix {
        let dim = 4i32;
        let dsub = 2i32;
        let ksub = 256usize;
        let m = 4i64;
        let n = 4i64;
        let nsubq = 2i32;
        let codesize = m as i32 * nsubq;

        let mut pq = ProductQuantizer::new(dim, dsub);

        // Fill all centroids far away, then set the 4 used ones.
        for v in pq.centroids.iter_mut() {
            *v = 100.0;
        }
        // Sub-q 0 (m=0): cstart = 0
        pq.centroids[0 * dsub as usize] = 1.0; // c0: [1, 0]
        pq.centroids[0 * dsub as usize + 1] = 0.0;
        pq.centroids[1 * dsub as usize] = 0.0; // c1: [0, 1]
        pq.centroids[1 * dsub as usize + 1] = 1.0;

        // Sub-q 1 (m=1): cstart = 1 * ksub * dsub = 512
        let cstart1 = 1 * ksub * dsub as usize;
        pq.centroids[cstart1 + 0 * dsub as usize] = 2.0; // c0: [2, 0]
        pq.centroids[cstart1 + 0 * dsub as usize + 1] = 0.0;
        pq.centroids[cstart1 + 1 * dsub as usize] = 0.0; // c1: [0, 2]
        pq.centroids[cstart1 + 1 * dsub as usize + 1] = 2.0;

        // codes: row 0=[0,0], row 1=[0,1], row 2=[1,0], row 3=[1,1]
        let codes = vec![0u8, 0, 0, 1, 1, 0, 1, 1];

        QuantMatrix {
            qnorm: false,
            m,
            n,
            codesize,
            codes,
            pq,
            norm_codes: None,
            npq: None,
        }
    }

    // -----------------------------------------------------------------------
    // Shape and accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_qm_shape() {
        let qm = make_test_qm();
        assert_eq!(qm.rows(), 4);
        assert_eq!(qm.cols(), 4);
    }

    // -----------------------------------------------------------------------
    // dot_row: approximate dot product via codebook lookup
    // -----------------------------------------------------------------------

    #[test]
    fn test_qm_dot_row_basic() {
        // VAL-DICT-016: dotRow computes approximate dot product via PQ codes.
        // Row 0 codes: [0,0] → reconstructed = [1,0,2,0]
        // x = [1,0,0,0]
        // dot = 1*1 + 0*0 + 0*2 + 0*0 = 1.0
        let qm = make_test_qm();
        let mut x = Vector::new(4);
        x[0] = 1.0;
        let result = qm.dot_row(&x, 0).expect("dot_row should succeed");
        assert!((result - 1.0).abs() < 1e-6, "expected 1.0, got {}", result);
    }

    #[test]
    fn test_qm_dot_row_row1() {
        // Row 1 codes: [0,1] → reconstructed = [1,0,0,2]
        // x = [0,0,0,1]
        // dot = 0*1 + 0*0 + 0*0 + 1*2 = 2.0
        let qm = make_test_qm();
        let mut x = Vector::new(4);
        x[3] = 1.0;
        let result = qm.dot_row(&x, 1).expect("dot_row should succeed");
        assert!((result - 2.0).abs() < 1e-6, "expected 2.0, got {}", result);
    }

    #[test]
    fn test_qm_dot_row_all_rows() {
        // Verify each row's dot product with a known vector.
        // x = [1,1,1,1]
        // Row 0 codes [0,0] → [1,0,2,0] → dot = 1+0+2+0 = 3
        // Row 1 codes [0,1] → [1,0,0,2] → dot = 1+0+0+2 = 3
        // Row 2 codes [1,0] → [0,1,2,0] → dot = 0+1+2+0 = 3
        // Row 3 codes [1,1] → [0,1,0,2] → dot = 0+1+0+2 = 3
        let qm = make_test_qm();
        let mut x = Vector::new(4);
        x[0] = 1.0;
        x[1] = 1.0;
        x[2] = 1.0;
        x[3] = 1.0;

        for i in 0..4i64 {
            let result = qm.dot_row(&x, i).expect("dot_row should succeed");
            assert!((result - 3.0).abs() < 1e-6, "row {}: expected 3.0, got {}", i, result);
        }
    }

    // -----------------------------------------------------------------------
    // add_row_to_vector: PQ reconstruction via addcode
    // -----------------------------------------------------------------------

    #[test]
    fn test_qm_add_row_to_vector_scale_one() {
        // VAL-DICT-016: addRowToVector adds quantized row reconstruction to x.
        // Row 0 codes [0,0] → centroid reconstruction = [1,0,2,0]
        // scale=1.0, x starts at zero → x = [1,0,2,0]
        let qm = make_test_qm();
        let mut x = Vector::new(4);
        qm.add_row_to_vector(&mut x, 0, 1.0);
        assert!((x[0] - 1.0).abs() < 1e-6, "x[0] expected 1.0, got {}", x[0]);
        assert!((x[1] - 0.0).abs() < 1e-6, "x[1] expected 0.0, got {}", x[1]);
        assert!((x[2] - 2.0).abs() < 1e-6, "x[2] expected 2.0, got {}", x[2]);
        assert!((x[3] - 0.0).abs() < 1e-6, "x[3] expected 0.0, got {}", x[3]);
    }

    #[test]
    fn test_qm_add_row_to_vector_with_scale() {
        // Row 1 codes [0,1] → reconstruction = [1,0,0,2], scale=2.0
        let qm = make_test_qm();
        let mut x = Vector::new(4);
        qm.add_row_to_vector(&mut x, 1, 2.0);
        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!((x[1] - 0.0).abs() < 1e-6);
        assert!((x[2] - 0.0).abs() < 1e-6);
        assert!((x[3] - 4.0).abs() < 1e-6); // 2 * 2.0
    }

    #[test]
    fn test_qm_add_row_to_vector_accumulates() {
        // Calling add_row_to_vector twice should accumulate.
        let qm = make_test_qm();
        let mut x = Vector::new(4);
        qm.add_row_to_vector(&mut x, 0, 1.0); // adds [1,0,2,0]
        qm.add_row_to_vector(&mut x, 1, 1.0); // adds [1,0,0,2]
        // total: [2,0,2,2]
        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!((x[1] - 0.0).abs() < 1e-6);
        assert!((x[2] - 2.0).abs() < 1e-6);
        assert!((x[3] - 2.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // average_rows_to_vector
    // -----------------------------------------------------------------------

    #[test]
    fn test_qm_average_rows_to_vector() {
        // Average rows 0 and 1:
        // Row 0 → [1,0,2,0], Row 1 → [1,0,0,2]
        // Average → [1,0,1,1]
        let qm = make_test_qm();
        let mut x = Vector::new(4);
        x[0] = 99.0; // should be zeroed before averaging
        qm.average_rows_to_vector(&mut x, &[0, 1]);
        assert!((x[0] - 1.0).abs() < 1e-6, "x[0]={}", x[0]);
        assert!((x[1] - 0.0).abs() < 1e-6, "x[1]={}", x[1]);
        assert!((x[2] - 1.0).abs() < 1e-6, "x[2]={}", x[2]);
        assert!((x[3] - 1.0).abs() < 1e-6, "x[3]={}", x[3]);
    }

    #[test]
    fn test_qm_average_rows_single() {
        // Average of single row = same as add_row_to_vector with scale 1.0
        let qm = make_test_qm();
        let mut x = Vector::new(4);
        qm.average_rows_to_vector(&mut x, &[2]);
        // Row 2 → [0,1,2,0], averaged over 1 = [0,1,2,0]
        assert!((x[0] - 0.0).abs() < 1e-6);
        assert!((x[1] - 1.0).abs() < 1e-6);
        assert!((x[2] - 2.0).abs() < 1e-6);
        assert!((x[3] - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // add_vector_to_row: must panic (write rejection)
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "Operation not permitted on quantized matrices")]
    fn test_qm_add_vector_to_row_panics() {
        // VAL-DICT-016: write operations must be rejected
        let mut qm = make_test_qm();
        let x = Vector::new(4);
        qm.add_vector_to_row(&x, 0, 1.0);
    }

    // -----------------------------------------------------------------------
    // Norm quantization (qnorm)
    // -----------------------------------------------------------------------

    /// Build a QuantMatrix with qnorm=true.
    ///
    /// npq has a single centroid at [2.0] and [0.5].
    /// norm_codes = [0, 1, 0, 1] → norms = [2.0, 0.5, 2.0, 0.5]
    fn make_qnorm_qm() -> QuantMatrix {
        let mut qm = make_test_qm();
        qm.qnorm = true;

        // Setup npq: dim=1, dsub=1 → nsubq=1, ksub=256
        let mut npq = ProductQuantizer::new(1, 1);
        // Set centroid 0 to [2.0], centroid 1 to [0.5]
        for v in npq.centroids.iter_mut() {
            *v = 1.0; // default
        }
        npq.centroids[0] = 2.0; // centroid 0
        npq.centroids[1] = 0.5; // centroid 1

        qm.norm_codes = Some(vec![0u8, 1, 0, 1]); // row norms: 2.0, 0.5, 2.0, 0.5
        qm.npq = Some(npq);
        qm
    }

    #[test]
    fn test_qm_qnorm_dot_row_scales_result() {
        // VAL-DICT-016: qnorm quantization scales dotRow result.
        // Row 0: codes [0,0] → base_dot(x, [1,0,2,0]), norm=2.0
        // x = [1,0,0,0] → base_dot = 1.0, result = 1.0 * 2.0 = 2.0
        let qm = make_qnorm_qm();
        let mut x = Vector::new(4);
        x[0] = 1.0;
        let result = qm.dot_row(&x, 0).expect("dot_row should succeed");
        assert!((result - 2.0).abs() < 1e-6, "expected 2.0, got {}", result);
    }

    #[test]
    fn test_qm_qnorm_dot_row_row1() {
        // Row 1: codes [0,1] → [1,0,0,2], norm=0.5
        // x = [0,0,0,1] → base_dot = 2.0, result = 2.0 * 0.5 = 1.0
        let qm = make_qnorm_qm();
        let mut x = Vector::new(4);
        x[3] = 1.0;
        let result = qm.dot_row(&x, 1).expect("dot_row should succeed");
        assert!((result - 1.0).abs() < 1e-6, "expected 1.0, got {}", result);
    }

    #[test]
    fn test_qm_qnorm_add_row_scales_reconstruction() {
        // Row 0: codes [0,0] → [1,0,2,0], norm=2.0, scale=1.0
        // addcode with alpha = scale * norm = 2.0
        // x = [2,0,4,0]
        let qm = make_qnorm_qm();
        let mut x = Vector::new(4);
        qm.add_row_to_vector(&mut x, 0, 1.0);
        assert!((x[0] - 2.0).abs() < 1e-6, "x[0]={}", x[0]);
        assert!((x[1] - 0.0).abs() < 1e-6, "x[1]={}", x[1]);
        assert!((x[2] - 4.0).abs() < 1e-6, "x[2]={}", x[2]);
        assert!((x[3] - 0.0).abs() < 1e-6, "x[3]={}", x[3]);
    }

    // -----------------------------------------------------------------------
    // Save / load round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_qm_save_load_roundtrip() {
        // VAL-DICT-012 / VAL-DICT-016: QuantMatrix must survive serialize → deserialize.
        let qm = make_test_qm();

        let mut buf = Vec::new();
        qm.save(&mut buf).expect("save should succeed");

        let mut cursor = Cursor::new(&buf);
        let qm2 = QuantMatrix::load(&mut cursor).expect("load should succeed");

        // Check dimensions.
        assert_eq!(qm2.m, qm.m);
        assert_eq!(qm2.n, qm.n);
        assert_eq!(qm2.qnorm, qm.qnorm);
        assert_eq!(qm2.codesize, qm.codesize);
        assert_eq!(qm2.codes, qm.codes);
        assert!(qm2.norm_codes.is_none());
        assert!(qm2.npq.is_none());
    }

    #[test]
    fn test_qm_save_load_qnorm_roundtrip() {
        // Round-trip with qnorm=true.
        let qm = make_qnorm_qm();

        let mut buf = Vec::new();
        qm.save(&mut buf).expect("save should succeed");

        let mut cursor = Cursor::new(&buf);
        let qm2 = QuantMatrix::load(&mut cursor).expect("load should succeed");

        assert_eq!(qm2.qnorm, true);
        assert_eq!(qm2.m, qm.m);
        assert_eq!(qm2.n, qm.n);
        assert_eq!(qm2.codesize, qm.codesize);
        assert_eq!(qm2.codes, qm.codes);
        assert_eq!(qm2.norm_codes, qm.norm_codes);

        // NPQ centroids must match.
        let nc1 = qm.npq.as_ref().unwrap();
        let nc2 = qm2.npq.as_ref().unwrap();
        assert_eq!(nc1.centroids.len(), nc2.centroids.len());
        for (a, b) in nc1.centroids.iter().zip(nc2.centroids.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_qm_save_load_preserves_dot_row() {
        // After round-trip, dot_row must produce the same results.
        let qm = make_test_qm();

        let mut buf = Vec::new();
        qm.save(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let qm2 = QuantMatrix::load(&mut cursor).unwrap();

        let mut x = Vector::new(4);
        x[0] = 1.0;
        x[1] = 0.5;
        x[2] = 0.25;
        x[3] = 0.125;

        for i in 0..4i64 {
            let r1 = qm.dot_row(&x, i).unwrap();
            let r2 = qm2.dot_row(&x, i).unwrap();
            assert_eq!(r1.to_bits(), r2.to_bits(), "row {}: dot_row mismatch", i);
        }
    }

    // -----------------------------------------------------------------------
    // Shape preservation
    // -----------------------------------------------------------------------

    #[test]
    fn test_qm_shape_preservation_after_load() {
        // Shape and key invariants must be preserved after save/load.
        let qm = make_test_qm();

        let mut buf = Vec::new();
        qm.save(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let qm2 = QuantMatrix::load(&mut cursor).unwrap();

        assert_eq!(qm2.rows(), 4);
        assert_eq!(qm2.cols(), 4);
        assert_eq!(qm2.pq.dim, 4);
        assert_eq!(qm2.pq.nsubq, 2);
    }

    // -----------------------------------------------------------------------
    // dot_row vs add_row_to_vector consistency
    // -----------------------------------------------------------------------

    #[test]
    fn test_qm_dot_row_add_row_consistency() {
        // dot_row(x, i) should equal x · add_row_to_vector(zeros, i)
        let qm = make_test_qm();

        let x_data = [0.3f32, 0.7, 1.1, 0.5];
        let mut x = Vector::new(4);
        for (i, &v) in x_data.iter().enumerate() {
            x[i] = v;
        }

        for i in 0..4i64 {
            let dot_result = qm.dot_row(&x, i).unwrap();

            let mut recon = Vector::new(4);
            qm.add_row_to_vector(&mut recon, i as i32, 1.0);
            let manual_dot: f32 = x_data.iter().zip(recon.data().iter()).map(|(&a, &b)| a * b).sum();

            assert!(
                (dot_result - manual_dot).abs() < 1e-6,
                "row {}: dot_row={} manual_dot={}",
                i,
                dot_result,
                manual_dot
            );
        }
    }

    // -----------------------------------------------------------------------
    // from_dense smoke test
    // -----------------------------------------------------------------------

    #[test]
    fn test_qm_from_dense_smoke() {
        // from_dense should produce a QuantMatrix with correct shape.
        // Use enough rows for k-means (need >= KSUB=256).
        let m = 300i64;
        let n = 4i64;
        let dsub = 2i32;
        let data: Vec<f32> = (0..m * n)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        let qm = QuantMatrix::from_dense(&data, m, n, dsub, false);
        assert_eq!(qm.rows(), m);
        assert_eq!(qm.cols(), n);
        assert_eq!(qm.codes.len(), qm.codesize as usize);
        assert!(!qm.qnorm);
        assert!(qm.norm_codes.is_none());
        assert!(qm.npq.is_none());
    }

    #[test]
    fn test_qm_from_dense_qnorm_smoke() {
        // from_dense with qnorm=true should populate norm_codes and npq.
        let m = 300i64;
        let n = 4i64;
        let dsub = 2i32;
        let data: Vec<f32> = (0..m * n)
            .map(|i| (i as f32 * 0.01).cos() + 0.5)
            .collect();

        let qm = QuantMatrix::from_dense(&data, m, n, dsub, true);
        assert_eq!(qm.rows(), m);
        assert!(qm.qnorm);
        assert!(qm.norm_codes.is_some());
        assert_eq!(qm.norm_codes.as_ref().unwrap().len(), m as usize);
        assert!(qm.npq.is_some());
    }

    // -----------------------------------------------------------------------
    // load rejects invalid data
    // -----------------------------------------------------------------------

    #[test]
    fn test_qm_load_negative_dims_rejected() {
        // Negative m or n should produce InvalidModel error.
        let mut buf = Vec::new();
        write_bool(&mut buf, false).unwrap();           // qnorm
        utils::write_i64(&mut buf, -1).unwrap();        // m (invalid)
        utils::write_i64(&mut buf, 4).unwrap();         // n
        utils::write_i32(&mut buf, 0).unwrap();         // codesize
        // PQ data would follow, but we expect an error before that.
        let mut cursor = Cursor::new(&buf);
        let result = QuantMatrix::load(&mut cursor);
        assert!(result.is_err(), "Expected error for negative m");
    }

    #[test]
    fn test_qm_load_negative_codesize_rejected() {
        let mut buf = Vec::new();
        write_bool(&mut buf, false).unwrap();
        utils::write_i64(&mut buf, 4).unwrap();         // m
        utils::write_i64(&mut buf, 4).unwrap();         // n
        utils::write_i32(&mut buf, -1).unwrap();        // codesize (invalid)
        let mut cursor = Cursor::new(&buf);
        let result = QuantMatrix::load(&mut cursor);
        assert!(result.is_err(), "Expected error for negative codesize");
    }
}
