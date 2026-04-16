// Matrix trait and DenseMatrix implementation
//
// DenseMatrix stores f32 values in row-major order with 64-byte aligned storage,
// matching the C++ fastText DenseMatrix class backed by `intgemm::AlignedVector<real>`.

use std::alloc::{self, Layout};
use std::convert::TryFrom;
use std::io::{Read, Write};

use crate::error::{FastTextError, Result};
use crate::model::MinstdRng;
use crate::utils;
use crate::vector::Vector;

/// Alignment in bytes for SIMD-friendly memory layout.
const ALIGNMENT: usize = 64;

// Matrix trait

/// Trait for matrix types, matching the C++ fastText abstract `Matrix` class.
pub trait Matrix {
    /// Return the number of rows.
    fn rows(&self) -> i64;

    /// Return the number of columns.
    fn cols(&self) -> i64;

    /// Compute the dot product of a vector with a specific matrix row.
    ///
    /// Returns the dot product value. NaN results are passed through as-is.
    fn dot_row(&self, vec: &Vector, i: i64) -> f32;

    /// Add `scale * vec` to row `i` of the matrix.
    fn add_vector_to_row(&mut self, vec: &Vector, i: i64, scale: f32);

    /// Add row `i` of the matrix to vector `x` (with scale 1.0).
    fn add_row_to_vector(&self, x: &mut Vector, i: i32, scale: f32);

    /// Average the specified rows into vector `x`.
    fn average_rows_to_vector(&self, x: &mut Vector, rows: &[i32]);

    /// Save the matrix in binary format.
    fn save<W: Write>(&self, writer: &mut W) -> Result<()>;

    /// Load the matrix from binary format.
    fn load<R: Read>(reader: &mut R) -> Result<Self>
    where
        Self: Sized;
}

// DenseMatrix

/// A dense matrix of `f32` values with 64-byte aligned row-major storage.
///
/// Matches the C++ fastText `DenseMatrix` class. Storage uses
/// `std::alloc::Layout` with 64-byte alignment for SIMD operations.
#[derive(Debug)]
pub struct DenseMatrix {
    /// Raw pointer to the aligned allocation.
    ptr: *mut f32,
    /// Number of rows.
    m: i64,
    /// Number of columns.
    n: i64,
    /// Cached element count (m * n as usize) to avoid recomputation on hot paths.
    size: usize,
}

// SAFETY: DenseMatrix owns its allocation exclusively.  `Send` is sound
// because the owned buffer can be transferred between threads.  `Sync` is
// required for `Arc<DenseMatrix>` in the Hogwild! training path, where
// concurrent unsynchronized writes to distinct (or overlapping) f32
// elements are intentional.  The `add_vector_to_row_unsync` method uses
// raw-pointer writes (not `&mut` references) so that Rust's aliasing
// rules are not violated.
unsafe impl Send for DenseMatrix {}
unsafe impl Sync for DenseMatrix {}

/// Compute `m * n` as `usize` with overflow checking.
///
/// Converts both dimensions from `i64` to `usize` (panicking if negative) and
/// then uses `checked_mul` to detect multiplication overflow.  This prevents
/// silent wrap-around when very large dimension values are passed.
///
/// # Panics
/// Panics with a descriptive message if either dimension is negative or if the
/// product overflows `usize`.
#[inline]
fn checked_dim_size(m: i64, n: i64) -> usize {
    let m_u = usize::try_from(m).expect("DenseMatrix row count (m) must be non-negative");
    let n_u = usize::try_from(n).expect("DenseMatrix column count (n) must be non-negative");
    m_u.checked_mul(n_u)
        .expect("DenseMatrix dimensions m*n overflow usize")
}

impl DenseMatrix {
    /// Create a new dense matrix with the given dimensions. All elements are zeroed.
    ///
    /// # Panics
    /// Panics if `m` or `n` is negative, or if `m * n` would overflow `usize`.
    pub fn new(m: i64, n: i64) -> Self {
        let size = checked_dim_size(m, n);
        if size == 0 {
            return DenseMatrix {
                ptr: std::ptr::null_mut(),
                m,
                n,
                size: 0,
            };
        }
        let layout = Layout::array::<f32>(size)
            .and_then(|l| l.align_to(ALIGNMENT))
            .expect("Invalid layout");
        // SAFETY: layout has non-zero size and valid alignment.
        let ptr = unsafe { alloc::alloc_zeroed(layout) as *mut f32 };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        DenseMatrix { ptr, m, n, size }
    }

    /// Create a `DenseMatrix` from existing data.
    ///
    /// Allocates a new aligned buffer and copies `data` into it.
    /// `data.len()` must equal `m * n`.
    pub fn from_data(m: i64, n: i64, data: &[f32]) -> Self {
        let mut dm = DenseMatrix::new(m, n);
        let size = (m as usize) * (n as usize);
        if size > 0 {
            dm.data_mut().copy_from_slice(&data[..size]);
        }
        dm
    }

    /// Return a slice of the entire matrix data in row-major order.
    #[inline]
    pub fn data(&self) -> &[f32] {
        if self.size == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for `m*n` f32 elements and is properly aligned.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Return a mutable slice of the entire matrix data in row-major order.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        if self.size == 0 {
            return &mut [];
        }
        // SAFETY: ptr is valid for `m*n` f32 elements and is properly aligned.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Access element at (i, j).
    #[inline]
    pub fn at(&self, i: i64, j: i64) -> f32 {
        // Convert to usize before multiplying to avoid i64 overflow.
        // The constructor invariant guarantees self.n >= 0 and fits in usize.
        let idx = (i as usize) * (self.n as usize) + (j as usize);
        self.data()[idx]
    }

    /// Set element at (i, j).
    #[inline]
    pub fn at_mut(&mut self, i: i64, j: i64) -> &mut f32 {
        // Convert to usize before multiplying to avoid i64 overflow.
        let idx = (i as usize) * (self.n as usize) + (j as usize);
        &mut self.data_mut()[idx]
    }

    /// Return a slice for row `i`.
    #[inline]
    pub fn row(&self, i: i64) -> &[f32] {
        // Convert to usize before multiplying to avoid i64 overflow.
        let start = (i as usize) * (self.n as usize);
        let end = start + (self.n as usize);
        &self.data()[start..end]
    }

    /// Return a mutable slice for row `i`.
    #[inline]
    pub fn row_mut(&mut self, i: i64) -> &mut [f32] {
        // Convert to usize before multiplying to avoid i64 overflow.
        let start = (i as usize) * (self.n as usize);
        let end = start + (self.n as usize);
        &mut self.data_mut()[start..end]
    }

    /// Performs a lock-free (Hogwild!) SGD update: `row[i] += scale * vec`.
    ///
    /// This method takes `&self` (not `&mut self`) and writes to the matrix data
    /// through raw pointers, deliberately circumventing Rust's aliasing rules so
    /// that multiple threads can update shared weight matrices concurrently —
    /// exactly as C++ fastText does.
    ///
    /// # Safety
    ///
    /// * The caller must ensure that concurrent writes to overlapping rows are
    ///   acceptable (as in Hogwild! SGD, where occasional data races on
    ///   individual `f32` values are tolerated for performance).
    /// * `i` must be a valid row index: `0 <= i < self.rows()`.
    /// * `vec.len()` must equal `self.cols()`.
    /// * The matrix must be alive (held via `Arc`) for the duration of the call.
    /// * Only `f32` element writes may occur — no structural mutations (resize,
    ///   etc.).
    pub unsafe fn add_vector_to_row_unsync(&self, vec: &Vector, i: i64, scale: f32) {
        debug_assert!(i >= 0 && i < self.m, "Row index out of bounds");
        debug_assert_eq!(vec.len(), self.n as usize);
        let n = self.n as usize;
        let start = (i as usize) * n;
        let src = vec.data();
        // Use raw pointer writes instead of creating a &mut slice, which would
        // be instant UB when called concurrently from Hogwild! threads.
        for (j, &s) in src[..n].iter().enumerate() {
            let p = self.ptr.add(start + j);
            p.write(p.read() + scale * s);
        }
    }

    /// Set all elements to zero.
    pub fn zero(&mut self) {
        let data = self.data_mut();
        for v in data.iter_mut() {
            *v = 0.0;
        }
    }

    /// Initialize matrix with uniform random values in [-a, a].
    ///
    /// Uses the C++ fastText approach: divide data into 10 blocks,
    /// each seeded with `block_index + seed` using minstd_rand (LCG).
    pub fn uniform(&mut self, a: f32, seed: i32) {
        let total = self.data().len();
        if total == 0 {
            return;
        }
        let block_size = total / 10;
        let data = self.data_mut();

        for block in 0..10 {
            let start = block_size * block;
            let end = if block == 9 {
                total
            } else {
                (block_size * (block + 1)).min(total)
            };
            // Seed matches C++: std::minstd_rand rng(block + seed)
            let mut rng = MinstdRng::new((block as u64).wrapping_add(seed as u32 as u64));
            for item in data.iter_mut().take(end).skip(start) {
                // uniform_real() returns [0, 1), map to [-a, a]
                let u = rng.uniform_real();
                *item = (u * 2.0 * a as f64 - a as f64) as f32;
            }
        }
    }

    /// Compute the L2 norm of row `i`.
    ///
    /// Returns `Err(EncounteredNaN)` if the result is NaN.
    pub fn l2_norm_row(&self, i: i64) -> Result<f32> {
        assert!(i >= 0 && i < self.m, "Row index out of bounds");
        let row = self.row(i);
        let mut norm = 0.0f64; // Use f64 accumulator matching C++ `auto norm = 0.0`
        for &val in row {
            norm += (val as f64) * (val as f64);
        }
        if norm.is_nan() {
            return Err(FastTextError::EncounteredNaN);
        }
        Ok((norm.sqrt()) as f32)
    }

    /// Multiply rows by the corresponding values in `nums`.
    ///
    /// Rows from `ib` to `ie` (exclusive) are scaled: `row[i] *= nums[i - ib]`.
    /// If `ie` is `None`, all rows from `ib` to the end of the matrix are processed.
    /// If a value in `nums` is 0, the row is skipped (left unchanged).
    /// Matches C++ `DenseMatrix::multiplyRow`.
    pub fn multiply_row(&mut self, nums: &[f32], ib: i64, ie: Option<i64>) {
        let ie = ie.unwrap_or(self.m);
        for i in ib..ie {
            let n = nums[(i - ib) as usize];
            if n != 0.0 {
                let row = self.row_mut(i);
                for v in row.iter_mut() {
                    *v *= n;
                }
            }
        }
    }

    /// Divide rows by the corresponding values in `denoms`.
    ///
    /// Rows from `ib` to `ie` (exclusive) are divided: `row[i] /= denoms[i - ib]`.
    /// If `ie` is `None`, all rows from `ib` to the end of the matrix are processed.
    /// If a value in `denoms` is 0, the row is left unchanged.
    /// Matches C++ `DenseMatrix::divideRow`.
    pub fn divide_row(&mut self, denoms: &[f32], ib: i64, ie: Option<i64>) {
        let ie = ie.unwrap_or(self.m);
        for i in ib..ie {
            let n = denoms[(i - ib) as usize];
            if n != 0.0 {
                let row = self.row_mut(i);
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
        }
    }
}

impl Clone for DenseMatrix {
    fn clone(&self) -> Self {
        let mut m = DenseMatrix::new(self.m, self.n);
        if !self.data().is_empty() {
            m.data_mut().copy_from_slice(self.data());
        }
        m
    }
}

impl Drop for DenseMatrix {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.size > 0 {
            let layout = Layout::array::<f32>(self.size)
                .and_then(|l| l.align_to(ALIGNMENT))
                .expect("Invalid layout in Drop");
            // SAFETY: ptr was allocated with this layout in new().
            unsafe {
                alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

// Matrix trait implementation for DenseMatrix

impl Matrix for DenseMatrix {
    #[inline]
    fn rows(&self) -> i64 {
        self.m
    }

    #[inline]
    fn cols(&self) -> i64 {
        self.n
    }

    fn dot_row(&self, vec: &Vector, i: i64) -> f32 {
        assert!(i >= 0 && i < self.m, "Row index out of bounds");
        assert_eq!(
            vec.len(),
            self.n as usize,
            "Vector size {} does not match matrix columns {}",
            vec.len(),
            self.n
        );
        let row = self.row(i);
        crate::simd::dot_impl(row, vec.data())
    }

    fn add_vector_to_row(&mut self, vec: &Vector, i: i64, scale: f32) {
        assert!(i >= 0 && i < self.m, "Row index out of bounds");
        assert_eq!(
            vec.len(),
            self.n as usize,
            "Vector size {} does not match matrix columns {}",
            vec.len(),
            self.n
        );
        let row = self.row_mut(i);
        crate::simd::add_vector_impl(row, vec.data(), scale);
    }

    fn add_row_to_vector(&self, x: &mut Vector, i: i32, scale: f32) {
        assert!(i >= 0 && (i as i64) < self.m, "Row index out of bounds");
        assert_eq!(
            x.len(),
            self.n as usize,
            "Vector size {} does not match matrix columns {}",
            x.len(),
            self.n
        );
        let row = self.row(i as i64);
        crate::simd::add_vector_impl(x.data_mut(), row, scale);
    }

    fn average_rows_to_vector(&self, x: &mut Vector, rows: &[i32]) {
        assert_eq!(
            x.len(),
            self.n as usize,
            "Vector size {} does not match matrix columns {}",
            x.len(),
            self.n
        );

        // Try SIMD-accelerated path for common dimensions
        #[cfg(target_arch = "aarch64")]
        {
            match self.n {
                512 | 256 | 64 | 32 | 16 => {
                    crate::simd::average_rows_fast_neon(x, rows, self);
                    return;
                }
                _ => {}
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            match self.n {
                512 | 256 | 64 | 32 | 16 => {
                    if is_x86_feature_detected!("avx2") {
                        // SAFETY: AVX2 availability verified by is_x86_feature_detected!
                        unsafe { crate::simd::average_rows_fast_avx2(x, rows, self) };
                    } else {
                        crate::simd::average_rows_fast_sse2(x, rows, self);
                    }
                    return;
                }
                _ => {}
            }
        }

        // Scalar fallback
        crate::simd::average_rows_scalar(x, rows, self);
    }

    /// **Note:** Values are written in little-endian byte order.  C++ fastText
    /// writes in native byte order, which is identical on x86/ARM (the only
    /// platforms fastText targets).  Big-endian platforms are not supported.
    fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        utils::write_i64(writer, self.m)?;
        utils::write_i64(writer, self.n)?;
        let data = self.data();
        for &val in data {
            utils::write_f32(writer, val)?;
        }
        Ok(())
    }

    fn load<R: Read>(reader: &mut R) -> Result<Self> {
        let m = utils::read_i64(reader)?;
        let n = utils::read_i64(reader)?;
        if m < 0 || n < 0 {
            return Err(FastTextError::InvalidModel(format!(
                "Invalid matrix dimensions: {}x{}",
                m, n
            )));
        }
        // Validate that m*n doesn't overflow usize before allocating.
        let m_u = usize::try_from(m).map_err(|_| {
            FastTextError::InvalidModel(format!("Matrix row count {} is too large", m))
        })?;
        let n_u = usize::try_from(n).map_err(|_| {
            FastTextError::InvalidModel(format!("Matrix column count {} is too large", n))
        })?;
        m_u.checked_mul(n_u).ok_or_else(|| {
            FastTextError::InvalidModel(format!(
                "Matrix dimensions {}x{} would overflow usize",
                m, n
            ))
        })?;
        let mut mat = DenseMatrix::new(m, n);
        let data = mat.data_mut();
        // Bulk-read the entire matrix as raw bytes, then reinterpret as f32.
        // This is safe because f32 has no invalid bit patterns and the data
        // is stored as little-endian (matching all supported platforms).
        let byte_slice =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) };
        reader
            .read_exact(byte_slice)
            .map_err(FastTextError::IoError)?;
        // On big-endian platforms we would need to byte-swap here, but
        // fastText only targets little-endian (x86/ARM).
        Ok(mat)
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{add_vector_scalar, dot_scalar};
    #[cfg(target_arch = "x86_64")]
    use crate::simd::{
        add_vector_simd_avx2, average_rows_fast_avx2, average_rows_scalar, dot_simd_avx2,
    };
    use std::io::Cursor;

    #[test]
    fn test_dense_matrix_alloc_safety_zero_size() {
        // Allocation with zero rows/cols must not panic.
        let m1 = DenseMatrix::new(0, 0);
        assert_eq!(m1.rows(), 0);
        assert_eq!(m1.cols(), 0);
        assert!(m1.data().is_empty());

        let m2 = DenseMatrix::new(0, 100);
        assert_eq!(m2.rows(), 0);
        assert_eq!(m2.cols(), 100);
        assert!(m2.data().is_empty());

        let m3 = DenseMatrix::new(100, 0);
        assert_eq!(m3.rows(), 100);
        assert_eq!(m3.cols(), 0);
        assert!(m3.data().is_empty());

        // Clone of zero-size matrix must also work without panic.
        let m4 = m1.clone();
        assert_eq!(m4.rows(), 0);
        assert_eq!(m4.cols(), 0);
    }

    #[test]
    fn test_dense_matrix_layout_overflow_check() {
        // Layout::array::<f32>(usize::MAX) must fail (checked arithmetic).
        assert!(Layout::array::<f32>(usize::MAX).is_err());
        // A very large size just over isize::MAX / 4 should also fail.
        let large = (isize::MAX as usize / std::mem::size_of::<f32>()) + 1;
        assert!(Layout::array::<f32>(large).is_err());
    }

    #[test]
    #[should_panic(expected = "must be non-negative")]
    fn test_dense_matrix_new_negative_m_panics() {
        // Negative row count must panic with a descriptive message.
        let _ = DenseMatrix::new(-1, 4);
    }

    #[test]
    #[should_panic(expected = "must be non-negative")]
    fn test_dense_matrix_new_negative_n_panics() {
        // Negative column count must panic with a descriptive message.
        let _ = DenseMatrix::new(4, -1);
    }

    #[test]
    fn test_dense_matrix_load_overflow_error() {
        // A binary blob with m and n values whose product would overflow usize
        // should be rejected with an InvalidModel error rather than panicking.
        use std::io::Cursor;

        // i64::MAX for both m and n — product (i64::MAX)^2 massively overflows usize.
        let m_val: i64 = i64::MAX;
        let n_val: i64 = i64::MAX;
        // Write header bytes (little-endian i64 m, then i64 n)
        let mut buf = Vec::new();
        buf.extend_from_slice(&m_val.to_le_bytes());
        buf.extend_from_slice(&n_val.to_le_bytes());
        // No data bytes needed — load() should fail at dimension validation.
        let mut cursor = Cursor::new(buf);
        let result = DenseMatrix::load(&mut cursor);
        assert!(
            result.is_err(),
            "Expected error for overflow dimensions, got Ok"
        );
        match result {
            Err(FastTextError::InvalidModel(_)) => {} // expected
            other => panic!("Expected InvalidModel error, got {:?}", other),
        }
    }

    #[test]
    fn test_dense_matrix_load_negative_dims_error() {
        // Negative dimensions in binary format must be rejected.
        use std::io::Cursor;
        let m_val: i64 = -5i64;
        let n_val: i64 = 10i64;
        let mut buf = Vec::new();
        buf.extend_from_slice(&m_val.to_le_bytes());
        buf.extend_from_slice(&n_val.to_le_bytes());
        let mut cursor = Cursor::new(buf);
        let result = DenseMatrix::load(&mut cursor);
        match result {
            Err(FastTextError::InvalidModel(_)) => {} // expected
            other => panic!("Expected InvalidModel error, got {:?}", other),
        }
    }

    #[test]
    fn test_dense_matrix_new() {
        let m = DenseMatrix::new(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        // All elements should be zero
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(m.at(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_dense_matrix_new_zero_rows() {
        let m = DenseMatrix::new(0, 4);
        assert_eq!(m.rows(), 0);
        assert_eq!(m.cols(), 4);
        assert!(m.data().is_empty());
    }

    #[test]
    fn test_dense_matrix_new_zero_cols() {
        let m = DenseMatrix::new(3, 0);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 0);
        assert!(m.data().is_empty());
    }

    #[test]
    fn test_dense_matrix_zero() {
        let mut m = DenseMatrix::new(2, 3);
        // Set some values
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;
        *m.at_mut(1, 0) = 4.0;
        *m.at_mut(1, 1) = 5.0;
        *m.at_mut(1, 2) = 6.0;

        m.zero();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(m.at(i, j), 0.0, "Element ({},{}) should be zero", i, j);
            }
        }
    }

    #[test]
    fn test_dense_matrix_row_major_layout() {
        let mut m = DenseMatrix::new(2, 3);
        // Set elements:
        // Row 0: [1, 2, 3]
        // Row 1: [4, 5, 6]
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;
        *m.at_mut(1, 0) = 4.0;
        *m.at_mut(1, 1) = 5.0;
        *m.at_mut(1, 2) = 6.0;

        // In row-major order, data should be [1, 2, 3, 4, 5, 6]
        let data = m.data();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 3.0);
        assert_eq!(data[3], 4.0);
        assert_eq!(data[4], 5.0);
        assert_eq!(data[5], 6.0);
    }

    #[test]
    fn test_dense_matrix_dot_row() {
        let mut m = DenseMatrix::new(2, 3);
        // Row 0: [1, 2, 3]
        // Row 1: [4, 5, 6]
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;
        *m.at_mut(1, 0) = 4.0;
        *m.at_mut(1, 1) = 5.0;
        *m.at_mut(1, 2) = 6.0;

        let mut v = Vector::new(3);
        v[0] = 1.0;
        v[1] = 1.0;
        v[2] = 1.0;

        // dot(row0, v) = 1+2+3 = 6
        assert!((m.dot_row(&v, 0) - 6.0).abs() < 1e-6);
        // dot(row1, v) = 4+5+6 = 15
        assert!((m.dot_row(&v, 1) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_dot_row_known_values() {
        let mut m = DenseMatrix::new(1, 4);
        *m.at_mut(0, 0) = 2.0;
        *m.at_mut(0, 1) = 3.0;
        *m.at_mut(0, 2) = 4.0;
        *m.at_mut(0, 3) = 5.0;

        let mut v = Vector::new(4);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;
        v[3] = 4.0;

        // 2*1 + 3*2 + 4*3 + 5*4 = 2 + 6 + 12 + 20 = 40
        assert!((m.dot_row(&v, 0) - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_dot_row_zero_vec() {
        let mut m = DenseMatrix::new(1, 3);
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;

        let v = Vector::new(3); // all zeros
        assert_eq!(m.dot_row(&v, 0), 0.0);
    }

    #[test]
    fn test_dense_matrix_dot_row_nan() {
        let mut m = DenseMatrix::new(1, 3);
        *m.at_mut(0, 0) = f32::NAN;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;

        let mut v = Vector::new(3);
        v[0] = 1.0;
        v[1] = 1.0;
        v[2] = 1.0;

        assert!(m.dot_row(&v, 0).is_nan());
    }

    #[test]
    fn test_dense_matrix_add_vector_to_row() {
        let mut m = DenseMatrix::new(2, 3);
        // Row 0: [1, 2, 3]
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;

        let mut v = Vector::new(3);
        v[0] = 10.0;
        v[1] = 20.0;
        v[2] = 30.0;

        m.add_vector_to_row(&v, 0, 1.0);

        assert!((m.at(0, 0) - 11.0).abs() < 1e-6);
        assert!((m.at(0, 1) - 22.0).abs() < 1e-6);
        assert!((m.at(0, 2) - 33.0).abs() < 1e-6);

        // Row 1 should be unchanged
        assert_eq!(m.at(1, 0), 0.0);
        assert_eq!(m.at(1, 1), 0.0);
        assert_eq!(m.at(1, 2), 0.0);
    }

    #[test]
    fn test_dense_matrix_add_vector_to_row_scaled() {
        let mut m = DenseMatrix::new(1, 3);
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;

        let mut v = Vector::new(3);
        v[0] = 10.0;
        v[1] = 20.0;
        v[2] = 30.0;

        m.add_vector_to_row(&v, 0, 0.5);

        assert!((m.at(0, 0) - 6.0).abs() < 1e-6);
        assert!((m.at(0, 1) - 12.0).abs() < 1e-6);
        assert!((m.at(0, 2) - 18.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_add_row_to_vector() {
        let mut m = DenseMatrix::new(2, 3);
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;
        *m.at_mut(1, 0) = 4.0;
        *m.at_mut(1, 1) = 5.0;
        *m.at_mut(1, 2) = 6.0;

        let mut v = Vector::new(3);
        v[0] = 10.0;
        v[1] = 20.0;
        v[2] = 30.0;

        m.add_row_to_vector(&mut v, 0, 1.0);
        assert!((v[0] - 11.0).abs() < 1e-6);
        assert!((v[1] - 22.0).abs() < 1e-6);
        assert!((v[2] - 33.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_add_row_to_vector_scaled() {
        let mut m = DenseMatrix::new(1, 3);
        *m.at_mut(0, 0) = 2.0;
        *m.at_mut(0, 1) = 4.0;
        *m.at_mut(0, 2) = 6.0;

        let mut v = Vector::new(3);
        v[0] = 1.0;
        v[1] = 1.0;
        v[2] = 1.0;

        m.add_row_to_vector(&mut v, 0, 0.5);
        assert!((v[0] - 2.0).abs() < 1e-6);
        assert!((v[1] - 3.0).abs() < 1e-6);
        assert!((v[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_average_rows_single() {
        let mut m = DenseMatrix::new(3, 3);
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;

        let mut v = Vector::new(3);
        m.average_rows_to_vector(&mut v, &[0]);

        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 2.0).abs() < 1e-6);
        assert!((v[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_average_rows_multiple() {
        let mut m = DenseMatrix::new(3, 3);
        // Row 0: [1, 2, 3]
        // Row 1: [4, 5, 6]
        // Row 2: [7, 8, 9]
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;
        *m.at_mut(1, 0) = 4.0;
        *m.at_mut(1, 1) = 5.0;
        *m.at_mut(1, 2) = 6.0;
        *m.at_mut(2, 0) = 7.0;
        *m.at_mut(2, 1) = 8.0;
        *m.at_mut(2, 2) = 9.0;

        let mut v = Vector::new(3);
        m.average_rows_to_vector(&mut v, &[0, 1, 2]);

        // Average: [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3] = [4, 5, 6]
        assert!((v[0] - 4.0).abs() < 1e-5);
        assert!((v[1] - 5.0).abs() < 1e-5);
        assert!((v[2] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_dense_matrix_average_rows_subset() {
        let mut m = DenseMatrix::new(4, 3);
        // Row 0: [1, 0, 0]
        // Row 1: [0, 1, 0]
        // Row 2: [0, 0, 1]
        // Row 3: [1, 1, 1]
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(1, 1) = 1.0;
        *m.at_mut(2, 2) = 1.0;
        *m.at_mut(3, 0) = 1.0;
        *m.at_mut(3, 1) = 1.0;
        *m.at_mut(3, 2) = 1.0;

        let mut v = Vector::new(3);
        // Average rows 0 and 3: [(1+1)/2, (0+1)/2, (0+1)/2] = [1, 0.5, 0.5]
        m.average_rows_to_vector(&mut v, &[0, 3]);

        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 0.5).abs() < 1e-6);
        assert!((v[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_average_rows_empty() {
        let m = DenseMatrix::new(3, 3);
        let mut v = Vector::new(3);
        v[0] = 999.0; // should be zeroed
        m.average_rows_to_vector(&mut v, &[]);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn test_dense_matrix_l2_norm_row() {
        let mut m = DenseMatrix::new(2, 2);
        *m.at_mut(0, 0) = 3.0;
        *m.at_mut(0, 1) = 4.0;
        *m.at_mut(1, 0) = 0.0;
        *m.at_mut(1, 1) = 0.0;

        assert!((m.l2_norm_row(0).unwrap() - 5.0).abs() < 1e-6);
        assert_eq!(m.l2_norm_row(1).unwrap(), 0.0);
    }

    #[test]
    fn test_dense_matrix_l2_norm_row_nan_detection() {
        let mut m = DenseMatrix::new(1, 3);
        *m.at_mut(0, 0) = f32::NAN;
        *m.at_mut(0, 1) = 1.0;
        *m.at_mut(0, 2) = 2.0;

        match m.l2_norm_row(0) {
            Err(FastTextError::EncounteredNaN) => {} // expected
            other => panic!("Expected EncounteredNaN, got {:?}", other),
        }
    }

    #[test]
    fn test_dense_matrix_multiply_row() {
        let mut m = DenseMatrix::new(3, 2);
        // Row 0: [1, 2]
        // Row 1: [3, 4]
        // Row 2: [5, 6]
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(1, 0) = 3.0;
        *m.at_mut(1, 1) = 4.0;
        *m.at_mut(2, 0) = 5.0;
        *m.at_mut(2, 1) = 6.0;

        let nums = [2.0, 3.0, 0.5];
        m.multiply_row(&nums, 0, Some(3));

        assert!((m.at(0, 0) - 2.0).abs() < 1e-6);
        assert!((m.at(0, 1) - 4.0).abs() < 1e-6);
        assert!((m.at(1, 0) - 9.0).abs() < 1e-6);
        assert!((m.at(1, 1) - 12.0).abs() < 1e-6);
        assert!((m.at(2, 0) - 2.5).abs() < 1e-6);
        assert!((m.at(2, 1) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_multiply_row_zero_skip() {
        let mut m = DenseMatrix::new(2, 2);
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(1, 0) = 3.0;
        *m.at_mut(1, 1) = 4.0;

        // Zero value should leave row unchanged
        let nums = [0.0, 2.0];
        m.multiply_row(&nums, 0, Some(2));

        assert_eq!(m.at(0, 0), 1.0); // unchanged
        assert_eq!(m.at(0, 1), 2.0); // unchanged
        assert!((m.at(1, 0) - 6.0).abs() < 1e-6);
        assert!((m.at(1, 1) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_multiply_row_default_ie() {
        let mut m = DenseMatrix::new(2, 2);
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(1, 0) = 3.0;
        *m.at_mut(1, 1) = 4.0;

        // ie = None means use m_ (all rows)
        let nums = [2.0, 3.0];
        m.multiply_row(&nums, 0, None);

        assert!((m.at(0, 0) - 2.0).abs() < 1e-6);
        assert!((m.at(0, 1) - 4.0).abs() < 1e-6);
        assert!((m.at(1, 0) - 9.0).abs() < 1e-6);
        assert!((m.at(1, 1) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_divide_row() {
        let mut m = DenseMatrix::new(2, 2);
        *m.at_mut(0, 0) = 4.0;
        *m.at_mut(0, 1) = 6.0;
        *m.at_mut(1, 0) = 8.0;
        *m.at_mut(1, 1) = 10.0;

        let denoms = [2.0, 5.0];
        m.divide_row(&denoms, 0, Some(2));

        assert!((m.at(0, 0) - 2.0).abs() < 1e-6);
        assert!((m.at(0, 1) - 3.0).abs() < 1e-6);
        assert!((m.at(1, 0) - 1.6).abs() < 1e-6);
        assert!((m.at(1, 1) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_divide_row_zero_denom() {
        let mut m = DenseMatrix::new(2, 2);
        *m.at_mut(0, 0) = 4.0;
        *m.at_mut(0, 1) = 6.0;
        *m.at_mut(1, 0) = 8.0;
        *m.at_mut(1, 1) = 10.0;

        // Zero denominator should leave the row unchanged
        let denoms = [0.0, 2.0];
        m.divide_row(&denoms, 0, Some(2));

        assert_eq!(m.at(0, 0), 4.0); // unchanged
        assert_eq!(m.at(0, 1), 6.0); // unchanged
        assert!((m.at(1, 0) - 4.0).abs() < 1e-6);
        assert!((m.at(1, 1) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_divide_row_default_ie() {
        let mut m = DenseMatrix::new(2, 2);
        *m.at_mut(0, 0) = 4.0;
        *m.at_mut(0, 1) = 6.0;
        *m.at_mut(1, 0) = 8.0;
        *m.at_mut(1, 1) = 10.0;

        let denoms = [2.0, 4.0];
        m.divide_row(&denoms, 0, None);

        assert!((m.at(0, 0) - 2.0).abs() < 1e-6);
        assert!((m.at(0, 1) - 3.0).abs() < 1e-6);
        assert!((m.at(1, 0) - 2.0).abs() < 1e-6);
        assert!((m.at(1, 1) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_simd_consistency() {
        // VAL-CORE-011: SIMD-accelerated matrix operations must match scalar fallback
        for &dim in &[16, 32, 64, 256, 512] {
            let mut m = DenseMatrix::new(3, dim);
            let n = dim as usize;

            // Fill matrix with deterministic values
            for i in 0..3 {
                for j in 0..n {
                    *m.at_mut(i as i64, j as i64) = ((i * n + j) as f32) * 0.01;
                }
            }

            let mut v = Vector::new(n);
            for j in 0..n {
                v[j] = ((n - j) as f32) * 0.01;
            }

            // Test dot_row: compare against scalar
            let simd_dot = m.dot_row(&v, 1);
            let row1 = m.row(1);
            let scalar_dot = dot_scalar(row1, v.data());

            let magnitude = simd_dot.abs().max(scalar_dot.abs()).max(1.0);
            let tolerance = magnitude * f32::EPSILON * n as f32;
            assert!(
                (simd_dot - scalar_dot).abs() < tolerance,
                "dot_row SIMD vs scalar mismatch for dim={}: SIMD={}, scalar={}",
                dim,
                simd_dot,
                scalar_dot,
            );

            // Test add_vector_to_row: compare SIMD and scalar results
            let mut m_simd = m.clone();
            let mut m_scalar = m.clone();

            m_simd.add_vector_to_row(&v, 0, 0.5);
            // Scalar add
            let row0_scalar = m_scalar.row_mut(0);
            add_vector_scalar(row0_scalar, v.data(), 0.5);

            for j in 0..n {
                let s = m_simd.at(0, j as i64);
                let sc = m_scalar.at(0, j as i64);
                let mag = s.abs().max(sc.abs()).max(1.0);
                let tol = mag * f32::EPSILON * 4.0;
                assert!(
                    (s - sc).abs() < tol,
                    "add_vector_to_row mismatch at j={} for dim={}: SIMD={}, scalar={}",
                    j,
                    dim,
                    s,
                    sc,
                );
            }

            // Test add_row_to_vector: compare SIMD and scalar results
            let mut v_simd = Vector::new(n);
            let mut v_scalar = Vector::new(n);
            for j in 0..n {
                v_simd[j] = j as f32 * 0.1;
                v_scalar[j] = j as f32 * 0.1;
            }

            m.add_row_to_vector(&mut v_simd, 2, 0.7);
            add_vector_scalar(v_scalar.data_mut(), m.row(2), 0.7);

            for j in 0..n {
                let s = v_simd[j];
                let sc = v_scalar[j];
                let mag = s.abs().max(sc.abs()).max(1.0);
                let tol = mag * f32::EPSILON * 4.0;
                assert!(
                    (s - sc).abs() < tol,
                    "add_row_to_vector mismatch at j={} for dim={}: SIMD={}, scalar={}",
                    j,
                    dim,
                    s,
                    sc,
                );
            }

            // Test average_rows_to_vector
            let mut v_avg_simd = Vector::new(n);
            let mut v_avg_scalar = Vector::new(n);

            m.average_rows_to_vector(&mut v_avg_simd, &[0, 1, 2]);

            // Scalar average
            v_avg_scalar.zero();
            for i in 0..3 {
                add_vector_scalar(v_avg_scalar.data_mut(), m.row(i), 1.0);
            }
            v_avg_scalar.mul(1.0 / 3.0);

            for j in 0..n {
                let s = v_avg_simd[j];
                let sc = v_avg_scalar[j];
                let mag = s.abs().max(sc.abs()).max(1.0);
                let tol = mag * f32::EPSILON * n as f32;
                assert!(
                    (s - sc).abs() < tol,
                    "average_rows_to_vector mismatch at j={} for dim={}: SIMD={}, scalar={}",
                    j,
                    dim,
                    s,
                    sc,
                );
            }
        }
    }

    #[test]
    fn test_dense_matrix_alignment() {
        for &(rows, cols) in &[(1, 1), (1, 16), (4, 64), (10, 100), (100, 256), (8, 512)] {
            let m = DenseMatrix::new(rows, cols);
            let ptr_addr = m.data().as_ptr() as usize;
            assert_eq!(
                ptr_addr % ALIGNMENT,
                0,
                "DenseMatrix {}x{} is not 64-byte aligned (addr: 0x{:x})",
                rows,
                cols,
                ptr_addr,
            );
        }
    }

    #[test]
    #[should_panic(expected = "Row index out of bounds")]
    fn test_dense_matrix_dot_row_out_of_bounds() {
        let m = DenseMatrix::new(2, 3);
        let v = Vector::new(3);
        let _ = m.dot_row(&v, 2);
    }

    #[test]
    #[should_panic(expected = "Vector size")]
    fn test_dense_matrix_dot_row_size_mismatch() {
        let m = DenseMatrix::new(2, 3);
        let v = Vector::new(4);
        let _ = m.dot_row(&v, 0);
    }

    #[test]
    #[should_panic(expected = "Row index out of bounds")]
    fn test_dense_matrix_add_vector_to_row_out_of_bounds() {
        let mut m = DenseMatrix::new(2, 3);
        let v = Vector::new(3);
        m.add_vector_to_row(&v, 2, 1.0);
    }

    #[test]
    #[should_panic(expected = "Row index out of bounds")]
    fn test_dense_matrix_add_row_to_vector_out_of_bounds() {
        let m = DenseMatrix::new(2, 3);
        let mut v = Vector::new(3);
        m.add_row_to_vector(&mut v, 2, 1.0);
    }

    #[test]
    #[should_panic(expected = "Row index out of bounds")]
    fn test_dense_matrix_l2_norm_row_out_of_bounds() {
        let m = DenseMatrix::new(2, 3);
        let _ = m.l2_norm_row(2);
    }

    #[test]
    fn test_dense_matrix_save_load_roundtrip() {
        let mut m = DenseMatrix::new(3, 4);
        for i in 0..3 {
            for j in 0..4 {
                *m.at_mut(i, j) = (i * 4 + j) as f32 * 1.5;
            }
        }

        // Save
        let mut buf = Vec::new();
        m.save(&mut buf).unwrap();

        // Expected size: 8 (m) + 8 (n) + 12*4 (data) = 64 bytes
        assert_eq!(buf.len(), 8 + 8 + 3 * 4 * 4);

        // Load
        let mut cursor = Cursor::new(&buf);
        let loaded = DenseMatrix::load(&mut cursor).unwrap();

        assert_eq!(loaded.rows(), 3);
        assert_eq!(loaded.cols(), 4);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(loaded.at(i, j), m.at(i, j), "Mismatch at ({}, {})", i, j,);
            }
        }
    }

    #[test]
    fn test_dense_matrix_save_load_empty() {
        let m = DenseMatrix::new(0, 0);

        let mut buf = Vec::new();
        m.save(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = DenseMatrix::load(&mut cursor).unwrap();

        assert_eq!(loaded.rows(), 0);
        assert_eq!(loaded.cols(), 0);
    }

    #[test]
    fn test_dense_matrix_save_load_header_format() {
        // Verify the binary format: m(i64), n(i64), then m*n f32 values
        let mut m = DenseMatrix::new(2, 3);
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;
        *m.at_mut(1, 0) = 4.0;
        *m.at_mut(1, 1) = 5.0;
        *m.at_mut(1, 2) = 6.0;

        let mut buf = Vec::new();
        m.save(&mut buf).unwrap();

        // Read header manually
        let mut cursor = Cursor::new(&buf);
        let rows = utils::read_i64(&mut cursor).unwrap();
        let cols = utils::read_i64(&mut cursor).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(cols, 3);

        // Read data
        for expected in &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            let val = utils::read_f32(&mut cursor).unwrap();
            assert_eq!(val, *expected);
        }
    }

    #[test]
    fn test_dense_matrix_load_truncated() {
        // Too-short buffer should fail
        let buf = vec![0u8; 4]; // Only 4 bytes, need at least 16 for header
        let mut cursor = Cursor::new(&buf);
        assert!(DenseMatrix::load(&mut cursor).is_err());
    }

    #[test]
    fn test_dense_matrix_uniform() {
        let mut m = DenseMatrix::new(10, 20);
        m.uniform(0.5, 0);

        // Check all values are in [-0.5, 0.5]
        let data = m.data();
        for &val in data {
            assert!(
                val >= -0.5 && val <= 0.5,
                "Uniform value {} out of range [-0.5, 0.5]",
                val,
            );
        }

        // Check not all zeros (should have some non-zero values)
        let non_zero_count = data.iter().filter(|&&v| v != 0.0).count();
        assert!(non_zero_count > 0, "All values are zero after uniform init");
    }

    #[test]
    fn test_dense_matrix_uniform_deterministic() {
        let mut m1 = DenseMatrix::new(10, 20);
        let mut m2 = DenseMatrix::new(10, 20);
        m1.uniform(0.5, 42);
        m2.uniform(0.5, 42);

        assert_eq!(m1.data(), m2.data(), "Same seed should produce same values");
    }

    #[test]
    fn test_dense_matrix_uniform_different_seeds() {
        let mut m1 = DenseMatrix::new(10, 20);
        let mut m2 = DenseMatrix::new(10, 20);
        m1.uniform(0.5, 42);
        m2.uniform(0.5, 43);

        assert_ne!(
            m1.data(),
            m2.data(),
            "Different seeds should produce different values"
        );
    }

    #[test]
    fn test_dense_matrix_clone() {
        let mut m = DenseMatrix::new(2, 3);
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(1, 2) = 5.0;

        let m2 = m.clone();
        assert_eq!(m2.rows(), 2);
        assert_eq!(m2.cols(), 3);
        assert_eq!(m2.at(0, 0), 1.0);
        assert_eq!(m2.at(1, 2), 5.0);

        // Verify independence
        *m.at_mut(0, 0) = 99.0;
        assert_eq!(m2.at(0, 0), 1.0);
    }

    #[test]
    fn test_dense_matrix_row_access() {
        let mut m = DenseMatrix::new(2, 3);
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;
        *m.at_mut(1, 0) = 4.0;
        *m.at_mut(1, 1) = 5.0;
        *m.at_mut(1, 2) = 6.0;

        let row0 = m.row(0);
        assert_eq!(row0, &[1.0, 2.0, 3.0]);

        let row1 = m.row(1);
        assert_eq!(row1, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_dense_matrix_multiply_row_with_offset() {
        let mut m = DenseMatrix::new(4, 2);
        for i in 0..4 {
            *m.at_mut(i, 0) = (i + 1) as f32;
            *m.at_mut(i, 1) = (i + 1) as f32 * 10.0;
        }

        // Multiply rows 1..3 with nums [2.0, 3.0]
        let nums = [2.0, 3.0];
        m.multiply_row(&nums, 1, Some(3));

        // Row 0: unchanged [1, 10]
        assert_eq!(m.at(0, 0), 1.0);
        assert_eq!(m.at(0, 1), 10.0);
        // Row 1: [2, 20] * 2.0 = [4, 40]
        assert!((m.at(1, 0) - 4.0).abs() < 1e-6);
        assert!((m.at(1, 1) - 40.0).abs() < 1e-6);
        // Row 2: [3, 30] * 3.0 = [9, 90]
        assert!((m.at(2, 0) - 9.0).abs() < 1e-6);
        assert!((m.at(2, 1) - 90.0).abs() < 1e-6);
        // Row 3: unchanged [4, 40]
        assert_eq!(m.at(3, 0), 4.0);
        assert_eq!(m.at(3, 1), 40.0);
    }

    #[test]
    fn test_dense_matrix_divide_row_with_offset() {
        let mut m = DenseMatrix::new(4, 2);
        for i in 0..4 {
            *m.at_mut(i, 0) = (i + 1) as f32 * 10.0;
            *m.at_mut(i, 1) = (i + 1) as f32 * 20.0;
        }

        // Divide rows 2..4 with denoms [5.0, 10.0]
        let denoms = [5.0, 10.0];
        m.divide_row(&denoms, 2, Some(4));

        // Row 0: unchanged [10, 20]
        assert_eq!(m.at(0, 0), 10.0);
        assert_eq!(m.at(0, 1), 20.0);
        // Row 1: unchanged [20, 40]
        assert_eq!(m.at(1, 0), 20.0);
        assert_eq!(m.at(1, 1), 40.0);
        // Row 2: [30, 60] / 5.0 = [6, 12]
        assert!((m.at(2, 0) - 6.0).abs() < 1e-6);
        assert!((m.at(2, 1) - 12.0).abs() < 1e-6);
        // Row 3: [40, 80] / 10.0 = [4, 8]
        assert!((m.at(3, 0) - 4.0).abs() < 1e-6);
        assert!((m.at(3, 1) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_dot_row_nan_in_vector() {
        let mut m = DenseMatrix::new(1, 3);
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;

        let mut v = Vector::new(3);
        v[0] = 1.0;
        v[1] = f32::NAN;
        v[2] = 1.0;

        assert!(m.dot_row(&v, 0).is_nan());
    }

    #[test]
    fn test_dense_matrix_large_dot_row() {
        let dim = 100;
        let mut m = DenseMatrix::new(10, dim);
        for i in 0..10 {
            for j in 0..dim as usize {
                *m.at_mut(i, j as i64) = 1.0;
            }
        }
        let mut v = Vector::new(dim as usize);
        for j in 0..dim as usize {
            v[j] = 1.0;
        }
        // dot of 100 ones with 100 ones = 100
        assert!((m.dot_row(&v, 0) - 100.0).abs() < 0.01);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_dense_matrix_avx2_consistency() {
        if !is_x86_feature_detected!("avx2") {
            // AVX2 not available on this CPU, skip runtime checks
            return;
        }
        for &dim in &[16_i64, 32, 64, 256, 512] {
            let mut m = DenseMatrix::new(3, dim);
            let n = dim as usize;

            // Fill matrix with deterministic values
            for i in 0..3 {
                for j in 0..n {
                    *m.at_mut(i as i64, j as i64) = ((i * n + j) as f32) * 0.01;
                }
            }

            let mut v = Vector::new(n);
            for j in 0..n {
                v[j] = ((n - j) as f32) * 0.01;
            }

            // Test dot_simd_avx2 vs scalar for dot_row
            let row1 = m.row(1);
            let avx2_dot = unsafe { dot_simd_avx2(row1, v.data()) };
            let scalar_dot = dot_scalar(row1, v.data());
            let tolerance = scalar_dot.abs().max(avx2_dot.abs()).max(1.0) * f32::EPSILON * n as f32;
            assert!(
                (avx2_dot - scalar_dot).abs() < tolerance,
                "dot_simd_avx2 vs scalar mismatch for dim={}: AVX2={}, scalar={}",
                dim,
                avx2_dot,
                scalar_dot,
            );

            // Test add_vector_simd_avx2 vs scalar for add_vector_to_row
            let mut dest_avx2: Vec<f32> = (0..n).map(|j| j as f32 * 0.5).collect();
            let mut dest_scalar: Vec<f32> = (0..n).map(|j| j as f32 * 0.5).collect();
            let src: Vec<f32> = (0..n).map(|j| j as f32 * 0.1).collect();
            unsafe { add_vector_simd_avx2(&mut dest_avx2, &src, 2.0) };
            add_vector_scalar(&mut dest_scalar, &src, 2.0);
            for j in 0..n {
                let mag = dest_avx2[j].abs().max(dest_scalar[j].abs()).max(1.0);
                let tol = mag * f32::EPSILON * 4.0;
                assert!(
                    (dest_avx2[j] - dest_scalar[j]).abs() < tol,
                    "add_vector_simd_avx2 vs scalar mismatch at j={} for dim={}: AVX2={}, scalar={}",
                    j,
                    dim,
                    dest_avx2[j],
                    dest_scalar[j],
                );
            }

            // Test add_vector_simd_avx2 vs scalar for add_row_to_vector
            let mut v_avx2 = Vector::new(n);
            let mut v_scalar = Vector::new(n);
            for j in 0..n {
                v_avx2[j] = j as f32 * 0.1;
                v_scalar[j] = j as f32 * 0.1;
            }
            let row2 = m.row(2);
            unsafe { add_vector_simd_avx2(v_avx2.data_mut(), row2, 0.7) };
            add_vector_scalar(v_scalar.data_mut(), row2, 0.7);
            for j in 0..n {
                let mag = v_avx2[j].abs().max(v_scalar[j].abs()).max(1.0);
                let tol = mag * f32::EPSILON * 4.0;
                assert!(
                    (v_avx2[j] - v_scalar[j]).abs() < tol,
                    "add_vector_simd_avx2 (add_row) vs scalar mismatch at j={} for dim={}: AVX2={}, scalar={}",
                    j,
                    dim,
                    v_avx2[j],
                    v_scalar[j],
                );
            }

            // Test average_rows_fast_avx2 vs scalar for average_rows_to_vector
            let mut v_avg_avx2 = Vector::new(n);
            let mut v_avg_scalar = Vector::new(n);
            unsafe { average_rows_fast_avx2(&mut v_avg_avx2, &[0, 1, 2], &m) };
            average_rows_scalar(&mut v_avg_scalar, &[0, 1, 2], &m);
            for j in 0..n {
                let mag = v_avg_avx2[j].abs().max(v_avg_scalar[j].abs()).max(1.0);
                let tol = mag * f32::EPSILON * n as f32;
                assert!(
                    (v_avg_avx2[j] - v_avg_scalar[j]).abs() < tol,
                    "average_rows_fast_avx2 vs scalar mismatch at j={} for dim={}: AVX2={}, scalar={}",
                    j,
                    dim,
                    v_avg_avx2[j],
                    v_avg_scalar[j],
                );
            }
        }
    }
}
