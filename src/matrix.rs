// Matrix trait and DenseMatrix implementation
//
// DenseMatrix stores f32 values in row-major order with 64-byte aligned storage,
// matching the C++ fastText DenseMatrix class backed by `intgemm::AlignedVector<real>`.

use std::alloc::{self, Layout};
use std::convert::TryFrom;
use std::io::{Read, Write};

use crate::error::{FastTextError, Result};
use crate::utils;
use crate::vector::Vector;

/// Alignment in bytes for SIMD-friendly memory layout.
const ALIGNMENT: usize = 64;

// ============================================================================
// Matrix trait
// ============================================================================

/// Trait for matrix types, matching the C++ fastText abstract `Matrix` class.
pub trait Matrix {
    /// Return the number of rows.
    fn rows(&self) -> i64;

    /// Return the number of columns.
    fn cols(&self) -> i64;

    /// Compute the dot product of a vector with a specific matrix row.
    ///
    /// Returns `Err(EncounteredNaN)` if the result is NaN.
    fn dot_row(&self, vec: &Vector, i: i64) -> Result<f32>;

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

// ============================================================================
// DenseMatrix
// ============================================================================

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
}

// SAFETY: DenseMatrix owns its allocation and provides &/&mut access through safe methods.
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
        DenseMatrix { ptr, m, n }
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
        // Use checked arithmetic to compute element count.  The constructor
        // guarantees m and n are non-negative and m*n fits in usize, so these
        // operations will never fail for a properly constructed DenseMatrix.
        let size = checked_dim_size(self.m, self.n);
        if size == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for `m*n` f32 elements and is properly aligned.
        unsafe { std::slice::from_raw_parts(self.ptr, size) }
    }

    /// Return a mutable slice of the entire matrix data in row-major order.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        // Use checked arithmetic to compute element count.  The constructor
        // guarantees m and n are non-negative and m*n fits in usize, so these
        // operations will never fail for a properly constructed DenseMatrix.
        let size = checked_dim_size(self.m, self.n);
        if size == 0 {
            return &mut [];
        }
        // SAFETY: ptr is valid for `m*n` f32 elements and is properly aligned.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, size) }
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
        let total = self.data().len(); // uses checked_dim_size internally
        if total == 0 {
            return;
        }
        let block_size = total / 10;
        let data = self.data_mut();

        // Process blocks (matching C++ uniformThread with single thread)
        for block in 0..10 {
            let start = block_size * block;
            let end = if block == 9 {
                total
            } else {
                (block_size * (block + 1)).min(total)
            };
            // Use minstd_rand equivalent: LCG with a=48271, c=0, m=2^31-1
            let mut rng_state = (block as u32).wrapping_add(seed as u32);
            // Seed the LCG
            if rng_state == 0 {
                rng_state = 1;
            }
            for item in data.iter_mut().take(end).skip(start) {
                // minstd_rand: state = (state * 48271) % 2147483647
                rng_state = ((rng_state as u64 * 48271) % 2147483647) as u32;
                // Map to [-a, a]: uniform_real_distribution
                let u = rng_state as f64 / 2147483647.0;
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
    /// If a value in `nums` is 0, the row is skipped (left unchanged).
    /// Matches C++ `DenseMatrix::multiplyRow`.
    pub fn multiply_row(&mut self, nums: &[f32], ib: i64, ie: i64) {
        let ie = if ie == -1 { self.m } else { ie };
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
    /// If a value in `denoms` is 0, the row is left unchanged.
    /// Matches C++ `DenseMatrix::divideRow`.
    pub fn divide_row(&mut self, denoms: &[f32], ib: i64, ie: i64) {
        let ie = if ie == -1 { self.m } else { ie };
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
        if !self.ptr.is_null() {
            // Reconstruct the size using the same checked arithmetic as new().
            // The constructor invariant ensures m,n >= 0 and m*n fits in usize,
            // so these conversions cannot fail for a properly constructed DenseMatrix.
            // Using unwrap_or to avoid a double-panic in drop.
            let m_u = usize::try_from(self.m).unwrap_or(0);
            let n_u = usize::try_from(self.n).unwrap_or(0);
            let size = m_u.checked_mul(n_u).unwrap_or(0);
            if size > 0 {
                let layout = Layout::array::<f32>(size)
                    .and_then(|l| l.align_to(ALIGNMENT))
                    .expect("Invalid layout in Drop");
                // SAFETY: ptr was allocated with this layout in new().
                unsafe {
                    alloc::dealloc(self.ptr as *mut u8, layout);
                }
            }
        }
    }
}

// ============================================================================
// Matrix trait implementation for DenseMatrix
// ============================================================================

impl Matrix for DenseMatrix {
    #[inline]
    fn rows(&self) -> i64 {
        self.m
    }

    #[inline]
    fn cols(&self) -> i64 {
        self.n
    }

    fn dot_row(&self, vec: &Vector, i: i64) -> Result<f32> {
        assert!(i >= 0 && i < self.m, "Row index out of bounds");
        assert_eq!(
            vec.len(),
            self.n as usize,
            "Vector size {} does not match matrix columns {}",
            vec.len(),
            self.n
        );
        let row = self.row(i);
        let d = dot_impl(row, vec.data());
        if d.is_nan() {
            return Err(FastTextError::EncounteredNaN);
        }
        Ok(d)
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
        add_vector_impl(row, vec.data(), scale);
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
        add_vector_impl(x.data_mut(), row, scale);
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
                    average_rows_fast_neon(x, rows, self);
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
                        unsafe { average_rows_fast_avx2(x, rows, self) };
                    } else {
                        average_rows_fast_sse2(x, rows, self);
                    }
                    return;
                }
                _ => {}
            }
        }

        // Scalar fallback
        average_rows_scalar(x, rows, self);
    }

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
        for val in data.iter_mut() {
            *val = utils::read_f32(reader)?;
        }
        Ok(mat)
    }
}

// ============================================================================
// Scalar fallback implementations
// ============================================================================

/// Scalar fallback for dot product of two slices.
#[allow(dead_code)]
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar fallback for dest += scale * src.
#[allow(dead_code)]
#[inline]
fn add_vector_scalar(dest: &mut [f32], src: &[f32], scale: f32) {
    for i in 0..dest.len() {
        dest[i] += scale * src[i];
    }
}

/// Scalar average_rows implementation.
fn average_rows_scalar(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    x.zero();
    for &row_idx in rows {
        let row = mat.row(row_idx as i64);
        add_vector_impl(x.data_mut(), row, 1.0);
    }
    if !rows.is_empty() {
        x.mul(1.0 / rows.len() as f32);
    }
}

// ============================================================================
// SIMD implementations
// ============================================================================

// ---------- aarch64 NEON ----------

#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_simd_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let chunks = n / 16;
    let remainder = n % 16;

    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let a0 = vld1q_f32(a_ptr.add(offset));
            let a1 = vld1q_f32(a_ptr.add(offset + 4));
            let a2 = vld1q_f32(a_ptr.add(offset + 8));
            let a3 = vld1q_f32(a_ptr.add(offset + 12));
            let b0 = vld1q_f32(b_ptr.add(offset));
            let b1 = vld1q_f32(b_ptr.add(offset + 4));
            let b2 = vld1q_f32(b_ptr.add(offset + 8));
            let b3 = vld1q_f32(b_ptr.add(offset + 12));
            sum0 = vfmaq_f32(sum0, a0, b0);
            sum1 = vfmaq_f32(sum1, a1, b1);
            sum2 = vfmaq_f32(sum2, a2, b2);
            sum3 = vfmaq_f32(sum3, a3, b3);
        }

        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);

        let mut result = vaddvq_f32(sum0);

        let rem_start = chunks * 16;
        for i in 0..remainder {
            result += a[rem_start + i] * b[rem_start + i];
        }

        result
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn add_vector_simd_neon(dest: &mut [f32], src: &[f32], scale: f32) {
    use std::arch::aarch64::*;
    let n = dest.len();
    let chunks = n / 16;
    let remainder = n % 16;

    unsafe {
        let scale_v = vdupq_n_f32(scale);
        let d_ptr = dest.as_mut_ptr();
        let s_ptr = src.as_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let d0 = vld1q_f32(d_ptr.add(offset));
            let d1 = vld1q_f32(d_ptr.add(offset + 4));
            let d2 = vld1q_f32(d_ptr.add(offset + 8));
            let d3 = vld1q_f32(d_ptr.add(offset + 12));
            let s0 = vld1q_f32(s_ptr.add(offset));
            let s1 = vld1q_f32(s_ptr.add(offset + 4));
            let s2 = vld1q_f32(s_ptr.add(offset + 8));
            let s3 = vld1q_f32(s_ptr.add(offset + 12));
            vst1q_f32(d_ptr.add(offset), vfmaq_f32(d0, scale_v, s0));
            vst1q_f32(d_ptr.add(offset + 4), vfmaq_f32(d1, scale_v, s1));
            vst1q_f32(d_ptr.add(offset + 8), vfmaq_f32(d2, scale_v, s2));
            vst1q_f32(d_ptr.add(offset + 12), vfmaq_f32(d3, scale_v, s3));
        }

        let rem_start = chunks * 16;
        for i in 0..remainder {
            dest[rem_start + i] += scale * src[rem_start + i];
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn average_rows_fast_neon(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    use std::arch::aarch64::*;

    if rows.is_empty() {
        x.zero();
        return;
    }

    let n = mat.n as usize;

    // First row: copy to x
    let first_row = mat.row(rows[0] as i64);
    x.data_mut().copy_from_slice(first_row);

    // Add remaining rows
    for &row_idx in &rows[1..] {
        let row = mat.row(row_idx as i64);
        add_vector_simd_neon(x.data_mut(), row, 1.0);
    }

    // Multiply by 1.0 / rows.len()
    let scale = 1.0 / rows.len() as f32;
    unsafe {
        let scale_v = vdupq_n_f32(scale);
        let chunks = n / 16;
        let remainder = n % 16;
        let d_ptr = x.data_mut().as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let v0 = vld1q_f32(d_ptr.add(offset));
            let v1 = vld1q_f32(d_ptr.add(offset + 4));
            let v2 = vld1q_f32(d_ptr.add(offset + 8));
            let v3 = vld1q_f32(d_ptr.add(offset + 12));
            vst1q_f32(d_ptr.add(offset), vmulq_f32(v0, scale_v));
            vst1q_f32(d_ptr.add(offset + 4), vmulq_f32(v1, scale_v));
            vst1q_f32(d_ptr.add(offset + 8), vmulq_f32(v2, scale_v));
            vst1q_f32(d_ptr.add(offset + 12), vmulq_f32(v3, scale_v));
        }

        let rem_start = chunks * 16;
        for i in 0..remainder {
            x.data_mut()[rem_start + i] *= scale;
        }
    }
}

// ---------- x86_64 SSE2 ----------

#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_simd_sse2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 16;
    let remainder = n % 16;

    unsafe {
        let mut sum0 = _mm_setzero_ps();
        let mut sum1 = _mm_setzero_ps();
        let mut sum2 = _mm_setzero_ps();
        let mut sum3 = _mm_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let a0 = _mm_loadu_ps(a_ptr.add(offset));
            let a1 = _mm_loadu_ps(a_ptr.add(offset + 4));
            let a2 = _mm_loadu_ps(a_ptr.add(offset + 8));
            let a3 = _mm_loadu_ps(a_ptr.add(offset + 12));
            let b0 = _mm_loadu_ps(b_ptr.add(offset));
            let b1 = _mm_loadu_ps(b_ptr.add(offset + 4));
            let b2 = _mm_loadu_ps(b_ptr.add(offset + 8));
            let b3 = _mm_loadu_ps(b_ptr.add(offset + 12));
            sum0 = _mm_add_ps(sum0, _mm_mul_ps(a0, b0));
            sum1 = _mm_add_ps(sum1, _mm_mul_ps(a1, b1));
            sum2 = _mm_add_ps(sum2, _mm_mul_ps(a2, b2));
            sum3 = _mm_add_ps(sum3, _mm_mul_ps(a3, b3));
        }

        sum0 = _mm_add_ps(sum0, sum1);
        sum2 = _mm_add_ps(sum2, sum3);
        sum0 = _mm_add_ps(sum0, sum2);

        let hi = _mm_movehl_ps(sum0, sum0);
        let s = _mm_add_ps(sum0, hi);
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        let mut result = _mm_cvtss_f32(s);

        let rem_start = chunks * 16;
        for i in 0..remainder {
            result += a[rem_start + i] * b[rem_start + i];
        }

        result
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn add_vector_simd_sse2(dest: &mut [f32], src: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let n = dest.len();
    let chunks = n / 16;
    let remainder = n % 16;

    unsafe {
        let scale_v = _mm_set1_ps(scale);
        let d_ptr = dest.as_mut_ptr();
        let s_ptr = src.as_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let d0 = _mm_loadu_ps(d_ptr.add(offset));
            let d1 = _mm_loadu_ps(d_ptr.add(offset + 4));
            let d2 = _mm_loadu_ps(d_ptr.add(offset + 8));
            let d3 = _mm_loadu_ps(d_ptr.add(offset + 12));
            let s0 = _mm_loadu_ps(s_ptr.add(offset));
            let s1 = _mm_loadu_ps(s_ptr.add(offset + 4));
            let s2 = _mm_loadu_ps(s_ptr.add(offset + 8));
            let s3 = _mm_loadu_ps(s_ptr.add(offset + 12));
            _mm_storeu_ps(d_ptr.add(offset), _mm_add_ps(d0, _mm_mul_ps(scale_v, s0)));
            _mm_storeu_ps(
                d_ptr.add(offset + 4),
                _mm_add_ps(d1, _mm_mul_ps(scale_v, s1)),
            );
            _mm_storeu_ps(
                d_ptr.add(offset + 8),
                _mm_add_ps(d2, _mm_mul_ps(scale_v, s2)),
            );
            _mm_storeu_ps(
                d_ptr.add(offset + 12),
                _mm_add_ps(d3, _mm_mul_ps(scale_v, s3)),
            );
        }

        let rem_start = chunks * 16;
        for i in 0..remainder {
            dest[rem_start + i] += scale * src[rem_start + i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn average_rows_fast_sse2(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    use std::arch::x86_64::*;

    if rows.is_empty() {
        x.zero();
        return;
    }

    let n = mat.n as usize;

    // First row: copy to x
    let first_row = mat.row(rows[0] as i64);
    x.data_mut().copy_from_slice(first_row);

    // Add remaining rows
    for &row_idx in &rows[1..] {
        let row = mat.row(row_idx as i64);
        add_vector_simd_sse2(x.data_mut(), row, 1.0);
    }

    // Multiply by 1.0 / rows.len()
    let scale = 1.0 / rows.len() as f32;
    unsafe {
        let scale_v = _mm_set1_ps(scale);
        let chunks = n / 4;
        let remainder = n % 4;
        let d_ptr = x.data_mut().as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let v = _mm_loadu_ps(d_ptr.add(offset));
            _mm_storeu_ps(d_ptr.add(offset), _mm_mul_ps(v, scale_v));
        }

        let rem_start = chunks * 4;
        for i in 0..remainder {
            x.data_mut()[rem_start + i] *= scale;
        }
    }
}

// ---------- x86_64 AVX2 ----------

/// AVX2-accelerated dot product of two slices.
///
/// Uses 8-wide f32 lanes (256-bit registers) with 4-way unrolling.
/// # Safety
/// Caller must ensure AVX2 is available (e.g., via `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_simd_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks4 = n / 32; // 4 AVX2 registers × 8 f32 = 32 elements per iteration
    let chunks1 = (n % 32) / 8; // remaining full 8-element chunks
    let remainder = n % 8;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks4 {
        let off = i * 32;
        let a0 = _mm256_loadu_ps(a_ptr.add(off));
        let a1 = _mm256_loadu_ps(a_ptr.add(off + 8));
        let a2 = _mm256_loadu_ps(a_ptr.add(off + 16));
        let a3 = _mm256_loadu_ps(a_ptr.add(off + 24));
        let b0 = _mm256_loadu_ps(b_ptr.add(off));
        let b1 = _mm256_loadu_ps(b_ptr.add(off + 8));
        let b2 = _mm256_loadu_ps(b_ptr.add(off + 16));
        let b3 = _mm256_loadu_ps(b_ptr.add(off + 24));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, b0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1, b1));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(a2, b2));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(a3, b3));
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    let rem_start4 = chunks4 * 32;
    for i in 0..chunks1 {
        let off = rem_start4 + i * 8;
        let a0 = _mm256_loadu_ps(a_ptr.add(off));
        let b0 = _mm256_loadu_ps(b_ptr.add(off));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, b0));
    }

    // Horizontal sum: add the two 128-bit halves, then reduce to scalar
    let low = _mm256_castps256_ps128(acc0);
    let high = _mm256_extractf128_ps(acc0, 1);
    let sum128 = _mm_add_ps(low, high);
    let hi = _mm_movehl_ps(sum128, sum128);
    let s = _mm_add_ps(sum128, hi);
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    let mut result = _mm_cvtss_f32(s);

    let rem_start = rem_start4 + chunks1 * 8;
    for i in 0..remainder {
        result += a[rem_start + i] * b[rem_start + i];
    }

    result
}

/// AVX2-accelerated `dest += scale * src` (8-wide fused multiply-add).
///
/// # Safety
/// Caller must ensure AVX2 is available (e.g., via `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_vector_simd_avx2(dest: &mut [f32], src: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let n = dest.len();
    let chunks4 = n / 32;
    let chunks1 = (n % 32) / 8;
    let remainder = n % 8;

    let scale_v = _mm256_set1_ps(scale);
    let d_ptr = dest.as_mut_ptr();
    let s_ptr = src.as_ptr();

    for i in 0..chunks4 {
        let off = i * 32;
        let d0 = _mm256_loadu_ps(d_ptr.add(off));
        let d1 = _mm256_loadu_ps(d_ptr.add(off + 8));
        let d2 = _mm256_loadu_ps(d_ptr.add(off + 16));
        let d3 = _mm256_loadu_ps(d_ptr.add(off + 24));
        let s0 = _mm256_loadu_ps(s_ptr.add(off));
        let s1 = _mm256_loadu_ps(s_ptr.add(off + 8));
        let s2 = _mm256_loadu_ps(s_ptr.add(off + 16));
        let s3 = _mm256_loadu_ps(s_ptr.add(off + 24));
        _mm256_storeu_ps(
            d_ptr.add(off),
            _mm256_add_ps(d0, _mm256_mul_ps(scale_v, s0)),
        );
        _mm256_storeu_ps(
            d_ptr.add(off + 8),
            _mm256_add_ps(d1, _mm256_mul_ps(scale_v, s1)),
        );
        _mm256_storeu_ps(
            d_ptr.add(off + 16),
            _mm256_add_ps(d2, _mm256_mul_ps(scale_v, s2)),
        );
        _mm256_storeu_ps(
            d_ptr.add(off + 24),
            _mm256_add_ps(d3, _mm256_mul_ps(scale_v, s3)),
        );
    }

    let rem_start4 = chunks4 * 32;
    for i in 0..chunks1 {
        let off = rem_start4 + i * 8;
        let d0 = _mm256_loadu_ps(d_ptr.add(off));
        let s0 = _mm256_loadu_ps(s_ptr.add(off));
        _mm256_storeu_ps(
            d_ptr.add(off),
            _mm256_add_ps(d0, _mm256_mul_ps(scale_v, s0)),
        );
    }

    let rem_start = rem_start4 + chunks1 * 8;
    for i in 0..remainder {
        dest[rem_start + i] += scale * src[rem_start + i];
    }
}

/// AVX2-accelerated row averaging: compute `x = average of selected rows`.
///
/// # Safety
/// Caller must ensure AVX2 is available (e.g., via `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn average_rows_fast_avx2(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    use std::arch::x86_64::*;

    if rows.is_empty() {
        x.zero();
        return;
    }

    let n = mat.n as usize;

    // First row: copy to x
    let first_row = mat.row(rows[0] as i64);
    x.data_mut().copy_from_slice(first_row);

    // Add remaining rows using AVX2
    for &row_idx in &rows[1..] {
        let row = mat.row(row_idx as i64);
        add_vector_simd_avx2(x.data_mut(), row, 1.0);
    }

    // Scale by 1.0 / rows.len() using AVX2
    let scale = 1.0 / rows.len() as f32;
    let scale_v = _mm256_set1_ps(scale);
    let n_chunks = n / 8;
    let remainder = n % 8;
    let d_ptr = x.data_mut().as_mut_ptr();

    for i in 0..n_chunks {
        let off = i * 8;
        let v = _mm256_loadu_ps(d_ptr.add(off));
        _mm256_storeu_ps(d_ptr.add(off), _mm256_mul_ps(v, scale_v));
    }

    let rem_start = n_chunks * 8;
    for i in 0..remainder {
        x.data_mut()[rem_start + i] *= scale;
    }
}

// ---------- dispatch functions ----------

/// Dispatch dot product to best available implementation.
#[inline]
fn dot_impl(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        dot_simd_neon(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability verified by is_x86_feature_detected!
            unsafe { dot_simd_avx2(a, b) }
        } else {
            dot_simd_sse2(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_scalar(a, b)
    }
}

/// Dispatch add_vector to best available implementation.
#[inline]
fn add_vector_impl(dest: &mut [f32], src: &[f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        add_vector_simd_neon(dest, src, scale)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability verified by is_x86_feature_detected!
            unsafe { add_vector_simd_avx2(dest, src, scale) }
        } else {
            add_vector_simd_sse2(dest, src, scale)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        add_vector_scalar(dest, src, scale)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // --- Allocation safety ---

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

    // --- Construction ---

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

    // --- Zero ---

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

    // --- Row-major layout verification ---

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

    // --- dot_row ---

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
        assert!((m.dot_row(&v, 0).unwrap() - 6.0).abs() < 1e-6);
        // dot(row1, v) = 4+5+6 = 15
        assert!((m.dot_row(&v, 1).unwrap() - 15.0).abs() < 1e-6);
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
        assert!((m.dot_row(&v, 0).unwrap() - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_matrix_dot_row_zero_vec() {
        let mut m = DenseMatrix::new(1, 3);
        *m.at_mut(0, 0) = 1.0;
        *m.at_mut(0, 1) = 2.0;
        *m.at_mut(0, 2) = 3.0;

        let v = Vector::new(3); // all zeros
        assert_eq!(m.dot_row(&v, 0).unwrap(), 0.0);
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

        match m.dot_row(&v, 0) {
            Err(FastTextError::EncounteredNaN) => {} // expected
            other => panic!("Expected EncounteredNaN, got {:?}", other),
        }
    }

    // --- add_vector_to_row ---

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

    // --- add_row_to_vector ---

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

    // --- average_rows ---

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

    // --- l2_norm_row ---

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

    // --- multiply_row ---

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
        m.multiply_row(&nums, 0, 3);

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
        m.multiply_row(&nums, 0, 2);

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

        // ie = -1 means use m_ (all rows)
        let nums = [2.0, 3.0];
        m.multiply_row(&nums, 0, -1);

        assert!((m.at(0, 0) - 2.0).abs() < 1e-6);
        assert!((m.at(0, 1) - 4.0).abs() < 1e-6);
        assert!((m.at(1, 0) - 9.0).abs() < 1e-6);
        assert!((m.at(1, 1) - 12.0).abs() < 1e-6);
    }

    // --- divide_row ---

    #[test]
    fn test_dense_matrix_divide_row() {
        let mut m = DenseMatrix::new(2, 2);
        *m.at_mut(0, 0) = 4.0;
        *m.at_mut(0, 1) = 6.0;
        *m.at_mut(1, 0) = 8.0;
        *m.at_mut(1, 1) = 10.0;

        let denoms = [2.0, 5.0];
        m.divide_row(&denoms, 0, 2);

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
        m.divide_row(&denoms, 0, 2);

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
        m.divide_row(&denoms, 0, -1);

        assert!((m.at(0, 0) - 2.0).abs() < 1e-6);
        assert!((m.at(0, 1) - 3.0).abs() < 1e-6);
        assert!((m.at(1, 0) - 2.0).abs() < 1e-6);
        assert!((m.at(1, 1) - 2.5).abs() < 1e-6);
    }

    // --- SIMD consistency ---

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
            let simd_dot = m.dot_row(&v, 1).unwrap();
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

    // --- Alignment ---

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

    // --- Bounds checking ---

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

    // --- Binary save/load round-trip ---

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

    // --- uniform ---

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

    // --- Clone ---

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

    // --- Row access ---

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

    // --- multiply_row with ib offset ---

    #[test]
    fn test_dense_matrix_multiply_row_with_offset() {
        let mut m = DenseMatrix::new(4, 2);
        for i in 0..4 {
            *m.at_mut(i, 0) = (i + 1) as f32;
            *m.at_mut(i, 1) = (i + 1) as f32 * 10.0;
        }

        // Multiply rows 1..3 with nums [2.0, 3.0]
        let nums = [2.0, 3.0];
        m.multiply_row(&nums, 1, 3);

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

    // --- divide_row with ib offset ---

    #[test]
    fn test_dense_matrix_divide_row_with_offset() {
        let mut m = DenseMatrix::new(4, 2);
        for i in 0..4 {
            *m.at_mut(i, 0) = (i + 1) as f32 * 10.0;
            *m.at_mut(i, 1) = (i + 1) as f32 * 20.0;
        }

        // Divide rows 2..4 with denoms [5.0, 10.0]
        let denoms = [5.0, 10.0];
        m.divide_row(&denoms, 2, 4);

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

    // --- NaN detection in dot_row with NaN in vector ---

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

        match m.dot_row(&v, 0) {
            Err(FastTextError::EncounteredNaN) => {} // expected
            other => panic!("Expected EncounteredNaN, got {:?}", other),
        }
    }

    // --- Large matrix operations ---

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
        assert!((m.dot_row(&v, 0).unwrap() - 100.0).abs() < 0.01);
    }

    // --- AVX2 consistency tests (x86_64 only) ---

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
