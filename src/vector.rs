// Vector: f32 array with SIMD-accelerated operations

use std::fmt;
use std::ops::{Index, IndexMut};

/// A vector of `f32` values used throughout fastText for embeddings,
/// hidden states, gradients, and output scores.
#[derive(Debug, Clone)]
pub struct Vector {
    data: Vec<f32>,
}

impl Vector {
    /// Create a new zero-initialized vector with the given size.
    pub fn new(size: usize) -> Self {
        Vector {
            data: vec![0.0; size],
        }
    }

    /// Set all elements to zero.
    pub fn zero(&mut self) {
        self.data.fill(0.0);
    }

    /// Return the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return true if the vector has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return a slice of the vector data.
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Return a mutable slice of the vector data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Compute the L2 (Euclidean) norm of the vector.
    pub fn norm(&self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Multiply all elements by a scalar.
    pub fn mul(&mut self, a: f32) {
        for v in &mut self.data {
            *v *= a;
        }
    }

    /// Add `scale * source` to this vector element-wise.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != source.len()`.
    pub fn add_vector(&mut self, source: &Vector, scale: f32) {
        assert_eq!(
            self.len(),
            source.len(),
            "Vector size mismatch: {} vs {}",
            self.len(),
            source.len()
        );
        add_vector_impl(self.data_mut(), source.data(), scale);
    }

    /// Compute the dot (inner) product with another vector.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != other.len()`.
    pub fn dot(&self, other: &Vector) -> f32 {
        assert_eq!(
            self.len(),
            other.len(),
            "Vector size mismatch: {} vs {}",
            self.len(),
            other.len()
        );
        dot_impl(self.data(), other.data())
    }

    /// Return the index of the maximum element.
    ///
    /// For an empty vector, returns 0 (matching C++ behavior where argmax
    /// on a zero-size vector is undefined).
    pub fn argmax(&self) -> usize {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in self.data.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        max_idx
    }
}

impl Index<usize> for Vector {
    type Output = f32;

    #[inline]
    fn index(&self, idx: usize) -> &f32 {
        &self.data[idx]
    }
}

impl IndexMut<usize> for Vector {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut f32 {
        &mut self.data[idx]
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, val) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{:.5}", val)?;
        }
        Ok(())
    }
}

// ============================================================================
// SIMD / scalar implementations
// ============================================================================

/// Scalar fallback for dot product.
#[allow(dead_code)]
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar fallback for add_vector (dest += scale * src).
#[allow(dead_code)]
#[inline]
fn add_vector_scalar(dest: &mut [f32], src: &[f32], scale: f32) {
    for i in 0..dest.len() {
        dest[i] += scale * src[i];
    }
}

// ---------- aarch64 NEON ----------

#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_simd_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let chunks = n / 16; // Process 16 floats at a time (4 NEON registers)
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

        // Combine accumulators
        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);

        let mut result = vaddvq_f32(sum0);

        // Handle remainder
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

        // Handle remainder
        let rem_start = chunks * 16;
        for i in 0..remainder {
            dest[rem_start + i] += scale * src[rem_start + i];
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

        // Combine accumulators
        sum0 = _mm_add_ps(sum0, sum1);
        sum2 = _mm_add_ps(sum2, sum3);
        sum0 = _mm_add_ps(sum0, sum2);

        // Horizontal sum of 4-lane SSE register
        let hi = _mm_movehl_ps(sum0, sum0);
        let s = _mm_add_ps(sum0, hi);
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        let mut result = _mm_cvtss_f32(s);

        // Handle remainder
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
            vst(d_ptr.add(offset), _mm_add_ps(d0, _mm_mul_ps(scale_v, s0)));
            vst(
                d_ptr.add(offset + 4),
                _mm_add_ps(d1, _mm_mul_ps(scale_v, s1)),
            );
            vst(
                d_ptr.add(offset + 8),
                _mm_add_ps(d2, _mm_mul_ps(scale_v, s2)),
            );
            vst(
                d_ptr.add(offset + 12),
                _mm_add_ps(d3, _mm_mul_ps(scale_v, s3)),
            );
        }

        // Handle remainder
        let rem_start = chunks * 16;
        for i in 0..remainder {
            dest[rem_start + i] += scale * src[rem_start + i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn vst(ptr: *mut f32, val: std::arch::x86_64::__m128) {
    std::arch::x86_64::_mm_storeu_ps(ptr, val);
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
        dot_simd_sse2(a, b)
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
        add_vector_simd_sse2(dest, src, scale)
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

    // --- Construction and zero ---

    #[test]
    fn test_vector_new() {
        let v = Vector::new(10);
        assert_eq!(v.len(), 10);
        assert!(!v.is_empty());
    }

    #[test]
    fn test_vector_new_zero_size() {
        let v = Vector::new(0);
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
        assert_eq!(v.data().len(), 0);
    }

    #[test]
    fn test_vector_zero() {
        let mut v = Vector::new(5);
        // Set some values
        for i in 0..5 {
            v[i] = (i + 1) as f32;
        }
        v.zero();
        for i in 0..5 {
            assert_eq!(v[i], 0.0, "Element {} should be zero", i);
        }
    }

    // --- Norm ---

    #[test]
    fn test_vector_norm_known() {
        // [3, 4] should have norm 5
        let mut v = Vector::new(2);
        v[0] = 3.0;
        v[1] = 4.0;
        assert!((v.norm() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vector_norm_zero() {
        let v = Vector::new(5);
        assert_eq!(v.norm(), 0.0);
    }

    #[test]
    fn test_vector_norm_single() {
        let mut v = Vector::new(1);
        v[0] = -7.0;
        assert!((v.norm() - 7.0).abs() < f32::EPSILON);
    }

    // --- Mul ---

    #[test]
    fn test_vector_mul_scalar() {
        let mut v = Vector::new(3);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;
        v.mul(2.0);
        assert_eq!(v[0], 2.0);
        assert_eq!(v[1], 4.0);
        assert_eq!(v[2], 6.0);
    }

    #[test]
    fn test_vector_mul_zero() {
        let mut v = Vector::new(3);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;
        v.mul(0.0);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn test_vector_mul_negative() {
        let mut v = Vector::new(3);
        v[0] = 1.0;
        v[1] = -2.0;
        v[2] = 3.0;
        v.mul(-1.0);
        assert_eq!(v[0], -1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], -3.0);
    }

    // --- add_vector ---

    #[test]
    fn test_vector_add_vector_basic() {
        let mut v = Vector::new(3);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;

        let mut other = Vector::new(3);
        other[0] = 10.0;
        other[1] = 20.0;
        other[2] = 30.0;

        v.add_vector(&other, 1.0);
        assert_eq!(v[0], 11.0);
        assert_eq!(v[1], 22.0);
        assert_eq!(v[2], 33.0);
    }

    #[test]
    fn test_vector_add_vector_scaled() {
        let mut v = Vector::new(3);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;

        let mut other = Vector::new(3);
        other[0] = 10.0;
        other[1] = 20.0;
        other[2] = 30.0;

        v.add_vector(&other, 0.5);
        assert_eq!(v[0], 6.0);
        assert_eq!(v[1], 12.0);
        assert_eq!(v[2], 18.0);
    }

    #[test]
    #[should_panic(expected = "Vector size mismatch")]
    fn test_vector_add_vector_size_mismatch() {
        let mut v = Vector::new(3);
        let other = Vector::new(4);
        v.add_vector(&other, 1.0);
    }

    // --- dot product ---

    #[test]
    fn test_vector_dot_basic() {
        let mut a = Vector::new(3);
        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;

        let mut b = Vector::new(3);
        b[0] = 4.0;
        b[1] = 5.0;
        b[2] = 6.0;

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((a.dot(&b) - 32.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vector_dot_orthogonal() {
        let mut a = Vector::new(2);
        a[0] = 1.0;
        a[1] = 0.0;

        let mut b = Vector::new(2);
        b[0] = 0.0;
        b[1] = 1.0;

        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    #[should_panic(expected = "Vector size mismatch")]
    fn test_vector_dot_size_mismatch() {
        let a = Vector::new(3);
        let b = Vector::new(4);
        a.dot(&b);
    }

    // --- argmax ---

    #[test]
    fn test_vector_argmax_first() {
        let mut v = Vector::new(5);
        v[0] = 10.0;
        v[1] = 1.0;
        v[2] = 2.0;
        v[3] = 3.0;
        v[4] = 4.0;
        assert_eq!(v.argmax(), 0);
    }

    #[test]
    fn test_vector_argmax_mid() {
        let mut v = Vector::new(5);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 10.0;
        v[3] = 3.0;
        v[4] = 4.0;
        assert_eq!(v.argmax(), 2);
    }

    #[test]
    fn test_vector_argmax_last() {
        let mut v = Vector::new(5);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;
        v[3] = 4.0;
        v[4] = 10.0;
        assert_eq!(v.argmax(), 4);
    }

    #[test]
    fn test_vector_argmax_single() {
        let mut v = Vector::new(1);
        v[0] = 42.0;
        assert_eq!(v.argmax(), 0);
    }

    // --- Size-1 vectors ---

    #[test]
    fn test_vector_size_one() {
        let mut v = Vector::new(1);
        v[0] = 5.0;
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], 5.0);
        assert!((v.norm() - 5.0).abs() < f32::EPSILON);
        v.mul(3.0);
        assert_eq!(v[0], 15.0);
        assert_eq!(v.argmax(), 0);
    }

    #[test]
    fn test_vector_size_one_dot() {
        let mut a = Vector::new(1);
        a[0] = 3.0;
        let mut b = Vector::new(1);
        b[0] = 4.0;
        assert!((a.dot(&b) - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vector_size_one_add() {
        let mut a = Vector::new(1);
        a[0] = 3.0;
        let mut b = Vector::new(1);
        b[0] = 4.0;
        a.add_vector(&b, 2.0);
        assert!((a[0] - 11.0).abs() < f32::EPSILON);
    }

    // --- Large vectors ---

    #[test]
    fn test_vector_large() {
        let size = 10000;
        let mut v = Vector::new(size);
        for i in 0..size {
            v[i] = i as f32;
        }
        assert_eq!(v.len(), size);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[9999], 9999.0);
        assert_eq!(v.argmax(), 9999);
    }

    #[test]
    fn test_vector_large_norm() {
        let size = 10000;
        let mut v = Vector::new(size);
        // All ones: norm = sqrt(10000) = 100
        for i in 0..size {
            v[i] = 1.0;
        }
        let n = v.norm();
        assert!(
            (n - 100.0).abs() < 0.01,
            "norm of 10000 ones: expected 100.0, got {}",
            n
        );
    }

    // --- SIMD vs scalar consistency ---

    #[test]
    fn test_simd_vs_scalar_dot_512() {
        let size = 512;
        let mut a = Vector::new(size);
        let mut b = Vector::new(size);
        for i in 0..size {
            a[i] = (i as f32) * 0.01;
            b[i] = ((size - i) as f32) * 0.01;
        }

        let simd_result = a.dot(&b);
        let scalar_result = dot_scalar(a.data(), b.data());

        // SIMD and scalar accumulate in different order, so allow relative tolerance
        let magnitude = simd_result.abs().max(scalar_result.abs()).max(1.0);
        let tolerance = magnitude * f32::EPSILON * size as f32;
        assert!(
            (simd_result - scalar_result).abs() < tolerance,
            "SIMD dot vs scalar mismatch: SIMD={}, scalar={}, diff={}, tol={}",
            simd_result,
            scalar_result,
            (simd_result - scalar_result).abs(),
            tolerance,
        );
    }

    #[test]
    fn test_simd_vs_scalar_norm_512() {
        let size = 512;
        let mut v = Vector::new(size);
        for i in 0..size {
            v[i] = (i as f32) * 0.01;
        }

        let simd_norm = v.norm();
        let scalar_dot = dot_scalar(v.data(), v.data());
        let scalar_norm = scalar_dot.sqrt();

        // Use relative tolerance proportional to magnitude
        let magnitude = simd_norm.abs().max(scalar_norm.abs()).max(1.0);
        let tolerance = magnitude * f32::EPSILON * size as f32;
        assert!(
            (simd_norm - scalar_norm).abs() < tolerance,
            "SIMD norm vs scalar mismatch: SIMD={}, scalar={}, diff={}, tol={}",
            simd_norm,
            scalar_norm,
            (simd_norm - scalar_norm).abs(),
            tolerance,
        );
    }

    #[test]
    fn test_simd_vs_scalar_add_vector_512() {
        let size = 512;

        // SIMD path
        let mut dest_simd = Vector::new(size);
        let mut src = Vector::new(size);
        for i in 0..size {
            dest_simd[i] = (i as f32) * 0.01;
            src[i] = ((size - i) as f32) * 0.01;
        }
        let mut dest_scalar = dest_simd.clone();

        dest_simd.add_vector(&src, 0.5);
        add_vector_scalar(dest_scalar.data_mut(), src.data(), 0.5);

        // add_vector operations are per-element (multiply+add), tolerance for FMA vs mul+add
        for i in 0..size {
            let magnitude = dest_simd[i].abs().max(dest_scalar.data()[i].abs()).max(1.0);
            let tolerance = magnitude * f32::EPSILON * 4.0;
            assert!(
                (dest_simd[i] - dest_scalar.data()[i]).abs() < tolerance,
                "SIMD add_vector vs scalar mismatch at index {}: SIMD={}, scalar={}",
                i,
                dest_simd[i],
                dest_scalar.data()[i],
            );
        }
    }

    // --- Clone ---

    #[test]
    fn test_vector_clone() {
        let mut v = Vector::new(4);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;
        v[3] = 4.0;

        let v2 = v.clone();
        assert_eq!(v2.len(), 4);
        assert_eq!(v2[0], 1.0);
        assert_eq!(v2[1], 2.0);
        assert_eq!(v2[2], 3.0);
        assert_eq!(v2[3], 4.0);

        // Ensure they are independent
        v[0] = 100.0;
        assert_eq!(v2[0], 1.0);
    }

    // --- Display ---

    #[test]
    fn test_vector_display() {
        let mut v = Vector::new(3);
        v[0] = 1.0;
        v[1] = 2.5;
        v[2] = -3.14;
        let s = format!("{}", v);
        assert!(s.contains("1.00000"));
        assert!(s.contains("2.50000"));
        assert!(s.contains("-3.14000"));
    }

    // --- Index ---

    #[test]
    fn test_vector_index() {
        let mut v = Vector::new(3);
        v[0] = 10.0;
        v[1] = 20.0;
        v[2] = 30.0;
        assert_eq!(v[0], 10.0);
        assert_eq!(v[1], 20.0);
        assert_eq!(v[2], 30.0);
    }

    #[test]
    #[should_panic]
    fn test_vector_index_out_of_bounds() {
        let v = Vector::new(3);
        let _ = v[3];
    }

    // --- Allocation safety ---

    #[test]
    fn test_vector_alloc_safety_zero_size() {
        // Allocation with size 0 must not panic and produce an empty vector.
        let v = Vector::new(0);
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
        assert_eq!(v.data().len(), 0);
        // Clone of zero-size vector must also work without panic.
        let v2 = v.clone();
        assert_eq!(v2.len(), 0);
    }

    // --- Edge cases ---

    #[test]
    fn test_vector_dot_zero_vectors() {
        let a = Vector::new(5);
        let b = Vector::new(5);
        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    fn test_vector_add_zero_scale() {
        let mut v = Vector::new(3);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;

        let mut other = Vector::new(3);
        other[0] = 10.0;
        other[1] = 20.0;
        other[2] = 30.0;

        v.add_vector(&other, 0.0);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_vector_argmax_all_negative() {
        let mut v = Vector::new(4);
        v[0] = -5.0;
        v[1] = -1.0;
        v[2] = -10.0;
        v[3] = -3.0;
        assert_eq!(v.argmax(), 1);
    }

    #[test]
    fn test_vector_argmax_ties_returns_first() {
        let mut v = Vector::new(4);
        v[0] = 1.0;
        v[1] = 5.0;
        v[2] = 5.0;
        v[3] = 5.0;
        // Should return the first occurrence of the maximum
        assert_eq!(v.argmax(), 1);
    }

    // --- SIMD consistency on non-aligned-to-16 sizes ---

    #[test]
    fn test_simd_vs_scalar_dot_non_aligned_sizes() {
        // Test sizes that are NOT multiples of 16 to exercise remainder handling
        for &size in &[1, 3, 7, 15, 17, 31, 33, 63, 65, 100, 127, 255, 511] {
            let mut a = Vector::new(size);
            let mut b = Vector::new(size);
            for i in 0..size {
                a[i] = (i as f32 + 1.0) * 0.1;
                b[i] = (size as f32 - i as f32) * 0.1;
            }

            let simd_result = a.dot(&b);
            let scalar_result = dot_scalar(a.data(), b.data());

            // Use relative tolerance proportional to result magnitude
            let magnitude = simd_result.abs().max(scalar_result.abs()).max(1.0);
            let tolerance = magnitude * f32::EPSILON * size as f32;
            assert!(
                (simd_result - scalar_result).abs() < tolerance,
                "SIMD vs scalar dot mismatch for size {}: SIMD={}, scalar={}, diff={}",
                size,
                simd_result,
                scalar_result,
                (simd_result - scalar_result).abs(),
            );
        }
    }

    #[test]
    fn test_simd_vs_scalar_add_non_aligned_sizes() {
        for &size in &[1, 3, 7, 15, 17, 31, 33, 63, 65, 100, 127, 255, 511] {
            let mut dest_simd = Vector::new(size);
            let mut src = Vector::new(size);
            for i in 0..size {
                dest_simd[i] = (i as f32 + 1.0) * 0.1;
                src[i] = (size as f32 - i as f32) * 0.1;
            }
            let mut dest_scalar = dest_simd.clone();

            dest_simd.add_vector(&src, 0.7);
            add_vector_scalar(dest_scalar.data_mut(), src.data(), 0.7);

            // Per-element tolerance: FMA vs mul+add can differ by 1 ULP at the value magnitude
            for i in 0..size {
                let magnitude = dest_simd[i].abs().max(dest_scalar.data()[i].abs()).max(1.0);
                let tolerance = magnitude * f32::EPSILON * 4.0;
                assert!(
                    (dest_simd[i] - dest_scalar.data()[i]).abs() < tolerance,
                    "SIMD vs scalar add_vector mismatch at index {} for size {}: SIMD={}, scalar={}",
                    i,
                    size,
                    dest_simd[i],
                    dest_scalar.data()[i],
                );
            }
        }
    }
}
