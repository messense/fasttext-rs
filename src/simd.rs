// Shared SIMD kernels for dot product, vector addition, and row averaging.
//
// These functions were previously duplicated between vector.rs and matrix.rs.
// All functions are `pub(crate)` — they are internal implementation details
// used by `Vector` and `DenseMatrix`.

use crate::matrix::{DenseMatrix, Matrix};
use crate::vector::Vector;

// Scalar fallback implementations

/// Scalar fallback for dot product of two slices.
#[allow(dead_code)]
#[inline]
pub(crate) fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar fallback for dest += scale * src.
#[allow(dead_code)]
#[inline]
pub(crate) fn add_vector_scalar(dest: &mut [f32], src: &[f32], scale: f32) {
    for i in 0..dest.len() {
        dest[i] += scale * src[i];
    }
}

/// Scalar average_rows implementation.
pub(crate) fn average_rows_scalar(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    x.zero();
    for &row_idx in rows {
        let row = mat.row(row_idx as i64);
        add_vector_impl(x.data_mut(), row, 1.0);
    }
    if !rows.is_empty() {
        x.mul(1.0 / rows.len() as f32);
    }
}

// NEON implementations (aarch64)

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn dot_simd_neon(a: &[f32], b: &[f32]) -> f32 {
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
pub(crate) fn add_vector_simd_neon(dest: &mut [f32], src: &[f32], scale: f32) {
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

#[cfg(target_arch = "aarch64")]
pub(crate) fn average_rows_fast_neon(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    use std::arch::aarch64::*;

    if rows.is_empty() {
        x.zero();
        return;
    }

    let n = mat.cols() as usize;

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

// SSE2 implementations (x86_64)

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn dot_simd_sse2(a: &[f32], b: &[f32]) -> f32 {
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
pub(crate) fn add_vector_simd_sse2(dest: &mut [f32], src: &[f32], scale: f32) {
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

        // Handle remainder
        let rem_start = chunks * 16;
        for i in 0..remainder {
            dest[rem_start + i] += scale * src[rem_start + i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn average_rows_fast_sse2(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    use std::arch::x86_64::*;

    if rows.is_empty() {
        x.zero();
        return;
    }

    let n = mat.cols() as usize;

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

// AVX2 implementations (x86_64)

/// AVX2-accelerated dot product of two slices.
///
/// Uses 8-wide f32 lanes (256-bit registers) with 4-way unrolling.
/// # Safety
/// Caller must ensure AVX2 is available (e.g., via `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot_simd_avx2(a: &[f32], b: &[f32]) -> f32 {
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
pub(crate) unsafe fn add_vector_simd_avx2(dest: &mut [f32], src: &[f32], scale: f32) {
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
pub(crate) unsafe fn average_rows_fast_avx2(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    use std::arch::x86_64::*;

    if rows.is_empty() {
        x.zero();
        return;
    }

    let n = mat.cols() as usize;

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

// Dispatch functions

/// Dispatch dot product to best available implementation.
#[inline]
pub(crate) fn dot_impl(a: &[f32], b: &[f32]) -> f32 {
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
pub(crate) fn add_vector_impl(dest: &mut [f32], src: &[f32], scale: f32) {
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
