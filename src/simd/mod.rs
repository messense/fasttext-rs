// Shared SIMD kernels for dot product, vector addition, and row averaging.
//
// Organized by target feature level:
//   - scalar   — portable fallback
//   - neon     — aarch64 NEON (128-bit with FMA)
//   - sse2     — x86_64 baseline (128-bit)
//   - avx      — x86_64 AVX (256-bit, no FMA)
//   - avx2     — x86_64 AVX2 + FMA (256-bit with fused multiply-add)
//   - avx512   — x86_64 AVX-512F (512-bit with FMA)
//
// The `dot_impl` and `add_vector_impl` dispatch functions select the best
// available implementation at runtime on x86_64 via `is_x86_feature_detected!`.

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "x86_64")]
mod sse2;

use crate::matrix::DenseMatrix;
use crate::vector::Vector;

// Scalar fallback implementations

/// Scalar fallback for dot product of two slices.
#[allow(dead_code)]
#[inline]
pub(crate) fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Scalar fallback for dest += scale * src.
#[allow(dead_code)]
#[inline]
pub(crate) fn add_vector_scalar(dest: &mut [f32], src: &[f32], scale: f32) {
    dest.iter_mut()
        .zip(src.iter())
        .for_each(|(d, &s)| *d += scale * s);
}

/// Scalar average_rows implementation.
#[allow(dead_code)]
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

// Dispatch functions

/// Dispatch dot product to best available implementation.
#[inline]
pub(crate) fn dot_impl(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx512::dot(a, b) }
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::dot(a, b) }
        } else if is_x86_feature_detected!("avx") {
            unsafe { avx::dot(a, b) }
        } else {
            sse2::dot(a, b)
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
        neon::add_vector(dest, src, scale)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx512::add_vector(dest, src, scale) }
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::add_vector(dest, src, scale) }
        } else if is_x86_feature_detected!("avx") {
            unsafe { avx::add_vector(dest, src, scale) }
        } else {
            sse2::add_vector(dest, src, scale)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        add_vector_scalar(dest, src, scale)
    }
}

/// Dispatch average_rows to best available implementation.
pub(crate) fn average_rows_impl(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::average_rows(x, rows, mat);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx512::average_rows(x, rows, mat) }
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::average_rows(x, rows, mat) }
        } else if is_x86_feature_detected!("avx") {
            unsafe { avx::average_rows(x, rows, mat) }
        } else {
            sse2::average_rows(x, rows, mat)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        average_rows_scalar(x, rows, mat);
    }
}
