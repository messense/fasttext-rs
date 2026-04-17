// AVX SIMD kernels (x86_64, 256-bit, no FMA).

use std::arch::x86_64::*;

use crate::matrix::{DenseMatrix, Matrix};
use crate::vector::Vector;

/// # Safety
/// Caller must ensure AVX is available.
#[target_feature(enable = "avx")]
pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks4 = n / 32;
    let chunks1 = (n % 32) / 8;
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

/// # Safety
/// Caller must ensure AVX is available.
#[target_feature(enable = "avx")]
pub(crate) unsafe fn add_vector(dest: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dest.len(), src.len());
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

/// # Safety
/// Caller must ensure AVX is available.
#[target_feature(enable = "avx")]
pub(crate) unsafe fn average_rows(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
    if rows.is_empty() {
        x.zero();
        return;
    }

    let n = mat.cols() as usize;

    let first_row = mat.row(rows[0] as i64);
    x.data_mut().copy_from_slice(first_row);

    for &row_idx in &rows[1..] {
        let row = mat.row(row_idx as i64);
        add_vector(x.data_mut(), row, 1.0);
    }

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
