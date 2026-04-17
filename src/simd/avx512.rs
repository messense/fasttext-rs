// AVX-512F SIMD kernels (x86_64, 512-bit registers with FMA).
//
// Processes 16 f32 values per instruction. Uses `_mm512_fmadd_ps` for fused
// multiply-add and `_mm512_reduce_add_ps` for horizontal sum.

use std::arch::x86_64::*;

use crate::matrix::{DenseMatrix, Matrix};
use crate::vector::Vector;

/// # Safety
/// Caller must ensure AVX-512F is available.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks4 = n / 64; // 4 × 16 f32 = 64 elements per unrolled iteration
    let chunks1 = (n % 64) / 16;
    let remainder = n % 16;

    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks4 {
        let off = i * 64;
        let a0 = _mm512_loadu_ps(a_ptr.add(off));
        let a1 = _mm512_loadu_ps(a_ptr.add(off + 16));
        let a2 = _mm512_loadu_ps(a_ptr.add(off + 32));
        let a3 = _mm512_loadu_ps(a_ptr.add(off + 48));
        let b0 = _mm512_loadu_ps(b_ptr.add(off));
        let b1 = _mm512_loadu_ps(b_ptr.add(off + 16));
        let b2 = _mm512_loadu_ps(b_ptr.add(off + 32));
        let b3 = _mm512_loadu_ps(b_ptr.add(off + 48));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        acc2 = _mm512_fmadd_ps(a2, b2, acc2);
        acc3 = _mm512_fmadd_ps(a3, b3, acc3);
    }

    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);

    let rem_start4 = chunks4 * 64;
    for i in 0..chunks1 {
        let off = rem_start4 + i * 16;
        let a0 = _mm512_loadu_ps(a_ptr.add(off));
        let b0 = _mm512_loadu_ps(b_ptr.add(off));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
    }

    let mut result = _mm512_reduce_add_ps(acc0);

    let rem_start = rem_start4 + chunks1 * 16;
    for i in 0..remainder {
        result += a[rem_start + i] * b[rem_start + i];
    }

    result
}

/// # Safety
/// Caller must ensure AVX-512F is available.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn add_vector(dest: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dest.len(), src.len());
    let n = dest.len();
    let chunks4 = n / 64;
    let chunks1 = (n % 64) / 16;
    let remainder = n % 16;

    let scale_v = _mm512_set1_ps(scale);
    let d_ptr = dest.as_mut_ptr();
    let s_ptr = src.as_ptr();

    for i in 0..chunks4 {
        let off = i * 64;
        let d0 = _mm512_loadu_ps(d_ptr.add(off));
        let d1 = _mm512_loadu_ps(d_ptr.add(off + 16));
        let d2 = _mm512_loadu_ps(d_ptr.add(off + 32));
        let d3 = _mm512_loadu_ps(d_ptr.add(off + 48));
        let s0 = _mm512_loadu_ps(s_ptr.add(off));
        let s1 = _mm512_loadu_ps(s_ptr.add(off + 16));
        let s2 = _mm512_loadu_ps(s_ptr.add(off + 32));
        let s3 = _mm512_loadu_ps(s_ptr.add(off + 48));
        _mm512_storeu_ps(d_ptr.add(off), _mm512_fmadd_ps(scale_v, s0, d0));
        _mm512_storeu_ps(d_ptr.add(off + 16), _mm512_fmadd_ps(scale_v, s1, d1));
        _mm512_storeu_ps(d_ptr.add(off + 32), _mm512_fmadd_ps(scale_v, s2, d2));
        _mm512_storeu_ps(d_ptr.add(off + 48), _mm512_fmadd_ps(scale_v, s3, d3));
    }

    let rem_start4 = chunks4 * 64;
    for i in 0..chunks1 {
        let off = rem_start4 + i * 16;
        let d0 = _mm512_loadu_ps(d_ptr.add(off));
        let s0 = _mm512_loadu_ps(s_ptr.add(off));
        _mm512_storeu_ps(d_ptr.add(off), _mm512_fmadd_ps(scale_v, s0, d0));
    }

    let rem_start = rem_start4 + chunks1 * 16;
    for i in 0..remainder {
        dest[rem_start + i] += scale * src[rem_start + i];
    }
}

/// # Safety
/// Caller must ensure AVX-512F is available.
#[target_feature(enable = "avx512f")]
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
    let scale_v = _mm512_set1_ps(scale);
    let n_chunks = n / 16;
    let remainder = n % 16;
    let d_ptr = x.data_mut().as_mut_ptr();

    for i in 0..n_chunks {
        let off = i * 16;
        let v = _mm512_loadu_ps(d_ptr.add(off));
        _mm512_storeu_ps(d_ptr.add(off), _mm512_mul_ps(v, scale_v));
    }

    let rem_start = n_chunks * 16;
    for i in 0..remainder {
        x.data_mut()[rem_start + i] *= scale;
    }
}
