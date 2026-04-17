// SSE2 SIMD kernels (x86_64 baseline).

use std::arch::x86_64::*;

use crate::matrix::{DenseMatrix, Matrix};
use crate::vector::Vector;

#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
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

#[inline]
pub(crate) fn add_vector(dest: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dest.len(), src.len());
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

pub(crate) fn average_rows(x: &mut Vector, rows: &[i32], mat: &DenseMatrix) {
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
