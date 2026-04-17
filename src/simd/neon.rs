// NEON SIMD kernels (aarch64).

use std::arch::aarch64::*;

use crate::matrix::{DenseMatrix, Matrix};
use crate::vector::Vector;

#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
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

#[inline]
pub(crate) fn add_vector(dest: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dest.len(), src.len());
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
