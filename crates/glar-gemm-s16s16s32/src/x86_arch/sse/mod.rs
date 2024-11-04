#[rustfmt::skip]
pub mod asm_ukernel;
// pub(crate) mod axpy_kernel;

pub(crate) use asm_ukernel::*;
// pub(crate) use axpy_kernel::*;

use paste::paste;
use seq_macro::seq;

use crate::{TA, TB, TC};

const VS: usize = 4;

const fn simd_vector_length() -> usize {
    VS
}

use crate::UnaryFnC;

#[target_feature(enable = "sse,sse2")]
pub unsafe fn axpy<F: UnaryFnC>(
    m: usize,
    n: usize,
    alpha: *const f32,
    a: *const TA,
    a_rs: usize,
    a_cs: usize,
    x: *const TB,
    incx: usize,
    beta: *const f32,
    y: *mut TC,
    incy: usize,
    f: F,
) {
    //    if a_cs == 1 && incx == 1 {
    //        axpy_d(m, n, alpha, a, a_rs, x, beta, y, incy);
    //        for i in 0..m {
    //            f.call(y.add(i*incy), m);
    //        }
    //        return;
    //    }
    // if a_rs == 1 && incy == 1 {
    //     axpy_v(m, n, alpha, a, a_cs, x, incx, beta, y);
    //     // move this inside axpy_v, and benchmark
    //     f.call(y, m);
    //     return;
    // }

    for i in 0..m {
        let y_cur = y.add(i * incy);
        let mut acc = 0i32;
        for j in 0..n {
            let a_cur = a.add(i * a_rs + j * a_cs);
            let x_cur = x.add(j * incx);
            acc += *a_cur as i32 * *x_cur as i32;
        }
        *y_cur = (*beta * *y_cur as f32 + *alpha * acc as f32) as i32;
        f.call(y_cur, 1);
    }
}

use glar_base::def_kernel_bb_v0;
def_kernel_bb_v0!(i16, i16, i32, f32, f32, T, 2, 2);

use super::pack_sse::packa_panel_8;
use glar_base::def_kernel_sb_v0;
def_kernel_sb_v0!(i16, i16, i32, f32, f32, T, packa_panel_8, 2, 2, 2);

pub(crate) unsafe fn kernel_sb<F: UnaryFnC>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    a: *const TB,
    a_rs: usize,
    a_cs: usize,
    b: *const TB,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap_buf: *mut TA,
    f: F,
) {
    if c_rs == 1 {
        kernel_sb_v0::<_, false>(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, f);
    } else {
        kernel_sb_v0::<_, true>(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, f);
    }
}

pub(crate) unsafe fn kernel<F: UnaryFnC>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap: *const TA,
    bp: *const TB,
    f: F,
) {
    let k_eff = (k + 1) / 2 * 2;
    if c_rs == 1 {
        kernel_bb::<_, false>(m, n, k_eff, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    } else {
        kernel_bb::<_, true>(m, n, k_eff, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    }
}
