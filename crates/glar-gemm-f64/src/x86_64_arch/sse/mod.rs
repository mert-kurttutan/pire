#[rustfmt::skip]
pub mod asm_ukernel;
pub(crate) mod axpy_kernel;

pub(crate) use asm_ukernel::*;
pub(crate) use axpy_kernel::*;

use paste::paste;
use seq_macro::seq;

use crate::{TA, TB, TC};

const VS: usize = 2;

const fn simd_vector_length() -> usize {
    VS
}

use crate::UnaryFnC;

#[target_feature(enable = "sse")]
pub unsafe fn axpy<F: UnaryFnC>(
    m: usize,
    n: usize,
    alpha: *const TA,
    a: *const TA,
    a_rs: usize,
    a_cs: usize,
    x: *const TB,
    incx: usize,
    beta: *const TC,
    y: *mut TC,
    incy: usize,
    f: F,
) {
    if a_cs == 1 && incx == 1 {
        axpy_d(m, n, alpha, a, a_rs, x, beta, y, incy);
        for i in 0..m {
            f.call(y.add(i * incy), 1);
        }
        return;
    }
    if a_rs == 1 && incy == 1 {
        axpy_v(m, n, alpha, a, a_cs, x, incx, beta, y);
        f.call(y, m);
        return;
    }

    for i in 0..m {
        let y_cur = y.add(i * incy);
        let mut acc = 0.0;
        for j in 0..n {
            let a_cur = a.add(i * a_rs + j * a_cs);
            let x_cur = x.add(j * incx);
            acc += *a_cur * *x_cur;
        }
        *y_cur = *beta * *y_cur + *alpha * acc;
        f.call(y_cur, 1);
    }
}

use glar_base::def_kernel_bb_v0;
def_kernel_bb_v0!(f64, f64, f64, f64, f64, T, 2, 4);

use glar_base::def_kernel_bs;

def_kernel_bs!(f64, f64, f64, f64, f64, 2, 4, 2, 1);

use super::pack_sse::packa_panel_4;

use glar_base::def_kernel_sb_v0;

def_kernel_sb_v0!(TA, TB, TC, TA, TC, T, packa_panel_4, 1, 2, 4);

// #[target_feature(enable = "sse")]
pub(crate) unsafe fn kernel_sb<F: UnaryFnC>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    beta: *const TC,
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

pub(crate) unsafe fn kernel_bs<F: UnaryFnC>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    beta: *const TC,
    b: *const TB,
    b_rs: usize,
    b_cs: usize,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap: *const TA,
    f: F,
) {
    if c_rs == 1 {
        kernel_bs_v0::<_, false>(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, f);
    } else {
        kernel_bs_v0::<_, true>(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, f);
    }
}

// #[target_feature(enable = "sse")]
pub(crate) unsafe fn kernel<F: UnaryFnC>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    beta: *const TC,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap: *const TA,
    bp: *const TB,
    f: F,
) {
    if c_rs == 1 {
        kernel_bb::<_, false>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    } else {
        kernel_bb::<_, true>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    }
}
