#[rustfmt::skip]
pub mod asm_ukernel;
pub(crate) mod axpy_kernel;

pub(crate) use asm_ukernel::*;

pub(crate) use axpy_kernel::*;

use paste::paste;
use seq_macro::seq;
use std::arch::asm;

const VS: usize = 8;

const fn simd_vector_length() -> usize {
    VS
}

use crate::UnaryFnC;

use half::f16;

#[target_feature(enable = "avx,fma")]
pub unsafe fn axpy<F: UnaryFnC>(
    m: usize,
    n: usize,
    alpha: *const f32,
    a: *const f16,
    a_rs: usize,
    a_cs: usize,
    x: *const f16,
    incx: usize,
    beta: *const f32,
    y: *mut f16,
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
        // move this inside axpy_v, and benchmark
        f.call(y, m);
        return;
    }
    let beta_f16 = f16::from_f32(*beta);
    let alpha_f16 = f16::from_f32(*alpha);
    for i in 0..m {
        let y_cur = y.add(i * incy);
        let mut acc = f16::ZERO;
        for j in 0..n {
            let a_cur = a.add(i * a_rs + j * a_cs);
            let x_cur = x.add(j * incx);
            acc += *a_cur * *x_cur;
        }
        *y_cur = beta_f16 * *y_cur + alpha_f16 * acc;
        f.call(y_cur, 1);
    }
}
type TA = f32;
type TB = f32;
type TC = f16;

use glar_base::def_kernel_bb_v0;
def_kernel_bb_v0!(TA, TB, TC, TA, TA, T, 2, 4);

// #[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn kernel<F: UnaryFnC>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
    ap: *const f32,
    bp: *const f32,
    f: F,
) {
    if c_rs == 1 {
        kernel_bb::<_, false>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    } else {
        kernel_bb::<_, true>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    }
    asm!("vzeroupper");
}
