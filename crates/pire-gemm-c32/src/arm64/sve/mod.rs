#[rustfmt::skip]
mod asm_ukernel;
mod axpy_kernel;

use asm_ukernel::*;
use axpy_kernel::*;

use crate::{UnaryFnC, TA, TB, TC};

unsafe fn simd_vector_length() -> usize {
    super::sve_vs()
}

const VS_MAX: usize = 2048 / (8 * core::mem::size_of::<TC>());
const ZERO: TC = TC::ZERO;

#[target_feature(enable = "neon")]
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
        let mut acc = TC::ZERO;
        for j in 0..n {
            let a_cur = a.add(i * a_rs + j * a_cs);
            let x_cur = x.add(j * incx);
            acc += *a_cur * *x_cur;
        }
        *y_cur = *beta * *y_cur + *alpha * acc;
        f.call(y_cur, 1);
    }
}

use pire_base::def_kernel_bb_v0;

def_kernel_bb_v0!(TC, TC, TC, TC, false, 1, 3, 8);

use super::pack_sve::packa_panel;

use pire_base::def_kernel_sb_v0;
def_kernel_sb_v0!(TC, TC, TC, TC, TC, false, packa_panel, 1, 3, 8);
