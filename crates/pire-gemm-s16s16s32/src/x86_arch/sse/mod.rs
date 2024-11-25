#[rustfmt::skip]
mod asm_ukernel;
// mod axpy_kernel;

use asm_ukernel::*;
// use axpy_kernel::*;

use crate::{UnaryFnC, TA, TB, TC};

const VS: usize = 4;
const VS_MAX: usize = VS;

const ZERO: i32 = 0;

const fn simd_vector_length() -> usize {
    VS
}

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

use pire_base::def_kernel_bb_v0;
def_kernel_bb_v0!(i16, i16, i32, f32, true, F, 2, 2, 2, 0, 0);

use super::pack_sse::packa_panel_8;
use pire_base::def_kernel_sb_v0;
def_kernel_sb_v0!(i16, i16, i16, i32, f32, true, F, packa_panel_8, 2, 2, 2, 0, 0);
