#[rustfmt::skip]
mod asm_ukernel;
// mod axpy_kernel;

use asm_ukernel::*;
// use axpy_kernel::*;

use crate::{UnaryFnC, TA, TB, TC};

const VS: usize = 8;

const fn simd_vector_length() -> usize {
    VS
}

#[target_feature(enable = "avx,avx2")]
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
    // if a_cs == 1 && incx == 1 {
    //     axpy_d(m, n, alpha, a, a_rs, x, beta, y, incy);
    //     for i in 0..m {
    //         f.call(y.add(i * incy), 1);
    //     }
    //     return;
    // }
    // if a_rs == 1 && incy == 1 {
    //     axpy_v(m, n, alpha, a, a_cs, x, incx, beta, y);
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

#[target_feature(enable = "avx,avx2")]
pub unsafe fn axpy2<F: UnaryFnC>(
    m: usize,
    n: usize,
    alpha: *const f32,
    a: *const TB,
    a_rs: usize,
    a_cs: usize,
    x: *const TA,
    incx: usize,
    beta: *const f32,
    y: *mut TC,
    incy: usize,
    f: F,
) {
    // if a_cs == 1 && incx == 1 {
    //     axpy_d(m, n, alpha, a, a_rs, x, beta, y, incy);
    //     for i in 0..m {
    //         f.call(y.add(i * incy), 1);
    //     }
    //     return;
    // }
    // if a_rs == 1 && incy == 1 {
    //     axpy_v(m, n, alpha, a, a_cs, x, incx, beta, y);
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
def_kernel_bb_v0!(i8, u8, i32, f32, F, 4, 2, 4);
// use pire_base::def_kernel_bb_pf1;
// def_kernel_bb_pf1!(i8, u8, i32, f32, F, 4, 2, 4, 96, 4);

use super::pack_avx::packa_panel_16;

use pire_base::def_kernel_sb_v0;
def_kernel_sb_v0!(i8, i8, u8, i32, f32, F, packa_panel_16, 4, 2, 4);
