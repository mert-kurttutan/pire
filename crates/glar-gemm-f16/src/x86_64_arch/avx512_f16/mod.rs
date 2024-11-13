#[rustfmt::skip]
mod asm_ukernel;
// mod axpy_kernel;

use asm_ukernel::*;
// use axpy_kernel::*;

use crate::{UnaryFnC, TA, TB, TC};

const VS: usize = 32;

// const fn simd_vector_length() -> usize {
//     VS
// }

// #[target_feature(enable = "avx")]
// pub unsafe fn axpy<F: UnaryFnC>(
//    m: usize, n: usize,
//    alpha: *const TA,
//    a: *const TA, a_rs: usize, a_cs: usize,
//    x: *const TB, incx: usize,
//    beta: *const TC,
//    y: *mut TC, incy: usize,
//    f: F,
// ) {
//    if a_cs == 1 && incx == 1 {
//     //    axpy_d(m, n, alpha, a, a_rs, x, beta, y, incy);
//     //    for i in 0..m {
//     //        f.call(y.add(i*incy), m);
//     //    }
//     //    return;
//    }
//    if a_rs == 1 && incy == 1 {
//         // axpy_v(m, n, alpha, a, a_cs, x, incx, beta, y);
//         // // move this inside axpy_v, and benchmark
//         // f.call(y, m);
//         // return;
//    }

//    if a_cs == 1 {
//        for i in 0..m {
//            let a_cur = a.add(i*a_rs);
//            let y_cur = y.add(i * incy);
//            let mut acc = f16::from_f32(0.0);
//            for j in 0..n {
//                let x_cur = x.add(j * incx);
//                acc += *a_cur.add(j) * *x_cur;
//            }
//            *y_cur = *beta * *y_cur + *alpha * acc;
//            f.call(y_cur, 1);
//        }
//        return;
//    }
//    if a_rs == 1 || true {
//        for i in 0..m {
//            let y_cur = y.add(i*incy);
//            let mut acc = f16::from_f32(0.0);
//            for j in 0..n {
//                let a_cur = a.add(j*a_cs);
//                let x_cur = x.add(j*incx);
//                acc += *a_cur.add(i) * *x_cur;
//            }
//            *y_cur = *beta * *y_cur + *alpha * acc;
//             f.call(y_cur, 1);
//        }
//        return;
//    }
// }

use glar_base::def_kernel_bb_pf1;
def_kernel_bb_pf1!(TA, TB, TC, TC, F, 1, 2, 15, 128, 4);

use glar_base::def_kernel_bs;
def_kernel_bs!(TA, TB, TC, TC, 2, 15);

use super::pack_avx::packa_panel_64_same;

use glar_base::def_kernel_sb_pf1;
def_kernel_sb_pf1!(TA, TA, TB, TC, TC, packa_panel_64_same, 1, 2, 15, 128, 8);
