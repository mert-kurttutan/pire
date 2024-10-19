#[rustfmt::skip]
pub mod asm_ukernel;
// pub(crate) mod axpy_kernel;

pub(crate) use asm_ukernel::*;
// pub(crate) use axpy_kernel::*;

use paste::paste;

use crate::{TA, TB, TC};

const VS: usize = 1;

use crate::MyFn;

use core::arch::x86::*;

#[target_feature(enable = "sse,sse2,sse3")]
pub(crate) unsafe fn scale_c(m: usize, n: usize, beta: *const TC, c: *mut TC, c_rs: usize, c_cs: usize) {
    if *beta == TC::ZERO {
        if c_rs == 1 {
            for j in 0..n {
                for i in 0..m {
                    *c.add(i + j * c_cs) = TC::ZERO;
                }
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * c_rs + j * c_cs) = TC::ZERO;
                }
            }
        }
    } else if *beta != TC::ONE {
        if c_rs == 1 {
            let beta_f64 = beta as *const f64;
            let beta_vr = _mm_set1_pd(*beta_f64);
            let beta_vi = _mm_set1_pd(*beta_f64.add(1));
            // let c_cs = c_cs * 2;
            let c = c;
            for j in 0..n {
                let mut mi = 0;
                while mi < m / VS * VS {
                    let c_v = _mm_loadu_pd(c.add(mi + j * c_cs) as *const f64);
                    let c_v_1 = _mm_mul_pd(c_v, beta_vr);
                    let c_v_2 = _mm_mul_pd(c_v, beta_vi);

                    let c_v_2 = _mm_shuffle_pd(c_v_2, c_v_2, 0b101);

                    let c_v = _mm_addsub_pd(c_v_1, c_v_2);

                    _mm_storeu_pd(c.add(mi + j * c_cs) as *mut f64, c_v);
                    mi += VS;
                }
                while mi < m {
                    *c.add(mi + j * c_cs) *= *beta;
                    mi += 1;
                }
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * c_rs + j * c_cs) *= *beta;
                }
            }
        }
    }
}

#[target_feature(enable = "sse")]
pub unsafe fn axpy<F: MyFn>(
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

use glar_base::def_kernel_bb_v0_no_beta;
def_kernel_bb_v0_no_beta!(TA, TB, TC, TA, TC, 1, 2, 1);

use glar_base::def_kernel_bs_no_beta;

def_kernel_bs_no_beta!(TA, TB, TC, TA, TC, 1, 2, 1);

use super::pack_sse::packa_panel_1;

use glar_base::def_kernel_sb_v0_no_beta;

def_kernel_sb_v0_no_beta!(TA, TB, TC, TA, TC, packa_panel_1, 1, 2, 1);

// #[target_feature(enable = "sse")]
pub(crate) unsafe fn kernel_sb<F: MyFn>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
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
        kernel_sb_v0::<_, false>(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, f);
    } else {
        kernel_sb_v0::<_, true>(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, f);
    }
}

pub(crate) unsafe fn kernel_bs<F: MyFn>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
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
        kernel_bs_v0::<_, false>(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, f);
    } else {
        kernel_bs_v0::<_, true>(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, f);
    }
}

// #[target_feature(enable = "sse")]
pub(crate) unsafe fn kernel<F: MyFn>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap: *const TA,
    bp: *const TB,
    f: F,
) {
    if c_rs == 1 {
        kernel_bb::<_, false>(m, n, k, alpha, c, c_rs, c_cs, ap, bp, f)
    } else {
        kernel_bb::<_, true>(m, n, k, alpha, c, c_rs, c_cs, ap, bp, f)
    }
}
