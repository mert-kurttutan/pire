#[rustfmt::skip]
pub mod asm_ukernel;
pub(crate) mod axpy_kernel;

pub(crate) use asm_ukernel::*;

pub(crate) use axpy_kernel::*;

use paste::paste;
use std::arch::asm;

const VS: usize = 8;

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

macro_rules! def_kernel_bb {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        paste! {
            #[target_feature(enable = "avx")]
            pub unsafe fn [<kernel_$MR x $NR _bb>]<F: UnaryFnC, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const f32,
                beta: *const f32,
                c: *mut f16, c_rs: usize, c_cs: usize,
                ap: *const f32, bp: *const f32,
                f: F,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let m_rounded = m / MR * MR;
                let n_rounded = n / NR * NR;
                let m_left = m % MR;
                let n_left = n % NR;

                let d_arr = [0, 0, c_rs];

                let mut m_i = 0;
                while m_i < m_rounded {
                    let c_cur0 = c.add(m_i * c_rs);
                    let ap_cur = ap.add(m_i * k);
                    let mut n_i = 0;
                    while n_i < n_rounded {
                        let bp_cur = bp.add(n_i * k);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        [<ukernel_$MR x $NR _bb>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, MR, f);
                        n_i += NR;
                    }
                    if n_left != 0 {
                        let bp_cur = bp.add(n_i * k);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        [<ukernel_$MR x n _bb>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, MR, n_left, f);
                    }
                    m_i += MR;
                }


                $(
                    if (m_left+VS-1) / VS * VS == $mr_left {
                        let c_cur0 = c.add(m_i * c_rs);
                        let ap_cur = ap.add(m_i * k);
                        let mut n_i = 0;
                        while n_i < n_rounded {
                            let bp_cur = bp.add(n_i * k);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left x $NR _bb>]::<_, true>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, m_left, f);
                            n_i += NR;
                        }
                        if n_left !=0 {
                            let bp_cur = bp.add(n_i * k);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left x n_bb>]::<_, true>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, m_left, n_left, f);
                        }
                    }
                )*

                asm!("vzeroupper");
            }
        }
    };
}

def_kernel_bb!(16, 4, 16, 8);

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
        kernel_16x4_bb::<_, false>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    } else {
        kernel_16x4_bb::<_, true>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    }
}
