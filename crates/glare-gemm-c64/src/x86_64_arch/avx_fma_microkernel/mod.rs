pub mod asm_ukernel;
// pub(crate) mod axpy_kernel;

pub(crate) use asm_ukernel::*;
// pub(crate) use axpy_kernel::*;

use paste::paste;
use std::arch::asm;

use crate::{TA,TB,TC};

const VS: usize = 2;

use crate::MyFn;

#[target_feature(enable = "avx,fma")]
pub unsafe fn axpy<F: MyFn>(
   m: usize, n: usize,
   alpha: *const TA,
   a: *const TA, a_rs: usize, a_cs: usize,
   x: *const TB, incx: usize,
   beta: *const TC,
   y: *mut TC, incy: usize,
   f: F,
) {
//    if a_cs == 1 && incx == 1 {
//        axpy_d(m, n, alpha, a, a_rs, x, beta, y, incy);
//        for i in 0..m {
//            f.call(y.add(i*incy), m);
//        }
//        return;
//    }
//    if a_rs == 1 && incy == 1 {
//         axpy_v(m, n, alpha, a, a_cs, x, incx, beta, y);
//         // move this inside axpy_v, and benchmark
//         f.call(y, m);
//         return;
//    }

   if a_cs == 1 {
       for i in 0..m {
           let a_cur = a.add(i*a_rs);
           let y_cur = y.add(i * incy);
           let mut acc = TC::ZERO;
           for j in 0..n {
               let x_cur = x.add(j * incx);
               acc += *a_cur.add(j) * *x_cur;
           }
           *y_cur = *beta * *y_cur + *alpha * acc;
           f.call(y_cur, 1);
       }
       return;
   }
   if a_rs == 1 || true {
       for i in 0..m {
           let y_cur = y.add(i*incy);
           let mut acc = TC::ZERO;
           for j in 0..n {
               let a_cur = a.add(j*a_cs);
               let x_cur = x.add(j*incx);
               acc += *a_cur.add(i) * *x_cur;
           }
           *y_cur = *beta * *y_cur + *alpha * acc;
            f.call(y_cur, 1);
       }
       return;
   }
}

macro_rules! def_kernel_bb {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        paste! {
            #[target_feature(enable = "avx")]
            pub unsafe fn [<kernel_$MR x $NR _bb>]<F: MyFn, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                c: *mut TC, c_rs: usize, c_cs: usize,
                ap: *const TA, bp: *const TB,
                f: F,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let mut m_iter = (m / MR);
                let m_left = m % MR;
                let mut ap_cur = ap;
                let mut c_cur0 = c;
                
                
                let n_iter0 = (n / NR);
                let n_left = (n % NR);
                let d_arr = [0, 0, c_rs, c_cs];
                
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut bp_cur = bp;
                    let mut c_cur1 = c_cur0;
                    while n_iter > 0 {
                        [<ukernel_$MR x $NR _bb>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, k, d_arr, MR, NR, f);
                        n_iter -= 1;
                        bp_cur = bp_cur.add(NR*k);
                        c_cur1 = c_cur1.add(NR*c_cs);
                    }
                    // let a_pft1_offset = ($MR+(n_iter0-n_iter)*2)*4*k;
                    if n_left != 0 {
                        [<ukernel_$MR x n _bb>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, k, d_arr, MR, n_left, f);
                    }
                    m_iter -= 1;
                    ap_cur = ap_cur.add(MR*k);
                    c_cur0 = c_cur0.add(MR*c_rs);
                }


                $(
                    if (m_left+VS-1) / VS *VS == $mr_left {
                        let mut n_iter = n_iter0;
                        let mut bp_cur = bp;
                        let mut c_cur1 = c_cur0;
                        while n_iter > 0 {
                            [<ukernel_$mr_left x $NR _bb_partial>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, k, d_arr, m_left, NR, f);
                            n_iter -= 1;
                            bp_cur = bp_cur.add(NR*k);
                            c_cur1 = c_cur1.add(NR*c_cs);
                        }
                        if n_left !=0 {
                            [<ukernel_$mr_left x n_bb_partial>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, k, d_arr, m_left, n_left, f);
                        }
                        return;
                    }
                )*

                asm!("vzeroupper");
            }
        }   
    };
}
use core::arch::x86_64::*;

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn scale_c(m: usize, n: usize, beta: *const TC, c: *mut TC, c_rs: usize, c_cs: usize) {
    if *beta == TC::ZERO {
        if c_rs == 1 {
            for j in 0..n {
                for i in 0..m {
                    *c.add(i + j*c_cs) = TC::ZERO;
                }
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i*c_rs + j*c_cs) = TC::ZERO;
                }
            }
        }
    } else if *beta != TC::ONE {
        if c_rs == 1 {
            let beta = beta as *const f64;
            let beta_vr = _mm256_set1_pd(*beta);
            let beta_vi = _mm256_set1_pd(*beta.add(1));
            let c_cs = c_cs * 2;
            let c = c as *mut f64;
            for j in 0..n {
                let mut mi = 0;
                while mi < m / 2 {
                    let c_v = _mm256_loadu_pd(c.add(mi*4 + j*c_cs));
                    let c_v_1 = _mm256_mul_pd(c_v, beta_vr);
                    let c_v_2 = _mm256_mul_pd(c_v, beta_vi);

                    let c_v_2 = _mm256_permute_pd(c_v_2, 0x5);

                    let c_v = _mm256_addsub_pd(c_v_1, c_v_2);

                    _mm256_storeu_pd(c.add(mi*8 + j*c_cs), c_v);
                    mi += 1;
                }
                // for i in 0..m {
                //     *c.add(i + j*c_cs) *= beta;
                // }
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i*c_rs + j*c_cs) *= *beta;
                }
            }
        }
    }
}

def_kernel_bb!(4, 3, 4, 2);


macro_rules! def_kernel_bs {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        paste! {
            #[target_feature(enable = "avx")]
            pub unsafe fn [<kernel_$MR x $NR _bs_v0>]<F: MyFn, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                b: *const TB, b_rs: usize, b_cs: usize,
                c: *mut TC, c_rs: usize, c_cs: usize,
                ap_cur: *const TA,
                f: F,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let mut m_iter = (m / MR);
                let m_left = m % MR;
                let mut c_cur0 = c;
                let mut ap_cur = ap_cur;
                
                let n_iter0 = (n / NR);
                let n_left = (n % NR);
                let d_arr = [b_rs, b_cs, c_rs, c_cs];
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut b_cur = b;
                    let mut c_cur1 = c_cur0;
                    while n_iter > 0 {
                        [<ukernel_$MR x $NR _bs>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, k, d_arr, MR, NR, f);
                        n_iter -= 1;
                        b_cur = b_cur.add(NR*b_cs);
                        c_cur1 = c_cur1.add(NR*c_cs);
                    }
                    if n_left != 0 {
                        [<ukernel_$MR xn_bs>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, k, d_arr, MR, n_left, f);
                    }
                    m_iter -= 1;
                    ap_cur = ap_cur.add(MR*k);
                    c_cur0 = c_cur0.add(MR*c_rs);
                }

                $(
                    if (m_left+VS-1) / VS *VS == $mr_left {
                        let mut n_iter = n_iter0;
                        let mut b_cur = b;
                        let mut c_cur1 = c_cur0;
                        while n_iter > 0 {
                            [<ukernel_$mr_left x $NR _bs_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, k, d_arr, m_left, NR, f);
                            // [<ukernel_$mr_left x $NR _bs>]::<_, true>(ap_cur, b_cur, c_cur1, alpha, beta, k, d_arr, m_left, NR, f);
                            n_iter -= 1;
                            b_cur = b_cur.add(NR*b_cs);
                            c_cur1 = c_cur1.add(NR*c_cs);
                        }
                        if n_left != 0 {
                            [<ukernel_$mr_left xn_bs_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, k, d_arr, m_left, n_left, f);
                        }
                        return;
                    }
                )*

                asm!("vzeroupper");
            }        
        }
    };
}

def_kernel_bs!(4, 3, 4, 2);

use super::pack_avx::packa_panel_4;
macro_rules! def_kernel_sb {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        paste! {
            #[target_feature(enable = "avx")]
            pub unsafe fn [<kernel_$MR x $NR _sb_v0>]<F: MyFn, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                a: *const TB, a_rs: usize, a_cs: usize,
                b: *const TA,
                c: *mut TC, c_rs: usize, c_cs: usize,
                ap_buf: *mut TA,
                f: F,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let mut m_iter = (m / MR);
                let m_left = m % MR;
                let mut c_cur0 = c;
                let mut a_cur = a;
                
                let n_iter0 = (n / NR);
                let n_left = (n % NR);
                let d_arr = [0, 0, c_rs, c_cs];
                let ap_cur = ap_buf;
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut b_cur = b;
                    let mut c_cur1 = c_cur0;
                    [<packa_panel_$MR>](MR, k, a_cur, a_rs, a_cs, ap_cur, VS);
                    while n_iter > 0 {
                        [<ukernel_$MR x $NR _bb>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, k, d_arr, MR, NR, f);                        
                        n_iter -= 1;
                        b_cur = b_cur.add(NR*k);
                        c_cur1 = c_cur1.add(NR*c_cs);
                    }
                    if n_left != 0 {
                        [<ukernel_$MR x n _bb>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, k, d_arr, MR, n_left, f);
                    }   
                    m_iter -= 1;
                    a_cur = a_cur.add(MR*a_rs);
                    c_cur0 = c_cur0.add(MR*c_rs);
                }

                $(
                    if (m_left+VS-1) / VS *VS == $mr_left {
                        [<packa_panel_$MR>](m_left, k, a_cur, a_rs, a_cs, ap_cur, VS);
                        let mut n_iter = n_iter0;
                        let mut b_cur = b;
                        let mut c_cur1 = c_cur0;
                        while n_iter > 0 {
                            [<ukernel_$mr_left x $NR _bb_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, k, d_arr, m_left, NR, f);
                            n_iter -= 1;
                            b_cur = b_cur.add(NR*k);
                            c_cur1 = c_cur1.add(NR*c_cs);
                        }
                        if n_left != 0 {
                            [<ukernel_$mr_left xn_bb_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, k, d_arr, m_left, n_left, f);
                        }
                        return;
                    }
                )*

                asm!("vzeroupper");
            }        
        }
    };
}

def_kernel_sb!(4, 3, 4, 2);



pub(crate) unsafe fn kernel_4x3_sb<F: MyFn>(
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    a: *const TB, a_rs: usize, a_cs: usize,
    b: *const TB,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap_buf: *mut TA,
    f: F,
 ) { 
    if c_rs == 1 {
        kernel_4x3_sb_v0::<_, false>(
            m, n, k,
            alpha,
            a, a_rs, a_cs,
            b,
            c, c_rs, c_cs,
            ap_buf,
            f
        );
    } else {
        kernel_4x3_sb_v0::<_, true>(
            m, n, k,
            alpha,
            a, a_rs, a_cs,
            b,
            c, c_rs, c_cs,
            ap_buf,
            f
        );
    }
 } 


 pub(crate) unsafe fn kernel_4x3_bs<F: MyFn>(
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    b: *const TB, b_rs: usize, b_cs: usize,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap: *const TA,
    f: F,
) {  
    if c_rs == 1 {
        kernel_4x3_bs_v0::<_, false>(
            m, n, k,
            alpha,
            b, b_rs, b_cs,
            c, c_rs, c_cs,
            ap,
            f
        );
    } else {
        kernel_4x3_bs_v0::<_, true>(
            m, n, k,
            alpha,
            b, b_rs, b_cs,
            c, c_rs, c_cs,
            ap,
            f
        );
    }
}

// #[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn kernel_4x3<F: MyFn>(
   m: usize, n: usize, k: usize,
   alpha: *const TA,
   c: *mut TC,
   c_rs: usize, c_cs: usize,
   ap: *const TA, bp: *const TB,
   f: F,
) {
    if c_rs == 1 {
        kernel_4x3_bb::<_, false>(m, n, k, alpha, c, c_rs, c_cs, ap, bp, f)
    } else {
        kernel_4x3_bb::<_, true>(m, n, k, alpha, c, c_rs, c_cs, ap, bp, f)
    }
}
