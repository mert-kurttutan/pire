pub mod asm_ukernel;
pub(crate) mod axpy_kernel;

pub(crate) use asm_ukernel::*;

pub(crate) use axpy_kernel::*;

use paste::paste;
use std::arch::asm;

const VS: usize = 8;

use crate::MyFn;

use half::f16;

#[target_feature(enable = "avx,fma")]
pub unsafe fn axpy<F: MyFn>(
   m: usize, n: usize,
   alpha: *const f32,
   a: *const f16, a_rs: usize, a_cs: usize,
   x: *const f16, incx: usize,
   beta: *const f32,
   y: *mut f16, incy: usize,
   f: F,
) {
   if a_cs == 1 && incx == 1 {
       axpy_d(m, n, alpha, a, a_rs, x, beta, y, incy);
       for i in 0..m {
           f.call(y.add(i*incy), m);
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
   if a_cs == 1 {
       for i in 0..m {
           let a_cur = a.add(i*a_rs);
           let y_cur = y.add(i * incy);
           let mut acc = f16::ZERO;
           for j in 0..n {
               let x_cur = x.add(j * incx);
               acc += *a_cur.add(j) * *x_cur;
           }
           *y_cur = beta_f16 * *y_cur + alpha_f16 * acc;
           f.call(y_cur, 1);
       }
       return;
   }
   if a_rs == 1 {
       for i in 0..m {
           let y_cur = y.add(i*incy);
           let mut acc = f16::ZERO;
           for j in 0..n {
               let a_cur = a.add(j*a_cs);
               let x_cur = x.add(j*incx);
               acc += *a_cur.add(i) * *x_cur;
           }
           *y_cur = beta_f16 * *y_cur + alpha_f16 * acc;
            f.call(y_cur, 1);
       }
       return;
   }
}

macro_rules! def_kernel_bb {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        paste! {
            #[target_feature(enable = "avx")]
            pub unsafe fn kernel_bb<F: MyFn, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const f32,
                beta: *const f32,
                c: *mut f16, c_rs: usize, c_cs: usize,
                ap: *const f32, bp: *const f32,
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
                        [<ukernel_$MR x $NR _bb>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, MR, f);
                        n_iter -= 1;
                        bp_cur = bp_cur.add(NR*k);
                        c_cur1 = c_cur1.add(NR*c_cs);
                    }
                    // let a_pft1_offset = ($MR+(n_iter0-n_iter)*2)*4*k;
                    if n_left != 0 {
                        [<ukernel_$MR x n _bb>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, MR, n_left, f);
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
                            [<ukernel_$mr_left x $NR _bb>]::<_, true>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, m_left, f);
                            n_iter -= 1;
                            bp_cur = bp_cur.add(NR*k);
                            c_cur1 = c_cur1.add(NR*c_cs);
                        }
                        if n_left !=0 {
                            [<ukernel_$mr_left x n_bb>]::<_, true>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, m_left, n_left, f);
                        }
                        return;
                    }
                )*

                asm!("vzeroupper");
            }
        }   
    };
}

def_kernel_bb!(24, 4, 24, 16, 8);


// #[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn kernel<F: MyFn>(
   m: usize, n: usize, k: usize,
   alpha: *const f32, beta: *const f32,
   c: *mut f16,
   c_rs: usize, c_cs: usize,
   ap: *const f32, bp: *const f32,
   f: F,
) {
    if c_rs == 1 {
        kernel_bb::<_, false>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    } else {
        kernel_bb::<_, true>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    }
}
