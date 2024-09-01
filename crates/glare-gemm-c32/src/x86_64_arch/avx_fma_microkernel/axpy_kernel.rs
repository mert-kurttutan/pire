use std::{arch::x86_64::*, ptr::copy_nonoverlapping};

use crate::{TA,TB,TC};

use super::VS;

const K_UNROLL: usize = 4;

// TODO: optimize axpy for m=1 case,
// for each loop we use, less than optimal number of registers, less than 16
// modify so that we use 16 registers for each loop step


#[inline(always)]
unsafe fn v_loadu_n(mem_addr: *const TC, n: usize) -> __m256 {
   let mut a_arr = [TC::ZERO; 4];
   copy_nonoverlapping(mem_addr, a_arr.as_mut_ptr(), n);
   _mm256_loadu_ps(a_arr.as_ptr() as *const f32)
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn acc_store(
    a: *const TA, lda: usize,
    c: *mut TC,
    cv: __m256, xt1: __m256, xt2: __m256, xt3: __m256,
) {
    let mut cv = cv;
    let a = a as *const f32;
    cv = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(lda)), xt1, cv);
    cv = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(lda*2)), xt2, cv);
    cv = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(lda*3)), xt3, cv);
    _mm256_storeu_ps(c as *mut f32, cv);
}

// #[target_feature(enable = "avx,fma")]
// pub(crate) unsafe fn acc_start<const BETA: usize>(
//     a: *const TA, y: *const TC,
//     xt0: __m256, beta_v: __m256,
// ) -> __m256 {
//     let a = a as *const f32;
//     let y = y as *const f32;
//     // if BETA == 1 {
//     //     _mm256_fmadd_ps(_mm256_loadu_ps(a), xt0, _mm256_loadu_ps(y))
//     // } else if BETA == 0 {
//     //     _mm256_mul_ps(_mm256_loadu_ps(a), xt0)
//     // } else {
//     //     let cx0 = _mm256_mul_ps(_mm256_loadu_ps(y), beta_v);
//     //     _mm256_fmadd_ps(_mm256_loadu_ps(a), xt0, cx0)
//     // }
// }

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn axpy_v_inner<const BETA: usize>(
    m: usize,
    a: *const TA, lda: usize,
    y: *mut TC,
    x_cur: *const TB,
    incx: usize,
    beta_v: __m256,
    beta: f32,
    alpha: *const TA,
) {
    // const K_UNROLL: usize = 4;
    // const MR: usize = 4;
    // let m_lane4 = m / (VS*MR) * VS*MR;
    // let m_lane = m / VS * VS;
    // let mut a = a;
    // let mut y = y;
    // let mut mi = 0usize;
    // let mut xt_arr = [TA::ZERO; K_UNROLL];
    // let mut xtv_arr = [_mm256_setzero_ps(); K_UNROLL];
    // for i in 0..K_UNROLL {
    //     xt_arr[i] = *alpha * *x_cur.add(i*incx);
    //     xtv_arr[i] = _mm256_broadcast_ss(&xt_arr[i]);
    // }
    // let x = xt_arr.as_ptr();
    // while mi < m_lane4 {
    //     seq!(i in 0..4 {
    //         let c~i = acc_start::<BETA>(a.add(VS*i), y.add(VS*i), xtv_arr[0], beta_v);
    //         acc_store(a.add(VS*i), lda, y.add(VS*i), c~i, xtv_arr[1], xtv_arr[2], xtv_arr[3]);
    //     });
    //     a = a.add(VS*4);
    //     y = y.add(VS*4);
    //     mi += VS*4;
    // }
    // while mi < m_lane {
    //     let c0 = acc_start::<BETA>(a, y, xtv_arr[0], beta_v);
    //     acc_store(a, lda, y, c0, xtv_arr[1], xtv_arr[2], xtv_arr[3]);
    //     a = a.add(VS);
    //     y = y.add(VS);
    //     mi += VS;
    // }
    // while mi < m {
    //     if BETA == 1 {
    //         *y = *a * *x + *y;
    //     } else if BETA == 0 {
    //         *y = *a * *x;
    //     } else {
    //         *y = *a * *x + *y * beta;
    //     }
    //     *y = *a.add(lda) * *x.add(1) + *y;
    //     *y = *a.add(lda*2) * *x.add(2) + *y;
    //     *y = *a.add(lda*3) * *x.add(3) + *y;
    //     a = a.add(1);
    //     y = y.add(1);
    //     mi += 1;
    // }
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn axpy_v_inner2<const BETA: usize>(
    m: usize,
    a: *const TA,
    y: *mut TC,
    x_cur: *const TB,
    beta_v: __m256,
    beta: f32,
    alpha: *const f32,
) {
    // const MR: usize = 4;
    // let m_lane4 = m / (VS*MR) * VS*MR;
    // let m_lane = m / VS * VS;
    // let mut a = a;
    // let mut y = y;
    // let mut mi = 0usize;
    // let xt = *x_cur * *alpha;
    // let x = &xt as *const f32;
    // let xt0 = _mm256_broadcast_ss(&*x);
    // while mi < m_lane4 {
    //     seq!(i in 0..4 {
    //         let c~i = acc_start::<BETA>(a.add(VS*i), y.add(VS*i), xt0, beta_v);
    //         _mm256_storeu_ps(y.add(VS*i), c~i);
    //     });
    //     a = a.add(VS*4);
    //     y = y.add(VS*4);
    //     mi += VS*4;
    // }
    // while mi < m_lane {
    //     let c0 = acc_start::<BETA>(a, y, xt0, beta_v);
    //     _mm256_storeu_ps(y, c0);
    //     a = a.add(VS);
    //     y = y.add(VS);
    //     mi += VS;
    // }
    // while mi < m {
    //     if BETA == 1 {
    //         *y = *a * *x + *y;
    //     } else if BETA == 0 {
    //         *y = *a * *x;
    //     } else {
    //         *y = *a * *x + *y * beta;
    //     }
    //     a = a.add(1);
    //     y = y.add(1);
    //     mi += 1;
    // }
}

// The inner should traver along m dimenson for better hw prefetching since they are contiguous in memory
// inner loop should work multiple k to utilize the registers while keeping hw prefetching happy, so tune unrolling param
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn axpy_v(
   m: usize, n: usize,
   alpha: *const TA,
   a: *const TA, lda: usize,
   x: *const TB, incx: usize,
   beta: *const TC,
   y: *mut TC,
) {
//     const K_UNROLL: usize = 4;
//    let mut beta = *beta;
//    let beta_v = _mm256_broadcast_ss(&beta);
//     let n_lane = n / K_UNROLL * K_UNROLL;

//     let mut ni = 0;
//     let mut a_cur = a;
//     let mut x_cur = x;
//     while ni < n_lane {
//         if beta == 1.0 {
//             axpy_v_inner::<1>(m, a_cur, lda, y, x_cur, incx, beta_v, beta, alpha);
//         } else if beta == 0.0 {
//             axpy_v_inner::<0>(m, a_cur, lda, y, x_cur, incx, beta_v, beta, alpha);
//         } else {
//             axpy_v_inner::<2>(m, a_cur, lda, y, x_cur, incx, beta_v, beta, alpha);
//         }
//         a_cur = a_cur.add(lda*K_UNROLL);
//         x_cur = x_cur.add(incx*K_UNROLL);
//        beta = 1.0;
//        ni += K_UNROLL;
//    }

//    while ni < n {
//      if beta == 1.0 {
//         axpy_v_inner2::<1>(m, a_cur, y, x_cur, beta_v, beta, alpha);
//     } else if beta == 0.0 {
//         axpy_v_inner2::<0>(m, a_cur, y, x_cur, beta_v, beta, alpha);
//     } else {
//         axpy_v_inner2::<2>(m, a_cur, y, x_cur, beta_v, beta, alpha);
//     }
//      a_cur = a_cur.add(lda);
//      x_cur = x_cur.add(incx);
//      ni += 1;
//     beta = 1.0;
// }
}

use seq_macro::seq;

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn acc_vec(
    x: __m256,
)-> TC {
    let mut acc_arr = [TC::ZERO; VS];
    // _mm256_storeu_ps(acc_arr.as_mut_ptr(), x);
    acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3] + acc_arr[4] + acc_arr[5] + acc_arr[6] + acc_arr[7]
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn axpy_d(
   m: usize, n: usize,
   alpha: *const TA,
   a: *const TA, lda: usize,
   x: *const TB,
   beta: *const TC,
   y: *mut TC, incy: usize
) {
//    let n_iter_unroll_vec = n / (K_UNROLL * VS);
//    let n_left_unroll_vec = n % (K_UNROLL * VS);
//    let n_iter_vec = n_left_unroll_vec / VS;
//    let n_left_vec = n_left_unroll_vec % VS;
//    let m3 = (m / 3) * 3;
//     let mut y_cur = y;
//     let mut a_cur0 = a;
//     let mut i = 0;
//    while i < m3 {
//        let mut a_cur = a_cur0;
//        let mut x_cur = x;
//        let mut acc_arr = [_mm256_setzero_ps(); 4*3];
//        let mut p = 0;
//        while p < n_iter_unroll_vec {
//         seq!(q in 0..3 {
//             acc_arr[q*4] = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*q)), _mm256_loadu_ps(x_cur), acc_arr[q*4]);
//             acc_arr[q*4+1] = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*q+VS)), _mm256_loadu_ps(x_cur.add(VS)), acc_arr[q*4+1]);
//             acc_arr[q*4+2] = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*q+VS*2)), _mm256_loadu_ps(x_cur.add(VS*2)), acc_arr[q*4+2]);
//             acc_arr[q*4+3] = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*q+VS*3)), _mm256_loadu_ps(x_cur.add(VS*3)), acc_arr[q*4+3]);
//         });
//            a_cur = a_cur.add(VS*K_UNROLL);
//            x_cur = x_cur.add(VS*K_UNROLL);
//            p += 1;
//        }

//        p = 0;
//        while p < n_iter_vec {
//             seq!(q in 0..3 {
//                 acc_arr[q*4] = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*q)), _mm256_loadu_ps(x_cur), acc_arr[q*4]);
//             });
//            a_cur = a_cur.add(VS);
//            x_cur = x_cur.add(VS);
//            p += 1;
//        }
//        let x_left_v = v_loadu_n(x_cur, n_left_vec);

//        // accumulate to scalar
//        seq!(q in 0..3 {
//         let a_lef_v = v_loadu_n(a_cur.add(lda*q), n_left_vec);
//         acc_arr[q*4] = _mm256_fmadd_ps(a_lef_v, x_left_v, acc_arr[q*4]);
//         acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+1]);
//         acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+2]);
//         acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+3]);
//        });

//        let acc1 = acc_vec(acc_arr[0]);
//        let acc2 = acc_vec(acc_arr[4]);
//          let acc3 = acc_vec(acc_arr[8]);
//        if *beta == 0.0 {
//            *y_cur = acc1 * *alpha;
//             *y_cur.add(incy) = acc2 * *alpha;
//             *y_cur.add(incy*2) = acc3 * *alpha;
//        } else {
//            *y_cur = *beta * *y_cur + acc1 * *alpha;
//             *y_cur.add(incy) = *beta * *y_cur.add(incy) + acc2 * *alpha;
//             *y_cur.add(incy*2) = *beta * *y_cur.add(incy*2) + acc3 * *alpha;
//        }
//          a_cur0 = a_cur0.add(3*lda);
//          y_cur = y_cur.add(3*incy);
//          i += 3;
//    }

//    while i < m {
//     let mut a_cur = a_cur0;
//     let mut x_cur = x;
//     let mut acc_arr = [_mm256_setzero_ps(); 4];
//     let mut p = 0;
//     while p < n_iter_unroll_vec {
//      seq!(q in 0..1 {
//          acc_arr[q*4] = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*q)), _mm256_loadu_ps(x_cur), acc_arr[q*4]);
//          acc_arr[q*4+1] = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*q+VS)), _mm256_loadu_ps(x_cur.add(VS)), acc_arr[q*4+1]);
//          acc_arr[q*4+2] = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*q+VS*2)), _mm256_loadu_ps(x_cur.add(VS*2)), acc_arr[q*4+2]);
//          acc_arr[q*4+3] = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*q+VS*3)), _mm256_loadu_ps(x_cur.add(VS*3)), acc_arr[q*4+3]);
//      });
//         a_cur = a_cur.add(VS*K_UNROLL);
//         x_cur = x_cur.add(VS*K_UNROLL);
//         p += 1;
//     }

//     p = 0;
//     while p < n_iter_vec {
//          seq!(q in 0..1 {
//              acc_arr[q*4] = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*q)), _mm256_loadu_ps(x_cur), acc_arr[q*4]);
//          });
//         a_cur = a_cur.add(VS);
//         x_cur = x_cur.add(VS);
//         p += 1;
//     }
//     let x_left_v = v_loadu_n(x_cur, n_left_vec);

//     // accumulate to scalar
//     seq!(q in 0..1 {
//      let a_lef_v = v_loadu_n(a_cur.add(lda*q), n_left_vec);
//      acc_arr[q*4] = _mm256_fmadd_ps(a_lef_v, x_left_v, acc_arr[q*4]);
//      acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+1]);
//      acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+2]);
//      acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+3]);
//     });

//     let acc1 = acc_vec(acc_arr[0]);
//     if *beta == 0.0 {
//         *y_cur = acc1 * *alpha;
//     } else {
//         *y_cur = *beta * *y_cur + acc1 * *alpha;
//     }
//       a_cur0 = a_cur0.add(lda);
//       y_cur = y_cur.add(incy);
//       i += 1;
// }
}