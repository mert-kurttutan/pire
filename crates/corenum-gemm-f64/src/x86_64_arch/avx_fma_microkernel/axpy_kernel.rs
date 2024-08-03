

use std::{arch::x86_64::*, ptr::copy_nonoverlapping};


use crate::{TA,TB,TC};


use super::super::VS;


const K_UNROLL: usize = 4;


// TODO: optimize axpy for m=1 case,
// for each loop we use, less than optimal number of registers, less than 16
// modify so that we use 16 registers for each loop step


#[inline(always)]
unsafe fn v_loadu_n(mem_addr: *const f32, n: usize) -> __m256 {
   let mut a_arr = [0_f32; 8];
   copy_nonoverlapping(mem_addr, a_arr.as_mut_ptr(), n);
   _mm256_loadu_ps(a_arr.as_ptr())
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
//    let mut beta = *beta;
//    let beta_v = _mm256_broadcast_ss(&beta);
//     let n_lane = n / 4 * 4;
//    let m_lane4 = m / (VS*4) * VS*4;
//    let m_lane = m / VS * VS;

//     let mut ni = 0;
//     let mut a_cur0 = a;
//     let mut x_cur0 = x;
//    while ni < n_lane {
//        let mut a_cur = a_cur0;
//        let mut y_cur = y;
//        let mut xt_arr = [0.0; 4];
//          for i in 0..4 {
//              xt_arr[i] = *alpha * *x_cur0.add(i*incx);
//          }
//         let x_cur = xt_arr.as_ptr();

//         let xt0 = _mm256_broadcast_ss(&*x_cur);
//         let xt1 = _mm256_broadcast_ss(&*x_cur.add(1));
//         let xt2 = _mm256_broadcast_ss(&*x_cur.add(2));
//         let xt3 = _mm256_broadcast_ss(&*x_cur.add(3));
//         let mut mi = 0usize;
//         if beta == 1.0 {
//             while mi < m_lane4 {
//                 seq!(i in 0..4{
//                     let mut c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i)), xt0, _mm256_loadu_ps(y_cur.add(VS*i)));
//                     c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i+lda)), xt1, c~i);
//                     c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i+lda*2)), xt2, c~i);
//                     c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i+lda*3)), xt3, c~i);
//                     _mm256_storeu_ps(y_cur.add(VS*i), c~i);
//                 });
//                 a_cur = a_cur.add(VS*4);
//                 y_cur = y_cur.add(VS*4);
//                 mi += VS*4;
//             }
//             while mi < m_lane {
//                 let mut c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur), xt0, _mm256_loadu_ps(y_cur));
//                 c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda)), xt1, c0);
//                 c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*2)), xt2, c0);
//                 c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*3)), xt3, c0);
//                 _mm256_storeu_ps(y_cur, c0);
//                 a_cur = a_cur.add(VS);
//                 y_cur = y_cur.add(VS);
//                 mi += VS;
//             }
//             while mi < m {
//                 *y_cur = *a_cur * *x_cur + *y_cur;
//                 *y_cur = *a_cur.add(lda) * *x_cur.add(1) + *y_cur;
//                 *y_cur = *a_cur.add(lda*2) * *x_cur.add(2) + *y_cur;
//                 *y_cur = *a_cur.add(lda*3) * *x_cur.add(3) + *y_cur;
//                 a_cur = a_cur.add(1);
//                 y_cur = y_cur.add(1);
//                 mi += 1;
//             }
//         } else if beta == 0.0 {
//             while mi < m_lane4 {
//                 seq!(i in 0..4{
//                     let mut c~i = _mm256_mul_ps(_mm256_loadu_ps(a_cur.add(VS*i)), xt0);
//                     c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i+lda)), xt1, c~i);
//                     c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i+lda*2)), xt2, c~i);
//                     c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i+lda*3)), xt3, c~i);
//                     _mm256_storeu_ps(y_cur.add(VS*i), c~i);
//                 });
//                 a_cur = a_cur.add(VS*4);
//                 y_cur = y_cur.add(VS*4);
//                 mi += VS*4;
//             }
//             while mi < m_lane {
//                 let mut c0 = _mm256_mul_ps(_mm256_loadu_ps(a_cur), xt0);
//                 c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda)), xt1, c0);
//                 c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*2)), xt2, c0);
//                 c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*3)), xt3, c0);
//                 _mm256_storeu_ps(y_cur, c0);
//                 a_cur = a_cur.add(VS);
//                 y_cur = y_cur.add(VS);
//                 mi += VS;
//             }
//             while mi < m {
//                 *y_cur = *a_cur * *x_cur;
//                 *y_cur = *a_cur.add(lda) * *x_cur.add(1) + *y_cur;
//                 *y_cur = *a_cur.add(lda*2) * *x_cur.add(2) + *y_cur;
//                 *y_cur = *a_cur.add(lda*3) * *x_cur.add(3) + *y_cur;
//                 a_cur = a_cur.add(1);
//                 y_cur = y_cur.add(1);
//                 mi += 1;
//             }
//         } else {
//             while mi < m_lane4 {
//                 seq!(i in 0..4 {
//                     let cx~i = _mm256_mul_ps(_mm256_loadu_ps(y_cur.add(VS*i)), beta_v);
//                     let mut c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i)), xt0, cx~i);
//                     c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i+lda)), xt1, c~i);
//                     c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i+lda*2)), xt2, c~i);
//                     c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i+lda*3)), xt3, c~i);
//                     _mm256_storeu_ps(y_cur.add(VS*i), c~i);
//                 });
//                 a_cur = a_cur.add(VS*4);
//                 y_cur = y_cur.add(VS*4);
//                 mi += VS*4;
//             }
//             while mi < m_lane {
//                 let cx0 = _mm256_mul_ps(_mm256_loadu_ps(y_cur), beta_v);
//                 let mut c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur), xt0, cx0);
//                 c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda)), xt1, c0);
//                 c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*2)), xt2, c0);
//                 c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(lda*3)), xt3, c0);
//                 _mm256_storeu_ps(y_cur, c0);
//                 a_cur = a_cur.add(VS);
//                 y_cur = y_cur.add(VS);
//                 mi += VS;
//             }
//             while mi < m {
//                 *y_cur = *a_cur * *x_cur + *y_cur * beta;
//                 *y_cur = *a_cur.add(lda) * *x_cur.add(1) + *y_cur;
//                 *y_cur = *a_cur.add(lda*2) * *x_cur.add(2) + *y_cur;
//                 *y_cur = *a_cur.add(lda*3) * *x_cur.add(3) + *y_cur;
//                 a_cur = a_cur.add(1);
//                 y_cur = y_cur.add(1);
//                 mi += 1;
//             }
//         }
//         a_cur0 = a_cur0.add(lda*4);
//         x_cur0 = x_cur0.add(4*incx);
//        beta = 1.0;
//        ni += 4;
//    }

//    while ni < n {
//     let mut a_cur = a_cur0;
//     let mut y_cur = y;
//       let xt = *x_cur0 * *alpha;
//      let x_cur = &xt as *const f32;

//      let xt0 = _mm256_broadcast_ss(&*x_cur);
//      let mut mi = 0usize;
//      if beta == 1.0 {
//          while mi < m_lane4 {
//              seq!(i in 0..4{
//                  let c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i)), xt0, _mm256_loadu_ps(y_cur.add(VS*i)));
//                  _mm256_storeu_ps(y_cur.add(VS*i), c~i);
//              });
//              a_cur = a_cur.add(VS*4);
//              y_cur = y_cur.add(VS*4);
//              mi += VS*4;
//          }
//          while mi < m_lane {
//              let c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur), xt0, _mm256_loadu_ps(y_cur));
//              _mm256_storeu_ps(y_cur, c0);
//              a_cur = a_cur.add(VS);
//              y_cur = y_cur.add(VS);
//              mi += VS;
//          }
//          while mi < m {
//              *y_cur = *a_cur * *x_cur + *y_cur;
//              a_cur = a_cur.add(1);
//              y_cur = y_cur.add(1);
//              mi += 1;
//          }
//      } else if beta == 0.0 {
//          while mi < m_lane4 {
//              seq!(i in 0..4{
//                  let c~i = _mm256_mul_ps(_mm256_loadu_ps(a_cur.add(VS*i)), xt0);
//                  _mm256_storeu_ps(y_cur.add(VS*i), c~i);
//              });
//              a_cur = a_cur.add(VS*4);
//              y_cur = y_cur.add(VS*4);
//              mi += VS*4;
//          }
//          while mi < m_lane {
//              let c0 = _mm256_mul_ps(_mm256_loadu_ps(a_cur), xt0);
//              _mm256_storeu_ps(y_cur, c0);
//              a_cur = a_cur.add(VS);
//              y_cur = y_cur.add(VS);
//              mi += VS;
//          }
//          while mi < m {
//              *y_cur = *a_cur * *x_cur;
//              a_cur = a_cur.add(1);
//              y_cur = y_cur.add(1);
//              mi += 1;
//          }
//      } else {
//          while mi < m_lane4 {
//              seq!(i in 0..4 {
//                  let cx~i = _mm256_mul_ps(_mm256_loadu_ps(y_cur.add(VS*i)), beta_v);
//                  let c~i = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*i)), xt0, cx~i);
//                  _mm256_storeu_ps(y_cur.add(VS*i), c~i);
//              });
//              a_cur = a_cur.add(VS*4);
//              y_cur = y_cur.add(VS*4);
//              mi += VS*4;
//          }
//          while mi < m_lane {
//              let cx0 = _mm256_mul_ps(_mm256_loadu_ps(y_cur), beta_v);
//              let c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur), xt0, cx0);
//              _mm256_storeu_ps(y_cur, c0);
//              a_cur = a_cur.add(VS);
//              y_cur = y_cur.add(VS);
//              mi += VS;
//          }
//          while mi < m {
//              *y_cur = *a_cur * *x_cur + *y_cur * beta;
//              a_cur = a_cur.add(1);
//              y_cur = y_cur.add(1);
//              mi += 1;
//          }
//      }
//      a_cur0 = a_cur0.add(lda);
//      x_cur0 = x_cur0.add(1*incx);
//      ni += 1;
//     beta = 1.0;
// }


}


use seq_macro::seq;


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
//    for i in 0..m {
//        let mut a_cur = a.add(i*lda);
//        let mut x = x;
//        let y_cur = y.add(i * incy);
//        let mut acc_0 = _mm256_setzero_ps();
//        let mut acc_1 = _mm256_setzero_ps();
//        let mut acc_2 = _mm256_setzero_ps();
//        let mut acc_3 = _mm256_setzero_ps();


//        let mut p = 0;
//        while p < n_iter_unroll_vec {
//            acc_0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur), _mm256_loadu_ps(x), acc_0);
//            acc_1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS)), _mm256_loadu_ps(x.add(VS)), acc_1);
//            acc_2 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*2)), _mm256_loadu_ps(x.add(VS*2)), acc_2);
//            acc_3 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*3)), _mm256_loadu_ps(x.add(VS*3)), acc_3);


//            a_cur = a_cur.add(VS*K_UNROLL);
//            x = x.add(VS*K_UNROLL);
//            p += 1;
//        }


//        p = 0;
//        while p < n_iter_vec {
//            acc_0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur), _mm256_loadu_ps(x), acc_0);
//            a_cur = a_cur.add(VS);
//            x = x.add(VS);
//            p += 1;
//        }
//        let a_left_v = v_loadu_n(a_cur, n_left_vec);
//        let x_left_v = v_loadu_n(x, n_left_vec);
//        acc_0 = _mm256_fmadd_ps(a_left_v, x_left_v, acc_0);


//        // accumulate to scalar
//        acc_0 = _mm256_add_ps(acc_0, acc_1);
//        acc_0 = _mm256_add_ps(acc_0, acc_2);
//        acc_0 = _mm256_add_ps(acc_0, acc_3);


//        // acc_0 = _mm256_hadd_ps(acc_0, acc_0);


//        let mut acc_arr = [0.0; VS];
//        _mm256_storeu_ps(acc_arr.as_mut_ptr(), acc_0);
//        // acc += acc_arr[0] + acc_arr[1] + acc_arr[4] + acc_arr[5];


//        let acc = acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3] + acc_arr[4] + acc_arr[5] + acc_arr[6] + acc_arr[7];
//        if *beta == 0.0 {
//            *y_cur = acc * *alpha;
//        } else {
//            *y_cur = *beta * *y_cur + acc * *alpha;
//        }
//    }
}