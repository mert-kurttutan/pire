

use std::{arch::x86_64::*, ptr::copy_nonoverlapping};


use crate::{TA,TB,TC};


// use super::super::VS;
const VS: usize = 8;


const K_UNROLL: usize = 4;



#[inline(always)]
unsafe fn v_storeu_n(mem_addr: *mut f32, a: __m256, n: usize) {
   let mut a_arr = [0_f32; 8];
   _mm256_storeu_ps(a_arr.as_mut_ptr(), a);
   copy_nonoverlapping(a_arr.as_ptr(), mem_addr, n);
}


#[inline(always)]
unsafe fn v_loadu_n(mem_addr: *const f32, n: usize) -> __m256 {
   let mut a_arr = [0_f32; 8];
   copy_nonoverlapping(mem_addr, a_arr.as_mut_ptr(), n);
   _mm256_loadu_ps(a_arr.as_ptr())
}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn axpy_v(
   m: usize, n: usize,
   alpha: *const TA,
   a: *const TA, lda: usize,
   x: *const TB, incx: usize,
   beta: *const TC,
   y: *mut TC,
) {
   let mut beta = *beta;


   let m_lane = m / VS * VS;
   let m_left = m % VS;

   let load_f = v_loadu_n;
   let store_f = v_storeu_n;
   for col in 0..n {
       let lhs = a.wrapping_offset(col as isize * lda as isize);
       let rhs = *x.wrapping_offset(col as isize * incx as isize);

       let beta_v = _mm256_broadcast_ss(&beta);


       let rhs = _mm256_mul_ps(_mm256_broadcast_ss(&*alpha), _mm256_broadcast_ss(&rhs));


       if beta == 0.0 {
           let mut row = 0usize;
           while row < m_lane {
               let dst_ptr = y.add(row);
               let lhs = _mm256_loadu_ps(lhs.add(row));
               _mm256_storeu_ps(
                   dst_ptr,
                   _mm256_mul_ps(lhs, rhs)
               );
               row += VS;
           }


           if m_left != 0 {
               let lhs = load_f(lhs.add(row), m_left);
               store_f(y.add(row), _mm256_mul_ps(lhs, rhs), m_left);
           }
       } else {
           let mut row = 0usize;
           while row < m_lane {
               let dst_ptr = y.add(row);
               let dst_v = if beta == 1.0 {
                   _mm256_loadu_ps(dst_ptr)
               } else {
                   _mm256_mul_ps(beta_v, _mm256_loadu_ps(dst_ptr))
               };
               let lhs = _mm256_loadu_ps(lhs.add(row));


               _mm256_storeu_ps(
                   dst_ptr,
                   _mm256_fmadd_ps(lhs, rhs, dst_v)
               );
               row += VS;
           }
           if m_left != 0 {
               let dst_v = if beta == 1.0 {
                   load_f(y.add(row), m_left)
               } else {
                   _mm256_mul_ps(beta_v, load_f(y.add(row), m_left))
               };


               let lhs = load_f(lhs.add(row), m_left);
               store_f(y.add(row), _mm256_fmadd_ps(lhs, rhs, dst_v), m_left);
           }
       }
       beta = 1.0;
   }
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
   let n_iter_unroll_vec = n / (K_UNROLL * VS);
   let n_left_unroll_vec = n % (K_UNROLL * VS);
   let n_iter_vec = n_left_unroll_vec / VS;
   let n_left_vec = n_left_unroll_vec % VS;
   for i in 0..m {
       let mut a_cur = a.add(i*lda);
       let mut x = x;
       let y_cur = y.add(i * incy);
       let mut acc_0 = _mm256_setzero_ps();
       let mut acc_1 = _mm256_setzero_ps();
       let mut acc_2 = _mm256_setzero_ps();
       let mut acc_3 = _mm256_setzero_ps();


       let mut p = 0;
       while p < n_iter_unroll_vec {
           acc_0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur), _mm256_loadu_ps(x), acc_0);
           acc_1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS)), _mm256_loadu_ps(x.add(VS)), acc_1);
           acc_2 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*2)), _mm256_loadu_ps(x.add(VS*2)), acc_2);
           acc_3 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur.add(VS*3)), _mm256_loadu_ps(x.add(VS*3)), acc_3);


           a_cur = a_cur.add(VS*K_UNROLL);
           x = x.add(VS*K_UNROLL);
           p += 1;
       }


       p = 0;
       while p < n_iter_vec {
           acc_0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_cur), _mm256_loadu_ps(x), acc_0);
           a_cur = a_cur.add(VS);
           x = x.add(VS);
           p += 1;
       }
       let a_left_v = v_loadu_n(a_cur, n_left_vec);
       let x_left_v = v_loadu_n(x, n_left_vec);
       acc_0 = _mm256_fmadd_ps(a_left_v, x_left_v, acc_0);


       // accumulate to scalar
       acc_0 = _mm256_add_ps(acc_0, acc_1);
       acc_0 = _mm256_add_ps(acc_0, acc_2);
       acc_0 = _mm256_add_ps(acc_0, acc_3);


       // acc_0 = _mm256_hadd_ps(acc_0, acc_0);


       let mut acc_arr = [0.0; VS];
       _mm256_storeu_ps(acc_arr.as_mut_ptr(), acc_0);
       // acc += acc_arr[0] + acc_arr[1] + acc_arr[4] + acc_arr[5];


       let acc = acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3] + acc_arr[4] + acc_arr[5] + acc_arr[6] + acc_arr[7];
       if *beta == 0.0 {
           *y_cur = acc * *alpha;
       } else {
           *y_cur = *beta * *y_cur + acc * *alpha;
       }
   }
}

