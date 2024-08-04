use std::{arch::x86_64::*, ptr::copy_nonoverlapping};

use crate::{TA,TB,TC};

use half::f16;

use super::super::VS;


const K_UNROLL: usize = 4;


// TODO: optimize axpy for m=1 case,
// for each loop we use, less than optimal number of registers, less than 16
// modify so that we use 16 registers for each loop step


#[inline(always)]
unsafe fn v_loadu_n(mem_addr: *const TC, n: usize) -> __m256 {
   let mut a_arr = [0_u16; 8];
   copy_nonoverlapping(mem_addr as *const u16, a_arr.as_mut_ptr(), n);
   _mm256_cvtph_ps(_mm_loadu_si128(a_arr.as_ptr() as *const __m128i))
   // _mm256_loadu_ps(a_arr.as_ptr())
}

#[target_feature(enable = "avx,f16c")]
unsafe fn v_load_ph(mem_addr: *const f16) -> __m256 {
    let mut a_arr = [0_u16; 8];
    copy_nonoverlapping(mem_addr as *const u16, a_arr.as_mut_ptr(), 8);
    _mm256_cvtph_ps(_mm_loadu_si128(a_arr.as_ptr() as *const __m128i))
}

#[target_feature(enable = "avx,f16c")]
unsafe fn v_store_ph(mem_addr: *mut f16, a: __m256) {
    let a_ph = _mm256_cvtps_ph(a, 0);
    _mm_storeu_si128(mem_addr as *mut __m128i, a_ph);
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn acc_store(
    a: *const TA, lda: usize,
    c: *mut TC,
    cv: __m256, xt1: __m256, xt2: __m256, xt3: __m256,
) {
    let mut cv = cv;
    cv = _mm256_fmadd_ps(v_load_ph(a.add(lda)), xt1, cv);
    cv = _mm256_fmadd_ps(v_load_ph(a.add(lda*2)), xt2, cv);
    cv = _mm256_fmadd_ps(v_load_ph(a.add(lda*3)), xt3, cv);
    let cv_ph = _mm256_cvtps_ph(cv, 0);
    _mm_storeu_si128(c as *mut __m128i, cv_ph);
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn axpy_v_inner<const BETA: usize>(
    m_lane4: usize, m_lane: usize, m: usize,
    a: *const TA, lda: usize,
    y: *mut TC,
    xt0: __m256, xt1: __m256, xt2: __m256, xt3: __m256,
    x: *const f32,
    beta_v: __m256,
    beta: f32,
) {
    let mut a = a;
    let mut y = y;
    let mut mi = 0usize;
    while mi < m_lane4 {
        seq!(i in 0..4 {
            let c~i = if BETA == 1 {
                _mm256_fmadd_ps(v_load_ph(a.add(VS*i)), xt0, v_load_ph(y.add(VS*i)))
            } else if BETA == 0 {
                _mm256_mul_ps(v_load_ph(a.add(VS*i)), xt0)
            } else {
                let cx~i = _mm256_mul_ps(v_load_ph(y.add(VS*i)), beta_v);
                _mm256_fmadd_ps(v_load_ph(a.add(VS*i)), xt0, cx~i)
            };
            acc_store(a.add(VS*i), lda, y.add(VS*i), c~i, xt1, xt2, xt3);
        });
        a = a.add(VS*4);
        y = y.add(VS*4);
        mi += VS*4;
    }
    while mi < m_lane {
        let c0 = if BETA == 1 {
            _mm256_fmadd_ps(v_load_ph(a), xt0, v_load_ph(y))
        } else if BETA == 0 {
            _mm256_mul_ps(v_load_ph(a), xt0)
        } else {
            let cx0 = _mm256_mul_ps(v_load_ph(y), beta_v);
            _mm256_fmadd_ps(v_load_ph(a), xt0, cx0)
        };
        acc_store(a, lda, y, c0, xt1, xt2, xt3);
        a = a.add(VS);
        y = y.add(VS);
        mi += VS;
    }
    while mi < m {
        if BETA == 1 {
            *y = *a * f16::from_f32(*x) + *y;
        } else if BETA == 0 {
            *y = *a * f16::from_f32(*x);
        } else {
            *y = *a * f16::from_f32(*x) + *y * f16::from_f32(beta);
        }
        *y = *a.add(lda) * f16::from_f32(*x.add(1)) + *y;
        *y = *a.add(lda*2) * f16::from_f32(*x.add(2)) + *y;
        *y = *a.add(lda*3) * f16::from_f32(*x.add(3)) + *y;
        a = a.add(1);
        y = y.add(1);
        mi += 1;
    }
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn axpy_v_inner2<const BETA: usize>(
    m_lane4: usize, m_lane: usize, m: usize,
    a: *const TA,
    y: *mut TC,
    xt0: __m256,
    x: *const f32,
    beta_v: __m256,
    beta: f32,
) {
    let mut a = a;
    let mut y = y;
    let mut mi = 0usize;
    while mi < m_lane4 {
        seq!(i in 0..4 {
            let c~i = if BETA == 1 {
                _mm256_fmadd_ps(v_load_ph(a.add(VS*i)), xt0, v_load_ph(y.add(VS*i)))
            } else if BETA == 0 {
                _mm256_mul_ps(v_load_ph(a.add(VS*i)), xt0)
            } else {
                let cx~i = _mm256_mul_ps(v_load_ph(y.add(VS*i)), beta_v);
                _mm256_fmadd_ps(v_load_ph(a.add(VS*i)), xt0, cx~i)
            };
            v_store_ph(y.add(VS*i), c~i);
        });
        a = a.add(VS*4);
        y = y.add(VS*4);
        mi += VS*4;
    }
    while mi < m_lane {
        let c0 = if BETA == 1 {
            _mm256_fmadd_ps(v_load_ph(a), xt0, v_load_ph(y))
        } else if BETA == 0 {
            _mm256_mul_ps(v_load_ph(a), xt0)
        } else {
            let cx0 = _mm256_mul_ps(v_load_ph(y), beta_v);
            _mm256_fmadd_ps(v_load_ph(a), xt0, cx0)
        };
        v_store_ph(y, c0);
        a = a.add(VS);
        y = y.add(VS);
        mi += VS;
    }
    while mi < m {
      if BETA == 1 {
            *y = *a * f16::from_f32(*x) + *y;
      } else if BETA == 0 {
            *y = *a * f16::from_f32(*x);
      } else {
            *y = *a * f16::from_f32(*x) + *y * f16::from_f32(beta);
      }
        a = a.add(1);
        y = y.add(1);
        mi += 1;
    }
}

// The inner should traver along m dimenson for better hw prefetching since they are contiguous in memory
// inner loop should work multiple k to utilize the registers while keeping hw prefetching happy, so tune unrolling param
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn axpy_v(
   m: usize, n: usize,
   alpha: *const f32,
   a: *const TA, lda: usize,
   x: *const TB, incx: usize,
   beta: *const f32,
   y: *mut TC,
) {
    const K_UNROLL: usize = 4;
    const MR: usize = 4;
   let mut beta = *beta;
   let beta_v = _mm256_broadcast_ss(&beta);
    let n_lane = n / K_UNROLL * K_UNROLL;
   let m_lane4 = m / (VS*MR) * VS*MR;
   let m_lane = m / VS * VS;

    let mut ni = 0;
    let mut a_cur = a;
    let mut x_cur = x;
    let mut xtv_arr = [_mm256_setzero_ps(); K_UNROLL];
   while ni < n_lane {
       let mut xt_arr = [0.0; K_UNROLL];
        for i in 0..K_UNROLL {
            xt_arr[i] = *alpha * (*x_cur.add(i*incx)).to_f32();
            xtv_arr[i] = _mm256_broadcast_ss(&xt_arr[i]);
        }
        let xt = xt_arr.as_ptr();

        if beta == 1.0 {
            axpy_v_inner::<1>(m_lane4, m_lane, m, a_cur, lda, y, xtv_arr[0], xtv_arr[1], xtv_arr[2], xtv_arr[3], xt, beta_v, beta);
        } else if beta == 0.0 {
            axpy_v_inner::<0>(m_lane4, m_lane, m, a_cur, lda, y, xtv_arr[0], xtv_arr[1], xtv_arr[2], xtv_arr[3], xt, beta_v, beta);
        } else {
            axpy_v_inner::<2>(m_lane4, m_lane, m, a_cur, lda, y, xtv_arr[0], xtv_arr[1], xtv_arr[2], xtv_arr[3], xt, beta_v, beta);
        }
        a_cur = a_cur.add(lda*K_UNROLL);
        x_cur = x_cur.add(incx*K_UNROLL);
       beta = 1.0;
       ni += K_UNROLL;
   }

   while ni < n {
      let xt = (*x_cur).to_f32() * *alpha;
        let xt_ptr = &xt as *const f32;
     let xt0 = _mm256_broadcast_ss(&*xt_ptr);
     if beta == 1.0 {
        axpy_v_inner2::<1>(m_lane4, m_lane, m, a_cur, y, xt0, xt_ptr, beta_v, beta);
    } else if beta == 0.0 {
        axpy_v_inner2::<0>(m_lane4, m_lane, m, a_cur, y, xt0, xt_ptr, beta_v, beta);
    } else {
        axpy_v_inner2::<2>(m_lane4, m_lane, m, a_cur, y, xt0, xt_ptr, beta_v, beta);
    }
     a_cur = a_cur.add(lda);
     x_cur = x_cur.add(incx);
     ni += 1;
    beta = 1.0;
}
}

use seq_macro::seq;

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn acc_vec(
    x: __m256,
)-> TC {
    let mut acc_arr = [0.0; VS];
    _mm256_storeu_ps(acc_arr.as_mut_ptr(), x);
    let x = acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3] + acc_arr[4] + acc_arr[5] + acc_arr[6] + acc_arr[7];
    f16::from_f32(x)
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn axpy_d(
   m: usize, n: usize,
   alpha: *const f32,
   a: *const TA, lda: usize,
   x: *const TB,
   beta: *const f32,
   y: *mut TC, incy: usize
) {
   let n_iter_unroll_vec = n / (K_UNROLL * VS);
   let n_left_unroll_vec = n % (K_UNROLL * VS);
   let n_iter_vec = n_left_unroll_vec / VS;
   let n_left_vec = n_left_unroll_vec % VS;
   let alpha_f16 = f16::from_f32(*alpha);
   let beta_f16 = f16::from_f32(*beta);
   let m3 = (m / 3) * 3;
    let mut y_cur = y;
    let mut a_cur0 = a;
    let mut i = 0;
   while i < m3 {
       let mut a_cur = a_cur0;
       let mut x_cur = x;
       let mut acc_arr = [_mm256_setzero_ps(); 4*3];
       let mut p = 0;
       while p < n_iter_unroll_vec {
        seq!(q in 0..3 {
            acc_arr[q*4] = _mm256_fmadd_ps(v_load_ph(a_cur.add(lda*q)), v_load_ph(x_cur), acc_arr[q*4]);
            acc_arr[q*4+1] = _mm256_fmadd_ps(v_load_ph(a_cur.add(lda*q+VS)), v_load_ph(x_cur.add(VS)), acc_arr[q*4+1]);
            acc_arr[q*4+2] = _mm256_fmadd_ps(v_load_ph(a_cur.add(lda*q+VS*2)), v_load_ph(x_cur.add(VS*2)), acc_arr[q*4+2]);
            acc_arr[q*4+3] = _mm256_fmadd_ps(v_load_ph(a_cur.add(lda*q+VS*3)), v_load_ph(x_cur.add(VS*3)), acc_arr[q*4+3]);
        });
           a_cur = a_cur.add(VS*K_UNROLL);
           x_cur = x_cur.add(VS*K_UNROLL);
           p += 1;
       }

       p = 0;
       while p < n_iter_vec {
            seq!(q in 0..3 {
                acc_arr[q*4] = _mm256_fmadd_ps(v_load_ph(a_cur.add(lda*q)), v_load_ph(x_cur), acc_arr[q*4]);
            });
           a_cur = a_cur.add(VS);
           x_cur = x_cur.add(VS);
           p += 1;
       }
       let x_left_v = v_loadu_n(x, n_left_vec);

       // accumulate to scalar
       seq!(q in 0..3 {
        let a_lef_v = v_loadu_n(a_cur.add(lda*q), n_left_vec);
        acc_arr[q*4] = _mm256_fmadd_ps(a_lef_v, x_left_v, acc_arr[q*4]);
        acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+1]);
        acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+2]);
        acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+3]);
       });

       let acc1 = acc_vec(acc_arr[0]);
       let acc2 = acc_vec(acc_arr[4]);
      let acc3 = acc_vec(acc_arr[8]);
       if *beta == 0.0 {
           *y_cur = acc1 * alpha_f16;
            *y_cur.add(incy) = acc2 * alpha_f16;
            *y_cur.add(incy*2) = acc3 * alpha_f16;
       } else {
           *y_cur = beta_f16 * *y_cur + acc1 * alpha_f16;
            *y_cur.add(incy) = beta_f16 * *y_cur.add(incy) + acc2 * alpha_f16;
            *y_cur.add(incy*2) = beta_f16 * *y_cur.add(incy*2) + acc3 * alpha_f16;
       }
         a_cur0 = a_cur0.add(3*lda);
         y_cur = y_cur.add(3*incy);
         i += 3;
   }

   while i < m {
    let mut a_cur = a_cur0;
    let mut x_cur = x;
    let mut acc_arr = [_mm256_setzero_ps(); 4];
    let mut p = 0;
    while p < n_iter_unroll_vec {
     seq!(q in 0..1 {
         acc_arr[q*4] = _mm256_fmadd_ps(v_load_ph(a_cur.add(lda*q)), v_load_ph(x_cur), acc_arr[q*4]);
         acc_arr[q*4+1] = _mm256_fmadd_ps(v_load_ph(a_cur.add(lda*q+VS)), v_load_ph(x_cur.add(VS)), acc_arr[q*4+1]);
         acc_arr[q*4+2] = _mm256_fmadd_ps(v_load_ph(a_cur.add(lda*q+VS*2)), v_load_ph(x_cur.add(VS*2)), acc_arr[q*4+2]);
         acc_arr[q*4+3] = _mm256_fmadd_ps(v_load_ph(a_cur.add(lda*q+VS*3)), v_load_ph(x_cur.add(VS*3)), acc_arr[q*4+3]);
     });
        a_cur = a_cur.add(VS*K_UNROLL);
        x_cur = x_cur.add(VS*K_UNROLL);
        p += 1;
    }

    p = 0;
    while p < n_iter_vec {
         seq!(q in 0..1 {
             acc_arr[q*4] = _mm256_fmadd_ps(v_load_ph(a_cur.add(lda*q)), v_load_ph(x_cur), acc_arr[q*4]);
         });
        a_cur = a_cur.add(VS);
        x_cur = x_cur.add(VS);
        p += 1;
    }
    let x_left_v = v_loadu_n(x, n_left_vec);

    // accumulate to scalar
    seq!(q in 0..1 {
     let a_lef_v = v_loadu_n(a_cur.add(lda*q), n_left_vec);
     acc_arr[q*4] = _mm256_fmadd_ps(a_lef_v, x_left_v, acc_arr[q*4]);
     acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+1]);
     acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+2]);
     acc_arr[q*4] = _mm256_add_ps(acc_arr[q*4], acc_arr[q*4+3]);
    });

    let acc1 = acc_vec(acc_arr[0]);
    if *beta == 0.0 {
        *y_cur = acc1 * alpha_f16;
    } else {
        *y_cur = beta_f16 * *y_cur + acc1 * alpha_f16;
    }
      a_cur0 = a_cur0.add(lda);
      y_cur = y_cur.add(incy);
      i += 1;
}
}