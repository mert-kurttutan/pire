use std::arch::x86_64::*;

use crate::{TA,TB,TC};
use seq_macro::seq;

use super::super::VS;


#[target_feature(enable = "avx,avx2")]
unsafe fn load_v<const N: usize>(a: *const TA) -> __m256i {
    _mm256_castps_si256(_mm256_broadcast_ss(&*(a as *const f32)))
}


// TODO: optimize axpy for m=1 case,
// for each loop we use, less than optimal number of registers, less than 16
// modify so that we use 16 registers for each loop step

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave<const PARTIAL: bool>(
    a: *const TA, lda: usize, kl: usize,
) -> (__m256i, __m256i) {

    let (a0, b0, c0, d0) = if PARTIAL {
        let mut a0 = if kl > 0 { _mm256_castsi128_si256(_mm_loadu_si128(a as *const __m128i)) } else { _mm256_setzero_si256() };
        let mut b0 = if kl > 1 { _mm256_castsi128_si256(_mm_loadu_si128(a.add(lda) as *const __m128i)) } else { _mm256_setzero_si256() };
        let mut c0 = if kl > 2 { _mm256_castsi128_si256(_mm_loadu_si128(a.add(lda*2) as *const __m128i)) } else { _mm256_setzero_si256() };
        let mut d0 = if kl > 3 { _mm256_castsi128_si256(_mm_loadu_si128(a.add(lda*3) as *const __m128i)) } else { _mm256_setzero_si256() };
        (a0, b0, c0, d0)
    } else {
        (
            _mm256_castsi128_si256(_mm_loadu_si128(a as *const __m128i)),
            _mm256_castsi128_si256(_mm_loadu_si128(a.add(lda) as *const __m128i)),
            _mm256_castsi128_si256(_mm_loadu_si128(a.add(lda*2) as *const __m128i)),
            _mm256_castsi128_si256(_mm_loadu_si128(a.add(lda*3) as *const __m128i))
        )
    };

    let t0 = _mm256_unpacklo_epi8(a0, b0);
    let t1 = _mm256_unpackhi_epi8(a0, b0);
    
    let p0 = _mm256_unpacklo_epi8(c0, d0);
    let p1 = _mm256_unpackhi_epi8(c0, d0);

    let d0 = _mm256_unpacklo_epi16(t0, p0);
    let d1 = _mm256_unpackhi_epi16(t0, p0);

    let e0 = _mm256_unpacklo_epi16(t1, p1);
    let e1 = _mm256_unpackhi_epi16(t1, p1);

    let d1 = _mm256_castsi256_si128(d1);
    let e1 = _mm256_castsi256_si128(e1);
    let d0 = _mm256_insertf128_si256(d0,d1,0x1);
    let e0 = _mm256_insertf128_si256(e0,e1,0x1);

    (d0,e0)

}

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave_single<const PARTIAL: bool>(
    a: *const TA, lda: usize, kl: usize,
) -> __m256i {

    let (a0, b0, c0, d0) = if PARTIAL {
        let mut a0 = if kl > 0 { load_v::<8>(a) } else { _mm256_setzero_si256() };
        let mut b0 = if kl > 1 { load_v::<8>(a.add(lda)) } else { _mm256_setzero_si256() };
        let mut c0 = if kl > 2 { load_v::<8>(a.add(lda*2)) } else { _mm256_setzero_si256() };
        let mut d0 = if kl > 3 { load_v::<8>(a.add(lda*3)) } else { _mm256_setzero_si256() };
        (a0, b0, c0, d0)
    } else {
        (
            load_v::<8>(a),
            load_v::<8>(a.add(lda)),
            load_v::<8>(a.add(lda*2)),
            load_v::<8>(a.add(lda*3))
        )
    };

    let t0 = _mm256_unpacklo_epi8(a0, b0);
    
    let p0 = _mm256_unpacklo_epi8(c0, d0);

    let d0 = _mm256_unpacklo_epi16(t0, p0);
    let d1 = _mm256_unpackhi_epi16(t0, p0);

    let d1 = _mm256_castsi256_si128(d1);
    let d0 = _mm256_insertf128_si256(d0,d1,0x1);

    d0
}

#[target_feature(enable = "avx,fma,avx2")]
unsafe fn fmadd(a: __m256i, b: __m256i, c: __m256i, one_v: __m256i) -> __m256i {
    let mut x = _mm256_maddubs_epi16(b, a);
    x = _mm256_madd_epi16(x, one_v);
    _mm256_add_epi32(x, c)
}

#[target_feature(enable = "avx,fma,avx2")]
unsafe fn beta_store<const BETA: usize>(
    dst: *mut TC,
    acc_v: __m256i, beta_v: __m256,
) {
    if BETA == 0 {
        _mm256_storeu_si256(dst as *mut __m256i, acc_v);
    } else if BETA == 1 {
        let acc_v = _mm256_add_epi32(acc_v, _mm256_loadu_si256(dst as *const __m256i));
        _mm256_storeu_si256(dst as *mut __m256i, acc_v);
    } else {
        let dst_v = _mm256_cvtepi32_ps(_mm256_loadu_si256(dst as *const __m256i));
        let acc_v = _mm256_cvtepi32_ps(acc_v);
        let acc_v = _mm256_fmadd_ps(dst_v, beta_v, acc_v);
        _mm256_storeu_si256(dst as *mut __m256i, _mm256_cvtps_epi32(acc_v));
    }
}

#[target_feature(enable = "avx,fma,avx2")]
unsafe fn load_b(
    b: *const TB
) -> __m256i {
    _mm256_castps_si256(_mm256_broadcast_ss(&*(b as *const f32)))
}

#[target_feature(enable = "avx,fma,avx2")]
pub(crate) unsafe fn axpy_v_inner<const BETA: usize>(
    m_lane4: usize, m_lane: usize, m: usize,
    a: *const TA, lda: usize,
    y: *mut TC,
    x_cur: *const TB,
    beta: f32,
    alpha: *const f32,
    incx: usize,
) {
    let beta_v = _mm256_broadcast_ss(&beta);
    let one_v = _mm256_set1_epi16(1_i16);
    const K_UNROLL: usize = 4;
    let mut xt_arr = [0_u8; K_UNROLL*4];
    let is_alpha_one = *alpha == 1.0;
    for i in 0..K_UNROLL {
        if is_alpha_one {
            xt_arr[4*i] = *x_cur.add(i*incx*4);
            xt_arr[4*i+1] = *x_cur.add((i*4+1)*incx);
            xt_arr[4*i+2] = *x_cur.add((i*4+2)*incx);
            xt_arr[4*i+3] = *x_cur.add((i*4+3)*incx);
        } else {
            xt_arr[4*i] = (*x_cur.add(i*incx*4) as f32 * *alpha) as u8;
            xt_arr[4*i+1] = (*x_cur.add((i*4+1)*incx) as f32 * *alpha) as u8;
            xt_arr[4*i+2] = (*x_cur.add((i*4+2)*incx) as f32 * *alpha) as u8;
            xt_arr[4*i+3] = (*x_cur.add((i*4+3)*incx) as f32 * *alpha) as u8;
        }
    }
    let xt0 = load_b(xt_arr.as_ptr());
    let xt1 = load_b(xt_arr.as_ptr().add(4));
    let xt2 = load_b(xt_arr.as_ptr().add(8));
    let xt3 = load_b(xt_arr.as_ptr().add(12));
    
    let x = xt_arr.as_ptr();

    let mut a = a;
    let mut y = y;
    let mut mi = 0usize;
    while mi < m_lane4 {
        seq!(i in 0..2 {
            let (a0, b0) = interleave::<false>(a.add(VS*i*2), lda, 4);
            let (q0, p0) = interleave::<false>(a.add(lda*4+VS*i*2), lda, 4);
            let (q10, p10) = interleave::<false>(a.add(lda*8+VS*i*2), lda, 4);
            let (q20, p20) = interleave::<false>(a.add(lda*12+VS*i*2), lda, 4);
            let c~i = {
                let mut acc~i = _mm256_maddubs_epi16(xt0, a0);
                acc~i = _mm256_madd_epi16(acc~i, one_v);
                acc~i = fmadd(q0, xt1, acc~i, one_v);
                acc~i = fmadd(q10, xt2, acc~i, one_v);
                acc~i = fmadd(q20, xt3, acc~i, one_v);
                acc~i
            };
            beta_store::<BETA>(y.add(VS*i*2), c~i, beta_v);

            let c~i = {
                let mut acc~i = _mm256_maddubs_epi16(xt0, b0);
                acc~i = _mm256_madd_epi16(acc~i, one_v);
                acc~i = fmadd(p0, xt1, acc~i, one_v);
                acc~i = fmadd(p10, xt2, acc~i, one_v);
                acc~i = fmadd(p20, xt3, acc~i, one_v);
                acc~i
            };
            beta_store::<BETA>(y.add(VS*i*2+VS), c~i, beta_v);
        });
        a = a.add(VS*4);
        y = y.add(VS*4);
        mi += VS*4;
    }

    while mi < m_lane / (VS*2) * (VS*2) {
        let (a0, b0) = interleave::<false>(a, lda, 4);
        let (q0, p0) = interleave::<false>(a.add(lda*4), lda, 4);
        let (q10, p10) = interleave::<false>(a.add(lda*8), lda, 4);
        let (q20, p20) = interleave::<false>(a.add(lda*12), lda, 4);
        let c0 = {
            let mut acc0 = _mm256_maddubs_epi16(xt0, a0);
            acc0 = _mm256_madd_epi16(acc0, one_v);
            acc0 = fmadd(q0, xt1, acc0, one_v);
            acc0 = fmadd(q10, xt2, acc0, one_v);
            acc0 = fmadd(q20, xt3, acc0, one_v);
            acc0
        };
        beta_store::<BETA>(y, c0, beta_v);

        let c0 = {
            let mut acc0 =_mm256_maddubs_epi16(xt0, b0);
            acc0 = _mm256_madd_epi16(acc0, one_v);
            acc0 = fmadd(p0, xt1, acc0, one_v);
            acc0 = fmadd(p10, xt2, acc0, one_v);
            acc0 = fmadd(p20, xt3, acc0, one_v);
            acc0
        };
        beta_store::<BETA>(y.add(VS), c0, beta_v);

        a = a.add(VS*2);
        y = y.add(VS*2);
        mi += VS*2;
    }
    while mi < m_lane {
        let a0 = interleave_single::<false>(a, lda, 4);
        let q0 = interleave_single::<false>(a.add(lda*4), lda, 4);
        let q10 = interleave_single::<false>(a.add(lda*8), lda, 4);
        let q20 = interleave_single::<false>(a.add(lda*12), lda, 4);
        let c0 = {
            let mut acc0 = _mm256_maddubs_epi16(xt0, a0);
            acc0 = _mm256_madd_epi16(acc0, one_v);
            acc0 = fmadd(q0, xt1, acc0, one_v);
            acc0 = fmadd(q10, xt2, acc0, one_v);
            acc0 = fmadd(q20, xt3, acc0, one_v);
            acc0
        };
        beta_store::<BETA>(y, c0, beta_v);

        a = a.add(VS);
        y = y.add(VS);
        mi += VS;
    }
    while mi < m {
        if BETA == 1 {
            *y = (*a as i32) * (*x as i32) + *y;
        } else if BETA == 0 {
            *y = (*a as i32) * (*x as i32);
        } else {
            *y = ((*a as f32) * (*x as f32) + *y as f32 * beta) as i32;
        }
        seq!(i in 1..16 {
            *y = (*a.add(lda*i) as i32) * (*x.add(i) as i32) + *y;
        });
        a = a.add(1);
        y = y.add(1);
        mi += 1;
    }
}

#[target_feature(enable = "avx,fma,avx2")]
pub(crate) unsafe fn axpy_v_inner2<const BETA: usize, const PARTIAL: bool>(
    m_lane4: usize, m_lane: usize, m: usize,
    a: *const TA, lda: usize,
    y: *mut TC,
    x_cur: *const TB,
    beta: f32,
    alpha: *const f32,
    incx: usize,
    kl: usize,
) {
    let beta_v = _mm256_broadcast_ss(&beta);
    const K_UNROLL: usize = 1;
    let mut xt_arr = [0_u8; K_UNROLL*4];
    let is_alpha_one = *alpha == 1.0;
    for i in 0..K_UNROLL {
        if is_alpha_one {
            xt_arr[2*i] = *x_cur.add(i*incx*2);
            xt_arr[2*i+1] = *x_cur.add((i*2+1)*incx);
        } else {
            xt_arr[2*i] = (*x_cur.add(i*incx*2) as f32 * *alpha) as u8;
            xt_arr[2*i+1] = (*x_cur.add((i*2+1)*incx) as f32 * *alpha) as u8;
        }
    }
    let xt0 = load_b(xt_arr.as_ptr());
    let x = xt_arr.as_ptr();
    let mut a = a;
    let mut y = y;
    let mut mi = 0usize;
    while mi < m_lane4 {
        seq!(i in 0..2 {
            let (a0, b0) = interleave::<PARTIAL>(a.add(VS*i*2), lda, kl);
            let c~i = _mm256_madd_epi16(a0, xt0);
            beta_store::<BETA>(y.add(VS*i*2), c~i, beta_v);

            let c~i = _mm256_madd_epi16(b0, xt0);
            beta_store::<BETA>(y.add(VS*i*2+VS), c~i, beta_v);
        });
        a = a.add(VS*4);
        y = y.add(VS*4);
        mi += VS*4;
    }

    while mi < m_lane / (VS*2) * (VS*2) {
        let (a0, b0) = interleave::<PARTIAL>(a, lda, kl);
        let c0 = _mm256_madd_epi16(a0, xt0);
        beta_store::<BETA>(y, c0, beta_v);

        let c0 = _mm256_madd_epi16(b0, xt0);
        beta_store::<BETA>(y.add(VS), c0, beta_v);
        a = a.add(VS*2);
        y = y.add(VS*2);
        mi += VS*2;
    }
    while mi < m_lane {
        let a0 = interleave_single::<PARTIAL>(a, lda, kl);
        let c0 = _mm256_madd_epi16(a0, xt0);
        beta_store::<BETA>(y, c0, beta_v);

        a = a.add(VS);
        y = y.add(VS);
        mi += VS;
    }
    while mi < m {
        if BETA == 1 {
            *y = (*a as i32) * (*x as i32) + *y;
        } else if BETA == 0 {
            *y = (*a as i32) * (*x as i32);
        } else {
            *y = ((*a as f32) * (*x as f32) + *y as f32 * beta) as i32;
        }
        for i in 1..kl {
            *y = (*a.add(lda*i) as i32) * (*x.add(i) as i32) + *y;
        }
        a = a.add(1);
        y = y.add(1);
        mi += 1;
    }
}

// The inner should traver along m dimenson for better hw prefetching since they are contiguous in memory
// inner loop should work multiple k to utilize the registers while keeping hw prefetching happy, so tune unrolling param
#[target_feature(enable = "avx,fma,avx2")]
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
    let n_lane = n / (K_UNROLL*4) * (K_UNROLL*4);
   let m_lane4 = m / (VS*MR) * VS*MR;
   let m_lane = m / VS * VS;

    let mut ni = 0;
    let mut a_cur = a;
    let mut x_cur = x;
    let mut inner_v1 = if beta == 1.0 {
        axpy_v_inner::<1>
    } else if beta == 0.0 {
        axpy_v_inner::<0>
    } else {
        axpy_v_inner::<2>
    };
    let mut inner_v2 = if beta == 1.0 {
        axpy_v_inner2::<1,false>
    } else if beta == 0.0 {
        axpy_v_inner2::<0,false>
    } else {
        axpy_v_inner2::<2,false>
    };
    let mut inner_v3 = if beta == 1.0 {
        axpy_v_inner2::<1,true>
    } else if beta == 0.0 {
        axpy_v_inner2::<0,true>
    } else {
        axpy_v_inner2::<2,true>
    };
   while ni < n_lane {
        inner_v1(m_lane4, m_lane, m, a_cur, lda, y, x_cur, beta, alpha, incx);
        a_cur = a_cur.add(lda*K_UNROLL*4);
        x_cur = x_cur.add(incx*K_UNROLL*4);
       beta = 1.0;
       inner_v1 = axpy_v_inner::<1>;
       inner_v2 = axpy_v_inner2::<1,false>;
       inner_v3 = axpy_v_inner2::<1,true>;
       ni += K_UNROLL*4;
   }

   while ni < (n / 4) * 4 {
        inner_v2(m_lane4, m_lane, m, a_cur, lda, y, x_cur, beta, alpha, incx, 4);
        a_cur = a_cur.add(lda*4);
        x_cur = x_cur.add(incx*4);
        beta = 1.0;
        inner_v2 = axpy_v_inner2::<1,false>;
        inner_v3 = axpy_v_inner2::<1,true>;
        ni += 4;
    }
    if n % 4 != 0 {
        inner_v3(m_lane4, m_lane, m, a_cur, lda, y, x_cur, beta, alpha, incx, n % 4);
    }
}