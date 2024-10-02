use std::arch::x86_64::*;

use crate::{TA, TB, TC};
use seq_macro::seq;

use super::super::VS;

// TODO: optimize axpy for m=1 case,
// for each loop we use, less than optimal number of registers, less than 16
// modify so that we use 16 registers for each loop step

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave<const PARTIAL: bool>(a: *const TA, lda: usize) -> (__m256i, __m256i) {
    let a0 = _mm256_loadu_si256(a as *const __m256i);
    let b0 = if PARTIAL { _mm256_setzero_si256() } else { _mm256_loadu_si256(a.add(lda) as *const __m256i) };
    let t0 = _mm256_unpacklo_epi16(a0, b0);
    let t1 = _mm256_unpackhi_epi16(a0, b0);
    let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
    let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
    (a0, b0)
}

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave_single<const PARTIAL: bool>(a: *const TA, lda: usize) -> __m256i {
    let a0 = _mm256_castsi128_si256(_mm_loadu_si128(a as *const __m128i));
    let b0 = if PARTIAL {
        _mm256_setzero_si256()
    } else {
        _mm256_castsi128_si256(_mm_loadu_si128(a.add(lda) as *const __m128i))
    };
    let t0 = _mm256_unpacklo_epi16(a0, b0);
    let t1 = _mm256_unpackhi_epi16(a0, b0);
    let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
    a0
}

#[target_feature(enable = "avx,fma,avx2")]
unsafe fn fmadd(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    _mm256_add_epi32(_mm256_madd_epi16(a, b), c)
}

#[target_feature(enable = "avx,fma,avx2")]
unsafe fn beta_store<const BETA: usize>(dst: *mut TC, acc_v: __m256i, beta_v: __m256) {
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
unsafe fn load_b(b: *const TB) -> __m256i {
    _mm256_castps_si256(_mm256_broadcast_ss(&*(b as *const f32)))
}

#[target_feature(enable = "avx,fma,avx2")]
pub(crate) unsafe fn axpy_v_inner<const BETA: usize>(
    m_lane4: usize,
    m_lane: usize,
    m: usize,
    a: *const TA,
    lda: usize,
    y: *mut TC,
    x_cur: *const TB,
    beta: f32,
    alpha: *const f32,
    incx: usize,
) {
    let beta_v = _mm256_broadcast_ss(&beta);
    const K_UNROLL: usize = 4;
    let mut xt_arr = [0_i16; K_UNROLL * 2];
    let is_alpha_one = *alpha == 1.0;
    for i in 0..K_UNROLL {
        if is_alpha_one {
            xt_arr[2 * i] = *x_cur.add(i * incx * 2);
            xt_arr[2 * i + 1] = *x_cur.add((i * 2 + 1) * incx);
        } else {
            xt_arr[2 * i] = (*x_cur.add(i * incx * 2) as f32 * *alpha) as i16;
            xt_arr[2 * i + 1] = (*x_cur.add((i * 2 + 1) * incx) as f32 * *alpha) as i16;
        }
    }
    let xt0 = load_b(xt_arr.as_ptr());
    let xt1 = load_b(xt_arr.as_ptr().add(2));
    let xt2 = load_b(xt_arr.as_ptr().add(4));
    let xt3 = load_b(xt_arr.as_ptr().add(6));

    let x = xt_arr.as_ptr();

    let mut a = a;
    let mut y = y;
    let mut mi = 0usize;
    while mi < m_lane4 {
        seq!(i in 0..2 {
            let (a0, b0) = interleave::<false>(a.add(VS*i*2), lda);
            let (q0, p0) = interleave::<false>(a.add(lda*2+VS*i*2), lda);
            let (q10, p10) = interleave::<false>(a.add(lda*4+VS*i*2), lda);
            let (q20, p20) = interleave::<false>(a.add(lda*6+VS*i*2), lda);
            let c~i = {
                let mut acc~i = _mm256_madd_epi16(a0, xt0);
                acc~i = fmadd(q0, xt1, acc~i);
                acc~i = fmadd(q10, xt2, acc~i);
                acc~i = fmadd(q20, xt3, acc~i);
                acc~i
            };
            beta_store::<BETA>(y.add(VS*i*2), c~i, beta_v);

            let c~i = {
                let mut acc~i = _mm256_madd_epi16(b0, xt0);
                acc~i = fmadd(p0, xt1, acc~i);
                acc~i = fmadd(p10, xt2, acc~i);
                acc~i = fmadd(p20, xt3, acc~i);
                acc~i
            };
            beta_store::<BETA>(y.add(VS*i*2+VS), c~i, beta_v);
        });
        a = a.add(VS * 4);
        y = y.add(VS * 4);
        mi += VS * 4;
    }

    while mi < m_lane / (VS * 2) * (VS * 2) {
        let (a0, b0) = interleave::<false>(a, lda);
        let (q0, p0) = interleave::<false>(a.add(lda * 2), lda);
        let (q10, p10) = interleave::<false>(a.add(lda * 4), lda);
        let (q20, p20) = interleave::<false>(a.add(lda * 6), lda);
        let c0 = {
            let mut acc0 = _mm256_madd_epi16(a0, xt0);
            acc0 = fmadd(q0, xt1, acc0);
            acc0 = fmadd(q10, xt2, acc0);
            acc0 = fmadd(q20, xt3, acc0);
            acc0
        };
        beta_store::<BETA>(y, c0, beta_v);

        let c0 = {
            let mut acc0 = _mm256_madd_epi16(b0, xt0);
            acc0 = fmadd(p0, xt1, acc0);
            acc0 = fmadd(p10, xt2, acc0);
            acc0 = fmadd(p20, xt3, acc0);
            acc0
        };
        beta_store::<BETA>(y.add(VS), c0, beta_v);

        a = a.add(VS * 2);
        y = y.add(VS * 2);
        mi += VS * 2;
    }
    while mi < m_lane {
        let a0 = interleave_single::<false>(a, lda);
        let q0 = interleave_single::<false>(a.add(lda * 2), lda);
        let q10 = interleave_single::<false>(a.add(lda * 4), lda);
        let q20 = interleave_single::<false>(a.add(lda * 6), lda);
        let c0 = {
            let mut acc0 = _mm256_madd_epi16(a0, xt0);
            acc0 = fmadd(q0, xt1, acc0);
            acc0 = fmadd(q10, xt2, acc0);
            acc0 = fmadd(q20, xt3, acc0);
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
        *y = (*a.add(lda) as i32) * (*x.add(1) as i32) + *y;
        *y = (*a.add(lda * 2) as i32) * (*x.add(2) as i32) + *y;
        *y = (*a.add(lda * 3) as i32) * (*x.add(3) as i32) + *y;
        *y = (*a.add(lda * 4) as i32) * (*x.add(4) as i32) + *y;
        *y = (*a.add(lda * 5) as i32) * (*x.add(5) as i32) + *y;
        *y = (*a.add(lda * 6) as i32) * (*x.add(6) as i32) + *y;
        *y = (*a.add(lda * 7) as i32) * (*x.add(7) as i32) + *y;
        a = a.add(1);
        y = y.add(1);
        mi += 1;
    }
}

#[target_feature(enable = "avx,fma,avx2")]
pub(crate) unsafe fn axpy_v_inner2<const BETA: usize, const PARTIAL: bool>(
    m_lane4: usize,
    m_lane: usize,
    m: usize,
    a: *const TA,
    lda: usize,
    y: *mut TC,
    x_cur: *const TB,
    beta: f32,
    alpha: *const f32,
    incx: usize,
) {
    let beta_v = _mm256_broadcast_ss(&beta);
    const K_UNROLL: usize = 1;
    let mut xt_arr = [0_i16; K_UNROLL * 2];
    let is_alpha_one = *alpha == 1.0;
    for i in 0..K_UNROLL {
        if is_alpha_one {
            xt_arr[2 * i] = *x_cur.add(i * incx * 2);
            xt_arr[2 * i + 1] = *x_cur.add((i * 2 + 1) * incx);
        } else {
            xt_arr[2 * i] = (*x_cur.add(i * incx * 2) as f32 * *alpha) as i16;
            xt_arr[2 * i + 1] = (*x_cur.add((i * 2 + 1) * incx) as f32 * *alpha) as i16;
        }
    }
    let xt0 = load_b(xt_arr.as_ptr());
    let x = xt_arr.as_ptr();
    let mut a = a;
    let mut y = y;
    let mut mi = 0usize;
    while mi < m_lane4 {
        seq!(i in 0..2 {
            let (a0, b0) = interleave::<PARTIAL>(a.add(VS*i*2), lda);
            let c~i = _mm256_madd_epi16(a0, xt0);
            beta_store::<BETA>(y.add(VS*i*2), c~i, beta_v);

            let c~i = _mm256_madd_epi16(b0, xt0);
            beta_store::<BETA>(y.add(VS*i*2+VS), c~i, beta_v);
        });
        a = a.add(VS * 4);
        y = y.add(VS * 4);
        mi += VS * 4;
    }

    while mi < m_lane / (VS * 2) * (VS * 2) {
        let (a0, b0) = interleave::<PARTIAL>(a, lda);
        let c0 = _mm256_madd_epi16(a0, xt0);
        beta_store::<BETA>(y, c0, beta_v);

        let c0 = _mm256_madd_epi16(b0, xt0);
        beta_store::<BETA>(y.add(VS), c0, beta_v);
        a = a.add(VS * 2);
        y = y.add(VS * 2);
        mi += VS * 2;
    }
    while mi < m_lane {
        let a0 = interleave_single::<PARTIAL>(a, lda);
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
        if !PARTIAL {
            *y = (*a.add(lda) as i32) * (*x.add(1) as i32) + *y;
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
    m: usize,
    n: usize,
    alpha: *const f32,
    a: *const TA,
    lda: usize,
    x: *const TB,
    incx: usize,
    beta: *const f32,
    y: *mut TC,
) {
    const K_UNROLL: usize = 4;
    const MR: usize = 4;
    let mut beta = *beta;
    let n_lane = n / (K_UNROLL * 2) * (K_UNROLL * 2);
    let m_lane4 = m / (VS * MR) * VS * MR;
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
        axpy_v_inner2::<1, false>
    } else if beta == 0.0 {
        axpy_v_inner2::<0, false>
    } else {
        axpy_v_inner2::<2, false>
    };
    let mut inner_v3 = if beta == 1.0 {
        axpy_v_inner2::<1, true>
    } else if beta == 0.0 {
        axpy_v_inner2::<0, true>
    } else {
        axpy_v_inner2::<2, true>
    };
    while ni < n_lane {
        inner_v1(m_lane4, m_lane, m, a_cur, lda, y, x_cur, beta, alpha, incx);
        a_cur = a_cur.add(lda * K_UNROLL * 2);
        x_cur = x_cur.add(incx * K_UNROLL * 2);
        beta = 1.0;
        inner_v1 = axpy_v_inner::<1>;
        inner_v2 = axpy_v_inner2::<1, false>;
        inner_v3 = axpy_v_inner2::<1, true>;
        ni += K_UNROLL * 2;
    }

    while ni < (n / 2) * 2 {
        inner_v2(m_lane4, m_lane, m, a_cur, lda, y, x_cur, beta, alpha, incx);
        a_cur = a_cur.add(lda * 2);
        x_cur = x_cur.add(incx * 2);
        beta = 1.0;
        inner_v2 = axpy_v_inner2::<1, false>;
        inner_v3 = axpy_v_inner2::<1, true>;
        ni += 2;
    }
    if n % 2 != 0 {
        inner_v3(m_lane4, m_lane, m, a_cur, lda, y, x_cur, beta, alpha, incx);
    }
}
