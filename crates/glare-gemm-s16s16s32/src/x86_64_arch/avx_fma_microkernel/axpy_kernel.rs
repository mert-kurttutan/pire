use std::arch::x86_64::*;

use crate::{TA,TB,TC};

use super::super::VS;


// TODO: optimize axpy for m=1 case,
// for each loop we use, less than optimal number of registers, less than 16
// modify so that we use 16 registers for each loop step

#[target_feature(enable = "avx,fma,avx2")]
pub(crate) unsafe fn interleave(
    a: *const TA, lda: usize,
) -> (__m256i, __m256i) {
    let a0 = _mm256_loadu_si256(a as *const __m256i);
    let b0 = _mm256_loadu_si256(a.add(lda) as *const __m256i);
    let t0 = _mm256_unpacklo_epi16(a0, b0);
    let t1 = _mm256_unpackhi_epi16(a0, b0);
    let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
    let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
    (a0, b0)
}

#[target_feature(enable = "avx,fma,avx2")]
pub(crate) unsafe fn interleave2(
    a: *const TA,
) -> (__m256i, __m256i) {
    let a0 = _mm256_loadu_si256(a as *const __m256i);
    let b0 = _mm256_setzero_si256();
    let t0 = _mm256_unpacklo_epi16(a0, b0);
    let t1 = _mm256_unpackhi_epi16(a0, b0);
    let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
    let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
    (a0, b0)
}

#[target_feature(enable = "avx,fma,avx2")]
pub(crate) unsafe fn interleave_single(
    a: *const TA, lda: usize,
) -> __m256i {
    let a0 = _mm256_castsi128_si256(_mm_loadu_si128(a as *const __m128i));
    let b0 = _mm256_castsi128_si256(_mm_loadu_si128(a.add(lda) as *const __m128i));
    let t0 = _mm256_unpacklo_epi16(a0, b0);
    let t1 = _mm256_unpackhi_epi16(a0, b0);
    let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
    a0
}

#[target_feature(enable = "avx,fma,avx2")]
pub(crate) unsafe fn interleave_single2(
    a: *const TA,
) -> __m256i {
    let a0 = _mm256_castsi128_si256(_mm_loadu_si128(a as *const __m128i));
    let b0 = _mm256_setzero_si256();
    let t0 = _mm256_unpacklo_epi16(a0, b0);
    let t1 = _mm256_unpackhi_epi16(a0, b0);
    let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
    // let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
    a0
}


#[target_feature(enable = "avx,fma,avx2")]
pub(crate) unsafe fn axpy_v_inner<const BETA: usize>(
    m_lane4: usize, m_lane: usize, m: usize,
    a: *const TA, lda: usize,
    y: *mut TC,
    xt0: __m256i, xt1: __m256i, 
    xt2: __m256i, xt3: __m256i,
    x: *const TB,
    beta_v: __m256,
    beta: f32,
) {
    let mut a = a;
    let mut y = y;
    let mut mi = 0usize;
    while mi < m_lane4 {
        seq!(i in 0..2 {
            let (a0, b0) = interleave(a.add(VS*i*2), lda);
            let (q0, p0) = interleave(a.add(lda*2+VS*i*2), lda);
            let (q10, p10) = interleave(a.add(lda*4+VS*i*2), lda);
            let (q20, p20) = interleave(a.add(lda*6+VS*i*2), lda);
            let c~i = {
                let mut acc~i = _mm256_madd_epi16(a0, xt0);
                acc~i = _mm256_add_epi32(acc~i, _mm256_madd_epi16(q0, xt1));
                acc~i = _mm256_add_epi32(acc~i, _mm256_madd_epi16(q10, xt2));
                acc~i = _mm256_add_epi32(acc~i, _mm256_madd_epi16(q20, xt3));
                acc~i
            };
            if BETA == 0 {
                _mm256_storeu_si256(y.add(VS*i*2) as *mut __m256i, c~i);
            } else if BETA == 1 {
                let c~i = _mm256_add_epi32(c~i, _mm256_loadu_si256(y.add(VS*i*2) as *const __m256i));
                _mm256_storeu_si256(y.add(VS*i*2) as *mut __m256i, c~i);
            } else {
                let cx~i = _mm256_cvtepi32_ps(_mm256_loadu_si256(y.add(VS*i*2) as *const __m256i));
                let c~i = _mm256_cvtepi32_ps(c~i);
                let c~i = _mm256_fmadd_ps(cx~i, beta_v, c~i);
                _mm256_storeu_si256(y.add(VS*i*2) as *mut __m256i, _mm256_cvtps_epi32(c~i));
            }

            let c~i = {
                let mut acc~i = _mm256_madd_epi16(b0, xt0);
                acc~i = _mm256_add_epi32(acc~i, _mm256_madd_epi16(p0, xt1));
                acc~i = _mm256_add_epi32(acc~i, _mm256_madd_epi16(p10, xt2));
                acc~i = _mm256_add_epi32(acc~i, _mm256_madd_epi16(p20, xt3));
                acc~i
            };
            if BETA == 0 {
                _mm256_storeu_si256(y.add(VS*i*2+VS) as *mut __m256i, c~i);
            } else if BETA == 1 {
                let c~i = _mm256_add_epi32(c~i, _mm256_loadu_si256(y.add(VS*i*2+VS) as *const __m256i));
                _mm256_storeu_si256(y.add(VS*i*2+VS) as *mut __m256i, c~i);
            } else {
                let cx~i = _mm256_cvtepi32_ps(_mm256_loadu_si256(y.add(VS*i*2+VS) as *const __m256i));
                let c~i = _mm256_cvtepi32_ps(c~i);
                let c~i = _mm256_fmadd_ps(cx~i, beta_v, c~i);
                _mm256_storeu_si256(y.add(VS*i*2+VS) as *mut __m256i, _mm256_cvtps_epi32(c~i));
            }
        });
        a = a.add(VS*4);
        y = y.add(VS*4);
        mi += VS*4;
    }

    while mi < m_lane / (VS*2) * (VS*2) {
        let (a0, b0) = interleave(a, lda);
        let (q0, p0) = interleave(a.add(lda*2), lda);
        let (q10, p10) = interleave(a.add(lda*4), lda);
        let (q20, p20) = interleave(a.add(lda*6), lda);
        let c0 = {
            let mut acc0 = _mm256_madd_epi16(a0, xt0);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q0, xt1));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q10, xt2));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q20, xt3));
            acc0
        };
        if BETA == 0 {
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else if BETA == 1 {
            let c0 = _mm256_add_epi32(c0, _mm256_loadu_si256(y as *const __m256i));
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else {
            let cx0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(y as *const __m256i));
            let c0 = _mm256_cvtepi32_ps(c0);
            let c0 = _mm256_fmadd_ps(cx0, beta_v, c0);
            _mm256_storeu_si256(y as *mut __m256i, _mm256_cvtps_epi32(c0));
        }

        let c0 = {
            let mut acc0 = _mm256_madd_epi16(b0, xt0);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, xt1));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p10, xt2));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p20, xt3));
            acc0
        };
        if BETA == 0 {
            _mm256_storeu_si256(y.add(VS) as *mut __m256i, c0);
        } else if BETA == 1 {
            let c0 = _mm256_add_epi32(c0, _mm256_loadu_si256(y.add(VS) as *const __m256i));
            _mm256_storeu_si256(y.add(VS) as *mut __m256i, c0);
        } else {
            let cx0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(y.add(VS) as *const __m256i));
            let c0 = _mm256_cvtepi32_ps(c0);
            let c0 = _mm256_fmadd_ps(cx0, beta_v, c0);
            _mm256_storeu_si256(y.add(VS) as *mut __m256i, _mm256_cvtps_epi32(c0));
        }
        a = a.add(VS*2);
        y = y.add(VS*2);
        mi += VS*2;
    }
    while mi < m_lane {
        let a0 = interleave_single(a, lda);
        let q0 = interleave_single(a.add(lda*2), lda);
        let q10 = interleave_single(a.add(lda*4), lda);
        let q20 = interleave_single(a.add(lda*6), lda);
        let c0 = {
            let mut acc0 = _mm256_madd_epi16(a0, xt0);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q0, xt1));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q10, xt2));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q20, xt3));
            acc0
        };
        if BETA == 0 {
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else if BETA == 1 {
            let c0 = _mm256_add_epi32(c0, _mm256_loadu_si256(y as *const __m256i));
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else {
            let cx0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(y as *const __m256i));
            let c0 = _mm256_cvtepi32_ps(c0);
            let c0 = _mm256_fmadd_ps(cx0, beta_v, c0);
            _mm256_storeu_si256(y as *mut __m256i, _mm256_cvtps_epi32(c0));
        }

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
        *y = (*a.add(lda*2) as i32) * (*x.add(2) as i32) + *y;
        *y = (*a.add(lda*3) as i32) * (*x.add(3) as i32) + *y;
        *y = (*a.add(lda*4) as i32) * (*x.add(4) as i32) + *y;
        *y = (*a.add(lda*5) as i32) * (*x.add(5) as i32) + *y;
        *y = (*a.add(lda*6) as i32) * (*x.add(6) as i32) + *y;
        *y = (*a.add(lda*7) as i32) * (*x.add(7) as i32) + *y;
        a = a.add(1);
        y = y.add(1);
        mi += 1;
    }
}

#[target_feature(enable = "avx,fma,avx2")]
pub(crate) unsafe fn axpy_v_inner2<const BETA: usize>(
    m_lane4: usize, m_lane: usize, m: usize,
    a: *const TA, lda: usize,
    y: *mut TC,
    xt0: __m256i,
    x: *const TB,
    beta_v: __m256,
    beta: f32,
) {
    let mut a = a;
    let mut y = y;
    let mut mi = 0usize;
    while mi < m_lane4 {
        seq!(i in 0..2 {
            let (a0, b0) = interleave(a.add(VS*i*2), lda);
            let c~i = {
                let acc~i = _mm256_madd_epi16(a0, xt0);
                acc~i
            };
            if BETA == 0 {
                _mm256_storeu_si256(y.add(VS*i*2) as *mut __m256i, c~i);
            } else if BETA == 1 {
                let c~i = _mm256_add_epi32(c~i, _mm256_loadu_si256(y.add(VS*i*2) as *const __m256i));
                _mm256_storeu_si256(y.add(VS*i*2) as *mut __m256i, c~i);
            } else {
                let cx~i = _mm256_cvtepi32_ps(_mm256_loadu_si256(y.add(VS*i*2) as *const __m256i));
                let c~i = _mm256_cvtepi32_ps(c~i);
                let c~i = _mm256_fmadd_ps(cx~i, beta_v, c~i);
                _mm256_storeu_si256(y.add(VS*i*2) as *mut __m256i, _mm256_cvtps_epi32(c~i));
            }

            let c~i = {
                let acc~i = _mm256_madd_epi16(b0, xt0);
                acc~i
            };
            if BETA == 0 {
                _mm256_storeu_si256(y.add(VS*i*2+VS) as *mut __m256i, c~i);
            } else if BETA == 1 {
                let c~i = _mm256_add_epi32(c~i, _mm256_loadu_si256(y.add(VS*i*2+VS) as *const __m256i));
                _mm256_storeu_si256(y.add(VS*i*2+VS) as *mut __m256i, c~i);
            } else {
                let cx~i = _mm256_cvtepi32_ps(_mm256_loadu_si256(y.add(VS*i*2+VS) as *const __m256i));
                let c~i = _mm256_cvtepi32_ps(c~i);
                let c~i = _mm256_fmadd_ps(cx~i, beta_v, c~i);
                _mm256_storeu_si256(y.add(VS*i*2+VS) as *mut __m256i, _mm256_cvtps_epi32(c~i));
            }
        });
        a = a.add(VS*4);
        y = y.add(VS*4);
        mi += VS*4;
    }

    while mi < m_lane / (VS*2) * (VS*2) {
        let (a0, b0) = interleave(a, lda);
        let c0 = {
            let acc0 = _mm256_madd_epi16(a0, xt0);
            acc0
        };
        if BETA == 0 {
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else if BETA == 1 {
            let c0 = _mm256_add_epi32(c0, _mm256_loadu_si256(y as *const __m256i));
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else {
            let cx0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(y as *const __m256i));
            let c0 = _mm256_cvtepi32_ps(c0);
            let c0 = _mm256_fmadd_ps(cx0, beta_v, c0);
            _mm256_storeu_si256(y as *mut __m256i, _mm256_cvtps_epi32(c0));
        }

        let c0 = {
            let acc0 = _mm256_madd_epi16(b0, xt0);
            acc0
        };
        if BETA == 0 {
            _mm256_storeu_si256(y.add(VS) as *mut __m256i, c0);
        } else if BETA == 1 {
            let c0 = _mm256_add_epi32(c0, _mm256_loadu_si256(y.add(VS) as *const __m256i));
            _mm256_storeu_si256(y.add(VS) as *mut __m256i, c0);
        } else {
            let cx0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(y.add(VS) as *const __m256i));
            let c0 = _mm256_cvtepi32_ps(c0);
            let c0 = _mm256_fmadd_ps(cx0, beta_v, c0);
            _mm256_storeu_si256(y.add(VS) as *mut __m256i, _mm256_cvtps_epi32(c0));
        }
        a = a.add(VS*2);
        y = y.add(VS*2);
        mi += VS*2;
    }
    while mi < m_lane {
        let a0 = interleave_single(a, lda);
        let c0 = {
            let acc0 = _mm256_madd_epi16(a0, xt0);
            acc0
        };
        if BETA == 0 {
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else if BETA == 1 {
            let c0 = _mm256_add_epi32(c0, _mm256_loadu_si256(y as *const __m256i));
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else {
            let cx0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(y as *const __m256i));
            let c0 = _mm256_cvtepi32_ps(c0);
            let c0 = _mm256_fmadd_ps(cx0, beta_v, c0);
            _mm256_storeu_si256(y as *mut __m256i, _mm256_cvtps_epi32(c0));
        }

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
        a = a.add(1);
        y = y.add(1);
        mi += 1;
    }
}



#[target_feature(enable = "avx,fma,avx2")]
pub(crate) unsafe fn axpy_v_inner3<const BETA: usize>(
    m_lane4: usize, m_lane: usize, m: usize,
    a: *const TA,
    y: *mut TC,
    xt0: __m256i,
    x: *const TB,
    beta_v: __m256,
    beta: f32,
) {
    let mut a = a;
    let mut y = y;
    let mut mi = 0usize;
    while mi < m_lane4 {
        seq!(i in 0..2 {
            let (a0, b0) = interleave2(a.add(VS*i*2));
            let c~i = {
                let acc~i = _mm256_madd_epi16(a0, xt0);
                acc~i
            };
            if BETA == 0 {
                _mm256_storeu_si256(y.add(VS*i*2) as *mut __m256i, c~i);
            } else if BETA == 1 {
                let c~i = _mm256_add_epi32(c~i, _mm256_loadu_si256(y.add(VS*i*2) as *const __m256i));
                _mm256_storeu_si256(y.add(VS*i*2) as *mut __m256i, c~i);
            } else {
                let cx~i = _mm256_cvtepi32_ps(_mm256_loadu_si256(y.add(VS*i*2) as *const __m256i));
                let c~i = _mm256_cvtepi32_ps(c~i);
                let c~i = _mm256_fmadd_ps(cx~i, beta_v, c~i);
                _mm256_storeu_si256(y.add(VS*i*2) as *mut __m256i, _mm256_cvtps_epi32(c~i));
            }

            let c~i = {
                let acc~i = _mm256_madd_epi16(b0, xt0);
                acc~i
            };
            if BETA == 0 {
                _mm256_storeu_si256(y.add(VS*i*2+VS) as *mut __m256i, c~i);
            } else if BETA == 1 {
                let c~i = _mm256_add_epi32(c~i, _mm256_loadu_si256(y.add(VS*i*2+VS) as *const __m256i));
                _mm256_storeu_si256(y.add(VS*i*2+VS) as *mut __m256i, c~i);
            } else {
                let cx~i = _mm256_cvtepi32_ps(_mm256_loadu_si256(y.add(VS*i*2+VS) as *const __m256i));
                let c~i = _mm256_cvtepi32_ps(c~i);
                let c~i = _mm256_fmadd_ps(cx~i, beta_v, c~i);
                _mm256_storeu_si256(y.add(VS*i*2+VS) as *mut __m256i, _mm256_cvtps_epi32(c~i));
            }
        });
        a = a.add(VS*4);
        y = y.add(VS*4);
        mi += VS*4;
    }

    while mi < m_lane / (VS*2) * (VS*2) {
        let (a0, b0) = interleave2(a);
        let c0 = {
            let acc0 = _mm256_madd_epi16(a0, xt0);
            acc0
        };
        if BETA == 0 {
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else if BETA == 1 {
            let c0 = _mm256_add_epi32(c0, _mm256_loadu_si256(y as *const __m256i));
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else {
            let cx0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(y as *const __m256i));
            let c0 = _mm256_cvtepi32_ps(c0);
            let c0 = _mm256_fmadd_ps(cx0, beta_v, c0);
            _mm256_storeu_si256(y as *mut __m256i, _mm256_cvtps_epi32(c0));
        }

        let c0 = {
            let acc0 = _mm256_madd_epi16(b0, xt0);
            acc0
        };
        if BETA == 0 {
            _mm256_storeu_si256(y.add(VS) as *mut __m256i, c0);
        } else if BETA == 1 {
            let c0 = _mm256_add_epi32(c0, _mm256_loadu_si256(y.add(VS) as *const __m256i));
            _mm256_storeu_si256(y.add(VS) as *mut __m256i, c0);
        } else {
            let cx0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(y.add(VS) as *const __m256i));
            let c0 = _mm256_cvtepi32_ps(c0);
            let c0 = _mm256_fmadd_ps(cx0, beta_v, c0);
            _mm256_storeu_si256(y.add(VS) as *mut __m256i, _mm256_cvtps_epi32(c0));
        }
        a = a.add(VS*2);
        y = y.add(VS*2);
        mi += VS*2;
    }
    while mi < m_lane {
        let a0 = interleave_single2(a);
        let c0 = {
            let acc0 = _mm256_madd_epi16(a0, xt0);
            acc0
        };
        if BETA == 0 {
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else if BETA == 1 {
            let c0 = _mm256_add_epi32(c0, _mm256_loadu_si256(y as *const __m256i));
            _mm256_storeu_si256(y as *mut __m256i, c0);
        } else {
            let cx0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(y as *const __m256i));
            let c0 = _mm256_cvtepi32_ps(c0);
            let c0 = _mm256_fmadd_ps(cx0, beta_v, c0);
            _mm256_storeu_si256(y as *mut __m256i, _mm256_cvtps_epi32(c0));
        }

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
   let beta_v = _mm256_broadcast_ss(&beta);
    let n_lane = n / (K_UNROLL*2) * (K_UNROLL*2);
   let m_lane4 = m / (VS*MR) * VS*MR;
   let m_lane = m / VS * VS;

    let mut ni = 0;
    let mut a_cur = a;
    let mut x_cur = x;
    let mut xtv_arr = [_mm256_setzero_si256(); K_UNROLL];
    let is_alpha_one = *alpha == 1.0;
   while ni < n_lane {
       let mut xt_arr = [0_i16; K_UNROLL*2];
       if is_alpha_one {
           for i in 0..K_UNROLL {
               xt_arr[2*i] = *x_cur.add(i*incx*2);
                xt_arr[2*i+1] = *x_cur.add((i*2+1)*incx);
               xtv_arr[i] = _mm256_castps_si256(_mm256_broadcast_ss(&*(xt_arr.as_ptr().add(2*i) as *const f32)));
           }
        } else {
            for i in 0..K_UNROLL {
                xt_arr[2*i] = (*x_cur.add(i*incx*2) as f32 * *alpha) as i16;
                xt_arr[2*i+1] = (*x_cur.add((i*2+1)*incx) as f32 * *alpha) as i16;
                xtv_arr[i] = _mm256_castps_si256(_mm256_broadcast_ss(&*(xt_arr.as_ptr().add(2*i) as *const f32)));
            }
        }

        let xt = xt_arr.as_ptr();

        if beta == 1.0 {
            axpy_v_inner::<1>(m_lane4, m_lane, m, a_cur, lda, y, xtv_arr[0], xtv_arr[1], xtv_arr[2], xtv_arr[3], xt, beta_v, beta);
        } else if beta == 0.0 {
            axpy_v_inner::<0>(m_lane4, m_lane, m, a_cur, lda, y, xtv_arr[0], xtv_arr[1], xtv_arr[2], xtv_arr[3], xt, beta_v, beta);
        } else {
            axpy_v_inner::<2>(m_lane4, m_lane, m, a_cur, lda, y, xtv_arr[0], xtv_arr[1], xtv_arr[2], xtv_arr[3], xt, beta_v, beta);
        }
        a_cur = a_cur.add(lda*K_UNROLL*2);
        x_cur = x_cur.add(incx*K_UNROLL*2);
       beta = 1.0;
       ni += K_UNROLL*2;
   }

   while ni < (n / 2) *2 {
        let mut xt_arr = [0_i16; 2];
        if is_alpha_one {
            xt_arr[0] = *x_cur;
            xt_arr[1] = *x_cur.add(incx);
            xtv_arr[0] = _mm256_castps_si256(_mm256_broadcast_ss(&*(xt_arr.as_ptr() as *const f32)));
        } else {
            xt_arr[0] = (*x_cur as f32 * *alpha) as i16;
            xt_arr[1] = (*x_cur.add(incx) as f32 * *alpha) as i16;
            xtv_arr[0] = _mm256_castps_si256(_mm256_broadcast_ss(&*(xt_arr.as_ptr() as *const f32)));
        }

        let xt = xt_arr.as_ptr();

        if beta == 1.0 {
            axpy_v_inner2::<1>(m_lane4, m_lane, m, a_cur, lda, y, xtv_arr[0], xt, beta_v, beta);
        } else if beta == 0.0 {
            axpy_v_inner2::<0>(m_lane4, m_lane, m, a_cur, lda, y, xtv_arr[0], xt, beta_v, beta);
        } else {
            axpy_v_inner2::<2>(m_lane4, m_lane, m, a_cur, lda, y, xtv_arr[0], xt, beta_v, beta);
        }
        a_cur = a_cur.add(lda*2);
        x_cur = x_cur.add(incx*2);
        beta = 1.0;
        ni += 2;
    }
    if n % 2 != 0 {
        let mut xt_arr = [0_i16; 2];
        if is_alpha_one {
            xt_arr[0] = *x_cur;
            xtv_arr[0] = _mm256_castps_si256(_mm256_broadcast_ss(&*(xt_arr.as_ptr() as *const f32)));
        } else {
            xt_arr[0] = (*x_cur as f32 * *alpha) as i16;
            xtv_arr[0] = _mm256_castps_si256(_mm256_broadcast_ss(&*(xt_arr.as_ptr() as *const f32)));
        }

        let xt = xt_arr.as_ptr();

        if beta == 1.0 {
            axpy_v_inner3::<1>(m_lane4, m_lane, m, a_cur, y, xtv_arr[0], xt, beta_v, beta);
        } else if beta == 0.0 {
            axpy_v_inner3::<0>(m_lane4, m_lane, m, a_cur, y, xtv_arr[0], xt, beta_v, beta);
        } else {
            axpy_v_inner3::<2>(m_lane4, m_lane, m, a_cur, y, xtv_arr[0], xt, beta_v, beta);
        }
    }
}

use seq_macro::seq;
