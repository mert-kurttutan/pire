use seq_macro::seq;
use std::ptr::copy_nonoverlapping;

use half::f16;


use paste::paste;

use std::arch::x86_64::*;


#[target_feature(enable = "avx")]
pub(crate) unsafe fn storeu_ps<const M: usize>(
    src: __m256, dst: *mut f32
) {
    let mut temp_arr = [0.0; 8];
    _mm256_storeu_ps(temp_arr.as_mut_ptr(), src);
    copy_nonoverlapping(temp_arr.as_ptr(), dst, M);
}


#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn pack_t<const MR: usize>(
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let a = a as *const u16;
    let a0 = _mm256_cvtph_ps(_mm_loadu_si128(a as *const __m128i));
    let a1 = _mm256_cvtph_ps(_mm_loadu_si128(a.add(lda) as *const __m128i));
    let a2 = _mm256_cvtph_ps(_mm_loadu_si128(a.add(lda*2) as *const __m128i));
    let a3 = _mm256_cvtph_ps(_mm_loadu_si128(a.add(lda*3) as *const __m128i));

    // transpose
    let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
    let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
    let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
    let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

    let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
    let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
    let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
    let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));

    let a0 = _mm256_cvtph_ps(_mm_loadu_si128(a.add(lda*4) as *const __m128i));
    let a1 = _mm256_cvtph_ps(_mm_loadu_si128(a.add(lda*5) as *const __m128i));
    let a2 = _mm256_cvtph_ps(_mm_loadu_si128(a.add(lda*6) as *const __m128i));
    let a3 = _mm256_cvtph_ps(_mm_loadu_si128(a.add(lda*7) as *const __m128i));

    // transpose
    let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
    let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
    let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
    let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

    let x4 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
    let x5 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
    let x6 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
    let x7 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));

    // exchange hi of x0 and lo of x4
    let x0_t = _mm256_permute2f128_ps(x0, x4, 0b0010_0000);
    let x4_t = _mm256_permute2f128_ps(x0, x4, 0b0011_0001);
    // exchange hi of x1 and lo of x5
    let x1_t = _mm256_permute2f128_ps(x1, x5, 0b0010_0000);
    let x5_t = _mm256_permute2f128_ps(x1, x5, 0b0011_0001);
    // exchange hi of x2 and lo of x6
    let x2_t = _mm256_permute2f128_ps(x2, x6, 0b0010_0000);
    let x6_t = _mm256_permute2f128_ps(x2, x6, 0b0011_0001);
    // exchange hi of x3 and lo of x7
    let x3_t = _mm256_permute2f128_ps(x3, x7, 0b0010_0000);
    let x7_t = _mm256_permute2f128_ps(x3, x7, 0b0011_0001);

    _mm256_store_ps(ap, x0_t);
    _mm256_store_ps(ap.add(MR), x1_t);
    _mm256_store_ps(ap.add(MR*2), x2_t);
    _mm256_store_ps(ap.add(MR*3), x3_t);
    _mm256_store_ps(ap.add(MR*4), x4_t);
    _mm256_store_ps(ap.add(MR*5), x5_t);
    _mm256_store_ps(ap.add(MR*6), x6_t);
    _mm256_store_ps(ap.add(MR*7), x7_t); 
}


#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn pack_scalar_k(
    m_left: usize, k: usize,
    a: *const f16, a_rs: usize, a_cs: usize,
    ap: *mut f32, vs: usize
) {
    let mr = (m_left + vs - 1) / vs * vs;
    for i in 0..m_left  {
        for j in 0..k {
            *ap.add(j*mr+i) = (*a.add(j*a_cs + i*a_rs)).to_f32();
        }
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_scalar_k_2(
    m_left: usize, k: usize,
    a: *const u16, a_rs: usize, a_cs: usize,
    ap: *mut u16, vs: usize
) {
    let mr = (m_left + vs - 1) / vs * vs;
    for i in 0..m_left  {
        for j in 0..k {
            *ap.add(j*mr+i) = *a.add(j*a_cs + i*a_rs);
        }
    }
}


// #[target_feature(enable = "avx,f16c")]
// pub(crate) unsafe fn copy_packed<const M: usize>(a: *const f16, b: *mut f32) {

//     if M == 24 {
//         let a = a as *const u16;
//         let x = _mm256_cvtph_ps(_mm_loadu_si128(a as *const __m128i));
//         _mm256_storeu_ps(b, x);
//         let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(8) as *const __m128i));
//         _mm256_storeu_ps(b.add(8), x);
//         let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(16) as *const __m128i));
//         _mm256_storeu_ps(b.add(16), x);
//         return;
//     }
//     if M > 16 && M < 24 {
//         let mut at = [0u16; 8];
//         let mut bt = [0f32; 8];
//         let a = a as *const u16;
//         let x = _mm256_cvtph_ps(_mm_loadu_si128(a as *const __m128i));
//         _mm256_storeu_ps(b, x);
//         let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(8) as *const __m128i));
//         _mm256_storeu_ps(b.add(8), x);
//         copy_nonoverlapping(a.add(16), at.as_mut_ptr(), M-16);
//         let x = _mm256_cvtph_ps(_mm_loadu_si128(at.as_ptr() as *const __m128i));
//         _mm256_storeu_ps(bt.as_mut_ptr(), x);
//         copy_nonoverlapping(bt.as_ptr(), b.add(16), M-16);
//         return;
//     }
//     if M == 16 {
//         let a = a as *const u16;
//         let x = _mm256_cvtph_ps(_mm_loadu_si128(a as *const __m128i));
//         _mm256_storeu_ps(b, x);
//         let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(8) as *const __m128i));
//         _mm256_storeu_ps(b.add(8), x);
//         return;
//     }
//     if M > 8 && M < 16 {
//         let mut at = [0u16; 8];
//         let mut bt = [0f32; 8];
//         let a = a as *const u16;
//         let x = _mm256_cvtph_ps(_mm_loadu_si128(a as *const __m128i));
//         _mm256_storeu_ps(b, x);
//         copy_nonoverlapping(a.add(8), at.as_mut_ptr(), M-8);
//         let x = _mm256_cvtph_ps(_mm_loadu_si128(at.as_ptr() as *const __m128i));
//         _mm256_storeu_ps(bt.as_mut_ptr(), x);
//         copy_nonoverlapping(bt.as_ptr(), b.add(8), M-8);
//         return;
//     }
//     let mut at = [0u16; 8];
//     let mut bt = [0f32; 8];
//     copy_nonoverlapping(a as *const u16, at.as_mut_ptr(), M);
//     let x = _mm256_cvtph_ps(_mm_loadu_si128(at.as_ptr() as *const __m128i));
//     _mm256_storeu_ps(bt.as_mut_ptr(), x);
//     copy_nonoverlapping(bt.as_ptr(), b, M);
// }

#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn copy_packed<const M: usize>(a: *const f16, b: *mut f32) {

    if M == 48 {
        let a = a as *const u16;
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a as *const __m128i));
        _mm256_storeu_ps(b, x);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(8) as *const __m128i));
        _mm256_storeu_ps(b.add(8), x);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(16) as *const __m128i));
        _mm256_storeu_ps(b.add(16), x);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(24) as *const __m128i));
        _mm256_storeu_ps(b.add(24), x);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(32) as *const __m128i));
        _mm256_storeu_ps(b.add(32), x);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(40) as *const __m128i));
        _mm256_storeu_ps(b.add(40), x);
        return;
    }
    if M == 24 {
        let a = a as *const u16;
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a as *const __m128i));
        _mm256_storeu_ps(b, x);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(8) as *const __m128i));
        _mm256_storeu_ps(b.add(8), x);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(16) as *const __m128i));
        _mm256_storeu_ps(b.add(16), x);
        return;
    }
    if M >= 40 {
        let mut at = [0u16; 8];
        let mut bt = [0f32; 8];
        let a = a as *const u16;
        seq!(i in 0..5 {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(i*8) as *const __m128i));
            _mm256_storeu_ps(b.add(i*8), x);
        });
        copy_nonoverlapping(a.add(40), at.as_mut_ptr(), M-40);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(at.as_ptr() as *const __m128i));
        _mm256_storeu_ps(bt.as_mut_ptr(), x);
        copy_nonoverlapping(bt.as_ptr(), b.add(40), M-40);
        return;
    }
    if M >= 32 {
        let mut at = [0u16; 8];
        let mut bt = [0f32; 8];
        let a = a as *const u16;
        seq!(i in 0..4 {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(i*8) as *const __m128i));
            _mm256_storeu_ps(b.add(i*8), x);
        });
        copy_nonoverlapping(a.add(32), at.as_mut_ptr(), M-32);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(at.as_ptr() as *const __m128i));
        _mm256_storeu_ps(bt.as_mut_ptr(), x);
        copy_nonoverlapping(bt.as_ptr(), b.add(32), M-32);
        return;
    }
    if M >= 24 {
        let mut at = [0u16; 8];
        let mut bt = [0f32; 8];
        let a = a as *const u16;
        seq!(i in 0..3 {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(i*8) as *const __m128i));
            _mm256_storeu_ps(b.add(i*8), x);
        });
        copy_nonoverlapping(a.add(24), at.as_mut_ptr(), M-24);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(at.as_ptr() as *const __m128i));
        _mm256_storeu_ps(bt.as_mut_ptr(), x);
        copy_nonoverlapping(bt.as_ptr(), b.add(24), M-24);
        return;
    }
    if M >= 16 {
        let mut at = [0u16; 8];
        let mut bt = [0f32; 8];
        let a = a as *const u16;
        seq!(i in 0..2 {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(i*8) as *const __m128i));
            _mm256_storeu_ps(b.add(i*8), x);
        });
        copy_nonoverlapping(a.add(16), at.as_mut_ptr(), M-16);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(at.as_ptr() as *const __m128i));
        _mm256_storeu_ps(bt.as_mut_ptr(), x);
        copy_nonoverlapping(bt.as_ptr(), b.add(16), M-16);
        return;
    }
    if M >= 8 {
        let mut at = [0u16; 8];
        let mut bt = [0f32; 8];
        let a = a as *const u16;
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a as *const __m128i));
        _mm256_storeu_ps(b, x);
        copy_nonoverlapping(a.add(8), at.as_mut_ptr(), M-8);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(at.as_ptr() as *const __m128i));
        _mm256_storeu_ps(bt.as_mut_ptr(), x);
        copy_nonoverlapping(bt.as_ptr(), b.add(8), M-8);
        return;
    }
    let mut at = [0u16; 8];
    let mut bt = [0f32; 8];
    copy_nonoverlapping(a as *const u16, at.as_mut_ptr(), M);
    let x = _mm256_cvtph_ps(_mm_loadu_si128(at.as_ptr() as *const __m128i));
    _mm256_storeu_ps(bt.as_mut_ptr(), x);
    copy_nonoverlapping(bt.as_ptr(), b, M);
}


#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn pack_k_v0<const M: usize, const MR: usize>(
    k: usize,
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    while k_i < k8 {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        copy_packed::<M>(a, ap);
        copy_packed::<M>(a.add(lda), ap.add(MR));
        copy_packed::<M>(a.add(lda*2), ap.add(MR*2));
        copy_packed::<M>(a.add(lda*3), ap.add(MR*3));
        copy_packed::<M>(a.add(lda*4), ap.add(MR*4));
        copy_packed::<M>(a.add(lda*5), ap.add(MR*5));
        copy_packed::<M>(a.add(lda*6), ap.add(MR*6));
        copy_packed::<M>(a.add(lda*7), ap.add(MR*7));

        k_i += 8;
    }

    while k_i < k {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        copy_packed::<M>(a, ap);

        k_i += 1;
    }

}


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_k_v0_2<const M: usize, const MR: usize>(
    k: usize,
    a: *const u16, lda: usize,
    ap: *mut u16,
) {
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    while k_i < k8 {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        std::ptr::copy_nonoverlapping(a, ap, M);
        std::ptr::copy_nonoverlapping(a.add(lda), ap.add(MR), M);
        std::ptr::copy_nonoverlapping(a.add(lda*2), ap.add(MR*2), M);
        std::ptr::copy_nonoverlapping(a.add(lda*3), ap.add(MR*3), M);
        std::ptr::copy_nonoverlapping(a.add(lda*4), ap.add(MR*4), M);
        std::ptr::copy_nonoverlapping(a.add(lda*5), ap.add(MR*5), M);
        std::ptr::copy_nonoverlapping(a.add(lda*6), ap.add(MR*6), M);
        std::ptr::copy_nonoverlapping(a.add(lda*7), ap.add(MR*7), M);

        k_i += 8;
    }

    while k_i < k {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        std::ptr::copy_nonoverlapping(a, ap, M);

        k_i += 1;
    }

}

#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn pack_kx24_v1(
    k: usize,
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    const MR: usize = 24;
    while k_i < k8 {
        let a = a0.add(k_i);
        let ap = ap0.add(k_i*MR);
        pack_t::<MR>(a, lda, ap);
        pack_t::<MR>(a.add(8*lda), lda, ap.add(8));
        pack_t::<MR>(a.add(16*lda), lda, ap.add(16));
        k_i += 8;
    }

    while k_i < k {
        let a = a0.add(k_i);
        let ap = ap0.add(k_i*MR);
        seq!(i in 0..24 {
            copy_packed::<1>(a.add(i*lda), ap.add(i));
        });
        k_i += 1;
    }
}

// #[target_feature(enable = "avx,f16c")]
// pub(crate) unsafe fn pack_kx16_v1(
//     k: usize,
//     a: *const f16, lda: usize,
//     ap: *mut f32,
// ) {
//     let k8 = k / 8 * 8;
//     let mut k_i = 0;
//     const MR: usize = 16;
//     while k_i < k8 {
//         pack_t::<MR>(a, lda, ap);
//         pack_t::<MR>(a.add(8*lda), lda, ap.add(8));

//         ap = ap.add(MR*8);
//         a = a.add(8);
//         k_i += 1;
//     }

//     k_i = 0;

//     while k_i < k_left {
//         seq!(i in 0..16 {
//             copy_packed::<1>(a.add(i*lda), ap.add(i));
//         });

//         ap = ap.add(MR);
//         a = a.add(1);
//         k_i += 1;
//     }
// }


// #[target_feature(enable = "avx")]
// pub(crate) unsafe fn pack_kx6_v1(
//     k: usize,
//     a: *const f16, lda: usize,
//     ap: *mut f32,
// ) {
//     let mut k_i = 0;
//     const MR: usize = 6;
//     while k_i < k8 {
//         pack_t::<4>(a, lda, ap);
//         pack_t::<2>(a.add(4*lda), lda, ap.add(4));

//         ap = ap.add(MR*8);
//         a = a.add(8);
//         k_i += 1;
//     }

//     k_i = 0;

//     while k_i < k_left {
//         seq!(i in 0..12 {
//             copy_packed::<1>(a.add(i*lda), ap.add(i));
//         });

//         ap = ap.add(MR);
//         a = a.add(1);
//         k_i += 1;
//     }
// }



#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn pack_kx48_v1(
    k: usize,
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    const MR: usize = 48;
    while k_i < k8 {
        let a = a0.add(k_i);
        let ap = ap0.add(k_i*MR);
        pack_t::<MR>(a, lda, ap);
        pack_t::<MR>(a.add(8*lda), lda, ap.add(8));
        pack_t::<MR>(a.add(16*lda), lda, ap.add(16));
        pack_t::<MR>(a.add(24*lda), lda, ap.add(24));
        pack_t::<MR>(a.add(32*lda), lda, ap.add(32));
        pack_t::<MR>(a.add(40*lda), lda, ap.add(40));

        k_i += 8;
    }

    while k_i < k {
        let a = a0.add(k_i);
        let ap = ap0.add(k_i*MR);
        seq!(i in 0..48 {
            copy_packed::<1>(a.add(i*lda), ap.add(i));
        });
        k_i += 1;
    }
}


#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn pack_kx16_v1(
    k: usize,
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    const MR: usize = 16;
    while k_i < k8 {
        let a = a0.add(k_i);
        let ap = ap0.add(k_i*MR);
        pack_t::<MR>(a, lda, ap);
        pack_t::<MR>(a.add(8*lda), lda, ap.add(8));

        k_i += 8;
    }

    while k_i < k {
        let a = a0.add(k_i);
        let ap = ap0.add(k_i*MR);
        seq!(i in 0..16 {
            copy_packed::<1>(a.add(i*lda), ap.add(i));
        });
        k_i += 1;
    }
}



#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn pack_kx4_v1(
    k: usize,
    b: *const f16, ldb: usize,
    bp: *mut f32,
) {
    let k8 = k / 8 * 8;
    let b0 = b as *const u16;
    let bp0 = bp;

    let mut k_i = 0;
    const M: usize = 4;

    while k_i < k8 {
        let b = b0.add(k_i);
        let bp = bp0.add(k_i*M);
        let a0 = _mm256_cvtph_ps(_mm_loadu_si128(b as *const __m128i));
        let a1 = _mm256_cvtph_ps(_mm_loadu_si128(b.add(ldb) as *const __m128i));
        let a2 = _mm256_cvtph_ps(_mm_loadu_si128(b.add(ldb*2) as *const __m128i));
        let a3 = _mm256_cvtph_ps(_mm_loadu_si128(b.add(ldb*3) as *const __m128i));

        // transpose
        let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
        let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
        let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

        let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        storeu_ps::<M>(x0, bp);
        storeu_ps::<M>(x0_h, bp.add(M*4));

        let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        storeu_ps::<M>(x1, bp.add(M));
        storeu_ps::<M>(x1_h, bp.add(M+M*4));

        let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        storeu_ps::<M>(x2, bp.add(2*M));
        storeu_ps::<M>(x2_h, bp.add(2*M+M*4));

        let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        storeu_ps::<M>(x3, bp.add(3*M));
        storeu_ps::<M>(x3_h, bp.add(3*M+M*4));

        k_i += 8;
    }
    let b0 = b0 as *const f16;
    while k_i < k {
        let b = b0.add(k_i);
        let bp = bp0.add(k_i*M);
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        copy_packed::<1>(b.add(ldb*2), bp.add(2));
        copy_packed::<1>(b.add(ldb*3), bp.add(3));
        k_i += 1;
    }
}


#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn pack_kx8_v1(
    k: usize,
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    const MR: usize = 8;
    while k_i < k8 {
        let a = a0.add(k_i);
        let ap = ap0.add(k_i*MR);
        pack_t::<MR>(a, lda, ap);

        k_i += 8;
    }

    while k_i < k {
        let a = a0.add(k_i);
        let ap = ap0.add(k_i*MR);
        seq!(i in 0..8 {
            copy_packed::<1>(a.add(i*lda), ap.add(i));
        });
        k_i += 1;
    }
}




macro_rules! def_packb {
   ($nr:tt) => {
       seq!(NL in 1..$nr {
           paste! {
            // #[target_feature(enable = "avx")]
            pub(crate) unsafe fn [<packb_panel_ $nr>](
                   n: usize, k: usize,
                   b: *const f16, b_rs: usize, b_cs: usize,
                   bp: *mut f32,
               ) {
                    let b0 = b;
                    let bp0 = bp;
                    const NR: usize = $nr;
                    let n_rounded = n / NR * NR;
                    let mut n_idx = 0;
                   if b_rs == 1 {
                       let ldb = b_cs;
                       while n_idx < n_rounded {
                            let b = b0.add(n_idx);
                            let bp = bp0.add(n_idx*k);
                            pack_k_v0::<NR,NR>(k, b, ldb, bp);
                            n_idx += NR;
                        }
                       let n_left = n - n_idx;
                       #(
                           if n_left == NL {
                            let b = b0.add(n_idx);
                            let bp = bp0.add(n_idx*k);
                               pack_k_v0::<NL,NL>(k, b, ldb, bp);
                               return;
                           }
                       )*
                   } else if b_cs == 1 {
                       let ldb = b_rs;
                       while n_idx < n_rounded {
                            let b = b0.add(n_idx*ldb);
                            let bp = bp0.add(n_idx*k);
                            [<pack_kx$nr _v1>](k, b, ldb, bp);
                            n_idx += NR;
                       }
                       let n_left = n - n_idx;
                       if n_left > 0 {
                           let b = b0.add(n_idx*ldb);
                           let bp = bp0.add(n_idx*k);
                           pack_scalar_k(
                               n_left, k,
                               b, b_rs, b_cs,
                               bp, 1
                           );
                       }
                   }
               }   
           }
       });
   };
}


def_packb!(4);
def_packb!(8);
// def_packb!(6);


macro_rules! def_packa {
    ($mr:tt) => {
        paste! {
            // #[target_feature(enable = "avx")]
            pub(crate) unsafe fn [<packa_panel_ $mr>](
                m: usize, k: usize,
                a: *const f16, a_rs: usize, a_cs: usize,
                ap: *mut f32, vs: usize,
            ) {
                let ap0 = ap;
                let a0 = a;
                const MR: usize = $mr;
                let m_rounded = m / MR * MR;
                let mut m_idx = 0;
                if a_rs == 1 {
                    let lda = a_cs;
                    while m_idx < m_rounded {
                        let a = a0.add(m_idx);
                        let ap = ap0.add(m_idx*k);
                        pack_k_v0::<$mr, $mr>(k, a, lda, ap);
                        m_idx += MR;
                    }
                    let m_left = m - m_idx;
                    if m_left > 0 {
                        pack_scalar_k(
                            m_left, k,
                            a0.add(m_idx), a_rs, a_cs,
                            ap0.add(m_idx*k), vs
                        );
                    }

                } else if a_cs == 1 {
                    let lda = a_rs;
                    while m_idx < m_rounded {
                        let a = a0.add(m_idx*lda);
                        let ap = ap0.add(m_idx*k);
                        [<pack_kx$mr _v1>](k, a, lda, ap);
                        m_idx += MR;
                    }
                    let m_left = m - m_idx;
                    if m_left > 0 {
                        pack_scalar_k(
                            m_left, k,
                            a0.add(m_idx*lda), a_rs, a_cs,
                            ap0.add(m_idx*k), vs
                        );
                    }
                }
            }
        }
    };
}
def_packa!(16);
def_packa!(24);
def_packa!(48);

// def_packa!(16, 16, 8);








macro_rules! def_packb {
    ($nr:tt) => {
        seq!(NL in 1..$nr {
            paste! {
             // #[target_feature(enable = "avx")]
             pub(crate) unsafe fn [<packb_panel_ $nr _same>](
                    n: usize, k: usize,
                    b: *const f16, b_rs: usize, b_cs: usize,
                    bp: *mut f16,
                ) {
                    let b0 = b as *const u16;
                    let bp0 = bp as *mut u16;
                    const NR: usize = $nr;
                    let n_rounded = n / NR * NR;
                    let mut n_idx = 0;
                    if b_rs == 1 {
                        let ldb = b_cs;
                        while n_idx < n_rounded {
                            let b = b0.add(n_idx);
                            let bp = bp0.add(n_idx*k);
                            pack_k_v0_2::<$nr, $nr>(k, b, ldb, bp);
                            n_idx += NR;
                        }
                        let n_left = n - n_idx;
                        #(
                            if n_left == NL {
                                let b = b0.add(n_idx);
                                let bp = bp0.add(n_idx*k);
                                pack_k_v0_2::<NL,NL>(k, b, ldb, bp);
                                return;
                            }
                        )*
                    } else if b_cs == 1 {
                        let ldb = b_rs;
                        while n_idx < n_rounded {
                            let b = b0.add(n_idx*ldb);
                            let bp = bp0.add(n_idx*k);
                            // [<pack_kx$nr _v1>](k, b, ldb, bp);
                            pack_scalar_k_2(
                                $nr, k,
                                b, b_rs, b_cs,
                                bp, 1
                            );
                            n_idx += NR;
                        }
                        let n_left = n - n_idx;
                        if n_left > 0 {
                            let b = b0.add(n_idx*ldb);
                            let bp = bp0.add(n_idx*k);
                            pack_scalar_k_2(
                                n_left, k,
                                b, b_rs, b_cs,
                                bp, 1
                            );
                        }
                    }
                }   
            }
        });
    };
 }
 
 
 def_packb!(4);
 def_packb!(8);
 def_packb!(15);
 // def_packb!(6);
 
 
 macro_rules! def_packa {
     ($mr:tt) => {
         paste! {
             // #[target_feature(enable = "avx")]
             pub(crate) unsafe fn [<packa_panel_ $mr _same>](
                 m: usize, k: usize,
                 a: *const f16, a_rs: usize, a_cs: usize,
                 ap: *mut f16, vs: usize,
             ) {
                let ap0 = ap as *mut u16;
                let a0 = a as *const u16;
                const MR: usize = $mr;
                let m_rounded = m / MR * MR;
                let mut m_idx = 0;
                 if a_rs == 1 {
                     let lda = a_cs;
                     while m_idx < m_rounded {
                        let a = a0.add(m_idx);
                        let ap = ap0.add(m_idx*k);
                         pack_k_v0_2::<$mr, $mr>(k, a, lda, ap);
                         m_idx += MR;
                     }
                     let m_left = m - m_idx;
                     pack_scalar_k_2(
                        m_left, k, 
                        a0.add(m_idx), a_rs, a_cs,
                        ap0.add(m_idx*k), vs
                    );
 
                 } else if a_cs == 1 {
                     let lda = a_rs;
                     while m_idx < m_rounded {
                        let a = a0.add(m_idx*lda);
                        let ap = ap0.add(m_idx*k);
                        pack_scalar_k_2(
                            $mr, k,
                            a, a_rs, a_cs,
                            ap, vs
                        );
                         m_idx += MR;
                     }
                     let m_left = m - m_idx;
                     pack_scalar_k_2(
                        m_left, k, 
                        a0.add(m_idx*lda), a_rs, a_cs,
                        ap0.add(m_idx*k), vs
                    );
                 }
             }
         }
     };
 }
 def_packa!(16);
 def_packa!(24);
 def_packa!(48);
 def_packa!(64);
 
 // def_packa!(16, 16, 8);
