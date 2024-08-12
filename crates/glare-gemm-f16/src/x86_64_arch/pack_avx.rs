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
pub(crate) unsafe fn pack_scalar_k<const MR: usize>(
    m_left: usize, k: usize,
    a: *const f16, a_rs: usize, a_cs: usize,
    ap: *mut f32,
) {
    for i in 0..m_left  {
        for j in 0..k {
            *ap.add(j*MR+i) = (*a.add(j*a_cs + i*a_rs)).to_f32();
        }
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_scalar_k_2<const MR: usize>(
    m_left: usize, k: usize,
    a: *const u16, a_rs: usize, a_cs: usize,
    ap: *mut u16,
) {
    for i in 0..m_left  {
        for j in 0..k {
            *ap.add(j*MR+i) = *a.add(j*a_cs + i*a_rs);
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
    k_iter: usize, k_left: usize,
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        // use vector intrinscs
        copy_packed::<M>(a, ap);
        copy_packed::<M>(a.add(lda), ap.add(MR));
        copy_packed::<M>(a.add(lda*2), ap.add(MR*2));
        copy_packed::<M>(a.add(lda*3), ap.add(MR*3));
        copy_packed::<M>(a.add(lda*4), ap.add(MR*4));
        copy_packed::<M>(a.add(lda*5), ap.add(MR*5));
        copy_packed::<M>(a.add(lda*6), ap.add(MR*6));
        copy_packed::<M>(a.add(lda*7), ap.add(MR*7));

        ap = ap.add(MR*8);
        a = a.add(8*lda);

        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<M>(a, ap);

        ap = ap.add(MR);
        a = a.add(lda);
        k_i += 1;
    }

}


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_k_v0_2<const M: usize, const MR: usize>(
    k_iter: usize, k_left: usize,
    a: *const u16, lda: usize,
    ap: *mut u16,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        // use vector intrinscs
        std::ptr::copy_nonoverlapping(a, ap, M);
        std::ptr::copy_nonoverlapping(a.add(lda), ap.add(MR), M);
        std::ptr::copy_nonoverlapping(a.add(lda*2), ap.add(MR*2), M);
        std::ptr::copy_nonoverlapping(a.add(lda*3), ap.add(MR*3), M);
        std::ptr::copy_nonoverlapping(a.add(lda*4), ap.add(MR*4), M);
        std::ptr::copy_nonoverlapping(a.add(lda*5), ap.add(MR*5), M);
        std::ptr::copy_nonoverlapping(a.add(lda*6), ap.add(MR*6), M);
        std::ptr::copy_nonoverlapping(a.add(lda*7), ap.add(MR*7), M);

        ap = ap.add(MR*8);
        a = a.add(8*lda);

        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        std::ptr::copy_nonoverlapping(a, ap, M);

        ap = ap.add(MR);
        a = a.add(lda);
        k_i += 1;
    }

}

#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn pack_kx24_v1(
    k_iter: usize, k_left: usize,
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 24;
    while k_i < k_iter {
        pack_t::<MR>(a, lda, ap);
        pack_t::<MR>(a.add(8*lda), lda, ap.add(8));
        pack_t::<MR>(a.add(16*lda), lda, ap.add(16));

        ap = ap.add(MR*8);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        seq!(i in 0..24 {
            copy_packed::<1>(a.add(i*lda), ap.add(i));
        });

        ap = ap.add(MR);
        a = a.add(1);
        k_i += 1;
    }
}

// #[target_feature(enable = "avx,f16c")]
// pub(crate) unsafe fn pack_kx16_v1(
//     k_iter: usize, k_left: usize,
//     a: *const f16, lda: usize,
//     ap: *mut f32,
// ) {
//     let mut k_i = 0;
//     let mut a = a;
//     let mut ap = ap;
//     const MR: usize = 16;
//     while k_i < k_iter {
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
//     k_iter: usize, k_left: usize,
//     a: *const f16, lda: usize,
//     ap: *mut f32,
// ) {
//     let mut k_i = 0;
//     let mut a = a;
//     let mut ap = ap;
//     const MR: usize = 6;
//     while k_i < k_iter {
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
    k_iter: usize, k_left: usize,
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 48;
    while k_i < k_iter {
        pack_t::<MR>(a, lda, ap);
        pack_t::<MR>(a.add(8*lda), lda, ap.add(8));
        pack_t::<MR>(a.add(16*lda), lda, ap.add(16));
        pack_t::<MR>(a.add(24*lda), lda, ap.add(24));
        pack_t::<MR>(a.add(32*lda), lda, ap.add(32));
        pack_t::<MR>(a.add(40*lda), lda, ap.add(40));

        ap = ap.add(MR*8);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        seq!(i in 0..48 {
            copy_packed::<1>(a.add(i*lda), ap.add(i));
        });

        ap = ap.add(MR);
        a = a.add(1);
        k_i += 1;
    }
}


#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn pack_kx4_v1(
    k_iter: usize, k_left: usize,
    b: *const f16, ldb: usize,
    bp: *mut f32,
) {
    let mut b = b as *const u16;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 4;

    while k_i < k_iter {
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

        b = b.add(8);
        bp = bp.add(M*8);
        k_i += 1;
    }

    k_i = 0;
    let mut b = b as *const f16;
    while k_i <  k_left {
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        copy_packed::<1>(b.add(ldb*2), bp.add(2));
        copy_packed::<1>(b.add(ldb*3), bp.add(3));
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}


#[target_feature(enable = "avx,f16c")]
pub(crate) unsafe fn pack_kx8_v1(
    k_iter: usize, k_left: usize,
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 8;
    while k_i < k_iter {
        pack_t::<MR>(a, lda, ap);

        ap = ap.add(MR*8);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        seq!(i in 0..8 {
            copy_packed::<1>(a.add(i*lda), ap.add(i));
        });

        ap = ap.add(MR);
        a = a.add(1);
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
                   let k_iter = k / 8;
                   let k_left = k % 8;
                   let mut bp = bp;
                   let mut b = b;
                   const NR: usize = $nr;
                   const NR_LAST_STEP: usize = $nr;
                   let mut n_idx = 0;
                   if b_rs == 1 {
                       let ldb = b_cs;
                       while n_idx + NR_LAST_STEP <= n {
                            pack_k_v0::<$nr, $nr>(k_iter, k_left, b, ldb, bp);
                           n_idx += NR;
                           bp = bp.add(k*NR);
                           b = b.add(NR);
                       }
                       let n_left = n - n_idx;
                       #(
                           if n_left == NL {
                               pack_k_v0::<NL,NL>(k_iter, k_left, b, ldb, bp);
                               return;
                           }
                       )*
                   } else if b_cs == 1 {
                       let ldb = b_rs;
                       while n_idx + NR_LAST_STEP <= n {
                           [<pack_kx$nr _v1>](k_iter, k_left, b, ldb, bp);
                           n_idx += NR;
                           bp = bp.add(k*NR);
                           b =  b.add(NR*ldb);
                       }
                       let n_left = n - n_idx;
                       #(
                           if n_left == NL {
                                pack_scalar_k::<NL>(
                                    NL, k,
                                    b, b_rs, b_cs,
                                    bp
                                );
                               return;
                           }
                       )*
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
    ($mr:tt, $vs:tt) => {
        paste! {
            // #[target_feature(enable = "avx")]
            pub(crate) unsafe fn [<packa_panel_ $mr>](
                m_left: usize, k: usize,
                a: *const f16, a_rs: usize, a_cs: usize,
                ap: *mut f32,
            ) {
                let mut ap = ap;
                let mut a = a;
                const MR: usize = $mr;
                const MR_LAST_STEP: usize = $mr;
                let mut m_idx = 0;
                if a_rs == 1 {
                    let lda = a_cs;
                    let k_iter = k / 8;
                    let k_left = k % 8;
                    while m_idx + MR_LAST_STEP <= m_left {
                        pack_k_v0::<$mr, $mr>(k_iter, k_left, a, lda, ap);
                        m_idx += MR;
                        ap = ap.add(k * MR);
                        a = a.add(MR);
                    }
                    let m_left = m_left - m_idx;
                    seq!(mr_left in 1..$mr {
                        if m_left == mr_left {
                            pack_k_v0::<mr_left, {(mr_left+$vs-1)/ $vs* $vs}>(k_iter, k_left, a, lda, ap);
                            return;
                        }
                    });

                } else if a_cs == 1 {
                    let lda = a_rs;
                    let k_iter = k / 8;
                    let k_left = k % 8;
                    while m_idx + MR_LAST_STEP <= m_left {
                        [<pack_kx$mr _v1>](k_iter, k_left, a, lda, ap);
                        m_idx += MR;
                        ap = ap.add(k * MR);
                        a = a.add(MR*lda);
                    }
                    let m_left = m_left - m_idx;
                    seq!(mr_left in 1..$mr {
                        if m_left == mr_left {
                            pack_scalar_k::<{(mr_left+$vs-1)/ $vs* $vs}>(
                                mr_left, k,
                                a, a_rs, a_cs,
                                ap
                            );
                            return;
                        }
                    });
                }
            }
        }
    };
}

def_packa!(24, 8);
def_packa!(48, 16);

// def_packa!(16, 16, 8);








macro_rules! def_packb {
    ($nr:tt) => {
        seq!(NL in 1..$nr {
            paste! {
             // #[target_feature(enable = "avx")]
             pub(crate) unsafe fn [<packb_panel_ $nr _same>](
                    n: usize, k: usize,
                    b: *const u16, b_rs: usize, b_cs: usize,
                    bp: *mut u16,
                ) {
                    let k_iter = k / 8;
                    let k_left = k % 8;
                    let mut bp = bp;
                    let mut b = b;
                    const NR: usize = $nr;
                    const NR_LAST_STEP: usize = $nr;
                    let mut n_idx = 0;
                    if b_rs == 1 {
                        let ldb = b_cs;
                        while n_idx + NR_LAST_STEP <= n {
                             pack_k_v0_2::<$nr, $nr>(k_iter, k_left, b, ldb, bp);
                            n_idx += NR;
                            bp = bp.add(k*NR);
                            b = b.add(NR);
                        }
                        let n_left = n - n_idx;
                        #(
                            if n_left == NL {
                                pack_k_v0_2::<NL,NL>(k_iter, k_left, b, ldb, bp);
                                return;
                            }
                        )*
                    } else if b_cs == 1 {
                        let ldb = b_rs;
                        while n_idx + NR_LAST_STEP <= n {
                            // [<pack_kx$nr _v1>](k_iter, k_left, b, ldb, bp);
                            pack_scalar_k_2::<$nr>(
                                $nr, k,
                                b, b_rs, b_cs,
                                bp
                            );
                            n_idx += NR;
                            bp = bp.add(k*NR);
                            b =  b.add(NR*ldb);
                        }
                        let n_left = n - n_idx;
                        #(
                            if n_left == NL {
                                 pack_scalar_k_2::<NL>(
                                     NL, k,
                                     b, b_rs, b_cs,
                                     bp
                                 );
                                return;
                            }
                        )*
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
     ($mr:tt, $vs:tt) => {
         paste! {
             // #[target_feature(enable = "avx")]
             pub(crate) unsafe fn [<packa_panel_ $mr _same>](
                 m_left: usize, k: usize,
                 a: *const u16, a_rs: usize, a_cs: usize,
                 ap: *mut u16,
             ) {
                 let mut ap = ap;
                 let mut a = a;
                 const MR: usize = $mr;
                 const MR_LAST_STEP: usize = $mr;
                 let mut m_idx = 0;
                 if a_rs == 1 {
                     let lda = a_cs;
                     let k_iter = k / 8;
                     let k_left = k % 8;
                     while m_idx + MR_LAST_STEP <= m_left {
                         pack_k_v0_2::<$mr, $mr>(k_iter, k_left, a, lda, ap);
                         m_idx += MR;
                         ap = ap.add(k * MR);
                         a = a.add(MR);
                     }
                     let m_left = m_left - m_idx;
                     seq!(mr_left in 1..$mr {
                         if m_left == mr_left {
                             pack_k_v0_2::<mr_left, {(mr_left+$vs-1)/ $vs* $vs}>(k_iter, k_left, a, lda, ap);
                             return;
                         }
                     });
 
                 } else if a_cs == 1 {
                     let lda = a_rs;
                     let k_iter = k / 8;
                     let k_left = k % 8;
                     while m_idx + MR_LAST_STEP <= m_left {
                        //  [<pack_kx$mr _v1>](k_iter, k_left, a, lda, ap);
                        pack_scalar_k_2::<{($mr+$vs-1)/ $vs* $vs}>(
                            $mr, k,
                            a, a_rs, a_cs,
                            ap
                        );
                         m_idx += MR;
                         ap = ap.add(k * MR);
                         a = a.add(MR*lda);
                     }
                     let m_left = m_left - m_idx;
                     seq!(mr_left in 1..$mr {
                         if m_left == mr_left {
                             pack_scalar_k_2::<{(mr_left+$vs-1)/ $vs* $vs}>(
                                 mr_left, k,
                                 a, a_rs, a_cs,
                                 ap
                             );
                             return;
                         }
                     });
                 }
             }
         }
     };
 }
 
 def_packa!(24, 8);
 def_packa!(48, 16);
 
 // def_packa!(16, 16, 8);
 