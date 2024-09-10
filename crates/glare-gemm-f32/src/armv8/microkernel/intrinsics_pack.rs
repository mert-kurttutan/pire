use crate::{TA, TB};
use seq_macro::seq;
use std::ptr::copy_nonoverlapping;

use paste::paste;

use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_t<const MR: usize>(a: *const TA, lda: usize, ap: *mut TB) {
    // let a0 = _mm256_loadu_ps(a);
    // let a1 = _mm256_loadu_ps(a.add(lda));
    // let a2 = _mm256_loadu_ps(a.add(lda*2));
    // let a3 = _mm256_loadu_ps(a.add(lda*3));

    // // transpose
    // let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
    // let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
    // let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
    // let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

    // let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
    // let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
    // let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
    // let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));

    // let a0 = _mm256_loadu_ps(a.add(lda*4));
    // let a1 = _mm256_loadu_ps(a.add(lda*5));
    // let a2 = _mm256_loadu_ps(a.add(lda*6));
    // let a3 = _mm256_loadu_ps(a.add(lda*7));

    // // transpose
    // let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
    // let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
    // let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
    // let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

    // let x4 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
    // let x5 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
    // let x6 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
    // let x7 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));

    // // exchange hi of x0 and lo of x4
    // let x0_t = _mm256_permute2f128_ps(x0, x4, 0b0010_0000);
    // let x4_t = _mm256_permute2f128_ps(x0, x4, 0b0011_0001);
    // // exchange hi of x1 and lo of x5
    // let x1_t = _mm256_permute2f128_ps(x1, x5, 0b0010_0000);
    // let x5_t = _mm256_permute2f128_ps(x1, x5, 0b0011_0001);
    // // exchange hi of x2 and lo of x6
    // let x2_t = _mm256_permute2f128_ps(x2, x6, 0b0010_0000);
    // let x6_t = _mm256_permute2f128_ps(x2, x6, 0b0011_0001);
    // // exchange hi of x3 and lo of x7
    // let x3_t = _mm256_permute2f128_ps(x3, x7, 0b0010_0000);
    // let x7_t = _mm256_permute2f128_ps(x3, x7, 0b0011_0001);

    // _mm256_store_ps(ap, x0_t);
    // _mm256_store_ps(ap.add(MR), x1_t);
    // _mm256_store_ps(ap.add(MR*2), x2_t);
    // _mm256_store_ps(ap.add(MR*3), x3_t);
    // _mm256_store_ps(ap.add(MR*4), x4_t);
    // _mm256_store_ps(ap.add(MR*5), x5_t);
    // _mm256_store_ps(ap.add(MR*6), x6_t);
    // _mm256_store_ps(ap.add(MR*7), x7_t);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_scalar_k<const MR: usize>(
    m_left: usize,
    k: usize,
    a: *const TA,
    a_rs: usize,
    a_cs: usize,
    ap: *mut TA,
) {
    for i in 0..m_left {
        for j in 0..k {
            *ap.add(j * MR + i) = *a.add(j * a_cs + i * a_rs);
        }
    }
}

// #[target_feature(enable = "neon")]
// pub(crate) unsafe fn storeu_ps<const M: usize>(
//     src: __m256, dst: *mut f32
// ) {
//     let mut temp_arr = [0.0; 8];
//     _mm256_storeu_ps(temp_arr.as_mut_ptr(), src);
//     copy_nonoverlapping(temp_arr.as_ptr(), dst, M);
// }

#[target_feature(enable = "neon")]
pub(crate) unsafe fn copy_packed<const M: usize>(a: *const f32, b: *mut f32) {
    std::ptr::copy_nonoverlapping(a, b, M);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_k_v0<const M: usize, const MR: usize>(
    k_iter: usize,
    k_left: usize,
    a: *const TA,
    lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        // use vector intrinscs
        copy_packed::<M>(a, ap);
        copy_packed::<M>(a.add(lda), ap.add(MR));
        copy_packed::<M>(a.add(lda * 2), ap.add(MR * 2));
        copy_packed::<M>(a.add(lda * 3), ap.add(MR * 3));
        copy_packed::<M>(a.add(lda * 4), ap.add(MR * 4));
        copy_packed::<M>(a.add(lda * 5), ap.add(MR * 5));
        copy_packed::<M>(a.add(lda * 6), ap.add(MR * 6));
        copy_packed::<M>(a.add(lda * 7), ap.add(MR * 7));

        ap = ap.add(MR * 8);
        a = a.add(8 * lda);

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

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx24_v0(
    k_iter: usize,
    k_left: usize,
    a: *const TA,
    lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 24;
    while k_i < k_iter {
        // use vector intrinscs
        // seq!(i in 0..8 {
        //     let a0 = _mm256_loadu_ps(a.add(lda*i));
        //     let a1 = _mm256_loadu_ps(a.add(lda*i+8));
        //     let a2 = _mm256_loadu_ps(a.add(lda*i+16));
        //     _mm256_store_ps(ap.add(i*MR), a0);
        //     _mm256_store_ps(ap.add(i*MR+8), a1);
        //     _mm256_store_ps(ap.add(i*MR+16), a2);
        // });

        ap = ap.add(MR * 8);
        a = a.add(8 * lda);

        k_i += 1;
    }
    k_i = 0;
    while k_i < k_left {
        // let a0 = _mm256_loadu_ps(a);
        // let a1 = _mm256_loadu_ps(a.add(8));
        // let a2 = _mm256_loadu_ps(a.add(16));
        // _mm256_store_ps(ap, a0);
        // _mm256_store_ps(ap.add(8), a1);
        // _mm256_store_ps(ap.add(16), a2);

        ap = ap.add(MR);
        a = a.add(lda);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx16_v0(
    k_iter: usize,
    k_left: usize,
    a: *const TA,
    lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 16;
    while k_i < k_iter {
        // use vector intrinscs
        // seq!(i in 0..8 {
        //     let a0 = _mm256_loadu_ps(a.add(lda*i));
        //     let a1 = _mm256_loadu_ps(a.add(lda*i+8));
        //     _mm256_store_ps(ap.add(i*MR), a0);
        //     _mm256_store_ps(ap.add(i*MR+8), a1);
        // });

        ap = ap.add(MR * 8);
        a = a.add(8 * lda);

        k_i += 1;
    }
    k_i = 0;
    while k_i < k_left {
        // let a0 = _mm256_loadu_ps(a);
        // let a1 = _mm256_loadu_ps(a.add(8));
        // _mm256_store_ps(ap, a0);
        // _mm256_store_ps(ap.add(8), a1);

        ap = ap.add(MR);
        a = a.add(lda);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx6_v0(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 6;
    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb * 2), bp.add(M * 2));
        copy_packed::<M>(b.add(ldb * 3), bp.add(M * 3));
        copy_packed::<M>(b.add(ldb * 4), bp.add(M * 4));
        copy_packed::<M>(b.add(ldb * 5), bp.add(M * 5));
        copy_packed::<M>(b.add(ldb * 6), bp.add(M * 6));
        copy_packed::<M>(b.add(ldb * 7), bp.add(M * 7));
        b = b.add(8 * ldb);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx5_v0(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 5;
    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb * 2), bp.add(M * 2));
        copy_packed::<M>(b.add(ldb * 3), bp.add(M * 3));
        copy_packed::<M>(b.add(ldb * 4), bp.add(M * 4));
        copy_packed::<M>(b.add(ldb * 5), bp.add(M * 5));
        copy_packed::<M>(b.add(ldb * 6), bp.add(M * 6));
        copy_packed::<M>(b.add(ldb * 7), bp.add(M * 7));
        b = b.add(8 * ldb);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx4_v0(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 4;
    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb * 2), bp.add(M * 2));
        copy_packed::<M>(b.add(ldb * 3), bp.add(M * 3));
        copy_packed::<M>(b.add(ldb * 4), bp.add(M * 4));
        copy_packed::<M>(b.add(ldb * 5), bp.add(M * 5));
        copy_packed::<M>(b.add(ldb * 6), bp.add(M * 6));
        copy_packed::<M>(b.add(ldb * 7), bp.add(M * 7));
        b = b.add(8 * ldb);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx3_v0(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 3;

    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb * 2), bp.add(M * 2));
        copy_packed::<M>(b.add(ldb * 3), bp.add(M * 3));
        copy_packed::<M>(b.add(ldb * 4), bp.add(M * 4));
        copy_packed::<M>(b.add(ldb * 5), bp.add(M * 5));
        copy_packed::<M>(b.add(ldb * 6), bp.add(M * 6));
        copy_packed::<M>(b.add(ldb * 7), bp.add(M * 7));
        b = b.add(8 * ldb);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx2_v0(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 2;

    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb * 2), bp.add(M * 2));
        copy_packed::<M>(b.add(ldb * 3), bp.add(M * 3));
        copy_packed::<M>(b.add(ldb * 4), bp.add(M * 4));
        copy_packed::<M>(b.add(ldb * 5), bp.add(M * 5));
        copy_packed::<M>(b.add(ldb * 6), bp.add(M * 6));
        copy_packed::<M>(b.add(ldb * 7), bp.add(M * 7));
        b = b.add(8 * ldb);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx1_v0(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 1;

    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb * 2), bp.add(M * 2));
        copy_packed::<M>(b.add(ldb * 3), bp.add(M * 3));
        copy_packed::<M>(b.add(ldb * 4), bp.add(M * 4));
        copy_packed::<M>(b.add(ldb * 5), bp.add(M * 5));
        copy_packed::<M>(b.add(ldb * 6), bp.add(M * 6));
        copy_packed::<M>(b.add(ldb * 7), bp.add(M * 7));
        b = b.add(8 * ldb);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx24_v1(
    k_iter: usize,
    k_left: usize,
    a: *const TA,
    lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        pack_t::<24>(a, lda, ap);

        pack_t::<24>(a.add(8 * lda), lda, ap.add(8));

        pack_t::<24>(a.add(16 * lda), lda, ap.add(16));

        ap = ap.add(192);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<1>(a, ap);
        copy_packed::<1>(a.add(lda), ap.add(1));
        copy_packed::<1>(a.add(lda * 2), ap.add(2));
        copy_packed::<1>(a.add(lda * 3), ap.add(3));
        copy_packed::<1>(a.add(lda * 4), ap.add(4));
        copy_packed::<1>(a.add(lda * 5), ap.add(5));
        copy_packed::<1>(a.add(lda * 6), ap.add(6));
        copy_packed::<1>(a.add(lda * 7), ap.add(7));
        copy_packed::<1>(a.add(lda * 8), ap.add(8));
        copy_packed::<1>(a.add(lda * 9), ap.add(9));
        copy_packed::<1>(a.add(lda * 10), ap.add(10));
        copy_packed::<1>(a.add(lda * 11), ap.add(11));
        copy_packed::<1>(a.add(lda * 12), ap.add(12));
        copy_packed::<1>(a.add(lda * 13), ap.add(13));
        copy_packed::<1>(a.add(lda * 14), ap.add(14));
        copy_packed::<1>(a.add(lda * 15), ap.add(15));
        copy_packed::<1>(a.add(lda * 16), ap.add(16));
        copy_packed::<1>(a.add(lda * 17), ap.add(17));
        copy_packed::<1>(a.add(lda * 18), ap.add(18));
        copy_packed::<1>(a.add(lda * 19), ap.add(19));
        copy_packed::<1>(a.add(lda * 20), ap.add(20));
        copy_packed::<1>(a.add(lda * 21), ap.add(21));
        copy_packed::<1>(a.add(lda * 22), ap.add(22));
        copy_packed::<1>(a.add(lda * 23), ap.add(23));

        ap = ap.add(24);
        a = a.add(1);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx16_v1(
    k_iter: usize,
    k_left: usize,
    a: *const TA,
    lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        pack_t::<16>(a, lda, ap);

        pack_t::<16>(a.add(8 * lda), lda, ap.add(8));

        ap = ap.add(128);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<1>(a, ap);
        copy_packed::<1>(a.add(lda), ap.add(1));
        copy_packed::<1>(a.add(lda * 2), ap.add(2));
        copy_packed::<1>(a.add(lda * 3), ap.add(3));
        copy_packed::<1>(a.add(lda * 4), ap.add(4));
        copy_packed::<1>(a.add(lda * 5), ap.add(5));
        copy_packed::<1>(a.add(lda * 6), ap.add(6));
        copy_packed::<1>(a.add(lda * 7), ap.add(7));
        copy_packed::<1>(a.add(lda * 8), ap.add(8));
        copy_packed::<1>(a.add(lda * 9), ap.add(9));
        copy_packed::<1>(a.add(lda * 10), ap.add(10));
        copy_packed::<1>(a.add(lda * 11), ap.add(11));
        copy_packed::<1>(a.add(lda * 12), ap.add(12));
        copy_packed::<1>(a.add(lda * 13), ap.add(13));
        copy_packed::<1>(a.add(lda * 14), ap.add(14));
        copy_packed::<1>(a.add(lda * 15), ap.add(15));

        ap = ap.add(16);
        a = a.add(1);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx6_v1(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 6;
    const M1: usize = 4;
    const M2: usize = 2;
    while k_i < k_iter {
        // let a0 = _mm256_loadu_ps(b);
        // let a1 = _mm256_loadu_ps(b.add(ldb));
        // let a2 = _mm256_loadu_ps(b.add(ldb*2));
        // let a3 = _mm256_loadu_ps(b.add(ldb*3));

        // // transpose
        // let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        // let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
        // let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
        // let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

        // let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        // let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        // storeu_ps::<M1>(x0, bp);
        // storeu_ps::<M1>(x0_h, bp.add(M*4));

        // let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        // let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        // storeu_ps::<M1>(x1, bp.add(M));
        // storeu_ps::<M1>(x1_h, bp.add(M+M*4));

        // let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        // let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        // storeu_ps::<M1>(x2, bp.add(2*M));
        // storeu_ps::<M1>(x2_h, bp.add(2*M+M*4));

        // let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        // let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        // storeu_ps::<M1>(x3, bp.add(3*M));
        // storeu_ps::<M1>(x3_h, bp.add(3*M+M*4));

        // let a0 = _mm256_loadu_ps(b.add(ldb*4));
        // let a1 = _mm256_loadu_ps(b.add(ldb*5));

        // // transpose
        // let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        // let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));

        // let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        // let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        // storeu_ps::<M2>(x0, bp.add(4));
        // storeu_ps::<M2>(x0_h, bp.add(M*4+4));

        // let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        // let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        // storeu_ps::<M2>(x1, bp.add(M+4));
        // storeu_ps::<M2>(x1_h, bp.add(M+M*4+4));

        // let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        // let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        // storeu_ps::<M2>(x2, bp.add(2*M+4));
        // storeu_ps::<M2>(x2_h, bp.add(2*M+M*4+4));

        // let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        // let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        // storeu_ps::<M2>(x3, bp.add(3*M+4));
        // storeu_ps::<M2>(x3_h, bp.add(3*M+M*4+4));

        b = b.add(8);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        copy_packed::<1>(b.add(ldb * 2), bp.add(2));
        copy_packed::<1>(b.add(ldb * 3), bp.add(3));
        copy_packed::<1>(b.add(ldb * 4), bp.add(4));
        copy_packed::<1>(b.add(ldb * 5), bp.add(5));
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx5_v1(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 5;
    const M1: usize = 4;
    const M2: usize = 1;
    while k_i < k_iter {
        // let a0 = _mm256_loadu_ps(b);
        // let a1 = _mm256_loadu_ps(b.add(ldb));
        // let a2 = _mm256_loadu_ps(b.add(ldb*2));
        // let a3 = _mm256_loadu_ps(b.add(ldb*3));

        // // transpose
        // let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        // let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
        // let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
        // let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

        // let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        // let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        // storeu_ps::<M1>(x0, bp);
        // storeu_ps::<M1>(x0_h, bp.add(M*4));

        // let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        // let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        // storeu_ps::<M1>(x1, bp.add(M));
        // storeu_ps::<M1>(x1_h, bp.add(M+M*4));

        // let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        // let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        // storeu_ps::<M1>(x2, bp.add(2*M));
        // storeu_ps::<M1>(x2_h, bp.add(2*M+M*4));

        // let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        // let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        // storeu_ps::<M1>(x3, bp.add(3*M));
        // storeu_ps::<M1>(x3_h, bp.add(3*M+M*4));

        // let a0 = _mm256_loadu_ps(b.add(ldb*4));

        // // transpose
        // let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        // let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));

        // let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        // let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        // storeu_ps::<M2>(x0, bp.add(4));
        // storeu_ps::<M2>(x0_h, bp.add(M*4+4));

        // let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        // let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        // storeu_ps::<M2>(x1, bp.add(M+4));
        // storeu_ps::<M2>(x1_h, bp.add(M+M*4+4));

        // let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        // let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        // storeu_ps::<M2>(x2, bp.add(2*M+4));
        // storeu_ps::<M2>(x2_h, bp.add(2*M+M*4+4));

        // let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        // let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        // storeu_ps::<M2>(x3, bp.add(3*M+4));
        // storeu_ps::<M2>(x3_h, bp.add(3*M+M*4+4));

        b = b.add(8);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        copy_packed::<1>(b.add(ldb * 2), bp.add(2));
        copy_packed::<1>(b.add(ldb * 3), bp.add(3));
        copy_packed::<1>(b.add(ldb * 4), bp.add(4));
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx4_v1(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 4;

    while k_i < k_iter {
        // let a0 = _mm256_loadu_ps(b);
        // let a1 = _mm256_loadu_ps(b.add(ldb));
        // let a2 = _mm256_loadu_ps(b.add(ldb*2));
        // let a3 = _mm256_loadu_ps(b.add(ldb*3));

        // // transpose
        // let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        // let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
        // let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
        // let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

        // let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        // let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        // storeu_ps::<M>(x0, bp);
        // storeu_ps::<M>(x0_h, bp.add(M*4));

        // let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        // let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        // storeu_ps::<M>(x1, bp.add(M));
        // storeu_ps::<M>(x1_h, bp.add(M+M*4));

        // let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        // let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        // storeu_ps::<M>(x2, bp.add(2*M));
        // storeu_ps::<M>(x2_h, bp.add(2*M+M*4));

        // let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        // let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        // storeu_ps::<M>(x3, bp.add(3*M));
        // storeu_ps::<M>(x3_h, bp.add(3*M+M*4));

        b = b.add(8);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        copy_packed::<1>(b.add(ldb * 2), bp.add(2));
        copy_packed::<1>(b.add(ldb * 3), bp.add(3));
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx3_v1(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 3;
    // let a3 = _mm256_setzero_ps();
    while k_i < k_iter {
        // let a0 = _mm256_loadu_ps(b);
        // let a1 = _mm256_loadu_ps(b.add(ldb));
        // let a2 = _mm256_loadu_ps(b.add(ldb*2));

        // // transpose
        // let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        // let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
        // let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
        // let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

        // let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        // let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        // storeu_ps::<M>(x0, bp);
        // storeu_ps::<M>(x0_h, bp.add(M*4));

        // let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        // let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        // storeu_ps::<M>(x1, bp.add(M));
        // storeu_ps::<M>(x1_h, bp.add(M+M*4));

        // let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        // let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        // storeu_ps::<M>(x2, bp.add(2*M));
        // storeu_ps::<M>(x2_h, bp.add(2*M+M*4));

        // let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        // let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        // storeu_ps::<M>(x3, bp.add(3*M));
        // storeu_ps::<M>(x3_h, bp.add(3*M+M*4));

        b = b.add(8);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        copy_packed::<1>(b.add(ldb * 2), bp.add(2));
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx2_v1(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    // let t2 = _mm256_setzero_pd();
    // let t3 = _mm256_setzero_pd();
    const M: usize = 2;
    while k_i < k_iter {
        // let a0 = _mm256_loadu_ps(b);
        // let a1 = _mm256_loadu_ps(b.add(ldb));

        // // transpose
        // let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        // let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));

        // let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        // let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        // storeu_ps::<M>(x0, bp);
        // storeu_ps::<M>(x0_h, bp.add(M*4));

        // let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        // let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        // storeu_ps::<M>(x1, bp.add(M));
        // storeu_ps::<M>(x1_h, bp.add(M+M*4));

        // let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        // let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        // storeu_ps::<M>(x2, bp.add(2*M));
        // storeu_ps::<M>(x2_h, bp.add(2*M+M*4));

        // let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        // let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        // storeu_ps::<M>(x3, bp.add(3*M));
        // storeu_ps::<M>(x3_h, bp.add(3*M+M*4));

        b = b.add(8);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn pack_kx1_v1(
    k_iter: usize,
    k_left: usize,
    b: *const TB,
    _ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    // let t2 = _mm256_setzero_pd();
    // let t3 = _mm256_setzero_pd();
    // let a1 = _mm256_setzero_ps();
    const M: usize = 1;
    while k_i < k_iter {
        // let a0 = _mm256_loadu_ps(b);

        // // transpose
        // let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        // let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));

        // let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        // let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        // storeu_ps::<M>(x0, bp);
        // storeu_ps::<M>(x0_h, bp.add(M*4));

        // let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        // let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        // storeu_ps::<M>(x1, bp.add(M));
        // storeu_ps::<M>(x1_h, bp.add(M+M*4));

        // let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        // let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        // storeu_ps::<M>(x2, bp.add(2*M));
        // storeu_ps::<M>(x2_h, bp.add(2*M+M*4));

        // let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        // let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        // storeu_ps::<M>(x3, bp.add(3*M));
        // storeu_ps::<M>(x3_h, bp.add(3*M+M*4));

        b = b.add(8);
        bp = bp.add(M * 8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<1>(b, bp);
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}

macro_rules! def_packb {
   ($nr:tt) => {
       seq!(NL in 1..$nr {
           paste! {
            #[target_feature(enable = "neon")]
            pub(crate) unsafe fn [<packb_panel_$nr>](
                   n: usize, k: usize,
                   b: *const TB, b_rs: usize, b_cs: usize,
                   bp: *mut TB,
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
                        //    [<pack_kx$nr _v0>](k_iter, k_left, b, ldb, bp);
                        pack_k_v0::<NR, NR>(k_iter, k_left, b, ldb, bp);
                           n_idx += NR;
                           bp = bp.add(k*NR);
                           b = b.add(NR);
                       }
                       let n_left = n - n_idx;
                       #(
                           if n_left == NL {
                            //    [<pack_kx~NL _v0>](k_iter, k_left, b, ldb, bp);
                            pack_k_v0::<NL, NL>(k_iter, k_left, b, ldb, bp);
                               return;
                           }
                       )*
                   } else if b_cs == 1 {
                       let ldb = b_rs;
                       while n_idx + NR_LAST_STEP <= n {
                        //    [<pack_kx$nr _v1>](k_iter, k_left, b, ldb, bp);
                        pack_scalar_k::<NR>(
                            NR, k,
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
                            //    [<pack_kx~NL _v1>](k_iter, k_left, b, ldb, bp);
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
def_packb!(6);

macro_rules! mul8 {
    (24) => {
        24
    };
    (23) => {
        24
    };
    (22) => {
        24
    };
    (21) => {
        24
    };
    (20) => {
        24
    };
    (19) => {
        24
    };
    (18) => {
        24
    };
    (17) => {
        24
    };
    (16) => {
        16
    };
    (15) => {
        16
    };
    (14) => {
        16
    };
    (13) => {
        16
    };
    (12) => {
        16
    };
    (11) => {
        16
    };
    (10) => {
        16
    };
    (9) => {
        16
    };
    (8) => {
        8
    };
    (7) => {
        8
    };
    (6) => {
        8
    };
    (5) => {
        8
    };
    (4) => {
        8
    };
    (3) => {
        8
    };
    (2) => {
        8
    };
    (1) => {
        8
    };
}
macro_rules! mul8_2 {
    (24) => {
        16
    };
    (16) => {
        8
    };
    (8) => {
        0
    };
    (4) => {
        0
    };
    (2) => {
        0
    };
    (1) => {
        0
    };
}
macro_rules! def_packa {
    ($mr:tt, $($mr_left:tt),*) => {
        paste! {
            #[target_feature(enable = "neon")]
            pub(crate) unsafe fn [<packa_panel_$mr>](
                m_left: usize, k: usize,
                a: *const TA, a_rs: usize, a_cs: usize,
                ap: *mut TA,
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
                        // [<pack_kx$mr _v0>](k_iter, k_left, a, lda, ap);
                        pack_k_v0::<MR, MR>(k_iter, k_left, a, lda, ap);
                        m_idx += MR;
                        ap = ap.add(k * MR);
                        a = a.add(MR);
                    }
                    let m_left = m_left - m_idx;
                    seq!(mr_left in 1..$mr {
                        if m_left == mr_left {
                            pack_k_v0::<mr_left, {mul8!(mr_left)}>(k_iter, k_left, a, lda, ap);
                            return;
                        }
                    });

                } else if a_cs == 1 {
                    let lda = a_rs;
                    let k_iter = k / 8;
                    let k_left = k % 8;
                    while m_idx + MR_LAST_STEP <= m_left {
                        // [<pack_kx$mr _v1>](k_iter, k_left, a, lda, ap);
                        pack_scalar_k::<MR>(
                            MR, k,
                            a, a_rs, a_cs,
                            ap
                        );
                        m_idx += MR;
                        ap = ap.add(k * MR);
                        a = a.add(MR*lda);
                    }
                    let m_left = m_left - m_idx;
                    $(
                        if m_left > mul8_2!($mr_left) {
                            pack_scalar_k::<$mr_left>(
                                m_left, k,
                                a, a_rs, a_cs,
                                ap
                            );
                            return;
                        }
                    )*
                }
            }
        }
    };
}

def_packa!(24, 24, 16, 8);
def_packa!(16, 16, 8);
