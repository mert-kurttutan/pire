use seq_macro::seq;
use std::ptr::copy_nonoverlapping;
use crate::{TA,TB};


use paste::paste;

use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_t<const MR: usize>(
    a: *const TA, lda: usize,
    ap: *mut TB,
) {
    let a0 = _mm256_loadu_ps(a);
    let a1 = _mm256_loadu_ps(a.add(lda));
    let a2 = _mm256_loadu_ps(a.add(lda*2));
    let a3 = _mm256_loadu_ps(a.add(lda*3));

    // transpose
    let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
    let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
    let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
    let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

    let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
    let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
    let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
    let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));

    let a0 = _mm256_loadu_ps(a.add(lda*4));
    let a1 = _mm256_loadu_ps(a.add(lda*5));
    let a2 = _mm256_loadu_ps(a.add(lda*6));
    let a3 = _mm256_loadu_ps(a.add(lda*7));

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


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_scalar_k(
    m_left: usize, k: usize,
    a: *const TA, a_rs: usize, a_cs: usize,
    ap: *mut TA, vs: usize
) {
    let mr = (m_left + vs - 1) / vs * vs;
    for i in 0..m_left  {
        for j in 0..k {
            *ap.add(j*mr+i) = *a.add(j*a_cs + i*a_rs);
        }
    }
}


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_k_v1<const M: usize, const MR: usize>(
    k: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    for i in 0..M  {
        for j in 0..k {
            *ap.add(j*MR+i) = *a.add(j + i*lda);
        }
    }
}


#[target_feature(enable = "avx")]
pub(crate) unsafe fn storeu_ps<const M: usize>(
    src: __m256, dst: *mut f32
) {
    let mut temp_arr = [0.0; 8];
    _mm256_storeu_ps(temp_arr.as_mut_ptr(), src);
    copy_nonoverlapping(temp_arr.as_ptr(), dst, M);
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn copy_packed<const M: usize>(a: *const f32, b: *mut f32) {
    std::ptr::copy_nonoverlapping(a, b, M);
}


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_k_v0<const M: usize, const MR: usize>(
    k: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
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
pub(crate) unsafe fn pack_kx24_v0(
    k: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    const MR: usize = 24;
    while k_i < k8 {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        seq!(i in 0..8 {
            let a0 = _mm256_loadu_ps(a.add(lda*i));
            let a1 = _mm256_loadu_ps(a.add(lda*i+8));
            let a2 = _mm256_loadu_ps(a.add(lda*i+16));
            _mm256_store_ps(ap.add(i*MR), a0);
            _mm256_store_ps(ap.add(i*MR+8), a1);
            _mm256_store_ps(ap.add(i*MR+16), a2);
        });

        k_i += 8;
    }

    while k_i < k {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        let a0 = _mm256_loadu_ps(a);
        let a1 = _mm256_loadu_ps(a.add(8));
        let a2 = _mm256_loadu_ps(a.add(16));
        _mm256_store_ps(ap, a0);
        _mm256_store_ps(ap.add(8), a1);
        _mm256_store_ps(ap.add(16), a2);

        k_i += 1;
    }
}


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx24_v1(
    k: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
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
            *ap.add(i) = *a.add(i*lda);
        });

        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx4_v1(
    k: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let k8 = k / 8 * 8;
    let b0 = b;
    let bp0 = bp;

    let mut k_i = 0;
    const M: usize = 4;

    while k_i < k8 {
        let b = b0.add(k_i);
        let bp = bp0.add(k_i*M);
        let a0 = _mm256_loadu_ps(b);
        let a1 = _mm256_loadu_ps(b.add(ldb));
        let a2 = _mm256_loadu_ps(b.add(ldb*2));
        let a3 = _mm256_loadu_ps(b.add(ldb*3));

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


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx48_v0(
    k: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    const MR: usize = 48;
    while k_i < k8 {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        // use vector intrinscs
        seq!(i in 0..8 {
            let a0 = _mm256_loadu_ps(a.add(lda*i));
            let a1 = _mm256_loadu_ps(a.add(lda*i+8));
            let a2 = _mm256_loadu_ps(a.add(lda*i+16));
            let a3 = _mm256_loadu_ps(a.add(lda*i+24));
            let a4 = _mm256_loadu_ps(a.add(lda*i+32));
            let a5 = _mm256_loadu_ps(a.add(lda*i+40));
            _mm256_store_ps(ap.add(i*MR), a0);
            _mm256_store_ps(ap.add(i*MR+8), a1);
            _mm256_store_ps(ap.add(i*MR+16), a2);
            _mm256_store_ps(ap.add(i*MR+24), a3);
            _mm256_store_ps(ap.add(i*MR+32), a4);
            _mm256_store_ps(ap.add(i*MR+40), a5);
        });

        k_i += 8;
    }

    while k_i < k {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        let a0 = _mm256_loadu_ps(a);
        let a1 = _mm256_loadu_ps(a.add(8));
        let a2 = _mm256_loadu_ps(a.add(16));
        let a3 = _mm256_loadu_ps(a.add(24));
        let a4 = _mm256_loadu_ps(a.add(32));
        let a5 = _mm256_loadu_ps(a.add(40));
        _mm256_store_ps(ap, a0);
        _mm256_store_ps(ap.add(8), a1);
        _mm256_store_ps(ap.add(16), a2);
        _mm256_store_ps(ap.add(24), a3);
        _mm256_store_ps(ap.add(32), a4);
        _mm256_store_ps(ap.add(40), a5);

        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx48_v1(
    k: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
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
            *ap.add(i) = *a.add(i*lda);
        });

        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx8_v1(
    k: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
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
            *ap.add(i) = *a.add(i*lda);
        });
        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx16_v0(
    k: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    const MR: usize = 16;
    while k_i < k8 {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        seq!(i in 0..8 {
            let a0 = _mm256_loadu_ps(a.add(lda*i));
            let a1 = _mm256_loadu_ps(a.add(lda*i+8));
            _mm256_store_ps(ap.add(i*MR), a0);
            _mm256_store_ps(ap.add(i*MR+8), a1);
        });

        k_i += 8;
    }

    while k_i < k {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        let a0 = _mm256_loadu_ps(a);
        let a1 = _mm256_loadu_ps(a.add(8));
        _mm256_store_ps(ap, a0);
        _mm256_store_ps(ap.add(8), a1);

        k_i += 1;
    }
}


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx16_v1(
    k: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
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
            *ap.add(i) = *a.add(i*lda);
        });

        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx6_v1(
    k: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    let b0 = b;
    let bp0 = bp;
    const M: usize = 6;
    const M1: usize = 4;
    const M2: usize = 2;
    while k_i < k8 {
        let b = b0.add(k_i);
        let bp = bp0.add(k_i*M);
        let a0 = _mm256_loadu_ps(b);
        let a1 = _mm256_loadu_ps(b.add(ldb));
        let a2 = _mm256_loadu_ps(b.add(ldb*2));
        let a3 = _mm256_loadu_ps(b.add(ldb*3));

        // transpose
        let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
        let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
        let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

        let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        storeu_ps::<M1>(x0, bp);
        storeu_ps::<M1>(x0_h, bp.add(M*4));

        let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        storeu_ps::<M1>(x1, bp.add(M));
        storeu_ps::<M1>(x1_h, bp.add(M+M*4));

        let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        storeu_ps::<M1>(x2, bp.add(2*M));
        storeu_ps::<M1>(x2_h, bp.add(2*M+M*4));

        let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        storeu_ps::<M1>(x3, bp.add(3*M));
        storeu_ps::<M1>(x3_h, bp.add(3*M+M*4));


        let a0 = _mm256_loadu_ps(b.add(ldb*4));
        let a1 = _mm256_loadu_ps(b.add(ldb*5));

        // transpose
        let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));

        let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        storeu_ps::<M2>(x0, bp.add(4));
        storeu_ps::<M2>(x0_h, bp.add(M*4+4));

        let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        storeu_ps::<M2>(x1, bp.add(M+4));
        storeu_ps::<M2>(x1_h, bp.add(M+M*4+4));

        let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        storeu_ps::<M2>(x2, bp.add(2*M+4));
        storeu_ps::<M2>(x2_h, bp.add(2*M+M*4+4));

        let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        storeu_ps::<M2>(x3, bp.add(3*M+4));
        storeu_ps::<M2>(x3_h, bp.add(3*M+M*4+4));

        k_i += 8;
    }

    while k_i < k {
        let b = b0.add(k_i);
        let bp = bp0.add(k_i*M);
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        copy_packed::<1>(b.add(ldb*2), bp.add(2));
        copy_packed::<1>(b.add(ldb*3), bp.add(3));
        copy_packed::<1>(b.add(ldb*4), bp.add(4));
        copy_packed::<1>(b.add(ldb*5), bp.add(5));
        k_i += 1;
    }
}


macro_rules! def_packb {
    ($nr:tt) => {
         paste! {
         // #[target_feature(enable = "avx")]
         pub(crate) unsafe fn [<packb_panel_ $nr>](
                 n: usize, k: usize,
                 b: *const TB, b_rs: usize, b_cs: usize,
                 bp: *mut TB,
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
                    seq!(NL in 1..$nr {
                        if n_left == NL {
                            let b = b0.add(n_idx);
                            let bp = bp0.add(n_idx*k);
                            pack_k_v0::<NL,NL>(k, b, ldb, bp);
                            return;
                        }
                    });
                 } else if b_cs == 1 {
                     let ldb = b_rs;
                    while n_idx < n_rounded {
                        let b = b0.add(n_idx*ldb);
                        let bp = bp0.add(n_idx*k);
                        [<pack_kx$nr _v1>](k, b, ldb, bp);
                        n_idx += NR;
                    }
                     let n_left = n - n_idx;
                    seq!(NL in 1..$nr {
                        if n_left == NL {
                            let b = b0.add(n_idx*ldb);
                            let bp = bp0.add(n_idx*k);
                            pack_k_v1::<NL,NL>(k, b, ldb, bp);
                            return;
                        }
                    });
                 }
             }   
         }
    };
}


def_packb!(4);
def_packb!(8);
def_packb!(6);

macro_rules! def_packa {
    ($mr:tt) => {
        paste! {
            // #[target_feature(enable = "avx")]
            pub(crate) unsafe fn [<packa_panel_ $mr>](
                m: usize, k: usize,
                a: *const TA, a_rs: usize, a_cs: usize,
                ap: *mut TA, vs: usize
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
                        [<pack_kx$mr _v0>](k, a, lda, ap);
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

def_packa!(24);
def_packa!(48);
def_packa!(16);
