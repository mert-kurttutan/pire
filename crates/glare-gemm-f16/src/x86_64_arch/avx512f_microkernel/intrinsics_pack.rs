use seq_macro::seq;
use std::ptr::copy_nonoverlapping;
use crate::{TA,TB};

use half::f16;


use paste::paste;

use std::arch::x86_64::*;


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
    if M == 32 {
        let a = a as *const u16;
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a as *const __m128i));
        _mm256_storeu_ps(b, x);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(8) as *const __m128i));
        _mm256_storeu_ps(b.add(8), x);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(16) as *const __m128i));
        _mm256_storeu_ps(b.add(16), x);
        let x = _mm256_cvtph_ps(_mm_loadu_si128(a.add(24) as *const __m128i));
        _mm256_storeu_ps(b.add(24), x);
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
pub(crate) unsafe fn pack_kx32_v1(
    k_iter: usize, k_left: usize,
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 32;
    while k_i < k_iter {
        pack_t::<MR>(a, lda, ap);
        pack_t::<MR>(a.add(8*lda), lda, ap.add(8));
        pack_t::<MR>(a.add(16*lda), lda, ap.add(16));
        pack_t::<MR>(a.add(24*lda), lda, ap.add(24));

        ap = ap.add(MR*8);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        seq!(i in 0..32 {
            copy_packed::<1>(a.add(i*lda), ap.add(i));
        });

        ap = ap.add(MR);
        a = a.add(1);
        k_i += 1;
    }
}


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx12_v1(
    k_iter: usize, k_left: usize,
    a: *const f16, lda: usize,
    ap: *mut f32,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 6;
    while k_i < k_iter {
        pack_t::<4>(a, lda, ap);
        pack_t::<2>(a.add(4*lda), lda, ap.add(4));

        ap = ap.add(MR*8);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        seq!(i in 0..12 {
            copy_packed::<1>(a.add(i*lda), ap.add(i));
        });

        ap = ap.add(MR);
        a = a.add(1);
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
            #[target_feature(enable = "avx")]
            pub(crate) unsafe fn [<packb_panel_$nr>](
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


def_packb!(8);
def_packb!(12);


macro_rules! mul8 {
    (48) => { 48 };
    (47) => { 48 };
    (46) => { 48 };
    (45) => { 48 };
    (44) => { 48 };
    (43) => { 48 };
    (42) => { 48 };
    (41) => { 48 };
    (40) => { 48 };
    (39) => { 48 };
    (38) => { 48 };
    (37) => { 48 };
    (36) => { 48 };
    (35) => { 48 };
    (34) => { 48 };
    (33) => { 48 };
    (32) => { 32 };
    (31) => { 32 };
    (30) => { 32 };
    (29) => { 32 };
    (28) => { 32 };
    (27) => { 32 };
    (26) => { 32 };
    (25) => { 32 };
    (24) => { 32 };
    (23) => { 32 };
    (22) => { 32 };
    (21) => { 32 };
    (20) => { 32 };
    (19) => { 32 };
    (18) => { 32 };
    (17) => { 32 };
    (16) => { 16 };
    (15) => { 16 };
    (14) => { 16 };
    (13) => { 16 };
    (12) => { 16 };
    (11) => { 16 };
    (10) => { 16 };
    (9) => { 16 };
    (8) => { 16 };
    (7) => { 16 };
    (6) => { 16 };
    (5) => { 16 };
    (4) => { 16 };
    (3) => { 16 };
    (2) => { 16 };
    (1) => { 16 };
}
macro_rules! mul8_2 {
    (48) => { 32 };
    (32) => { 16 };
    (16) => { 0 };
    (4) => { 0 };
    (2) => { 0 };
    (1) => { 0 };
}
macro_rules! def_packa {
    ($mr:tt, $($mr_left:tt),*) => {
        paste! {
            #[target_feature(enable = "avx")]
            pub(crate) unsafe fn [<packa_panel_$mr>](
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
                            pack_k_v0::<mr_left, {mul8!(mr_left)}>(k_iter, k_left, a, lda, ap);
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

def_packa!(48, 48, 32, 16);
def_packa!(32, 32, 16);
