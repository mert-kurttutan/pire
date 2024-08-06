use seq_macro::seq;
use crate::{TA,TB};

use paste::paste;

use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_t4<const MR: usize>(
    b: *const TA, ldb: usize,
    bp: *mut TB,
) {
    let a0 = _mm256_loadu_pd(b);
    let a1 = _mm256_loadu_pd(b.add(ldb));
    let a2 = _mm256_loadu_pd(b.add(ldb*2));
    let a3 = _mm256_loadu_pd(b.add(ldb*3));

    // transpose
    let t0 = _mm256_unpacklo_pd(a0, a1);
    let t1 = _mm256_unpackhi_pd(a0, a1);
    let t2 = _mm256_unpacklo_pd(a2, a3);
    let t3 = _mm256_unpackhi_pd(a2, a3);

    let x0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    let x0_h = _mm256_permute2f128_pd(t0, t2, 0x31);

    _mm256_storeu_pd(bp, x0);
    _mm256_store_pd(bp.add(MR*2), x0_h);

    let x1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    let x1_h = _mm256_permute2f128_pd(t1, t3, 0x31);

    _mm256_storeu_pd(bp.add(MR), x1);
    _mm256_store_pd(bp.add(MR*3), x1_h);

    // k = 4
    let a0 = _mm256_loadu_pd(b.add(4));
    let a1 = _mm256_loadu_pd(b.add(ldb+4));
    let a2 = _mm256_loadu_pd(b.add(ldb*2+4));
    let a3 = _mm256_loadu_pd(b.add(ldb*3+4));

    // transpose
    let t0 = _mm256_unpacklo_pd(a0, a1);
    let t1 = _mm256_unpackhi_pd(a0, a1);
    let t2 = _mm256_unpacklo_pd(a2, a3);
    let t3 = _mm256_unpackhi_pd(a2, a3);

    let x0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    let x0_h = _mm256_permute2f128_pd(t0, t2, 0x31);

    _mm256_storeu_pd(bp.add(MR*4), x0);
    _mm256_store_pd(bp.add(MR*6), x0_h);

    let x1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    let x1_h = _mm256_permute2f128_pd(t1, t3, 0x31);

    _mm256_storeu_pd(bp.add(MR*5), x1);
    _mm256_store_pd(bp.add(MR*7), x1_h);

}


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_scalar_k<const MR: usize>(
    m_left: usize, k: usize,
    a: *const TA, a_rs: usize, a_cs: usize,
    ap: *mut TA,
) {
    for i in 0..m_left  {
        for j in 0..k {
            *ap.add(j*MR+i) = *a.add(j*a_cs + i*a_rs);
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
pub(crate) unsafe fn copy_packed<const M: usize>(a: *const f64, b: *mut f64) {
    std::ptr::copy_nonoverlapping(a, b, M);
}


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_k_v0<const M: usize, const MR: usize>(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
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
pub(crate) unsafe fn pack_kx24_v0(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 24;
    while k_i < k_iter {
        // use vector intrinscs
        seq!(i in 0..8 {
            let a0 = _mm256_loadu_pd(a.add(lda*i));
            let a1 = _mm256_loadu_pd(a.add(lda*i+4));
            let a2 = _mm256_loadu_pd(a.add(lda*i+8));
            let a3 = _mm256_loadu_pd(a.add(lda*i+12));
            let a4 = _mm256_loadu_pd(a.add(lda*i+16));
            let a5 = _mm256_loadu_pd(a.add(lda*i+20));
            _mm256_store_pd(ap.add(i*MR), a0);
            _mm256_store_pd(ap.add(i*MR+4), a1);
            _mm256_store_pd(ap.add(i*MR+8), a2);
            _mm256_store_pd(ap.add(i*MR+12), a3);
            _mm256_store_pd(ap.add(i*MR+16), a4);
            _mm256_store_pd(ap.add(i*MR+20), a5);
        });

        ap = ap.add(MR*8);
        a = a.add(8*lda);

        k_i += 1;
    }
    k_i = 0;
    while k_i < k_left {
        let a0 = _mm256_loadu_pd(a);
        let a1 = _mm256_loadu_pd(a.add(4));
        let a2 = _mm256_loadu_pd(a.add(8));
        let a3 = _mm256_loadu_pd(a.add(12));
        let a4 = _mm256_loadu_pd(a.add(16));
        let a5 = _mm256_loadu_pd(a.add(20));
        _mm256_store_pd(ap, a0);
        _mm256_store_pd(ap.add(4), a1);
        _mm256_store_pd(ap.add(8), a2);
        _mm256_store_pd(ap.add(12), a3);
        _mm256_store_pd(ap.add(16), a4);
        _mm256_store_pd(ap.add(20), a5);

        ap = ap.add(MR);
        a = a.add(lda);
        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx16_v0(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 16;
    while k_i < k_iter {
        // use vector intrinscs
        seq!(i in 0..8 {
            let a0 = _mm256_loadu_pd(a.add(lda*i));
            let a1 = _mm256_loadu_pd(a.add(lda*i+8));
            let a2 = _mm256_loadu_pd(a.add(lda*i+16));
            let a3 = _mm256_loadu_pd(a.add(lda*i+24));
            _mm256_store_pd(ap.add(i*MR), a0);
            _mm256_store_pd(ap.add(i*MR+8), a1);
            _mm256_store_pd(ap.add(i*MR+16), a2);
            _mm256_store_pd(ap.add(i*MR+24), a3);
        });

        ap = ap.add(MR*8);
        a = a.add(8*lda);

        k_i += 1;
    }
    k_i = 0;
    while k_i < k_left {
        let a0 = _mm256_loadu_pd(a);
        let a1 = _mm256_loadu_pd(a.add(8));
        let a2 = _mm256_loadu_pd(a.add(16));
        let a3 = _mm256_loadu_pd(a.add(24));
        _mm256_store_pd(ap, a0);
        _mm256_store_pd(ap.add(8), a1);
        _mm256_store_pd(ap.add(16), a2);
        _mm256_store_pd(ap.add(24), a3);

        ap = ap.add(MR);
        a = a.add(lda);
        k_i += 1;
    }
}


#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx24_v1(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 24;
    while k_i < k_iter {
        pack_t4::<MR>(a, lda, ap);
        pack_t4::<MR>(a.add(4*lda), lda, ap.add(4));
        pack_t4::<MR>(a.add(8*lda), lda, ap.add(8));
        pack_t4::<MR>(a.add(12*lda), lda, ap.add(12));
        pack_t4::<MR>(a.add(16*lda), lda, ap.add(16));
        pack_t4::<MR>(a.add(20*lda), lda, ap.add(20));

        ap = ap.add(MR*8);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        seq!(i in 0..24 {
            copy_packed::<1>(a.add(lda*i), ap.add(i));
        });

        ap = ap.add(MR);
        a = a.add(1);
        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx16_v1(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 16;
    while k_i < k_iter {
        pack_t4::<MR>(a, lda, ap);
        pack_t4::<MR>(a.add(4*lda), lda, ap.add(4));
        pack_t4::<MR>(a.add(8*lda), lda, ap.add(8));
        pack_t4::<MR>(a.add(12*lda), lda, ap.add(12));

        ap = ap.add(MR*8);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        seq!(i in 0..16 {
            copy_packed::<1>(a.add(lda*i), ap.add(i));
        });

        ap = ap.add(MR);
        a = a.add(1);
        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx12_v1(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 12;
    while k_i < k_iter {
        pack_t4::<MR>(a, lda, ap);
        pack_t4::<MR>(a.add(4*lda), lda, ap.add(4));
        pack_t4::<MR>(a.add(8*lda), lda, ap.add(8));

        ap = ap.add(MR*8);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        seq!(i in 0..12 {
            copy_packed::<1>(a.add(lda*i), ap.add(i));
        });

        ap = ap.add(MR);
        a = a.add(1);
        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx8_v1(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 8;
    while k_i < k_iter {
        pack_t4::<MR>(a, lda, ap);
        pack_t4::<MR>(a.add(4*lda), lda, ap.add(4));

        ap = ap.add(MR*8);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        seq!(i in 0..8 {
            copy_packed::<1>(a.add(lda*i), ap.add(i));
        });

        ap = ap.add(MR);
        a = a.add(1);
        k_i += 1;
    }
}

macro_rules! def_packb {
    ($nr:tt) => {
         paste! {
         #[target_feature(enable = "avx")]
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
                         pack_k_v0::<NR,NR>(k_iter, k_left, b, ldb, bp);
                         n_idx += NR;
                         bp = bp.add(k*NR);
                         b = b.add(NR);
                     }
                     let n_left = n - n_idx;
                     seq!(NL in 1..$nr {
                         if n_left == NL {
                             pack_k_v0::<NL,NL>(k_iter, k_left, b, ldb, bp);
                             return;
                         }
                     });
                 } else if b_cs == 1 {
                     let ldb = b_rs;
                     while n_idx + NR_LAST_STEP <= n {
                         [<pack_kx$nr _v1>](k_iter, k_left, b, ldb, bp);
                         n_idx += NR;
                         bp = bp.add(k*NR);
                         b =  b.add(NR*ldb);
                     }
                     let n_left = n - n_idx;
                     seq!(NL in 1..$nr {
                         if n_left == NL {
                             pack_k_v1::<NL,NL>(k, b, ldb, bp);
                             return;
                         }
                     });
                 }
             }   
         }
    };
 }
 
 
 def_packb!(8);
 def_packb!(12);


macro_rules! def_packa {
    ($mr:tt) => {
        paste! {
            #[target_feature(enable = "avx")]
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
                        [<pack_kx$mr _v0>](k_iter, k_left, a, lda, ap);
                        m_idx += MR;
                        ap = ap.add(k * MR);
                        a = a.add(MR);
                    }
                    let m_left = m_left - m_idx;
                    seq!(mr_left in 1..$mr {
                        if m_left == mr_left {
                            pack_k_v0::<mr_left, {(mr_left+7)/ 8 * 8}>(k_iter, k_left, a, lda, ap);
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
                            pack_scalar_k::<{(mr_left+7)/ 8 * 8}>(
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

def_packa!(24);
def_packa!(16);
