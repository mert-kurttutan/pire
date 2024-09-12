use crate::{TA, TB};
use seq_macro::seq;
use std::ptr::copy_nonoverlapping;

use paste::paste;

use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_t4<const MR: usize>(b: *const TA, ldb: usize, bp: *mut TB) {
    let a0 = _mm256_loadu_pd(b);
    let a1 = _mm256_loadu_pd(b.add(ldb));
    let a2 = _mm256_loadu_pd(b.add(ldb * 2));
    let a3 = _mm256_loadu_pd(b.add(ldb * 3));

    // transpose
    let t0 = _mm256_unpacklo_pd(a0, a1);
    let t1 = _mm256_unpackhi_pd(a0, a1);
    let t2 = _mm256_unpacklo_pd(a2, a3);
    let t3 = _mm256_unpackhi_pd(a2, a3);

    let x0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    let x0_h = _mm256_permute2f128_pd(t0, t2, 0x31);

    storeu_ps::<4>(x0, bp);
    storeu_ps::<4>(x0_h, bp.add(MR * 2));

    let x1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    let x1_h = _mm256_permute2f128_pd(t1, t3, 0x31);

    storeu_ps::<4>(x1, bp.add(MR));
    storeu_ps::<4>(x1_h, bp.add(MR * 3));

    // k = 4
    let a0 = _mm256_loadu_pd(b.add(4));
    let a1 = _mm256_loadu_pd(b.add(ldb + 4));
    let a2 = _mm256_loadu_pd(b.add(ldb * 2 + 4));
    let a3 = _mm256_loadu_pd(b.add(ldb * 3 + 4));

    // transpose
    let t0 = _mm256_unpacklo_pd(a0, a1);
    let t1 = _mm256_unpackhi_pd(a0, a1);
    let t2 = _mm256_unpacklo_pd(a2, a3);
    let t3 = _mm256_unpackhi_pd(a2, a3);

    let x0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    let x0_h = _mm256_permute2f128_pd(t0, t2, 0x31);
    storeu_ps::<4>(x0, bp.add(MR * 4));
    storeu_ps::<4>(x0_h, bp.add(MR * 6));

    let x1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    let x1_h = _mm256_permute2f128_pd(t1, t3, 0x31);
    storeu_ps::<4>(x1, bp.add(MR * 5));
    storeu_ps::<4>(x1_h, bp.add(MR * 7));
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_t2<const MR: usize>(b: *const TA, ldb: usize, bp: *mut TB) {
    let mut a0 = [0f64; 8];
    let mut a1 = [0f64; 8];
    copy_nonoverlapping(b, a0.as_mut_ptr(), 8);
    copy_nonoverlapping(b.add(ldb), a1.as_mut_ptr(), 8);

    let mut ap0 = [0f64; 2];
    let mut ap1 = [0f64; 2];
    let mut ap2 = [0f64; 2];
    let mut ap3 = [0f64; 2];
    let mut ap4 = [0f64; 2];
    let mut ap5 = [0f64; 2];
    let mut ap6 = [0f64; 2];
    let mut ap7 = [0f64; 2];

    ap0[0] = a0[0];
    ap0[1] = a1[0];

    ap1[0] = a0[1];
    ap1[1] = a1[1];

    ap2[0] = a0[2];
    ap2[1] = a1[2];

    ap3[0] = a0[3];
    ap3[1] = a1[3];

    ap4[0] = a0[4];
    ap4[1] = a1[4];

    ap5[0] = a0[5];
    ap5[1] = a1[5];

    ap6[0] = a0[6];
    ap6[1] = a1[6];

    ap7[0] = a0[7];
    ap7[1] = a1[7];

    copy_nonoverlapping(ap0.as_ptr(), bp, 2);
    copy_nonoverlapping(ap1.as_ptr(), bp.add(MR), 2);
    copy_nonoverlapping(ap2.as_ptr(), bp.add(MR * 2), 2);
    copy_nonoverlapping(ap3.as_ptr(), bp.add(MR * 3), 2);
    copy_nonoverlapping(ap4.as_ptr(), bp.add(MR * 4), 2);
    copy_nonoverlapping(ap5.as_ptr(), bp.add(MR * 5), 2);
    copy_nonoverlapping(ap6.as_ptr(), bp.add(MR * 6), 2);
    copy_nonoverlapping(ap7.as_ptr(), bp.add(MR * 7), 2);
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_scalar_k(
    m_left: usize,
    k: usize,
    a: *const TA,
    a_rs: usize,
    a_cs: usize,
    ap: *mut TA,
    vs: usize,
) {
    let mr = (m_left + vs - 1) / vs * vs;
    for i in 0..m_left {
        for j in 0..k {
            *ap.add(j * mr + i) = *a.add(j * a_cs + i * a_rs);
        }
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_k_v1<const M: usize, const MR: usize>(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    for i in 0..M {
        for j in 0..k {
            *ap.add(j * MR + i) = *a.add(j + i * lda);
        }
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn storeu_ps<const M: usize>(src: __m256d, dst: *mut f64) {
    let mut temp_arr = [0.0; 4];
    _mm256_storeu_pd(temp_arr.as_mut_ptr(), src);
    copy_nonoverlapping(temp_arr.as_ptr(), dst, M);
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn copy_packed<const M: usize>(a: *const f64, b: *mut f64) {
    copy_nonoverlapping(a, b, M);
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_k_v0<const M: usize, const MR: usize>(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    let mut a = a;
    let mut ap = ap;
    let a0 = a;
    let ap0 = ap;
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    while k_i < k8 {
        a = a0.add(k_i * lda);
        ap = ap0.add(k_i * MR);
        copy_packed::<M>(a, ap);
        copy_packed::<M>(a.add(lda), ap.add(MR));
        copy_packed::<M>(a.add(lda * 2), ap.add(MR * 2));
        copy_packed::<M>(a.add(lda * 3), ap.add(MR * 3));
        copy_packed::<M>(a.add(lda * 4), ap.add(MR * 4));
        copy_packed::<M>(a.add(lda * 5), ap.add(MR * 5));
        copy_packed::<M>(a.add(lda * 6), ap.add(MR * 6));
        copy_packed::<M>(a.add(lda * 7), ap.add(MR * 7));

        k_i += 8;
    }

    while k_i < k {
        a = a0.add(k_i * lda);
        ap = ap0.add(k_i * MR);
        copy_packed::<M>(a, ap);

        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx12_v0(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    const MR: usize = 12;
    let mut a = a;
    let mut ap = ap;
    let a0 = a;
    let ap0 = ap;
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    while k_i < k8 {
        a = a0.add(k_i * lda);
        ap = ap0.add(k_i * MR);
        seq!(i in 0..8 {
            let a0 = _mm256_loadu_pd(a.add(lda*i));
            let a1 = _mm256_loadu_pd(a.add(lda*i+4));
            let a2 = _mm256_loadu_pd(a.add(lda*i+8));
            _mm256_store_pd(ap.add(i*MR), a0);
            _mm256_store_pd(ap.add(i*MR+4), a1);
            _mm256_store_pd(ap.add(i*MR+8), a2);
        });

        k_i += 8;
    }

    while k_i < k {
        a = a0.add(k_i * lda);
        ap = ap0.add(k_i * MR);
        let a0 = _mm256_loadu_pd(a);
        let a1 = _mm256_loadu_pd(a.add(4));
        let a2 = _mm256_loadu_pd(a.add(8));
        _mm256_store_pd(ap, a0);
        _mm256_store_pd(ap.add(4), a1);
        _mm256_store_pd(ap.add(8), a2);

        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx8_v0(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    const MR: usize = 8;
    let mut a = a;
    let mut ap = ap;
    let a0 = a;
    let ap0 = ap;
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    while k_i < k8 {
        a = a0.add(k_i * lda);
        ap = ap0.add(k_i * MR);
        seq!(i in 0..8 {
            let a0 = _mm256_loadu_pd(a.add(lda*i));
            let a1 = _mm256_loadu_pd(a.add(lda*i+4));
            _mm256_store_pd(ap.add(i*MR), a0);
            _mm256_store_pd(ap.add(i*MR+4), a1);
        });

        k_i += 8;
    }
    while k_i < k {
        a = a0.add(k_i * lda);
        ap = ap0.add(k_i * MR);
        let a0 = _mm256_loadu_pd(a);
        let a1 = _mm256_loadu_pd(a.add(4));
        _mm256_store_pd(ap, a0);
        _mm256_store_pd(ap.add(4), a1);

        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx24_v0(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    const MR: usize = 24;
    let mut a = a;
    let mut ap = ap;
    let a0 = a;
    let ap0 = ap;
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    while k_i < k8 {
        a = a0.add(k_i * lda);
        ap = ap0.add(k_i * MR);
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

        k_i += 8;
    }
    while k_i < k {
        a = a0.add(k_i * lda);
        ap = ap0.add(k_i * MR);
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

        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx24_v1(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    const MR: usize = 24;
    let mut a = a;
    let mut ap = ap;
    let a0 = a;
    let ap0 = ap;
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    while k_i < k8 {
        a = a0.add(k_i);
        ap = ap0.add(k_i * MR);
        pack_t4::<MR>(a, lda, ap);
        pack_t4::<MR>(a.add(4 * lda), lda, ap.add(4));
        pack_t4::<MR>(a.add(8 * lda), lda, ap.add(8));
        pack_t4::<MR>(a.add(12 * lda), lda, ap.add(12));
        pack_t4::<MR>(a.add(16 * lda), lda, ap.add(16));
        pack_t4::<MR>(a.add(20 * lda), lda, ap.add(20));

        k_i += 8;
    }

    while k_i < k {
        a = a0.add(k_i);
        ap = ap0.add(k_i * MR);
        seq!(i in 0..24 {
            copy_packed::<1>(a.add(lda*i), ap.add(i));
        });

        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx12_v1(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    const MR: usize = 12;
    let mut a = a;
    let mut ap = ap;
    let a0 = a;
    let ap0 = ap;
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    while k_i < k8 {
        a = a0.add(k_i);
        ap = ap0.add(k_i * MR);
        pack_t4::<MR>(a, lda, ap);
        pack_t4::<MR>(a.add(4 * lda), lda, ap.add(4));
        pack_t4::<MR>(a.add(8 * lda), lda, ap.add(8));

        k_i += 8;
    }

    while k_i < k {
        a = a0.add(k_i);
        ap = ap0.add(k_i * MR);
        seq!(i in 0..12 {
            copy_packed::<1>(a.add(lda*i), ap.add(i));
        });

        k_i += 1;
    }
}

// #[target_feature(enable = "avx")]
// pub(crate) unsafe fn pack_kx8_v1(
//     k: usize,
//     a: *const TA, lda: usize,
//     ap: *mut TA,
// ) {
//     let mut k_i = 0;
//     let mut a = a;
//     let mut ap = ap;
//     const MR: usize = 8;
//     while k_i < k8 {
//         pack_t4::<MR>(a, lda, ap);
//         pack_t4::<MR>(a.add(4*lda), lda, ap.add(4));

//         ap = ap.add(MR*8);
//         a = a.add(8);
//         k_i += 1;
//     }

//     while k_i < k {
//         seq!(i in 0..8 {
//             copy_packed::<1>(a.add(lda*i), ap.add(i));
//         });

//         ap = ap.add(MR);
//         a = a.add(1);
//         k_i += 1;
//     }
// }

// #[target_feature(enable = "avx")]
// pub(crate) unsafe fn pack_kx6_v1(
//     k: usize,
//     a: *const TA, lda: usize,
//     ap: *mut TA,
// ) {
//     let mut k_i = 0;
//     let mut a = a;
//     let mut ap = ap;
//     const MR: usize = 6;
//     while k_i < k8 {
//         pack_t4::<MR>(a, lda, ap);
//         pack_t4::<MR>(a.add(4*lda), lda, ap.add(4));

//         ap = ap.add(MR*8);
//         a = a.add(8);
//         k_i += 1;
//     }

//     k_i = 0;

//     while k_i < k {
//         seq!(i in 0..6 {
//             copy_packed::<1>(a.add(lda*i), ap.add(i));
//         });

//         ap = ap.add(MR);
//         a = a.add(1);
//         k_i += 1;
//     }
// }

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx8_v1(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    const MR: usize = 8;
    let mut a = a;
    let mut ap = ap;
    let a0 = a;
    let ap0 = ap;
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    while k_i < k8 {
        a = a0.add(k_i);
        ap = ap0.add(k_i * MR);
        pack_t4::<MR>(a, lda, ap);
        pack_t4::<MR>(a.add(4 * lda), lda, ap.add(4));

        k_i += 8;
    }

    while k_i < k {
        a = a0.add(k_i);
        ap = ap0.add(k_i * MR);
        seq!(i in 0..8 {
            copy_packed::<1>(a.add(lda*i), ap.add(i));
        });

        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx4_v1(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    const MR: usize = 4;
    let mut a = a;
    let mut ap = ap;
    let a0 = a;
    let ap0 = ap;
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    while k_i < k8 {
        a = a0.add(k_i);
        ap = ap0.add(k_i * MR);
        pack_t4::<MR>(a, lda, ap);

        k_i += 8;
    }

    while k_i < k {
        a = a0.add(k_i);
        ap = ap0.add(k_i * MR);
        seq!(i in 0..4 {
            copy_packed::<1>(a.add(lda*i), ap.add(i));
        });

        k_i += 1;
    }
}

#[target_feature(enable = "avx")]
pub(crate) unsafe fn pack_kx6_v1(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    const MR: usize = 6;
    let mut a = a;
    let mut ap = ap;
    let a0 = a;
    let ap0 = ap;
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    while k_i < k8 {
        a = a0.add(k_i);
        ap = ap0.add(k_i * MR);
        pack_t4::<MR>(a, lda, ap);
        pack_t2::<MR>(a.add(lda * 4), lda, ap.add(4));

        k_i += 8;
    }

    while k_i < k {
        a = a0.add(k_i);
        ap = ap0.add(k_i * MR);
        seq!(i in 0..6 {
            copy_packed::<1>(a.add(lda*i), ap.add(i));
        });

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
                    while m_idx < m_rounded{
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
def_packa!(8);
def_packa!(12);
def_packa!(24);
// def_packa!(8);
