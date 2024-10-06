use crate::{TA, TB};
use seq_macro::seq;
use std::ptr::copy_nonoverlapping;

use paste::paste;

#[target_feature(enable = "sse")]
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

#[target_feature(enable = "sse")]
pub(crate) unsafe fn pack_k_v1<const M: usize, const MR: usize>(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    for i in 0..M {
        for j in 0..k {
            *ap.add(j * MR + i) = *a.add(j + i * lda);
        }
    }
}

#[target_feature(enable = "sse")]
pub(crate) unsafe fn copy_packed<const M: usize>(a: *const f64, b: *mut f64) {
    copy_nonoverlapping(a, b, M);
}

#[target_feature(enable = "sse")]
pub(crate) unsafe fn pack_k_v0<const M: usize, const MR: usize>(k: usize, a: *const TA, lda: usize, ap: *mut TA) {
    let k8 = k / 8 * 8;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    while k_i < k8 {
        let a = a0.add(k_i * lda);
        let ap = ap0.add(k_i * MR);
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
        let a = a0.add(k_i * lda);
        let ap = ap0.add(k_i * MR);
        copy_packed::<M>(a, ap);
        k_i += 1;
    }
}

macro_rules! def_packb {
    ($nr:tt) => {
        paste! {
        // #[target_feature(enable = "sse")]
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
                       pack_k_v1::<NR,NR>(k, b, ldb, bp);
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

def_packb!(2);

// def_packb!(6);

macro_rules! def_packa {
    ($mr:tt) => {
        paste! {
            // #[target_feature(enable = "sse")]
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
                        pack_k_v0::<MR,MR>(k, a, lda, ap);
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
                        pack_k_v1::<MR,MR>(k, a, lda, ap);
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

def_packa!(4);
