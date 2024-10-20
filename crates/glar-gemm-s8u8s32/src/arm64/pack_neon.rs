// use crate::{TA, TB};
use crate::TA;
// use seq_macro::seq;

use paste::paste;

// use std::ptr::copy_nonoverlapping;

#[target_feature(enable = "neon")]
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
    let k4 = k / 8 * 8;
    let kl = k % 8;
    let kl_4 = if kl == 0 { 0 } else { 8 };
    for i in 0..m_left {
        let mut j = 0;
        while j < k4 {
            *ap.add(j * mr + i * 8) = *a.add(j * a_cs + i * a_rs);
            *ap.add(j * mr + i * 8 + 1) = *a.add((j + 1) * a_cs + i * a_rs);
            *ap.add(j * mr + i * 8 + 2) = *a.add((j + 2) * a_cs + i * a_rs);
            *ap.add(j * mr + i * 8 + 3) = *a.add((j + 3) * a_cs + i * a_rs);
            *ap.add(j * mr + i * 8 + 4) = *a.add((j + 4) * a_cs + i * a_rs);
            *ap.add(j * mr + i * 8 + 5) = *a.add((j + 5) * a_cs + i * a_rs);
            *ap.add(j * mr + i * 8 + 6) = *a.add((j + 6) * a_cs + i * a_rs);
            *ap.add(j * mr + i * 8 + 7) = *a.add((j + 7) * a_cs + i * a_rs);
            j += 8;
        }
        let mut jl = 0;
        while jl < kl {
            *ap.add(j * mr + i * 8 + jl) = *a.add((j + jl) * a_cs + i * a_rs);
            jl += 1;
        }
        while jl < kl_4 {
            *ap.add(j * mr + i * 8 + jl) = 0;
            jl += 1;
        }
    }
}

// macro_rules! def_packb {
//     ($nr:tt) => {
//         paste! {
//         #[target_feature(enable = "neon")]
//         pub(crate) unsafe fn [<packb_panel_ $nr>](
//                 n: usize, k: usize,
//                 b: *const TB, b_rs: usize, b_cs: usize,
//                 bp: *mut TB,
//             ) {
//                 let k_eff = (k+7) / 8 * 8;
//                 let bp0 = bp as *mut i8;
//                 let b0 = b as *const i8;
//                 const NR: usize = $nr;
//                 let n_rounded = n / NR * NR;
//                 let mut n_idx = 0;
//                 if b_rs == 1 {
//                     let ldb = b_cs;
//                     while n_idx < n_rounded {
//                         let b = b0.add(n_idx);
//                         let bp = bp0.add(n_idx*k_eff);
//                         // pack_k_v0::<NR,NR>(k, b, ldb, bp);
//                         pack_scalar_k(
//                             NR, k,
//                             b, 1, ldb,
//                             bp, 1
//                         );
//                         n_idx += NR;
//                     }
//                     let n_left = n - n_idx;
//                     if n_left > 0 {
//                         pack_scalar_k(
//                             n_left, k,
//                             b0.add(n_idx), b_rs, b_cs,
//                             bp0.add(n_idx*k_eff), 1
//                         );
//                     }
//                 } else if b_cs == 1 {
//                     let ldb = b_rs;
//                     while n_idx < n_rounded {
//                         let b = b0.add(n_idx*ldb);
//                         let bp = bp0.add(n_idx*k_eff);
//                         // pack_k_v1::<NR,NR>(k, b, ldb, bp);
//                         pack_scalar_k(
//                             NR, k,
//                             b, b_rs, b_cs,
//                             bp, 1
//                         );
//                         n_idx += NR;
//                     }
//                     let n_left = n - n_idx;
//                     if n_left > 0 {
//                         pack_scalar_k(
//                             n_left, k,
//                             b0.add(n_idx*ldb), b_rs, b_cs,
//                             bp0.add(n_idx*k_eff), 1
//                         );
//                     }
//                 }
//             }
//         }
//     };
// }

// def_packb!(12);

macro_rules! def_packa {
    ($mr:tt) => {
        paste! {
            #[target_feature(enable = "neon")]
            pub(crate) unsafe fn [<packa_panel_ $mr>](
                m: usize, k: usize,
                a: *const TA, a_rs: usize, a_cs: usize,
                ap: *mut TA, vs: usize,
            ) {
                let mr = $mr;
                let k_eff = (k+7) / 8 * 8;
                let ap0 = ap;
                let a0 = a;
                let m_rounded = m / mr * mr;
                let mut m_idx = 0;
                if a_rs == 1 {
                    let lda = a_cs;
                    while m_idx < m_rounded {
                        let a = a0.add(m_idx);
                        let ap = ap0.add(m_idx*k_eff);
                        pack_scalar_k(
                            mr, k,
                            a, 1, lda,
                            ap, vs
                        );
                        // pack_k_v0::<$mr,$mr>(k, a, lda, ap);
                        m_idx += mr;
                    }
                    let m_left = m - m_idx;
                    if m_left > 0 {
                        pack_scalar_k(
                            m_left, k,
                            a0.add(m_idx), a_rs, a_cs,
                            ap0.add(m_idx*k_eff), vs
                        );
                    }

                } else if a_cs == 1 {
                    let lda = a_rs;
                    while m_idx < m_rounded {
                        let a = a0.add(m_idx*lda);
                        let ap = ap0.add(m_idx*k_eff);
                        pack_scalar_k(
                            mr, k,
                            a, a_rs, a_cs,
                            ap, vs
                        );
                        // pack_k_v1::<$mr,$mr>(k, a, lda, ap);
                        m_idx += mr;
                    }
                    let m_left = m - m_idx;
                    if m_left > 0 {
                        pack_scalar_k(
                            m_left, k,
                            a0.add(m_idx*lda), a_rs, a_cs,
                            ap0.add(m_idx*k_eff), vs
                        );
                    }
                }
            }
        }
    };
}

def_packa!(8);
