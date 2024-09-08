use seq_macro::seq;
use crate::{TA,TB};


use paste::paste;

// use std::arch::x86_64::*;


#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn pack_scalar_k(
    m_left: usize, k: usize,
    a: *const TA, a_rs: usize, a_cs: usize,
    ap: *mut TA, vs: usize
) {
    let mr = (m_left + vs - 1) / vs * vs;
    let k4 = k / 4 * 4;
    let kl = k % 4;
    let kl_4 = if kl == 0 { 0 } else { 4 };
    for i in 0..m_left  {
        let mut j = 0;
        while j < k4 {
            *ap.add(j*mr+i*4) = *a.add(j*a_cs + i*a_rs);
            *ap.add(j*mr+i*4+1) = *a.add((j+1)*a_cs + i*a_rs);
            *ap.add(j*mr+i*4+2) = *a.add((j+2)*a_cs + i*a_rs);
            *ap.add(j*mr+i*4+3) = *a.add((j+3)*a_cs + i*a_rs);
            j += 4;
        }
        let mut jl = 0;
        while jl < kl {
            *ap.add(j*mr+i*4+jl) = *a.add((j+jl)*a_cs + i*a_rs);
            jl += 1;
        }
        while jl < kl_4 {
            *ap.add(j*mr+i*4+jl) = 0;
            jl += 1;
        }

    }
}

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave_t<const M: usize>(
    a: *const TA, ap: *mut TA, lda: usize
) {
    if M == 1 {
        let mut t0 = [0_i8; 4];
        seq!(i in 0..1 {
            std::ptr::copy_nonoverlapping(a.add(lda*i), t0.as_mut_ptr().add(4*i), 4);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 4);
        return;
    }
    if M == 2 {
        let mut t0 = [0_i8; 8];
        seq!(i in 0..2 {
            std::ptr::copy_nonoverlapping(a.add(lda*i), t0.as_mut_ptr().add(4*i), 4);
        });

        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 8);
        return;
    }
    if M == 3 {
        let mut t0 = [0_i8; 12];
        seq!(i in 0..3 {
            std::ptr::copy_nonoverlapping(a.add(lda*i), t0.as_mut_ptr().add(4*i), 4);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 12);
        return;
    }
    if M == 4 {
        let mut t0 = [0_i8; 16];
        seq!(i in 0..4 {
            std::ptr::copy_nonoverlapping(a.add(lda*i), t0.as_mut_ptr().add(4*i), 4);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 16);
        return;
    }

    if M == 16 {
        let mut t0 = [0_i8; 64];
        seq!(i in 0..16 {
            std::ptr::copy_nonoverlapping(a.add(lda*i), t0.as_mut_ptr().add(4*i), 4);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 64);
        return;
    }
}

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave<const M: usize>(
    a: *const TA, ap: *mut TA, lda: usize
) {
    if M == 1 {
        let mut t0 = [0_i8; 4];
        seq!(i in 0..1 {
            t0[i*4] = *a.add(i);
            t0[i*4+1] = *a.add(lda+i);
            t0[i*4+2] = *a.add(2*lda+i);
            t0[i*4+3] = *a.add(3*lda+i);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 4);
        return;
    }
    if M == 2 {
        let mut t0 = [0_i8; 8];
        seq!(i in 0..2 {
            t0[i*4] = *a.add(i);
            t0[i*4+1] = *a.add(lda+i);
            t0[i*4+2] = *a.add(2*lda+i);
            t0[i*4+3] = *a.add(3*lda+i);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 8);
        return;
    }
    if M == 3 {
        let mut t0 = [0_i8; 12];
        seq!(i in 0..3 {
            t0[i*4] = *a.add(i);
            t0[i*4+1] = *a.add(lda+i);
            t0[i*4+2] = *a.add(2*lda+i);
            t0[i*4+3] = *a.add(3*lda+i);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 12);
        return;
    }
    if M == 4 {
        let mut t0 = [0_i8; 16];
        seq!(i in 0..4 {
            t0[i*4] = *a.add(i);
            t0[i*4+1] = *a.add(lda+i);
            t0[i*4+2] = *a.add(2*lda+i);
            t0[i*4+3] = *a.add(3*lda+i);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 16);
        return;
    }

    if M == 16 {
        let mut t0 = [0_i8; 64];
        seq!(i in 0..16 {
            t0[i*4] = *a.add(i);
            t0[i*4+1] = *a.add(lda+i);
            t0[i*4+2] = *a.add(2*lda+i);
            t0[i*4+3] = *a.add(3*lda+i);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 64);
        return;
    }
}

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave_left<const M: usize>(
    a: *const TA, ap: *mut TA, kl: usize, lda: usize
) {
    if M == 1 {
        let mut t0 = [0_i8; 4];
        for i in 0..kl {
            t0[i] = *a.add(i*lda);
        }
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 4);
        return;
    }
    if M == 2 {
        let mut t0 = [0_i8; 8];
        for i in 0..kl {
            t0[i] = *a.add(i*lda);
            t0[i+4] = *a.add(i*lda+1);
        }
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 8);
        return;
    }
    if M == 3 {
        let mut t0 = [0_i8; 12];
        for i in 0..kl {
            t0[i] = *a.add(i*lda);
            t0[i+4] = *a.add(i*lda+1);
            t0[i+8] = *a.add(i*lda+2);
        }
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 12);
        return;
    }
    if M == 4 {
        let mut t0 = [0_i8; 16];
        for i in 0..kl {
            t0[i] = *a.add(i*lda);
            t0[i+4] = *a.add(i*lda+1);
            t0[i+8] = *a.add(i*lda+2);
            t0[i+12] = *a.add(i*lda+3);
        }
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 16);
        return;
    }

    if M == 16 {
        let mut t0 = [0_i8; 64];
        for i in 0..kl {
            seq!(j in 0..16 {
                t0[i+4*j] = *a.add(i*lda+j);
            });
        }
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 64);
        return;
    }
}


#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave_left_t<const M: usize>(
    a: *const TA, ap: *mut TA, kl: usize, lda: usize
) {
    if M == 1 {
        let mut t0 = [0_i8; 4];
        std::ptr::copy_nonoverlapping(a, t0.as_mut_ptr(), kl);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 4);
        return;
    }
    if M == 2 {
        let mut t0 = [0_i8; 8];
        std::ptr::copy_nonoverlapping(a, t0.as_mut_ptr(), kl);
        std::ptr::copy_nonoverlapping(a.add(lda), t0.as_mut_ptr().add(4), kl);

        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 8);
        return;
    }
    if M == 3 {
        let mut t0 = [0_i8; 12];
        std::ptr::copy_nonoverlapping(a, t0.as_mut_ptr(), kl);
        std::ptr::copy_nonoverlapping(a.add(lda), t0.as_mut_ptr().add(4), kl);
        std::ptr::copy_nonoverlapping(a.add(2*lda), t0.as_mut_ptr().add(8), kl);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 12);
        return;
    }
    if M == 4 {
        let mut t0 = [0_i8; 16];
        std::ptr::copy_nonoverlapping(a, t0.as_mut_ptr(), kl);
        std::ptr::copy_nonoverlapping(a.add(lda), t0.as_mut_ptr().add(4), kl);
        std::ptr::copy_nonoverlapping(a.add(2*lda), t0.as_mut_ptr().add(8), kl);
        std::ptr::copy_nonoverlapping(a.add(3*lda), t0.as_mut_ptr().add(12), kl);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 16);
        return;
    }

    if M == 16 {
        let mut t0 = [0_i8; 64];
        seq!(i in 0..16 {
            std::ptr::copy_nonoverlapping(a.add(lda*i), t0.as_mut_ptr().add(4*i), kl);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 64);
        return;
    }
}

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn pack_k_v0<const M: usize, const MR: usize>(
    k: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let k8 = k / 8 * 8;
    let k4 = k / 4 * 4;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    while k_i < k8 {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        seq!(i in 0..2 {
            interleave::<M>(a.add(lda*4*i), ap.add(MR*4*i), lda);
        });

        k_i += 8;
    }

    while k_i < k4 {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        interleave::<M>(a, ap, lda);
        k_i += 4;
    }

    let kl = k % 4;
    if kl != 0 {
        let a = a0.add(k_i*lda);
        let ap = ap0.add(k_i*MR);
        interleave_left::<M>(a, ap, kl, lda);
    }
}


#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn pack_k_v1<const M: usize, const MR: usize>(
    k: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let k8 = k / 8 * 8;
    let k4 = k / 4 * 4;
    let mut k_i = 0;
    let a0 = a;
    let ap0 = ap;
    while k_i < k8 {
        let a = a0.add(k_i);
        let ap = ap0.add(k_i*MR);
        seq!(i in 0..2 {
            interleave_t::<M>(a.add(4*i), ap.add(MR*4*i), lda);
        });

        k_i += 8;
    }

    while k_i < k4 {
        let a = a0.add(k_i);
        let ap = ap0.add(k_i*MR);
        interleave_t::<M>(a, ap, lda);
        k_i += 4;
    }

    let kl = k % 4;
    if kl != 0 {
        let a = a0.add(k_i);
        let ap = ap0.add(k_i*MR);
        interleave_left_t::<M>(a, ap, kl, lda);
    }
}

// #[target_feature(enable = "avx,avx2")]
// pub(crate) unsafe fn pack_kx16_v0(
//     k: usize,
//     a: *const TA, lda: usize,
//     ap: *mut TA,
// ) {
//     let mut k_i = 0;
//     let mut a = a;
//     let mut ap = ap;
//     const MR: usize = 16;
//     while k_i < k8 {
//         // use vector intrinscs
//         seq!(i in 0..4 {
//             let a0 = _mm256_loadu_si256(a.add(lda*2*i) as *const __m256i);
//             let b0 = _mm256_loadu_si256(a.add(lda*(2*i+1)) as *const __m256i);
//             let t0 = _mm256_unpacklo_epi16(a0, b0);
//             let t1 = _mm256_unpackhi_epi16(a0, b0);
//             let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
//             let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
//             _mm256_storeu_si256(ap.add(MR*2*i) as *mut __m256i, a0);
//             _mm256_storeu_si256(ap.add(MR*2*i+16) as *mut __m256i, b0);
//         });

//         ap = ap.add(MR*8);
//         a = a.add(8*lda);

//         k_i += 1;
//     }
//     k_i = 0;
//     while k_i < k_left / 2 {
//         let a0 = _mm256_loadu_si256(a as *const __m256i);
//         let b0 = _mm256_loadu_si256(a.add(lda) as *const __m256i);
//         let t0 = _mm256_unpacklo_epi16(a0, b0);
//         let t1 = _mm256_unpackhi_epi16(a0, b0);
//         let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
//         let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
//         _mm256_storeu_si256(ap as *mut __m256i, a0);
//         _mm256_storeu_si256(ap.add(16) as *mut __m256i, b0);

//         ap = ap.add(MR*2);
//         a = a.add(lda*2);
//         k_i += 1;
//     }
//     if k_left % 2 != 0 {
//         let a0 = _mm256_loadu_si256(a as *const __m256i);
//         let b0 = _mm256_setzero_si256();
//         let t0 = _mm256_unpacklo_epi16(a0, b0);
//         let t1 = _mm256_unpackhi_epi16(a0, b0);
//         let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
//         let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
//         _mm256_storeu_si256(ap as *mut __m256i, a0);
//         _mm256_storeu_si256(ap.add(16) as *mut __m256i, b0);
//     }
// }


// #[target_feature(enable = "avx,avx2")]
// pub(crate) unsafe fn pack_kx16_v1(
//     k: usize,
//     a: *const TA, lda: usize,
//     ap: *mut TA,
// ) {
//     let mut k_i = 0;
//     let mut a = a;
//     let mut ap = ap;
//     const MR: usize = 16;
//     while k_i < k8 {
//         // pack_t::<MR>(a, lda, ap);
//         // pack_t::<MR>(a.add(8*lda), lda, ap.add(8));

//         ap = ap.add(MR*8);
//         a = a.add(8);
//         k_i += 1;
//     }

//     k_i = 0;

//     while k_i < k_left {
//         seq!(i in 0..16 {
//             *ap.add(i) = *a.add(i*lda);
//         });

//         ap = ap.add(MR);
//         a = a.add(1);
//         k_i += 1;
//     }
// }


macro_rules! def_packb {
   ($nr:tt) => {
        paste! {
        #[target_feature(enable = "avx,avx2")]
        pub(crate) unsafe fn [<packb_panel_ $nr>](
                n: usize, k: usize,
                b: *const TB, b_rs: usize, b_cs: usize,
                bp: *mut TB,
            ) {
                let k_eff = (k+3) / 4 * 4;
                let bp0 = bp as *mut i8;
                let b0 = b as *const i8;
                const NR: usize = $nr;
                let n_rounded = n / NR * NR;
                let mut n_idx = 0;
                if b_rs == 1 {
                    let ldb = b_cs;
                    while n_idx < n_rounded {
                        let b = b0.add(n_idx);
                        let bp = bp0.add(n_idx*k_eff);
                        pack_k_v0::<NR,NR>(k, b, ldb, bp);
                        n_idx += NR;
                    }
                    let n_left = n - n_idx;
                    seq!(NL in 1..$nr {
                        if n_left == NL {
                            let b = b0.add(n_idx);
                            let bp = bp0.add(n_idx*k_eff);
                            pack_k_v0::<NL,NL>(k, b, ldb, bp);
                            return;
                        }
                    });
                } else if b_cs == 1 {
                    let ldb = b_rs;
                    while n_idx < n_rounded {
                        let b = b0.add(n_idx*ldb);
                        let bp = bp0.add(n_idx*k_eff);
                        pack_k_v1::<NR,NR>(k, b, ldb, bp);
                        n_idx += NR;
                    }
                    let n_left = n - n_idx;
                    seq!(NL in 1..$nr {
                        if n_left == NL {
                            let b = b0.add(n_idx*ldb);
                            let bp = bp0.add(n_idx*k_eff);
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

macro_rules! def_packa {
    ($mr:tt) => {
        paste! {
            #[target_feature(enable = "avx,avx2")]
            pub(crate) unsafe fn [<packa_panel_ $mr>](
                m: usize, k: usize,
                a: *const TA, a_rs: usize, a_cs: usize,
                ap: *mut TA, vs: usize
            ) {
                let k_eff = (k+3) / 4 * 4;
                let ap0 = ap;
                let a0 = a;
                const MR: usize = $mr;
                let m_rounded = m / MR * MR;
                let mut m_idx = 0;
                if a_rs == 1 {
                    let lda = a_cs;
                    while m_idx < m_rounded {
                        let a = a0.add(m_idx);
                        let ap = ap0.add(m_idx*k_eff);
                        pack_k_v0::<$mr,$mr>(k, a, lda, ap);
                        m_idx += MR;
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
                        pack_k_v1::<$mr,$mr>(k, a, lda, ap);
                        m_idx += MR;
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

def_packa!(16);
