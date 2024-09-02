use seq_macro::seq;
use std::ptr::copy_nonoverlapping;
use crate::{TA,TB};


use paste::paste;


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
pub(crate) unsafe fn copy_packed<const M: usize>(a: *const TA, b: *mut TA) {
    let a = a as *const f64;
    let b = b as *mut f64;
    copy_nonoverlapping(a, b, M*2);
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


macro_rules! def_packb {
   ($nr:tt) => {
        paste! {
        // #[target_feature(enable = "avx")]
        pub(crate) unsafe fn [<packb_panel_ $nr>](
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
                        pack_k_v1::<NR,NR>(k, b, ldb, bp);
                        // [<pack_kx$nr _v1>](k_iter, k_left, b, ldb, bp);
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


def_packb!(2);
def_packb!(4);
def_packb!(3);

macro_rules! def_packa {
    ($mr:tt) => {
        paste! {
            // #[target_feature(enable = "avx")]
            pub(crate) unsafe fn [<packa_panel_ $mr>](
                m_left: usize, k: usize,
                a: *const TA, a_rs: usize, a_cs: usize,
                ap: *mut TA, vs: usize
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
                        pack_k_v0::<MR,MR>(k_iter, k_left, a, lda, ap);
                        m_idx += MR;
                        ap = ap.add(k * MR);
                        a = a.add(MR);
                    }
                    let m_left = m_left - m_idx;
                    pack_scalar_k(
                        m_left, k,
                        a, a_rs, a_cs,
                        ap, vs
                    );

                } else if a_cs == 1 {
                    let lda = a_rs;
                    while m_idx + MR_LAST_STEP <= m_left {
                        // [<pack_kx$mr _v1>](k_iter, k_left, a, lda, ap);
                        pack_k_v1::<MR,MR>(k, a, lda, ap);
                        m_idx += MR;
                        ap = ap.add(k * MR);
                        a = a.add(MR*lda);
                    }
                    let m_left = m_left - m_idx;
                    pack_scalar_k(
                        m_left, k,
                        a, a_rs, a_cs,
                        ap, vs
                    );
                }
            }
        }
    };
}

// def_packa!(12);
def_packa!(12);
def_packa!(4);
