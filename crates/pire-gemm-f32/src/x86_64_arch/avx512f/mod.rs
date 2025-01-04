#[rustfmt::skip]
mod asm_ukernel;

pub(crate) use asm_ukernel::*;

use crate::{UnaryFnC, TA, TB, TC};

const VS: usize = 16;
const VS_MAX: usize = VS;

const fn simd_vector_length() -> usize {
    VS
}
const ZERO: f32 = 0.0;

use pire_base::def_kernel_bb_v0;
def_kernel_bb_v0!(TA, TB, TC, TC, false, T, 1, 3, 8, 96, 8);

use pire_base::def_kernel_bs;
def_kernel_bs!(TA, TB, TC, TC, 3, 8);

// use pire_base::def_kernel_ss;
// def_kernel_ss!(TA, TB, TC, TC, 3, 8);

use super::pack_avx::packa_panel_48;

use pire_base::def_kernel_sb_v0;
def_kernel_sb_v0!(TA, TA, TB, TC, TC, false, T, packa_panel_48, 1, 3, 8, 96, 8);

pub(crate) unsafe fn dot_kernel<F: UnaryFnC>(
    m: usize,
    n: usize,
    k: usize,
    a: *const TA,
    b: *const TB,
    c: *mut TC,
    a_rs: usize,
    a_cs: usize,
    b_rs: usize,
    b_cs: usize,
    c_rs: usize,
    c_cs: usize,
    alpha: *const TA,
    beta: *const TC,
    f: F,
) {
    let lda = if a_rs == 1 { a_cs } else { a_rs };
    let ldb = if b_rs == 1 { b_cs } else { b_rs };
    let mr = 3;
    let nr = 8;
    let (m, n, lda, ldb, a, b, c_rs, c_cs) = if n < 4 {
        (n, m, ldb, lda, b, a, c_cs, c_rs)
    } else {
        (m, n, lda, ldb, a, b, c_rs, c_cs)
    };
    // let (mr, nr) = if m == 3 {
    //     (nr, mr)
    // } else {
    //     (mr, nr)
    // };
    let ukernel_func = ukernel_rcc3;
    let mut mi = 0;
    while mi < m / mr * mr {
        let mut ni = 0;
        while ni < n {
            let nr_cur = nr.min(n - ni);
            ukernel_func(
                a.add(mi * lda),
                b.add(ni * ldb),
                c.add(mi * c_rs + ni * c_cs),
                alpha,
                beta,
                lda,
                ldb,
                c_rs,
                c_cs,
                mr,
                nr_cur,
                k,
            );
            ni += nr;
        }
        mi += mr;
    }
    let mr_left = m - mi;
    let mr_left_ukernel_func = if mr_left == 2 { ukernel_rcc2 } else { ukernel_rcc1 };
    if mr_left > 0 {
        let mut ni = 0;
        while ni < n {
            let nr_cur = nr.min(n - ni);
            mr_left_ukernel_func(
                a.add(mi * lda),
                b.add(ni * ldb),
                c.add(mi * c_rs + ni * c_cs),
                alpha,
                beta,
                lda,
                ldb,
                c_rs,
                c_cs,
                mr_left,
                nr_cur,
                k,
            );
            ni += nr;
        }
    }
    for i in 0..m {
        for j in 0..n {
            let c_ij = c.add(i * c_rs + j * c_cs);
            f.call(c_ij, 1);
        }
    }
}
