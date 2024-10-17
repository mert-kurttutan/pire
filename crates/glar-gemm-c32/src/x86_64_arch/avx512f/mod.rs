#[rustfmt::skip]
pub mod asm_ukernel;

pub(crate) use asm_ukernel::*;

use paste::paste;
use std::arch::asm;

use crate::{TA, TB, TC};

const VS: usize = 8;

use crate::MyFn;

use glar_base::def_kernel_bb_pf1_no_beta;

def_kernel_bb_pf1_no_beta!(TA, TB, TC, TA, TC, 3, 4, 64, 8, 3, 2, 1);

use glar_base::def_kernel_bs_no_beta;

def_kernel_bs_no_beta!(TA, TB, TC, TA, TC, 3, 4, 3, 2, 1);

use super::pack_avx::packa_panel_24;
use glar_base::def_kernel_sb_pf1_no_beta;

def_kernel_sb_pf1_no_beta!(TA, TB, TC, TA, TC, packa_panel_24, 3, 4, 96, 8, 3, 2, 1);

pub(crate) unsafe fn kernel_bs<F: MyFn>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    b: *const TB,
    b_rs: usize,
    b_cs: usize,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap: *const TA,
    f: F,
) {
    if c_rs == 1 {
        kernel_bs_v0::<_, false>(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, f);
    } else {
        kernel_bs_v0::<_, true>(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, f);
    }
    asm!("vzeroupper");
}

pub(crate) unsafe fn kernel_sb<F: MyFn>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    a: *const TB,
    a_rs: usize,
    a_cs: usize,
    b: *const TB,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap_buf: *mut TA,
    f: F,
) {
    if c_rs == 1 {
        kernel_sb_v0::<_, false>(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, f);
    } else {
        kernel_sb_v0::<_, true>(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, f);
    }
    asm!("vzeroupper");
}

pub(crate) unsafe fn kernel<F: MyFn>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap: *const TA,
    bp: *const TB,
    f: F,
) {
    if c_rs == 1 {
        kernel_bb::<_, false>(m, n, k, alpha, c, c_rs, c_cs, ap, bp, f)
    } else {
        kernel_bb::<_, true>(m, n, k, alpha, c, c_rs, c_cs, ap, bp, f)
    }
    asm!("vzeroupper");
}
