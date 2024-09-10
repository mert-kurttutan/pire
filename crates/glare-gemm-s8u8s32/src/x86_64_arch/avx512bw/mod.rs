pub mod asm_ukernel;
// pub(crate) mod axpy_kernel;

pub(crate) use asm_ukernel::*;
// pub(crate) use axpy_kernel::*;

use std::arch::asm;
use paste::paste;

use crate::{TA,TB,TC};

use crate::MyFn;

const VS: usize = 16;

use glare_base::def_kernel_bb_pf1;

def_kernel_bb_pf1!(
    i8, u8, i32, f32, f32,
    32, 8, 64, 8, 32, 16
);


use super::pack_avx::packa_panel_32;


use glare_base::def_kernel_sb_pf1;

def_kernel_sb_pf1!(
    i8, u8, i32, f32, f32,
    4,
    32, 8, 64, 8, 32, 16
);

// #[target_feature(enable = "avx2")]
pub(crate) unsafe fn kernel_sb<F: MyFn>(
    m: usize, n: usize, k: usize,
    alpha: *const f32,
    beta: *const f32,
    a: *const TA, a_rs: usize, a_cs: usize,
    b: *const TB,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap_buf: *mut TA,
    f: F,
 ) { 
    if c_rs == 1 {
        kernel_32x8_sb_v0::<_, false>(
            m, n, k,
            alpha, beta,
            a, a_rs, a_cs,
            b,
            c, c_rs, c_cs,
            ap_buf,
            f
        );
    } else {
        kernel_32x8_sb_v0::<_, true>(
            m, n, k,
            alpha, beta,
            a, a_rs, a_cs,
            b,
            c, c_rs, c_cs,
            ap_buf,
            f
        );
    }
 } 

// #[target_feature(enable = "avx2")]
pub(crate) unsafe fn kernel<F: MyFn>(
    m: usize, n: usize, k: usize,
    alpha: *const f32, beta: *const f32,
    c: *mut TC,
    c_rs: usize, c_cs: usize,
    ap: *const TA, bp: *const TB,
    f: F,
 ) {
    let k_eff = (k+3)/4 * 4;
     if c_rs == 1 {
         kernel_bb::<_, false>(m, n, k_eff, alpha, beta, c, c_rs, c_cs, ap, bp, f)
     } else {
         kernel_bb::<_, true>(m, n, k_eff, alpha, beta, c, c_rs, c_cs, ap, bp, f)
     }
 }
 