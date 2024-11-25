#[rustfmt::skip]
mod asm_ukernel;
// mod axpy_kernel;

use asm_ukernel::*;
// use axpy_kernel::*;

use crate::{UnaryFnC, TA, TB, TC};

const VS: usize = 8;
// const VS_MAX: usize = VS;

const ZERO: f64 = 0.0;

use pire_base::def_kernel_bb_pf1;
def_kernel_bb_pf1!(TA, TB, TC, TC, false, 1, 3, 8, 64, 16);

use pire_base::def_kernel_bs;
def_kernel_bs!(TA, TB, TC, TC, 3, 8);

use super::pack_avx::packa_panel_24;
use pire_base::def_kernel_sb_pf1;
def_kernel_sb_pf1!(TA, TA, TB, TC, TC, false, packa_panel_24, 1, 3, 8, 96, 8);
