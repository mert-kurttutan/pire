#[rustfmt::skip]
mod asm_ukernel;
// mod axpy_kernel;

use asm_ukernel::*;
// use axpy_kernel::*;

use crate::{UnaryFnC, TA, TB, TC};

const VS: usize = 8;

use pire_base::def_kernel_bb_pf1;
def_kernel_bb_pf1!(TA, TB, TC, TC, F, 1, 3, 8, 64, 16);

use pire_base::def_kernel_bs;
def_kernel_bs!(TA, TB, TC, TC, 3, 8);

use super::pack_avx::packa_panel_24;
use pire_base::def_kernel_sb_pf1;
def_kernel_sb_pf1!(TA, TA, TB, TC, TC, packa_panel_24, 1, 3, 8, 96, 8);
