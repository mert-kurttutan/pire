#[rustfmt::skip]
mod asm_ukernel;

use asm_ukernel::*;

use crate::{UnaryFnC, TA, TB, TC};

const VS: usize = 4;
const VS_MAX: usize = VS;

const ZERO: TC = TC::ZERO;

const fn simd_vector_length() -> usize {
    VS
}

use pire_base::def_kernel_bb_v0;
def_kernel_bb_v0!(TA, TB, TC, TC, false, T, 1, 3, 4, 64, 8);

use pire_base::def_kernel_bs;
def_kernel_bs!(TA, TB, TC, TC, 3, 4);

use super::pack_avx::packa_panel_12;
use pire_base::def_kernel_sb_pf1;
def_kernel_sb_pf1!(TA, TA, TB, TC, TC, false, packa_panel_12, 1, 3, 4, 96, 8);
