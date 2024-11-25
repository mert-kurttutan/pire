#[rustfmt::skip]
mod asm_ukernel;
use asm_ukernel::*;

use crate::{UnaryFnC, TA, TB, TC};

const VS: usize = 8;
const VS_MAX: usize = VS;

const ZERO: TC = TC::ZERO;

const fn simd_vector_length() -> usize {
    VS
}

use pire_base::def_kernel_bb_v0;
def_kernel_bb_v0!(TA, TB, TC, TC, false, T, 1, 3, 4, 64, 8);

use pire_base::def_kernel_bs;
def_kernel_bs!(TA, TB, TC, TC, 3, 4);

use super::pack_avx::packa_panel_24;
use pire_base::def_kernel_sb_v0;
def_kernel_sb_v0!(TA, TA, TB, TC, TC, false, T, packa_panel_24, 1, 3, 4, 96, 8);
