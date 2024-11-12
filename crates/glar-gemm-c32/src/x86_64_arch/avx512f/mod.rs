#[rustfmt::skip]
mod asm_ukernel;
use asm_ukernel::*;

use crate::{UnaryFnC, TA, TB, TC};

const VS: usize = 8;

// const fn simd_vector_length() -> usize {
//     VS
// }

use glar_base::def_kernel_bb_pf1;
def_kernel_bb_pf1!(TA, TB, TC, TC, F, 1, 3, 4, 64, 8);

use glar_base::def_kernel_bs;
def_kernel_bs!(TA, TB, TC, TC, 3, 4);

use super::pack_avx::packa_panel_24;
use glar_base::def_kernel_sb_pf1;
def_kernel_sb_pf1!(TA, TA, TB, TC, TC, packa_panel_24, 1, 3, 4, 96, 8);
