#[rustfmt::skip]
mod asm_ukernel;
// mod axpy_kernel;

use asm_ukernel::*;
// use axpy_kernel::*;

use crate::UnaryFnC;

const VS: usize = 16;

// const fn simd_vector_length() -> usize {
//     VS
// }

use glar_base::def_kernel_bb_pf1;
def_kernel_bb_pf1!(i8, u8, i32, f32, F, 4, 3, 8, 64, 8);

use super::pack_avx::packa_panel_48;

use glar_base::def_kernel_sb_pf1;
def_kernel_sb_pf1!(i8, i8, u8, i32, f32, packa_panel_48, 4, 3, 8, 64, 8);
