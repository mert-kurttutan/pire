#[rustfmt::skip]
mod asm_ukernel;
// mod axpy_kernel;

use asm_ukernel::*;
// use axpy_kernel::*;

use crate::UnaryFnC;

const VS: usize = 16;
const VS_MAX: usize = VS;

const ZERO: i32 = 0;

const fn simd_vector_length() -> usize {
    VS
}

use pire_base::def_kernel_bb_v0;
def_kernel_bb_v0!(i16, i16, i32, f32, false, T, 2, 3, 8, 64, 8);

use super::pack_avx::packa_panel_48;

use pire_base::def_kernel_sb_v0;
def_kernel_sb_v0!(i16, i16, i16, i32, f32, false, T, packa_panel_48, 2, 3, 8, 64, 8);
