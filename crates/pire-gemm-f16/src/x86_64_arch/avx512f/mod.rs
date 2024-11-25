#[rustfmt::skip]
mod asm_ukernel;

use asm_ukernel::*;

const VS: usize = 16;
const VS_MAX: usize = VS;

const fn simd_vector_length() -> usize {
    VS
}

use crate::UnaryFnC;

use half::f16;

const ZERO: f16 = f16::ZERO;

use pire_base::def_kernel_bb_v0;
def_kernel_bb_v0!(f32, f32, f16, f32, false, T, 1, 3, 8, 96, 8);
