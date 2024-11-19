#[rustfmt::skip]
mod asm_ukernel;

use asm_ukernel::*;

const VS: usize = 16;

// const fn simd_vector_length() -> usize {
//     VS
// }

use crate::UnaryFnC;

use half::f16;

use pire_base::def_kernel_bb_pf1;
def_kernel_bb_pf1!(f32, f32, f16, f32, F, 1, 3, 8, 96, 8);
