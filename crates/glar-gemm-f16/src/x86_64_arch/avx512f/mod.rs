#[rustfmt::skip]
pub mod asm_ukernel;

pub(crate) use asm_ukernel::*;

use paste::paste;
use seq_macro::seq;
use std::arch::asm;

const VS: usize = 16;

// const fn simd_vector_length() -> usize {
//     VS
// }

use crate::UnaryFnC;

use half::f16;

use glar_base::def_kernel_bb_pf1;

def_kernel_bb_pf1!(f32, f32, f16, f32, f32, F, 3, 8, 96, 8);

pub(crate) unsafe fn kernel<F: UnaryFnC>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
    ap: *const f32,
    bp: *const f32,
    f: F,
) {
    if c_rs == 1 {
        kernel_bb::<_, false>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    } else {
        kernel_bb::<_, true>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    }
    asm!("vzeroupper");
}
