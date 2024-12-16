use seq_macro::seq;
use pire_base::{
    def_ukernel_avx, mem,
    init_ab_avx,
    acc_3, acc_2, acc_1, store_3, store_2, store_1,
    fmadd_3, fmadd_2, fmadd_1,
    step_3, step_2, step_1,
};
use crate::{TC, TC_SIZE};

use super::super::avx::asm_ukernel::{
    inc_b, inc_b_k_unroll,
    vzeroall,
    mask_ptr, load_mask,
    load_beta,
    vs, v_i,
    loadp_unit, storep_unit,
};

type TA = f32;
type TB = f32;
type TS = f32;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vcvtph2ps ", $m0, ", %ymm2\n",
            "vaddps %ymm2, %ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vcvtph2ps ", $m0, ", %ymm2\n",
            "vfmadd231ps %ymm2, %ymm0, %ymm", $r1, "\n",
        ) 
    };
}

macro_rules! c_load {
    () => {
        concat!(
            "mov 16({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x3}\n",
            "lea ({cx}, {x3},), {x1}\n",
            "lea ({x1}, {x3},), {x2}\n",
            "lea ({x2}, {x3},), {x3}\n",
        )
    };
}

macro_rules! vbroadcast {
    () => { "vbroadcastss" };
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231ps %ymm", $r1, ", %ymm", $r2,", %ymm", $r3, "\n",
        ) 
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                vbroadcast!(), " ({alphax}),%ymm1", "\n",
                #(
                    "vmulps %ymm1, %ymm", r, ",%ymm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! br_3 {
    ($nr:tt) => {3};
}
macro_rules! br_2 {
    (0) => { 2 };
    (1) => { 3 };
    (2) => { 2 };
    (3) => { 3 };
}

macro_rules! br_1 {
    (0) => { 1 };
    (1) => { 2 };
    (2) => { 3 };
    (3) => { 1 };
}

macro_rules! cr {
    (0,0) => { 4 };
    (1,0) => { 5 };
    (2,0) => { 6 };
    (0,1) => { 7 };
    (1,1) => { 8 };
    (2,1) => { 9 };
    (0,2) => { 10 };
    (1,2) => { 11 };
    (2,2) => { 12 };
    (0,3) => { 13 };
    (1,3) => { 14 };
    (2,3) => { 15 };
}

macro_rules! vzero_kernel {
    () => { vzeroall!(4, 15) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(4, 15) };
}

macro_rules! load_b {
    (B, $nr:tt, $ni:tt, $K:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $nr, "*4+", $ni, "*4({bx}), %ymm", $r, "\n",
        )
    };
}

// NOTE: BS ukernel for f16 is hard to implement since it requires loading single f16 in a strided fashion
// we can do this, it will require avx2, and som other issues, which I dont want to deal with
// Additiionally and more importantly, it wont be performant neough since it reqiures to convert additioanl
// computation, it wont benefit from vectorization since we load single f16 in strided layout.

// Dont use partial since partially (and efficiently at the same time) is hard instead copy to c buffer

def_ukernel_avx!(1, step_3, acc_3, store_3, 3, 4, B, C, ukernel_bbc);

def_ukernel_avx!(1, step_3, acc_3, store_3, 3, 4, B, C, ukernel_3_bbp);
def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 4, B, C, ukernel_2_bbp);
def_ukernel_avx!(1, step_1, acc_1, store_1, 1, 4, B, C, ukernel_1_bbp);
