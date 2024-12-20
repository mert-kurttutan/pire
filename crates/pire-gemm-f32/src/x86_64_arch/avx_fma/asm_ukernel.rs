use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_avx, def_ukernel_avx_2,
    init_ab_avx, init_ab_2,
    acc_3, acc_2, acc_1,
    store_3, store_2, store_1,
    step_3, step_2, step_1,
    mem, b_mem,
};

use super::super::avx::asm_ukernel::{
    mask_ptr, load_mask,
    vs, v_i, mask_and_offset,
    vzeroall, vbroadcast,
    loadp_unit, storep_unit,
    alpha_scale_0, load_beta,
};
macro_rules! bs {
    () => { "4" };
}
type TS = TC;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vaddps ", $m0, ",%ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vaddps %ymm2, %ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vfmadd231ps ", $m0, ",%ymm0,%ymm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vfmadd231ps %ymm2, %ymm0,%ymm", $r1, "\n",
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

macro_rules! c_load_2 {
    () => {
        concat!(
            "mov ({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x3}\n",
            "lea ({cx}, {x3},), {x1}\n",
            "lea ({x1}, {x3},), {x2}\n",
            "lea ({x2}, {x3},), {x3}\n",
        )
    };
}

macro_rules! vfmadd {
    ($i:tt, $j:tt, $b_macro:tt) => {
        concat!(
            "vfmadd231ps %ymm", $i, ", %ymm", $b_macro!($j),", %ymm", cr!($i,$j), "\n",
        )  
    };
}

macro_rules! inc_b {
    (S,$nr:tt) => { "add {x1},{bx} \n add {x1},{x3} \n" };
    (B,$nr:tt) => { "" };
    ($nr:tt) => { "" };
}

macro_rules! inc_b_k_unroll {
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $4*", $K, "*", $X, ", {bx}", "\n",
        )
    };
    (S, $X:tt, $K:tt) => { "" };
}

macro_rules! vzero_kernel {
    () => { vzeroall!(4,15) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(4,15) };
}

macro_rules! prefetch {
    (B, $nr:tt, 0, 0) => {
        "prefetcht0 384({bx})\n"
    };
    ($b_layout:tt, $nr:tt, $ni:tt, $K:tt) => {
        ""
    };
}

macro_rules! load_b {
    ($b_layout:tt, $nr:tt, $ni:tt, $K:tt, $b_macro:tt) => {
        concat!(
            prefetch!($b_layout, $nr, $ni, $K),
            vbroadcast!(), " ", b_mem!($b_layout,$nr,$ni,$K), ",%ymm", $b_macro!($ni), "\n",
        )
    };
}
macro_rules! br_3 {
    ($nr:tt) => { 3 };
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
    (3) => { 5 };
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

def_ukernel_avx!(1, step_3, acc_3, store_3, 3, 4, B, P, ukernel_3_bbp);
def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 4, B, P, ukernel_2_bbp);
def_ukernel_avx!(1, step_1, acc_1, store_1, 1, 4, B, P, ukernel_1_bbp);

def_ukernel_avx!(1, step_3, acc_3, store_3, 3, 4, S, C, ukernel_bsc);

def_ukernel_avx!(1, step_3, acc_3, store_3, 3, 4, S, P, ukernel_3_bsp);
def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 4, S, P, ukernel_2_bsp);
def_ukernel_avx!(1, step_1, acc_1, store_1, 1, 4, S, P, ukernel_1_bsp);

def_ukernel_avx_2!(1, step_3, acc_3, store_3, 3, 4, 4, 32);
