use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_avx,
    init_ab_avx,
    acc_2, acc_1, store_2, store_1,
    step_2, step_1,
    mem, b_mem,
};
type TS = TC;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

macro_rules! vs {
    () => { "0x20" };
}
macro_rules! bs {
    () => { "4" };
}
pub(crate) use vs;

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x20+" , $m) };
}
pub(crate) use v_i;

#[inline(always)]
pub(crate) fn mask_and_offset(m: usize) -> ([u32;16], usize) {
    let mask: [u32; 16] = [
        u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let mask_offset = if m % VS == 0 { 0 } else { VS - (m %VS)};

    (mask, mask_offset)
}

macro_rules! mask_ptr {
    (P, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let (mask, mask_offset) = mask_and_offset($m);
        let $nm = mask.as_ptr().add(mask_offset);
        let $mask_ptr = $nm;
    };
    (C, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let mask = [0xFFFFFFFF_u64];
        let $nm = mask.as_ptr();
        let $mask_ptr = $nm;
    };
}
pub(crate) use mask_ptr;

macro_rules! load_mask {
    (P) => { "vmovdqu ({maskx}), %ymm1"
    };
    (C) => { "/* {maskx} */" }
}
pub(crate) use load_mask;

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("vpxor %ymm",r,",%ymm",r,",%ymm",r,"\n",)*)
        })
    }
}
pub(crate) use vzeroall;

macro_rules! vbroadcast {
    () => { "vbroadcastss" };
}
pub(crate) use vbroadcast;

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
            "vmulps ", $m0, ",%ymm0,%ymm2", "\n",
            "vaddps %ymm2,%ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vmulps %ymm2, %ymm0,%ymm3", "\n",
            "vaddps %ymm3,%ymm", $r1, ",%ymm", $r1, "\n",
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

macro_rules! vfmadd {
    ($i:tt, $j:tt, $b_macro:tt) => {
        concat!(
            "vmulps %ymm", $i, ", %ymm", $b_macro!($j),", %ymm", dr!($i,$j), "\n",
            "vaddps %ymm", dr!($i,$j), ", %ymm", cr!($i,$j), ", %ymm", cr!($i,$j), "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr, B) => {
        concat!(
            "vmovaps ", mem!($m0, concat!("0x20*", $r1)), ", %ymm", $r1, "\n",
        )
    };
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovaps ", mem!($m0, concat!("0x20*", $r1)), ", %ymm", $r1, "\n",
        )
    };
}

pub(crate) use loadp_unit;

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (P, $r1:expr, $m0:expr) => {
        concat!(
            "vmaskmovps %ymm", $r1, ", %ymm1, ", $m0,  "\n",
        )
    };
}

pub(crate) use storep_unit;

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
pub(crate) use alpha_scale_0;

macro_rules! load_beta {
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %ymm0\n",
        )
    }
}
pub(crate) use load_beta;

macro_rules! inc_b {
    (S,$nr:tt) => { "add {x1},{bx} \n add {x1},{x3} \n" };
    (B,$nr:tt) => { "" };
}

macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => { "" };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $4*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}

macro_rules! vzero_kernel {
    () => { vzeroall!(4,11) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(4,11) };
}

macro_rules! load_b {
    ($b_layout:tt, $nr:tt, $ni:tt, $K:tt, $b_macro:tt) => {
        concat!(
            vbroadcast!(), " ", b_mem!($b_layout,$nr,$ni,$K), ",%ymm", $b_macro!($ni), "\n",
        )
    };
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
    (0,1) => { 6 };
    (1,1) => { 7 };
    (0,2) => { 8 };
    (1,2) => { 9 };
    (0,3) => { 10 };
    (1,3) => { 11 };
}

macro_rules! dr {
    (0,0) => { 12 };
    (1,0) => { 13 };
    (0,1) => { 14 };
    (1,1) => { 15 };
    (0,2) => { 12 };
    (1,2) => { 13 };
    (0,3) => { 14 };
    (1,3) => { 15 };
}

def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 4, B, C, ukernel_bbc);

def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 4, B, P, ukernel_2_bbp);
def_ukernel_avx!(1, step_1, acc_1, store_1, 1, 4, B, P, ukernel_1_bbp);

def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 4, S, C, ukernel_bsc);

def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 4, S, P, ukernel_2_bsp);
def_ukernel_avx!(1, step_1, acc_1, store_1, 1, 4, S, P, ukernel_1_bsp);
