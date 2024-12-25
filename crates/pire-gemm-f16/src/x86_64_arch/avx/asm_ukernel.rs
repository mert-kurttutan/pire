use seq_macro::seq;
use crate::{TC, TC_SIZE};
use pire_base::{
    def_ukernel_avx, mem, b_mem,
    init_ab_avx,
    acc_2, store_2, acc_1, store_1,
    step_2, step_1,
};

type TA = f32;
type TB = f32;

type TS = f32;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

macro_rules! vs {
    () => { "0x20" };
}
pub(crate) use vs;

macro_rules! bs {
    () => { "4" }
}
pub(crate) use bs;

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x10+" , $m) };
}
pub(crate) use v_i;
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
            "vmulps %ymm2, %ymm0,%ymm2", "\n",
            "vaddps %ymm2, %ymm", $r1, ",%ymm", $r1, "\n",
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
}
pub(crate) use loadp_unit;

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vcvtps2ph $0x00, %ymm", $r1, ", ", $m0, "\n",
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

macro_rules! load_beta {
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %ymm0\n",
        )
    }
}

pub(crate) use load_beta;

macro_rules! inc_b {
    (B,$nr:tt) => { "" };
}

pub(crate) use inc_b;

macro_rules! inc_b_k_unroll {
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $4*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}

pub(crate) use inc_b_k_unroll;

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

macro_rules! vzero_kernel {
    () => { vzeroall!(4, 11) };
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

macro_rules! mask_ptr {
    (C, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let mask = [0xFFFF_u32];
        let $nm = mask.as_ptr();
        let $mask_ptr = $nm;
    };
}

pub(crate) use mask_ptr;

macro_rules! load_mask {
    (C) => { "/* {maskx} */" }
}

pub(crate) use load_mask;

// NOTE: BS ukernel for f16 is hard to implement since it requires loading single f16 in a strided fashion
// we can do this, it will require avx2, and som other issues, which I dont want to deal with
// Additiionally and more importantly, it wont be performant neough since it reqiures to convert additioanl
// computation, it wont benefit from vectorization since we load single f16 in strided layout.

// Dont use partial since partially (and efficiently at the same time) is hard instead copy to c buffer

def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 4, B, C, ukernel_bbc);

def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 4, B, C, ukernel_2_bbp);
def_ukernel_avx!(1, step_1, acc_1, store_1, 1, 4, B, C, ukernel_1_bbp);
