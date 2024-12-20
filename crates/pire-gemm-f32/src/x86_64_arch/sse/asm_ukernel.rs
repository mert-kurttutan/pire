use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_sse, 
    mem, init_ab_avx,
    acc_2, acc_1, store_2, store_1,
    step_2, step_1,
    b_mem,
};

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


type TS = TC;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

macro_rules! load_mask {
    ($is_partial:tt) => { "" };
}

macro_rules! vs {
    () => { "0x10" };
}
macro_rules! bs {
    () => { "4" };
}
macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x10+" , $m) };
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "movups ", $m0, ", %xmm2", "\n",
            "addps %xmm2,%xmm", $r1, "\n",
        ) 
    };
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "movups ", $m0, ", %xmm2", "\n",
            "mulps %xmm0,%xmm2", "\n",
            "addps %xmm2,%xmm", $r1, "\n",
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
            concat!(#("xorps %xmm",r,",%xmm",r,"\n",)*)
        })
    }
}

macro_rules! vbroadcast {
    () => { "movss" };
}

macro_rules! vfmadd {
    ($i:tt, $j:tt, $b_macro:tt) => {
        concat!(
            "movups %xmm", $b_macro!($j), ", %xmm", dr!($i,$j), "\n",
            "mulps %xmm", $i, ", %xmm", dr!($i,$j), "\n",
            "addps %xmm", dr!($i,$j), ", %xmm", cr!($i,$j), "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "movaps ", mem!($m0, concat!("0x10*", $r1)), ", %xmm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "movups %xmm", $r1, ", ", $m0,  "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                vbroadcast!(), " ({alphax}),%xmm1", "\n",
                "shufps ", "$0, %xmm1, %xmm1", "\n",
                #(
                    "mulps %xmm1, %xmm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %xmm0\n",
            "shufps ", "$0, %xmm0, %xmm0", "\n",
        )
    }
}

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

macro_rules! load_b {
    ($b_layout:tt, $nr:tt, $ni:tt, $K:tt, $b_macro:tt) => {
        concat!(
            vbroadcast!(), " ", b_mem!($b_layout,$nr,$ni,$K), ",%xmm", $b_macro!($ni), "\n",
            "shufps ", "$0, %xmm", $b_macro!($ni), ", %xmm", $b_macro!($ni), "\n",
        )
    };
}

macro_rules! vzero_kernel {
    () => { vzeroall!(4,11) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(4,11) };
}


def_ukernel_sse!(1, step_2, acc_2, store_2, 2, 4, B, C, ukernel_bbc);

def_ukernel_sse!(1, step_2, acc_2, store_2, 2, 4, B, C, ukernel_2_bbp);
def_ukernel_sse!(1, step_1, acc_1, store_1, 1, 4, B, C, ukernel_1_bbp);

def_ukernel_sse!(1, step_2, acc_2, store_2, 2, 4, S, C, ukernel_bsc);

def_ukernel_sse!(1, step_2, acc_2, store_2, 2, 4, S, C, ukernel_2_bsp);
def_ukernel_sse!(1, step_1, acc_1, store_1, 1, 4, S, C, ukernel_1_bsp);
