use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_sse,
    acc_1, store_1, step_1,
    mem,
};

macro_rules! br_1 {
    (0) => {1};
    (1) => {1};
    (2) => {1};
    (3) => {1};
}

macro_rules! cr {
    (0,0) => { 4 };
    (0,1) => { 5 };
    (0,2) => { 6 };
    (0,3) => { 7 };
}

macro_rules! dr {
    ($i:tt, $j:tt) => { 2 };
}

type TS = f32;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

macro_rules! vs {
    () => { "0x10" };
}

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x10+" , $m) };
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r:expr, 1) => {
        concat!(
            "movups ", $m0, ", %xmm2", "\n",
            "paddd ", "%xmm2", ", %xmm", $r, "\n",
            // "paddd ", $m0, ", %xmm", $r, "\n",
        ) 
    };
    (C, $m0:expr, $r:expr, 2) => {
        concat!(
            "cvtdq2ps %xmm", $r, ",%xmm", $r, "\n",
            "movups ", $m0, ",%xmm2", "\n",
            "cvtdq2ps %xmm2", ",%xmm2", "\n",
            "mulps %xmm0, %xmm2", "\n",
            "addps %xmm2, %xmm", $r, "\n",
            "cvtps2dq %xmm", $r, ",%xmm", $r, "\n",
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
    () => {
        "movss"
    };
}

macro_rules! vfmadd {
    ($i:tt, $j:tt, $b_macro:tt) => {
        concat!(
            "movups %xmm", $b_macro!($j), ", %xmm", dr!($i,$j), "\n",
            "pmaddubsw %xmm", $i, ", %xmm", dr!($i,$j), "\n",
            "pmaddwd ", "%xmm3", ", %xmm", dr!($i,$j), "\n",
            "paddd %xmm", dr!($i,$j), ", %xmm", cr!($i,$j), "\n",
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
                "mov ({ptr_arrx}), {ax}\n",
                vbroadcast!(), " ({ax}),%xmm1", "\n",
                "shufps ", "$0, %xmm1, %xmm1", "\n",
                #(
                    "cvtdq2ps %xmm", r, ",%xmm", r, "\n",
                    "mulps %xmm1, %xmm", r, "\n",
                    "cvtps2dq %xmm", r, ",%xmm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "mov 4({ptr_arrx}), {ax}\n",
            vbroadcast!(), " ({ax}), %xmm0\n",
            "shufps $0, %xmm0, %xmm0\n",
        )
    }
}

// only non contigous along m and n direction which is not changed frequently during iteration along k direction
/*

ax -> cs_a
x2 -> cs_b
x3 -> ax + 3*cs_a
x4 -> bx + 3*cs_b

*/


macro_rules! init_ab {
    (B) => {
        concat!(
            // move 2 1_i16 to xmm3
            "mov $0x10001, {x0:e}", "\n",
            "movd {x0:e}, %xmm3", "\n",
            "shufps $0,%xmm3,%xmm3\n",
            "mov 4({dim_arrx}),{x0}", "\n",
        )
    };
}


macro_rules! c_load {
    () => {
        concat!(
            "mov ({dim_arrx}),{x0}", "\n",
            "lea ({x0}, {x0}, 2), {bx}", "\n",
            "lea ({cx}, {bx},), {bx}", "\n",
        )
    };
}


macro_rules! vzero_kernel {
    () => { vzeroall!(4,7) };
}

macro_rules! inc_b {
    (S,$nr:tt) => {
        "add {x0},{bx} \n"
    };
    (B,$nr:tt) => {
        ""
    };
}

macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => {
        ""
    };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $4*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}


macro_rules! alpha_scale {
    () => {alpha_scale_0!(4,7)};
}

macro_rules! load_b {
    (B, $nr:tt, $ni:tt, $K:tt, $b_macro:tt) => {
        concat!(
            "movss ", $K, "*", $nr, "*4+", $ni, "*4({bx}), %xmm", $b_macro!($ni), "\n",
            "shufps $0, %xmm", $b_macro!($ni), ", %xmm", $b_macro!($ni), "\n",
        )
    };
}

def_ukernel_sse!(4, step_1, acc_1, store_1, 1, 4, B, C, ukernel_bbc);

def_ukernel_sse!(4, step_1, acc_1, store_1, 1, 4, B, C, ukernel_1_bbp);