use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_sse,
    acc_2, acc_1, store_2, store_1,
    step_2, step_1,
    mem,
};

macro_rules! br_2 {
    (0) => {2};
    (1) => {2};
}

macro_rules! br_1 {
    (0) => {1};
    (1) => {2};
}

macro_rules! cr {
    (0,0) => { 4 };
    (1,0) => { 5 };
    (0,1) => { 6 };
    (1,1) => { 7 };
}

type TS = TC;

const ZERO_SCALAR: f64 = 0.0;
const ONE_SCALAR: f64 = 1.0;

macro_rules! vs {
    () => { "0x10" };
}

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x10+" , $m) };
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "movupd ", $m0, ", %xmm2", "\n",
            "addpd %xmm2,%xmm", $r1, "\n",
        ) 
    };
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "movupd ", $m0, ", %xmm2", "\n",
            "mulpd %xmm0,%xmm2", "\n",
            "addpd %xmm2,%xmm", $r1, "\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("xorpd %xmm",r,",%xmm",r,"\n",)*)
        })
    }
}

macro_rules! vbroadcast {
    () => {
        "movsd"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "movupd %xmm", $r2, ", %xmm", $r4, "\n",
            "mulpd %xmm", $r1, ", %xmm", $r4, "\n",
            "addpd %xmm", $r4, ", %xmm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "movapd ", mem!($m0, concat!("0x10*", $r1)), ", %xmm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "movupd %xmm", $r1, ", ", $m0,  "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "mov ({ptr_arrx}), {ax}\n",
                vbroadcast!(), " ({ax}),%xmm1", "\n",
                "shufpd ", "$0, %xmm1, %xmm1", "\n",
                #(
                    "mulpd %xmm1, %xmm", r, "\n",
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
            "shufpd ", "$0, %xmm0, %xmm0", "\n",
        )
    }
}

// only non contigous along m and n direction which is not changed frequently during iteration along k direction
/*

x1 -> cs_a
x2 -> cs_b
x4 -> cx + 3*cs_b

*/

macro_rules! init_ab {
    (B) => {
        concat!(
            "mov 4({dim_arrx}),{x0}", "\n",
        )
    };
}

macro_rules! c_load {
    () => {
        concat!(
            "mov ({dim_arrx}),{x0}", "\n",
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
            "add $8*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}

macro_rules! alpha_scale {
    () => {
        alpha_scale_0!(4,7)
    };
}

macro_rules! load_b {
    (B, $nr:tt, $ni:tt, $K:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $nr, "*8+", $ni, "*8({bx}), %xmm", $r, "\n",
            "shufpd ", "$0, %xmm", $r, ",%xmm", $r, "\n",
        )
    };
}

macro_rules! fmadd_2 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_2!($ni), cr!(0,$ni), 3),
            vfmadd!(1, br_2!($ni), cr!(1,$ni), 3),
        )
    };
}

macro_rules! fmadd_1 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_1!($ni), cr!(0,$ni), 3),
        )
    };
}

def_ukernel_sse!(1, step_2, acc_2, store_2, 2, 2, B, C, ukernel_bbc);

def_ukernel_sse!(1, step_2, acc_2, store_2, 2, 2, B, C, ukernel_2_bbp);
def_ukernel_sse!(1, step_1, acc_1, store_1, 1, 2, B, C, ukernel_1_bbp);
