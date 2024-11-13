use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC, UnaryFnC, TC_SIZE};
use glar_base::{c_mem, def_ukernel_sse, mem};
use super::VS;

type TS = TC;

const ZERO: f32 = 0.0;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

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
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "movups %xmm", $r2, ", %xmm", $r4, "\n",
            "mulps %xmm", $r1, ", %xmm", $r4, "\n",
            "addps %xmm", $r4, ", %xmm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "movaps ", $m0, ",%xmm", $r1, "\n",
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
                "mov ({ptr_arrx}), {bx}\n",
                vbroadcast!(), " ({bx}),%xmm1", "\n",
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
            "mov 4({ptr_arrx}), {ax}\n",
            vbroadcast!(), " ({ax}), %xmm0\n",
            "shufps ", "$0, %xmm0, %xmm0", "\n",
            "xorps %xmm3,%xmm3\n",
            "ucomiss %xmm3,%xmm0\n",
        )
    }
}

macro_rules! acc_p {
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1, $q),
            beta_fmadd!($layout, mem!($m0, "0x10"), $r2, $q),
        )
    };
    ($layout:tt, $m0:expr, $q:tt, $r1:expr) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1, $q),
        )
    };
}


macro_rules! loadp {
    (2, $m0:expr) => {
        concat!(
            loadp_unit!($m0, 0),
            loadp_unit!(mem!($m0, "0x10"), 1),
        )
    };
    (1, $m0:expr) => {
        concat!(
            loadp_unit!($m0, 0),
        )
    };
}

macro_rules! storep {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            storep_unit!(C, $r1, $m0),
            storep_unit!($layout, $r2, mem!($m0, "0x10")),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            storep_unit!($layout, $r1, $m0),
        )
    };
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
            "mov 12({dim_arrx}),{x0}", "\n",
        )
    };
    (S) => {
        concat!(
            // mov cs_b to reg
            "mov 12({dim_arrx}),{x0}", "\n",
        )
    };
}


macro_rules! c_load {
    (2) => {
        concat!(
            "mov 8({dim_arrx}),{x0}", "\n",
        )
    };
    (1) => {
        concat!(
            "mov 8({dim_arrx}),{x0}", "\n",
        )
    };
}


macro_rules! vzero_kernel {
    () => { vzeroall!(4,7) };
}

macro_rules! inc_b {
    (S,2) => {
        "add {x0},{bx} \n"
    };
    (S,1) => {
        "add {x0},{bx} \n"
    };
    (B,$nr:tt) => {
        ""
    };
}

macro_rules! inc_a_k_unroll {
    ($X:tt, $K:tt) => {
        concat!(
            "add $16*", $K, "*", $X, ",{ax}", "\n",
        )
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
    (2, 2) => {
        alpha_scale_0!(4,7)
    };
    (2, 1) => {
        alpha_scale_0!(4,5)
    };

    (1, 2) => {
        alpha_scale_0!(4,5)
    };
    (1, 1) => {
        alpha_scale_0!(4,4)
    };
}

macro_rules! c_reg_2x2 {
    (0,0) => { 4 };
    (1,0) => { 5 };
    (0,1) => { 6 };
    (1,1) => { 7 };
}

macro_rules! c_reg_1x2 {
    (0,0) => { 4 };
    (0,1) => { 5 };
}


macro_rules! acc_2x2 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!($layout, c_mem!($ni), $q, c_reg_2x2!(0,$ni), c_reg_2x2!(1,$ni))
    };
}

macro_rules! store_2x2 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_2x2!(0,$ni), c_reg_2x2!(1,$ni))
    };
}

macro_rules! acc_1x2 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!($layout, c_mem!($ni), $q, c_reg_1x2!(0,$ni))
    };
}

macro_rules! store_1x2 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_1x2!(0,$ni))
    };
}

macro_rules! load_b {
    (S, 0, $K:tt, $X:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ({bx}),%xmm", $r, "\n",
            "shufps ", "$0, %xmm", $r, ",%xmm", $r, "\n",
        )
    };
    (S, 1, $K:tt, $X:tt, $r:expr) => {
        concat!(
            // "prefetcht0 64({cx},{x2},1) \n",
            // vbroadcast!(), " ({cx},{x2},1),%xmm", $r, "\n",
            "shufps ", "$0, %xmm", $r, ",%xmm", $r, "\n",
        )
    };
    (B, $N:tt, $K:tt, $X:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*4+", $N, "*4({bx}), %xmm", $r, "\n",
            "shufps ", "$0, %xmm", $r, ",%xmm", $r, "\n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt, $K:tt) => {
        loadp!($mr, concat!($mr,"*16*",$K,"({ax})"))
    };
}

macro_rules! fmadd_2v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 4, 3),
            vfmadd!(1, 2, 5, 3),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 6, 3),
            vfmadd!(1, 2, 7, 3),
        )
    };
}

macro_rules! fmadd_1v {
    (0) => {
        concat!(vfmadd!(0, 1, 4, 6))
    };
    (1) => {
        concat!(vfmadd!(0, 2, 5, 7))
    };
}

macro_rules! b_num_2x2 {
    (0) => {2};
    (1) => {2};
}

macro_rules! b_num_1x2 {
    (0) => {1};
    (1) => {2};
}

// ***************************** 2x2 ******************************* //
macro_rules! step_2x2 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_2x2!(n)),
                    fmadd_2v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 1x2 ******************************* //
macro_rules! step_1x2 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(1, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_1x2!(n)),
                    fmadd_1v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

def_ukernel_sse!(1, step_2x2, acc_2x2, store_2x2, 2, 2, 2, 3, B, C, ukernel_bbc);

def_ukernel_sse!(1, step_2x2, acc_2x2, store_2x2, 2, 2, 2, 3, B, C, ukernel_2_bbp);
def_ukernel_sse!(1, step_1x2, acc_1x2, store_1x2, 1, 2, 2, 3, B, C, ukernel_1_bbp);


def_ukernel_sse!(1, step_2x2, acc_2x2, store_2x2, 2, 2, 1, 2, B, C, ukernel_n_bbc);

def_ukernel_sse!(1, step_2x2, acc_2x2, store_2x2, 2, 2, 1, 2, B, C, ukernel_2xn_bbp);
def_ukernel_sse!(1, step_1x2, acc_1x2, store_1x2, 1, 2, 1, 2, B, C, ukernel_1xn_bbp);

