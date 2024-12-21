use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    mem, def_ukernel_sse,
    acc_1, store_1,
    step_1_c,
};
type TS = TC;

const ZERO_SCALAR: TC = TC::ZERO;
const ONE_SCALAR: TC = TC::ONE;

macro_rules! vs {
    () => { "0x10" };
}

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x10+" , $m) };
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "movupd ", $m0, ",%xmm2", "\n",
            "addpd ", "%xmm2", ",%xmm", $r1, "\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "movupd ", $m0, ", %xmm5", "\n",
            complex_mul!(5, 7),
            "addpd %xmm5, %xmm", $r1, "\n",
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
    ($i:tt, $j:tt, $b_macro:tt, 0) => {
        concat!(
            "movupd %xmm", $b_macro!($j,0), ", %xmm", dr!($i,$j), "\n",
            "mulpd %xmm", $i, ", %xmm", dr!($i,$j), "\n",
            "addpd %xmm", dr!($i,$j), ", %xmm", cr!($i,$j), "\n",
        ) 
    };
    ($i:tt, $j:tt, $b_macro:tt, 1) => {
        concat!(
            "movupd %xmm", $b_macro!($j,1), ", %xmm", dr!($i,$j), "\n",
            "mulpd %xmm", $i, ", %xmm", dr!($i,$j), "\n",
            "addpd %xmm", dr!($i,$j), ", %xmm", cr!($i,$j,1), "\n",
        ) 
    };
}

macro_rules! complex_mul {
    ($r0:tt, $rt:tt) => {
        concat!(
            "movapd %xmm", $r0, ", %xmm", $rt, "\n",
            "shufpd $0b101, %xmm", $r0, ", %xmm", $rt, "\n",
            "mulpd %xmm0, %xmm", $r0, "\n",
            "mulpd %xmm1, %xmm", $rt, "\n",
            "addsubpd %xmm", $rt, ", %xmm", $r0, "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    () => {
        concat!(
            "mov ({ptr_arrx}), {bx}\n",
            vbroadcast!(), " 0({bx}),%xmm0", "\n",
            "shufpd ", "$0, %xmm0, %xmm0", "\n",

            vbroadcast!(), " 8({bx}),%xmm1", "\n",
            "shufpd ", "$0, %xmm1, %xmm1", "\n",
        
            complex_mul!(4, 5),
            complex_mul!(6, 7),
        )
    }
}

macro_rules! v_to_c {
    ($r0:tt, $r1:tt) => {
        concat!(
            "shufpd $0b101, %xmm", $r1, ", %xmm", $r1, "\n",
            "addsubpd %xmm", $r1, ", %xmm", $r0, "\n",
        )
    }
}

macro_rules! permute_complex {
    () => {
        concat!(
            // permute even and odd elements
            v_to_c!(4, 5),
            v_to_c!(6, 7),
        )
    }
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

macro_rules! load_beta {
    () => {
        concat!(
            "mov 4({ptr_arrx}), {bx}\n",
            "movsd ({bx}), %xmm0 \n",
            "shufpd ", "$0, %xmm0, %xmm0", "\n",
            "movsd 8({bx}), %xmm1 \n",
            "shufpd ", "$0, %xmm1, %xmm1", "\n",
        )
    }
}


// only non contigous along m and n direction which is not changed frequently during iteration along k direction
/*

x1 -> cs_a
x2 -> cs_b
x3 -> ax + 3*cs_a
x4 -> bx + 3*cs_b

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
            permute_complex!(),
            "mov ({dim_arrx}),{x0}", "\n",
        )
    };
}


macro_rules! vzero_kernel {
    () => { vzeroall!(4,7) };
}

macro_rules! inc_b {
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
            "add $16*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}


macro_rules! alpha_scale {
    () => {
        alpha_scale_0!()
    };
}

macro_rules! br_1 {
    (0,0) => {1}; (0,1) => {2};
    (1,0) => {1}; (1,1) => {2};
}

macro_rules! cr {
    (0,0) => { 4 };
    (0,1) => { 6 };

    (0,0,0) => { 4 };
    (0,1,0) => { 6 };

    (0,0,1) => { 5 };
    (0,1,1) => { 7 };
}

macro_rules! dr {
    (0,0) => { 3 };
    (0,1) => { 3 };
}

macro_rules! load_b {
    (B, $nr:tt, $ni:tt, $K:tt, $b_macro:tt, $i:tt) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $nr, "*16+", $ni, "*16+8*", $i, "({bx}), %xmm", $b_macro!($ni,$i), "\n",
            "shufpd $0, %xmm", $b_macro!($ni,$i), ", %xmm", $b_macro!($ni,$i), "\n",
        )
    };
}

def_ukernel_sse!(1, step_1_c, acc_1, store_1, 1, 2, B, C, ukernel_bbc);

def_ukernel_sse!(1, step_1_c, acc_1, store_1, 1, 2, B, C, ukernel_1_bbp);

