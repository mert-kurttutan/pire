use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_sse, 
    mem, init_ab_avx,
    acc_2, store_2, acc_1, store_1,
    step_2_c, step_1_c,
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

macro_rules! load_mask {
    ($is_partial:tt) => { "" };
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "addpd ", $m0, ",%xmm", $r1, "\n",
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
macro_rules! c_load {
    () => {
        concat!(
            permute_complex!(),
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

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr, B) => {
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

macro_rules! alpha_scale {
    () => {
        concat!(
            "movsd ({alphax}), %xmm0 \n",
            "shufpd ", "$0, %xmm0, %xmm0", "\n",
            "movsd 8({alphax}), %xmm1 \n",
            "shufpd ", "$0, %xmm1, %xmm1", "\n",
        
            complex_mul!(4, 5),
            complex_mul!(6, 7),
            complex_mul!(8, 9),
            complex_mul!(10, 11),
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
            v_to_c!(8, 9),
            v_to_c!(10, 11),
        )
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "movsd ({betax}), %xmm0 \n",
            "shufpd ", "$0, %xmm0, %xmm0", "\n",
            "movsd 8({betax}), %xmm1 \n",
            "shufpd ", "$0, %xmm1, %xmm1", "\n",
        )
    }
}

macro_rules! vzero_kernel {
    () => {vzeroall!(4,11)};
}	

macro_rules! inc_b {
    (S,$nr:tt) => { "add {x1},{bx} \n" };
    (B,$nr:tt) => { "" };
}

macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => { "" };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $16*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}


macro_rules! cr {
    (0,0) => { 4 };
    (1,0) => { 6 };
    (0,1) => { 8 };
    (1,1) => { 10 };

    (0,0,0) => { 4 };
    (1,0,0) => { 6 };
    (0,1,0) => { 8 };
    (1,1,0) => { 10 };

    (0,0,1) => { 5 };
    (1,0,1) => { 7 };
    (0,1,1) => { 9 };
    (1,1,1) => { 11 };
}

macro_rules! dr {
    (0,0) => { 12 };
    (1,0) => { 13 };
    (0,1) => { 14 };
    (1,1) => { 15 };
}

macro_rules! br_2 {
    (0,0) => {2}; (0,1) => {3};
    (1,0) => {2}; (1,1) => {3};
}

macro_rules! br_1 {
    (0,0) => {1}; (0,1) => {2};
    (1,0) => {3}; (1,1) => {1};
}

macro_rules! load_b {
    (S, $nr:tt, 0, $K:tt, $b_macro:tt, $i:tt) => {
        concat!(
            vbroadcast!(), " ({bx}),%xmm", $b_macro!(0,$i), "\n",
            "shufpd $0, %xmm", $b_macro!(0,$i), ", %xmm", $b_macro!(0,$i), "\n",
        )
    };
    (S, $nr:tt, 1, $K:tt, $b_macro:tt, $i:tt) => {
        concat!(
            "prefetcht0 64({bx},{x2},1) \n",
            vbroadcast!(), " ({bx},{x2},1),%xmm", $b_macro!(1,$i), "\n",
            "shufpd $0, %xmm", $b_macro!(1,$i), ", %xmm", $b_macro!(1,$i), "\n",
        )
    };
    (B, $nr:tt, $ni:tt, $K:tt, $b_macro:tt, $i:tt) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $nr, "*16+", $ni, "*16+8*", $i, "({bx}), %xmm", $b_macro!($ni,$i), "\n",
            "shufpd $0, %xmm", $b_macro!($ni,$i), ", %xmm", $b_macro!($ni,$i), "\n",
        )
    };
}

def_ukernel_sse!(1, step_2_c, acc_2, store_2, 2, 2, B, C, ukernel_bbc);

def_ukernel_sse!(1, step_2_c, acc_2, store_2, 2, 2, B, C, ukernel_2_bbp);
def_ukernel_sse!(1, step_1_c, acc_1, store_1, 1, 2, B, C, ukernel_1_bbp);

def_ukernel_sse!(1, step_2_c, acc_2, store_2, 2, 2, S, C, ukernel_bsc);

def_ukernel_sse!(1, step_2_c, acc_2, store_2, 2, 2, S, C, ukernel_2_bsp);
def_ukernel_sse!(1, step_1_c, acc_1, store_1, 1, 2, S, C, ukernel_1_bsp);