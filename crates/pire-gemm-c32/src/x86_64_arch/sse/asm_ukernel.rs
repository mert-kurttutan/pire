use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_sse,
    mem, init_ab_avx,
    acc_2, store_2, acc_1, store_1,
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
            "addps ", $m0, ",%xmm", $r1, "\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "movups ", $m0, ", %xmm5", "\n",
            complex_mul!(5, 7),
            "addps %xmm5, %xmm", $r1, "\n",
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
    () => { "movss" };
}

macro_rules! vfmadd {
    ($r1:expr, $b1:expr, $b2:expr, $r2:expr, $r3:expr, $r4:expr, $r5:expr) => {
        concat!(
            "movups %xmm", $b1, ", %xmm", $r4, "\n",
            "mulps %xmm", $r1, ", %xmm", $r4, "\n",
            "addps %xmm", $r4, ", %xmm", $r2, "\n",

            "movups %xmm", $b2, ", %xmm", $r5, "\n",
            "mulps %xmm", $r1, ", %xmm", $r5, "\n",
            "addps %xmm", $r5, ", %xmm", $r3, "\n",
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

macro_rules! complex_mul {
    ($r0:tt, $rt:tt) => {
        concat!(
            "movaps %xmm", $r0, ", %xmm", $rt, "\n",
            "shufps $0xb1, %xmm", $r0, ", %xmm", $rt, "\n",
            "mulps %xmm0, %xmm", $r0, "\n",
            "mulps %xmm1, %xmm", $rt, "\n",
            "addsubps %xmm", $rt, ", %xmm", $r0, "\n",
        )
    };
}

macro_rules! alpha_scale {
    () => {
        concat!(
            "movss ({alphax}), %xmm0 \n",
            "shufps ", "$0, %xmm0, %xmm0", "\n",
            "movss 4({alphax}), %xmm1 \n",
            "shufps ", "$0, %xmm1, %xmm1", "\n",
        
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
            "shufps $0xb1, %xmm", $r1, ", %xmm", $r1, "\n",
            "addsubps %xmm", $r1, ", %xmm", $r0, "\n",
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
            "movss ({betax}), %xmm0 \n",
            "shufps ", "$0, %xmm0, %xmm0", "\n",
            "movss 4({betax}), %xmm1 \n",
            "shufps ", "$0, %xmm1, %xmm1", "\n",
        )
    }
}

macro_rules! vzero_kernel {
    () => {vzeroall!(4,11)};
}	

macro_rules! inc_b {
    (S,$nr:tt) => {
        "add {x1},{bx} \n"
    };
    (B,$nr:tt) => {
        ""
    };
}

macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => { "" };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $8*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}

macro_rules! cr {
    (0,0) => { 4 };
    (1,0) => { 6 };
    (0,1) => { 8 };
    (1,1) => { 10 };
}

macro_rules! dr {
    (0,0) => { 5 };
    (1,0) => { 7 };
    (0,1) => { 9 };
    (1,1) => { 11 };
}


macro_rules! load_b {
    (S, $nr:tt, 0, $K:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx}),%xmm", $r1, "\n",
            "shufps $0, %xmm", $r1, ", %xmm", $r1, "\n",
        )
    };
    (S, $nr:tt, 1, $K:tt, $r1:expr, $r2:expr) => {
        concat!(
            "prefetcht0 64({bx},{x2},1) \n",
            vbroadcast!(), " ({bx},{x2},1),%xmm", $r1, "\n",
            "shufps $0, %xmm", $r1, ", %xmm", $r1, "\n",
        )
    };
    (B, $nr:tt, $ni:tt, $K:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $nr, "*8+", $ni, "*8({bx}), %xmm", $r1, "\n",
            "shufps $0, %xmm", $r1, ", %xmm", $r1, "\n",
            vbroadcast!(), " ", $K, "*", $nr, "*8+", $ni, "*8+4({bx}), %xmm", $r2, "\n",
            "shufps $0, %xmm", $r2, ", %xmm", $r2, "\n",
        )
    };
}

macro_rules! fmadd_2 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, 2, 3, cr!(0,$ni), dr!(0,$ni), 12, 13),
            vfmadd!(1, 2, 3, cr!(1,$ni), dr!(1,$ni), 14, 15),
        )
    };
}

macro_rules! fmadd_1 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, 2, 3, cr!(0,$ni), dr!(0,$ni), 12, 13),
        )
    };
}

// ***************************** 2 ******************************* //
macro_rules! step_2 {
    ($b_layout:tt, $nr:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, $nr, n, $K, 2, 3),
                    fmadd_2!(n),
                )*
            )
        })
    };
}

// ***************************** 1 ******************************* //
macro_rules! step_1 {
    ($b_layout:tt, $nr:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, $nr, n, $K, 2, 3),
                    fmadd_1!(n),
                )*
            )
        })
    };
}

def_ukernel_sse!(1, step_2, acc_2, store_2, 2, 2, B, C, ukernel_bbc);

def_ukernel_sse!(1, step_2, acc_2, store_2, 2, 2, B, C, ukernel_2_bbp);
def_ukernel_sse!(1, step_1, acc_1, store_1, 1, 2, B, C, ukernel_1_bbp);

def_ukernel_sse!(1, step_2, acc_2, store_2, 2, 2, S, C, ukernel_bsc);


def_ukernel_sse!(1, step_2, acc_2, store_2, 2, 2, S, C, ukernel_2_bsp);
def_ukernel_sse!(1, step_1, acc_1, store_1, 1, 2, S, C, ukernel_1_bsp);
