use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    prefetch_0, def_ukernel_avx, 
    init_ab_avx,
    def_ukernel_avx_2, init_ab_2,
    mem,
    acc_3, store_3, acc_2, store_2, acc_1, store_1,
};

use super::super::avx::asm_ukernel::{
    loadp_unit, storep_unit,
    mask_and_offset,
    mask_ptr,
    inc_b_k_unroll,
    inc_b,
    load_mask,
    vzeroall,
    load_beta,
    vbroadcast,
    vs, v_i
};


type TS = TC;

const ZERO_SCALAR: TC = TC::ZERO;
const ONE_SCALAR: TC = TC::ONE;

macro_rules! vs {
    () => { "0x20" };
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "vaddpd ", $m0, ",%ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr,1) => {
        concat!(
            "vmaskmovpd ", $m0, ", %ymm2", ", %ymm5",  "\n",
            "vaddpd %ymm5, %ymm", $r1, ", %ymm", $r1, "\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "vmovupd ", $m0, ", %ymm5", "\n",
            complex_mul!(5, 7),
            "vaddpd %ymm5, %ymm", $r1, ", %ymm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr,2) => {
        concat!(
            "vmaskmovpd ", $m0, ", %ymm2", ", %ymm5",  "\n",
            complex_mul!(5, 7),
            "vaddpd %ymm5, %ymm", $r1, ", %ymm", $r1, "\n",
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

macro_rules! c_load_2 {
    () => {
        concat!(
            permute_complex!(),
            "mov ({dim_arrx}),{x0}\n",
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

macro_rules! vbroadcast {
    () => {
        "vbroadcastsd"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $b1:expr, $b2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vfmadd231pd ", "%ymm",$b1, ", %ymm", $r1,", %ymm", $r3, "\n",
            "vfmadd231pd ", "%ymm",$b2, ", %ymm", $r1,", %ymm", $r4, "\n",
        ) 
    };
    ($r1:expr, $b1:expr, $r3:expr) => {
        concat!(
            "vfmadd231pd ", "%ymm",$b1, ", %ymm", $r1,", %ymm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovapd ", mem!($m0, concat!("0x20*", $r1)), ", %ymm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovupd %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (P, $r1:expr, $m0:expr) => {
        concat!(
            "vmaskmovpd %ymm", $r1, ", %ymm2, ", $m0,  "\n",
        )
    };
}

macro_rules! complex_mul {
    ($r0:tt, $rt:tt) => {
        concat!(
            "vpermilpd $0b101, %ymm", $r0, ", %ymm", $rt, "\n",
            "vmulpd %ymm0, %ymm", $r0, ", %ymm", $r0, "\n",
            "vmulpd %ymm1, %ymm", $rt, ", %ymm", $rt, "\n",
            "vaddsubpd %ymm", $rt, ", %ymm", $r0, ", %ymm", $r0, "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    () => {
        concat!(

            "vbroadcastsd ({alphax}), %ymm0 \n",
            "vbroadcastsd 8({alphax}), %ymm1 \n",
            
            complex_mul!(4, 5),
            complex_mul!(6, 7),
            complex_mul!(8, 9),
            complex_mul!(10, 11),
            complex_mul!(12, 13),
            complex_mul!(14, 15),
        )
    }
}


macro_rules! v_to_c {
    ($r0:tt, $r1:tt) => {
        concat!(
            "vpermilpd $0b101, %ymm", $r1, ", %ymm", $r1, "\n",
            "vaddsubpd %ymm", $r1, ", %ymm", $r0, ", %ymm", $r0, "\n",
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
            v_to_c!(12, 13),
            v_to_c!(14, 15),
        )
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "vbroadcastsd ({betax}), %ymm0\n",
            "vbroadcastsd 8({betax}), %ymm1\n",
        )
    }
}


macro_rules! vzero_kernel {
    () => {vzeroall!(4,15)};
}	

macro_rules! inc_b {
    (S,$nr:tt) => {
        "add {x1},{bx} \n"
    };
    (B,$nr:tt) => {
        ""
    };
    ($nr:tt) => {
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

macro_rules! cr {
    (0,0) => { 4 };
    (1,0) => { 6 };
    (2,0) => { 8 };
    (0,1) => { 10 };
    (1,1) => { 12 };
    (2,1) => { 14 };
}

macro_rules! dr {
    (0,0) => { 5 };
    (1,0) => { 7 };
    (2,0) => { 9 };
    (0,1) => { 11 };
    (1,1) => { 13 };
    (2,1) => { 15 };
}

macro_rules! load_b {
    (S, $nr:tt, 0, $K:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx}),%ymm", $r1, "\n",
        )
    };
    (S, $nr:tt, 1, $K:tt, $r1:expr, $r2:expr) => {
        concat!(
            "prefetcht0 64({bx},{x2},1) \n",
            vbroadcast!(), " ({bx},{x2},1),%ymm", $r1, "\n",
        )
    };
    (S, $nr:tt, 2, $K:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx},{x2},2),%ymm", $r1, "\n",
        )
    };
    (B, $nr:tt, $ni:tt, $K:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $nr, "*16+", $ni, "*16({bx}), %ymm", $r1, "\n",
            vbroadcast!(), " ", $K, "*", $nr, "*16+", $ni, "*16+8({bx}), %ymm", $r2, "\n",
        )
    };
}

macro_rules! load_b1 {
    (S, $nr:tt, 0, $K:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ({bx}),%ymm", $r, "\n",
        )
    };
    (S, $nr:tt, 1, $K:tt, $r:expr) => {
        concat!(
            "prefetcht0 64({bx},{x2},1) \n",
            vbroadcast!(), " ({bx},{x2},1),%ymm", $r, "\n",
        )
    };
    (B, $nr:tt, $ni:tt, $K:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $nr, "*16+", $ni, "*16({bx}), %ymm", $r, "\n",
        )
    };
}


macro_rules! load_b2 {
    (S, $nr:tt, 0, $K:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " 8({bx}),%ymm", $r, "\n",
        )
    };
    (S, $nr:tt, 1, $K:tt, $r:expr) => {
        concat!(
            "prefetcht0 64({bx},{x2},1) \n",
            vbroadcast!(), " 8({bx},{x2},1),%ymm", $r, "\n",
        )
    };
    (B, $nr:tt, $ni:tt, $K:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $nr, "*16+", $ni, "*16+8({bx}), %ymm", $r, "\n",
        )
    };
}

macro_rules! fmadd_3 {
    ($ni:tt,0) => {
        concat!(
            vfmadd!(0, 3, cr!(0, $ni)),
            vfmadd!(1, 3, cr!(1, $ni)),
            vfmadd!(2, 3, cr!(2, $ni)),
        )
    };
    ($ni:tt,1) => {
        concat!(
            vfmadd!(0, 3, dr!(0, $ni)),
            vfmadd!(1, 3, dr!(1, $ni)),
            vfmadd!(2, 3, dr!(2, $ni)),
        )
    };
}

macro_rules! fmadd_2 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, 2, 3, cr!(0, $ni), dr!(0, $ni)),
            vfmadd!(1, 2, 3, cr!(1, $ni), dr!(1, $ni)),
        )
    };
}

macro_rules! fmadd_1 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, 2, 3, cr!(0, $ni), dr!(0, $ni)),
        )
    };
}

// ***************************** 3 ******************************* //
macro_rules! step_3 {
    ($b_layout:tt, $nr:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b1!($b_layout, $nr, n, $K, 3),
                    fmadd_3!(n,0),
                    load_b2!($b_layout, $nr, n, $K, 3),
                    fmadd_3!(n,1),
                )*
            )
        })
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

def_ukernel_avx!(1, step_3, acc_3, store_3, 3, 2, B, P, ukernel_3_bbp);
def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 2, B, P, ukernel_2_bbp);
def_ukernel_avx!(1, step_1, acc_1, store_1, 1, 2, B, P, ukernel_1_bbp);

def_ukernel_avx!(1, step_3, acc_3, store_3, 3, 2, S, C, ukernel_bsc);

def_ukernel_avx!(1, step_3, acc_3, store_3, 3, 2, S, P, ukernel_3_bsp);
def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 2, S, P, ukernel_2_bsp);
def_ukernel_avx!(1, step_1, acc_1, store_1, 1, 2, S, P, ukernel_1_bsp);

def_ukernel_avx_2!(1, step_3, acc_3, store_3, 3, 2, 4, 64);
