use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_avx, 
    init_ab_avx,
    mem,
    acc_2, store_2, acc_1, store_1,
};

type TS = TC;

const ZERO_SCALAR: TC = TC::ZERO;
const ONE_SCALAR: TC = TC::ONE;

macro_rules! vs {
    () => { "0x20" };
}
pub(crate) use vs;

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x20+" , $m) };
}
pub(crate) use v_i;

#[inline(always)]
pub(crate) fn mask_and_offset(m: usize) -> ([u64;8], usize) {
    let mask: [u64; 8] = [
        u64::MAX, u64::MAX, u64::MAX, u64::MAX,
        0, 0, 0, 0,
    ];
    let mask_offset = if m % VS == 0 { 0 } else { VS - (m %VS)};
    // mask offset for 128 bit part of ymm register
    // since c64 is 128 bit wide
    (mask, mask_offset*2)
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
    (P) => { "vmovdqu ({maskx}), %ymm2" };
    (C) => { "/* {maskx} */" }
}

pub(crate) use load_mask;

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

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("vpxor %ymm",r,",%ymm",r,",%ymm",r,"\n",)*)
        })
    }
}
pub(crate) use vzeroall;

macro_rules! vbroadcast {
    () => { "vbroadcastsd" };
}
pub(crate) use vbroadcast;

macro_rules! vfmadd {
    ($r1:expr, $b1:expr, $b2:expr, $r2:expr, $r3:expr, $r4:expr, $r5:expr) => {
        concat!(
            "vmulpd %ymm", $r1, ", %ymm", $b1,", %ymm", $r4, "\n",
            "vaddpd %ymm", $r4, ", %ymm", $r2, ", %ymm", $r2, "\n",

            "vmulpd %ymm", $r1, ", %ymm", $b2,", %ymm", $r5, "\n",
            "vaddpd %ymm", $r5, ", %ymm", $r3, ", %ymm", $r3, "\n",
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
pub(crate) use loadp_unit;

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
pub(crate) use storep_unit;

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

macro_rules! alpha_scale {
    () => {
        concat!(
            "vbroadcastsd ({alphax}), %ymm0 \n",
            "vbroadcastsd 8({alphax}), %ymm1 \n",
            
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
pub(crate) use load_beta;

macro_rules! vzero_kernel {
    () => {vzeroall!(4,11)};
}
pub(crate) use vzero_kernel;

macro_rules! inc_b {
    (S,$nr:tt) => { "add {x1},{bx} \n" };
    (B,$nr:tt) => { "" };
}
pub(crate) use inc_b;
macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => { "" };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $16*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}
pub(crate) use inc_b_k_unroll;

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
    (B, $nr:tt, $ni:tt, $K:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $nr, "*16+", $ni, "*16({bx}), %ymm", $r1, "\n",
            vbroadcast!(), " ", $K, "*", $nr, "*16+", $ni, "*16+8({bx}), %ymm", $r2, "\n",
        )
    };
}

macro_rules! fmadd_2 {
    (0) => {
        concat!(
            vfmadd!(0, 2, 3, 4, 5, 12, 13),
            vfmadd!(1, 2, 3, 6, 7, 14, 15),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 3, 8, 9, 12, 13),
            vfmadd!(1, 2, 3, 10, 11, 14, 15),
        )
    };
}

macro_rules! fmadd_1 {
    (0) => {
        concat!(
            vfmadd!(0, 2, 3, 4, 5, 12, 13),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 3, 6, 7, 14, 15),
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

macro_rules! cr_2 {
    (0,0) => { 4 };
    (1,0) => { 6 };
    (0,1) => { 8 };
    (1,1) => { 10 };
}

macro_rules! cr_1 {
    (0,0) => { 4 };
    (0,1) => { 6 };
}

def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 2, B, C, ukernel_bbc);

def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 2, B, P, ukernel_2_bbp);
def_ukernel_avx!(1, step_1, acc_1, store_1, 1, 2, B, P, ukernel_1_bbp);

def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 2, S, C, ukernel_bsc);

def_ukernel_avx!(1, step_2, acc_2, store_2, 2, 2, S, P, ukernel_2_bsp);
def_ukernel_avx!(1, step_1, acc_1, store_1, 1, 2, S, P, ukernel_1_bsp);
