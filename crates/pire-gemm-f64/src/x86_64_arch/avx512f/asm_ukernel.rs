use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_avx512, def_ukernel_avx512_2,
    acc_3, acc_2, acc_1, store_3, store_2, store_1,
    init_ab, b_mem,
    step_3, step_2, step_1,
    init_ab_2, mem,
};

type TS = TC;

const ZERO_SCALAR: f64 = 0.0;
const ONE_SCALAR: f64 = 1.0;

macro_rules! vs {
    () => { "0x40" };
}

macro_rules! bs {
    () => { "8" };
}
macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x40+" , $m) };
}

macro_rules! mask_ptr {
    (P, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let $nm = if $m % VS == 0 && $m > 0 { 0xFF } else { (1_u8 << ($m % VS)) - 1 };
        let $mask_ptr = (&$nm) as *const u8;
    };
    (C, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let $nm = 0xFF_u8;
        let $mask_ptr = (&$nm) as *const u8;
    };
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vfmadd231pd ", $m0, ",%zmm0,%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vmovupd ", $m0, ", %zmm1 {{%k1}}", "\n",
            "vfmadd231pd %zmm1,%zmm0,%zmm", $r1, "\n",
        )
    };

    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vaddpd ", $m0, ",%zmm", $r1, ",%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vaddpd ", $m0, ",%zmm", $r1, ",%zmm", $r1, "{{%k1}}\n",
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

macro_rules! c_load_2 {
    () => {
        concat!(
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
            concat!(#("vpxorq %zmm",r,",%zmm",r,",%zmm",r,"\n",)*)
        })
    }
}

macro_rules! vbroadcast {
    () => { "vbroadcastsd" };
}

macro_rules! vfmadd {
    ($i:tt, $j:tt, $b_macro:tt) => {
        concat!(
            "vfmadd231pd %zmm", $i, ", %zmm", $b_macro!($j),", %zmm", cr!($i,$j), "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovapd ", mem!($m0, concat!("0x40*", $r1)), ", %zmm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovupd %zmm", $r1, ", ", $m0, "\n",
        )
    };
    (P, $r1:expr, $m0:expr) => {
        concat!(
            "vmovupd %zmm", $r1, ", ", $m0, " {{%k1}}\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                vbroadcast!(), " ({alphax}), %zmm1", "\n",
                #(
                    "vmulpd %zmm1, %zmm", r, ",%zmm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! load_mask {
    (P) => { "kmovw ({maskx}), %k1" };
    (C) => { "/* {maskx} */" };
}

macro_rules! load_beta {
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %zmm0\n",
        )
    }
}

macro_rules! vzero_kernel {
    () => { vzeroall!(8,31) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(8,31) };
}

macro_rules! inc_b {
    (B,$nr:tt) => {
        concat!(
            "add $8*", $nr, ", {bx}", "\n",
        )
    };
    ($nr:tt) => {
        concat!(
            "add $8*", $nr, ", {bx}", "\n",
        )
    };
    (S,$nr:tt) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n"
    };
}

macro_rules! prefetch {
    (B, 0) => {
        "prefetcht0 768({bx})\n"
    };
    (B, 1) => {
        "prefetcht0 768({bx})\n"
    };
    (B, 2) => {
        "prefetcht0 768+64({ax})\n"
    };
    (B, 3) => {
        "prefetcht0 768+128({ax})\n"
    };
    ($b_layout:tt, $ni:tt) => {
        ""
    };
}

macro_rules! load_b {
    ($b_layout:tt, $ni:tt, $b_macro:tt) => {
        concat!(
            prefetch!($b_layout, $ni),
            vbroadcast!(), " ", b_mem!($b_layout,0,$ni,0), ",%zmm", $b_macro!($ni), "\n",
        )
    };
}

macro_rules! cr {
    (0,0) => { 8 };
    (1,0) => { 9 };
    (2,0) => { 10 };
    (0,1) => { 11 };
    (1,1) => { 12 };
    (2,1) => { 13 };
    (0,2) => { 14 };
    (1,2) => { 15 };
    (2,2) => { 16 };
    (0,3) => { 17 };
    (1,3) => { 18 };
    (2,3) => { 19 };
    (0,4) => { 20 };
    (1,4) => { 21 };
    (2,4) => { 22 };
    (0,5) => { 23 };
    (1,5) => { 24 };
    (2,5) => { 25 };
    (0,6) => { 26 };
    (1,6) => { 27 };
    (2,6) => { 28 };
    (0,7) => { 29 };
    (1,7) => { 30 };
    (2,7) => { 31 };
}

macro_rules! br_3 {
    (0) => { 3 };
    (1) => { 4 };
    (2) => { 5 };
    (3) => { 6 };
    (4) => { 7 };
    (5) => { 3 };
    (6) => { 4 };
    (7) => { 5 };
}

macro_rules! br_2 {
    (0) => { 2 };
    (1) => { 3 };
    (2) => { 4 };
    (3) => { 5 };
    (4) => { 6 };
    (5) => { 7 };
    (6) => { 2 };
    (7) => { 3 };
}

macro_rules! br_1 {
    (0) => { 1 };
    (1) => { 2 };
    (2) => { 3 };
    (3) => { 4 };
    (4) => { 5 };
    (5) => { 6 };
    (6) => { 7 };
    (7) => { 9 };
}


def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 8, B, P, ukernel_3_bbp);
def_ukernel_avx512!(1, step_2, acc_2, store_2, 2, 8, B, P, ukernel_2_bbp);
def_ukernel_avx512!(1, step_1, acc_1, store_1, 1, 8, B, P, ukernel_1_bbp);

def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 8, S, C, ukernel_bsc);

def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 8, S, P, ukernel_3_bsp);
def_ukernel_avx512!(1, step_2, acc_2, store_2, 2, 8, S, P, ukernel_2_bsp);
def_ukernel_avx512!(1, step_1, acc_1, store_1, 1, 8, S, P, ukernel_1_bsp);

def_ukernel_avx512_2!(1, step_3, acc_3, store_3, 3, 8, 8, 64);
