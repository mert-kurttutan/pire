use seq_macro::seq;
use crate::{TC, TC_SIZE};
use super::VS;
use pire_base::{
    def_ukernel_avx512, mem, b_mem,
    acc_3, acc_2, acc_1, store_3, store_2, store_1,
    init_ab,
    step_3, step_2, step_1,
    init_ab_2, def_ukernel_avx512_2,
};

type TS = f32;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 0.0;

type TA = f32;
type TB = f32;

macro_rules! vs {
    () => { "0x40" };
}

macro_rules! bs {
    () => { "4" }
}

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x20+" , $m) };
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "vcvtph2ps ", $m0, ", %zmm2", "\n",
            "vfmadd231ps %zmm2,%zmm0,%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr,2) => {
        concat!(
            "vcvtph2ps ", $m0, ", %zmm2{{%k1}}", "\n",
            "vfmadd231ps %zmm2,%zmm0,%zmm", $r1, "\n",
        )
    };

    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "vcvtph2ps ", $m0, ", %zmm2", "\n",
            "vaddps %zmm2,%zmm", $r1, ",%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr,1) => {
        concat!(
            "vcvtph2ps ", $m0, ", %zmm2{{%k1}}", "\n",
            "vaddps %zmm2,%zmm", $r1, ",%zmm", $r1, "\n",
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
    () => { "vbroadcastss" };
}

macro_rules! vfmadd {
    ($i:tt, $j:tt, $b_macro:tt) => {
        concat!(
            "vfmadd231ps %zmm", $i, ", %zmm", $b_macro!($j),", %zmm", cr!($i,$j), "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr, B) => {
        concat!(
            "vmovaps ", mem!($m0, concat!("0x40*", $r1)), ", %zmm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vcvtps2ph $0x00, %zmm", $r1, ", ", $m0, "\n",
        )
    };
    (P, $r1:expr, $m0:expr) => {
        concat!(
            "vcvtps2ph $0x00, %zmm", $r1, ", ", $m0, " {{%k1}}\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                vbroadcast!(), " ({alphax}),%zmm1", "\n",
                #(
                    "vmulps %zmm1, %zmm", r, ",%zmm", r, "\n",
                )*
            )
        })
    }
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
            "add $4*", $nr, ", {bx}", "\n",
        )
    };
    ($nr:tt) => {
        concat!(
            "add $4*", $nr, ", {bx}", "\n",
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
    (7) => { 1 };
}

macro_rules! load_b {
    (B, $ni:tt, $b_macro:tt) => {
        concat!(
            vbroadcast!(), " ", b_mem!(B,0,$ni,0), ",%zmm", $b_macro!($ni), "\n",
        )
    };
}

macro_rules! mask_ptr {
    (P, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let $nm = if $m % VS == 0 && $m > 0 { 0xFFFF } else { (1_u16 << ($m % VS)) - 1 };
        let $mask_ptr = (&$nm) as *const u16;
    };
    (C, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let $nm = 0xFFFF_u16;
        let $mask_ptr = (&$nm) as *const u16;
    };
}

macro_rules! load_mask {
    (P) => { "kmovw ({maskx}), %k1" };
    (C) => { "/* {maskx} */" };
}

def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 8, B, B, P, ukernel_3_bbp);
def_ukernel_avx512!(1, step_2, acc_2, store_2, 2, 8, B, B, P, ukernel_2_bbp);
def_ukernel_avx512!(1, step_1, acc_1, store_1, 1, 8, B, B, P, ukernel_1_bbp);

def_ukernel_avx512_2!(1, step_3, acc_3, store_3, 3, 8, 8, 32);