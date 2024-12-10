use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_avx512, def_ukernel_avx512_2,
    acc_2, acc_1, store_2, store_1,
    step_2, step_1,
    mem,
};

type TS = f32;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

macro_rules! dr_2 {
    (0,0) => { 24 };
    (1,0) => { 25 };
    (0,1) => { 26 };
    (1,1) => { 27 };
    (0,2) => { 28 };
    (1,2) => { 29 };
    (0,3) => { 30 };
    (1,3) => { 24 };
    (0,4) => { 25 };
    (1,4) => { 26 };
    (0,5) => { 27 };
    (1,5) => { 28 };
    (0,6) => { 29 };
    (1,6) => { 30 };
    (0,7) => { 24 };
    (1,7) => { 25 };
}

macro_rules! dr_1 {
    (0,0) => { 24 };
    (0,1) => { 25 };
    (0,2) => { 26 };
    (0,3) => { 27 };
    (0,4) => { 28 };
    (0,5) => { 29 };
    (0,6) => { 30 };
    (0,7) => { 24 };
    (0,8) => { 25 };
    (0,9) => { 26 };
    (0,10) => { 27 };
    (0,11) => { 28 };
}


macro_rules! cr_2 {
    (0,0) => { 8 };
    (1,0) => { 9 };
    (0,1) => { 10 };
    (1,1) => { 11 };
    (0,2) => { 12 };
    (1,2) => { 13 };
    (0,3) => { 14 };
    (1,3) => { 15 };
    (0,4) => { 16 };
    (1,4) => { 17 };
    (0,5) => { 18 };
    (1,5) => { 19 };
    (0,6) => { 20 };
    (1,6) => { 21 };
    (0,7) => { 22 };
    (1,7) => { 23 };
}

macro_rules! cr_1 {
    (0,0) => { 9 };
    (0,1) => { 10 };
    (0,2) => { 11 };
    (0,3) => { 12 };
    (0,4) => { 13 };
    (0,5) => { 14 };
    (0,6) => { 15 };
    (0,7) => { 16 };
    (0,8) => { 17 };
    (0,9) => { 18 };
    (0,10) => { 19 };
    (0,11) => { 20 };
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
    (7) => { 8 };
}

macro_rules! vs {
    () => { "0x40" };
}

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x40+" , $m) };
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r:expr, 1) => {
        concat!(
            "vpaddd ", $m0, ", %zmm", $r, ", %zmm", $r, "\n",
        ) 
    };
    (C, $m0:expr, $r:expr, 2) => {
        concat!(
            "vcvtdq2ps %zmm", $r, ",%zmm", $r, "\n",
            "vcvtdq2ps ", $m0, ",%zmm30", "\n",
            "vfmadd231ps %zmm30,%zmm0,%zmm", $r, "\n",
            "vcvtps2dq %zmm", $r, ",%zmm", $r, "\n",
        ) 
    };
    (P, $m0:expr, $r:expr, 1) => {
        concat!(
            "vmovups ", $m0, ", %zmm30 {{%k1}}", "\n",
            "vpaddd %zmm30, %zmm", $r, ", %zmm", $r, "\n",
        ) 
    };

    (P, $m0:expr, $r:expr, 2) => {
        concat!(
            "vcvtdq2ps ", $m0, ", %zmm30 {{%k1}}", "\n",
            "vcvtdq2ps %zmm", $r, ",%zmm", $r, "\n",
            "vfmadd231ps %zmm30,%zmm0,%zmm", $r, "\n",
            "vcvtps2dq %zmm", $r, ",%zmm", $r, "\n",
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
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vpmaddubsw %zmm", $r1, ", %zmm", $r2, ", %zmm", $r4, "\n",
            "vpmaddwd %zmm", $r4, ", %zmm31", ", %zmm", $r4, "\n",
            "vpaddd %zmm", $r4, ", %zmm", $r3, ", %zmm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovaps ", mem!($m0, concat!("0x40*", $r1)), ", %zmm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %zmm", $r1, ", ", $m0,  "\n",
        )
    };
    (P, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %zmm", $r1, ", ", $m0, " {{%k1}}\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "vbroadcastss ({alphax}),%zmm1", "\n",
                #(
                    "vcvtdq2ps %zmm", r, ",%zmm", r, "\n",
                    "vmulps %zmm1, %zmm", r, ",%zmm", r, "\n",
                    "vcvtps2dq %zmm", r, ",%zmm", r, "\n",
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
            "movw $0x1, {x5:x}", "\n",
            "vpbroadcastw {x5:e}, %zmm31", "\n",
            "/* {x4} */", "\n",
            "/* {x3} */", "\n",
            "/* {x2} */", "\n",
            "/* {x1} */", "\n",
            "mov 24({dim_arrx}),{x0}", "\n",
        )
    };
    (S) => { "" };
}

macro_rules! init_ab_2 {
    (B) => {
        concat!(
            "movw $0x1, {x5:x}", "\n",
            "vpbroadcastw {x5:e}, %zmm31", "\n",
            "/* {x4} */", "\n",
            "/* {x3} */", "\n",
            "/* {x2} */", "\n",
            "/* {x1} */", "\n",
            "mov 8({dim_arrx}),{x0}", "\n",
        )
    };
    (S) => { "" };
}

macro_rules! vzero_kernel {
    () => { vzeroall!(8,23) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(8,23) };
}

macro_rules! inc_b {
    (S, $nr:tt) => { "" };
    (B, $nr:tt) => {
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

macro_rules! load_b {
    (B, $N:tt, $r:expr) => {
        concat!(
            vbroadcast!(), "  ", $N, "*4({bx}), %zmm", $r, "\n",
        )
    };
}

macro_rules! fmadd_2 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_2!($ni), cr_2!(0,$ni), dr_2!(0,$ni)),
            vfmadd!(1, br_2!($ni), cr_2!(1,$ni), dr_2!(1,$ni)),
        )
    };
}

macro_rules! fmadd_1 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_1!($ni), cr_1!(0,$ni), dr_1!(0,$ni)),
        )
    };
}

macro_rules! mask_ptr {
    (P, $m:tt, $nm:ident, $mask_ptr:tt) => {
        let $nm = if $m % VS == 0 && $m > 0 { 0xFFFF } else { (1_u16 << ($m % VS)) - 1 };
        let $mask_ptr = (&$nm) as *const u16;
    };
    (C, $m:tt, $nm:ident, $mask_ptr:tt) => {
        let $nm = 0xFFFF_u16;
        let $mask_ptr = (&$nm) as *const u16;
    };
}

macro_rules! load_mask {
    (P) => { "kmovw ({maskx}), %k1" };
    (C) => { "/* {maskx} */" }
}

def_ukernel_avx512!(4, step_2, acc_2, store_2, 2, 8, B, P, ukernel_2_bbp);
def_ukernel_avx512!(4, step_1, acc_1, store_1, 1, 8, B, P, ukernel_1_bbp);

def_ukernel_avx512_2!(4, step_2, acc_2, store_2, 2, 8, 32, 32);
