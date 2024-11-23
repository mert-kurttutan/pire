use seq_macro::seq;
use std::arch::asm;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE, UnaryFnC};
use pire_base::{
    c_mem, def_ukernel_avx512, def_ukernel_avx512_2,
    acc_3x8, acc_2x12, acc_1x12, store_3x8, store_2x12, store_1x12,
    c_reg_3x8, c_reg_2x12, c_reg_1x12, init_ab, b_num_3x8, b_num_2x12, b_num_1x12,
    fmadd_3x8, fmadd_2x12, fmadd_1x12, b_reg, load_a_avx512, storep_avx512, acc_p_avx512,
    init_ab_2,
};

type TS = TC;

const ZERO: f64 = 0.0;

const ZERO_SCALAR: f64 = 0.0;
const ONE_SCALAR: f64 = 1.0;


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
    () => {
        "vbroadcastsd"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231pd %zmm", $r1, ", %zmm", $r2,", %zmm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovapd ", $m0, ",%zmm", $r1, "\n",
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
            "vxorpd %ymm3,%ymm3,%ymm3\n",
            "vucomisd %xmm3,%xmm0\n",
        )
    }
}

macro_rules! vzero_kernel {
    () => { vzeroall!(8,31) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(8,31) };
}

macro_rules! inc_a {
    ($mr:tt) => {
        concat!(
            "add $64*", $mr, ", {ax}", "\n",
        )
    };
}

macro_rules! inc_b {
    (S,12) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n"
    };
    (S,11) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n"
    };
    (S,10) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n"
    };
    (S,9) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n"
    };
    (S,8) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n"
    };
    (S,7) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n"
    };
    (S,6) => {
        "add {x1},{bx} \n add {x1},{x3} \n"
    };
    (S,5) => {
        "add {x1},{bx} \n add {x1},{x3} \n"
    };
    (S,4) => {
        "add {x1},{bx} \n add {x1},{x3} \n"
    };
    (S,3) => {
        "add {x1},{bx} \n"
    };
    (S,2) => {
        "add {x1},{bx} \n"
    };
    (S,1) => {
        "add {x1},{bx} \n"
    };
    (B,$nr:tt) => {
        concat!(
            "add $8*", $nr, ", {bx}", "\n",
        )
    };
}

macro_rules! load_b {
    (S, $ni:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", b_reg!($ni), ",%zmm", $r, "\n",
        )
    };
    (B, $ni:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", $ni, "*8({bx}), %zmm", $r, "\n",
        )
    };
}

// ***************************** 3x8 ******************************* //
macro_rules! step_3x8 {
    (8, B) => {
        concat!(
            load_a_avx512!(3),
            "addq $192, {ax} \n",
            load_b!(B, 0, 3),
            fmadd_3x8!(0),
            load_b!(B, 1, 4),
            fmadd_3x8!(1),
            "prefetcht0 384({ax}) \n",
            "prefetcht0 64({bx}) \n",
            load_b!(B, 2, 5),
            fmadd_3x8!(2),
            load_b!(B, 3, 6),
            fmadd_3x8!(3),
            "prefetcht0 448({ax}) \n",
            load_b!(B, 4, 7),
            fmadd_3x8!(4),
            load_b!(B, 5, 3),
            fmadd_3x8!(5),
            "prefetcht0 512({ax}) \n",
            load_b!(B, 6, 4),
            fmadd_3x8!(6),
            load_b!(B, 7, 5),
            fmadd_3x8!(7),
            "addq $64, {bx} \n",
        )
    };
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx512!(3),
                inc_a!(3),
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, b_num_3x8!(n)),
                    fmadd_3x8!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 2x12 ******************************* //
macro_rules! step_2x12 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx512!(2),
                inc_a!(2),
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, b_num_2x12!(n)),
                    fmadd_2x12!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 1x12 ******************************* //
macro_rules! step_1x12 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx512!(1),
                inc_a!(1),
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, b_num_1x12!(n)),
                    fmadd_1x12!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
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

def_ukernel_avx512!(1, step_3x8, acc_3x8, store_3x8, 3, 8, B, P, ukernel_3_bbp);
def_ukernel_avx512!(1, step_2x12, acc_2x12, store_2x12, 2, 8, B, P, ukernel_2_bbp);
def_ukernel_avx512!(1, step_1x12, acc_1x12, store_1x12, 1, 8, B, P, ukernel_1_bbp);

def_ukernel_avx512!(1, step_3x8, acc_3x8, store_3x8, 3, 8, S, C, ukernel_bsc);

def_ukernel_avx512!(1, step_3x8, acc_3x8, store_3x8, 3, 8, S, P, ukernel_3_bsp);
def_ukernel_avx512!(1, step_2x12, acc_2x12, store_2x12, 2, 8, S, P, ukernel_2_bsp);
def_ukernel_avx512!(1, step_1x12, acc_1x12, store_1x12, 1, 8, S, P, ukernel_1_bsp);


def_ukernel_avx512_2!(1, step_3x8, acc_3x8, store_3x8, 3, 8, 8, 64);
