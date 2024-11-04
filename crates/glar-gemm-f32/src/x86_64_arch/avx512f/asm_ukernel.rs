use seq_macro::seq;
use std::arch::asm;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE, UnaryFnC};
use glar_base::{
    load_buf, store_buf, c_mem, dim_to_reg, def_ukernel_avx512,def_ukernel_avx512_2,
    cum_seq, acc_3x8, acc_2x12, acc_1x12, store_3x8, store_2x12, store_1x12,
    c_reg_3x8, c_reg_2x12, c_reg_1x12, init_ab, b_num_3x8, b_num_2x12, b_num_1x12,
    fmadd_3x8, fmadd_2x12, fmadd_1x12, b_reg, 
    load_a_avx512, storep_avx512, acc_p_avx512, init_ab_2,
};

type TS = TC;

const ZERO: f32 = 0.0;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;


macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vfmadd231ps ", $m0, ",%zmm0,%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vmovups ", $m0, ", %zmm1 {{%k1}}", "\n",
            "vfmadd231ps %zmm1,%zmm0,%zmm", $r1, "\n",
        )
    };

    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vaddps ", $m0, ",%zmm", $r1, ",%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vaddps ", $m0, ",%zmm", $r1, ",%zmm", $r1, "{{%k1}}\n",
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
        "vbroadcastss"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231ps %zmm", $r1, ", %zmm", $r2,", %zmm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovaps ", $m0, ",%zmm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %zmm", $r1, ", ", $m0, "\n",
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
                vbroadcast!(), " ({alphax}),%zmm1", "\n",
                #(
                    "vmulps %zmm1, %zmm", r, ",%zmm", r, "\n",
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
            "vxorps %ymm3,%ymm3,%ymm3\n",
            "vucomiss %xmm3,%xmm0\n",
        )
    }
}

macro_rules! vzero_kernel {
    () => { vzeroall!(8,31) };
}

macro_rules! alpha_scale {
    ($mr:tt,$nr:tt) => { dim_to_reg!(alpha_scale_0, $mr, $nr) };
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
            "add $4*", $nr, ", {bx}", "\n",
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
            vbroadcast!(), " ", $ni, "*4({bx}), %zmm", $r, "\n",
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
            "prefetcht0 128({bx}) \n",
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
            "addq $32, {bx} \n",
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
        let $nm = if $m % VS == 0 && $m > 0 { 0xFFFF } else { (1_u16 << ($m % VS)) - 1 };
        let $mask_ptr = &$nm as *const u16;
    };
    (C, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let $nm = 0xFFFF_u16;
        let $mask_ptr = &$nm as *const u16;
    };
}

def_ukernel_avx512!(1, step_3x8, acc_3x8, store_3x8, 3, 8, 8, 9, B, P, ukernel_3_bbp);
def_ukernel_avx512!(1, step_2x12, acc_2x12, store_2x12, 2, 8, 8, 9, B, P, ukernel_2_bbp);
def_ukernel_avx512!(1, step_1x12, acc_1x12, store_1x12, 1, 8, 8, 9, B, P, ukernel_1_bbp);

def_ukernel_avx512!(1, step_3x8, acc_3x8, store_3x8, 3, 8, 8, 9, S, C, ukernel_bsc);

def_ukernel_avx512!(1, step_3x8, acc_3x8, store_3x8, 3, 8, 8, 9, S, P, ukernel_3_bsp);
def_ukernel_avx512!(1, step_2x12, acc_2x12, store_2x12, 2, 8, 8, 9, S, P, ukernel_2_bsp);
def_ukernel_avx512!(1, step_1x12, acc_1x12, store_1x12, 1, 8, 8, 9, S, P, ukernel_1_bsp);


def_ukernel_avx512!(1, step_3x8, acc_3x8, store_3x8, 3, 8, 1, 8, B, C, ukernel_n_bbc);

def_ukernel_avx512!(1, step_3x8, acc_3x8, store_3x8, 3, 8, 1, 8, B, P, ukernel_3xn_bbp);
def_ukernel_avx512!(1, step_2x12, acc_2x12, store_2x12, 2, 8, 1, 8, B, P, ukernel_2xn_bbp);
def_ukernel_avx512!(1, step_1x12, acc_1x12, store_1x12, 1, 8, 1, 8, B, P, ukernel_1xn_bbp);

def_ukernel_avx512!(1, step_3x8, acc_3x8, store_3x8, 3, 8, 1, 8, S, C, ukernel_n_bsc);

def_ukernel_avx512!(1, step_3x8, acc_3x8, store_3x8, 3, 8, 1, 8, S, P, ukernel_3xn_bsp);
def_ukernel_avx512!(1, step_2x12, acc_2x12, store_2x12, 2, 8, 1, 8, S, P, ukernel_2xn_bsp);
def_ukernel_avx512!(1, step_1x12, acc_1x12, store_1x12, 1, 8, 1, 8, S, P, ukernel_1xn_bsp);



// based on l1 prefetching scheme is from openblas impl for skylax
// see: https://github.com/OpenMathLib/OpenBLAS/pull/2300
// this is adapted to our ukernel of 3x8
// seems to stem from high bandwith of l1 cache (compared to other uarch e.g. haswell
// where the same l1 prefetching does not benefit as much)

def_ukernel_avx512_2!(1, step_3x8, acc_3x8, store_3x8, 3, 8, 8, 32);
