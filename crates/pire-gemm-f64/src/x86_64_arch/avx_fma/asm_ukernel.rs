use seq_macro::seq;
use std::arch::asm;
use super::VS;
use crate::{TA, TB, TC, UnaryFnC, TC_SIZE};
use pire_base::{
    c_mem, def_ukernel_avx,
    c_reg_3x4, c_reg_2x6, c_reg_1x6, acc_3x4, acc_2x6, acc_1x6,
    store_3x4, store_2x6, store_1x6, b_num_2x6, b_num_1x6, init_ab_avx,
    load_a_avx, storep_avx, acc_p_avx, def_ukernel_avx_2, init_ab_2, prefetch_0,
};

type TS = TC;
const ZERO: f64 = 0.0;

const ZERO_SCALAR: f64 = 0.0;
const ONE_SCALAR: f64 = 1.0;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vaddpd ", $m0, ",%ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vmaskmovpd ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vaddpd %ymm2, %ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vfmadd231pd ", $m0, ",%ymm0,%ymm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vmaskmovpd ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vfmadd231pd %ymm2, %ymm0,%ymm", $r1, "\n",
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
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231pd %ymm", $r1, ", %ymm", $r2,", %ymm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovapd ", $m0, ",%ymm", $r1, "\n",
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
            "vmaskmovpd %ymm", $r1, ", %ymm1, ", $m0,  "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                vbroadcast!(), " ({alphax}),%ymm1", "\n",
                #(
                    "vmulpd %ymm1, %ymm", r, ",%ymm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %ymm0\n",
            "vxorps %ymm3,%ymm3,%ymm3\n",
            "vucomiss %xmm3,%xmm0\n",
        )
    }
}

macro_rules! inc_b {
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
        ""
    };
}

macro_rules! inc_a_k_unroll {
    ($X:tt, $K:tt) => {
        concat!(
            "add $32*", $K, "*", $X, ",{ax}", "\n",
        )
    };
}

macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => {
        ""
    };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $8*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}

macro_rules! vzero_kernel {
    () => { vzeroall!(4,15) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(4,15) };
}


macro_rules! load_b {
    (S, 0, $K:tt, $X:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ({bx}),%ymm", $r, "\n",
        )
    };
    (S, 1, $K:tt, $X:tt, $r:expr) => {
        concat!(
            "prefetcht0 64({bx},{x2},1) \n",
            vbroadcast!(), " ({bx},{x2},1),%ymm", $r, "\n",
        )
    };
    (S, 2, $K:tt, $X:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ({bx},{x2},2),%ymm", $r, "\n",
        )
    };
    (S, 3, $K:tt, $X:tt, $r:expr) => {
        concat!(
            "prefetcht0 64({x3}) \n",
            vbroadcast!(), " ({x3}),%ymm", $r, "\n",
        )
    };
    (B, $N:tt, $K:tt, $X:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*8+", $N, "*8({bx}), %ymm", $r, "\n",
        )
    };
}

macro_rules! fmadd_3v {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, 3, c_reg_3x4!(0,$ni)),
            vfmadd!(1, 3, c_reg_3x4!(1,$ni)),
            vfmadd!(2, 3, c_reg_3x4!(2,$ni)),
        )
    };
}

macro_rules! fmadd_2v {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, b_num_2x6!($ni), c_reg_2x6!(0,$ni)),
            vfmadd!(1, b_num_2x6!($ni), c_reg_2x6!(1,$ni)),
        )
    };
}

macro_rules! fmadd_1v {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, b_num_1x6!($ni), c_reg_1x6!(0,$ni)),
        )
    };
}

// ***************************** 3x4 ******************************* //
macro_rules! step_3x4 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx!(3, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, 3),
                    fmadd_3v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 2x6 ******************************* //
macro_rules! step_2x6 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx!(2, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_2x6!(n)),
                    fmadd_2v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 1x6 ******************************* //
macro_rules! step_1x6 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx!(1, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_1x6!(n)),
                    fmadd_1v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

#[inline(always)]
fn mask_and_offset(m: usize) -> ([u64;8], usize) {
    let mask: [u64; 8] = [
        u64::MAX, u64::MAX, u64::MAX, u64::MAX,
        0, 0, 0, 0,
    ];
    let mask_offset = if m % VS == 0 { 0 } else { VS - (m %VS)};

    (mask, mask_offset)
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

macro_rules! load_mask {
    (P) => {
        "vmovdqu ({maskx}), %ymm1"
    };
    (C) => {
        "/* {maskx} */"
    }
}

def_ukernel_avx!(1, step_3x4, acc_3x4, store_3x4, 3, 4, B, P, ukernel_3_bbp);
def_ukernel_avx!(1, step_2x6, acc_2x6, store_2x6, 2, 4, B, P, ukernel_2_bbp);
def_ukernel_avx!(1, step_1x6, acc_1x6, store_1x6, 1, 4, B, P, ukernel_1_bbp);

def_ukernel_avx!(1, step_3x4, acc_3x4, store_3x4, 3, 4, S, C, ukernel_bsc);

def_ukernel_avx!(1, step_3x4, acc_3x4, store_3x4, 3, 4, S, P, ukernel_3_bsp);
def_ukernel_avx!(1, step_2x6, acc_2x6, store_2x6, 2, 4, S, P, ukernel_2_bsp);
def_ukernel_avx!(1, step_1x6, acc_1x6, store_1x6, 1, 4, S, P, ukernel_1_bsp);

// based on l1 prefetching scheme is from openblas impl for skylax
// see: https://github.com/OpenMathLib/OpenBLAS/pull/2300
// this is adapted to our ukernel of 3x4
// seems to stem from high bandwith of l1 cache (compared to other uarch e.g. haswell
// where the same l1 prefetching does not benefit as much)
def_ukernel_avx_2!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 4, 64);
