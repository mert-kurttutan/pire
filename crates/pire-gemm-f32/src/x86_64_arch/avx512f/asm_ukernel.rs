use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_avx512, def_ukernel_avx512_2,
    acc_3, acc_2, acc_1, store_3, store_2, store_1,
    step_3, step_2, step_1,
    init_ab_2, init_ab,
    def_ukernel_avx512_dot,
    b_mem, mem,
};

type TS = TC;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

macro_rules! vs {
    () => { "0x40" };
}
macro_rules! bs {
    () => { "4" };
}
macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x40+" , $m) };
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
    ($m0:expr, $r1:expr, C) => {
        concat!(
            "vmovups ", mem!($m0, concat!("0x40*", $r1)), ", %zmm", $r1, "\n",
        )
    };
    ($m0:expr, $r1:expr, B) => {
        concat!(
            "vmovaps ", mem!($m0, concat!("0x40*", $r1)), ", %zmm", $r1, "\n",
        )
    };
    ($m0:expr, $r1:expr, P) => {
        concat!(
            "vmovups ", mem!($m0, concat!("0x40*", $r1)), ", %zmm", $r1, "{{%k1}}\n",
        )
    };
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovaps ", mem!($m0, concat!("0x40*", $r1)), ", %zmm", $r1, "\n",
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
    (S,$nr:tt) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n"
    };
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

macro_rules! prefetch {
    (B, 0) => {
        "prefetcht0 768({bx})\n"
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
    (7) => { 1 };
}

def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 8, B, B, P, ukernel_3_bbp);
def_ukernel_avx512!(1, step_2, acc_2, store_2, 2, 8, B, B, P, ukernel_2_bbp);
def_ukernel_avx512!(1, step_1, acc_1, store_1, 1, 8, B, B, P, ukernel_1_bbp);

def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 8, B, S, C, ukernel_bsc);

def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 8, B, S, P, ukernel_3_bsp);
def_ukernel_avx512!(1, step_2, acc_2, store_2, 2, 8, B, S, P, ukernel_2_bsp);
def_ukernel_avx512!(1, step_1, acc_1, store_1, 1, 8, B, S, P, ukernel_1_bsp);


// based on l1 prefetching scheme is from openblas impl for skylax
// see: https://github.com/OpenMathLib/OpenBLAS/pull/2300
// this is adapted to our ukernel of 3
// seems to stem from high bandwith of l1 cache (compared to other uarch e.g. haswell
// where the same l1 prefetching does not benefit as much)

// def_ukernel_avx512_2!(1, step_3, acc_3, store_3, 3, 8, 8, 32);


def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 8, C, S, C, ukernel_ssc);

def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 8, P, S, P, ukernel_3_ssp);
def_ukernel_avx512!(1, step_2, acc_2, store_2, 2, 8, P, S, P, ukernel_2_ssp);
def_ukernel_avx512!(1, step_1, acc_1, store_1, 1, 8, P, S, P, ukernel_1_ssp);
def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 8, C, B, C, ukernel_bbc);

macro_rules! reduce_add {
    ($r0:tt, $r1:tt) => {
        concat!(
            "vunpcklps %zmm", $r1, ", %zmm", $r0, ", %zmm0", "\n",
            "vunpckhps %zmm", $r1, ", %zmm", $r0, ", %zmm1", "\n",
            "vaddps %zmm0, %zmm1, %zmm", $r0, "\n",

            "vpermilps $0b01001110, %zmm", $r0, ", %zmm", $r1, "\n",
            "vaddps %zmm", $r0, ", %zmm", $r1, ", %zmm", $r0, "\n",

            "vshuff32x4 $0b10110001, %zmm", $r0, ", %zmm", $r0, ", %zmm", $r1, "\n",
            "vaddps %zmm", $r0, ", %zmm", $r1, ", %zmm", $r0, "\n",

            "vshuff32x4 $0b10110011, %zmm", $r0, ", %zmm", $r0, ", %zmm", $r1, "\n",
            "vaddps %zmm", $r0, ", %zmm", $r1, ", %zmm", $r0, "\n",

            // move second f32 to zmm $r1
            "vpermilps $0b01000001, %zmm", $r0, ", %zmm", $r1, "\n",
        )
    }
}

macro_rules! init_ab_dot {
    () => {
        concat!(
            "/* {x4} */",
            "mov 32({dim_arrx}), {x0}", "\n",
            "mov ({dim_arrx}), {x1}", "\n",
            "lea ({x1}, {x1}, 2), {x3}", "\n",
            "lea ({ax}, {x3}, 1), {x2}", "\n",
            "lea ({x2}, {x3}, 1), {x3}", "\n",
            "mov 8({dim_arrx}), {x5}", "\n",
            "lea ({bx}, {x5}, 2), {x4}", "\n",
            "lea ({x4}, {x5}, 1), {x4}", "\n",

        )
    }
}

macro_rules! a_mem_dot {
    (0) => { "({ax})" };
    (1) => { "({ax},{x1})" };
    (2) => { "({ax},{x1},2)" };
    (3) => { "({x2})" };
    (4) => { "({x2},{x1})" };
    (5) => { "({x2},{x1},2)" };
    (6) => { "({x3})" };
}

macro_rules! b_mem_dot {
    (0) => { "({bx})" };
    (1) => { "({bx},{x5})" };
    (2) => { "({bx},{x5},2)" };
    (3) => { "({x4})" };
    (4) => { "({x4},{x5})" };
    (5) => { "({x4},{x5},2)" };
    (6) => { "({x4})" };
}

macro_rules! b_macro_dot {
    (0) => { 7 };
    (1) => { 30 };
    (2) => { 31 };
    (3) => { 7 };
    (4) => { 30 };
    (5) => { 31 };
    (6) => { 7 };
}

macro_rules! b_macro_dot_2 {
    (0) => { 7 };
    (1) => { 30 };
    (2) => { 31 };
    (3) => { 7 };
    (4) => { 30 };
    (5) => { 31 };
    (6) => { 31 };
    (7) => { 31 };
}
macro_rules! c_macro_dot {
    (0,0) => { 8 };
    (0,1) => { 9 };
    (0,2) => { 10 };
    (1,0) => { 11 };
    (1,1) => { 12 };
    (1,2) => { 13 };
    (2,0) => { 14 };
    (2,1) => { 15 };
    (2,2) => { 16 };
    (3,0) => { 17 };
    (3,1) => { 18 };
    (3,2) => { 19 };
    (4,0) => { 20 };
    (4,1) => { 21 };
    (4,2) => { 22 };
    (5,0) => { 23 };
    (5,1) => { 24 };
    (5,2) => { 25 };
    (6,0) => { 26 };
    (6,1) => { 27 };
    (6,2) => { 28 };
}

macro_rules! c_macro_dot_2 {
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
}


macro_rules! load_a_dot {
    ($nr:tt) => {
        concat!(
            seq!(n in 0..$nr {
                concat!(
                    #(
                        "vmovups ", a_mem_dot!(n), ", %zmm", n, "\n",
                    )*
                )
            })
        )
    };
}

macro_rules! load_a_dot_partial {
    ($nr:tt) => {
        concat!(
            seq!(n in 0..$nr {
                concat!(
                    #(
                        "vpxorq %zmm", n, ", %zmm", n, ", %zmm", n, "\n",
                        "vmovups ", a_mem_dot!(n), ", %zmm", n, " {{%k1}}\n",
                    )*
                )
            })
        )
    };
}

macro_rules! step_dot {
    (7, $nr:tt, $b_macro:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    "vmovups ", b_mem_dot!(n), ", %zmm", $b_macro!(n), "\n",
                    "vfmadd231ps %zmm0, %zmm", $b_macro!(n), ", %zmm", $c_macro!(0,n), "\n",
                    "vfmadd231ps %zmm1, %zmm", $b_macro!(n), ", %zmm", $c_macro!(1,n), "\n",
                    "vfmadd231ps %zmm2, %zmm", $b_macro!(n), ", %zmm", $c_macro!(2,n), "\n",
                    "vfmadd231ps %zmm3, %zmm", $b_macro!(n), ", %zmm", $c_macro!(3,n), "\n",
                    "vfmadd231ps %zmm4, %zmm", $b_macro!(n), ", %zmm", $c_macro!(4,n), "\n",
                    "vfmadd231ps %zmm5, %zmm", $b_macro!(n), ", %zmm", $c_macro!(5,n), "\n",
                    "vfmadd231ps %zmm6, %zmm", $b_macro!(n), ", %zmm", $c_macro!(6,n), "\n",
                )*
            )
        })
    };
    (6, $nr:tt, $b_macro:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    "vmovups ", b_mem_dot!(n), ", %zmm", $b_macro!(n), "\n",
                    "vfmadd231ps %zmm0, %zmm", $b_macro!(n), ", %zmm", $c_macro!(0,n), "\n",
                    "vfmadd231ps %zmm1, %zmm", $b_macro!(n), ", %zmm", $c_macro!(1,n), "\n",
                    "vfmadd231ps %zmm2, %zmm", $b_macro!(n), ", %zmm", $c_macro!(2,n), "\n",
                    "vfmadd231ps %zmm3, %zmm", $b_macro!(n), ", %zmm", $c_macro!(3,n), "\n",
                    "vfmadd231ps %zmm4, %zmm", $b_macro!(n), ", %zmm", $c_macro!(4,n), "\n",
                    "vfmadd231ps %zmm5, %zmm", $b_macro!(n), ", %zmm", $c_macro!(5,n), "\n",
                )*
            )
        })
    };
    (5, $nr:tt, $b_macro:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    "vmovups ", b_mem_dot!(n), ", %zmm", $b_macro!(n), "\n",
                    "vfmadd231ps %zmm0, %zmm", $b_macro!(n), ", %zmm", $c_macro!(0,n), "\n",
                    "vfmadd231ps %zmm1, %zmm", $b_macro!(n), ", %zmm", $c_macro!(1,n), "\n",
                    "vfmadd231ps %zmm2, %zmm", $b_macro!(n), ", %zmm", $c_macro!(2,n), "\n",
                    "vfmadd231ps %zmm3, %zmm", $b_macro!(n), ", %zmm", $c_macro!(3,n), "\n",
                    "vfmadd231ps %zmm4, %zmm", $b_macro!(n), ", %zmm", $c_macro!(4,n), "\n",
                )*
            )
        })
    };
    (4, $nr:tt, $b_macro:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    "vmovups ", b_mem_dot!(n), ", %zmm", $b_macro!(n), "\n",
                    "vfmadd231ps %zmm0, %zmm", $b_macro!(n), ", %zmm", $c_macro!(0,n), "\n",
                    "vfmadd231ps %zmm1, %zmm", $b_macro!(n), ", %zmm", $c_macro!(1,n), "\n",
                    "vfmadd231ps %zmm2, %zmm", $b_macro!(n), ", %zmm", $c_macro!(2,n), "\n",
                    "vfmadd231ps %zmm3, %zmm", $b_macro!(n), ", %zmm", $c_macro!(3,n), "\n",
                )*
            )
        })
    };
    (3, $nr:tt, $b_macro:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    "vmovups ", b_mem_dot!(n), ", %zmm", $b_macro!(n), "\n",
                    "vfmadd231ps %zmm0, %zmm", $b_macro!(n), ", %zmm", $c_macro!(0,n), "\n",
                    "vfmadd231ps %zmm1, %zmm", $b_macro!(n), ", %zmm", $c_macro!(1,n), "\n",
                    "vfmadd231ps %zmm2, %zmm", $b_macro!(n), ", %zmm", $c_macro!(2,n), "\n",
                )*
            )
        })
    };
    (2, $nr:tt, $b_macro:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    "vmovups ", b_mem_dot!(n), ", %zmm", $b_macro!(n), "\n",
                    "vfmadd231ps %zmm0, %zmm", $b_macro!(n), ", %zmm", $c_macro!(0,n), "\n",
                    "vfmadd231ps %zmm1, %zmm", $b_macro!(n), ", %zmm", $c_macro!(1,n), "\n",
                )*
            )
        })
    };
    (1, $nr:tt, $b_macro:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    "vmovups ", b_mem_dot!(n), ", %zmm", $b_macro!(n), "\n",
                    "vfmadd231ps %zmm0, %zmm", $b_macro!(n), ", %zmm", $c_macro!(0,n), "\n",
                )*
            )
        })
    };
}

macro_rules! c_load_dot {
    () => {
        concat!(
            reduce_add!(8,9),
            reduce_add!(10,11),
            reduce_add!(12,13),
            reduce_add!(14,15),
            reduce_add!(16,17),
            reduce_add!(18,19),
            reduce_add!(20,21),
            reduce_add!(22,23),
            reduce_add!(24,25),
            reduce_add!(26,27),
            reduce_add!(28,29),
            reduce_add!(30,31),
            "mov 16({dim_arrx}),{x0}", "\n",	
            "mov 24({dim_arrx}),{x1}", "\n",
        )
    }
}

macro_rules! c_mem_dot {
    (0) => { "({x2})" };
    (1) => { "({x2},{x0})" };
    (2) => { "({x2},{x0},2)" };
    (3) => { "({x3})" };
    (4) => { "({x3},{x0})" };
    (5) => { "({x3},{x0},2)" };
    (6) => { "({x4})" };
}

macro_rules! acc_dot {
    (7, $nr:tt, 2, $c_macro:tt) => {
        seq!(n in 0..$nr {

            concat!(
                "mov {cx}, {x2}", "\n",
                "lea ({x0}, {x0}, 2), {x4}", "\n",	
                "lea ({x2}, {x4}, 1), {x3}", "\n",	
                "lea ({x3}, {x4}, 1), {x4}", "\n",
                #(
                    "vaddss ", c_mem_dot!(0), ", %xmm", $c_macro!(0,n), ", %xmm", $c_macro!(0,n), "\n",
                    "vaddss ", c_mem_dot!(1), ", %xmm", $c_macro!(1,n), ", %xmm", $c_macro!(1,n), "\n",
                    "vaddss ", c_mem_dot!(2), ", %xmm", $c_macro!(2,n), ", %xmm", $c_macro!(2,n), "\n",
                    "vaddss ", c_mem_dot!(3), ", %xmm", $c_macro!(3,n), ", %xmm", $c_macro!(3,n), "\n",
                    "vaddss ", c_mem_dot!(4), ", %xmm", $c_macro!(4,n), ", %xmm", $c_macro!(4,n), "\n",
                    "vaddss ", c_mem_dot!(5), ", %xmm", $c_macro!(5,n), ", %xmm", $c_macro!(5,n), "\n",
                    "vaddss ", c_mem_dot!(6), ", %xmm", $c_macro!(6,n), ", %xmm", $c_macro!(6,n), "\n",
                    "add {x1},{x2} \n add {x1},{x3} \n add {x1},{x4} \n",
                )*
            )
        })
    };
    (6, $nr:tt, 2, $c_macro:tt) => {
        seq!(n in 0..$nr {

            concat!(
                "mov {cx}, {x2}", "\n",
                "lea ({x0}, {x0}, 2), {x4}", "\n",	
                "lea ({x2}, {x4}, 1), {x3}", "\n",	
                "lea ({x3}, {x4}, 1), {x4}", "\n",
                #(
                    "vaddss ", c_mem_dot!(0), ", %xmm", $c_macro!(0,n), ", %xmm", $c_macro!(0,n), "\n",
                    "vaddss ", c_mem_dot!(1), ", %xmm", $c_macro!(1,n), ", %xmm", $c_macro!(1,n), "\n",
                    "vaddss ", c_mem_dot!(2), ", %xmm", $c_macro!(2,n), ", %xmm", $c_macro!(2,n), "\n",
                    "vaddss ", c_mem_dot!(3), ", %xmm", $c_macro!(3,n), ", %xmm", $c_macro!(3,n), "\n",
                    "vaddss ", c_mem_dot!(4), ", %xmm", $c_macro!(4,n), ", %xmm", $c_macro!(4,n), "\n",
                    "vaddss ", c_mem_dot!(5), ", %xmm", $c_macro!(5,n), ", %xmm", $c_macro!(5,n), "\n",
                    "add {x1},{x2} \n add {x1},{x3} \n add {x1},{x4} \n",
                )*
            )
        })
    };
    (3, $nr:tt, 2, $c_macro:tt) => {
        seq!(n in 0..$nr {

            concat!(
                "mov {cx}, {x2}", "\n",
                "lea ({x0}, {x0}, 2), {x4}", "\n",	
                "lea ({x2}, {x4}, 1), {x3}", "\n",	
                "lea ({x3}, {x4}, 1), {x4}", "\n",
                #(
                    "vaddss ", c_mem_dot!(0), ", %xmm", $c_macro!(0,n), ", %xmm", $c_macro!(0,n), "\n",
                    "vaddss ", c_mem_dot!(1), ", %xmm", $c_macro!(1,n), ", %xmm", $c_macro!(1,n), "\n",
                    "vaddss ", c_mem_dot!(2), ", %xmm", $c_macro!(2,n), ", %xmm", $c_macro!(2,n), "\n",
                    "add {x1},{x2} \n add {x1},{x3} \n add {x1},{x4} \n",
                )*
            )
        })
    };
    (2, $nr:tt, 2, $c_macro:tt) => {
        seq!(n in 0..$nr {

            concat!(
                "mov {cx}, {x2}", "\n",
                "lea ({x0}, {x0}, 2), {x4}", "\n",	
                "lea ({x2}, {x4}, 1), {x3}", "\n",	
                "lea ({x3}, {x4}, 1), {x4}", "\n",
                #(
                    "vaddss ", c_mem_dot!(0), ", %xmm", $c_macro!(0,n), ", %xmm", $c_macro!(0,n), "\n",
                    "vaddss ", c_mem_dot!(1), ", %xmm", $c_macro!(1,n), ", %xmm", $c_macro!(1,n), "\n",
                    "add {x1},{x2} \n add {x1},{x3} \n add {x1},{x4} \n",
                )*
            )
        })
    };
    (1, $nr:tt, 2, $c_macro:tt) => {
        seq!(n in 0..$nr {

            concat!(
                "mov {cx}, {x2}", "\n",
                "lea ({x0}, {x0}, 2), {x4}", "\n",	
                "lea ({x2}, {x4}, 1), {x3}", "\n",	
                "lea ({x3}, {x4}, 1), {x4}", "\n",
                #(
                    "vaddss ", c_mem_dot!(0), ", %xmm", $c_macro!(0,n), ", %xmm", $c_macro!(0,n), "\n",
                    "add {x1},{x2} \n add {x1},{x3} \n add {x1},{x4} \n",
                )*
            )
        })
    };
}

macro_rules! store_dot {
    (7, $nr:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {

            concat!(
                "mov {cx}, {x2}", "\n",
                "lea ({x0}, {x0}, 2), {x4}", "\n",	
                "lea ({x2}, {x4}, 1), {x3}", "\n",	
                "lea ({x3}, {x4}, 1), {x4}", "\n",
                #(
                    "vmovss %xmm", $c_macro!(0,n), ", ", c_mem_dot!(0), "\n",
                    "vmovss %xmm", $c_macro!(1,n), ", ", c_mem_dot!(1), "\n",
                    "vmovss %xmm", $c_macro!(2,n), ", ", c_mem_dot!(2), "\n",
                    "vmovss %xmm", $c_macro!(3,n), ", ", c_mem_dot!(3), "\n",
                    "vmovss %xmm", $c_macro!(4,n), ", ", c_mem_dot!(4), "\n",
                    "vmovss %xmm", $c_macro!(5,n), ", ", c_mem_dot!(5), "\n",
                    "vmovss %xmm", $c_macro!(6,n), ", ", c_mem_dot!(6), "\n",
                    "add {x1},{x2} \n add {x1},{x3} \n add {x1},{x4} \n",
                )*
            )
        })
    };
    (6, $nr:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {

            concat!(
                "mov {cx}, {x2}", "\n",
                "lea ({x0}, {x0}, 2), {x4}", "\n",	
                "lea ({x2}, {x4}, 1), {x3}", "\n",	
                "lea ({x3}, {x4}, 1), {x4}", "\n",
                #(
                    "vmovss %xmm", $c_macro!(0,n), ", ", c_mem_dot!(0), "\n",
                    "vmovss %xmm", $c_macro!(1,n), ", ", c_mem_dot!(1), "\n",
                    "vmovss %xmm", $c_macro!(2,n), ", ", c_mem_dot!(2), "\n",
                    "vmovss %xmm", $c_macro!(3,n), ", ", c_mem_dot!(3), "\n",
                    "vmovss %xmm", $c_macro!(4,n), ", ", c_mem_dot!(4), "\n",
                    "vmovss %xmm", $c_macro!(5,n), ", ", c_mem_dot!(5), "\n",
                    "add {x1},{x2} \n add {x1},{x3} \n add {x1},{x4} \n",
                )*
            )
        })
    };
    (3, $nr:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {

            concat!(
                "mov {cx}, {x2}", "\n",
                "lea ({x0}, {x0}, 2), {x4}", "\n",	
                "lea ({x2}, {x4}, 1), {x3}", "\n",	
                "lea ({x3}, {x4}, 1), {x4}", "\n",
                #(
                    "vmovss %xmm", $c_macro!(0,n), ", ", c_mem_dot!(0), "\n",
                    "vmovss %xmm", $c_macro!(1,n), ", ", c_mem_dot!(1), "\n",
                    "vmovss %xmm", $c_macro!(2,n), ", ", c_mem_dot!(2), "\n",
                    "add {x1},{x2} \n add {x1},{x3} \n add {x1},{x4} \n",
                )*
            )
        })
    };
    (2, $nr:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {

            concat!(
                "mov {cx}, {x2}", "\n",
                "lea ({x0}, {x0}, 2), {x4}", "\n",	
                "lea ({x2}, {x4}, 1), {x3}", "\n",	
                "lea ({x3}, {x4}, 1), {x4}", "\n",
                #(
                    "vmovss %xmm", $c_macro!(0,n), ", ", c_mem_dot!(0), "\n",
                    "vmovss %xmm", $c_macro!(1,n), ", ", c_mem_dot!(1), "\n",
                    "add {x1},{x2} \n add {x1},{x3} \n add {x1},{x4} \n",
                )*
            )
        })
    };
    (1, $nr:tt, $c_macro:tt) => {
        seq!(n in 0..$nr {

            concat!(
                "mov {cx}, {x2}", "\n",
                "lea ({x0}, {x0}, 2), {x4}", "\n",	
                "lea ({x2}, {x4}, 1), {x3}", "\n",	
                "lea ({x3}, {x4}, 1), {x4}", "\n",
                #(
                    "vmovss %xmm", $c_macro!(0,n), ", ", c_mem_dot!(0), "\n",
                    "add {x1},{x2} \n add {x1},{x3} \n add {x1},{x4} \n",
                )*
            )
        })
    };
}


def_ukernel_avx512_dot!(16, step_dot, acc_dot, store_dot, b_macro_dot, c_macro_dot, 7, 3, ukernel_rcc7);
def_ukernel_avx512_dot!(16, step_dot, acc_dot, store_dot, b_macro_dot, c_macro_dot, 6, 3, ukernel_rcc6);
def_ukernel_avx512_dot!(16, step_dot, acc_dot, store_dot, b_macro_dot_2, c_macro_dot_2, 3, 6, ukernel_rcc3);
def_ukernel_avx512_dot!(16, step_dot, acc_dot, store_dot, b_macro_dot_2, c_macro_dot_2, 2, 6, ukernel_rcc2);
def_ukernel_avx512_dot!(16, step_dot, acc_dot, store_dot, b_macro_dot_2, c_macro_dot_2, 1, 6, ukernel_rcc1);
