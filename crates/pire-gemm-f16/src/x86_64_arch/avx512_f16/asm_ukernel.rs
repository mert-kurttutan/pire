use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_avx512,def_ukernel_avx512_2,
    init_ab_2,
    acc_2, store_2, acc_1, store_1,
    step_2, step_1,
};
use half::f16;

type TS = TC;

const ZERO_SCALAR: f16 = f16::ZERO;
const ONE_SCALAR: f16 = f16::ONE;

macro_rules! vs {
    () => { "0x40" };
}

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x20+" , $m) };
}

#[macro_export]
macro_rules! init_ab {
    (B) => {
        concat!(
            "/* {x5} */\n",
            "/* {x4} */\n",
            "/* {x3} */\n",
            "/* {x2} */\n",
            "/* {x1} */\n",
            "mov 24({dim_arrx}),{x0}\n",
        )
    };
    (S) => {
        concat!(
            // mov cs_b to reg
            "mov ({dim_arrx}), {x1}\n",
            "mov 8({dim_arrx}), {x2}\n",
            "lea ({x2}, {x2}, 4), {x5}\n",
            "lea ({bx}, {x2}, 1), {x3}\n",
            "lea ({bx}, {x5}, 1), {x4}\n",
            "lea ({x4}, {x5}, 1), {x5}\n",
            "mov 24({dim_arrx}),{x0}\n",
        )
    };
}

macro_rules! cr {
    (0,0) => { 2 }; (1,0) => { 3 };
    (0,1) => { 4 }; (1,1) => { 5 };
    (0,2) => { 6 }; (1,2) => { 7 };
    (0,3) => { 8 }; (1,3) => { 9 };
    (0,4) => { 10 }; (1,4) => { 11 };
    (0,5) => { 12 }; (1,5) => { 13 };
    (0,6) => { 14 }; (1,6) => { 15 };
    (0,7) => { 16 }; (1,7) => { 17 };
    (0,8) => { 18 }; (1,8) => { 19 };
    (0,9) => { 20 }; (1,9) => { 21 };
    (0,10) => { 22 }; (1,10) => { 23 };
    (0,11) => { 24 }; (1,11) => { 25 };
    (0,12) => { 26 }; (1,12) => { 27 };
    (0,13) => { 28 }; (1,13) => { 29 };
    (0,14) => { 30 }; (1,14) => { 31 };
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vfmadd231ph ", $m0, ",%zmm0,%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vmovdqu16 ", $m0, ", %zmm1 {{%k1}}", "\n",
            "vfmadd231ph %zmm1,%zmm0,%zmm", $r1, "\n",
        )
    };
    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vaddph ", $m0, ",%zmm", $r1, ",%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vaddph ", $m0, ",%zmm", $r1, ",%zmm", $r1, "{{%k1}}\n",
        )
    };
}

macro_rules! c_load {
    () => {
        concat!(
            "mov 16({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x4}\n",
            "lea ({cx}, {x4},), {x1}\n",
            "lea ({x1}, {x4},), {x2}\n",
            "lea ({x2}, {x4},), {x3}\n",
            "lea ({x3}, {x4},), {x4}\n",
        )
    };
}

macro_rules! c_load_2 {
    () => {
        concat!(
            "mov ({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x4}\n",
            "lea ({cx}, {x4},), {x1}\n",
            "lea ({x1}, {x4},), {x2}\n",
            "lea ({x2}, {x4},), {x3}\n",
            "lea ({x3}, {x4},), {x4}\n",
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
    () => { "vpbroadcastw" };
}

macro_rules! vfmadd {
    ($i:tt, $j:tt, $b_layout:tt, $b_macro:tt) => {
        concat!(
            "vfmadd231ph ", $b_macro!($b_layout,$j), "{{1to32}}", ", %zmm", $i,", %zmm", cr!($i,$j), "\n",
        ) 
    };
}

macro_rules! bd {
    (B, $i:tt) => {
        concat!($i, "*2({bx})")
    };
    (S, 0) => {
        "0({bx})"
    };
    (S, 1) => {
        "0({bx}, {x2})"
    };
    (S, 2) => {
        "0({bx}, {x2}, 2)"
    };
    (S, 3) => {
        "0({x3}, {x2}, 2)"
    };
    (S, 4) => {
        "0({bx}, {x2}, 4)"
    };
    (S, 5) => {
        "0({x4})"
    };
    (S, 6) => {
        "0({x4}, {x2})"
    };
    (S, 7) => {
        "0({x4}, {x2}, 2)"
    };
    (S, 8) => {
        "0({bx}, {x2}, 8)"
    };
    (S, 9) => {
        "0({x3}, {x2}, 8)"
    };
    (S, 10) => {
        "0({x5})"
    };
    (S, 11) => {
        "0({x5}, {x2})"
    };
    (S, 12) => {
        "0({x5}, {x2}, 2)"
    };
    (S, 13) => {
        "0({x4}, {x2}, 8)"
    };
    (S, 14) => {
        "0({x5}, {x2}, 4)"
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr, B) => {
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
            "vmovdqu16 %zmm", $r1, ", ", $m0, " {{%k1}}\n",
        )
    };
}


macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                vbroadcast!(), " ({alphax}),%zmm1", "\n",
                #(
                    "vmulph %zmm1, %zmm", r, ",%zmm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! load_mask {
    (P) => { "kmovd ({maskx}), %k1" };
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
    () => { vzeroall!(2,31) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(2,31) };
}

macro_rules! inc_b {
    (S,$nr:tt) => { "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n" };
    (B,$nr:tt) => {
        concat!(
            "add $2*", $nr, ", {bx}", "\n",
        )
    };
    ($nr:tt) => {
        concat!(
            "add $2*", $nr, ", {bx}", "\n",
        )
    };
}

macro_rules! fmadd_2 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, $ni, B, bd),
            vfmadd!(1, $ni, B, bd),
        )
    };
    ($b_layout:tt, $ni:tt) => {
        concat!(
            vfmadd!(0, $ni, $b_layout, bd),
            vfmadd!(1, $ni, $b_layout, bd),
        )
    };
}

macro_rules! fmadd_1 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, $ni, B, bd),
        )
    };
    ($b_layout:tt, $ni:tt) => {
        concat!(
            vfmadd!(0, $ni, $b_layout, bd),
        )
    };
}

// ***************************** 2 ******************************* //
macro_rules! step_2 {
    (B, 15) => {
        concat!(
            fmadd_2!(0),
            fmadd_2!(1),
            "prefetcht0 256({ax}) \n",
            fmadd_2!(2),
            "prefetcht0 64({bx}) \n",
            fmadd_2!(3),
            fmadd_2!(4),
            fmadd_2!(5),
            fmadd_2!(6),
            fmadd_2!(7),
            fmadd_2!(8),
            fmadd_2!(9),
            fmadd_2!(10),
            "prefetcht0 320({ax}) \n",
            fmadd_2!(11),
            fmadd_2!(12),
            fmadd_2!(13),
            fmadd_2!(14),
        )
        
    };
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                "prefetcht0 64({bx}) \n",
                #(
                    fmadd_2!($b_layout, n),
                )*
            )
        })
    };
}
// ***************************** 1 ******************************* //
macro_rules! step_1 {
    (15, B) => {
        concat!(
            fmadd_1!(0),
            fmadd_1!(1),
            "prefetcht0 256({ax}) \n",
            fmadd_1!(2),
            "prefetcht0 64({bx}) \n",
            fmadd_1!(3),
            fmadd_1!(4),
            fmadd_1!(5),
            fmadd_1!(6),
            fmadd_1!(7),
            fmadd_1!(8),
            fmadd_1!(9),
            fmadd_1!(10),
            "prefetcht0 320({ax}) \n",
            fmadd_1!(11),
            fmadd_1!(12),
            fmadd_1!(13),
            fmadd_1!(14),
        )
        
    };
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                "prefetcht0 64({bx}) \n",
                #(
                    fmadd_1!($b_layout, n),
                )*
            )
        })
    };
}

macro_rules! mask_ptr {
    (P, $m:expr, $nm:ident, $mask_ptr:ident) => {
        let $nm = if $m % VS == 0 && $m > 0 { 0xFFFFFFFF } else { (1_u32 << ($m % VS)) - 1 };
        let $mask_ptr = (&$nm) as *const u32;
    };
    (C, $m:expr, $nm:ident, $mask_ptr:ident) => {
        let $nm = 0xFFFFFFFF_u32;
        let $mask_ptr = (&$nm) as *const u32;
    };
}

def_ukernel_avx512!(1, step_2, acc_2, store_2, 2, 15, B, B, P, ukernel_2_bbp);
def_ukernel_avx512!(1, step_1, acc_1, store_1, 1, 15, B, B, P, ukernel_1_bbp);

def_ukernel_avx512!(1, step_2, acc_2, store_2, 2, 15, B, S, C, ukernel_bsc);

def_ukernel_avx512!(1, step_2, acc_2, store_2, 2, 15, B, S, P, ukernel_2_bsp);
def_ukernel_avx512!(1, step_1, acc_1, store_1, 1, 15, B, S, P, ukernel_1_bsp);


// based on l1 prefetching scheme is from openblas impl for skylax
// see: httph://github.com/OpenMathLib/OpenBLAS/pull/2300
// this is adapted to our ukernel of 3x8
// seems to stem from high bandwith of l1 cache (compared to other uarch e.g. haswell
// where the same l1 prefetching does not benefit as much)

def_ukernel_avx512_2!(1, step_2, acc_2, store_2, 2, 15, 16, 32);
