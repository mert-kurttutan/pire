use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_avx,
    acc_2, acc_1, store_2, store_1,
    mem,
    step_2, step_1,
    // def_ukernel_avx_2,
    // prefetch_0,
};

type TS = f32;


const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

macro_rules! br_2 {
    (0) => { 2 };
    (1) => { 3 };
    (2) => { 2 };
    (3) => { 3 };
}

macro_rules! br_1 {
    (0) => { 1 };
    (1) => { 2 };
    (2) => { 3 };
    (3) => { 5 };
}

macro_rules! cr {
    (0,0) => { 4 };
    (1,0) => { 5 };
    (0,1) => { 6 };
    (1,1) => { 7 };
    (0,2) => { 8 };
    (1,2) => { 9 };
    (0,3) => { 10 };
    (1,3) => { 11 };
}

macro_rules! dr {
    (0,0) => { 12 };
    (1,0) => { 13 };
    (0,1) => { 14 };
    (1,1) => { 12 };
    (0,2) => { 13 };
    (1,2) => { 14 };
    (0,3) => { 12 };
    (1,3) => { 13 };
}

macro_rules! vs {
    () => { "0x20" };
}

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x20+" , $m) };
}
macro_rules! beta_fmadd {
    (C, $m0:expr, $r:expr, 1) => {
        concat!(
            "vpaddd ", $m0, ", %ymm", $r, ", %ymm", $r, "\n",
        ) 
    };
    (C, $m0:expr, $r:expr, 2) => {
        concat!(
            "vcvtdq2ps %ymm", $r, ",%ymm", $r, "\n",
            "vcvtdq2ps ", $m0, ",%ymm2", "\n",
            "vfmadd231ps %ymm2,%ymm0,%ymm", $r, "\n",
            "vcvtps2dq %ymm", $r, ",%ymm", $r, "\n",
        ) 
    };
    (P, $m0:expr, $r:expr, 1) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vpaddd %ymm2, %ymm", $r, ", %ymm", $r, "\n",
        ) 
    };

    (P, $m0:expr, $r:expr, 2) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vcvtdq2ps %ymm2", ",%ymm2", "\n",
            "vcvtdq2ps %ymm", $r, ",%ymm", $r, "\n",
            "vfmadd231ps %ymm2,%ymm0,%ymm", $r, "\n",
            "vcvtps2dq %ymm", $r, ",%ymm", $r, "\n",
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

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("vpxor %ymm",r,",%ymm",r,",%ymm",r,"\n",)*)
        })
    }
}

macro_rules! vbroadcast {
    () => { "vbroadcastss" };
}
macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vpmaddubsw %ymm", $r1, ", %ymm", $r2, ", %ymm", $r4, "\n",
            "vpmaddwd %ymm", $r4, ", %ymm15", ", %ymm", $r4, "\n",
            "vpaddd %ymm", $r4, ", %ymm", $r3, ", %ymm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovaps ", mem!($m0, concat!("0x20*", $r1)), ", %ymm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (P, $r1:expr, $m0:expr) => {
        concat!(
            "vmaskmovps %ymm", $r1, ", %ymm1, ", $m0,  "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "vbroadcastss ({alphax}),%ymm1", "\n",
                #(
                    "vcvtdq2ps %ymm", r, ",%ymm", r, "\n",
                    "vmulps %ymm1, %ymm", r, ",%ymm", r, "\n",
                    "vcvtps2dq %ymm", r, ",%ymm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %ymm0\n",
        )
    }
}

macro_rules! init_ab_avx {
    (B) => {
        concat!(
            // move 2 1_i16 to xmm15
            "mov $0x10001, {x3:e}", "\n",
            "movd {x3:e}, %xmm15", "\n",
            "vbroadcastss %xmm15, %ymm15", "\n",
            "/* {x3} */", "\n",
            "/* {x2} */", "\n",
            "/* {x1} */", "\n",
            "mov 24({dim_arrx}),{x0}", "\n",
        )
    };
    (S) => { "" };
}

macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => { "" };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $4*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}

macro_rules! inc_b {
    (S, $nr:tt) => { "" };
    (B, $nr:tt) => { "" };
}

macro_rules! load_b {
    (B, $nr:tt, $ni:tt, $K:tt, $r:expr) => {
        concat!(
            "vbroadcastss ", $K, "*", $nr, "*4+", $ni, "*4({bx}), %ymm", $r, "\n",
        )
    };
}

macro_rules! fmadd_2 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_2!($ni), cr!(0, $ni), dr!(0, $ni)),
            vfmadd!(1, br_2!($ni), cr!(1, $ni), dr!(1, $ni)),
        )
    };
}

macro_rules! fmadd_1 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_1!($ni), cr!(0, $ni), dr!(0, $ni)),
        )
    };
}

macro_rules! vzero_kernel {
    () => { vzeroall!(4,11) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(4,11) };
}

#[inline(always)]
fn mask_and_offset(m: usize) -> ([u32;16], usize) {
    let mask: [u32; 16] = [
        u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX,
        0, 0, 0, 0, 0, 0, 0, 0,
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
        let mask = [0xFFFF_u32];
        let $nm = mask.as_ptr();
        let $mask_ptr = $nm;
    };
}

macro_rules! load_mask {
    (P) => { "vmovdqu ({maskx}), %ymm1" };
    (C) => { "/* {maskx} */" }
}

def_ukernel_avx!(4, step_2, acc_2, store_2, 2, 4, B, P, ukernel_2_bbp);
def_ukernel_avx!(4, step_1, acc_1, store_1, 1, 4, B, P, ukernel_1_bbp);

// def_ukernel_avx_2!(4, step_2, acc_2, store_2, 2, 4, 16, 32);
def_ukernel_avx!(4,step_2, acc_2, store_2, 2, 4, B, C, ukernel_bbc);
