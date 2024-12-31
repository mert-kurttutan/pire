use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_avx512, def_ukernel_avx512_2,
    acc_3, acc_2, acc_1, store_3, store_2, store_1,
    step_3, step_2, step_1,
    init_ab_2, init_ab,
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

use std::arch::x86_64::*;


#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn ukernel_rcc2(
    a: *const f32, b: *const f32, c: *mut f32, 
    alpha: *const f32, beta: *const f32,
    a_rs: usize, a_cs: usize, b_rs: usize, b_cs: usize, c_rs: usize, c_cs: usize,
    m: usize, n: usize, k: usize,
) {
    // use dot kernel 
    // assume m=6, n=3
    let mut ki = 0;
    let mut c_vec_arr = [_mm512_setzero_ps(); 7*3];
    let mut a_cur = a;
    let mut b_cur = b;
    let dim_arr = [k];
    let mut c00: __m512;
    let mut c10: __m512;
    let mut c20: __m512;
    let mut c30: __m512;
    let mut c40: __m512;
    let mut c50: __m512;
    let mut c60: __m512;

    let mut c01: __m512;
    let mut c11: __m512;
    let mut c21: __m512;
    let mut c31: __m512;
    let mut c41: __m512;
    let mut c51: __m512;
    let mut c61: __m512;

    let mut c02: __m512;
    let mut c12: __m512;
    let mut c22: __m512;
    let mut c32: __m512;
    let mut c42: __m512;
    let mut c52: __m512;
    let mut c62: __m512;
    // _mm_prefetch(c as *const i8, _MM_HINT_T0);
    // _mm_prefetch(c.add(64) as *const i8, _MM_HINT_T1);
    // _mm_prefetch(c.add(c_cs) as *const i8, _MM_HINT_T1);
    // _mm_prefetch(c.add(c_cs+64) as *const i8, _MM_HINT_T1);
    // _mm_prefetch(c.add(c_cs*2) as *const i8, _MM_HINT_T1);
    // _mm_prefetch(c.add(c_cs*2+64) as *const i8, _MM_HINT_T1);

    core::arch::asm!(
        "/* {x4} */",
        "vpxorq %zmm8, %zmm8, %zmm8",
        "vpxorq %zmm9, %zmm9, %zmm9",
        "vpxorq %zmm10, %zmm10, %zmm10",
        "vpxorq %zmm11, %zmm11, %zmm11",
        "vpxorq %zmm12, %zmm12, %zmm12",
        "vpxorq %zmm13, %zmm13, %zmm13",
        "vpxorq %zmm14, %zmm14, %zmm14",
        "vpxorq %zmm15, %zmm15, %zmm15",
        "vpxorq %zmm16, %zmm16, %zmm16",
        "vpxorq %zmm17, %zmm17, %zmm17",
        "vpxorq %zmm18, %zmm18, %zmm18",
        "vpxorq %zmm19, %zmm19, %zmm19",
        "vpxorq %zmm20, %zmm20, %zmm20",
        "vpxorq %zmm21, %zmm21, %zmm21",
        "vpxorq %zmm22, %zmm22, %zmm22",
        "vpxorq %zmm23, %zmm23, %zmm23",
        "vpxorq %zmm24, %zmm24, %zmm24",
        "vpxorq %zmm25, %zmm25, %zmm25",
        "vpxorq %zmm26, %zmm26, %zmm26",
        "vpxorq %zmm27, %zmm27, %zmm27",
        "vpxorq %zmm28, %zmm28, %zmm28",
        // "mov ({dim_arrx}),{x0}\n",


        // "mov 40({dim_arrx}), {x5}",
        "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

        "2:", // KITER
        "vmovups 0({ax}), %zmm0",
        "vmovups 0({ax},{x1}), %zmm1",
        "vmovups 0({ax},{x1},2), %zmm2",
        "vmovups 0({x2}), %zmm3",
        "vmovups 0({x2},{x1}), %zmm4",
        "vmovups 0({x2},{x1},2), %zmm5",
        "vmovups 0({x3}), %zmm6",

        "vmovups 0({bx}), %zmm7",

        "vfmadd231ps %zmm0, %zmm7, %zmm8",
        "vfmadd231ps %zmm1, %zmm7, %zmm9",
        "vfmadd231ps %zmm2, %zmm7, %zmm10",
        "vfmadd231ps %zmm3, %zmm7, %zmm11",
        "vfmadd231ps %zmm4, %zmm7, %zmm12",
        "vfmadd231ps %zmm5, %zmm7, %zmm13",
        "vfmadd231ps %zmm6, %zmm7, %zmm14",

        "vmovups 0({bx},{x5}), %zmm30",
        "vfmadd231ps %zmm0, %zmm30, %zmm15",
        "vfmadd231ps %zmm1, %zmm30, %zmm16",
        "vfmadd231ps %zmm2, %zmm30, %zmm17",
        "vfmadd231ps %zmm3, %zmm30, %zmm18",
        "vfmadd231ps %zmm4, %zmm30, %zmm19",
        "vfmadd231ps %zmm5, %zmm30, %zmm20",
        "vfmadd231ps %zmm6, %zmm30, %zmm21",

        "vmovups 0({bx},{x5},2), %zmm31",
        "vfmadd231ps %zmm0, %zmm31, %zmm22",
        "vfmadd231ps %zmm1, %zmm31, %zmm23",
        "vfmadd231ps %zmm2, %zmm31, %zmm24",
        "vfmadd231ps %zmm3, %zmm31, %zmm25",
        "vfmadd231ps %zmm4, %zmm31, %zmm26",
        "vfmadd231ps %zmm5, %zmm31, %zmm27",
        "vfmadd231ps %zmm6, %zmm31, %zmm28",

        "add $64, {ax}",
        "add $64, {x2}",
        "add $64, {x3}",
        "add $64, {bx}",

        "dec {x0}", "jne 2b", // KITER
        "3:",
        // "vzeroupper",

        ax = inout(reg) a => _,
        bx = inout(reg) b => _,
        // cx = inout(reg) c => _,
        // dim_arrx = inout(reg) dim_arr.as_ptr() => _,

        x0 = inout(reg) k / 16 => _,
        x1 = inout(reg) a_rs*4 => _,
        x2 = inout(reg) a.add(3*a_rs) => _,
        x3 = inout(reg) a.add(6*a_rs) => _,
        x4 = out(reg) _,
        x5 = inout(reg) b_cs*4 => _,
        out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _, out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        out("zmm8") c00, out("zmm9") c10, out("zmm10") c20, out("zmm11") c30, out("zmm12") c40, out("zmm13") c50, out("zmm14") c60, out("zmm15") c01,
        out("zmm16") c11, out("zmm17") c21, out("zmm18") c31, out("zmm19") c41, out("zmm20") c51, out("zmm21") c61, out("zmm22") c02, out("zmm23") c12,
        out("zmm24") c22, out("zmm25") c32, out("zmm26") c42, out("zmm27") c52, out("zmm28") c62, out("zmm29") _, out("zmm30") _, out("zmm31") _,
        options(att_syntax)
    );
    *c += _mm512_reduce_add_ps(c00);
    *c.add(1) += _mm512_reduce_add_ps(c10);
    *c.add(2) += _mm512_reduce_add_ps(c20);
    *c.add(3) += _mm512_reduce_add_ps(c30);
    *c.add(4) += _mm512_reduce_add_ps(c40);
    *c.add(5) += _mm512_reduce_add_ps(c50);
    *c.add(6) += _mm512_reduce_add_ps(c60);

    *c.add(c_cs) += _mm512_reduce_add_ps(c01);
    *c.add(c_cs+1) += _mm512_reduce_add_ps(c11);
    *c.add(c_cs+2) += _mm512_reduce_add_ps(c21);
    *c.add(c_cs+3) += _mm512_reduce_add_ps(c31);
    *c.add(c_cs+4) += _mm512_reduce_add_ps(c41);
    *c.add(c_cs+5) += _mm512_reduce_add_ps(c51);
    *c.add(c_cs+6) += _mm512_reduce_add_ps(c61);

    *c.add(c_cs*2) += _mm512_reduce_add_ps(c02);
    *c.add(c_cs*2+1) += _mm512_reduce_add_ps(c12);
    *c.add(c_cs*2+2) += _mm512_reduce_add_ps(c22);
    *c.add(c_cs*2+3) += _mm512_reduce_add_ps(c32);
    *c.add(c_cs*2+4) += _mm512_reduce_add_ps(c42);
    *c.add(c_cs*2+5) += _mm512_reduce_add_ps(c52);
    *c.add(c_cs*2+6) += _mm512_reduce_add_ps(c62);

}



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

pub(crate) unsafe fn ukernel_rcc3(
    a: *const f32, b: *const f32, c: *mut f32, 
    alpha: *const f32, beta: *const f32,
    a_rs: usize, a_cs: usize, b_rs: usize, b_cs: usize, c_rs: usize, c_cs: usize,
    m: usize, n: usize, k: usize,
) {
    // use dot kernel 
    // assume m=6, n=3
    let dim_arr = [k, c_rs*4, c_cs*4];

    core::arch::asm!(
        "/* {x4} */",
        "vpxorq %zmm8, %zmm8, %zmm8",
        "vpxorq %zmm9, %zmm9, %zmm9",
        "vpxorq %zmm10, %zmm10, %zmm10",
        "vpxorq %zmm11, %zmm11, %zmm11",
        "vpxorq %zmm12, %zmm12, %zmm12",
        "vpxorq %zmm13, %zmm13, %zmm13",
        "vpxorq %zmm14, %zmm14, %zmm14",
        "vpxorq %zmm15, %zmm15, %zmm15",
        "vpxorq %zmm16, %zmm16, %zmm16",
        "vpxorq %zmm17, %zmm17, %zmm17",
        "vpxorq %zmm18, %zmm18, %zmm18",
        "vpxorq %zmm19, %zmm19, %zmm19",
        "vpxorq %zmm20, %zmm20, %zmm20",
        "vpxorq %zmm21, %zmm21, %zmm21",
        "vpxorq %zmm22, %zmm22, %zmm22",
        "vpxorq %zmm23, %zmm23, %zmm23",
        "vpxorq %zmm24, %zmm24, %zmm24",
        "vpxorq %zmm25, %zmm25, %zmm25",
        "vpxorq %zmm26, %zmm26, %zmm26",
        "vpxorq %zmm27, %zmm27, %zmm27",
        "vpxorq %zmm28, %zmm28, %zmm28",
        // "vpxorq %zmm29, %zmm29, %zmm29",
        // "mov ({dim_arrx}),{x0}\n",


        // "mov 40({dim_arrx}), {x5}",
        "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

        "2:", // KITER
        "vmovups 0({ax}), %zmm0",
        "vmovups 0({ax},{x1}), %zmm1",
        "vmovups 0({ax},{x1},2), %zmm2",
        "vmovups 0({x2}), %zmm3",
        "vmovups 0({x2},{x1}), %zmm4",
        "vmovups 0({x2},{x1},2), %zmm5",
        "vmovups 0({x3}), %zmm6",

        "vmovups 0({bx}), %zmm7",

        "vfmadd231ps %zmm0, %zmm7, %zmm8",
        "vfmadd231ps %zmm1, %zmm7, %zmm9",
        "vfmadd231ps %zmm2, %zmm7, %zmm10",
        "vfmadd231ps %zmm3, %zmm7, %zmm11",
        "vfmadd231ps %zmm4, %zmm7, %zmm12",
        "vfmadd231ps %zmm5, %zmm7, %zmm13",
        "vfmadd231ps %zmm6, %zmm7, %zmm14",

        "vmovups 0({bx},{x5}), %zmm30",
        "vfmadd231ps %zmm0, %zmm30, %zmm15",
        "vfmadd231ps %zmm1, %zmm30, %zmm16",
        "vfmadd231ps %zmm2, %zmm30, %zmm17",
        "vfmadd231ps %zmm3, %zmm30, %zmm18",
        "vfmadd231ps %zmm4, %zmm30, %zmm19",
        "vfmadd231ps %zmm5, %zmm30, %zmm20",
        "vfmadd231ps %zmm6, %zmm30, %zmm21",

        "vmovups 0({bx},{x5},2), %zmm31",
        "vfmadd231ps %zmm0, %zmm31, %zmm22",
        "vfmadd231ps %zmm1, %zmm31, %zmm23",
        "vfmadd231ps %zmm2, %zmm31, %zmm24",
        "vfmadd231ps %zmm3, %zmm31, %zmm25",
        "vfmadd231ps %zmm4, %zmm31, %zmm26",
        "vfmadd231ps %zmm5, %zmm31, %zmm27",
        "vfmadd231ps %zmm6, %zmm31, %zmm28",

        "add $64, {ax}",
        "add $64, {x2}",
        "add $64, {x3}",
        "add $64, {bx}",

        "dec {x0}", "jne 2b", // KITER
        "3:",
        // reduce sum all c vectors
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
        "mov 8({dim_arrx}),{x0}",
        "mov 16({dim_arrx}),{x1}",
        "lea ({x0}, {x0}, 2), {x3}",
        "lea ({cx}, {x3}, 1), {x2}",
        "lea ({x2}, {x3}, 1), {x3}",

        "vaddss ({cx}), %xmm8, %xmm8",
        "vmovss %xmm8, 0({cx})",
        "vaddss ({cx},{x0}), %xmm9, %xmm9",
        "vmovss %xmm9, 0({cx},{x0})",
        "vaddss ({cx},{x0},2), %xmm10, %xmm10",
        "vmovss %xmm10, 0({cx},{x0},2)",
        "vaddss ({x2}), %xmm11, %xmm11",
        "vmovss %xmm11, 0({x2})",
        "vaddss ({x2},{x0}), %xmm12, %xmm12",
        "vmovss %xmm12, 0({x2},{x0})",
        "vaddss ({x2},{x0},2), %xmm13, %xmm13",
        "vmovss %xmm13, 0({x2},{x0},2)",
        "vaddss ({x3}), %xmm14, %xmm14",
        "vmovss %xmm14, 0({x3})",

        "add {x1},{cx} \n add {x1},{x2} \n add {x1},{x3} \n",

        "vaddss ({cx}), %xmm15, %xmm15",
        "vmovss %xmm15, 0({cx})",
        "vaddss ({cx},{x0}), %xmm16, %xmm16",
        "vmovss %xmm16, 0({cx},{x0})",
        "vaddss ({cx},{x0},2), %xmm17, %xmm17",
        "vmovss %xmm17, 0({cx},{x0},2)",
        "vaddss ({x2}), %xmm18, %xmm18",
        "vmovss %xmm18, 0({x2})",
        "vaddss ({x2},{x0}), %xmm19, %xmm19",
        "vmovss %xmm19, 0({x2},{x0})",
        "vaddss ({x2},{x0},2), %xmm20, %xmm20",
        "vmovss %xmm20, 0({x2},{x0},2)",
        "vaddss ({x3}), %xmm21, %xmm21",
        "vmovss %xmm21, 0({x3})",

        "add {x1},{cx} \n add {x1},{x2} \n add {x1},{x3} \n",

        "vaddss ({cx}), %xmm22, %xmm22",
        "vmovss %xmm22, 0({cx})",
        "vaddss ({cx},{x0}), %xmm23, %xmm23",
        "vmovss %xmm23, 0({cx},{x0})",
        "vaddss ({cx},{x0},2), %xmm24, %xmm24",
        "vmovss %xmm24, 0({cx},{x0},2)",
        "vaddss ({x2}), %xmm25, %xmm25",
        "vmovss %xmm25, 0({x2})",
        "vaddss ({x2},{x0}), %xmm26, %xmm26",
        "vmovss %xmm26, 0({x2},{x0})",
        "vaddss ({x2},{x0},2), %xmm27, %xmm27",
        "vmovss %xmm27, 0({x2},{x0},2)",
        "vaddss ({x3}), %xmm28, %xmm28",
        "vmovss %xmm28, 0({x3})",


        "vzeroupper",

        ax = inout(reg) a => _,
        bx = inout(reg) b => _,
        cx = inout(reg) c => _,
        dim_arrx = inout(reg) dim_arr.as_ptr() => _,

        x0 = inout(reg) k / 16 => _,
        x1 = inout(reg) a_rs*4 => _,
        x2 = inout(reg) a.add(3*a_rs) => _,
        x3 = inout(reg) a.add(6*a_rs) => _,
        x4 = out(reg) _,
        x5 = inout(reg) b_cs*4 => _,
        out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _, out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _, out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
        out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _, out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
        out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _, out("zmm28") _, out("zmm29") _, out("zmm30") _, out("zmm31") _,
        options(att_syntax)
    );

}


