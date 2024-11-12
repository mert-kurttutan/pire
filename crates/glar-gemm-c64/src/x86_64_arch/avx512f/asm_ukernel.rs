use seq_macro::seq;
use std::arch::asm;
use super::VS;
use crate::{TA, TB, TC, UnaryFnC, TC_SIZE};
use glar_base::{
    c_mem, def_ukernel_avx512, init_ab,
    load_a_avx512, storep_avx512, acc_p_avx512,
    def_ukernel_avx512_2, init_ab_2,
};

type TS = TC;

const ZERO: TC = TC::ZERO;

const ZERO_SCALAR: TC = TC::ZERO;
const ONE_SCALAR: TC = TC::ONE;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "vaddpd ", $m0, ",%zmm", $r1, ",%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr,1) => {
        concat!(
            "vmovupd ", $m0, ", %zmm2 {{%k1}}", "\n",
            "vaddpd %zmm2, %zmm", $r1, ", %zmm", $r1, "\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "vmovupd ", $m0, ", %zmm3", "\n",
            complex_mul!(3, 4, 0),
            "vaddpd %zmm3, %zmm", $r1, ", %zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr,2) => {
        concat!(
            "vmovupd ", $m0, ", %zmm3 {{%k1}}", "\n",
            complex_mul!(3, 4, 0),
            "vaddpd %zmm3, %zmm", $r1, ", %zmm", $r1, "\n",
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
    ($r1:expr, $b1:expr, $b2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vfmadd231pd ", "%zmm", $b1, ", %zmm", $r1,", %zmm", $r3, "\n",
            "vfmadd231pd ", "%zmm", $b2, ", %zmm", $r1,", %zmm", $r4, "\n",
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


macro_rules! complex_mul {
    ($r0:tt, $rt:tt, $rs:tt) => {
        concat!(
            "vpermilpd $0b1010101, %zmm", $r0, ", %zmm", $rt, "\n",
            "vmulpd %zmm1, %zmm", $r0, ", %zmm", $r0, "\n",
            "vmulpd %zmm2, %zmm", $rt, ", %zmm", $rt, "\n",
            "vfmaddsub231pd %zmm0, %zmm", $r0, ", %zmm", $rt, "\n",
            "vmovapd %zmm", $rt, ", %zmm", $r0, "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        concat!(
            permute_complex!(3, 4),
            vbroadcast!(), " ({alphax}), %zmm1 \n",
            vbroadcast!(), " 8({alphax}), %zmm2 \n",

            complex_mul!(8,9,0),
            complex_mul!(10,11,0),
            complex_mul!(12,13,0),
            complex_mul!(14,15,0),
            complex_mul!(16,17,0),
            complex_mul!(18,19,0),
            complex_mul!(20,21,0),
            complex_mul!(22,23,0),
            complex_mul!(24,25,0),
            complex_mul!(26,27,0),
            complex_mul!(28,29,0),
            complex_mul!(30,31,0),
        )
    };
}

macro_rules! v_to_c {
    ($r0:tt, $r1:tt, $rs:tt) => {
        concat!(
            "vpermilpd $0b1010101, %zmm", $r1, ", %zmm", $r1, "\n",
            // "vfmadd231pd %zmm", $r1, ", %zmm", $rs, ", %zmm", $r0, "\n",
            "vfmaddsub231pd %zmm0, %zmm", $r0, ", %zmm", $r1, "\n",
            "vmovapd %zmm", $r1, ", %zmm", $r0, "\n",
        )
    }
}

macro_rules! permute_complex {
    ($mr:tt, $nr:tt) => {
        concat!(
            // permute even and odd elements
            // "vmovupd ({alternate}), %zmm0 \n",
            "mov $0x3FF0000000000000, {ax:r} \n",
            "movd {ax:r}, %xmm0 \n",
            "vbroadcastsd %xmm0, %zmm0 \n",

            v_to_c!(8, 9, 0),
            v_to_c!(10, 11, 0),
            v_to_c!(12, 13, 0),
            v_to_c!(14, 15, 0),
            v_to_c!(16, 17, 0),
            v_to_c!(18, 19, 0),
            v_to_c!(20, 21, 0),
            v_to_c!(22, 23, 0),
            v_to_c!(24, 25, 0),
            v_to_c!(26, 27, 0),
            v_to_c!(28, 29, 0),
            v_to_c!(30, 31, 0),
        )
    };
}

macro_rules! load_beta {
    () => {
        concat!(
            "vbroadcastsd ({betax}), %zmm1\n",
            "vbroadcastsd 8({betax}), %zmm2\n",
        )
    }
}


macro_rules! vzero_kernel {
    () => {vzeroall!(8,31)};
}

macro_rules! alpha_scale {
    (3,4) => {alpha_scale_0!(8,31)};
    (3,3) => {alpha_scale_0!(8,25)};
    (3,2) => {alpha_scale_0!(8,19)};
    (3,1) => {alpha_scale_0!(8,13)};

    (2,6) => {alpha_scale_0!(8,31)};
    (2,5) => {alpha_scale_0!(8,27)};
    (2,4) => {alpha_scale_0!(8,23)};
    (2,3) => {alpha_scale_0!(8,19)};
    (2,2) => {alpha_scale_0!(8,15)};
    (2,1) => {alpha_scale_0!(8,11)};

    (1,6) => {alpha_scale_0!(20,31)};
    (1,5) => {alpha_scale_0!(20,29)};
    (1,4) => {alpha_scale_0!(20,27)};
    (1,3) => {alpha_scale_0!(20,25)};
    (1,2) => {alpha_scale_0!(20,23)};
    (1,1) => {alpha_scale_0!(20,21)};
}

macro_rules! inc_a {
    ($mr:tt) => {
        concat!(
            "add $64*", $mr, ", {ax}", "\n",
        )
    };
}

macro_rules! inc_b {
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
            "add $16*", $nr, ", {bx}", "\n",
        )
    };
}
macro_rules! c_reg_3x4 {
    (0,0) => { 8 };
    (1,0) => { 10 };
    (2,0) => { 12 };
    (0,1) => { 14 };
    (1,1) => { 16 };
    (2,1) => { 18 };
    (0,2) => { 20 };
    (1,2) => { 22 };
    (2,2) => { 24 };
    (0,3) => { 26 };
    (1,3) => { 28 };
    (2,3) => { 30 };
}

macro_rules! c_reg_2x6 {
    (0,0) => { 8 };
    (1,0) => { 10 };
    (0,1) => { 12 };
    (1,1) => { 14 };
    (0,2) => { 16 };
    (1,2) => { 18 };
    (0,3) => { 20 };
    (1,3) => { 22 };
    (0,4) => { 24 };
    (1,4) => { 26 };
    (0,5) => { 28 };
    (1,5) => { 30 };
}

macro_rules! c_reg_1x6 {
    (0,0) => { 20 };
    (0,1) => { 22 };
    (0,2) => { 24 };
    (0,3) => { 26 };
    (0,4) => { 28 };
    (0,5) => { 30 };
}


macro_rules! acc_3x4 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx512!($layout, c_mem!($ni), $q, c_reg_3x4!(0,$ni), c_reg_3x4!(1,$ni), c_reg_3x4!(2,$ni))
    };
}

macro_rules! store_3x4 {
    ($ni:tt, $layout:tt) => {
        storep_avx512!($layout, c_mem!($ni), c_reg_3x4!(0,$ni), c_reg_3x4!(1,$ni), c_reg_3x4!(2,$ni))
    };
}

macro_rules! acc_2x6 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx512!($layout, c_mem!($ni), $q, c_reg_2x6!(0,$ni), c_reg_2x6!(1,$ni))
    };
}

macro_rules! store_2x6 {
    ($ni:tt, $layout:tt) => {
        storep_avx512!($layout, c_mem!($ni), c_reg_2x6!(0,$ni), c_reg_2x6!(1,$ni))
    };
}

macro_rules! acc_1x6 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx512!($layout, c_mem!($ni), $q, c_reg_1x6!(0,$ni))
    };
}

macro_rules! store_1x6 {
    ($ni:tt, $layout:tt) => {
        storep_avx512!($layout, c_mem!($ni), c_reg_1x6!(0,$ni))
    };
}

macro_rules! load_b {
    (S, 0, $r1:expr, $r2:expr) => {
        concat!(
            "prefetcht0 64({bx},{x1},8) \n",
            vbroadcast!(), " ({bx}),%zmm", $r1, "\n",
            vbroadcast!(), " 8({bx}),%zmm", $r2, "\n",
        )
    };
    (S, 1, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx},{x2},1),%zmm", $r1, "\n",
            vbroadcast!(), " 8({bx},{x2},1),%zmm", $r2, "\n",
        )
    };
    (S, 2, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx},{x2},2),%zmm", $r1, "\n",
            vbroadcast!(), " 8({bx},{x2},2),%zmm", $r2, "\n",
        )
    };
    (S, 3, $r1:expr, $r2:expr) => {
        concat!(
            "prefetcht0 64({x3},{x1},8) \n",
            vbroadcast!(), " ({x3}),%zmm", $r1, "\n",
            vbroadcast!(), " 8({x3}),%zmm", $r2, "\n",
        )
    };
    (S, 4, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({x3},{x2},1),%zmm", $r1, "\n",
            vbroadcast!(), " 8({x3},{x2},1),%zmm", $r2, "\n",
        )
    };
    (S, 5, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({x3},{x2},2),%zmm", $r1, "\n",
            vbroadcast!(), " 8({x3},{x2},2),%zmm", $r2, "\n",
        )
    };
    (B, $N:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ", $N, "*16({bx}), %zmm", $r1, "\n",
            vbroadcast!(), " ", $N, "*16+8({bx}), %zmm", $r2, "\n",
        )
    };
}

macro_rules! fmadd_3v {
    (0) => {
        concat!(
            vfmadd!(0, 3, 4, 8, 9),
            vfmadd!(1, 3, 4, 10, 11),
            vfmadd!(2, 3, 4, 12, 13),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 5, 6, 14, 15),
            vfmadd!(1, 5, 6, 16, 17),
            vfmadd!(2, 5, 6, 18, 19),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 7, 3, 20, 21),
            vfmadd!(1, 7, 3, 22, 23),
            vfmadd!(2, 7, 3, 24, 25),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 4, 5, 26, 27),
            vfmadd!(1, 4, 5, 28, 29),
            vfmadd!(2, 4, 5, 30, 31),
        )
    };
}

macro_rules! fmadd_2v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 3, 8, 9),
            vfmadd!(1, 2, 3, 10, 11),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 4, 5, 12, 13),
            vfmadd!(1, 4, 5, 14, 15),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 6, 7, 16, 17),
            vfmadd!(1, 6, 7, 18, 19),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 2, 3, 20, 21),
            vfmadd!(1, 2, 3, 22, 23),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 4, 5, 24, 25),
            vfmadd!(1, 4, 5, 26, 27),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 6, 7, 28, 29),
            vfmadd!(1, 6, 7, 30, 31),
        )
    };
}

macro_rules! fmadd_1v {
    (0) => {
        concat!(
            vfmadd!(0, 1, 2, 20, 21),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 3, 4, 22, 23),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 5, 6, 24, 25),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 7, 8, 26, 27),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 9, 10, 28, 29),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 11, 12, 30, 31),
        )
    };
}

macro_rules! b_num_3x4 {
    (0,0) => {3}; (0,1) => {4};
    (1,0) => {5}; (1,1) => {6};
    (2,0) => {7}; (2,1) => {3};
    (3,0) => {4}; (3,1) => {5};
}

macro_rules! b_num_2x6 {
    (0,0) => {2}; (0,1) => {3};
    (1,0) => {4}; (1,1) => {5};
    (2,0) => {6}; (2,1) => {7};
    (3,0) => {2}; (3,1) => {3};
    (4,0) => {4}; (4,1) => {5};
    (5,0) => {6}; (5,1) => {7};
}

macro_rules! b_num_1x6 {
    (0,0) => {1}; (0,1) => {2};
    (1,0) => {3}; (1,1) => {4};
    (2,0) => {5}; (2,1) => {6};
    (3,0) => {7}; (3,1) => {8};
    (4,0) => {9}; (4,1) => {10};
    (5,0) => {11}; (5,1) => {12};
}

// ***************************** 3x4 ******************************* //
macro_rules! step_3x4 {
    (4, B) => {
        concat!(
            load_a_avx512!(3),
            "addq $192, {ax} \n",
            load_b!(B, 0, 3, 4),
            fmadd_3v!(0),
            load_b!(B, 1, 5, 6),
            fmadd_3v!(1),
            "prefetcht0 384({ax}) \n",
            "prefetcht0 64({bx}) \n",
            load_b!(B, 2, 7, 3),
            fmadd_3v!(2),
            load_b!(B, 3, 4, 5),
            fmadd_3v!(3),
            "prefetcht0 448({ax}) \n",
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
                    load_b!($b_layout, n, b_num_3x4!(n,0), b_num_3x4!(n,1)),
                    fmadd_3v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 2x6 ******************************* //
macro_rules! step_2x6 {
    (6, B) => {
        concat!(
            load_a_avx512!(2),
            "addq $128, {ax} \n",
            load_b!(B, 0, 2, 3),
            fmadd_2v!(0),
            "prefetcht0 320({ax}) \n",
            load_b!(B, 1, 4, 5),
            fmadd_2v!(1),
            "prefetcht0 64({bx}) \n",
            load_b!(B, 2, 6, 7),
            fmadd_2v!(2),
            "prefetcht0 384({ax}) \n",
            load_b!(B, 3, 2, 3),
            fmadd_2v!(3),
            load_b!(B, 4, 4, 5),
            fmadd_2v!(4),
            load_b!(B, 5, 6, 7),
            fmadd_2v!(5),
            "addq $96, {bx} \n",
        )
    };
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx512!(2),
                inc_a!(2),
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, b_num_2x6!(n,0), b_num_2x6!(n,1)),
                    fmadd_2v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 1x6 ******************************* //
macro_rules! step_1x6 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx512!(1),
                inc_a!(1),
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, b_num_1x6!(n,0), b_num_1x6!(n,1)),
                    fmadd_1v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! mask_ptr {
    (P, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let $nm = if $m % VS == 0 && $m > 0 { 0xFF } else { (1_u8 << (($m % VS)*2)) - 1 };
        let $mask_ptr = &$nm as *const u8;
    };
    (C, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let $nm = 0xFF_u8;
        let $mask_ptr = &$nm as *const u8;
    };
}

macro_rules! load_mask {
    (P) => {
        "kmovw ({maskx}), %k1"
    };
    (C) => {
        "/* {maskx} */"
    }
}

def_ukernel_avx512!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 4, 5, B, P, ukernel_3_bbp);
def_ukernel_avx512!(1, step_2x6, acc_2x6, store_2x6, 2, 4, 4, 5, B, P, ukernel_2_bbp);
def_ukernel_avx512!(1, step_1x6, acc_1x6, store_1x6, 1, 4, 4, 5, B, P, ukernel_1_bbp);

def_ukernel_avx512!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 4, 5, S, C, ukernel_bsc);

def_ukernel_avx512!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 4, 5, S, P, ukernel_3_bsp);
def_ukernel_avx512!(1, step_2x6, acc_2x6, store_2x6, 2, 4, 4, 5, S, P, ukernel_2_bsp);
def_ukernel_avx512!(1, step_1x6, acc_1x6, store_1x6, 1, 4, 4, 5, S, P, ukernel_1_bsp);


def_ukernel_avx512!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 1, 4, B, C, ukernel_n_bbc);

def_ukernel_avx512!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 1, 4, B, P, ukernel_3xn_bbp);
def_ukernel_avx512!(1, step_2x6, acc_2x6, store_2x6, 2, 4, 1, 4, B, P, ukernel_2xn_bbp);
def_ukernel_avx512!(1, step_1x6, acc_1x6, store_1x6, 1, 4, 1, 4, B, P, ukernel_1xn_bbp);

def_ukernel_avx512!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 1, 4, S, C, ukernel_n_bsc);

def_ukernel_avx512!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 1, 4, S, P, ukernel_3xn_bsp);
def_ukernel_avx512!(1, step_2x6, acc_2x6, store_2x6, 2, 4, 1, 4, S, P, ukernel_2xn_bsp);
def_ukernel_avx512!(1, step_1x6, acc_1x6, store_1x6, 1, 4, 1, 4, S, P, ukernel_1xn_bsp);


def_ukernel_avx512_2!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 4, 32);
