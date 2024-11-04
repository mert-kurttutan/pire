use seq_macro::seq;
use std::arch::asm;
use crate::UnaryFnC;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use glar_base::{
    load_buf, store_buf, c_mem, def_ukernel_avx512,
    c_reg_2x12, c_reg_1x12, b_num_2x12, b_num_1x12, dim_to_reg,
    cum_seq, load_a_avx512, storep_avx512, acc_p_avx512, init_ab, init_ab_2,
    def_ukernel_avx512_2,
};

type TS = f32;

const ZERO: i32 = 0;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

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
            "vmovups ", $m0, ", %zmm30 {{%k1}}", "\n",
            "vcvtdq2ps %zmm30,%zmm30", "\n",
            "vcvtdq2ps %zmm", $r, ",%zmm", $r, "\n",
            "vfmadd231ps %zmm30,%zmm0,%zmm", $r, "\n",
            "vcvtps2dq %zmm", $r, ",%zmm", $r, "\n",
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
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vpmaddwd %zmm", $r1, ", %zmm", $r2, ", %zmm", $r4, "\n",
            "vpaddd %zmm", $r4, ", %zmm", $r3, ", %zmm", $r3, "\n",
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
                // jmp to 8 if alpha is equal to one
                "cmp $0x3f800000, {alphax} \n", // 0x3f800000 is 1 in float
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
            "vxorps %zmm31,%zmm31,%zmm31\n",
            "vucomiss %xmm31,%xmm0\n",
        )
    }
}

macro_rules! acc_2x8 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx512!($layout, c_mem!($ni), $q, c_reg_2x12!(0,$ni), c_reg_2x12!(1,$ni))
    };
}

macro_rules! store_2x8 {
    ($ni:tt, $layout:tt) => {
        storep_avx512!($layout, c_mem!($ni), c_reg_2x12!(0,$ni), c_reg_2x12!(1,$ni))
    };
}

macro_rules! acc_1x8 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx512!($layout, c_mem!($ni), $q, c_reg_1x12!(0,$ni))
    };
}

macro_rules! store_1x8 {
    ($ni:tt, $layout:tt) => {
        storep_avx512!($layout, c_mem!($ni), c_reg_1x12!(0,$ni))
    };
}

macro_rules! load_b {
    (B, $N:tt, $r:expr) => {
        concat!(
            vbroadcast!(), "  ", $N, "*4({bx}), %zmm", $r, "\n",
        )
    };
}

macro_rules! fmadd_2v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 8, 24),
            vfmadd!(1, 2, 9, 25),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 3, 10, 26),
            vfmadd!(1, 3, 11, 27),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 4, 12, 28),
            vfmadd!(1, 4, 13, 29),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 5, 14, 30),
            vfmadd!(1, 5, 15, 31),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 6, 16, 24),
            vfmadd!(1, 6, 17, 25),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 7, 18, 26),
            vfmadd!(1, 7, 19, 27),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 2, 20, 28),
            vfmadd!(1, 2, 21, 29),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 3, 22, 30),
            vfmadd!(1, 3, 23, 31),
        )
    };
}

macro_rules! fmadd_1v {
    (0) => {
        concat!(
            vfmadd!(0, 1, 20, 10),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 21, 11),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 3, 22, 12),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 4, 23, 13),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 5, 24, 14),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 6, 25, 15),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 7, 26, 16),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 8, 27, 17),
        )
    };
}

// ***************************** 2x8 ******************************* //
macro_rules! step_2x8 {
    (8, B) => {
        concat!(

            load_a_avx512!(2),
            "add $128, {ax}\n",
            load_b!(B, 0, 2),
            fmadd_2v!(0),

            load_b!(B, 1, 3),
            fmadd_2v!(1),

            load_b!(B, 2, 4),
            "prefetcht0 256({ax}) \n",
            fmadd_2v!(2),

            load_b!(B, 3, 5),
            fmadd_2v!(3),

            load_b!(B, 4, 6),
            "prefetcht0 320({ax}) \n",
            fmadd_2v!(4),

            load_b!(B, 5, 7),
            "prefetcht0 64({bx}) \n",
            fmadd_2v!(5),

            load_b!(B, 6, 2),
            fmadd_2v!(6),

            load_b!(B, 7, 3),
            fmadd_2v!(7),

            "add $32, {bx}\n",	
        )
    };
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx512!(2),
                "add $128, {ax}\n",
                #(
                    load_b!($b_layout, n, b_num_2x12!(n)),
                    fmadd_2v!(n),
                )*
                "add $4*", $nr, ", {bx}\n",
            )
        })
    };
}

// ***************************** 1x8 ******************************* //
macro_rules! step_1x8 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx512!(1),
                "add $64, {ax}\n",
                #(
                    load_b!($b_layout, n, b_num_1x12!(n)),
                    fmadd_1v!(n),
                )*
                "add $4*", $nr, ", {bx}\n",
            )
        })
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
    (P) => {
        "kmovw ({maskx}), %k1"
    };
    (C) => {
        "/* {maskx} */"
    }
}

macro_rules! vzero_kernel {
    () => { vzeroall!(8,23) };
}

macro_rules! alpha_scale {
    ($mr:tt,$nr:tt) => { dim_to_reg!(alpha_scale_0, $mr, $nr) };
}
 
def_ukernel_avx512!(2, step_2x8, acc_2x8, store_2x8, 2, 8, 8, 9, B, P, ukernel_2_bbp);
def_ukernel_avx512!(2, step_1x8, acc_1x8, store_1x8, 1, 8, 8, 9, B, P, ukernel_1_bbp);


def_ukernel_avx512!(2, step_2x8, acc_2x8, store_2x8, 2, 8, 1, 8, B, C, ukernel_n_bbc);

def_ukernel_avx512!(2, step_2x8, acc_2x8, store_2x8, 2, 8, 1, 8, B, P, ukernel_2xn_bbp);
def_ukernel_avx512!(2, step_1x8, acc_1x8, store_1x8, 1, 8, 1, 8, B, P, ukernel_1xn_bbp);

def_ukernel_avx512_2!(2, step_2x8, acc_2x8, store_2x8, 2, 8, 16, 32);
