use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    c_mem, def_ukernel_avx512, def_ukernel_avx512_2,
    b_num_3x8, b_num_2x12, b_num_1x12, c_reg_3x8, c_reg_2x12, c_reg_1x12,
    load_a_avx512, storep_avx512, acc_p_avx512,
    acc_3x8, store_3x8, acc_2x12, store_2x12, acc_1x12, store_1x12, init_ab, init_ab_2,
};
type TS = f32;

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
            "vcvtdq2ps ", $m0, ",%zmm7", "\n",
            "vfmadd231ps %zmm7,%zmm0,%zmm", $r, "\n",
            "vcvtps2dq %zmm", $r, ",%zmm", $r, "\n",
        ) 
    };
    (P, $m0:expr, $r:expr, 1) => {
        concat!(
            "vmovups ", $m0, ", %zmm7 {{%k1}}", "\n",
            "vpaddd %zmm7, %zmm", $r, ", %zmm", $r, "\n",
        ) 
    };

    (P, $m0:expr, $r:expr, 2) => {
        concat!(
            "vcvtdq2ps ", $m0, ", %zmm7 {{%k1}}", "\n",
            "vcvtdq2ps %zmm", $r, ",%zmm", $r, "\n",
            "vfmadd231ps %zmm7,%zmm0,%zmm", $r, "\n",
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
    () => {
        "vbroadcastss"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vpdpbusds %zmm", $r1, ", %zmm", $r2, ", %zmm", $r3, "\n",
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
            "vxorps %zmm6,%zmm6,%zmm6\n",
            "vucomiss %xmm6,%xmm0\n",
        )
    }
}

macro_rules! vzero_kernel {
    () => { vzeroall!(8,31) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(8,31) };
}

macro_rules! load_b {
    (B, $N:tt, $r:expr) => {
        concat!(
            vbroadcast!(), "  ", $N, "*4({bx}), %zmm", $r, "\n",
        )
    };
}

macro_rules! fmadd_3v {
    (0) => {
        concat!(
            vfmadd!(0, 3, 8),
            vfmadd!(1, 3, 9),
            vfmadd!(2, 3, 10),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 4, 11),
            vfmadd!(1, 4, 12),
            vfmadd!(2, 4, 13),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 5, 14),
            vfmadd!(1, 5, 15),
            vfmadd!(2, 5, 16),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 6, 17),
            vfmadd!(1, 6, 18),
            vfmadd!(2, 6, 19),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 7, 20),
            vfmadd!(1, 7, 21),
            vfmadd!(2, 7, 22),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 3, 23),
            vfmadd!(1, 3, 24),
            vfmadd!(2, 3, 25),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 4, 26),
            vfmadd!(1, 4, 27),
            vfmadd!(2, 4, 28),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 5, 29),
            vfmadd!(1, 5, 30),
            vfmadd!(2, 5, 31),
        )
    };
}

macro_rules! fmadd_2v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 8),
            vfmadd!(1, 2, 9),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 3, 10),
            vfmadd!(1, 3, 11),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 4, 12),
            vfmadd!(1, 4, 13),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 5, 14),
            vfmadd!(1, 5, 15),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 6, 16),
            vfmadd!(1, 6, 17),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 7, 18),
            vfmadd!(1, 7, 19),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 2, 20),
            vfmadd!(1, 2, 21),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 3, 22),
            vfmadd!(1, 3, 23),
        )
    };
    (8) => {
        concat!(
            vfmadd!(0, 4, 24),
            vfmadd!(1, 4, 25),
        )
    };
    (9) => {
        concat!(
            vfmadd!(0, 5, 26),
            vfmadd!(1, 5, 27),
        )
    };
    (10) => {
        concat!(
            vfmadd!(0, 6, 28),
            vfmadd!(1, 6, 29),
        )
    };
    (11) => {
        concat!(
            vfmadd!(0, 7, 30),
            vfmadd!(1, 7, 31),
        )
    };
}

macro_rules! fmadd_1v {
    (0) => {
        concat!(
            vfmadd!(0, 1, 9),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 10),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 3, 11),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 4, 12),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 5, 13),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 6, 14),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 7, 15),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 8, 16),
        )
    };
}

// ***************************** 3x8 ******************************* //
macro_rules! step_3x8 {
    (8, B) => {
        concat!(

            load_a_avx512!(3),
            "add $192, {ax}\n",
            load_b!(B, 0, 3),
            fmadd_3v!(0),

            load_b!(B, 1, 4),
            fmadd_3v!(1),

            load_b!(B, 2, 5),
            "prefetcht0 256({ax}) \n",
            fmadd_3v!(2),

            load_b!(B, 3, 6),
            fmadd_3v!(3),

            load_b!(B, 4, 7),
            "prefetcht0 320({ax}) \n",
            fmadd_3v!(4),

            load_b!(B, 5, 3),
            "prefetcht0 192({bx}) \n",
            fmadd_3v!(5),

            load_b!(B, 6, 4),
            fmadd_3v!(6),

            load_b!(B, 7, 5),
            fmadd_3v!(7),

            "add $32, {bx}\n",	
        )
    };
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx512!(3),
                "add $192, {ax}\n",
                #(
                    load_b!($b_layout, n, b_num_3x8!(n)),
                    fmadd_3v!(n),
                )*
                "add $4*", $nr, ", {bx}\n",
            )
        })
    };
}

// ***************************** 2x12 ******************************* //
macro_rules! step_2x12 {
    (12, B) => {
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
            load_b!(B, 8, 4),
            fmadd_2v!(8),
            load_b!(B, 9, 5),
            fmadd_2v!(9),
            load_b!(B, 10, 6),
            fmadd_2v!(10),
            load_b!(B, 11, 7),
            fmadd_2v!(11),

            "add $48, {bx}\n",	
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

// ***************************** 1x12 ******************************* //
macro_rules! step_1x12 {
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

def_ukernel_avx512!(4, step_3x8, acc_3x8, store_3x8, 3, 8, B, P, ukernel_3_bbp);
def_ukernel_avx512!(4, step_2x12, acc_2x12, store_2x12, 2, 8, B, P, ukernel_2_bbp);
def_ukernel_avx512!(4, step_1x12, acc_1x12, store_1x12, 1, 8, B, P, ukernel_1_bbp);

def_ukernel_avx512_2!(4, step_3x8, acc_3x8, store_3x8, 3, 8, 32, 32);
