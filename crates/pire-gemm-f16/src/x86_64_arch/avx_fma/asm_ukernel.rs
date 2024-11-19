use seq_macro::seq;
use std::arch::asm;
use half::f16;
use pire_base::{
    c_mem, def_ukernel_avx, mem,
    c_reg_3x4, c_reg_2x6, c_reg_1x6, acc_3x4, acc_2x6, acc_1x6,
    store_3x4, store_2x6, store_1x6, b_num_2x6, b_num_1x6, init_ab_avx, load_a_avx,
};
use crate::{UnaryFnC, TC, TC_SIZE};

type TA = f32;
type TB = f32;
type TS = f32;

const VS: usize = 8;
const ZERO: f16 = f16::ZERO;
const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vcvtph2ps ", $m0, ", %ymm2\n",
            "vaddps %ymm2, %ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "vcvtph2ps ", $m0, ", %ymm2\n",
            "vfmadd231ps %ymm2, %ymm0, %ymm", $r1, "\n",
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
    () => {
        "vbroadcastss"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231ps %ymm", $r1, ", %ymm", $r2,", %ymm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovaps ", $m0, ",%ymm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vcvtps2ph $0x00, %ymm", $r1, ", ", $m0, "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                vbroadcast!(), " ({alphax}),%ymm1", "\n",
                #(
                    "vmulps %ymm1, %ymm", r, ",%ymm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    // ($partial:tt) => {
    //     concat!(
    //         vbroadcast!(), " ({betax}), %ymm0\n",
    //         "vxorps %ymm3,%ymm3,%ymm3\n",
    //         "vucomiss %xmm3,%xmm0\n",
    //         load_mask!($partial), "\n",
    //         "je 6f", // STORE
    //     )
    // };
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %ymm0\n",
            "vxorps %ymm3,%ymm3,%ymm3\n",
            "vucomiss %xmm3,%xmm0\n",
        )
    };
}

macro_rules! acc_p_avx {
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1, $q),
            beta_fmadd!(C, mem!($m0, "0x10"), $r2, $q),
            beta_fmadd!($layout, mem!($m0, "0x20"), $r3, $q),
        )
    };
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1, $q),
            beta_fmadd!($layout, mem!($m0, "0x10"), $r2, $q),
        )
    };
    ($layout:tt, $m0:expr, $q:tt, $r1:expr) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1, $q),
        )
    };
}

macro_rules! storep_avx {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            storep_unit!(C, $r1, $m0),
            storep_unit!(C, $r2, mem!($m0, "0x10")),
            storep_unit!($layout, $r3, mem!($m0, "0x20")),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            storep_unit!(C, $r1, $m0),
            storep_unit!($layout, $r2, mem!($m0, "0x10")),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            storep_unit!($layout, $r1, $m0),
        )
    };
}

macro_rules! inc_b {
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
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $4*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}

macro_rules! vzero_kernel {
    () => { vzeroall!(4, 15) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(4, 15) };
}



macro_rules! load_b {
    (B, $N:tt, $K:tt, $X:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*4+", $N, "*4({bx}), %ymm", $r, "\n",
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

macro_rules! mask_ptr {
    (C, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let mask = [0xFFFF_u32];
        let $nm = mask.as_ptr();
        let $mask_ptr = $nm;
    };
}

macro_rules! load_mask {
    (C) => {
        "/* {maskx} */"
    }
}

// NOTE: BS ukernel for f16 is hard to implement since it requires loading single f16 in a strided fashion
// we can do this, it will require avx2, and som other issues, which I dont want to deal with
// Additiionally and more importantly, it wont be performant neough since it reqiures to convert additioanl
// computation, it wont benefit from vectorization since we load single f16 in strided layout.

// Dont use partial since partially (and efficiently at the same time) is hard instead copy to c buffer

def_ukernel_avx!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 4, 5, B, C, ukernel_bbc);

def_ukernel_avx!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 4, 5, B, C, ukernel_3_bbp);
def_ukernel_avx!(1, step_2x6, acc_2x6, store_2x6, 2, 4, 4, 5, B, C, ukernel_2_bbp);
def_ukernel_avx!(1, step_1x6, acc_1x6, store_1x6, 1, 4, 4, 5, B, C, ukernel_1_bbp);


def_ukernel_avx!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 1, 4, B, C, ukernel_n_bbc);

def_ukernel_avx!(1, step_3x4, acc_3x4, store_3x4, 3, 4, 1, 4, B, C, ukernel_3xn_bbp);
def_ukernel_avx!(1, step_2x6, acc_2x6, store_2x6, 2, 4, 1, 4, B, C, ukernel_2xn_bbp);
def_ukernel_avx!(1, step_1x6, acc_1x6, store_1x6, 1, 4, 1, 4, B, C, ukernel_1xn_bbp);



