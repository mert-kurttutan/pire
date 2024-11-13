use seq_macro::seq;
use std::arch::asm;
use half::f16;
use crate::{UnaryFnC, TC, TC_SIZE};
use glar_base::{
    c_mem, def_ukernel_avx, mem,
    init_ab_avx, dim_to_reg_avx, c_reg_2x4, c_reg_1x4,
    b_num_2x4, b_num_1x4,
    load_a_avx,
};

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
            "vmulps %ymm2, %ymm0,%ymm2", "\n",
            "vaddps %ymm2, %ymm", $r1, ",%ymm", $r1, "\n",
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
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vmulps %ymm", $r1, ", %ymm", $r2,", %ymm", $r4, "\n",
            "vaddps %ymm", $r4, ", %ymm", $r3, ", %ymm", $r3, "\n",
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
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %ymm0\n",
            "vxorps %ymm3,%ymm3,%ymm3\n",
            "vucomiss %xmm3,%xmm0\n",
        )
    }
}

macro_rules! acc_p {
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

macro_rules! storep {
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
    () => { vzeroall!(4, 11) };
}

macro_rules! alpha_scale {
    ($mr:tt,$nr:tt) => { dim_to_reg_avx!(alpha_scale_0, $mr, $nr) };
}

macro_rules! acc_2x4 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!($layout, c_mem!($ni), $q, c_reg_2x4!(0,$ni), c_reg_2x4!(1,$ni))
    };
}

macro_rules! store_2x4 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_2x4!(0,$ni), c_reg_2x4!(1,$ni))
    };
}

macro_rules! acc_1x4 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!($layout, c_mem!($ni), $q, c_reg_1x4!(0,$ni))
    };
}

macro_rules! store_1x4 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_1x4!(0,$ni))
    };
}


macro_rules! load_b {
    (B, $N:tt, $K:tt, $X:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*4+", $N, "*4({bx}), %ymm", $r, "\n",
        )
    };
}

macro_rules! fmadd_2v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 4, 12),
            vfmadd!(1, 2, 5, 13),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 3, 6, 14),
            vfmadd!(1, 3, 7, 15),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 2, 8, 12),
            vfmadd!(1, 2, 9, 13),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 3, 10, 14),
            vfmadd!(1, 3, 11, 15),
        )
    };
}

macro_rules! fmadd_1v {
    (0) => {
        concat!(vfmadd!(0, 1, 7, 11))
    };
    (1) => {
        concat!(vfmadd!(0, 2, 8, 12))
    };
    (2) => {
        concat!(vfmadd!(0, 3, 9, 13))
    };
    (3) => {
        concat!(vfmadd!(0, 4, 10, 14))
    };
}

// ***************************** 2x4 ******************************* //
macro_rules! step_2x4 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx!(2, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_2x4!(n)),
                    fmadd_2v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 1x4 ******************************* //
macro_rules! step_1x4 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx!(1, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_1x4!(n)),
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

def_ukernel_avx!(1, step_2x4, acc_2x4, store_2x4, 2, 4, 4, 5, B, C, ukernel_bbc);

def_ukernel_avx!(1, step_2x4, acc_2x4, store_2x4, 2, 4, 4, 5, B, C, ukernel_2_bbp);
def_ukernel_avx!(1, step_1x4, acc_1x4, store_1x4, 1, 4, 4, 5, B, C, ukernel_1_bbp);

def_ukernel_avx!(1, step_2x4, acc_2x4, store_2x4, 2, 4, 1, 4, B, C, ukernel_n_bbc);

def_ukernel_avx!(1, step_2x4, acc_2x4, store_2x4, 2, 4, 1, 4, B, C, ukernel_2xn_bbp);
def_ukernel_avx!(1, step_1x4, acc_1x4, store_1x4, 1, 4, 1, 4, B, C, ukernel_1xn_bbp);
