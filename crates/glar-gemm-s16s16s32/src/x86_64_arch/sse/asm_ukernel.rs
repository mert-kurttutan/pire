use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC, UnaryFnC, TC_SIZE};
use super::VS;
use glar_base::{
    c_mem, def_ukernel_sse, 
    mem, c_reg_2x4, c_reg_1x4,
    b_num_2x4, b_num_1x4, dim_to_reg_avx,
    init_ab_avx,
};

type TS = f32;

const ZERO: i32 = 0;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;
macro_rules! beta_fmadd {
    (C, $m0:expr, $r:expr, 1) => {
        concat!(
            "movups ", $m0, ", %xmm2", "\n",
            "paddd ", "%xmm2", ", %xmm", $r, "\n",
            // "paddd ", $m0, ", %xmm", $r, "\n",
        ) 
    };
    (C, $m0:expr, $r:expr, 2) => {
        concat!(
            "cvtdq2ps %xmm", $r, ",%xmm", $r, "\n",
            "movups ", $m0, ",%xmm2", "\n",
            "cvtdq2ps %xmm2", ",%xmm2", "\n",
            "mulps %xmm0, %xmm2", "\n",
            "addps %xmm2, %xmm", $r, "\n",
            "cvtps2dq %xmm", $r, ",%xmm", $r, "\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("xorps %xmm",r,",%xmm",r,"\n",)*)
        })
    }
}

macro_rules! vbroadcast {
    () => {
        "movss"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "movups %xmm", $r2, ", %xmm", $r4, "\n",
            "pmaddwd %xmm", $r1, ", %xmm", $r4, "\n",
            "paddd %xmm", $r4, ", %xmm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "movaps ", $m0, ",%xmm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "movups %xmm", $r1, ", ", $m0,  "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                // jmp to 8 if alpha is equal to one
                "cmp $0x3f800000, {alphax} \n", // 0x3f800000 is 1 in float
                "movss ({alphax}),%xmm1", "\n",
                "shufps $0,%xmm1,%xmm1", "\n",
                "je 8f \n",
                #(
                    "cvtdq2ps %xmm", r, ",%xmm", r, "\n",
                    "mulps %xmm1, %xmm", r, "\n",
                    "cvtps2dq %xmm", r, ",%xmm", r, "\n",
                )*
                "8:", "\n",
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %xmm0\n",
            "shufps $0, %xmm0, %xmm0\n",
            "xorps %xmm3,%xmm3\n",
            "ucomiss %xmm3,%xmm0\n",
        )
    }
}

macro_rules! acc_p {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $b:tt) => {
        concat!(
            beta_fmadd!(C, $m0, $r1, $b),
            beta_fmadd!($layout, mem!($m0, "0x10"), $r2, $b),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $b:tt) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1, $b),
        )
    };
}


macro_rules! loadp {
    (2, $m0:expr) => {
        concat!(
            loadp_unit!($m0, 0),
            loadp_unit!(mem!($m0, "0x10"), 1),
        )
    };
    (1, $m0:expr) => {
        concat!(
            loadp_unit!($m0, 0),
        )
    };
}

macro_rules! storep {
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

macro_rules! inc_a_k_unroll {
    ($X:tt, $K:tt) => {
        concat!(
            "add $16*", $K, "*", $X, ",{ax}", "\n",
        )
    };
}

macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => {
        ""
    };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $4*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}

macro_rules! acc_2x4 {
    ($ni:tt, $layout:tt, $b:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_2x4!(0,$ni), c_reg_2x4!(1,$ni), $b)
    };
}

macro_rules! store_2x4 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_2x4!(0,$ni), c_reg_2x4!(1,$ni))
    };
}

macro_rules! acc_1x4 {
    ($ni:tt, $layout:tt, $b:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_1x4!(0,$ni), $b)
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
            vbroadcast!(), "  ", $K, "*", $X, "*4+", $N, "*4({bx}), %xmm", $r, "\n",
            "shufps $0, %xmm", $r, ", %xmm", $r, "\n",
        )
    };
}

macro_rules! load_a {
    ($mr:tt, $K:tt) => {
        loadp!($mr, concat!($mr,"*16*",$K,"({ax})"))
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
        concat!(
            vfmadd!(0, 1, 5, 9),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 6, 10),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 3, 7, 11),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 4, 8, 12),
        )
    };
}

macro_rules! vzero_kernel {
    () => { vzeroall!(4,11) };
}

macro_rules! alpha_scale {
    ($mr:tt,$nr:tt) => { dim_to_reg_avx!(alpha_scale_0, $mr, $nr) };
}

// ***************************** 2x4 ******************************* //
macro_rules! step_2x4 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_2x4!(n)),
                    fmadd_2v!(n),
                )*
            )
        })
    };
}

// ***************************** 1x4 ******************************* //
macro_rules! step_1x4 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(1, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_1x4!(n)),
                    fmadd_1v!(n),
                )*
            )
        })
    };
}

def_ukernel_sse!(2, step_2x4, acc_2x4, store_2x4, 2, 4, 4, 5, B, C, ukernel_bbc);

def_ukernel_sse!(2, step_2x4, acc_2x4, store_2x4, 2, 4, 4, 5, B, C, ukernel_2_bbp);
def_ukernel_sse!(2, step_1x4, acc_1x4, store_1x4, 1, 4, 4, 5, B, C, ukernel_1_bbp);


def_ukernel_sse!(2, step_2x4, acc_2x4, store_2x4, 2, 4, 1, 4, B, C, ukernel_n_bbc);

def_ukernel_sse!(2, step_2x4, acc_2x4, store_2x4, 2, 4, 1, 4, B, C, ukernel_2xn_bbp);
def_ukernel_sse!(2, step_1x4, acc_1x4, store_1x4, 1, 4, 1, 4, B, C, ukernel_1xn_bbp);
