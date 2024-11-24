use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC, TC_SIZE};
use crate::UnaryFnC;
use pire_base::{c_mem, def_ukernel_sse};
type TS = TC;

const ZERO_SCALAR: TC = TC::ZERO;
const ONE_SCALAR: TC = TC::ONE;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "movups ", $m0, ",%xmm2", "\n",
            "addps ", "%xmm2", ",%xmm", $r1, "\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "movups ", $m0, ", %xmm5", "\n",
            complex_mul!(5, 7),
            "addps %xmm5, %xmm", $r1, "\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("xorpd %xmm",r,",%xmm",r,"\n",)*)
        })
    }
}

macro_rules! vbroadcast {
    () => {
        "movss"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $b1:expr, $r2:expr, $r4:expr) => {
        concat!(
            "movups %xmm", $b1, ", %xmm", $r4, "\n",
            "mulps %xmm", $r1, ", %xmm", $r4, "\n",
            "addps %xmm", $r4, ", %xmm", $r2, "\n",
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

macro_rules! complex_mul {
    ($r0:tt, $rt:tt) => {
        concat!(
            "movaps %xmm", $r0, ", %xmm", $rt, "\n",
            "shufps $0xb1, %xmm", $r0, ", %xmm", $rt, "\n",
            "mulps %xmm0, %xmm", $r0, "\n",
            "mulps %xmm1, %xmm", $rt, "\n",
            "addsubps %xmm", $rt, ", %xmm", $r0, "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        concat!(
            "mov ({ptr_arrx}), {bx}\n",
            vbroadcast!(), " ({bx}),%xmm0", "\n",
            "shufps ", "$0, %xmm0, %xmm0", "\n",

            vbroadcast!(), " 4({bx}),%xmm1", "\n",
            "shufps ", "$0, %xmm1, %xmm1", "\n",
        
            complex_mul!(4, 5),
            complex_mul!(6, 7),
        )
    }
}

macro_rules! v_to_c {
    ($r0:tt, $r1:tt) => {
        concat!(
            "shufps $0xb1, %xmm", $r1, ", %xmm", $r1, "\n",
            "addsubps %xmm", $r1, ", %xmm", $r0, "\n",
        )
    }
}

macro_rules! permute_complex {
    () => {
        concat!(
            // permute even and odd elements
            v_to_c!(4, 5),
            v_to_c!(6, 7),
        )
    }
}

macro_rules! acc_p {
    ($layout:tt, $m0:expr, $q:tt, $r1:expr) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1, $q),
        )
    };
}


macro_rules! loadp {
    (2, $m0:expr) => {
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

macro_rules! load_beta {
    () => {
        concat!(
            "mov 4({ptr_arrx}), {bx}\n",
            "movss ({bx}), %xmm0 \n",
            "shufps ", "$0, %xmm0, %xmm0", "\n",
            "movss 4({bx}), %xmm1 \n",
            "shufps ", "$0, %xmm1, %xmm1", "\n",
        )
    }
}

// only non contigous along m and n direction which is not changed frequently during iteration along k direction
/*

bx -> cs_a
x2 -> cs_b
x3 -> ax + 3*cs_a
x4 -> bx + 3*cs_b

*/


macro_rules! init_ab {
    (B) => {
        concat!(
            "mov 4({dim_arrx}),{x0}", "\n",
        )
    };
}


macro_rules! c_load {
    () => {
        concat!(
            permute_complex!(),
            "mov 0({dim_arrx}),{x0}", "\n",
        )
    };
}


macro_rules! vzero_kernel {
    () => { vzeroall!(4,7) };
}

macro_rules! inc_b {
    (S,2) => {
        "add {x0},{bx} \n"
    };
    (S,1) => {
        "add {x0},{bx} \n"
    };
    (B,$nr:tt) => {
        ""
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
            "add $8*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}


macro_rules! alpha_scale {
    () => {
        alpha_scale_0!(4,7)
    };
}

macro_rules! c_reg_1x2 {
    (0,0) => { 4 };
    (0,1) => { 6 };
}

macro_rules! acc_1x2 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!($layout, c_mem!($ni), $q, c_reg_1x2!(0,$ni))
    };
}

macro_rules! store_1x2 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_1x2!(0,$ni))
    };
}


macro_rules! load_b {
    (S, 0, $K:tt, $X:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx}),%xmm", $r1, "\n",
            "shufps $0, %xmm", $r1, ", %xmm", $r1, "\n",
        )
    };
    (S, 1, $K:tt, $X:tt, $r1:expr, $r2:expr) => {
        concat!(
            // "prefetcht0 64({bx},{x2},1) \n",
            // vbroadcast!(), " ({bx},{x2},1),%xmm", $r1, "\n",
            "shufps $0, %xmm", $r1, ", %xmm", $r1, "\n",
        )
    };
    (B, $N:tt, $K:tt, $X:tt, $r1:expr, 0) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*8+", $N, "*8({bx}), %xmm", $r1, "\n",
            "shufps $0, %xmm", $r1, ", %xmm", $r1, "\n",
        )
    };

    (B, $N:tt, $K:tt, $X:tt, $r1:expr, 1) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*8+", $N, "*8+4({bx}), %xmm", $r1, "\n",
            "shufps $0, %xmm", $r1, ", %xmm", $r1, "\n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt, $K:tt) => {
        loadp!($mr, concat!($mr,"*8*",$K,"({ax})"))
    };
}

macro_rules! fmadd_1v {
    (0, 0) => {
        concat!(
            vfmadd!(0, 1, 4, 2),
        )
    };
    (0, 1) => {
        concat!(
            vfmadd!(0, 1, 5, 3),
        )
    };
    (1, 0) => {
        concat!(
            vfmadd!(0, 1, 6, 2),
        )
    };
    (1, 1) => {
        concat!(
            vfmadd!(0, 1, 7, 3),
        )
    };
}

// ***************************** 1x2 ******************************* //
macro_rules! step_1x2 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, 1, 0),
                    fmadd_1v!(n, 0),
                    load_b!($b_layout, n, $K, $nr, 1, 1),
                    fmadd_1v!(n, 1),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

def_ukernel_sse!(1, step_1x2, acc_1x2, store_1x2, 1, 2, B, C, ukernel_bbc);

def_ukernel_sse!(1, step_1x2, acc_1x2, store_1x2, 1, 2, B, C, ukernel_1_bbp);


