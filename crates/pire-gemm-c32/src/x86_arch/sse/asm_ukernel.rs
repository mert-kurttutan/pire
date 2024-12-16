use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    mem, def_ukernel_sse,
    acc_1, store_1,
};
type TS = TC;

const ZERO_SCALAR: TC = TC::ZERO;
const ONE_SCALAR: TC = TC::ONE;

macro_rules! vs {
    () => { "0x10" };
}

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x10+" , $m) };
}

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

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "movaps ", mem!($m0, concat!("0x10*", $r1)), ", %xmm", $r1, "\n",
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
    (B,$nr:tt) => {
        ""
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

macro_rules! cr {
    (0,0) => { 4 };
    (0,1) => { 6 };
}

macro_rules! load_b {
    (B, $nr:tt, $ni:tt, $K:tt, $r:expr, 0) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $nr, "*8+", $ni, "*8({bx}), %xmm", $r, "\n",
            "shufps $0, %xmm", $r, ", %xmm", $r, "\n",
        )
    };

    (B, $nr:tt, $ni:tt, $K:tt, $r:expr, 1) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $nr, "*8+", $ni, "*8+4({bx}), %xmm", $r, "\n",
            "shufps $0, %xmm", $r, ", %xmm", $r, "\n",
        )
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

// ***************************** 1 ******************************* //
macro_rules! step_1 {
    ($b_layout:tt, $nr:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, $nr, n, $K, 1, 0),
                    fmadd_1v!(n, 0),
                    load_b!($b_layout, $nr, n, $K, 1, 1),
                    fmadd_1v!(n, 1),
                )*
            )
        })
    };
}

def_ukernel_sse!(1, step_1, acc_1, store_1, 1, 2, B, C, ukernel_bbc);

def_ukernel_sse!(1, step_1, acc_1, store_1, 1, 2, B, C, ukernel_1_bbp);


