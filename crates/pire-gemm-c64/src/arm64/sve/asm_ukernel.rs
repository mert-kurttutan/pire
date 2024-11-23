use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC, UnaryFnC, TC_SIZE};
use pire_base::{c_mem, prefetch_0, def_ukernel_sve, mem};
use super::super::sve_vs;

const ZERO: TC = TC::ZERO;

const ONE_SCALAR: TC = TC::ONE;
const ZERO_SCALAR: TC = TC::ZERO;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "ld1d {{ z1.d }}, p0/z, ", $m0, "\n",
            "fadd z", $r1, ".d, p0/m, z", $r1, ".d, z1.d\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,1) => {
        concat!(
            "ld1d {{ z1.d }}, p1/z, ", $m0, "\n",
            "fadd z", $r1, ".d, p0/m, z", $r1, ".d, z1.d\n",
        ) 
    };


    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1d {{ z1.d }}, p0/z, ", $m0, "\n",
            "fcmla z", $r1, ".d, p0/m, z1.d, z0.d, #0 \n",
            "fcmla z", $r1, ".d, p0/m, z1.d, z0.d, #90 \n",
        ) 
    };
    (M, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1d {{ z1.d }}, p1/z, ", $m0, "\n",
            "fcmla z", $r1, ".d, p0/m, z1.d, z0.d, #0\n",
            "fcmla z", $r1, ".d, p0/m, z1.d, z0.d, #90\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("dup z", r, ".d, #0 \n",)*)
        })
    }
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "fcmla z", $r3, ".d", ", p0/m, z", $r1,".d, z", $r2, ".d, #0 \n",
            "fcmla z", $r3, ".d", ", p0/m, z", $r1,".d, z", $r2, ".d, #90 \n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "ld1d {{ z", $r1, ".d }}, p0/z, ", $m0, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "st1d {{ z", $r1, ".d }}, p0, ", $m0, "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
            "st1d {{ z", $r1, ".d }}, p1, ", $m0, "\n",
        )
    };
}

macro_rules! complex_mul {
    ($r0:tt, $rt:tt) => {
        concat!(
            vzeroall!(4,4),
            "fcmla z4.d, p0/m, z", $r0, ".d, z7.d, #0\n",
            "fcmla z4.d, p0/m, z", $r0, ".d, z7.d, #90\n",
            // copy from z4 to $r0
            "mov z", $r0, ".d, z4.d\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    () => {
        concat!(
            "ld1rqd {{ z7.d }}, p0/z, [{alphax}]", "\n",

            complex_mul!(8, 2),
            complex_mul!(9, 3),
            complex_mul!(10, 4),
            complex_mul!(11, 5),
            complex_mul!(12, 6),
            complex_mul!(13, 2),
            complex_mul!(14, 3),
            complex_mul!(15, 4),
            complex_mul!(16, 5),
            complex_mul!(17, 6),
            complex_mul!(18, 2),
            complex_mul!(19, 3),
            complex_mul!(20, 4),
            complex_mul!(21, 5),
            complex_mul!(22, 6),
            complex_mul!(23, 2),
            complex_mul!(24, 3),
            complex_mul!(25, 4),
            complex_mul!(26, 5),
            complex_mul!(27, 6),
            complex_mul!(28, 2),
            complex_mul!(29, 3),
            complex_mul!(30, 4),
            complex_mul!(31, 5),

            "13:", "\n",

        )
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "ld1rqd {{ z0.d }}, p0/z, [{betax}]", "\n",
            "/* {betax} */", "\n",

            "fcmp d0,#0.0", "\n",
        )
    }
}


macro_rules! acc_p {
    (C, $m0:expr, $q:tt, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $q),
            beta_fmadd!(C, mem!($m0, "1", "MUL VL"), $r2, $q),
            beta_fmadd!(C, mem!($m0, "2", "MUL VL"), $r3, $q),
        )
    };

    (M, $m0:expr, $q:tt, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $q),
            beta_fmadd!(C, mem!($m0, "1", "MUL VL"), $r2, $q),
            "whilelo p1.d, {m_s}, {m_e}", "\n",
            beta_fmadd!(M, mem!($m0, "2", "MUL VL"), $r3, $q),
        )
    };
    (C, $m0:expr, $q:tt, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $q),
            beta_fmadd!(C, mem!($m0, "1", "MUL VL"), $r2, $q),
        )
    };

    (M, $m0:expr, $q:tt, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $q),
            "whilelo p1.d, {m_s}, {m_e}", "\n",
            beta_fmadd!(M, mem!($m0, "1", "MUL VL"), $r2, $q),
        )
    };
    (C, $m0:expr, $q:tt, $r1:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $q),
        )
    };

    (M, $m0:expr, $q:tt, $r1:expr) => {
        concat!(
            "whilelo p1.d, {m_s}, {m_e}", "\n",
            beta_fmadd!(M, mem!($m0), $r1, $q),
        )
    };
}

macro_rules! loadp {
    (3, $m0:expr) => {
        concat!(
            loadp_unit!(mem!($m0), 0),
            loadp_unit!(mem!($m0, "1", "MUL VL"), 1),
            loadp_unit!(mem!($m0, "2", "MUL VL"), 2),
        )
    };
    (2, $m0:expr) => {
        concat!(
            loadp_unit!(mem!($m0), 0),
            loadp_unit!(mem!($m0, "1", "MUL VL"), 1),
        )
    };
    (1, $m0:expr) => {
        concat!(
            loadp_unit!(mem!($m0), 0),
        )
    };
}

macro_rules! storep {
    (C, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            storep_unit!(C, $r1, mem!($m0)),
            storep_unit!(C, $r2, mem!($m0, "1", "MUL VL")),
            storep_unit!(C, $r3, mem!($m0, "2", "MUL VL")),
        )
    };
    (M, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            storep_unit!(C, $r1, mem!($m0)),
            storep_unit!(C, $r2, mem!($m0, "1", "MUL VL")),
            "whilelo p1.d, {m_s}, {m_e}", "\n",
            storep_unit!(M, $r3, mem!($m0, "2", "MUL VL")),
        )
    };
    (C, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            storep_unit!(C, $r1, mem!($m0)),
            storep_unit!(C, $r2, mem!($m0, "1", "MUL VL")),
        )
    };
    (M, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            storep_unit!(C, $r1, mem!($m0)),
            "whilelo p1.d, {m_s}, {m_e}", "\n",
            storep_unit!(M, $r2, mem!($m0, "1", "MUL VL")),
        )
    };
    (C, $m0:expr, $r1:expr) => {
        concat!(
            storep_unit!(C, $r1, mem!($m0)),
        )
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "whilelo p1.d, {m_s}, {m_e}", "\n",
            storep_unit!(M, $r1, mem!($m0)),
        )
    };
}

// only non contigous along m and n direction which is not changed frequently during iteration along k direction
/*

x1 -> cs_a
x2 -> cs_b
x4 -> cx + 3*cs_b

*/


macro_rules! init_ab {
    (B) => {
        concat!(
            "/* {x7} */", "\n",
            "/* {x6} */", "\n",
            "/* {x5} */", "\n",
            "/* {x4} */", "\n",
            "/* {x3} */", "\n",
            "/* {x2} */", "\n",
            "/* {x1} */", "\n",
            // // multiply {m_e} by 2
            "lsl {m_e}, {m_e}, #1", "\n",
            "ldr {x0}, [{dim_arrx}, #24]", "\n",
        )
    };
    (S) => {
        concat!(
            // multiply {m_e} by 2
            "lsl {m_e}, {m_e}, #1", "\n",
            // mov cs_b to reg
            "mov ({dim_arrx}), {x1}", "\n",
            // "mov 8({dim_arrx}), {x2}", "\n",
            "ldr {x0}, [{dim_arrx}, #24]", "\n",
        )
    };
}


macro_rules! c_load {
    () => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
        )
    };
}


macro_rules! vzero_kernel {
    () => {vzeroall!(8,31)};
}

macro_rules! inc_b {
    (S,$nr:tt) => {
        "add {x1},{cx} \n"
    };
    (S,1) => {
        "add {x1},{cx} \n"
    };
    (B,$nr:tt) => {
        ""
    };
}


macro_rules! alpha_scale {
    () => {
        alpha_scale_0!()
    };
}

macro_rules! c_reg_3x8 {
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

macro_rules! acc_3x8 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!(
            $layout, c_mem!($ni), $q, c_reg_3x8!(0,$ni), c_reg_3x8!(1,$ni), c_reg_3x8!(2,$ni)
        )
    };
}

macro_rules! store_3x8 {
    ($ni:tt, $layout:tt) => {
        storep!(
            $layout, c_mem!($ni), c_reg_3x8!(0,$ni), c_reg_3x8!(1,$ni), c_reg_3x8!(2,$ni)
        )
    };
}

macro_rules! acc_2x8 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!(
            $layout, c_mem!($ni), $q, c_reg_3x8!(0,$ni), c_reg_3x8!(1,$ni)
        )
    };
}

macro_rules! store_2x8 {
    ($ni:tt, $layout:tt) => {
        storep!(
            $layout, c_mem!($ni), c_reg_3x8!(0,$ni), c_reg_3x8!(1,$ni)
        )
    };
}

macro_rules! acc_1x8 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!(
            $layout, c_mem!($ni), $q, c_reg_3x8!(0,$ni)
        )
    };
}

macro_rules! store_1x8 {
    ($ni:tt, $layout:tt) => {
        storep!(
            $layout, c_mem!($ni), c_reg_3x8!(0,$ni)
        )
    };
}

macro_rules! load_b {
    (B, 0) => {
        concat!(
            "ld1rqd {{ z3.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*16 \n",
        )
    };
    (B, 1) => {
        concat!(
            "ld1rqd {{ z4.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*16 \n",
        )
    };
    (B, 2) => {
        concat!(
            "ld1rqd {{ z5.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*16 \n",
        )
    };
    (B, 3) => {
        concat!(
            "ld1rqd {{ z6.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*16 \n",
        )
    };
    (B, 4) => {
        concat!(
            "ld1rqd {{ z7.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*16 \n",
        )
    };
    (B, 5) => {
        concat!(
            "ld1rqd {{ z3.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*16 \n",
        )
    };
    (B, 6) => {
        concat!(
            "ld1rqd {{ z4.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*16 \n",
        )
    };
    (B, 7) => {
        concat!(
            "ld1rqd {{ z5.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*16 \n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt) => {
        loadp!($mr, "{ax}")
    };
}

macro_rules! fmadd_1x8 {
    (0) => {
        concat!(
            vfmadd!(0, 3, 8),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 4, 11),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 5, 14),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 6, 17),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 7, 20),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 3, 23),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 4, 26),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 5, 29),
        )
    };
}


macro_rules! step_1x8 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(1),
                "add {ax}, {ax}, {incax} \n",
                #(
                    load_b!($b_layout, n),
                    fmadd_1x8!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}


macro_rules! fmadd_2x8 {
    (0) => {
        concat!(
            vfmadd!(0, 3, 8),
            vfmadd!(1, 3, 9),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 4, 11),
            vfmadd!(1, 4, 12),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 5, 14),
            vfmadd!(1, 5, 15),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 6, 17),
            vfmadd!(1, 6, 18),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 7, 20),
            vfmadd!(1, 7, 21),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 3, 23),
            vfmadd!(1, 3, 24),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 4, 26),
            vfmadd!(1, 4, 27),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 5, 29),
            vfmadd!(1, 5, 30),
        )
    };
}

macro_rules! step_2x8 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2),
                "add {ax}, {ax}, {incax} \n",
                #(
                    load_b!($b_layout, n),
                    fmadd_2x8!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}



macro_rules! fmadd_3x8 {
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



macro_rules! step_3x8 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(3),
                "add {ax}, {ax}, {incax} \n",
                #(
                    load_b!($b_layout, n),
                    fmadd_3x8!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! prefetch_c {
    () => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0}\n ",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "prfm pldl1keep, [{cx}] \n",
            "prfm pldl1keep, [{cx},#64]\n",
            "prfm pldl1keep, [{cx},#128]\n",
            "prfm pldl1keep, [{x1}] \n",
            "prfm pldl1keep, [{x1},#64]\n",
            "prfm pldl1keep, [{x1},#128]\n",
            "prfm pldl1keep, [{x2}] \n",
            "prfm pldl1keep, [{x2},#64]\n",
            "prfm pldl1keep, [{x2},#128]\n",
            "prfm pldl1keep, [{x3}] \n",
            "prfm pldl1keep, [{x3},#64]\n",
            "prfm pldl1keep, [{x3},#128]\n",
        )
    };
}

def_ukernel_sve!(step_1x8, acc_1x8, store_1x8, 1, 8, 8, 9, B, M, ukernel_1_bbp);
def_ukernel_sve!(step_1x8, acc_1x8, store_1x8, 1, 8, 1, 8, B, M, ukernel_1xn_bbp);
def_ukernel_sve!(step_2x8, acc_2x8, store_2x8, 2, 8, 8, 9, B, M, ukernel_2_bbp);
def_ukernel_sve!(step_2x8, acc_2x8, store_2x8, 2, 8, 1, 8, B, M, ukernel_2xn_bbp);

def_ukernel_sve!(step_3x8, acc_3x8, store_3x8, 3, 8, 8, 9, B, M, ukernel_3_bbp);
def_ukernel_sve!(step_3x8, acc_3x8, store_3x8, 3, 8, 1, 8, B, M, ukernel_3xn_bbp);


def_ukernel_sve!(step_3x8, acc_3x8, store_3x8, 3, 8, 1, 8, B, C, ukernel_n_bbc);
def_ukernel_sve!(step_3x8, acc_3x8, store_3x8, 3, 8, 8, 9, B, C, ukernel_bbc);
