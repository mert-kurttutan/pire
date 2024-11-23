use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC, UnaryFnC, TC_SIZE};
use pire_base::{c_mem, prefetch_0, mem, def_ukernel_sve};
use super::super::sve_vs;

const ZERO: TC = TC::ZERO;

const ONE_SCALAR: TC = TC::ONE;
const ZERO_SCALAR: TC = TC::ZERO;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1h {{ z1.h }}, p0/z, ", $m0, "\n",
            "fmla z", $r1, ".h, z1.h, z0.h[0]\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1h {{ z1.h }}, p1/z, ", $m0, "\n",
            "fmla z", $r1, ".h, z1.h, z0.h[0]\n",
        ) 
    };
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "ld1h {{ z1.h }}, p0/z, ", $m0, "\n",
            "fadd z", $r1, ".h, z", $r1, ".h, z1.h\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,1) => {
        concat!(
            "ld1h {{ z1.h }}, p1/z, ", $m0, "\n",
            "fadd z", $r1, ".h, z", $r1, ".h, z1.h\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("dup z", r, ".h, #0 \n",)*)
        })
    }
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr, $i:expr) => {
        concat!(
            "fmla z", $r3, ".h", ", z", $r1,".h, z", $r2, ".h[", $i, "]\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "ld1h {{ z", $r1, ".h }}, p0/z, ", $m0, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "st1h {{ z", $r1, ".h }}, p0, ", $m0, "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
            "st1h {{ z", $r1, ".h }}, p1, ", $m0, "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "ld1rqh {{ z1.h }}, p0/z, [{alphax}]", "\n",
                #(
                    "fmul  z", r, ".h, z", r, ".h, z1.h[0]\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "ld1rqh {{ z0.h }}, p0/z, [{betax}]", "\n",
            "/* {betax} */", "\n",

            "fcmp h0,#0.0", "\n",
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
            "whilelo p1.h, {m_s}, {m_e}", "\n",
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
            "whilelo p1.h, {m_s}, {m_e}", "\n",
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
            "whilelo p1.h, {m_s}, {m_e}", "\n",
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
            "whilelo p1.h, {m_s}, {m_e}", "\n",
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
            "whilelo p1.h, {m_s}, {m_e}", "\n",
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
            "whilelo p1.h, {m_s}, {m_e}", "\n",
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
            "ldr {x0}, [{dim_arrx}, #24]", "\n",
        )
    };
    (S) => {
        concat!(
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
        alpha_scale_0!(8,31)
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
    (B, 1) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*2 \n",
        )
    };
    (B, 2) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*2 \n",
        )
    };
    (B, 3) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #3*2 \n",
        )
    };
    (B, 4) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*2 \n",
        )
    };
    (B, 5) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #5*2 \n",
        )
    };
    (B, 6) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #6*2 \n",
        )
    };
    (B, 7) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #7*2 \n",
        )
    };
    (B, 8) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #8*2 \n",
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
            vfmadd!(0, 3, 8, 0),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 3, 11, 1),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 3, 14, 2),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 3, 17, 3),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 3, 20, 4),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 3, 23, 5),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 3, 26, 6),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 3, 29, 7),
        )
    };
}

macro_rules! step_1x8 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(1),
                "add {ax}, {ax}, {incax} \n",
                load_b!($b_layout, $nr),
                #(
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
            vfmadd!(0, 3, 8, 0),
            vfmadd!(1, 3, 9, 0),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 3, 11, 1),
            vfmadd!(1, 3, 12, 1),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 3, 14, 2),
            vfmadd!(1, 3, 15, 2),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 3, 17, 3),
            vfmadd!(1, 3, 18, 3),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 3, 20, 4),
            vfmadd!(1, 3, 21, 4),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 3, 23, 5),
            vfmadd!(1, 3, 24, 5),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 3, 26, 6),
            vfmadd!(1, 3, 27, 6),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 3, 29, 7),
            vfmadd!(1, 3, 30, 7),
        )
    };
}

macro_rules! step_2x8 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2),
                "add {ax}, {ax}, {incax} \n",
                load_b!($b_layout, $nr),
                #(
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
            vfmadd!(0, 3, 8, 0),
            vfmadd!(1, 3, 9, 0),
            vfmadd!(2, 3, 10, 0),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 3, 11, 1),
            vfmadd!(1, 3, 12, 1),
            vfmadd!(2, 3, 13, 1),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 3, 14, 2),
            vfmadd!(1, 3, 15, 2),
            vfmadd!(2, 3, 16, 2),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 3, 17, 3),
            vfmadd!(1, 3, 18, 3),
            vfmadd!(2, 3, 19, 3),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 3, 20, 4),
            vfmadd!(1, 3, 21, 4),
            vfmadd!(2, 3, 22, 4),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 3, 23, 5),
            vfmadd!(1, 3, 24, 5),
            vfmadd!(2, 3, 25, 5),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 3, 26, 6),
            vfmadd!(1, 3, 27, 6),
            vfmadd!(2, 3, 28, 6),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 3, 29, 7),
            vfmadd!(1, 3, 30, 7),
            vfmadd!(2, 3, 31, 7),
        )
    };
}




macro_rules! step_3x8 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(3),
                "add {ax}, {ax}, {incax} \n",
                load_b!($b_layout, $nr),
                #(
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
