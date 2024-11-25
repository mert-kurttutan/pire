use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{c_mem, prefetch_0, def_ukernel_sve,mem};
use super::super::sve_vs;

const ONE_SCALAR: TC = 1.0;
const ZERO_SCALAR: TC = 0.0;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1w {{ z1.s }}, p0/z, ", $m0, "\n",
            "fmla z", $r1, ".s, z1.s, z0.s[0]\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1w {{ z1.s }}, p1/z, ", $m0, "\n",
            "fmla z", $r1, ".s, z1.s, z0.s[0]\n",
        ) 
    };
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "ld1w {{ z1.s }}, p0/z, ", $m0, "\n",
            "fadd z", $r1, ".s, z", $r1, ".s, z1.s\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,1) => {
        concat!(
            "ld1w {{ z1.s }}, p1/z, ", $m0, "\n",
            "fadd z", $r1, ".s, z", $r1, ".s, z1.s\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("dup z", r, ".s, #0 \n",)*)
        })
    }
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr, $i:expr) => {
        concat!(
            "fmla z", $r3, ".s", ", z", $r1,".s, z", $r2, ".s[", $i, "]\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "ld1w {{ z", $r1, ".s }}, p0/z, ", $m0, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "st1w {{ z", $r1, ".s }}, p0, ", $m0, "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
            "st1w {{ z", $r1, ".s }}, p1, ", $m0, "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "ld1rqw {{ z1.s }}, p0/z, [{alphax}]", "\n",
                #(
                    "fmul  z", r, ".s, z", r, ".s, z1.s[0]\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "ld1rqw {{ z0.s }}, p0/z, [{betax}]", "\n",
            "/* {betax} */", "\n",

            "fcmp s0,#0.0", "\n",
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
            "whilelo p1.s, {m_s}, {m_e}", "\n",
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
            "whilelo p1.s, {m_s}, {m_e}", "\n",
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
            "whilelo p1.s, {m_s}, {m_e}", "\n",
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
            "whilelo p1.s, {m_s}, {m_e}", "\n",
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
            "whilelo p1.s, {m_s}, {m_e}", "\n",
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
            "whilelo p1.s, {m_s}, {m_e}", "\n",
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
            "ld1rqw {{ z3.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*4 \n",
        )
    };
    (B, 2) => {
        concat!(
            "ld1rqw {{ z3.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*4 \n",
        )
    };
    (B, 3) => {
        concat!(
            "ld1rqw {{ z3.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #3*4 \n",
        )
    };
    (B, 4) => {
        concat!(
            "ld1rqw {{ z3.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
        )
    };
    (B, 5) => {
        concat!(
            "ld1rqw {{ z3.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
            "ld1rqw {{ z4.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*4 \n",
        )
    };
    (B, 6) => {
        concat!(
            "ld1rqw {{ z3.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
            "ld1rqw {{ z4.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*4 \n",
        )
    };
    (B, 7) => {
        concat!(
            "ld1rqw {{ z3.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
            "ld1rqw {{ z4.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #3*4 \n",
        )
    };
    (B, 8) => {
        concat!(
            "ld1rqw {{ z3.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
            "ld1rqw {{ z4.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
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
            vfmadd!(0, 4, 20, 0),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 4, 23, 1),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 4, 26, 2),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 4, 29, 3),
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
            vfmadd!(0, 4, 20, 0),
            vfmadd!(1, 4, 21, 0),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 4, 23, 1),
            vfmadd!(1, 4, 24, 1),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 4, 26, 2),
            vfmadd!(1, 4, 27, 2),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 4, 29, 3),
            vfmadd!(1, 4, 30, 3),
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
            vfmadd!(0, 4, 20, 0),
            vfmadd!(1, 4, 21, 0),
            vfmadd!(2, 4, 22, 0),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 4, 23, 1),
            vfmadd!(1, 4, 24, 1),
            vfmadd!(2, 4, 25, 1),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 4, 26, 2),
            vfmadd!(1, 4, 27, 2),
            vfmadd!(2, 4, 28, 2),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 4, 29, 3),
            vfmadd!(1, 4, 30, 3),
            vfmadd!(2, 4, 31, 3),
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
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
            "prfm pldl1keep, [{cx}] \n",
            "prfm pldl1keep, [{cx},#64]\n",
            "prfm pldl1keep, [{cx},#96]\n",
            "prfm pldl1keep, [{x1}] \n",
            "prfm pldl1keep, [{x1},#64]\n",
            "prfm pldl1keep, [{x1},#96]\n",
            "prfm pldl1keep, [{x2}] \n",
            "prfm pldl1keep, [{x2},#64]\n",
            "prfm pldl1keep, [{x2},#96]\n",
            "prfm pldl1keep, [{x3}] \n",
            "prfm pldl1keep, [{x3},#64]\n",
            "prfm pldl1keep, [{x3},#96]\n",
            "prfm pldl1keep, [{x4}] \n",
            "prfm pldl1keep, [{x4},#64]\n",
            "prfm pldl1keep, [{x4},#96]\n",
            "prfm pldl1keep, [{x5}] \n",
            "prfm pldl1keep, [{x5},#64]\n",
            "prfm pldl1keep, [{x5},#96]\n",
            "prfm pldl1keep, [{x6}] \n",
            "prfm pldl1keep, [{x6},#64]\n",
            "prfm pldl1keep, [{x6},#96]\n",
            "prfm pldl1keep, [{x7}] \n",
            "prfm pldl1keep, [{x7},#64]\n",
            "prfm pldl1keep, [{x7},#96]\n",
        )
    };
}

def_ukernel_sve!(step_1x8, acc_1x8, store_1x8, 1, 8, B, M, ukernel_1_bbp);
def_ukernel_sve!(step_2x8, acc_2x8, store_2x8, 2, 8, B, M, ukernel_2_bbp);
def_ukernel_sve!(step_3x8, acc_3x8, store_3x8, 3, 8, B, M, ukernel_3_bbp);



def_ukernel_sve!(step_3x8, acc_3x8, store_3x8, 3, 8, B, C, ukernel_bbc);
