use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC, UnaryFnC, TC_SIZE};
use glar_base::{load_buf, store_buf, c_mem, prefetch_0, cum_seq, mem, def_ukernel_sve_i8mm};
use super::super::sve_vs;

const ZERO: TC = 0i32;

const ONE_SCALAR: f32 = 0f32;
const ZERO_SCALAR: f32 = 0f32;
type TS = f32;


macro_rules! unzip_tuple {
    ($r1:tt, $r2:tt,$rt1:tt,$rt2:tt) => {
        concat!(
            "uzp1 z", $rt1, ".d, z", $r1, ".d, z", $r2, ".d\n",
            "uzp2 z", $rt2, ".d, z", $r1, ".d, z", $r2, ".d\n",
            // copy uzp1 to z8 and uzp2 to z11
            "orr z", $r1, ".b, z", $rt1, ".b, z", $rt1, ".b\n",
            "orr z", $r2, ".b, z", $rt2, ".b, z", $rt2, ".b\n",
        )
    };
}

macro_rules! unzip_c {
    () => {
        concat!(
            unzip_tuple!(8, 9, 1, 2),
            unzip_tuple!(10, 11, 3, 4),

            unzip_tuple!(12, 13, 5, 6),
            unzip_tuple!(14, 15, 7, 1),

            unzip_tuple!(16, 17, 2, 3),
            unzip_tuple!(18, 19, 4, 5),

            unzip_tuple!(20, 21, 6, 7),
            unzip_tuple!(22, 23, 1, 2),
            
            unzip_tuple!(24, 25, 3, 4),
            unzip_tuple!(26, 27, 5, 6),
            
            unzip_tuple!(28, 29, 7, 1),
            unzip_tuple!(30, 31, 2, 3),
        )
    }
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "ld1w {{ z1.s }}, p0/z, ", $m0, "\n",
            "add z", $r1, ".s, p0/m, z", $r1, ".s, z1.s\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,1) => {
        concat!(
            "ld1w {{ z1.s }}, p1/z, ", $m0, "\n",
            "add z", $r1, ".s, p1/m, z", $r1, ".s, z1.s\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1w {{ z1.s }}, p0/z, ", $m0, "\n",
            "scvtf z1.s, p0/m, z1.s\n",
            "scvtf z", $r1, ".s, p0/m, z", $r1, ".s\n",
            "fmla z", $r1, ".s, z1.s, z0.s[0]\n",
            "fcvtzs z", $r1, ".s, p0/m, z", $r1, ".s\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1w {{ z1.s }}, p1/z, ", $m0, "\n",
            "scvtf z1.s, p0/m, z1.s\n",
            "scvtf z", $r1, ".s, p0/m, z", $r1, ".s\n",
            "fmla z", $r1, ".s, z1.s, z0.s[0]\n",
            "fcvtzs z", $r1, ".s, p0/m, z", $r1, ".s\n",
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
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "usmmla z", $r3, ".s", ", z", $r2,".b, z", $r1, ".b\n",
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
                    "scvtf z", r, ".s, p0/m, z", r, ".s\n",
                    "fmul  z", r, ".s, z", r, ".s, z1.s[0]\n",
                    "fcvtzs z", r, ".s, p0/m, z", r, ".s\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "ld1rqw {{ z0.s }}, p0/z, [{betax}]", "\n",
            // "/* {betax} */", "\n",
        )
    }
}


macro_rules! acc_p {
    (C, $m0:expr, $r1:expr, $r2:expr, $idx:tt) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $idx),
            beta_fmadd!(C, mem!($m0, "1", "MUL VL"), $r2, $idx),
        )
    };

    (M, $m0:expr, $r1:expr, $r2:expr, $idx:tt) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $idx),
            "whilelo p1.s, {m_s}, {m_e}", "\n",
            beta_fmadd!(M, mem!($m0, "1", "MUL VL"), $r2, $idx),
        )
    };

    (C, $m0:expr, $r1:expr, $idx:tt) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $idx),
        )
    };

    (M, $m0:expr, $r1:expr, $idx:tt) => {
        concat!(
            "whilelo p1.s, {m_s}, {m_e}", "\n",
            beta_fmadd!(M, mem!($m0), $r1, $idx),
        )
    };

}


macro_rules! loadp {
    (2, $m0:expr) => {
        concat!(
            loadp_unit!(mem!($m0), 0),
            loadp_unit!(mem!($m0, "1", "MUL VL"), 1),
            loadp_unit!(mem!($m0, "2", "MUL VL"), 2),
            loadp_unit!(mem!($m0, "3", "MUL VL"), 3),
        )
    };
    (1, $m0:expr) => {
        concat!(
            loadp_unit!(mem!($m0), 0),
            loadp_unit!(mem!($m0, "1", "MUL VL"), 1),
        )
    };
}

macro_rules! storep {
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
            "/* {x11} */", "\n",
            "/* {x10} */", "\n",
            "/* {x9} */", "\n",
            "/* {x8} */", "\n",
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
            unzip_c!(),
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
            "add {x8}, {x7}, {x0} \n",
            "add {x9}, {x8}, {x0} \n",
            "add {x10}, {x9}, {x0} \n",
            "add {x11}, {x10}, {x0} \n",
        )
    };
    (11) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
            "add {x8}, {x7}, {x0} \n",
            "add {x9}, {x8}, {x0} \n",
            "add {x10}, {x9}, {x0} \n",
        )
    };
    (10) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
            "add {x8}, {x7}, {x0} \n",
            "add {x9}, {x8}, {x0} \n",
        )
    };
    (9) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
            "add {x8}, {x7}, {x0} \n",
        )
    };
    (8) => {
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
    (7) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
        )
    };
    (6) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
        )
    };
    (5) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
        )
    };
    (4) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
        )
    };
    (3) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
        )
    };
    (2) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
        )
    };
    (1) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
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

macro_rules! c_reg_2x12 {
    (0,0) => { 8 };
    (0,1) => { 9 };

    (1,0) => { 10 };
    (1,1) => { 11 };

    (0,2) => { 12 };
    (0,3) => { 13 };

    (1,2) => { 14 };
    (1,3) => { 15 };

    (0,4) => { 16 };
    (0,5) => { 17 };

    (1,4) => { 18 };
    (1,5) => { 19 };

    (0,6) => { 20 };
    (0,7) => { 21 };

    (1,6) => { 22 };
    (1,7) => { 23 };

    (0,8) => { 24 };
    (0,9) => { 25 };

    (1,8) => { 26 };
    (1,9) => { 27 };

    (0,10) => { 28 };
    (0,11) => { 29 };

    (1,10) => { 30 };
    (1,11) => { 31 };
}


macro_rules! acc_2x12 {
    ($ni:tt, $layout:tt, $idx:tt) => {
        acc_p!(
            $layout, c_mem!($ni), c_reg_2x12!(0,$ni), c_reg_2x12!(1,$ni), $idx
        )
    };
}

macro_rules! store_2x12 {
    ($ni:tt, $layout:tt) => {
        storep!(
            $layout, c_mem!($ni), c_reg_2x12!(0,$ni), c_reg_2x12!(1,$ni)
        )
    };
}

macro_rules! acc_1x12 {
    ($ni:tt, $layout:tt, $idx:tt) => {
        acc_p!(
            $layout, c_mem!($ni), c_reg_2x12!(0,$ni), $idx
        )
    };
}

macro_rules! store_1x12 {
    ($ni:tt, $layout:tt) => {
        storep!(
            $layout, c_mem!($ni), c_reg_2x12!(0,$ni)
        )
    };
}

macro_rules! load_b {
    (B, 0) => {
        concat!(
            "ld1rqd {{ z4.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 2) => {
        concat!(
            "ld1rqd {{ z5.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 4) => {
        concat!(
            "ld1rqd {{ z6.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 6) => {
        concat!(
            "ld1rqd {{ z7.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 8) => {
        concat!(
            "ld1rqd {{ z4.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 10) => {
        concat!(
            "ld1rqd {{ z5.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };

    (B, $nr:tt) => {
        "add {bx}, {bx}, #8 \n"
    };
}


macro_rules! load_a {
    ($mr:tt) => {
        loadp!($mr, "{ax}")
    };
}

macro_rules! fmadd_1x12 {
    (0) => {
        concat!(
            vfmadd!(0, 4, 8),
            vfmadd!(1, 4, 9),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 5, 12),
            vfmadd!(1, 5, 13),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 6, 16),
            vfmadd!(1, 6, 17),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 7, 20),
            vfmadd!(1, 7, 21),
        )
    };
    (8) => {
        concat!(
            vfmadd!(0, 4, 24),
            vfmadd!(1, 4, 25),
        )
    };
    (10) => {
        concat!(
            vfmadd!(0, 5, 28),
            vfmadd!(1, 5, 29),
        )
    };
    ($nr:tt) => {""};
}

macro_rules! step_1x12 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(1),
                "add {ax}, {ax}, {incax} \n",
                #(
                    load_b!($b_layout, n),
                    fmadd_1x12!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! fmadd_2x12 {
    (0) => {
        concat!(
            vfmadd!(0, 4, 8),
            vfmadd!(1, 4, 9),
            vfmadd!(2, 4, 10),
            vfmadd!(3, 4, 11),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 5, 12),
            vfmadd!(1, 5, 13),
            vfmadd!(2, 5, 14),
            vfmadd!(3, 5, 15),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 6, 16),
            vfmadd!(1, 6, 17),
            vfmadd!(2, 6, 18),
            vfmadd!(3, 6, 19),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 7, 20),
            vfmadd!(1, 7, 21),
            vfmadd!(2, 7, 22),
            vfmadd!(3, 7, 23),
        )
    };
    (8) => {
        concat!(
            vfmadd!(0, 4, 24),
            vfmadd!(1, 4, 25),
            vfmadd!(2, 4, 26),
            vfmadd!(3, 4, 27),
        )
    };
    (10) => {
        concat!(
            vfmadd!(0, 5, 28),
            vfmadd!(1, 5, 29),
            vfmadd!(2, 5, 30),
            vfmadd!(3, 5, 31),
        )
    };
    ($nr:tt) => {""};
}

macro_rules! step_2x12 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2),
                "add {ax}, {ax}, {incax} \n",
                #(
                    load_b!($b_layout, n),
                    fmadd_2x12!(n),
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
            "add {x8}, {x7}, {x0} \n",
            "add {x9}, {x8}, {x0} \n",
            "add {x10}, {x9}, {x0} \n",
            "add {x11}, {x10}, {x0} \n",
            "prfm pldl1keep, [{cx}] \n",
            "prfm pldl1keep, [{cx},#64]\n",
            "prfm pldl1keep, [{x1}] \n",
            "prfm pldl1keep, [{x1},#64]\n",
            "prfm pldl1keep, [{x2}] \n",
            "prfm pldl1keep, [{x2},#64]\n",
            "prfm pldl1keep, [{x3}] \n",
            "prfm pldl1keep, [{x3},#64]\n",
            "prfm pldl1keep, [{x4}] \n",
            "prfm pldl1keep, [{x4},#64]\n",
            "prfm pldl1keep, [{x5}] \n",
            "prfm pldl1keep, [{x5},#64]\n",
            "prfm pldl1keep, [{x6}] \n",
            "prfm pldl1keep, [{x6},#64]\n",
            "prfm pldl1keep, [{x7}] \n",
            "prfm pldl1keep, [{x7},#64]\n",
            "prfm pldl1keep, [{x8}] \n",
            "prfm pldl1keep, [{x8},#64]\n",
            "prfm pldl1keep, [{x9}] \n",
            "prfm pldl1keep, [{x9},#64]\n",
            "prfm pldl1keep, [{x10}] \n",
            "prfm pldl1keep, [{x10},#64]\n",
            "prfm pldl1keep, [{x11}] \n",
            "prfm pldl1keep, [{x11},#64]\n",
        )
    };
}

const MAX_VS: usize = 64;

def_ukernel_sve_i8mm!(step_1x12, acc_1x12, store_1x12, 1, 12, 12, 13, B, M, ukernel_1_bbp);
def_ukernel_sve_i8mm!(step_1x12, acc_1x12, store_1x12, 1, 12, 1, 12, B, M, ukernel_1xn_bbp);
def_ukernel_sve_i8mm!(step_2x12, acc_2x12, store_2x12, 2, 12, 12, 13, B, M, ukernel_2_bbp);
def_ukernel_sve_i8mm!(step_2x12, acc_2x12, store_2x12, 2, 12, 1, 12, B, M, ukernel_2xn_bbp);


def_ukernel_sve_i8mm!(step_2x12, acc_2x12, store_2x12, 2, 12, 1, 12, B, C, ukernel_n_bbc);


#[target_feature(enable="neon,sve,i8mm")]
pub(crate) unsafe fn ukernel_bbc<F: UnaryFnC, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const f32, beta: *const f32,
    k: usize,
    d_arr: [usize; 3], c_cs: usize,
    m: usize, _n: usize,
    f: F,
) {
    let vs = sve_vs();
    let inc_a = 2 * vs * 8;
    let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, c_cs*TC_SIZE, k / 32, (k % 32) / 8];
    let mut cf = c;
    let mr = vs * 2;
    let mut c_buf = [0i32; MAX_VS * 2 * 6];
    let alpha_st = if *alpha == 1f32 {
        0i32
    } else {
        1i32
    };
    let beta_st = if *beta == 0f32 {
        0i32
    } else if *beta == 1f32 {
        1i32
    } else {
        2i32
    };
    if BUF {
        load_buf(c, d_arr[2], c_cs, &mut c_buf, m, 12, mr);
        dim_arr[2] = mr*TC_SIZE;
        cf = c_buf.as_mut_ptr();
    }
    asm!(
        "ptrue p0.s",
        "ptrue p1.s",
        vzero_kernel!(),

        prefetch_c!(),

        init_ab!(B),
        
        // 3 -> CONSIDKLEFT
        "cmp {x0}, #0", "BEQ 3f",
        
        // 2 -> KITER
        "2:",
        prefetch_0!(256, "{bx}"),
        step_2x12!(12, B),
        step_2x12!(12, B),
        prefetch_0!(256, "{bx}"),
        step_2x12!(12, B),
        step_2x12!(12, B),

        "sub {x0}, {x0}, #1",
        // 2 -> KITER
        "cmp {x0}, 0",
        "BNE 2b",

        // 3 -> CONSIDKLEFT
        "3:",

        "ldr {x0}, [{dim_arrx}, #32]",
        "cmp {x0}, #0",

        // 5 -> POSTACCUM
        "BEQ 5f",
        // 4 -> KLEFT
        "4:",
        step_2x12!(12, B),

        "sub {x0}, {x0}, #1",

        // 4 -> KLEFT
        "cmp {x0}, 0",
        "BNE 4b",

        // 5 -> POSTACCUM
        "5:",
        c_load!(),
        "cmp {alpha_st:w}, #0",
        "BEQ 13f",
        alpha_scale!(),
        "13:",

        "cmp {beta_st:w}, #0",
        "BEQ 6f",

        "cmp {beta_st:w}, #1",
        "BEQ 9f",

        // 6 -> BETAZERO
        load_beta!(),
        cum_seq!(acc_2x12,12,C,2),
        "B 6f",

        "9:",
        // 9 -> BETAONE
        cum_seq!(acc_2x12,12,C,1),

        // 6 -> BETAZERO
        "6:",
        cum_seq!(store_2x12,12,C),
        
        // 7 -> DDONE
        "7:",
        ax = inout(reg) a => _,
        bx = inout(reg) b => _,
        cx = inout(reg) cf => _,
        dim_arrx = inout(reg) dim_arr.as_ptr() => _,
        alphax = inout(reg) alpha => _,
        betax = inout(reg) beta => _,
        incax = in(reg) inc_a as u64,
        alpha_st = in(reg) alpha_st,
        beta_st = in(reg) beta_st,
        x0 = out(reg) _,
        x1 = out(reg) _,
        x2 = out(reg) _,
        x3 = out(reg) _,
        x4 = out(reg) _,
        x5 = out(reg) _,
        x6 = out(reg) _,
        x7 = out(reg) _,
        x8 = out(reg) _,
        x9 = out(reg) _,
        x10 = out(reg) _,
        x11 = out(reg) _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
    );
    if BUF {
        for j in 0..12 {
            f.call(cf.add(j*mr), mr);
        }
        store_buf(c, d_arr[2], c_cs, &c_buf, m, 12, mr);
    } else {
        for j in 0..12 {
            f.call(cf.add(j*c_cs), m);
        }
    }
}