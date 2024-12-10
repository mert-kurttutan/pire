use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    prefetch_0, def_ukernel_sve,mem,
    acc_3, acc_2, acc_1,
    store_3, store_2, store_1,
};
use super::super::sve_vs;

const ONE_SCALAR: TC = 1.0;
const ZERO_SCALAR: TC = 0.0;

macro_rules! v_i {
    ($m0:tt, $ni:tt) => {
        concat!("[", $m0, ", #", $ni, ", MUL VL]")
    }
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1d {{ z1.d }}, p0/z, ", $m0, "\n",
            "fmla z", $r1, ".d, z1.d, z0.d[0]\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1d {{ z1.d }}, p1/z, ", $m0, "\n",
            "fmla z", $r1, ".d, z1.d, z0.d[0]\n",
        ) 
    };
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "ld1d {{ z1.d }}, p0/z, ", $m0, "\n",
            "fadd z", $r1, ".d, z", $r1, ".d, z1.d\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,1) => {
        concat!(
            "ld1d {{ z1.d }}, p1/z, ", $m0, "\n",
            "fadd z", $r1, ".d, z", $r1, ".d, z1.d\n",
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
    ($r1:expr, $r2:expr, $r3:expr, $i:expr) => {
        concat!(
            "fmla z", $r3, ".d", ", z", $r1,".d, z", $r2, ".d[", $i, "]\n",
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

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "ld1rqd {{ z1.d }}, p0/z, [{alphax}]", "\n",
                #(
                    "fmul  z", r, ".d, z", r, ".d, z1.d[0]\n",
                )*
            )
        })
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
    (B,$nr:tt) => {
        ""
    };
}

macro_rules! alpha_scale {
    () => {
        alpha_scale_0!(8,31)
    };
}
macro_rules! cr_3 {
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
macro_rules! cr_2 {
    (0,0) => { 8 };
    (1,0) => { 9 };

    (0,1) => { 11 };
    (1,1) => { 12 };

    (0,2) => { 14 };
    (1,2) => { 15 };

    (0,3) => { 17 };
    (1,3) => { 18 };

    (0,4) => { 20 };
    (1,4) => { 21 };

    (0,5) => { 23 };
    (1,5) => { 24 };

    (0,6) => { 26 };
    (1,6) => { 27 };

    (0,7) => { 29 };
    (1,7) => { 30 };
}
macro_rules! cr_1 {
    (0,0) => { 8 };

    (0,1) => { 11 };

    (0,2) => { 14 };

    (0,3) => { 17 };

    (0,4) => { 20 };

    (0,5) => { 23 };

    (0,6) => { 26 };

    (0,7) => { 29 };
}

macro_rules! load_b {
    (B, 1) => {
        concat!(
            "ld1rqd {{ z3.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*8 \n",
        )
    };
    (B, 2) => {
        concat!(
            "ld1rqd {{ z3.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
        )
    };
    (B, 3) => {
        concat!(
            "ld1rqd {{ z3.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z4.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*8 \n",
        )
    };
    (B, 4) => {
        concat!(
            "ld1rqd {{ z3.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z4.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
        )
    };
    (B, 5) => {
        concat!(
            "ld1rqd {{ z3.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z4.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z5.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*8 \n",
        )
    };
    (B, 6) => {
        concat!(
            "ld1rqd {{ z3.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z4.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z5.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
        )
    };
    (B, 7) => {
        concat!(
            "ld1rqd {{ z3.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z4.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z5.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z6.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*8 \n",
        )
    };
    (B, 8) => {
        concat!(
            "ld1rqd {{ z3.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z4.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z5.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
            "ld1rqd {{ z6.d }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*8 \n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt) => {
        loadp!($mr, "{ax}")
    };
}

macro_rules! fmadd_1 {
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
            vfmadd!(0, 4, 14, 0),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 4, 17, 1),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 5, 20, 0),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 5, 23, 1),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 6, 26, 0),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 6, 29, 1),
        )
    };
}

macro_rules! step_1 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(1),
                "add {ax}, {ax}, {incax} \n",
                load_b!($b_layout, $nr),
                #(
                    fmadd_1!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! fmadd_2 {
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
            vfmadd!(0, 4, 14, 0),
            vfmadd!(1, 4, 15, 0),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 4, 17, 1),
            vfmadd!(1, 4, 18, 1),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 5, 20, 0),
            vfmadd!(1, 5, 21, 0),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 5, 23, 1),
            vfmadd!(1, 5, 24, 1),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 6, 26, 0),
            vfmadd!(1, 6, 27, 0),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 6, 29, 1),
            vfmadd!(1, 6, 30, 1),
        )
    };
}

macro_rules! step_2 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2),
                "add {ax}, {ax}, {incax} \n",
                load_b!($b_layout, $nr),
                #(
                    fmadd_2!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}



macro_rules! fmadd_3 {
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
            vfmadd!(0, 4, 14, 0),
            vfmadd!(1, 4, 15, 0),
            vfmadd!(2, 4, 16, 0),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 4, 17, 1),
            vfmadd!(1, 4, 18, 1),
            vfmadd!(2, 4, 19, 1),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 5, 20, 0),
            vfmadd!(1, 5, 21, 0),
            vfmadd!(2, 5, 22, 0),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 5, 23, 1),
            vfmadd!(1, 5, 24, 1),
            vfmadd!(2, 5, 25, 1),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 6, 26, 0),
            vfmadd!(1, 6, 27, 0),
            vfmadd!(2, 6, 28, 0),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 6, 29, 1),
            vfmadd!(1, 6, 30, 1),
            vfmadd!(2, 6, 31, 1),
        )
    };
}


macro_rules! step_3 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(3),
                "add {ax}, {ax}, {incax} \n",
                load_b!($b_layout, $nr),
                #(
                    fmadd_3!(n),
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

def_ukernel_sve!(step_1, acc_1, store_1, 1, 8, B, M, ukernel_1_bbp);
def_ukernel_sve!(step_2, acc_2, store_2, 2, 8, B, M, ukernel_2_bbp);
def_ukernel_sve!(step_3, acc_3, store_3, 3, 8, B, M, ukernel_3_bbp);

def_ukernel_sve!(step_3, acc_3, store_3, 3, 8, B, C, ukernel_bbc);
