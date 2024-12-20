use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_sve,
    acc_3, acc_2, acc_1,
    store_3, store_2, store_1,
    step_3, step_2, step_1,
};
use super::super::sve_vs;

const ONE_SCALAR: TC = 1.0;
const ZERO_SCALAR: TC = 0.0;

macro_rules! cr {
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

macro_rules! br_3 {
    (0) => { "z3.s[0]" };
    (1) => { "z3.s[1]" };
    (2) => { "z3.s[2]" };
    (3) => { "z3.s[3]" };
    (4) => { "z4.s[0]" };
    (5) => { "z4.s[1]" };
    (6) => { "z4.s[2]" };
    (7) => { "z4.s[3]" };
}

macro_rules! br_2 {
    (0) => { "z3.s[0]" };
    (1) => { "z3.s[1]" };
    (2) => { "z3.s[2]" };
    (3) => { "z3.s[3]" };
    (4) => { "z4.s[0]" };
    (5) => { "z4.s[1]" };
    (6) => { "z4.s[2]" };
    (7) => { "z4.s[3]" };
}

macro_rules! br_1 {
    (0) => { "z3.s[0]" };
    (1) => { "z3.s[1]" };
    (2) => { "z3.s[2]" };
    (3) => { "z3.s[3]" };
    (4) => { "z4.s[0]" };
    (5) => { "z4.s[1]" };
    (6) => { "z4.s[2]" };
    (7) => { "z4.s[3]" };
}

macro_rules! v_i {
    ($m0:tt, $ni:tt) => {
        concat!("[", $m0, ", #", $ni, ", MUL VL]")
    }
}

macro_rules! set_predicate {
    (M) => { "mov {m_s}, #0 \n whilelo p1.s, {m_s}, {m_e} \n" };
    (C) => { "/* {m_s}, {m_e} */" }
}

macro_rules! inc_a {
    ($mr:tt) => {
        concat!("add {ax}, {ax}, {incax} \n")
    };
}

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
    ($i:tt, $j:tt, $b_macro:tt) => {
        concat!(
            "fmla z", cr!($i,$j), ".s", ", z", $i,".s, ", $b_macro!($j), "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "ld1w {{ z", $r1, ".s }}, p0/z, [", $m0, ", #", $r1, ", MUL VL]\n",
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

macro_rules! load_b {
    (B, 0, $b_macro:tt) => {
        concat!(
            "ld1rqw {{ z3.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*4 \n",
        )
    };
    // (B, 2) => {
    //     concat!(
    //         "ld1rqw {{ z3.s }}, p0/z, [{bx}]", "\n",
    //         "add {bx}, {bx}, #2*4 \n",
    //     )
    // };
    // (B, 3) => {
    //     concat!(
    //         "ld1rqw {{ z3.s }}, p0/z, [{bx}]", "\n",
    //         "add {bx}, {bx}, #3*4 \n",
    //     )
    // };
    (B, 4, $b_macro:tt) => {
        concat!(
            "ld1rqw {{ z3.s }}, p0/z, [{bx}, #0x10]", "\n",
            // "add {bx}, {bx}, #4*4 \n",
        )
    };
    (B, $ni:tt, $b_macro:tt) => {
        ""
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

def_ukernel_sve!(step_1, acc_1, store_1, 1, 8, B, M, ukernel_1_bbp);
def_ukernel_sve!(step_2, acc_2, store_2, 2, 8, B, M, ukernel_2_bbp);
def_ukernel_sve!(step_3, acc_3, store_3, 3, 8, B, M, ukernel_3_bbp);



def_ukernel_sve!(step_3, acc_3, store_3, 3, 8, B, C, ukernel_bbc);
