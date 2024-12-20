use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_neon_alt,
    acc_3, acc_2, acc_1,
    store_3, store_2, store_1,
    step_3_c, step_2_c, step_1_c,
};

const ONE_SCALAR: TC = TC::ONE;
const ZERO_SCALAR: TC = TC::ZERO;

macro_rules! br_3 {
    (0,0) => { "v3.s[0]" };
    (1,0) => { "v3.s[2]" };
    (2,0) => { "v4.s[0]" };
    (3,0) => { "v4.s[2]" };

    (0,1) => { "v3.s[1]" };
    (1,1) => { "v3.s[3]" };
    (2,1) => { "v4.s[1]" };
    (3,1) => { "v4.s[3]" };
}

macro_rules! br_2 {
    (0,0) => { "v3.s[0]" };
    (1,0) => { "v3.s[2]" };
    (2,0) => { "v4.s[0]" };
    (3,0) => { "v4.s[2]" };

    (0,1) => { "v3.s[1]" };
    (1,1) => { "v3.s[3]" };
    (2,1) => { "v4.s[1]" };
    (3,1) => { "v4.s[3]" };
}
macro_rules! br_1 {
    (0,0) => { "v3.s[0]" };
    (1,0) => { "v3.s[2]" };
    (2,0) => { "v4.s[0]" };
    (3,0) => { "v4.s[2]" };

    (0,1) => { "v3.s[1]" };
    (1,1) => { "v3.s[3]" };
    (2,1) => { "v4.s[1]" };
    (3,1) => { "v4.s[3]" };
}

macro_rules! cr {
    (0,0) => { 8 };
    (1,0) => { 10 };
    (2,0) => { 12 };

    (0,1) => { 14 };
    (1,1) => { 16 };
    (2,1) => { 18 };

    (0,2) => { 20 };
    (1,2) => { 22 };
    (2,2) => { 24 };
    
    (0,3) => { 26 };
    (1,3) => { 28 };
    (2,3) => { 30 };

    (0,0,1) => { 9 };
    (1,0,1) => { 11 };
    (2,0,1) => { 13 };

    (0,1,1) => { 15 };
    (1,1,1) => { 17 };
    (2,1,1) => { 19 };

    (0,2,1) => { 21 };
    (1,2,1) => { 23 };
    (2,2,1) => { 25 };
    
    (0,3,1) => { 27 };
    (1,3,1) => { 29 };
    (2,3,1) => { 31 };
}

macro_rules! vzero_kernel {
    () => {
        seq!(r in 8..=31 {
            concat!(#("dup v", r, ".4s, wzr \n",)*)
        })
    }
}

macro_rules! alt_arr {
    ($vec_name:ident) => {
        let $vec_name = [-1.0f32, 1.0f32, -1.0f32, 1.0f32];
    }
}
macro_rules! v_i {
    ($m0:tt, $ni:tt) => {
        concat!("[", $m0, ", #", $ni, "*0x10]")
    }
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "fadd  v", $r1, ".4s, v", $r1, ".4s, v1.4s\n",
        ) 
    };
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "fmul v2.4s, v1.4s, v0.s[0]\n",
            "fmul v3.4s, v1.4s, v0.s[1]\n",
            v_to_c!(2, 3),
            "fadd  v", $r1, ".4s, v", $r1, ".4s, v2.4s\n",
        ) 
    };
}

macro_rules! vfmadd {
    ($i:tt, $j:tt, $b_macro:tt, $il:tt) => {
        concat!(
            "fmla v", cr!($i,$j), ".4s", ", v", $i,".4s, ", $b_macro!($j,$il), "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "ldr q", $r1, ", [", $m0, ", #", $r1, "*0x10] \n",
        )
    };
}

macro_rules! storep_unit {
    ($layout:tt, $r1:expr, $m0:expr) => {
        concat!(
            "str q", $r1, ", ", $m0,  "\n",
        )
    };
}

macro_rules! complex_mul {
    ($r0:tt, $rt:tt) => {
        concat!(
            "rev64 v", $rt, ".4s, v", $r0, ".4s\n",

            "fmul v", $r0, ".4s, v", $r0, ".4s, v1.s[0]\n",
            "fmul v", $rt, ".4s, v", $rt, ".4s, v1.s[1]\n",

            "fmla v", $r0, ".4s, v", $rt, ".4s, v7.4s\n",
        )
    };
}

macro_rules! alpha_scale {
    () => {
        concat!(
            "ldr d1, [{alphax}]", "\n",
            complex_mul!(8, 9),
            complex_mul!(10, 11),
            complex_mul!(12, 13),
            complex_mul!(14, 15),
            complex_mul!(16, 17),
            complex_mul!(18, 19),
            complex_mul!(20, 21),
            complex_mul!(22, 23),
            complex_mul!(24, 25),
            complex_mul!(26, 27),
            complex_mul!(28, 29),
            complex_mul!(30, 31),
        )
    }
}

macro_rules! load_beta {
    () => { "ldr d0, [{betax}]\n" }
}

macro_rules! v_to_c {
    ($r0:tt, $r1:tt) => {
        concat!(
            "rev64 v", $r1, ".4s, v", $r1, ".4s\n",
            // use altx
            "fmla v", $r0, ".4s, v", $r1, ".4s, v7.4s\n",
        )
    }
}

macro_rules! permute_complex {
    () => {
        concat!(
            // load altx
            "ldr q7, [{altx}]", "\n",
            // permute even and odd elements
            v_to_c!(8, 9),
            v_to_c!(10, 11),
            v_to_c!(12, 13),
            v_to_c!(14, 15),
            v_to_c!(16, 17),
            v_to_c!(18, 19),
            v_to_c!(20, 21),
            v_to_c!(22, 23),
            v_to_c!(24, 25),
            v_to_c!(26, 27),
            v_to_c!(28, 29),
            v_to_c!(30, 31),
        )
    }
}

macro_rules! init_ab {
    (B) => {
        concat!(
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
            permute_complex!(),
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
        )
    };
}

macro_rules! inc_a {
    ($mr:tt) => {
        concat!("add {ax}, {ax}, #16*", $mr, " \n")
    };
}

macro_rules! inc_b {
    (S,$nr:tt) => {
        "add {x1},{cx} \n"
    };
    (B,$nr:tt) => {
        concat!("add {bx}, {bx}, #8*", $nr, " \n")
    };
}

macro_rules! load_b {
    (B, 0, $b_macro:tt, $i:tt) => {
        concat!(
            "ldr q3, [{bx}]", "\n",
        )
    };
    (B, 4, $b_macro:tt, $i:tt) => {
        concat!(
            "ldr q4, [{bx}, #0x10]", "\n",
        )
    };
    (B, $ni:tt, $b_macro:tt, $i:tt) => {
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
            "prfm pldl1keep, [{cx}] \n",
            "prfm pldl1keep, [{cx},#64]\n",
            "prfm pldl1keep, [{cx},#96]\n",
            "prfm pldl1keep, [{x1}] \n",
            "prfm pldl1keep, [{x1},#64]\n",
            "prfm pldl1keep, [{x1},#96]\n",
            // "prfm pldl1keep, [{x2}] \n",
            // "prfm pldl1keep, [{x2},#64]\n",
            // "prfm pldl1keep, [{x2},#96]\n",
            // "prfm pldl1keep, [{x3}] \n",
            // "prfm pldl1keep, [{x3},#64]\n",
            // "prfm pldl1keep, [{x3},#96]\n",
        )
    };
}

def_ukernel_neon_alt!(step_3_c, acc_3, store_3, 3, 4, B, C, ukernel_bbc);
def_ukernel_neon_alt!(step_3_c, acc_3, store_3, 3, 4, B, C, ukernel_3_bbp);
def_ukernel_neon_alt!(step_2_c, acc_2, store_2, 2, 4, B, C, ukernel_2_bbp);
def_ukernel_neon_alt!(step_1_c, acc_1, store_1, 1, 4, B, C, ukernel_1_bbp);


