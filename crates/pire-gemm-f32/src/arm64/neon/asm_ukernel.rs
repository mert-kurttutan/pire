use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC, UnaryFnC, TC_SIZE};
use pire_base::{c_mem, prefetch_0, def_ukernel_neon, mem};
use super::VS;

const ZERO: TC = 0f32;

const ONE_SCALAR: TC = 1f32;
const ZERO_SCALAR: TC = 0f32;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "fmla v", $r1, ".4s, v1.4s, v0.s[0]\n",
        ) 
    };
    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "fmla v", $r1, ".4s, v1.4s, v0.s[0]\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("dup v", r, ".4s, wzr \n",)*)
        })
    }
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr, $i:expr) => {
        concat!(
            "fmla v", $r3, ".4s", ", v", $r1,".4s, v", $r2, ".s[", $i, "]\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "ldr q", $r1, ", ", $m0, "\n",
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

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "ldr s1, [{alphax}]", "\n",
                #(
                    "fmul  v", r, ".4s, v", r, ".4s, v1.s[0]\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "ldr s0, [{betax}]", "\n",
            "/* {betax} */", "\n",

            "fcmp s0,#0.0", "\n",
        )
    }
}


macro_rules! acc_p {
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr, $r3:expr, $r4:expr, $r5:expr, $r6:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $q),
            beta_fmadd!(C, mem!($m0, "0x10"), $r2, $q),
            beta_fmadd!(C, mem!($m0, "0x20"), $r3, $q),
            beta_fmadd!(C, mem!($m0, "0x30"), $r4, $q),
            beta_fmadd!(C, mem!($m0, "0x40"), $r5, $q),
            beta_fmadd!(C, mem!($m0, "0x50"), $r6, $q),
        )
    };
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $q),
            beta_fmadd!(C, mem!($m0, "0x10"), $r2, $q),
            beta_fmadd!(C, mem!($m0, "0x20"), $r3, $q),
            beta_fmadd!(C, mem!($m0, "0x30"), $r4, $q),
        )
    };
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $q),
            beta_fmadd!(C, mem!($m0, "0x10"), $r2, $q),
        )
    };
}


macro_rules! loadp {
    (3, $m0:expr) => {
        concat!(
            loadp_unit!(mem!($m0), 0),
            loadp_unit!(mem!($m0, "0x10"), 1),
            loadp_unit!(mem!($m0, "0x20"), 2),
            loadp_unit!(mem!($m0, "0x30"), 3),
            loadp_unit!(mem!($m0, "0x40"), 4),
            loadp_unit!(mem!($m0, "0x50"), 5),
        )
    };
    (2, $m0:expr) => {
        concat!(
            loadp_unit!(mem!($m0), 0),
            loadp_unit!(mem!($m0, "0x10"), 1),
            loadp_unit!(mem!($m0, "0x20"), 2),
            loadp_unit!(mem!($m0, "0x30"), 3),
        )
    };
    (1, $m0:expr) => {
        concat!(
            loadp_unit!(mem!($m0), 0),
            loadp_unit!(mem!($m0, "0x10"), 1),
        )
    };
}

macro_rules! storep {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr, $r5:expr, $r6:expr) => {
        concat!(
            storep_unit!(C, $r1, mem!($m0)),
            storep_unit!(C, $r2, mem!($m0, "0x10")),
            storep_unit!(C, $r3, mem!($m0, "0x20")),
            storep_unit!(C, $r4, mem!($m0, "0x30")),
            storep_unit!(C, $r5, mem!($m0, "0x40")),
            storep_unit!(C, $r6, mem!($m0, "0x50")),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            storep_unit!(C, $r1, mem!($m0)),
            storep_unit!(C, $r2, mem!($m0, "0x10")),
            storep_unit!(C, $r3, mem!($m0, "0x20")),
            storep_unit!(C, $r4, mem!($m0, "0x30")),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            storep_unit!(C, $r1, mem!($m0)),
            storep_unit!(C, $r2, mem!($m0, "0x10")),
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
        concat!("add {bx}, {bx}, #4*", $nr, " \n")
    };
}

macro_rules! alpha_scale {
    () => {
        alpha_scale_0!(8,31)
    };
}

macro_rules! c_reg_3x4 {
    (0,0) => { 8 };
    (1,0) => { 9 };
    (2,0) => { 10 };
    (3,0) => { 11 };
    (4,0) => { 12 };
    (5,0) => { 13 };

    (0,1) => { 14 };
    (1,1) => { 15 };
    (2,1) => { 16 };
    (3,1) => { 17 };
    (4,1) => { 18 };
    (5,1) => { 19 };

    (0,2) => { 20 };
    (1,2) => { 21 };
    (2,2) => { 22 };
    (3,2) => { 23 };
    (4,2) => { 24 };
    (5,2) => { 25 };

    (0,3) => { 26 };
    (1,3) => { 27 };
    (2,3) => { 28 };
    (3,3) => { 29 };
    (4,3) => { 30 };
    (5,3) => { 31 };
}

macro_rules! c_reg_2x6 {
    (0,0) => { 8 };
    (1,0) => { 9 };
    (2,0) => { 10 };
    (3,0) => { 11 };

    (0,1) => { 12 };
    (1,1) => { 13 };
    (2,1) => { 14 };
    (3,1) => { 15 };

    (0,2) => { 16 };
    (1,2) => { 17 };
    (2,2) => { 18 };
    (3,2) => { 19 };

    (0,3) => { 20 };
    (1,3) => { 21 };
    (2,3) => { 22 };
    (3,3) => { 23 };

    (0,4) => { 24 };
    (1,4) => { 25 };
    (2,4) => { 26 };
    (3,4) => { 27 };

    (0,5) => { 28 };
    (1,5) => { 29 };
    (2,5) => { 30 };
    (3,5) => { 31 };
}

macro_rules! c_reg_1x6 {
    (0,0) => { 8 };
    (1,0) => { 9 };

    (0,1) => { 10 };
    (1,1) => { 11 };

    (0,2) => { 12 };
    (1,2) => { 13 };

    (0,3) => { 14 };
    (1,3) => { 15 };

    (0,4) => { 16 };
    (1,4) => { 17 };

    (0,5) => { 18 };
    (1,5) => { 19 };
}


macro_rules! acc_3x4 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!(
            $layout, c_mem!($ni), $q, c_reg_3x4!(0,$ni), c_reg_3x4!(1,$ni), c_reg_3x4!(2,$ni), c_reg_3x4!(3,$ni), c_reg_3x4!(4,$ni), c_reg_3x4!(5,$ni)
        )
    };
}

macro_rules! store_3x4 {
    ($ni:tt, $layout:tt) => {
        storep!(
            $layout, c_mem!($ni), c_reg_3x4!(0,$ni), c_reg_3x4!(1,$ni), c_reg_3x4!(2,$ni), c_reg_3x4!(3,$ni), c_reg_3x4!(4,$ni), c_reg_3x4!(5,$ni)
        )
    };
}

macro_rules! acc_2x6 {
    ($ni:tt, $layout:tt,$q:tt) => {
        acc_p!($layout, c_mem!($ni), $q, c_reg_2x6!(0,$ni), c_reg_2x6!(1,$ni), c_reg_2x6!(2,$ni), c_reg_2x6!(3,$ni))
    };
}

macro_rules! store_2x6 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_2x6!(0,$ni), c_reg_2x6!(1,$ni), c_reg_2x6!(2,$ni), c_reg_2x6!(3,$ni))
    };
}

macro_rules! acc_1x6 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!($layout, c_mem!($ni), $q, c_reg_1x6!(0,$ni), c_reg_1x6!(1,$ni))
    };
}

macro_rules! store_1x6 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_1x6!(0,$ni), c_reg_1x6!(1,$ni))
    };
}

macro_rules! load_b {
    (B, $r:expr) => {
        concat!(
            "ldr q", $r, ", [{bx}]", "\n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt) => {
        loadp!($mr, "{ax}")
    };
}

macro_rules! fmadd_3v {
    (0) => {
        concat!(
            vfmadd!(0, 6, 8, 0),
            vfmadd!(1, 6, 9, 0),
            vfmadd!(2, 6, 10, 0),
            vfmadd!(3, 6, 11, 0),
            vfmadd!(4, 6, 12, 0),
            vfmadd!(5, 6, 13, 0),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 6, 14, 1),
            vfmadd!(1, 6, 15, 1),
            vfmadd!(2, 6, 16, 1),
            vfmadd!(3, 6, 17, 1),
            vfmadd!(4, 6, 18, 1),
            vfmadd!(5, 6, 19, 1),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 6, 20, 2),
            vfmadd!(1, 6, 21, 2),
            vfmadd!(2, 6, 22, 2),
            vfmadd!(3, 6, 23, 2),
            vfmadd!(4, 6, 24, 2),
            vfmadd!(5, 6, 25, 2),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 6, 26, 3),
            vfmadd!(1, 6, 27, 3),
            vfmadd!(2, 6, 28, 3),
            vfmadd!(3, 6, 29, 3),
            vfmadd!(4, 6, 30, 3),
            vfmadd!(5, 6, 31, 3),
        )
    };
}

macro_rules! fmadd_2v {
    (0) => {
        concat!(
            vfmadd!(0, 4, 8, 0),
            vfmadd!(1, 4, 9, 0),
            vfmadd!(2, 4, 10, 0),
            vfmadd!(3, 4, 11, 0),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 4, 12, 1),
            vfmadd!(1, 4, 13, 1),
            vfmadd!(2, 4, 14, 1),
            vfmadd!(3, 4, 15, 1),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 4, 16, 2),
            vfmadd!(1, 4, 17, 2),
            vfmadd!(2, 4, 18, 2),
            vfmadd!(3, 4, 19, 2),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 4, 20, 3),
            vfmadd!(1, 4, 21, 3),
            vfmadd!(2, 4, 22, 3),
            vfmadd!(3, 4, 23, 3),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 5, 24, 0),
            vfmadd!(1, 5, 25, 0),
            vfmadd!(2, 5, 26, 0),
            vfmadd!(3, 5, 27, 0),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 5, 28, 1),
            vfmadd!(1, 5, 29, 1),
            vfmadd!(2, 5, 30, 1),
            vfmadd!(3, 5, 31, 1),
        )
    };
}


macro_rules! fmadd_1v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 8, 0),
            vfmadd!(1, 2, 9, 0),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 10, 1),
            vfmadd!(1, 2, 11, 1),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 2, 12, 2),
            vfmadd!(1, 2, 13, 2),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 2, 14, 3),
            vfmadd!(1, 2, 15, 3),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 3, 16, 0),
            vfmadd!(1, 3, 17, 0),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 3, 18, 1),
            vfmadd!(1, 3, 19, 1),
        )
    };
}


// ***************************** 3x4 ******************************* //
macro_rules! step_3x4 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(3),
                "add {ax}, {ax}, #4*3*8 \n",
                load_b!($b_layout, 6),
                #(
                    fmadd_3v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 2x6 ******************************* //
macro_rules! step_2x6 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2),
                "add {ax}, {ax}, #4*2*8 \n",
                load_b!($b_layout, 4),
                #(
                    fmadd_2v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 2x6 ******************************* //
macro_rules! step_1x6 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(1),
                "add {ax}, {ax}, #4*1*8 \n",
                load_b!($b_layout, 2),
                #(
                    fmadd_1v!(n),
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
        )
    };
}

def_ukernel_neon!(step_3x4, acc_3x4, store_3x4, 3, 4, 4, 5, B, C, ukernel_bbc);

def_ukernel_neon!(step_3x4, acc_3x4, store_3x4, 3, 4, 4, 5, B, C, ukernel_3_bbp);
def_ukernel_neon!(step_2x6, acc_2x6, store_2x6, 2, 4, 4, 5, B, C, ukernel_2_bbp);
def_ukernel_neon!(step_1x6, acc_1x6, store_1x6, 1, 4, 4, 5, B, C, ukernel_1_bbp);

def_ukernel_neon!(step_3x4, acc_3x4, store_3x4, 3, 4, 1, 4, B, C, ukernel_n_bbc);

def_ukernel_neon!(step_3x4, acc_3x4, store_3x4, 3, 4, 1, 4, B, C, ukernel_3xn_bbp);
def_ukernel_neon!(step_2x6, acc_2x6, store_2x6, 2, 4, 1, 4, B, C, ukernel_2xn_bbp);
def_ukernel_neon!(step_1x6, acc_1x6, store_1x6, 1, 4, 1, 4, B, C, ukernel_1xn_bbp);