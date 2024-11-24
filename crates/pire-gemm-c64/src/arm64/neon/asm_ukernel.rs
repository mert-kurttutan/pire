use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC, UnaryFnC, TC_SIZE};
use pire_base::{c_mem, prefetch_0, def_ukernel_neon_alt, mem};
use super::VS;

const ZERO: TC = TC::ZERO;

const ONE_SCALAR: TC = TC::ONE;
const ZERO_SCALAR: TC = TC::ZERO;

macro_rules! alt_arr {
    ($vec_name:ident) => {
        let $vec_name = [-1.0f64, 1.0f64];
    }
}


macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            // "fmla v", $r1, ".2d, v1.2d, v0.s[0], #0\n",
            "fadd  v", $r1, ".2d, v", $r1, ".2d, v1.2d\n",

        ) 
    };
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "fmul v2.2d, v1.2d, v0.d[0]\n",
            "fmul v3.2d, v1.2d, v0.d[1]\n",
            v_to_c!(2, 3),
            "fadd  v", $r1, ".2d, v", $r1, ".2d, v2.2d\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("dup v", r, ".2d, xzr \n",)*)
        })
    }
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr, $i1:expr) => {
        concat!(
            "fmla v", $r3, ".2d", ", v", $r1,".2d, v", $r2, ".d[", $i1, "]\n",
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

macro_rules! complex_mul {
    ($r0:tt, $rt:tt) => {
        concat!(
            "ext v", $rt, ".16b, v", $r0, ".16b, v", $r0, ".16b, #8\n",

            "fmul v", $r0, ".2d, v", $r0, ".2d, v1.d[0]\n",
            "fmul v", $rt, ".2d, v", $rt, ".2d, v1.d[1]\n",

            "fmla v", $r0, ".2d, v", $rt, ".2d, v7.2d\n",


        )
    };
}

macro_rules! alpha_scale_0 {
    () => {
        concat!(
            "ldr q1, [{alphax}]", "\n",
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
    () => {
        concat!(
            "ldr q0, [{betax}]", "\n",
        )
    }
}

macro_rules! v_to_c {
    ($r0:tt, $r1:tt) => {
        concat!(
            "ext v", $r1, ".16b, v", $r1, ".16b, v", $r1, ".16b, #8\n",
            // use altx
            "fmla v", $r0, ".2d, v", $r1, ".2d, v7.2d\n",
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
            permute_complex!(),
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
        concat!("add {bx}, {bx}, #16*", $nr, " \n")
    };
}

macro_rules! alpha_scale {
    () => {
        alpha_scale_0!()
    };
}

macro_rules! c_reg_3x2 {
    (0,0) => { 8 };
    (1,0) => { 10 };
    (2,0) => { 12 };
    (3,0) => { 14 };
    (4,0) => { 16 };
    (5,0) => { 18 };

    (0,1) => { 20 };
    (1,1) => { 22 };
    (2,1) => { 24 };
    (3,1) => { 26 };
    (4,1) => { 28 };
    (5,1) => { 30 };
}

macro_rules! c_reg_2x3 {
    (0,0) => { 8 };
    (1,0) => { 10 };
    (2,0) => { 12 };
    (3,0) => { 14 };

    (0,1) => { 16 };
    (1,1) => { 18 };
    (2,1) => { 20 };
    (3,1) => { 22 };

    (0,2) => { 24 };
    (1,2) => { 16 };
    (2,2) => { 28 };
    (3,2) => { 30 };
}

macro_rules! c_reg_1x3 {
    (0,0) => { 8 };
    (1,0) => { 10 };

    (0,1) => { 12 };
    (1,1) => { 14 };

    (0,2) => { 16 };
    (1,2) => { 18 };
}



macro_rules! acc_3x2 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!(
            $layout, c_mem!($ni), $q, c_reg_3x2!(0,$ni), c_reg_3x2!(1,$ni), c_reg_3x2!(2,$ni), c_reg_3x2!(3,$ni), c_reg_3x2!(4,$ni), c_reg_3x2!(5,$ni)
        )
    };
}

macro_rules! store_3x2 {
    ($ni:tt, $layout:tt) => {
        storep!(
            $layout, c_mem!($ni), c_reg_3x2!(0,$ni), c_reg_3x2!(1,$ni), c_reg_3x2!(2,$ni), c_reg_3x2!(3,$ni), c_reg_3x2!(4,$ni), c_reg_3x2!(5,$ni)
        )
    };
}

macro_rules! acc_2x3 {
    ($ni:tt, $layout:tt,$q:tt) => {
        acc_p!($layout, c_mem!($ni), $q, c_reg_2x3!(0,$ni), c_reg_2x3!(1,$ni), c_reg_2x3!(2,$ni), c_reg_2x3!(3,$ni))
    };
}

macro_rules! store_2x3 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_2x3!(0,$ni), c_reg_2x3!(1,$ni), c_reg_2x3!(2,$ni), c_reg_2x3!(3,$ni))
    };
}

macro_rules! acc_1x3 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!($layout, c_mem!($ni), $q, c_reg_1x3!(0,$ni), c_reg_1x3!(1,$ni))
    };
}

macro_rules! store_1x3 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_1x3!(0,$ni), c_reg_1x3!(1,$ni))
    };
}

macro_rules! load_b {
    (B, $r:expr) => {
        concat!(
            "ldr q", $r, ", [{bx}]", "\n",
        )
    };
    (B, $r1:expr, $r2:expr) => {
        concat!(
            "ldr q", $r1, ", [{bx}]", "\n",
            "ldr q", $r2, ", [{bx}, #0x10]", "\n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt) => {
        loadp!($mr, "{ax}")
    };
}

macro_rules! fmadd_3v {
    (0,0) => {
        concat!(
            vfmadd!(0, 6, 8, 0),
            vfmadd!(1, 6, 10, 0),
            vfmadd!(2, 6, 12, 0),
            vfmadd!(3, 6, 14, 0),
            vfmadd!(4, 6, 16, 0),
            vfmadd!(5, 6, 18, 0),
        )
    };
    (0,1) => {
        concat!(
            vfmadd!(0, 6, 9, 1),
            vfmadd!(1, 6, 11, 1),
            vfmadd!(2, 6, 13, 1),
            vfmadd!(3, 6, 15, 1),
            vfmadd!(4, 6, 17, 1),
            vfmadd!(5, 6, 19, 1),
        )
    };
    (1,0) => {
        concat!(
            vfmadd!(0, 7, 20, 0),
            vfmadd!(1, 7, 22, 0),
            vfmadd!(2, 7, 24, 0),
            vfmadd!(3, 7, 26, 0),
            vfmadd!(4, 7, 28, 0),
            vfmadd!(5, 7, 30, 0),
        )
    };
    (1,1) => {
        concat!(
            vfmadd!(0, 7, 21, 1),
            vfmadd!(1, 7, 23, 1),
            vfmadd!(2, 7, 25, 1),
            vfmadd!(3, 7, 27, 1),
            vfmadd!(4, 7, 29, 1),
            vfmadd!(5, 7, 31, 1),
        )
    };
}

macro_rules! fmadd_2v {
    (0,0) => {
        concat!(
            vfmadd!(0, 4, 8, 0),
            vfmadd!(1, 4, 10, 0),
            vfmadd!(2, 4, 12, 0),
            vfmadd!(3, 4, 14, 0),
        )
    };
    (0,1) => {
        concat!(
            vfmadd!(0, 4, 9, 1),
            vfmadd!(1, 4, 11, 1),
            vfmadd!(2, 4, 13, 1),
            vfmadd!(3, 4, 15, 1),
        )
    };
    (1,0) => {
        concat!(
            vfmadd!(0, 5, 16, 0),
            vfmadd!(1, 5, 18, 0),
            vfmadd!(2, 5, 20, 0),
            vfmadd!(3, 5, 22, 0),
        )
    };
    (1,1) => {
        concat!(
            vfmadd!(0, 5, 17, 1),
            vfmadd!(1, 5, 19, 1),
            vfmadd!(2, 5, 21, 1),
            vfmadd!(3, 5, 23, 1),
        )
    };
    (2,0) => {
        concat!(
            vfmadd!(0, 6, 24, 0),
            vfmadd!(1, 6, 26, 0),
            vfmadd!(2, 6, 28, 0),
            vfmadd!(3, 6, 30, 0),
        )
    };
    (2,1) => {
        concat!(
            vfmadd!(0, 6, 25, 1),
            vfmadd!(1, 6, 27, 1),
            vfmadd!(2, 6, 29, 1),
            vfmadd!(3, 6, 31, 1),
        )
    };
}


macro_rules! fmadd_1v {
    (0,0) => {
        concat!(
            vfmadd!(0, 2, 8, 0),
            vfmadd!(1, 2, 10, 0),
        )
    };
    (0,1) => {
        concat!(
            vfmadd!(0, 2, 9, 1),
            vfmadd!(1, 2, 11, 1),
        )
    };
    (1,0) => {
        concat!(
            vfmadd!(0, 3, 12, 0),
            vfmadd!(1, 3, 14, 0),
        )
    };
    (1,1) => {
        concat!(
            vfmadd!(0, 3, 13, 1),
            vfmadd!(1, 3, 15, 1),
        )
    };
    (2,0) => {
        concat!(
            vfmadd!(0, 4, 14, 0),
            vfmadd!(1, 4, 16, 0),
        )
    };
    (2,1) => {
        concat!(
            vfmadd!(0, 4, 15, 1),
            vfmadd!(1, 4, 17, 1),
        )
    };
}


// ***************************** 3x2 ******************************* //
macro_rules! step_3x2 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(3),
                "add {ax}, {ax}, #16*3*2 \n",
                load_b!($b_layout, 6, 7),
                #(
                    fmadd_3v!(n,0),
                    fmadd_3v!(n,1),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 2x3 ******************************* //
macro_rules! step_2x3 {
    ($nr:tt,  $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2),
                "add {ax}, {ax}, #16*2*2 \n",
                load_b!($b_layout, 4, 5),
                #(
                    fmadd_2v!(n,0),
                    fmadd_2v!(n,1),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 2x3 ******************************* //
macro_rules! step_1x3 {
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2),
                "add {ax}, {ax}, #16*1*2 \n",
                load_b!($b_layout, 2, 3),
                #(
                    fmadd_1v!(n,0),
                    fmadd_1v!(n,1),
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
            // "prfm pldl1keep, [{x2}] \n",
            // "prfm pldl1keep, [{x2},#64]\n",
            // "prfm pldl1keep, [{x2},#96]\n",
            // "prfm pldl1keep, [{x3}] \n",
            // "prfm pldl1keep, [{x3},#64]\n",
            // "prfm pldl1keep, [{x3},#96]\n",
        )
    };
}

def_ukernel_neon_alt!(step_3x2, acc_3x2, store_3x2, 3, 2, B, C, ukernel_bbc);

def_ukernel_neon_alt!(step_3x2, acc_3x2, store_3x2, 3, 2, B, C, ukernel_3_bbp);
def_ukernel_neon_alt!(step_2x3, acc_2x3, store_2x3, 2, 2, B, C, ukernel_2_bbp);
def_ukernel_neon_alt!(step_1x3, acc_1x3, store_1x3, 1, 2, B, C, ukernel_1_bbp);

