use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC};
use glar_base::{load_buf, store_buf, c_mem2};

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

macro_rules! mem {
    ($m0:tt, $b0:tt, $b1:tt) => {
        concat!("[", $m0, ", #", $b0, "+", $b1, "]")
    };
    ($m0:tt, $b0:tt) => {
        concat!("[", $m0, ", #", $b0, "]")
    };
    ($m0:tt) => {
        concat!("[", $m0, "]")
    };
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr, $i1:expr) => {
        concat!(
            "fmla v", $r3, ".2d", ", v", $r1,".2d, v", $r2, ".d[", $i1, "]\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($layout:tt, $m0:expr, $r1:expr) => {
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

macro_rules! asm_alpha_scale_0 {
    ($mr:tt,$nr:tt) => {
        concat!(
            // check if s1 is 0 bit
            "cmp {is_alpha_one:w}, #0", "\n",

            "BEQ 13f", "\n",
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
            "13:", "\n",
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
    ($mr:tt,$nr:tt) => {
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
    (6, $layout:tt, $m0:expr, $b0:expr) => {
        concat!(
            loadp_unit!($layout, mem!($m0, $b0), 0),
            loadp_unit!($layout, mem!($m0, $b0, "0x10"), 1),
            loadp_unit!($layout, mem!($m0, $b0, "0x20"), 2),
            loadp_unit!($layout, mem!($m0, $b0, "0x30"), 3),
            loadp_unit!($layout, mem!($m0, $b0, "0x40"), 4),
            loadp_unit!($layout, mem!($m0, $b0, "0x50"), 5),
        )
    };
    (4, $layout:tt, $m0:expr, $b0:expr) => {
        concat!(
            loadp_unit!($layout, mem!($m0, $b0), 0),
            loadp_unit!($layout, mem!($m0, $b0, "0x10"), 1),
            loadp_unit!($layout, mem!($m0, $b0, "0x20"), 2),
            loadp_unit!($layout, mem!($m0, $b0, "0x30"), 3),
        )
    };
    (2, $layout:tt, $m0:expr, $b0:expr) => {
        concat!(
            loadp_unit!($layout, mem!($m0, $b0), 0),
            loadp_unit!($layout, mem!($m0, $b0, "0x10"), 1),
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


macro_rules! asm_init_ab {
    ($KER:tt,B,B) => {
        concat!(
            "/* {x5} */", "\n",
            "/* {x4} */", "\n",

            "/* {x3} */", "\n",

            "/* {x2} */", "\n",

            "/* {x1} */", "\n",
            "ldr {x0}, [{dim_arrx}, #24]", "\n",
            "cmp {x0}, #0",
        )
    };
    ($ker:tt,B,S) => {
        concat!(
            // mov cs_b to reg
            "mov ({dim_arrx}), {x1}", "\n",
            // "mov 8({dim_arrx}), {x2}", "\n",
            "ldr {x0}, [{dim_arrx}, #24]", "\n",
            "cmp {x0}, #0",
        )
    };
}


macro_rules! asm_c_load {
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


macro_rules! asm_vzeroall {
    ($mr:tt,$nr:tt) => {vzeroall!(8,31)};
    // (8,1) => {vzeroall!(4,5)};

    // (4,2) => {vzeroall!(4,5)};
    // (4,1) => {vzeroall!(4,4)};
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

macro_rules! inc_a_k_unroll {
    (C, $X:tt, $K:tt) => {
        ""
    };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add {ax}, {ax}, #16*", $K, "*", $X, " \n",
        )
    };
}

macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => {
        ""
    };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add {bx}, {bx}, #16*", $K, "*", $X, " \n",
        )
    };
}

macro_rules! asm_alpha_scale {
    ($mr:tt, $nr:tt) => {
        asm_alpha_scale_0!(8,31)
    };
    (8, 1) => {
        asm_alpha_scale_0!(4,5)
    };

    (4, 2) => {
        asm_alpha_scale_0!(4,5)
    };
    (4, 1) => {
        asm_alpha_scale_0!(4,4)
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
            $layout, c_mem2!($ni), $q, c_reg_3x2!(0,$ni), c_reg_3x2!(1,$ni), c_reg_3x2!(2,$ni), c_reg_3x2!(3,$ni), c_reg_3x2!(4,$ni), c_reg_3x2!(5,$ni)
        )
    };
}

macro_rules! store_3x2 {
    ($ni:tt, $layout:tt) => {
        storep!(
            $layout, c_mem2!($ni), c_reg_3x2!(0,$ni), c_reg_3x2!(1,$ni), c_reg_3x2!(2,$ni), c_reg_3x2!(3,$ni), c_reg_3x2!(4,$ni), c_reg_3x2!(5,$ni)
        )
    };
}

macro_rules! acc_2x3 {
    ($ni:tt, $layout:tt,$q:tt) => {
        acc_p!($layout, c_mem2!($ni), $q, c_reg_2x3!(0,$ni), c_reg_2x3!(1,$ni), c_reg_2x3!(2,$ni), c_reg_2x3!(3,$ni))
    };
}

macro_rules! store_2x3 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem2!($ni), c_reg_2x3!(0,$ni), c_reg_2x3!(1,$ni), c_reg_2x3!(2,$ni), c_reg_2x3!(3,$ni))
    };
}

macro_rules! acc_1x3 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!($layout, c_mem2!($ni), $q, c_reg_1x3!(0,$ni), c_reg_1x3!(1,$ni))
    };
}

macro_rules! store_1x3 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem2!($ni), c_reg_1x3!(0,$ni), c_reg_1x3!(1,$ni))
    };
}



macro_rules! cum_seq {
    ($step_macro:tt, $nr:tt, $layout:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!(n, $layout),)*)
        })
    };
    ($step_macro:tt, $nr:tt, $layout:tt, $q:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!(n, $layout,$q),)*)
        })
    };
}

macro_rules! load_b {
    (B, $ni:tt, $K:tt, $X:tt, $r:expr) => {
        concat!(
            "ldr q", $r, ", [{bx}, #", $ni, "*16+", $K, "*", $X, "*16]", "\n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt, B, $K:tt) => {
        loadp!($mr, B, "{ax}", concat!($mr,"*16*",$K))
    };
    ($mr:tt, C, $K:tt) => {
        loadp!($mr, C, "[{ax}]")
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
            vfmadd!(0, 5, 8, 0),
            vfmadd!(1, 5, 10, 0),
            vfmadd!(2, 5, 12, 0),
            vfmadd!(3, 5, 14, 0),
        )
    };
    (0,1) => {
        concat!(
            vfmadd!(0, 5, 9, 1),
            vfmadd!(1, 5, 11, 1),
            vfmadd!(2, 5, 13, 1),
            vfmadd!(3, 5, 15, 1),
        )
    };
    (1,0) => {
        concat!(
            vfmadd!(0, 6, 16, 0),
            vfmadd!(1, 6, 18, 0),
            vfmadd!(2, 6, 20, 0),
            vfmadd!(3, 6, 22, 0),
        )
    };
    (1,1) => {
        concat!(
            vfmadd!(0, 6, 17, 1),
            vfmadd!(1, 6, 19, 1),
            vfmadd!(2, 6, 21, 1),
            vfmadd!(3, 6, 23, 1),
        )
    };
    (2,0) => {
        concat!(
            vfmadd!(0, 7, 24, 0),
            vfmadd!(1, 7, 26, 0),
            vfmadd!(2, 7, 28, 0),
            vfmadd!(3, 7, 30, 0),
        )
    };
    (2,1) => {
        concat!(
            vfmadd!(0, 7, 25, 1),
            vfmadd!(1, 7, 27, 1),
            vfmadd!(2, 7, 29, 1),
            vfmadd!(3, 7, 31, 1),
        )
    };
}


macro_rules! fmadd_1v {
    (0,0) => {
        concat!(
            vfmadd!(0, 5, 8, 0),
            vfmadd!(1, 5, 10, 0),
        )
    };
    (0,1) => {
        concat!(
            vfmadd!(0, 5, 9, 1),
            vfmadd!(1, 5, 11, 1),
        )
    };
    (1,0) => {
        concat!(
            vfmadd!(0, 6, 12, 0),
            vfmadd!(1, 6, 14, 0),
        )
    };
    (1,1) => {
        concat!(
            vfmadd!(0, 6, 13, 1),
            vfmadd!(1, 6, 15, 1),
        )
    };
    (2,0) => {
        concat!(
            vfmadd!(0, 7, 14, 0),
            vfmadd!(1, 7, 16, 0),
        )
    };
    (2,1) => {
        concat!(
            vfmadd!(0, 7, 15, 1),
            vfmadd!(1, 7, 17, 1),
        )
    };
}


// ***************************** 3x2 ******************************* //
macro_rules! step_3x2 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(6, $a_layout, $K),
                load_b!($b_layout, 0, $K, $nr, 6),
                load_b!($b_layout, 1, $K, $nr, 7),
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
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(4, $a_layout, $K),
                load_b!($b_layout, 0, $K, $nr, 5),
                load_b!($b_layout, 1, $K, $nr, 6),
                load_b!($b_layout, 2, $K, $nr, 7),
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
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2, $a_layout, $K),
                load_b!($b_layout, 0, $K, $nr, 5),
                load_b!($b_layout, 1, $K, $nr, 6),
                load_b!($b_layout, 2, $K, $nr, 7),
                #(
                    fmadd_1v!(n,0),
                    fmadd_1v!(n,1),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! prefetch_0 {
    ($dist:tt, $reg:tt, $k_i:tt) => {
        concat!(
            "prfm pldl1keep, [", $reg, ", #", $k_i, "*64+", $dist, "] \n",
        )
    };
}

use crate::MyFn;

macro_rules! prefetch_c {
    (24, 4) => {
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
    (24, $nr:tt) => {
        ""
        // seq!(j in 0..$nr {
        //     concat!(
        // 		#(

        // 		)*
        // 	)
        // });
    };
    (16, $nr:tt) => {
            ""
        // seq!(j in 0..$nr {
        //     _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
        // });
    };
    (8, $nr:tt) => {
            ""
        // seq!(j in 0..$nr {
        //     _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
        // });
    }
}


macro_rules! def_ukernel {
    (
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $a_layout:tt, $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        pub(crate) unsafe fn $func_name<F: MyFn, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize,
            f: F,
        ) {
            let k_iter = k / 4;
            let k_left = k % 4;
            let mut dim_arr = [d_arr[0]*16, d_arr[1]*16, d_arr[3]*16, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let alt = [-1.0f64, 1.0f64];
            let c_cs = d_arr[3];
            let is_alpha_one = if *alpha == TA::ONE {
                0i32
            } else {
                1i32
            };
            let is_beta = if *beta == TB::ZERO {
                0i32
            } else if *beta == TB::ONE {
                1i32
            } else {
                2i32
            };
            if BUF || m != $mr {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, $mr);
                dim_arr[2] = $mr*16;
                cf = c_buf.as_mut_ptr();
            }
            asm!(
                asm_vzeroall!($mr,$nr),

                prefetch_c!(24,4),
        
                asm_init_ab!($mr,$a_layout,$b_layout),
                
                // 3 -> CONSIDKLEFT
                "BEQ 3f",
                
                // 2 -> KITER
                "2:",
                prefetch_0!(192, "{bx}", 0),
                $step_macro!($nr, $a_layout, $b_layout, 0),
                $step_macro!($nr, $a_layout, $b_layout, 1),
                prefetch_0!(256, "{bx}", 0),
                $step_macro!($nr, $a_layout, $b_layout, 2),
                $step_macro!($nr, $a_layout, $b_layout, 3),

                inc_a_k_unroll!($a_layout, $mr, 4),
                inc_b_k_unroll!($b_layout, $nr, 4),
        
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
                $step_macro!($nr, $a_layout, $b_layout, 0),
                inc_a_k_unroll!($a_layout, $mr, 1),
                inc_b_k_unroll!($b_layout, $nr, 1),

                "sub {x0}, {x0}, #1",
        
                // 4 -> KLEFT
                "cmp {x0}, 0",
                "BNE 4b",
        
                // 5 -> POSTACCUM
                "5:",
                permute_complex!($mr, $nr),
                asm_c_load!($nr),
                // scale by alpha
                asm_alpha_scale!($mr, $nr),

                "cmp {is_beta:w}, #0", "\n",
                "BEQ 6f",

                "cmp {is_beta:w}, #1", "\n",
                "BEQ 15f",

                load_beta!(),
                cum_seq!($acc_macro,$nr,$is_partial,2),
                // jumpt to 6
                "B 6f",
                "15:",
                cum_seq!($acc_macro,$nr,$is_partial,1),

                // 6 -> BETAZERO
                "6:",
                cum_seq!($store_macro,$nr,$is_partial),
                
                // 7 -> DDONE
                "7:",
                ax = inout(reg) a => _,
                bx = inout(reg) b => _,
                cx = inout(reg) cf => _,
                dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                alphax = inout(reg) alpha => _,
                is_alpha_one = in(reg) is_alpha_one,
                is_beta = in(reg) is_beta,
                betax = inout(reg) beta => _,
                x0 = out(reg) _,
                x1 = out(reg) _,
                x2 = out(reg) _,
                x3 = out(reg) _,
                x4 = out(reg) _,
                x5 = out(reg) _,
                altx = inout(reg) alt.as_ptr() => _,

                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            );
            if BUF || m != $mr {
                for j in 0..$nr {
                    f.call(cf.add(j*$mr), $mr);
                }
                store_buf(c, d_arr[2], c_cs, &c_buf, m, $nr, $mr);
            } else {
                for j in 0..$nr {
                    f.call(cf.add(j*c_cs), m);
                }
            }
        }
    };
}

macro_rules! def_ukernelxn {
    (
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $a_layout:tt, $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        pub(crate) unsafe fn $func_name<F: MyFn, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize, n: usize,
            f: F,
        ) {
            let k_iter = k / 4;
            let k_left = k % 4;
            let mut dim_arr = [d_arr[0]*16, d_arr[1]*16, d_arr[3]*16, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let is_alpha_one = if *alpha == TA::ONE {
                0i32
            } else {
                1i32
            };
            let is_beta = if *beta == TB::ZERO {
                0i32
            } else if *beta == TB::ONE {
                1i32
            } else {
                2i32
            };
            let c_cs = d_arr[3];
            let alt = [-1.0f64, 1.0f64];
            if BUF || m != $mr {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, $mr);
                dim_arr[2] = $mr*16;
                cf = c_buf.as_mut_ptr();
            }
            let _ = 'blk: {
                seq!(ni in 1..$nr {
                    if ni == n {
                        asm!(
                            asm_vzeroall!($mr,ni),
                
                            asm_init_ab!($mr,$a_layout,$b_layout),
                        
                            // 3 -> CONSIDKLEFT
                            "BEQ 3f",
                        
                            // 2 -> KITER
                            "2:",
                            prefetch_0!(128, "{cx}", 0),
                            $step_macro!(ni, $a_layout, $b_layout, 0),
                            $step_macro!(ni, $a_layout, $b_layout, 1),
                            $step_macro!(ni, $a_layout, $b_layout, 2),
                            $step_macro!(ni, $a_layout, $b_layout, 3),
            
                            inc_a_k_unroll!($a_layout, $mr, 4),
                            inc_b_k_unroll!($b_layout, ni, 4),
                
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
                            $step_macro!(ni, $a_layout, $b_layout, 0),
                            inc_a_k_unroll!($a_layout, $mr, 1),
                            inc_b_k_unroll!($b_layout, ni, 1),

                            "sub {x0}, {x0}, #1",
                
                            // 4 -> KLEFT
                            "cmp {x0}, 0",
                            "BNE 4b",
                
                            // 5 -> POSTACCUM
                            "5:",
                            permute_complex!($mr, $nr),
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),
                            "cmp {is_beta:w}, #0", "\n",
                            "BEQ 6f",

                            "cmp {is_beta:w}, #1", "\n",
                            "BEQ 15f",
            
                            load_beta!(),
                            cum_seq!($acc_macro,ni,$is_partial,2),
                            "B 6f",

                            "15:",
                            // 6 -> BETAZERO
                            cum_seq!($acc_macro,ni,$is_partial,1),

                            // 6 -> BETAZERO
                            "6:",
                            cum_seq!($store_macro,ni,$is_partial),
                            
                            // 7 -> DDONE
                            "7:",
                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) cf => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            is_alpha_one = in(reg) is_alpha_one,
                            betax = inout(reg) beta => _,
                            is_beta = in(reg) is_beta,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            x4 = out(reg) _,
                            x5 = out(reg) _,
                            altx = inout(reg) alt.as_ptr() => _,
                            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                            out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                            out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                        );
                        break 'blk;
                    }
                });
            };
            if BUF || m != $mr {
                for j in 0..n {
                    f.call(cf.add(j*$mr), $mr);
                }
                store_buf(c, d_arr[2], c_cs, &c_buf, m, n, $mr);
            } else {
                for j in 0..n {
                    f.call(cf.add(j*c_cs), m);
                }
            }
        }
    };
}

def_ukernel!(step_3x2, acc_3x2, store_3x2, 6, 2, B, B, C, ukernel_bb);

def_ukernel!(step_3x2, acc_3x2, store_3x2, 6, 2, B, B, C, ukernel_3_bb_partial);
def_ukernel!(step_2x3, acc_2x3, store_2x3, 4, 2, B, B, C, ukernel_2_bb_partial);
def_ukernel!(step_1x3, acc_1x3, store_1x3, 2, 2, B, B, C, ukernel_1_bb_partial);

def_ukernelxn!(step_3x2, acc_3x2, store_3x2, 6, 2, B, B, C, ukernel_n_bb);

def_ukernelxn!(step_3x2, acc_3x2, store_3x2, 6, 2, B, B, C, ukernel_3xn_bb_partial);
def_ukernelxn!(step_2x3, acc_2x3, store_2x3, 4, 2, B, B, C, ukernel_2xn_bb_partial);
def_ukernelxn!(step_1x3, acc_1x3, store_1x3, 2, 2, B, B, C, ukernel_1xn_bb_partial);

