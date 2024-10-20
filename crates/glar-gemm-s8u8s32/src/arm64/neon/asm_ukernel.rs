use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC, TC_SIZE};
use glar_base::{load_buf, store_buf, c_mem, prefetch_0};


macro_rules! unzip_tuple {
    ($r1:tt, $r2:tt,$rt1:tt,$rt2:tt) => {
        concat!(
            "uzp1 v", $rt1, ".2d, v", $r1, ".2d, v", $r2, ".2d\n",
            "uzp2 v", $rt2, ".2d, v", $r1, ".2d, v", $r2, ".2d\n",
            // copy uzp1 to z8 and uzp2 to v11
            "orr v", $r1, ".16b, v", $rt1, ".16b, v", $rt1, ".16b\n",
            "orr v", $r2, ".16b, v", $rt2, ".16b, v", $rt2, ".16b\n",
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
            "ldr q1, ", $m0, "\n",
            "add v", $r1, ".4s, v", $r1, ".4s, v1.4s\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,1) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "add v", $r1, ".4s, v", $r1, ".4s, v1.4s\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "scvtf v1.4s, v1.4s\n",
            "scvtf v", $r1, ".4s, v", $r1, ".4s\n",
            "fmla v", $r1, ".4s, v1.4s, v0.s[0]\n",
            "fcvtzs v", $r1, ".4s, v", $r1, ".4s\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,2) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "scvtf v1.4s, v1.4s\n",
            "scvtf v", $r1, ".4s, v", $r1, ".4s\n",
            "fmla v", $r1, ".4s, v1.4s, v0.s[0]\n",
            "fcvtzs v", $r1, ".4s, v", $r1, ".4s\n",
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

macro_rules! mem {
    ($m0:tt, $b0:tt, $b1:tt) => {
        concat!("[", $m0, ", ", $b0, ", ", $b1, "]")
    };
    ($m0:tt, $b0:tt) => {
        concat!("[", $m0, ", #", $b0, "]")
    };
    ($m0:tt) => {
        concat!("[", $m0, "]")
    };
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "usmmla v", $r3, ".4s", ", v", $r2,".16b, v", $r1, ".16b\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($l:tt, $m0:expr, $r1:expr) => {
        concat!(
            "ldr q", $r1, ", ", $m0, "\n",
        )
    };
}

macro_rules! storep_unit {
    ($l:tt, $r1:expr, $m0:expr) => {
        concat!(
            "str q", $r1, ", ", $m0,  "\n",
        )
    };
}

macro_rules! asm_alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "cmp {alpha_st:w}, #0", "\n",
                "BEQ 13f", "\n",

                "ldr s1, [{alphax}]", "\n",

                #(
                    "scvtf v", r, ".4s, v", r, ".4s\n",
                    "fmul  v", r, ".4s, v", r, ".4s, v1.s[0]\n",
                    "fcvtzs v", r, ".4s, v", r, ".4s\n",
                )*

                "13:", "\n",
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "ldr s0, [{betax}]", "\n",
            "/* {betax} */", "\n",
        )
    }
}


macro_rules! acc_p {
    (C, $m0:expr, $r1:expr, $r2:expr, $idx:tt) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $idx),
            beta_fmadd!(C, mem!($m0, "0x10"), $r2, $idx),
        )
    };

    (M, $m0:expr, $r1:expr, $r2:expr, $idx:tt) => {
        concat!(
            beta_fmadd!(M, mem!($m0), $r1, $idx),
            beta_fmadd!(M, mem!($m0, "0x10"), $r2, $idx),
        )
    };

    (C, $m0:expr, $r1:expr, $idx:tt) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1, $idx),
        )
    };

    (M, $m0:expr, $r1:expr, $idx:tt) => {
        concat!(
            beta_fmadd!(M, mem!($m0), $r1, $idx),
        )
    };
}


macro_rules! loadp {
    (2, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, mem!($m0), 0),
            loadp_unit!($layout, mem!($m0, "0x10"), 1),
            loadp_unit!($layout, mem!($m0, "0x20"), 2),
            loadp_unit!($layout, mem!($m0, "0x30"), 3),
        )
    };

    (1, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, mem!($m0), 0),
            loadp_unit!($layout, mem!($m0, "0x10"), 1),
        )
    };
}

macro_rules! storep {
    (C, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            storep_unit!(C, $r1, mem!($m0)),
            storep_unit!(C, $r2, mem!($m0, "0x10")),
        )
    };

    (M, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            storep_unit!(M, $r1, mem!($m0)),
            storep_unit!(M, $r2, mem!($m0, "0x10")),
        )
    };

    (C, $m0:expr, $r1:expr) => {
        concat!(
            storep_unit!(C, $r1, mem!($m0)),
        )
    };

    (M, $m0:expr, $r1:expr) => {
        concat!(
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


macro_rules! asm_init_ab {
    ($KER:tt,B,B) => {
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
    (12) => {
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


macro_rules! asm_vzeroall {
    ($mr:tt,$nr:tt) => {vzeroall!(8,31)};
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


macro_rules! asm_alpha_scale {
    ($mr:tt, $nr:tt) => {
        asm_alpha_scale_0!(8,31)
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


macro_rules! c_reg_1x12 {
    (0,0) => { 8 };
    (0,1) => { 9 };

    (0,2) => { 12 };
    (0,3) => { 13 };

    (0,4) => { 16 };
    (0,5) => { 17 };

    (0,6) => { 20 };
    (0,7) => { 21 };

    (0,8) => { 24 };
    (0,9) => { 25 };

    (0,10) => { 28 };
    (0,11) => { 29 };
}


macro_rules! acc_1x12 {
    ($ni:tt, $layout:tt, $idx:tt) => {
        acc_p!(
            $layout, c_mem!($ni), c_reg_1x12!(0,$ni), $idx
        )
    };
}

macro_rules! store_1x12 {
    ($ni:tt, $layout:tt) => {
        storep!(
            $layout, c_mem!($ni), c_reg_1x12!(0,$ni)
        )
    };
}


macro_rules! cum_seq {
    ($step_macro:tt, $nr:tt, $layout:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!(n, $layout),)*)
        })
    };
    ($step_macro:tt, $nr:tt, $layout:tt, $idx:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!(n, $layout,$idx),)*)
        })
    };
}

macro_rules! load_b {
    (B, 0) => {
        concat!(
            "ld1 {{v4.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 2) => {
        concat!(
            "ld1 {{v5.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 4) => {
        concat!(
            "ld1 {{v6.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 6) => {
        concat!(
            "ld1 {{v7.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 8) => {
        concat!(
            "ld1 {{v4.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 10) => {
        concat!(
            "ld1 {{v5.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };

    (B, $nr:tt) => {
        "add {bx}, {bx}, #8 \n"
    };
}


macro_rules! load_a {
    ($mr:tt, B) => {
        loadp!($mr, B, "{ax}")
    };
    ($mr:tt, C) => {
        loadp!($mr, C, "[{ax}]")
    };
}

macro_rules! fmadd_3v2 {
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
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2, $a_layout),
                "add {ax}, {ax}, #64 \n",
                #(
                    load_b!($b_layout, n),
                    fmadd_3v2!(n),
                )*
                // "add {bx}, {bx}, #256 \n",
                inc_b!($b_layout,$nr), 
            )
        })
    };
}


macro_rules! fmadd_1v2 {
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
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(1, $a_layout),
                "add {ax}, {ax}, #32 \n",
                #(
                    load_b!($b_layout, n),
                    fmadd_1v2!(n),
                )*
                // "add {bx}, {bx}, #256 \n",
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

use crate::UnaryFnC;

macro_rules! prefetch_c {
    (2, $nr:tt) => {
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
            "prfm pldl1keep, [{cx},#56]\n",
            "prfm pldl1keep, [{x1}] \n",
            "prfm pldl1keep, [{x1},#56]\n",
            "prfm pldl1keep, [{x2}] \n",
            "prfm pldl1keep, [{x2},#56]\n",
            "prfm pldl1keep, [{x3}] \n",
            "prfm pldl1keep, [{x3},#56]\n",
            "prfm pldl1keep, [{x4}] \n",
            "prfm pldl1keep, [{x4},#56]\n",
            "prfm pldl1keep, [{x5}] \n",
            "prfm pldl1keep, [{x5},#56]\n",
            "prfm pldl1keep, [{x6}] \n",
            "prfm pldl1keep, [{x6},#56]\n",
            "prfm pldl1keep, [{x7}] \n",
            "prfm pldl1keep, [{x7},#56]\n",
            "prfm pldl1keep, [{x8}] \n",
            "prfm pldl1keep, [{x8},#56]\n",
            "prfm pldl1keep, [{x9}] \n",
            "prfm pldl1keep, [{x9},#56]\n",
            "prfm pldl1keep, [{x10}] \n",
            "prfm pldl1keep, [{x10},#56]\n",
            "prfm pldl1keep, [{x11}] \n",
            "prfm pldl1keep, [{x11},#56]\n",
        )
    };
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
        #[target_feature(enable="neon,i8mm")]
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const f32, beta: *const f32,
            k: usize,
            d_arr: [usize; 4],
            m: usize,
            f: F,
        ) {
            const MR: usize = $mr * 4;
            let (k_i, k_l) = (k / 32, (k % 32) / 8);
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*TC_SIZE, k_i, k_l];
            let mut cf = c;
            let c_cs = d_arr[3];
            let mut c_buf = [0i32; MR * $nr];
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
            if BUF || m != MR {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, MR);
                dim_arr[2] = MR*TC_SIZE;
                cf = c_buf.as_mut_ptr();
            }
            asm!(
                asm_vzeroall!($mr,$nr),

                prefetch_c!(2,12),
        
                asm_init_ab!($mr,$a_layout,$b_layout),
                
                // 3 -> CONSIDKLEFT
                "BEQ 3f",
                
                // 2 -> KITER
                "2:",
                prefetch_0!(128, "{bx}"),
                $step_macro!($nr, $a_layout, $b_layout),
                $step_macro!($nr, $a_layout, $b_layout),
                $step_macro!($nr, $a_layout, $b_layout),
                $step_macro!($nr, $a_layout, $b_layout),
        
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
                $step_macro!($nr, $a_layout, $b_layout),

                "sub {x0}, {x0}, #1",
        
                // 4 -> KLEFT
                "cmp {x0}, 0",
                "BNE 4b",
        
                // 5 -> POSTACCUM
                "5:",
                asm_c_load!($nr),
                // scale by alpha
                "/* {alphax} */",
                asm_alpha_scale!($mr, $nr),

                unzip_c!(),

                "cmp {beta_st:w}, #0", "\n",
                "BEQ 6f",
        
                "cmp {beta_st:w}, #1", "\n",
                "BEQ 9f",

                // 6 -> BETAZERO
                load_beta!(),
                cum_seq!($acc_macro,$nr,$is_partial,2),
                "B 6f",

                "9:",
                // 9 -> BETAONE
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
                betax = inout(reg) beta => _,
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
            if BUF || m != MR {
                for j in 0..$nr {
                    f.call(cf.add(j*MR), MR);
                }
                store_buf(c, d_arr[2], c_cs, &c_buf, m, $nr, MR);
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
        #[target_feature(enable="neon,i8mm")]
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const f32, beta: *const f32,
            k: usize,
            d_arr: [usize; 4],
            m: usize, n: usize,
            f: F,
        ) {
            const MR: usize = $mr * 4;
            let (k_i, k_l) = (k / 32, (k % 32) / 8);
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*TC_SIZE, k_i, k_l];
            let mut cf = c;
            let c_cs = d_arr[3];
            let mut c_buf = [0i32; MR * $nr];
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
            if BUF || m != MR {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, MR);
                dim_arr[2] = MR*TC_SIZE;
                cf = c_buf.as_mut_ptr();
            }
            let _ = 'blk: {
                seq!(ni in 1..$nr {
                    if ni == n {
                        asm!(
                            prefetch_c!(2,12),
                            asm_vzeroall!($mr,ni),
                
                            asm_init_ab!($mr,$a_layout,$b_layout),
                        
                            // 3 -> CONSIDKLEFT
                            "BEQ 3f",
                        
                            // 2 -> KITER
                            "2:",
                            prefetch_0!(128, "{bx}"),
                            $step_macro!(ni, $a_layout, $b_layout),
                            $step_macro!(ni, $a_layout, $b_layout),
                            $step_macro!(ni, $a_layout, $b_layout),
                            $step_macro!(ni, $a_layout, $b_layout),
                
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
                            $step_macro!(ni, $a_layout, $b_layout),

                            "sub {x0}, {x0}, #1",
                
                            // 4 -> KLEFT
                            "cmp {x0}, 0",
                            "BNE 4b",
                
                            // 5 -> POSTACCUM
                            "5:",
                            asm_c_load!(ni),
                            // scale by alpha
                            "/* {alphax} */",
                            asm_alpha_scale!($mr, ni),

                            unzip_c!(),

                            "cmp {beta_st:w}, #0", "\n",
                            "BEQ 6f",
                    
                            "cmp {beta_st:w}, #1", "\n",
                            "BEQ 9f",

                            // 6 -> BETAZERO
                            load_beta!(),
                            cum_seq!($acc_macro,ni,$is_partial,2),
                            "B 6f",

                            "9:",
                            // 9 -> BETAONE
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
                            betax = inout(reg) beta => _,
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
                        break 'blk;
                    }
                });
            };
            if BUF || m != MR {
                for j in 0..n {
                    f.call(cf.add(j*MR), MR);
                }
                store_buf(c, d_arr[2], c_cs, &c_buf, m, n, MR);
            } else {
                for j in 0..n {
                    f.call(cf.add(j*c_cs), m);
                }
            }
        }
    };
}

def_ukernel!(step_2x12, acc_2x12, store_2x12, 2, 12, B, B, M, ukernel_2_bb_partial);
def_ukernel!(step_1x12, acc_1x12, store_1x12, 1, 12, B, B, M, ukernel_1_bb_partial);

def_ukernelxn!(step_2x12, acc_2x12, store_2x12, 2, 12, B, B, C, ukernel_n_bb);

def_ukernelxn!(step_2x12, acc_2x12, store_2x12, 2, 12, B, B, M, ukernel_2xn_bb_partial);
def_ukernelxn!(step_1x12, acc_1x12, store_1x12, 1, 12, B, B, M, ukernel_1xn_bb_partial);


#[target_feature(enable="neon,i8mm")]
pub(crate) unsafe fn ukernel_bb<F: UnaryFnC, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const f32, beta: *const f32,
    k: usize,
    d_arr: [usize; 4],
    m: usize,
    f: F,
) {
    const MR: usize = 2 * 4;
    let (k_i, k_l) = (k / 32, (k % 32) / 8);
    let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*TC_SIZE, k_i, k_l];
    let mut cf = c;
    let mut c_buf = [0i32; MR * 12];
    let c_cs = d_arr[3];
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
        load_buf(c, d_arr[2], c_cs, &mut c_buf, m, 12, MR);
        dim_arr[2] = MR*TC_SIZE;
        cf = c_buf.as_mut_ptr();
    }
    asm!(
        asm_vzeroall!(2,12),

        prefetch_c!(2,12),

        asm_init_ab!(12,B,B),
        
        // 3 -> CONSIDKLEFT
        "BEQ 3f",
        
        // 2 -> KITER
        "2:",
        prefetch_0!(256, "{bx}"),
        step_2x12!(12, B, B),
        step_2x12!(12, B, B),
        prefetch_0!(256, "{bx}"),
        step_2x12!(12, B, B),
        step_2x12!(12, B, B),

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
        step_2x12!(12, B, B),

        "sub {x0}, {x0}, #1",

        // 4 -> KLEFT
        "cmp {x0}, 0",
        "BNE 4b",

        // 5 -> POSTACCUM
        "5:",
        asm_c_load!(12),
        // scale by alpha
        "/* {alphax} */",
        asm_alpha_scale!(2,12),

        unzip_c!(),

        "cmp {beta_st:w}, #0", "\n",
        "BEQ 6f",

        "cmp {beta_st:w}, #1", "\n",
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
            f.call(cf.add(j*MR), MR);
        }
        store_buf(c, d_arr[2], c_cs, &c_buf, m, 12, MR);
    } else {
        for j in 0..12 {
            f.call(cf.add(j*c_cs), m);
        }
    }
}