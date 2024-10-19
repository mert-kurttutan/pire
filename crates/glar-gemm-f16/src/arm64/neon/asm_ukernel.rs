use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC, TC_SIZE};
use glar_base::{load_buf, store_buf, c_mem, prefetch_0};
use half::f16;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "fmla v", $r1, ".8h, v1.8h, v0.h[0]\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("dup v", r, ".8h, wzr \n",)*)
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
    ($r1:expr, $r2:expr, $r3:expr, $i:expr) => {
        concat!(
            "fmla v", $r3, ".8h", ", v", $r1,".8h, v", $r2, ".h[", $i, "]\n",
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

macro_rules! asm_alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "ldr h1, [{alphax}]", "\n",
                #(
                    "fmul  v", r, ".8h, v", r, ".8h, v1.h[0]\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "ldr h0, [{betax}]", "\n",
            "/* {betax} */", "\n",

            "fcmp h0,#0.0", "\n",
        )
    }
}


macro_rules! acc_p {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr, $r5:expr, $r6:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1),
            beta_fmadd!(C, mem!($m0, "0x10"), $r2),
            beta_fmadd!(C, mem!($m0, "0x20"), $r3),
            beta_fmadd!(C, mem!($m0, "0x30"), $r4),
            beta_fmadd!(C, mem!($m0, "0x40"), $r5),
            beta_fmadd!(C, mem!($m0, "0x50"), $r6),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1),
            beta_fmadd!(C, mem!($m0, "0x10"), $r2),
            beta_fmadd!(C, mem!($m0, "0x20"), $r3),
            beta_fmadd!(C, mem!($m0, "0x30"), $r4),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1),
            beta_fmadd!(C, mem!($m0, "0x10"), $r2),
        )
    };
}


macro_rules! loadp {
    (48, $layout:tt, $m0:expr, $b0:expr) => {
        concat!(
            loadp_unit!($layout, mem!($m0, $b0), 0),
            loadp_unit!($layout, mem!($m0, $b0, "0x10"), 1),
            loadp_unit!($layout, mem!($m0, $b0, "0x20"), 2),
            loadp_unit!($layout, mem!($m0, $b0, "0x30"), 3),
            loadp_unit!($layout, mem!($m0, $b0, "0x40"), 4),
            loadp_unit!($layout, mem!($m0, $b0, "0x50"), 5),
        )
    };
    (32, $layout:tt, $m0:expr, $b0:expr) => {
        concat!(
            loadp_unit!($layout, mem!($m0, $b0), 0),
            loadp_unit!($layout, mem!($m0, $b0, "0x10"), 1),
            loadp_unit!($layout, mem!($m0, $b0, "0x20"), 2),
            loadp_unit!($layout, mem!($m0, $b0, "0x30"), 3),
        )
    };
    (16, $layout:tt, $m0:expr, $b0:expr) => {
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
            "add {ax}, {ax}, #2*", $K, "*", $X, " \n",
        )
    };
}

macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => {
        ""
    };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add {bx}, {bx}, #2*", $K, "*", $X, " \n",
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
    ($ni:tt, $layout:tt) => {
        acc_p!(
            $layout, c_mem!($ni), c_reg_3x4!(0,$ni), c_reg_3x4!(1,$ni), c_reg_3x4!(2,$ni), c_reg_3x4!(3,$ni), c_reg_3x4!(4,$ni), c_reg_3x4!(5,$ni)
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
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_2x6!(0,$ni), c_reg_2x6!(1,$ni), c_reg_2x6!(2,$ni), c_reg_2x6!(3,$ni))
    };
}

macro_rules! store_2x6 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_2x6!(0,$ni), c_reg_2x6!(1,$ni), c_reg_2x6!(2,$ni), c_reg_2x6!(3,$ni))
    };
}

macro_rules! acc_1x6 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_1x6!(0,$ni), c_reg_1x6!(1,$ni))
    };
}

macro_rules! store_1x6 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_1x6!(0,$ni), c_reg_1x6!(1,$ni))
    };
}



macro_rules! cum_seq {
    ($step_macro:tt, $nr:tt, $layout:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!(n, $layout),)*)
        })
    };
}

macro_rules! load_b {
    (B, 0, $K:tt, $X:tt, $r:expr) => {
        concat!(
            "ldr d", $r, ", [{bx}, #", $K, "*", $X, "*2]", "\n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt, B, $K:tt) => {
        loadp!($mr, B, "{ax}", concat!($mr,"*2*",$K))
    };
    ($mr:tt, C, $K:tt) => {
        loadp!($mr, C, "[{ax}]")
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
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(48, $a_layout, $K),
                load_b!($b_layout, 0, $K, $nr, 6),
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
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(32, $a_layout, $K),
                load_b!($b_layout, 0, $K, $nr, 4),
                load_b!($b_layout, 0, $K, $nr, 5),
                #(
                    fmadd_2v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 1x6 ******************************* //
macro_rules! step_1x6 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(16, $a_layout, $K),
                load_b!($b_layout, 0, $K, $nr, 2),
                load_b!($b_layout, 0, $K, $nr, 3),
                #(
                    fmadd_1v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
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
            "prfm pldl1keep, [{x2}] \n",
            "prfm pldl1keep, [{x2},#64]\n",
            "prfm pldl1keep, [{x2},#96]\n",
            "prfm pldl1keep, [{x3}] \n",
            "prfm pldl1keep, [{x3},#64]\n",
            "prfm pldl1keep, [{x3},#96]\n",
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
        #[target_feature(enable = "fp16")]
        pub(crate) unsafe fn $func_name<F: MyFn, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize,
            f: F,
        ) {
            let (k_i, k_l) = (k / 4, k % 4);
            let mut dim_arr = [d_arr[0]*2, d_arr[1]*2, d_arr[3]*TC_SIZE, k_i, k_l];
            let mut cf = c;
            let mut c_buf = [f16::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF || m != $mr {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, $mr);
                dim_arr[2] = $mr*TC_SIZE;
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
                prefetch_0!(128, "{bx}"),
                $step_macro!($nr, $a_layout, $b_layout, 0),
                $step_macro!($nr, $a_layout, $b_layout, 1),
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
                asm_c_load!($nr),
                // scale by alpha
                asm_alpha_scale!($mr, $nr),

                load_beta!(),

                // 6 -> BETAZERO
                "BEQ 6f",
                cum_seq!($acc_macro,$nr,$is_partial),

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
                x0 = out(reg) _,
                x1 = out(reg) _,
                x2 = out(reg) _,
                x3 = out(reg) _,
                x4 = out(reg) _,
                x5 = out(reg) _,
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
        #[target_feature(enable = "fp16")]
        pub(crate) unsafe fn $func_name<F: MyFn, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize, n: usize,
            f: F,
        ) {
            let (k_i, k_l) = (k / 4, k % 4);
            let mut dim_arr = [d_arr[0]*2, d_arr[1]*2, d_arr[3]*TC_SIZE, k_i, k_l];
            let mut cf = c;
            let mut c_buf = [f16::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF || m != $mr {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, $mr);
                dim_arr[2] = $mr*TC_SIZE;
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
                            prefetch_0!(128, "{cx}"),
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
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),

                            load_beta!(),

                            // 6 -> BETAZERO
                            "BEQ 6f",
                            cum_seq!($acc_macro,ni,$is_partial),

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
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            x4 = out(reg) _,
                            x5 = out(reg) _,
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

def_ukernel!(step_3x4, acc_3x4, store_3x4, 48, 4, B, B, C, ukernel_bb);

def_ukernel!(step_3x4, acc_3x4, store_3x4, 48, 4, B, B, C, ukernel_3_bb_partial);
def_ukernel!(step_2x6, acc_2x6, store_2x6, 32, 4, B, B, C, ukernel_2_bb_partial);
def_ukernel!(step_1x6, acc_1x6, store_1x6, 16, 4, B, B, C, ukernel_1_bb_partial);

def_ukernelxn!(step_3x4, acc_3x4, store_3x4, 48, 4, B, B, C, ukernel_n_bb);

def_ukernelxn!(step_3x4, acc_3x4, store_3x4, 48, 4, B, B, C, ukernel_3xn_bb_partial);
def_ukernelxn!(step_2x6, acc_2x6, store_2x6, 32, 4, B, B, C, ukernel_2xn_bb_partial);
def_ukernelxn!(step_1x6, acc_1x6, store_1x6, 16, 4, B, B, C, ukernel_1xn_bb_partial);
