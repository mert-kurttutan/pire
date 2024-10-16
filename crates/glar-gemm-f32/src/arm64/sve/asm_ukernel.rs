use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC};
use glar_base::{load_buf, store_buf, c_mem2};

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "ld1w {{ z1.s }}, p0/z, ", $m0, "\n",
            "fmla z", $r1, ".s, z1.s, z0.s[0]\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "ld1w {{ z1.s }}, p1/z, ", $m0, "\n",
            "fmla z", $r1, ".s, z1.s, z0.s[0]\n",
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
    ($r1:expr, $r2:expr, $r3:expr, $i:expr) => {
        concat!(
            "fmla z", $r3, ".s", ", z", $r1,".s, z", $r2, ".s[", $i, "]\n",
        ) 
    };
}

macro_rules! loadp_unit {
    (B, $m0:expr, $r1:expr) => {
        concat!(
            "ld1w {{ z", $r1, ".s }}, p0/z, ", $m0, "\n",
        )
    };
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "ld1w {{ z", $r1, ".s }}, p0/z, ", $m0, "\n",
        )
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "ld1w {{ z", $r1, ".s }}, p1/z, ", $m0, "\n",
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

macro_rules! asm_alpha_scale_0 {
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
    (C, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, mem!($m0), $r1),
            beta_fmadd!(C, mem!($m0, "#1", "MUL VL"), $r2),
            beta_fmadd!(C, mem!($m0, "#2", "MUL VL"), $r3),
        )
    };

    (M, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "whilelo p1.s, {m_s}, {m_e0}", "\n",
            beta_fmadd!(M, mem!($m0), $r1),
            "whilelo p1.s, {m_s}, {m_e1}", "\n",
            beta_fmadd!(M, mem!($m0, "#1", "MUL VL"), $r2),
            "whilelo p1.s, {m_s}, {m_e2}", "\n",
            beta_fmadd!(M, mem!($m0, "#2", "MUL VL"), $r3),
        )
    };
}


macro_rules! loadp {
    (48, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, mem!($m0), 0),
            loadp_unit!($layout, mem!($m0, "#1", "MUL VL"), 1),
            loadp_unit!($layout, mem!($m0, "#2", "MUL VL"), 2),
            loadp_unit!($layout, mem!($m0, "#3", "MUL VL"), 3),
            loadp_unit!($layout, mem!($m0, "#4", "MUL VL"), 4),
            loadp_unit!($layout, mem!($m0, "#5", "MUL VL"), 5),
        )
    };
    (24, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, mem!($m0), 0),
            loadp_unit!($layout, mem!($m0, "#1", "MUL VL"), 1),
            loadp_unit!($layout, mem!($m0, "#2", "MUL VL"), 2),
        )
    };
}

macro_rules! storep {
    (C, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            storep_unit!(C, $r1, mem!($m0)),
            storep_unit!(C, $r2, mem!($m0, "#1", "MUL VL")),
            storep_unit!(C, $r3, mem!($m0, "#2", "MUL VL")),
        )
    };
    (M, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "whilelo p1.s, {m_s}, {m_e0}", "\n",
            storep_unit!(M, $r1, mem!($m0)),
            "whilelo p1.s, {m_s}, {m_e1}", "\n",
            storep_unit!(M, $r2, mem!($m0, "#1", "MUL VL")),
            "whilelo p1.s, {m_s}, {m_e2}", "\n",
            storep_unit!(M, $r3, mem!($m0, "#2", "MUL VL")),
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

macro_rules! c_reg_24x8 {
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

macro_rules! acc_24x8 {
    ($ni:tt, $layout:tt) => {
        acc_p!(
            $layout, c_mem2!($ni), c_reg_24x8!(0,$ni), c_reg_24x8!(1,$ni), c_reg_24x8!(2,$ni)
        )
    };
}

macro_rules! store_24x8 {
    ($ni:tt, $layout:tt) => {
        storep!(
            $layout, c_mem2!($ni), c_reg_24x8!(0,$ni), c_reg_24x8!(1,$ni), c_reg_24x8!(2,$ni)
        )
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
    (B, 1) => {
        concat!(
            "ld1rqw {{ z5.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*4 \n",
        )
    };
    (B, 2) => {
        concat!(
            "ld1rqw {{ z5.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*4 \n",
        )
    };
    (B, 3) => {
        concat!(
            "ld1rqw {{ z5.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #3*4 \n",
        )
    };
    (B, 4) => {
        concat!(
            "ld1rqw {{ z5.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
        )
    };
    (B, 5) => {
        concat!(
            "ld1rqw {{ z5.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
            "ld1rqw {{ z6.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*4 \n",
        )
    };
    (B, 6) => {
        concat!(
            "ld1rqw {{ z5.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
            "ld1rqw {{ z6.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*4 \n",
        )
    };
    (B, 7) => {
        concat!(
            "ld1rqw {{ z5.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
            "ld1rqw {{ z6.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #3*4 \n",
        )
    };
    (B, 8) => {
        concat!(
            "ld1rqw {{ z5.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
            "ld1rqw {{ z6.s }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*4 \n",
        )
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
            vfmadd!(0, 5, 8, 0),
            vfmadd!(1, 5, 9, 0),
            vfmadd!(2, 5, 10, 0),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 5, 11, 1),
            vfmadd!(1, 5, 12, 1),
            vfmadd!(2, 5, 13, 1),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 5, 14, 2),
            vfmadd!(1, 5, 15, 2),
            vfmadd!(2, 5, 16, 2),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 5, 17, 3),
            vfmadd!(1, 5, 18, 3),
            vfmadd!(2, 5, 19, 3),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 6, 20, 0),
            vfmadd!(1, 6, 21, 0),
            vfmadd!(2, 6, 22, 0),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 6, 23, 1),
            vfmadd!(1, 6, 24, 1),
            vfmadd!(2, 6, 25, 1),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 6, 26, 2),
            vfmadd!(1, 6, 27, 2),
            vfmadd!(2, 6, 28, 2),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 6, 29, 3),
            vfmadd!(1, 6, 30, 3),
            vfmadd!(2, 6, 31, 3),
        )
    };

    (0,M) => {
        concat!(
            vfmadd!(0, 5, 8, 0),
            vfmadd!(1, 5, 9, 0),
            vfmadd!(2, 5, 10, 0),
        )
    };
    (1,M) => {
        concat!(
            vfmadd!(0, 5, 11, 1),
            vfmadd!(1, 5, 12, 1),
            vfmadd!(2, 5, 13, 1),
        )
    };
    (2,M) => {
        concat!(
            vfmadd!(0, 5, 14, 2),
            vfmadd!(1, 5, 15, 2),
            vfmadd!(2, 5, 16, 2),
        )
    };
    (3,M) => {
        concat!(
            vfmadd!(0, 5, 17, 3),
            vfmadd!(1, 5, 18, 3),
            vfmadd!(2, 5, 19, 3),
        )
    };
    (4,M) => {
        concat!(
            vfmadd!(0, 6, 20, 0),
            vfmadd!(1, 6, 21, 0),
            vfmadd!(2, 6, 22, 0),
        )
    };
    (5,M) => {
        concat!(
            vfmadd!(0, 6, 23, 1),
            vfmadd!(1, 6, 24, 1),
            vfmadd!(2, 6, 25, 1),
        )
    };
    (6,M) => {
        concat!(
            vfmadd!(0, 6, 26, 2),
            vfmadd!(1, 6, 27, 2),
            vfmadd!(2, 6, 28, 2),
        )
    };
    (7,M) => {
        concat!(
            vfmadd!(0, 6, 29, 3),
            vfmadd!(1, 6, 30, 3),
            vfmadd!(2, 6, 31, 3),
        )
    };
}


macro_rules! step_24x8 {
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(24, $a_layout),
                "add {ax}, {ax}, #4*24 \n",
                load_b!($b_layout, $nr),
                #(
                    fmadd_3v2!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
    ($nr:tt, $a_layout:tt, $b_layout:tt, M) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(24, $a_layout),
                "add {ax}, {ax}, #4*24 \n",
                load_b!($b_layout, $nr),
                #(
                    fmadd_3v2!(n,M),
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
    (48, 4) => {
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
    (24, $nr:tt) => {
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

#[inline(always)]
unsafe fn sve_vs() -> usize {
    8
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
        #[target_feature(enable="neon,sve")]
        pub(crate) unsafe fn $func_name<F: MyFn, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize,
            f: F,
        ) {
            let vs = sve_vs();
            // let m_l = m % vs;
            // let m_l = if m_l == 0 { vs } else { m_l };
            let k_iter = k / 4;
            let k_left = k % 4;
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*4, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [0f32;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, $mr);
                dim_arr[2] = $mr*4;
                cf = c_buf.as_mut_ptr();
            }
            asm!(
                "ptrue p0.s",
                "/* {m_s} */", "\n",
                "/* {m_e0} */", "\n",
                "/* {m_e1} */", "\n",
                "/* {m_e2} */", "\n",
                asm_vzeroall!($mr,$nr),

                prefetch_c!(24,4),
        
                asm_init_ab!($mr,$a_layout,$b_layout),
                
                // 3 -> CONSIDKLEFT
                "BEQ 3f",
                
                // 2 -> KITER
                "2:",
                prefetch_0!(128, "{bx}", 0),
                $step_macro!($nr, $a_layout, $b_layout,M),
                $step_macro!($nr, $a_layout, $b_layout,M),
                $step_macro!($nr, $a_layout, $b_layout,M),
                $step_macro!($nr, $a_layout, $b_layout,M),
        
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
                $step_macro!($nr, $a_layout, $b_layout,M),

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
                m_s = in(reg) 0 as u64,
                m_e0 = in(reg) (m) as u64,
                m_e1 = in(reg) (m - vs.min(m)) as u64,
                m_e2 = in(reg) (m - (2*vs).min(m)) as u64,
                x0 = out(reg) _,
                x1 = out(reg) _,
                x2 = out(reg) _,
                x3 = out(reg) _,
                x4 = out(reg) _,
                x5 = out(reg) _,
                x6 = out(reg) _,
                x7 = out(reg) _,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            );
            if BUF {
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
        #[target_feature(enable="neon,sve")]
        pub(crate) unsafe fn $func_name<F: MyFn, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize, n: usize,
            f: F,
        ) {
            let vs = sve_vs();
            let k_iter = k / 4;
            let k_left = k % 4;
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*4, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [0f32;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, $mr);
                dim_arr[2] = $mr*4;
                cf = c_buf.as_mut_ptr();
            }
            let _ = 'blk: {
                seq!(ni in 1..$nr {
                    if ni == n {
                        asm!(
                            "ptrue p0.s",
                            "/* {m_s} */", "\n",
                            "/* {m_e0} */", "\n",
                            "/* {m_e1} */", "\n",
                            "/* {m_e2} */", "\n",
                            prefetch_c!(24,4),
                            asm_vzeroall!($mr,ni),
                
                            asm_init_ab!($mr,$a_layout,$b_layout),
                        
                            // 3 -> CONSIDKLEFT
                            "BEQ 3f",
                        
                            // 2 -> KITER
                            "2:",
                            prefetch_0!(128, "{bx}", 0),
                            $step_macro!(ni, $a_layout, $b_layout,M),
                            $step_macro!(ni, $a_layout, $b_layout,M),
                            $step_macro!(ni, $a_layout, $b_layout,M),
                            $step_macro!(ni, $a_layout, $b_layout,M),
                
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
                            $step_macro!(ni, $a_layout, $b_layout,M),

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
                            m_s = in(reg) 0 as u64,
                            m_e0 = in(reg) (m) as u64,
                            m_e1 = in(reg) (m - vs.min(m)) as u64,
                            m_e2 = in(reg) (m - (2*vs).min(m)) as u64,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            x4 = out(reg) _,
                            x5 = out(reg) _,
                            x6 = out(reg) _,
                            x7 = out(reg) _,
                            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                            out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                            out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                        );
                        break 'blk;
                    }
                });
            };
            if BUF {
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

def_ukernel!(step_24x8, acc_24x8, store_24x8, 24, 8, B, B, M, ukernel_24x8_bb_partial);


def_ukernelxn!(step_24x8, acc_24x8, store_24x8, 24, 8, B, B, C, ukernel_24xn_bb);

def_ukernelxn!(step_24x8, acc_24x8, store_24x8, 24, 8, B, B, M, ukernel_24xn_bb_partial);



#[target_feature(enable="neon,sve")]
pub(crate) unsafe fn ukernel_24x8_bb<F: MyFn, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const TA, beta: *const TB,
    k: usize,
    d_arr: [usize; 4],
    m: usize,
    f: F,
) {
    let k_iter = k / 4;
    let k_left = k % 4;
    let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*4, k_iter, k_left];
    let mut cf = c;
    let mut c_buf = [0f32; 2048 * 3 * 4];
    let c_cs = d_arr[3];
    if BUF {
        let mr = sve_vs() * 3;
        load_buf(c, d_arr[2], c_cs, &mut c_buf, m, 8, mr);
        dim_arr[2] = mr*4;
        cf = c_buf.as_mut_ptr();
    }
    asm!(
        "ptrue p0.s",
        "ptrue p1.s",
        asm_vzeroall!(24,8),

        prefetch_c!(24,8),

        asm_init_ab!(24,B,B),
        
        // 3 -> CONSIDKLEFT
        "BEQ 3f",
        
        // 2 -> KITER
        "2:",
        prefetch_0!(128, "{bx}", 0),
        step_24x8!(8, B, B),
        step_24x8!(8, B, B),
        step_24x8!(8, B, B),
        step_24x8!(8, B, B),

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
        step_24x8!(8, B, B),

        "sub {x0}, {x0}, #1",

        // 4 -> KLEFT
        "cmp {x0}, 0",
        "BNE 4b",

        // 5 -> POSTACCUM
        "5:",
        asm_c_load!(8),
        // scale by alpha
        asm_alpha_scale!(24,8),

        load_beta!(),

        // 6 -> BETAZERO
        "BEQ 6f",
        cum_seq!(acc_24x8,8,C),

        // 6 -> BETAZERO
        "6:",
        cum_seq!(store_24x8,8,C),
        
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
        x6 = out(reg) _,
        x7 = out(reg) _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
    );
    if BUF {
        let mr = sve_vs() * 3;
        for j in 0..4 {
            f.call(cf.add(j*mr), mr);
        }
        store_buf(c, d_arr[2], c_cs, &c_buf, m, 8, mr);
    } else {
        for j in 0..8 {
            f.call(cf.add(j*c_cs), m);
        }
    }
}