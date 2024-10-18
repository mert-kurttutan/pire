use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC};
use glar_base::{load_buf, store_buf, c_mem, prefetch_0};

use half::f16;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "ld1h {{ z1.h }}, p0/z, ", $m0, "\n",
            "fmla z", $r1, ".h, z1.h, z0.h[0]\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "ld1h {{ z1.h }}, p1/z, ", $m0, "\n",
            "fmla z", $r1, ".h, z1.h, z0.h[0]\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("dup z", r, ".h, #0 \n",)*)
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
            "fmla z", $r3, ".h", ", z", $r1,".h, z", $r2, ".h[", $i, "]\n",
        ) 
    };
}

macro_rules! loadp_unit {
    (B, $m0:expr, $r1:expr) => {
        concat!(
            "ld1h {{ z", $r1, ".h }}, p0/z, ", $m0, "\n",
        )
    };
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "ld1h {{ z", $r1, ".h }}, p0/z, ", $m0, "\n",
        )
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "ld1h {{ z", $r1, ".h }}, p1/z, ", $m0, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "st1h {{ z", $r1, ".h }}, p0, ", $m0, "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
            "st1h {{ z", $r1, ".h }}, p1, ", $m0, "\n",
        )
    };
}

macro_rules! asm_alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "ld1rqh {{ z1.h }}, p0/z, [{alphax}]", "\n",
                #(
                    "fmul  z", r, ".h, z", r, ".h, z1.h[0]\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "ld1rqh {{ z0.h }}, p0/z, [{betax}]", "\n",
            "/* {betax} */", "\n",

            "fcmp h0,#0.0", "\n",
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
            "whilelo p1.h, {m_s}, {m_e0}", "\n",
            beta_fmadd!(M, mem!($m0), $r1),
            "whilelo p1.h, {m_s}, {m_e1}", "\n",
            beta_fmadd!(M, mem!($m0, "#1", "MUL VL"), $r2),
            "whilelo p1.h, {m_s}, {m_e2}", "\n",
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
            "whilelo p1.h, {m_s}, {m_e0}", "\n",
            storep_unit!(M, $r1, mem!($m0)),
            "whilelo p1.h, {m_s}, {m_e1}", "\n",
            storep_unit!(M, $r2, mem!($m0, "#1", "MUL VL")),
            "whilelo p1.h, {m_s}, {m_e2}", "\n",
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

macro_rules! c_reg_3x8 {
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

macro_rules! acc_3x8 {
    ($ni:tt, $layout:tt) => {
        acc_p!(
            $layout, c_mem!($ni), c_reg_3x8!(0,$ni), c_reg_3x8!(1,$ni), c_reg_3x8!(2,$ni)
        )
    };
}

macro_rules! store_3x8 {
    ($ni:tt, $layout:tt) => {
        storep!(
            $layout, c_mem!($ni), c_reg_3x8!(0,$ni), c_reg_3x8!(1,$ni), c_reg_3x8!(2,$ni)
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
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #1*2 \n",
        )
    };
    (B, 2) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #2*2 \n",
        )
    };
    (B, 3) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #3*2 \n",
        )
    };
    (B, 4) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #4*2 \n",
        )
    };
    (B, 5) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #5*2 \n",
        )
    };
    (B, 6) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #6*2 \n",
        )
    };
    (B, 7) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #7*2 \n",
        )
    };
    (B, 8) => {
        concat!(
            "ld1rqh {{ z3.h }}, p0/z, [{bx}]", "\n",
            "add {bx}, {bx}, #8*2 \n",
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
            vfmadd!(0, 3, 14, 2),
            vfmadd!(1, 3, 15, 2),
            vfmadd!(2, 3, 16, 2),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 3, 17, 3),
            vfmadd!(1, 3, 18, 3),
            vfmadd!(2, 3, 19, 3),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 3, 20, 4),
            vfmadd!(1, 3, 21, 4),
            vfmadd!(2, 3, 22, 4),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 3, 23, 5),
            vfmadd!(1, 3, 24, 5),
            vfmadd!(2, 3, 25, 5),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 3, 26, 6),
            vfmadd!(1, 3, 27, 6),
            vfmadd!(2, 3, 28, 6),
        )
    };
    (7) => {
        concat!(
            vfmadd!(0, 3, 29, 7),
            vfmadd!(1, 3, 30, 7),
            vfmadd!(2, 3, 31, 7),
        )
    };

    (0,M) => {
        concat!(
            vfmadd!(0, 3, 8, 0),
            vfmadd!(1, 3, 9, 0),
            vfmadd!(2, 3, 10, 0),
        )
    };
    (1,M) => {
        concat!(
            vfmadd!(0, 3, 11, 1),
            vfmadd!(1, 3, 12, 1),
            vfmadd!(2, 3, 13, 1),
        )
    };
    (2,M) => {
        concat!(
            vfmadd!(0, 3, 14, 2),
            vfmadd!(1, 3, 15, 2),
            vfmadd!(2, 3, 16, 2),
        )
    };
    (3,M) => {
        concat!(
            vfmadd!(0, 3, 17, 3),
            vfmadd!(1, 3, 18, 3),
            vfmadd!(2, 3, 19, 3),
        )
    };
    (4,M) => {
        concat!(
            vfmadd!(0, 3, 20, 4),
            vfmadd!(1, 3, 21, 4),
            vfmadd!(2, 3, 22, 4),
        )
    };
    (5,M) => {
        concat!(
            vfmadd!(0, 3, 23, 5),
            vfmadd!(1, 3, 24, 5),
            vfmadd!(2, 3, 25, 5),
        )
    };
    (6,M) => {
        concat!(
            vfmadd!(0, 3, 26, 6),
            vfmadd!(1, 3, 27, 6),
            vfmadd!(2, 3, 28, 6),
        )
    };
    (7,M) => {
        concat!(
            vfmadd!(0, 3, 29, 7),
            vfmadd!(1, 3, 30, 7),
            vfmadd!(2, 3, 31, 7),
        )
    };

}


macro_rules! step_3x8 {
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(48, $a_layout),
                "add {ax}, {ax}, #2*48 \n",
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
                load_a!(48, $a_layout),
                "add {ax}, {ax}, #2*48 \n",
                load_b!($b_layout, $nr),
                #(
                    fmadd_3v2!(n,M),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
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
    (12, $nr:tt) => {
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
    16
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
            let (k_i, k_l) = (k / 4, k % 4);
            let mut dim_arr = [d_arr[0]*2, d_arr[1]*2, d_arr[3]*2, k_i, k_l];
            let mut cf = c;
            let mut c_buf = [f16::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, $mr);
                dim_arr[2] = $mr*2;
                cf = c_buf.as_mut_ptr();
            }
            asm!(
                "ptrue p0.h",
                "/* {m_s} */", "\n",
                "/* {m_e0} */", "\n",
                "/* {m_e1} */", "\n",
                "/* {m_e2} */", "\n",
                asm_vzeroall!($mr,$nr),

                prefetch_c!(12,8),
        
                asm_init_ab!($mr,$a_layout,$b_layout),
                
                // 3 -> CONSIDKLEFT
                "BEQ 3f",
                
                // 2 -> KITER
                "2:",
                prefetch_0!(128, "{bx}"),
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
            let (k_i, k_l) = (k / 4, k % 4);
            let mut dim_arr = [d_arr[0]*2, d_arr[1]*2, d_arr[3]*2, k_i, k_l];
            let mut cf = c;
            let mut c_buf = [f16::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, $mr);
                dim_arr[2] = $mr*2;
                cf = c_buf.as_mut_ptr();
            }
            let _ = 'blk: {
                seq!(ni in 1..$nr {
                    if ni == n {
                        asm!(
                            "ptrue p0.h",
                            "/* {m_s} */", "\n",
                            "/* {m_e0} */", "\n",
                            "/* {m_e1} */", "\n",
                            "/* {m_e2} */", "\n",
                            prefetch_c!(12,8),
                            asm_vzeroall!($mr,ni),
                
                            asm_init_ab!($mr,$a_layout,$b_layout),
                        
                            // 3 -> CONSIDKLEFT
                            "BEQ 3f",
                        
                            // 2 -> KITER
                            "2:",
                            prefetch_0!(128, "{bx}"),
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

def_ukernel!(step_3x8, acc_3x8, store_3x8, 48, 8, B, B, M, ukernel_bb_partial);


def_ukernelxn!(step_3x8, acc_3x8, store_3x8, 48, 8, B, B, C, ukernel_n_bb);

def_ukernelxn!(step_3x8, acc_3x8, store_3x8, 48, 8, B, B, M, ukernel_n_bb_partial);



#[target_feature(enable="neon,sve")]
pub(crate) unsafe fn ukernel_bb<F: MyFn, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const TA, beta: *const TB,
    k: usize,
    d_arr: [usize; 4],
    m: usize,
    f: F,
) {
    let (k_i, k_l) = (k / 4, k % 4);
    let mut dim_arr = [d_arr[0]*2, d_arr[1]*2, d_arr[3]*2, k_i, k_l];
    let mut cf = c;
    let mut c_buf = [f16::ZERO; 2048 * 3 * 4];
    let c_cs = d_arr[3];
    if BUF {
        let mr = sve_vs() * 3;
        load_buf(c, d_arr[2], c_cs, &mut c_buf, m, 8, mr);
        dim_arr[2] = mr*2;
        cf = c_buf.as_mut_ptr();
    }
    asm!(
        "ptrue p0.h",
        "ptrue p1.h",
        asm_vzeroall!(12,8),

        prefetch_c!(12,8),

        asm_init_ab!(24,B,B),
        
        // 3 -> CONSIDKLEFT
        "BEQ 3f",
        
        // 2 -> KITER
        "2:",
        prefetch_0!(128, "{bx}"),
        step_3x8!(8, B, B),
        step_3x8!(8, B, B),
        step_3x8!(8, B, B),
        step_3x8!(8, B, B),

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
        step_3x8!(8, B, B),

        "sub {x0}, {x0}, #1",

        // 4 -> KLEFT
        "cmp {x0}, 0",
        "BNE 4b",

        // 5 -> POSTACCUM
        "5:",
        asm_c_load!(8),
        // scale by alpha
        asm_alpha_scale!(12,8),

        load_beta!(),

        // 6 -> BETAZERO
        "BEQ 6f",
        cum_seq!(acc_3x8,8,C),

        // 6 -> BETAZERO
        "6:",
        cum_seq!(store_3x8,8,C),
        
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
        for j in 0..8 {
            f.call(cf.add(j*mr), mr);
        }
        store_buf(c, d_arr[2], c_cs, &c_buf, m, 8, mr);
    } else {
        for j in 0..8 {
            f.call(cf.add(j*c_cs), m);
        }
    }
}