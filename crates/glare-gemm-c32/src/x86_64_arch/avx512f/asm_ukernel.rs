use seq_macro::seq;
use std::arch::asm;
use super::VS;
use crate::{TA, TB, TC};
use crate::MyFn;
use crate::{load_buf, store_buf};
use glare_base::c_mem;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "vaddps ", $m0, ",%zmm", $r1, ",%zmm", $r1, "\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "vmovupd ", $m0, ", %zmm1 {{%k1}}", "\n",
            "vaddps %zmm1, %zmm", $r1, ",%zmm", $r1, "\n",
        )
    };
    (C, $m0:expr) => {
        concat!(
            "vfmadd231ps ", $m0, ",%zmm0,%zmm", 0, "\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("vpxorq %zmm",r,",%zmm",r,",%zmm",r,"\n",)*)
        })
    }
}

macro_rules! vmovp {
    (B) => {
        "vmovaps "
    };
    ($layout:tt) => {
        "vmovups "
    };
}

macro_rules! vbroadcast {
    () => {
        "vbroadcastss"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $m2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vfmadd231ps ", $m2, "{{1to16}}", ", %zmm", $r1,", %zmm", $r3, "\n",
            "vfmadd231ps ", "4+", $m2, "{{1to16}}", ", %zmm", $r1,", %zmm", $r4, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovp!($layout), $m0, ",%zmm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %zmm", $r1, ", ", $m0, "\n",
        )
    };
    (B, $r1:expr, $m0:expr) => {
        concat!(
            "vmovaps %zmm", $r1, ", ", $m0, "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
            "vmovupd %zmm", $r1, ", ", $m0, " {{%k1}}\n",
        )
    };
}


macro_rules! complex_mul {
    ($r0:tt, $rt:tt, $rs:tt) => {
        concat!(
            "vpermilps $0xb1, %zmm", $r0, ", %zmm", $rt, "\n",
            "vmulps %zmm0, %zmm", $r0, ", %zmm", $r0, "\n",
            "vmulps %zmm1, %zmm", $rt, ", %zmm", $rt, "\n",
            "vfmadd231ps %zmm", $rt, ", %zmm", $rs, ", %zmm", $r0, "\n",
        )
    };
}

macro_rules! asm_alpha_scale_0 {
    (24, 4) => {
        concat!(
            vbroadcast!(), " ({alphax}), %zmm0 \n",
            vbroadcast!(), " 4({alphax}), %zmm1 \n",
            "vbroadcastsd ({alternate}), %zmm3 \n",

            complex_mul!(4, 7, 3),
            complex_mul!(6, 9, 3),
            complex_mul!(8, 11, 3),
            complex_mul!(10, 13, 3),
            complex_mul!(12, 15, 3),
            complex_mul!(14, 17, 3),
            complex_mul!(16, 19, 3),
            complex_mul!(18, 21, 3),
            complex_mul!(20, 23, 3),
            complex_mul!(22, 25, 3),
            complex_mul!(24, 27, 3),
            complex_mul!(26, 29, 3),
        )
    };
    (16, 7) => {
        concat!(
            vbroadcast!(), " ({alphax}), %zmm0 \n",
            vbroadcast!(), " 4({alphax}), %zmm1 \n",
            "vbroadcastsd ({alternate}), %zmm3 \n",

            complex_mul!(4, 7, 3),
            complex_mul!(6, 9, 3),
            complex_mul!(8, 11, 3),
            complex_mul!(10, 13, 3),
            complex_mul!(12, 15, 3),
            complex_mul!(14, 17, 3),
            complex_mul!(16, 19, 3),
            complex_mul!(18, 21, 3),
            complex_mul!(20, 23, 3),
            complex_mul!(22, 25, 3),
            complex_mul!(24, 27, 3),
            complex_mul!(26, 29, 3),
        )
    };
    ($r0:tt, $r1:tt) => {
        concat!(
            vbroadcast!(), " ({alphax}), %zmm0 \n",
            vbroadcast!(), " 4({alphax}), %zmm1 \n",
            "vbroadcastsd ({alternate}), %zmm3 \n",

            complex_mul!(2, 5, 3),
            complex_mul!(4, 7, 3),
            complex_mul!(6, 9, 3),
            complex_mul!(8, 11, 3),
            complex_mul!(10, 13, 3),
            complex_mul!(12, 15, 3),
            complex_mul!(14, 17, 3),
            complex_mul!(16, 19, 3),
            complex_mul!(18, 21, 3),
            complex_mul!(20, 23, 3),
            complex_mul!(22, 25, 3),
            complex_mul!(24, 27, 3),
            complex_mul!(26, 29, 3),
            complex_mul!(28, 31, 3),
            complex_mul!(30, 5, 3),
        )
    }
}

macro_rules! v_to_c {
    ($r0:tt, $r1:tt, $rs:tt) => {
        concat!(
            "vpermilps $0xb1, %zmm", $r1, ", %zmm", $r1, "\n",
            "vfmadd231ps %zmm", $r1, ", %zmm", $rs, ", %zmm", $r0, "\n",
        )
    }
}

macro_rules! permute_complex {
    (24, 4) => {
        concat!(
            // "vmulps %zmm1, %zmm31, %zmm31 \n",
            // permute even and odd elements
            "vbroadcastsd ({alternate}), %zmm0 \n",

            v_to_c!(4, 5, 0),
            v_to_c!(6, 7, 0),
            v_to_c!(8, 9, 0),
            v_to_c!(10, 11, 0),
            v_to_c!(12, 13, 0),
            v_to_c!(14, 15, 0),
            v_to_c!(16, 17, 0),
            v_to_c!(18, 19, 0),
            v_to_c!(20, 21, 0),
            v_to_c!(22, 23, 0),
            v_to_c!(24, 25, 0),
            v_to_c!(26, 27, 0),
        )
    };
    (16, 7) => {
        concat!(
            // "vmulps %zmm1, %zmm31, %zmm31 \n",
            // permute even and odd elements
            "vbroadcastsd ({alternate}), %zmm0 \n",
            v_to_c!(2, 3, 0),
            v_to_c!(4, 5, 0),
            v_to_c!(6, 7, 0),
            v_to_c!(8, 9, 0),
            v_to_c!(10, 11, 0),
            v_to_c!(12, 13, 0),
            v_to_c!(14, 15, 0),
            v_to_c!(16, 17, 0),
            v_to_c!(18, 19, 0),
            v_to_c!(20, 21, 0),
            v_to_c!(22, 23, 0),
            v_to_c!(24, 25, 0),
            v_to_c!(26, 27, 0),
            v_to_c!(28, 29, 0),
        )
    };
    ($mr:tt, $nr:tt) => {
        concat!(
            // "vmulps %zmm1, %zmm31, %zmm31 \n",
            // permute even and odd elements
            "vbroadcastsd ({alternate}), %zmm0 \n",

            v_to_c!(2, 3, 0),
            v_to_c!(4, 5, 0),
            v_to_c!(6, 7, 0),
            v_to_c!(8, 9, 0),
            v_to_c!(10, 11, 0),
            v_to_c!(12, 13, 0),
            v_to_c!(14, 15, 0),
            v_to_c!(16, 17, 0),
            v_to_c!(18, 19, 0),
            v_to_c!(20, 21, 0),
            v_to_c!(22, 23, 0),
            v_to_c!(24, 25, 0),
            v_to_c!(26, 27, 0),
            v_to_c!(28, 29, 0),
            v_to_c!(30, 31, 0),
        )
    }
}


macro_rules! mem {
    ($m0:tt, $b0:tt) => {
        concat!($b0, "+", $m0)
    };
}

macro_rules! acc_p {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!(C, mem!($m0, "0x40"), $r2),
            beta_fmadd!($layout, mem!($m0, "0x80"), $r3),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!($layout, mem!($m0, "0x40"), $r2),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1),
        )
    };
}

macro_rules! loadp {
    (24, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x40"), 1),
            loadp_unit!($layout, mem!($m0, "0x80"), 2),
        )
    };
    (16, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x40"), 1),
        )
    };
    (8, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
        )
    };
}


macro_rules! storep {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            storep_unit!(C, $r1, $m0),
            storep_unit!(C, $r2, mem!($m0, "0x40")),
            storep_unit!($layout, $r3, mem!($m0, "0x80")),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            storep_unit!(C, $r1, $m0),
            storep_unit!($layout, $r2, mem!($m0, "0x40")),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            storep_unit!($layout, $r1, $m0),
        )
    };
}


// only non contigous along m and n direction which is not changed frequently during iteration along k direction
/*

x1 -> cs_a
x2 -> cs_b
x3 -> ax + 3*cs_a
x4 -> bx + 3*cs_b

*/


macro_rules! asm_init_ab {
    ($KER:tt,B,B) => {
        concat!(
            "/* {x5} */", "\n",
            "/* {x4} */", "\n",
            "/* {x3} */", "\n",
            "/* {x2} */", "\n",
            "/* {x1} */", "\n",
            "mov 24({dim_arrx}),{x0}", "\n",
            "test {x0},{x0}", "\n",
        )
    };
    ($ker:tt,B,S) => {
        concat!(
            // mov cs_b to reg
            "mov ({dim_arrx}), {x1}", "\n",
            "mov 8({dim_arrx}), {x2}", "\n",
            "lea ({x2}, {x2}, 2), {x5}", "\n",
            "lea ({bx}, {x5}, 1), {x3}", "\n",
            "lea ({x3}, {x5}, 1), {x4}", "\n",
            "lea ({x4}, {x5}, 1), {x5}", "\n",

            "mov 24({dim_arrx}),{x0}", "\n",
            "test {x0},{x0}", "\n",
        )
    };
}


macro_rules! asm_c_load {

    (7) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
            "lea ({x0}, {x0}, 2), {x2}", "\n",
            "lea ({cx}, {x2},), {x1}", "\n",
            "lea ({x1}, {x2},), {x2}", "\n",

        )
    };
    (6) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
            "lea ({x0}, {x0}, 2), {x1}", "\n",
            "lea ({cx}, {x1},), {x1}", "\n",
        )
    };
    (5) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
            "lea ({x0}, {x0}, 2), {x1}", "\n",
            "lea ({cx}, {x1},), {x1}", "\n",
        )
    };
    (4) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
            "lea ({x0}, {x0}, 2), {x1}", "\n",
            "lea ({cx}, {x1},), {x1}", "\n",
        )
    };
    (3) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
        )
    };
    (2) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
        )
    };
    (1) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
        )
    };
}


macro_rules! asm_vzeroall {

    (24,4) => {vzeroall!(4,27)};
    (24,3) => {vzeroall!(4,21)};
    (24,2) => {vzeroall!(4,15)};
    (24,1) => {vzeroall!(4,9)};

    (16,7) => {vzeroall!(2,29)};
    (16,6) => {vzeroall!(2,25)};
    (16,5) => {vzeroall!(2,21)};
    (16,4) => {vzeroall!(2,17)};
    (16,3) => {vzeroall!(2,13)};
    (16,2) => {vzeroall!(2,9)};
    (16,1) => {vzeroall!(2,5)};

    (8,7) => {vzeroall!(2,15)};
    (8,6) => {vzeroall!(2,13)};
    (8,5) => {vzeroall!(2,11)};
    (8,4) => {vzeroall!(2,9)};
    (8,3) => {vzeroall!(2,7)};
    (8,2) => {vzeroall!(2,5)};
    (8,1) => {vzeroall!(2,3)};
}

macro_rules! asm_alpha_scale {
    (24,4) => {asm_alpha_scale_0!(4,27)};
    (24,3) => {asm_alpha_scale_0!(4,21)};
    (24,2) => {asm_alpha_scale_0!(4,15)};
    (24,1) => {asm_alpha_scale_0!(4,11)};

    (16,7) => {asm_alpha_scale_0!(2,29)};
    (16,6) => {asm_alpha_scale_0!(2,25)};
    (16,5) => {asm_alpha_scale_0!(2,21)};
    (16,4) => {asm_alpha_scale_0!(2,17)};
    (16,3) => {asm_alpha_scale_0!(2,13)};
    (16,2) => {asm_alpha_scale_0!(2,9)};
    (16,1) => {asm_alpha_scale_0!(2,5)};

    (8,7) => {asm_alpha_scale_0!(2,15)};
    (8,6) => {asm_alpha_scale_0!(2,13)};
    (8,5) => {asm_alpha_scale_0!(2,11)};
    (8,4) => {asm_alpha_scale_0!(2,9)};
    (8,3) => {asm_alpha_scale_0!(2,7)};
    (8,2) => {asm_alpha_scale_0!(2,5)};
    (8,1) => {asm_alpha_scale_0!(2,3)};
}

macro_rules! inc_a {
    (C) => {
        "add {x1}, {ax} \n"
    };
    (B, $mr:tt) => {
        concat!(
            "add $8*", $mr, ", {ax}", "\n",
        )
    };
}

macro_rules! inc_b {
    (S,7) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n"
    };
    (S,6) => {
        "add {x1},{bx} \n add {x1},{x3} \n"
    };
    (S,5) => {
        "add {x1},{bx} \n add {x1},{x3} \n"
    };
    (S,4) => {
        "add {x1},{bx} \n add {x1},{x3} \n"
    };
    (S,3) => {
        "add {x1},{bx} \n"
    };
    (S,2) => {
        "add {x1},{bx} \n"
    };
    (S,1) => {
        "add {x1},{bx} \n"
    };
    (B,$nr:tt) => {
        concat!(
            "add $8*", $nr, ", {bx}", "\n",
        )
    };
}

macro_rules! c_reg_24x4 {
    (0,0) => { 4 };
    (1,0) => { 6 };
    (2,0) => { 8 };
    (0,1) => { 10 };
    (1,1) => { 12 };
    (2,1) => { 14 };
    (0,2) => { 16 };
    (1,2) => { 18 };
    (2,2) => { 20 };
    (0,3) => { 22 };
    (1,3) => { 24 };
    (2,3) => { 26 };
}

macro_rules! c_reg_16x7 {
    (0,0) => { 2 };
    (1,0) => { 4 };
    (0,1) => { 6 };
    (1,1) => { 8 };
    (0,2) => { 10 };
    (1,2) => { 12 };
    (0,3) => { 14 };
    (1,3) => { 16 };
    (0,4) => { 18 };
    (1,4) => { 20 };
    (0,5) => { 22 };
    (1,5) => { 24 };
    (0,6) => { 26 };
    (1,6) => { 28 };
}

macro_rules! c_reg_8x7 {
    (0,0) => { 2 };
    (0,1) => { 4 };
    (0,2) => { 6 };
    (0,3) => { 8 };
    (0,4) => { 10 };
    (0,5) => { 12 };
    (0,6) => { 14 };
}

macro_rules! acc_24x4 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_24x4!(0,$ni), c_reg_24x4!(1,$ni), c_reg_24x4!(2,$ni))
    };
}

macro_rules! store_24x4 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_24x4!(0,$ni), c_reg_24x4!(1,$ni), c_reg_24x4!(2,$ni))
    };
}

macro_rules! acc_16x7 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_16x7!(0,$ni), c_reg_16x7!(1,$ni))
    };
}

macro_rules! store_16x7 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_16x7!(0,$ni), c_reg_16x7!(1,$ni))
    };
}

macro_rules! acc_8x7 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_8x7!(0,$ni))
    };
}

macro_rules! store_8x7 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_8x7!(0,$ni))
    };
}

macro_rules! cum_seq {
    ($step_macro:tt, $nr:tt, $layout:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!(n, $layout),)*)
        })
    };
}

macro_rules! load_a {
    ($mr:tt, B) => {
        loadp!($mr, B, "0({ax})")
    };
    ($mr:tt, C) => {
        loadp!($mr, C, "0({ax})")
    };
}

macro_rules! fmadd_3v {
    (0, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 4, 5),
            vfmadd!(1, $m, 6, 7),
            vfmadd!(2, $m, 8, 9),
        )
    };
    (1, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 10, 11),
            vfmadd!(1, $m, 12, 13),
            vfmadd!(2, $m, 14, 15),
        )
    };
    (2, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 16, 17),
            vfmadd!(1, $m, 18, 19),
            vfmadd!(2, $m, 20, 21),
        )
    };
    (3, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 22, 23),
            vfmadd!(1, $m, 24, 25),
            vfmadd!(2, $m, 26, 27),
        )
    };
}

macro_rules! fmadd_2v {
    (0, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 2, 3),
            vfmadd!(1, $m, 4, 5),
        )
    };
    (1, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 6, 7),
            vfmadd!(1, $m, 8, 9),
        )
    };
    (2, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 10, 11),
            vfmadd!(1, $m, 12, 13),
        )
    };
    (3, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 14, 15),
            vfmadd!(1, $m, 16, 17),
        )
    };
    (4, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 18, 19),
            vfmadd!(1, $m, 20, 21),
        )
    };
    (5, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 22, 23),
            vfmadd!(1, $m, 24, 25),
        )
    };
    (6, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 26, 27),
            vfmadd!(1, $m, 28, 29),
        )
    };
}

macro_rules! fmadd_1v {
    (0, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 2, 3),
        )
    };
    (1, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 4, 5),
        )
    };
    (2, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 6, 7),
        )
    };
    (3, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 8, 9),
        )
    };
    (4, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 10, 11),
        )
    };
    (5, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 12, 13),
        )
    };
    (6, $m:expr) => {
        concat!(
            vfmadd!(0, $m, 14, 15),
        )
    };
}

macro_rules! bd {
    (B, $i:tt) => {
        concat!($i, "*8({bx})")
    };
    (S, 0) => {
        "0({bx})"
    };
    (S, 1) => {
        "0({bx}, {x2})"
    };
    (S, 2) => {
        "0({bx}, {x2}, 2)"
    };
    (S, 3) => {
        "0({x3})"
    };
    (S, 4) => {
        "0({x3}, {x2})"
    };
    (S, 5) => {
        "0({x3}, {x2}, 2)"
    };
    (S, 6) => {
        "0({x4})"
    };
    (S, 7) => {
        "0({x4}, {x2})"
    };
}

// ***************************** 24x4 ******************************* //
macro_rules! step_24x4 {
    (4, B, B) => {
        concat!(
            load_a!(24, B),
            "addq $192, {ax} \n",
            fmadd_3v!(0, bd!(B, 0)),
            fmadd_3v!(1, bd!(B, 1)),
            "prefetcht0 384({ax}) \n",
            "prefetcht0 64({bx}) \n",
            fmadd_3v!(2, bd!(B, 2)),
            fmadd_3v!(3, bd!(B, 3)),
            "prefetcht0 448({ax}) \n",
            "addq $32, {bx} \n",
        )
    };
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(24, $a_layout),
                inc_a!($a_layout,24),
                "prefetcht0 64({bx}) \n",
                #(
                    fmadd_3v!(n, bd!($b_layout, n)),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 16x7 ******************************* //
macro_rules! step_16x7 {
    (7, B, B) => {
        concat!(
            load_a!(16, B),
            "addq $128, {ax} \n",
            fmadd_2v!(0, bd!(B, 0)),
            "prefetcht0 320({ax}) \n",
            fmadd_2v!(1, bd!(B, 1)),
            "prefetcht0 64({bx}) \n",
            fmadd_2v!(2, bd!(B, 2)),
            "prefetcht0 384({ax}) \n",
            fmadd_2v!(3, bd!(B, 3)),
            fmadd_2v!(4, bd!(B, 4)),
            fmadd_2v!(5, bd!(B, 5)),
            fmadd_2v!(6, bd!(B, 6)),
            "addq $56, {bx} \n",
        )
    };
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(16, $a_layout),
                inc_a!($a_layout,16),
                "prefetcht0 64({bx}) \n",
                #(
                    fmadd_2v!(n, bd!($b_layout, n)),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 8x7 ******************************* //
macro_rules! step_8x7 {
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(8, $a_layout),
                inc_a!($a_layout,8),
                "prefetcht0 64({bx}) \n",
                #(
                    fmadd_1v!(n, bd!($b_layout, n)),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! prefetch_c {
    (24, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
        });
    };
    (16, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
        });
    };
    (8, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(4+j*$ldc) as *const i8, 3);
        });
    };
}

macro_rules! mask_ptr {
    (M, $m:tt, $nm:ident) => {
        let $nm = if $m % VS == 0 && $m > 0 { 0xFF } else { (1_u8 << ($m % VS)) - 1 };
    };
    (C, $m:tt, $nm:ident) => {
        let $nm = 0xFF_u8;
    };
}

macro_rules! load_mask_ptr_asm {
    (M) => {
        "kmovw ({maskx}), %k1"
    };
    (C) => {
        "/* {maskx} */"
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
            alpha: *const TA,
            k: usize,
            d_arr: [usize; 4],
            m: usize,
            f: F,
        ) {
            mask_ptr!($is_partial, m, x);
            let mask_ptr = (&x) as *const u8;
            let k_iter = k / 4;
            let k_left = k % 4;
            let mut dim_arr = [d_arr[0]*8, d_arr[1]*8, d_arr[3]*8, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            let alt_arr = [-1.0f32, 1.0f32];
            let alt_buf = alt_arr.as_ptr();
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr);
                dim_arr[2] = m*8;
                cf = c_buf.as_mut_ptr();
            }
            // prefetch for c
            use std::arch::x86_64::_mm_prefetch;
            prefetch_c!($mr,$nr,c,c_cs);
            asm!(
                asm_vzeroall!($mr,$nr),
        
                asm_init_ab!($mr,$a_layout,$b_layout),
                
                // 3 -> CONSIDKLEFT
                "je 3f",
                
                // 2 -> KITER
                "2:",
                $step_macro!($nr, $a_layout, $b_layout),
                $step_macro!($nr, $a_layout, $b_layout),
                $step_macro!($nr, $a_layout, $b_layout),
                $step_macro!($nr, $a_layout, $b_layout),
        
                "dec {x0}",
                // 2 -> KITER
                "jne 2b",

                // 3 -> CONSIDKLEFT
                "3:",
                "mov 32({dim_arrx}),{x0}",
                "test {x0},{x0}",

                // 5 -> POSTACCUM
                "je 5f",
                // 4 -> KLEFT
                "4:",
                $step_macro!($nr, $a_layout, $b_layout),

                "dec {x0}",
        
                // 4 -> KLEFT
                "jne 4b",
        
                // 5 -> POSTACCUM
                "5:",
                permute_complex!($mr, $nr),
                asm_c_load!($nr),
                // scale by alpha
                asm_alpha_scale!($mr, $nr),

                load_mask_ptr_asm!($is_partial),				
                // 6 -> BETAZERO
                cum_seq!($acc_macro,$nr,$is_partial),

                // 6 -> BETAZERO
                cum_seq!($store_macro,$nr,$is_partial),
                
                // 7 -> DDONE
                "7:",
                // "vzeroupper",
                ax = inout(reg) a => _,
                bx = inout(reg) b => _,
                cx = inout(reg) cf => _,
                dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                alphax = inout(reg) alpha => _,
                maskx = inout(reg) mask_ptr => _,
                alternate = inout(reg) alt_buf => _,
                x0 = out(reg) _,
                x1 = out(reg) _,
                x2 = out(reg) _,
                x3 = out(reg) _,
                x4 = out(reg) _,
                x5 = out(reg) _,
                out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
                out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
                out("k1") _,
                options(att_syntax)
            );
            if BUF {
                for j in 0..$nr {
                    f.call(cf.add(j*m), m);
                }
                store_buf(c, d_arr[2], c_cs, &c_buf, m, $nr);
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
            alpha: *const TA,
            k: usize,
            d_arr: [usize; 4],
            m: usize, n: usize,
            f: F,
        ) {
            mask_ptr!($is_partial, m, x);
            let mask_ptr = (&x) as *const u8;
            let k_iter = k / 4;
            let k_left = k % 4;
            let mut dim_arr = [d_arr[0]*8, d_arr[1]*8, d_arr[3]*8, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let alt_arr = [-1.0f32, 1.0f32];
            let alt_buf = alt_arr.as_ptr();
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n);
                dim_arr[2] = m*8;
                cf = c_buf.as_mut_ptr();
            }
            use std::arch::x86_64::_mm_prefetch;
            let _ = 'blk: {
                seq!(ni in 1..$nr {
                    if ni == n {
                        // prefetch for c
                        prefetch_c!($mr,ni,c,c_cs);
                        asm!(
                            asm_vzeroall!($mr,ni),
                
                            asm_init_ab!($mr,$a_layout,$b_layout),
                        
                            // 3 -> CONSIDKLEFT
                            "je 3f",
                        
                            // 2 -> KITER
                            "2:",
                            $step_macro!(ni, $a_layout, $b_layout),
                            $step_macro!(ni, $a_layout, $b_layout),
                            $step_macro!(ni, $a_layout, $b_layout),
                            $step_macro!(ni, $a_layout, $b_layout),
                
                            "dec {x0}",
                            // 2 -> KITER
                            "jne 2b",

                            // 3 -> CONSIDKLEFT
                            "3:",
                            "mov 32({dim_arrx}),{x0}",
                            "test {x0},{x0}",

                            // 5 -> POSTACCUM
                            "je 5f",
                            // 4 -> KLEFT
                            "4:",
                            $step_macro!(ni, $a_layout, $b_layout),

                            "dec {x0}",
                
                            // 4 -> KLEFT
                            "jne 4b",
                
                            // 5 -> POSTACCUM
                            "5:",
                            permute_complex!($mr, ni),
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),


                            load_mask_ptr_asm!($is_partial),				
                            // 6 -> BETAZERO
                            cum_seq!($acc_macro,ni,$is_partial),

                            // 6 -> BETAZERO
                            cum_seq!($store_macro,ni,$is_partial),
                            
                            // 7 -> DDONE
                            "7:",
                            // "vzeroupper",
                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) cf => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            maskx = inout(reg) mask_ptr => _,
                            alternate = inout(reg) alt_buf => _,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            x4 = out(reg) _,
                            x5 = out(reg) _,
                            out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                            out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                            out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
                            out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
                            out("k1") _,
                            options(att_syntax)
                        );
                        break 'blk;
                    }
                });
            };
            if BUF {
                for j in 0..n {
                    f.call(cf.add(j*m), m);
                }
                store_buf(c, d_arr[2], c_cs, &c_buf, m, n);
            } else {
                for j in 0..n {
                    f.call(cf.add(j*c_cs), m);
                }
            }
        }
    };
}

// def_ukernel!(step_24x4, acc_24x4, store_24x4, 24, 4, B, B, C, 4, ukernel_24x4_bb);
// def_ukernel!(step_16x7, acc_16x7, store_16x7, 16, 4, B, B, C, 4, ukernel_16x4_bb);
// def_ukernel!(step_8x7, acc_8x7, store_8x7, 8, 4, B, B, C, 4, ukernel_8x4_bb);

def_ukernel!(step_24x4, acc_24x4, store_24x4, 24, 4, B, B, M, ukernel_24x4_bb_partial);
def_ukernel!(step_16x7, acc_16x7, store_16x7, 16, 4, B, B, M, ukernel_16x4_bb_partial);
def_ukernel!(step_8x7, acc_8x7, store_8x7, 8, 4, B, B, M, ukernel_8x4_bb_partial);

def_ukernel!(step_24x4, acc_24x4, store_24x4, 24, 4, B, S, C, ukernel_24x4_bs);

def_ukernel!(step_24x4, acc_24x4, store_24x4, 24, 4, B, S, M, ukernel_24x4_bs_partial);
def_ukernel!(step_16x7, acc_16x7, store_16x7, 16, 4, B, S, M, ukernel_16x4_bs_partial);
def_ukernel!(step_8x7, acc_8x7, store_8x7, 8, 4, B, S, M, ukernel_8x4_bs_partial);


def_ukernelxn!(step_24x4, acc_24x4, store_24x4, 24, 4, B, B, C, ukernel_24xn_bb);
def_ukernelxn!(step_16x7, acc_16x7, store_16x7, 16, 7, B, B, C, ukernel_16xn_bb);
// def_ukernelxn!(step_8x7, acc_8x7, store_8x7, 8, 7, B, B, C, 4, ukernel_8xn_bb);

def_ukernelxn!(step_24x4, acc_24x4, store_24x4, 24, 4, B, B, M, ukernel_24xn_bb_partial);
def_ukernelxn!(step_16x7, acc_16x7, store_16x7, 16, 7, B, B, M, ukernel_16xn_bb_partial);
def_ukernelxn!(step_8x7, acc_8x7, store_8x7, 8, 7, B, B, M, ukernel_8xn_bb_partial);

def_ukernelxn!(step_24x4, acc_24x4, store_24x4, 24, 4, B, S, C, ukernel_24xn_bs);

def_ukernelxn!(step_24x4, acc_24x4, store_24x4, 24, 4, B, S, M, ukernel_24xn_bs_partial);
def_ukernelxn!(step_16x7, acc_16x7, store_16x7, 16, 4, B, S, M, ukernel_16xn_bs_partial);
def_ukernelxn!(step_8x7, acc_8x7, store_8x7, 8, 4, B, S, M, ukernel_8xn_bs_partial);


def_ukernel!(step_16x7, acc_16x7, store_16x7, 16, 7, B, B, M, ukernel_16x7_bb_partial);
def_ukernel!(step_8x7, acc_8x7, store_8x7, 8, 7, B, B, M, ukernel_8x7_bb_partial);

// based on l1 prefetching scheme is from openblas impl for skylax
// see: https://github.com/OpenMathLib/OpenBLAS/pull/2300
// this is adapted to our ukernel of 24x4
// seems to stem from high bandwith of l1 cache (compared to other uarch e.g. haswell
// where the same l1 prefetching does not benefit as much)
pub(crate) unsafe fn ukernel_24x4_bb<F: MyFn, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const TA,
    k: usize,
    d_arr: [usize; 4],
    a_pft1_offset: usize,
    f: F,
) {
    let k_left0 = k % 4;
    let k_left = if k_left0 == 0 {4} else {k_left0};
    let k_iter = (k - k_left) / 4;
    let mut dim_arr = [d_arr[3]*8, k_iter, k_left, a_pft1_offset];
    let alt_arr = [-1.0f32, 1.0f32];
    let alt_buf = alt_arr.as_ptr();
    let mut cf = c;
    let mut c_buf = [TC::ZERO; 24 * 4];
    let c_cs = d_arr[3];
    if BUF {
        load_buf(c, d_arr[2], c_cs, &mut c_buf, 24, 4);
        dim_arr[2] = 24*8;
        cf = c_buf.as_mut_ptr();
    }
    asm!(
        asm_vzeroall!(24,4),
        "mov 8({dim_arrx}),{x0}",
        "test {x0},{x0}",
        "je 3f",
        // "je 3f",
        "mov {cx}, {x2}",
        "mov {ax}, {x5}",
        "mov 24({dim_arrx}),{x1}",
        "add {x1}, {x5}",
        "mov ({dim_arrx}),{x1}",
        "2:",
        step_24x4!(4, B, B),

        "movq $64*4, {x4}",
        // divisiblity by 4
        "testq $3, {x0}",
        "cmovz {x1},{x4}",

        step_24x4!(4, B, B),

        "prefetcht1 ({x2})",

        "subq $64*3, {x2}",
        "addq {x4}, {x2}",

        step_24x4!(4, B, B),

        "prefetcht1 ({x5})",
        "addq $32, {x5}",

        "testq $63, {x0}",
        "cmovz {cx},{x2}",

        step_24x4!(4, B, B),

        "dec {x0}",
        "jne 2b",
        "3:",
        "mov 16({dim_arrx}),{x0}",
        "test {x0},{x0}",

        // 5 -> POSTACCUM
        "je 5f",
        "mov {cx}, {x2}",
        "mov ({dim_arrx}),{x1}",
        "4:",
        "prefetcht0 ({x2})",
        "prefetcht0 64({x2})",
        "prefetcht0 128({x2})",
        step_24x4!(4, B, B),

        "add {x1}, {x2}",
        "dec {x0}",
        "jne 4b",
        "5:",
        permute_complex!(24, 4),
        "mov ({dim_arrx}),{x0}",
        "lea ({x0}, {x0}, 2), {x3}",
        "lea ({cx}, {x3},), {x1}",
        "lea ({x1}, {x3},), {x2}",
        // scale by alpha
        asm_alpha_scale!(24, 4),

        // 6 -> BETAZERO
        cum_seq!(acc_24x4,4,C),

        // 6 -> BETAZERO
        cum_seq!(store_24x4,4,C),

        "7:",
        ax = inout(reg) a => _, 
        bx = inout(reg) b => _, 
        cx = inout(reg) cf => _,
        dim_arrx = inout(reg) dim_arr.as_ptr() => _, 
        alphax = inout(reg) alpha => _, 
        alternate = inout(reg) alt_buf => _,
        x0 = out(reg) _, 
        x1 = out(reg)_, 
        x2 = out(reg) _, 
        x3 = out(reg) _, 
        x4 = out(reg) _,
        x5 = out(reg) _, 
        out("xmm0") _, out("xmm1") _,
        out("xmm2") _, out("xmm3") _, out("xmm4") _, out("xmm5") _, out("xmm6") _,
        out("xmm7") _, out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
        out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
        options(att_syntax)
    );
    if BUF {
        for j in 0..7 {
            f.call(cf.add(j*16), 16);
        }
        store_buf(c, d_arr[2], c_cs, &c_buf, 16, 7);
    } else {
        for j in 0..7 {
            f.call(cf.add(j*c_cs), 16);
        }
    }
}




// based on l1 prefetching scheme is from openblas impl for skylax
// see: https://github.com/OpenMathLib/OpenBLAS/pull/2300
// this is adapted to our ukernel of 24x4
// seems to stem from high bandwith of l1 cache (compared to other uarch e.g. haswell
// where the same l1 prefetching does not benefit as much)
pub(crate) unsafe fn ukernel_16x7_bb<F: MyFn, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const TA,
    k: usize,
    d_arr: [usize; 4],
    a_pft1_offset: usize,
    f: F,
) {
    let k_left0 = k % 8;
    let k_left = if k_left0 == 0 {8} else {k_left0};
    let k_iter = (k - k_left) / 4;
    let mut dim_arr = [d_arr[3]*8, k_iter, k_left, a_pft1_offset];
    let alt_arr = [-1.0f32, 1.0f32];
    let alt_buf = alt_arr.as_ptr();
    let mut cf = c;
    let mut c_buf = [TC::ZERO; 16 * 7];
    let c_cs = d_arr[3];
    if BUF {
        load_buf(c, d_arr[2], c_cs, &mut c_buf, 16, 7);
        dim_arr[2] = 16*8;
        cf = c_buf.as_mut_ptr();
    }
    asm!(
        asm_vzeroall!(16,7),
        "mov 8({dim_arrx}),{x0}",
        "test {x0},{x0}",
        "je 3f",
        // "je 3f",
        "mov {cx}, {x2}",
        "mov {ax}, {x5}",
        "mov 24({dim_arrx}),{x1}",
        "add {x1}, {x5}",
        "mov ({dim_arrx}),{x1}",
        "2:",
        step_16x7!(7, B, B),

        "movq $64*4, {x4}",
        // divisiblity by 4
        "testq $3, {x0}",
        "cmovz {x1},{x4}",

        step_16x7!(7, B, B),

        "prefetcht1 ({x2})",

        "subq $64*3, {x2}",
        "addq {x4}, {x2}",

        step_16x7!(7, B, B),

        "prefetcht1 ({x5})",
        "addq $32, {x5}",

        "testq $63, {x0}",
        "cmovz {cx},{x2}",

        step_16x7!(7, B, B),

        "dec {x0}",
        "jne 2b",
        "3:",
        "mov 16({dim_arrx}),{x0}",
        "test {x0},{x0}",

        // 5 -> POSTACCUM
        "je 5f",
        "mov {cx}, {x2}",
        "mov ({dim_arrx}),{x1}",
        "4:",
        "prefetcht0 ({x2})",
        "prefetcht0 64({x2})",
        // "prefetcht0 128({x2})",
        step_16x7!(7, B, B),

        "add {x1}, {x2}",
        "dec {x0}",
        "jne 4b",
        "5:",
        permute_complex!(16, 7),
        "mov ({dim_arrx}),{x0}",
        "lea ({x0}, {x0}, 2), {x3}",
        "lea ({cx}, {x3},), {x1}",
        "lea ({x1}, {x3},), {x2}",
        // scale by alpha
        asm_alpha_scale!(16, 7),

        // 6 -> BETAZERO
        cum_seq!(acc_16x7,7,C),

        // 6 -> BETAZERO
        cum_seq!(store_16x7,7,C),

        "7:",
        ax = inout(reg) a => _, 
        bx = inout(reg) b => _, 
        cx = inout(reg) cf => _,
        dim_arrx = inout(reg) dim_arr.as_ptr() => _, 
        alphax = inout(reg) alpha => _, 
        alternate = inout(reg) alt_buf => _,
        x0 = out(reg) _, 
        x1 = out(reg)_, 
        x2 = out(reg) _, 
        x3 = out(reg) _, 
        x4 = out(reg) _,
        x5 = out(reg) _, 
        out("xmm0") _, out("xmm1") _,
        out("xmm2") _, out("xmm3") _, out("xmm4") _, out("xmm5") _, out("xmm6") _,
        out("xmm7") _, out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
        out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
        options(att_syntax)
    );
    if BUF {
        for j in 0..7 {
            f.call(cf.add(j*16), 16);
        }
        store_buf(c, d_arr[2], c_cs, &c_buf, 16, 7);
    } else {
        for j in 0..7 {
            f.call(cf.add(j*c_cs), 16);
        }
    }
}