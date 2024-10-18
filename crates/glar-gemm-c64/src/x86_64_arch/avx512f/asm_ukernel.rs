use seq_macro::seq;
use std::arch::asm;
use super::VS;
use crate::{TA, TB, TC};
use crate::MyFn;
use glar_base::{load_buf, store_buf, c_mem};

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "vaddpd ", $m0, ",%zmm", $r1, ",%zmm", $r1, "\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "vmovupd ", $m0, ", %zmm1 {{%k1}}", "\n",
            "vaddpd %zmm1, %zmm", $r1, ",%zmm", $r1, "\n",
        )
    };
    (C, $m0:expr) => {
        concat!(
            "vfmadd231pd ", $m0, ",%zmm0,%zmm", 0, "\n",
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
        "vmovapd "
    };
    ($layout:tt) => {
        "vmovupd "
    };
}

macro_rules! vbroadcast {
    () => {
        "vbroadcastsd"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $b1:expr, $b2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vfmadd231pd ", "%zmm",$b1, ", %zmm", $r1,", %zmm", $r3, "\n",
            "vfmadd231pd ", "%zmm",$b2, ", %zmm", $r1,", %zmm", $r4, "\n",
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
            "vmovupd %zmm", $r1, ", ", $m0, "\n",
        )
    };
    (B, $r1:expr, $m0:expr) => {
        concat!(
            "vmovapd %zmm", $r1, ", ", $m0, "\n",
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
            "vpermilpd $0b1010101, %zmm", $r0, ", %zmm", $rt, "\n",
            "vmulpd %zmm0, %zmm", $r0, ", %zmm", $r0, "\n",
            "vmulpd %zmm1, %zmm", $rt, ", %zmm", $rt, "\n",
            "vfmadd231pd %zmm", $rt, ", %zmm", $rs, ", %zmm", $r0, "\n",
        )
    };
}

macro_rules! asm_alpha_scale_0 {
    (12, 4) => {
        concat!(
            vbroadcast!(), " ({alphax}), %zmm0 \n",
            vbroadcast!(), " 8({alphax}), %zmm1 \n",
            "vmovupd ({alternate}), %zmm3 \n",

            complex_mul!(8,9,3),
            complex_mul!(10,11,3),
            complex_mul!(12,13,3),
            complex_mul!(14,15,3),
            complex_mul!(16,17,3),
            complex_mul!(18,19,3),
            complex_mul!(20,21,3),
            complex_mul!(22,23,3),
            complex_mul!(24,25,3),
            complex_mul!(26,27,3),
            complex_mul!(28,29,3),
            complex_mul!(30,31,3),
        )
    };
    (8, 6) => {
        concat!(
            vbroadcast!(), " ({alphax}), %zmm0 \n",
            vbroadcast!(), " 8({alphax}), %zmm1 \n",
            "vmovupd ({alternate}), %zmm3 \n",

            complex_mul!(8,9,3),
            complex_mul!(10,11,3),
            complex_mul!(12,13,3),
            complex_mul!(14,15,3),
            complex_mul!(16,17,3),
            complex_mul!(18,19,3),
            complex_mul!(20,21,3),
            complex_mul!(22,23,3),
            complex_mul!(24,25,3),
            complex_mul!(26,27,3),
            complex_mul!(28,29,3),
            complex_mul!(30,31,3),
        )
    };
    ($r0:tt, $r1:tt) => {
        concat!(
            vbroadcast!(), " ({alphax}), %zmm0 \n",
            vbroadcast!(), " 8({alphax}), %zmm1 \n",
            "vmovupd ({alternate}), %zmm3 \n",

            complex_mul!(8,9,3),
            complex_mul!(10,11,3),
            complex_mul!(12,13,3),
            complex_mul!(14,15,3),
            complex_mul!(16,17,3),
            complex_mul!(18,19,3),
            complex_mul!(20,21,3),
            complex_mul!(22,23,3),
            complex_mul!(24,25,3),
            complex_mul!(26,27,3),
            complex_mul!(28,29,3),
            complex_mul!(30,31,3),
        )
    }
}

macro_rules! v_to_c {
    ($r0:tt, $r1:tt, $rs:tt) => {
        concat!(
            "vpermilpd $0b1010101, %zmm", $r1, ", %zmm", $r1, "\n",
            "vfmadd231pd %zmm", $r1, ", %zmm", $rs, ", %zmm", $r0, "\n",
        )
    }
}

macro_rules! permute_complex {
    (12, 4) => {
        concat!(
            // permute even and odd elements
            "vmovupd ({alternate}), %zmm0 \n",

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
    };
    (8, 6) => {
        concat!(
            // permute even and odd elements
            "vmovupd ({alternate}), %zmm0 \n",
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
    };
    ($mr:tt, $nr:tt) => {
        concat!(
            // permute even and odd elements
            "vmovupd ({alternate}), %zmm0 \n",

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
    (12, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x40"), 1),
            loadp_unit!($layout, mem!($m0, "0x80"), 2),
        )
    };
    (8, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x40"), 1),
        )
    };
    (4, $layout:tt, $m0:expr) => {
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
    (12,4) => {vzeroall!(8,31)};
    (12,3) => {vzeroall!(8,25)};
    (12,2) => {vzeroall!(8,19)};
    (12,1) => {vzeroall!(8,13)};

    (8,6) => {vzeroall!(8,31)};
    (8,5) => {vzeroall!(8,27)};
    (8,4) => {vzeroall!(8,23)};
    (8,3) => {vzeroall!(8,19)};
    (8,2) => {vzeroall!(8,15)};
    (8,1) => {vzeroall!(8,11)};

    (4,6) => {vzeroall!(20,31)};
    (4,5) => {vzeroall!(20,29)};
    (4,4) => {vzeroall!(20,27)};
    (4,3) => {vzeroall!(20,25)};
    (4,2) => {vzeroall!(20,23)};
    (4,1) => {vzeroall!(20,21)};
}

macro_rules! asm_alpha_scale {
    (12,4) => {asm_alpha_scale_0!(8,31)};
    (12,3) => {asm_alpha_scale_0!(8,25)};
    (12,2) => {asm_alpha_scale_0!(8,19)};
    (12,1) => {asm_alpha_scale_0!(8,13)};

    (8,6) => {asm_alpha_scale_0!(8,31)};
    (8,5) => {asm_alpha_scale_0!(8,27)};
    (8,4) => {asm_alpha_scale_0!(8,23)};
    (8,3) => {asm_alpha_scale_0!(8,19)};
    (8,2) => {asm_alpha_scale_0!(8,15)};
    (8,1) => {asm_alpha_scale_0!(8,11)};

    (4,6) => {asm_alpha_scale_0!(20,31)};
    (4,5) => {asm_alpha_scale_0!(20,29)};
    (4,4) => {asm_alpha_scale_0!(20,27)};
    (4,3) => {asm_alpha_scale_0!(20,25)};
    (4,2) => {asm_alpha_scale_0!(20,23)};
    (4,1) => {asm_alpha_scale_0!(20,21)};
}

macro_rules! inc_a {
    (C) => {
        "add {x1}, {ax} \n"
    };
    (B, $mr:tt) => {
        concat!(
            "add $16*", $mr, ", {ax}", "\n",
        )
    };
}

macro_rules! inc_b {
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
            "add $16*", $nr, ", {bx}", "\n",
        )
    };
}
macro_rules! c_reg_3x4 {
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
}

macro_rules! c_reg_2x6 {
    (0,0) => { 8 };
    (1,0) => { 10 };
    (0,1) => { 12 };
    (1,1) => { 14 };
    (0,2) => { 16 };
    (1,2) => { 18 };
    (0,3) => { 20 };
    (1,3) => { 22 };
    (0,4) => { 24 };
    (1,4) => { 26 };
    (0,5) => { 28 };
    (1,5) => { 30 };
}

macro_rules! c_reg_1x6 {
    (0,0) => { 20 };
    (0,1) => { 22 };
    (0,2) => { 24 };
    (0,3) => { 26 };
    (0,4) => { 28 };
    (0,5) => { 30 };
}


macro_rules! acc_3x4 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_3x4!(0,$ni), c_reg_3x4!(1,$ni), c_reg_3x4!(2,$ni))
    };
}

macro_rules! store_3x4 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_3x4!(0,$ni), c_reg_3x4!(1,$ni), c_reg_3x4!(2,$ni))
    };
}

macro_rules! acc_2x6 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_2x6!(0,$ni), c_reg_2x6!(1,$ni))
    };
}

macro_rules! store_2x6 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_2x6!(0,$ni), c_reg_2x6!(1,$ni))
    };
}

macro_rules! acc_1x6 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_1x6!(0,$ni))
    };
}

macro_rules! store_1x6 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_1x6!(0,$ni))
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

macro_rules! load_b {
    (S, 0, $r1:expr, $r2:expr) => {
        concat!(
            "prefetcht0 64({bx},{x1},8) \n",
            vbroadcast!(), " ({bx}),%zmm", $r1, "\n",
            vbroadcast!(), " 8({bx}),%zmm", $r2, "\n",
        )
    };
    (S, 1, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx},{x2},1),%zmm", $r1, "\n",
            vbroadcast!(), " 8({bx},{x2},1),%zmm", $r2, "\n",
        )
    };
    (S, 2, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx},{x2},2),%zmm", $r1, "\n",
            vbroadcast!(), " 8({bx},{x2},2),%zmm", $r2, "\n",
        )
    };
    (S, 3, $r1:expr, $r2:expr) => {
        concat!(
            "prefetcht0 64({x3},{x1},8) \n",
            vbroadcast!(), " ({x3}),%zmm", $r1, "\n",
            vbroadcast!(), " 8({x3}),%zmm", $r2, "\n",
        )
    };
    (S, 4, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({x3},{x2},1),%zmm", $r1, "\n",
            vbroadcast!(), " 8({x3},{x2},1),%zmm", $r2, "\n",
        )
    };
    (S, 5, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({x3},{x2},2),%zmm", $r1, "\n",
            vbroadcast!(), " 8({x3},{x2},2),%zmm", $r2, "\n",
        )
    };
    (B, $N:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ", $N, "*16({bx}), %zmm", $r1, "\n",
            vbroadcast!(), " ", $N, "*16+8({bx}), %zmm", $r2, "\n",
        )
    };
}

macro_rules! fmadd_3v {
    (0) => {
        concat!(
            vfmadd!(0, 3, 4, 8, 9),
            vfmadd!(1, 3, 4, 10, 11),
            vfmadd!(2, 3, 4, 12, 13),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 5, 6, 14, 15),
            vfmadd!(1, 5, 6, 16, 17),
            vfmadd!(2, 5, 6, 18, 19),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 7, 3, 20, 21),
            vfmadd!(1, 7, 3, 22, 23),
            vfmadd!(2, 7, 3, 24, 25),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 4, 5, 26, 27),
            vfmadd!(1, 4, 5, 28, 29),
            vfmadd!(2, 4, 5, 30, 31),
        )
    };
}

macro_rules! fmadd_2v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 3, 8, 9),
            vfmadd!(1, 2, 3, 10, 11),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 4, 5, 12, 13),
            vfmadd!(1, 4, 5, 14, 15),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 6, 7, 16, 17),
            vfmadd!(1, 6, 7, 18, 19),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 2, 3, 20, 21),
            vfmadd!(1, 2, 3, 22, 23),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 4, 5, 24, 25),
            vfmadd!(1, 4, 5, 26, 27),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 6, 7, 28, 29),
            vfmadd!(1, 6, 7, 30, 31),
        )
    };
}

macro_rules! fmadd_1v {
    (0) => {
        concat!(
            vfmadd!(0, 1, 2, 20, 21),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 3, 4, 22, 23),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 5, 6, 24, 25),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 7, 8, 26, 27),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 9, 10, 28, 29),
        )
    };
    (5) => {
        concat!(
            vfmadd!(0, 11, 12, 30, 31),
        )
    };
}

macro_rules! b_num_3x4 {
    (0,0) => {3}; (0,1) => {4};
    (1,0) => {5}; (1,1) => {6};
    (2,0) => {7}; (2,1) => {3};
    (3,0) => {4}; (3,1) => {5};
}

macro_rules! b_num_2x6 {
    (0,0) => {2}; (0,1) => {3};
    (1,0) => {4}; (1,1) => {5};
    (2,0) => {6}; (2,1) => {7};
    (3,0) => {2}; (3,1) => {3};
    (4,0) => {4}; (4,1) => {5};
    (5,0) => {6}; (5,1) => {7};
}

macro_rules! b_num_1x6 {
    (0,0) => {1}; (0,1) => {2};
    (1,0) => {3}; (1,1) => {4};
    (2,0) => {5}; (2,1) => {6};
    (3,0) => {7}; (3,1) => {8};
    (4,0) => {9}; (4,1) => {10};
    (5,0) => {11}; (5,1) => {12};
}

// ***************************** 3x4 ******************************* //
macro_rules! step_3x4 {
    (4, B, B) => {
        concat!(
            load_a!(12, B),
            "addq $192, {ax} \n",
            load_b!(B, 0, 3, 4),
            fmadd_3v!(0),
            load_b!(B, 1, 5, 6),
            fmadd_3v!(1),
            "prefetcht0 384({ax}) \n",
            "prefetcht0 64({bx}) \n",
            load_b!(B, 2, 7, 3),
            fmadd_3v!(2),
            load_b!(B, 3, 4, 5),
            fmadd_3v!(3),
            "prefetcht0 448({ax}) \n",
            "addq $64, {bx} \n",
        )
    };
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(12, $a_layout),
                inc_a!($a_layout,12),
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, b_num_3x4!(n,0), b_num_3x4!(n,1)),
                    fmadd_3v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 2x6 ******************************* //
macro_rules! step_2x6 {
    (6, B, B) => {
        concat!(
            load_a!(8, B),
            "addq $128, {ax} \n",
            load_b!(B, 0, 2, 3),
            fmadd_2v!(0),
            "prefetcht0 320({ax}) \n",
            load_b!(B, 1, 4, 5),
            fmadd_2v!(1),
            "prefetcht0 64({bx}) \n",
            load_b!(B, 2, 6, 7),
            fmadd_2v!(2),
            "prefetcht0 384({ax}) \n",
            load_b!(B, 3, 2, 3),
            fmadd_2v!(3),
            load_b!(B, 4, 4, 5),
            fmadd_2v!(4),
            load_b!(B, 5, 6, 7),
            fmadd_2v!(5),
            "addq $96, {bx} \n",
        )
    };
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(8, $a_layout),
                inc_a!($a_layout,8),
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, b_num_2x6!(n,0), b_num_2x6!(n,1)),
                    fmadd_2v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 1x6 ******************************* //
macro_rules! step_1x6 {
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(4, $a_layout),
                inc_a!($a_layout,4),
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, b_num_1x6!(n,0), b_num_1x6!(n,1)),
                    fmadd_1v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! prefetch_c {
    (12, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(4+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
        });
    };
    (8, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(4+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
        });
    };
    (4, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(2+j*$ldc) as *const i8, 3);
        });
    };
}

macro_rules! mask_ptr {
    (M, $m:tt, $nm:ident) => {
        let $nm = if $m % VS == 0 && $m > 0 { 0xFF } else { (1_u8 << (($m % VS)*2)) - 1 };
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
            let mut dim_arr = [d_arr[0]*16, d_arr[1]*16, d_arr[3]*16, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            let alt_arr = [-1.0f64, 1.0f64, -1.0f64, 1.0f64, -1.0f64, 1.0f64, -1.0f64, 1.0f64];
            let alt_buf = alt_arr.as_ptr();
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, $mr);
                dim_arr[2] = $mr*16;
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
            let mut dim_arr = [d_arr[0]*16, d_arr[1]*16, d_arr[3]*16, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let alt_arr = [-1.0f64, 1.0f64, -1.0f64, 1.0f64, -1.0f64, 1.0f64, -1.0f64, 1.0f64];
            let alt_buf = alt_arr.as_ptr();
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, $mr);
                dim_arr[2] = $mr*16;
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

def_ukernel!(step_3x4, acc_3x4, store_3x4, 12, 4, B, B, M, ukernel_3_bb_partial);
def_ukernel!(step_2x6, acc_2x6, store_2x6, 8, 4, B, B, M, ukernel_2_bb_partial);
def_ukernel!(step_1x6, acc_1x6, store_1x6, 4, 4, B, B, M, ukernel_1_bb_partial);

def_ukernel!(step_3x4, acc_3x4, store_3x4, 12, 4, B, S, C, ukernel_bs);

def_ukernel!(step_3x4, acc_3x4, store_3x4, 12, 4, B, S, M, ukernel_3_bs_partial);
def_ukernel!(step_2x6, acc_2x6, store_2x6, 8, 4, B, S, M, ukernel_2_bs_partial);
def_ukernel!(step_1x6, acc_1x6, store_1x6, 4, 4, B, S, M, ukernel_1_bs_partial);


def_ukernelxn!(step_3x4, acc_3x4, store_3x4, 12, 4, B, B, C, ukernel_n_bb);

def_ukernelxn!(step_3x4, acc_3x4, store_3x4, 12, 4, B, B, M, ukernel_3xn_bb_partial);
def_ukernelxn!(step_2x6, acc_2x6, store_2x6, 8, 6, B, B, M, ukernel_2xn_bb_partial);
def_ukernelxn!(step_1x6, acc_1x6, store_1x6, 4, 6, B, B, M, ukernel_1xn_bb_partial);

def_ukernelxn!(step_3x4, acc_3x4, store_3x4, 12, 4, B, S, C, ukernel_n_bs);

def_ukernelxn!(step_3x4, acc_3x4, store_3x4, 12, 4, B, S, M, ukernel_3xn_bs_partial);
def_ukernelxn!(step_2x6, acc_2x6, store_2x6, 8, 4, B, S, M, ukernel_2xn_bs_partial);
def_ukernelxn!(step_1x6, acc_1x6, store_1x6, 4, 4, B, S, M, ukernel_1xn_bs_partial);


// based on l1 prefetching scheme is from openblas impl for skylax
// see: https://github.com/OpenMathLib/OpenBLAS/pull/2300
// this is adapted to our ukernel of 3x4
// seems to stem from high bandwith of l1 cache (compared to other uarch e.g. haswell
// where the same l1 prefetching does not benefit as much)
pub(crate) unsafe fn ukernel_bb<F: MyFn, const BUF: bool>(
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
    let mut dim_arr = [d_arr[3]*16, k_iter, k_left, a_pft1_offset];
    let alt_arr = [-1.0f64, 1.0f64, -1.0f64, 1.0f64, -1.0f64, 1.0f64, -1.0f64, 1.0f64];
    let alt_buf = alt_arr.as_ptr();
    let mut cf = c;
    let mut c_buf = [TC::ZERO; 12 * 4];
    let c_cs = d_arr[3];
    if BUF {
        load_buf(c, d_arr[2], c_cs, &mut c_buf, 12, 4, 12);
        dim_arr[0] = 12*16;
        cf = c_buf.as_mut_ptr();
    }
    asm!(
        asm_vzeroall!(12,4),
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
        step_3x4!(4, B, B),

        "movq $64*4, {x4}",
        // divisiblity by 4
        "testq $3, {x0}",
        "cmovz {x1},{x4}",

        step_3x4!(4, B, B),

        "prefetcht1 ({x2})",

        "subq $64*3, {x2}",
        "addq {x4}, {x2}",

        step_3x4!(4, B, B),

        "prefetcht1 ({x5})",
        "addq $32, {x5}",

        "testq $63, {x0}",
        "cmovz {cx},{x2}",

        step_3x4!(4, B, B),

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
        step_3x4!(4, B, B),

        "add {x1}, {x2}",
        "dec {x0}",
        "jne 4b",
        "5:",
        permute_complex!(12, 4),
        "mov ({dim_arrx}),{x0}",
        "lea ({x0}, {x0}, 2), {x3}",
        "lea ({cx}, {x3},), {x1}",
        "lea ({x1}, {x3},), {x2}",
        // scale by alpha
        asm_alpha_scale!(12, 4),

        // 6 -> BETAZERO
        cum_seq!(acc_3x4,4,C),

        // 6 -> BETAZERO
        cum_seq!(store_3x4,4,C),

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
        for j in 0..4 {
            f.call(cf.add(j*12), 12);
        }
        store_buf(c, d_arr[2], c_cs, &c_buf, 12, 4, 12);
    } else {
        for j in 0..4 {
            f.call(cf.add(j*c_cs), 12);
        }
    }
}
