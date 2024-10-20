use seq_macro::seq;
use std::arch::asm;
use std::arch::x86_64::_mm_prefetch;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use crate::UnaryFnC;
use glar_base::{load_buf, store_buf, c_mem};

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "vfmadd231pd ", $m0, ",%zmm0,%zmm", $r1, "\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            // "vmovupd ", $m0, ", %zmm1", ", %zmm2",  "\n",
            "vmovupd ", $m0, ", %zmm1 {{%k1}}", "\n",
            "vfmadd231pd %zmm1,%zmm0,%zmm", $r1, "\n",
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
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231pd %zmm", $r1, ", %zmm", $r2,", %zmm", $r3, "\n",
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

macro_rules! asm_alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                vbroadcast!(), " ({alphax}), %zmm1", "\n",
                #(
                    "vmulpd %zmm1, %zmm", r, ",%zmm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %zmm0\n",
            "vxorpd %ymm3,%ymm3,%ymm3\n",
            "vucomisd %xmm3,%xmm0\n",
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
    (12) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
            "lea ({x0}, {x0}, 2), {x3}", "\n",
            "lea ({cx}, {x3},), {x1}", "\n",
            "lea ({x1}, {x3},), {x2}", "\n",
            "lea ({x2}, {x3},), {x3}", "\n",
        )
    };
    (11) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
            "lea ({x0}, {x0}, 2), {x3}", "\n",
            "lea ({cx}, {x3},), {x1}", "\n",
            "lea ({x1}, {x3},), {x2}", "\n",
            "lea ({x2}, {x3},), {x3}", "\n",
        )
    };
    (10) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
            "lea ({x0}, {x0}, 2), {x3}", "\n",
            "lea ({cx}, {x3},), {x1}", "\n",
            "lea ({x1}, {x3},), {x2}", "\n",
            "lea ({x2}, {x3},), {x3}", "\n",
        )
    };
    (9) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
            "lea ({x0}, {x0}, 2), {x3}", "\n",
            "lea ({cx}, {x3},), {x1}", "\n",
            "lea ({x1}, {x3},), {x2}", "\n",
        )
    };
    (8) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
            "lea ({x0}, {x0}, 2), {x3}", "\n",
            "lea ({cx}, {x3},), {x1}", "\n",
            "lea ({x1}, {x3},), {x2}", "\n",
        )
    };
    (7) => {
        concat!(
            "mov 16({dim_arrx}),{x0}", "\n",
            "lea ({x0}, {x0}, 2), {x3}", "\n",
            "lea ({cx}, {x3},), {x1}", "\n",
            "lea ({x1}, {x3},), {x2}", "\n",
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

    (24,8) => {vzeroall!(8,31)};
    (24,7) => {vzeroall!(8,28)};
    (24,6) => {vzeroall!(8,25)};
    (24,5) => {vzeroall!(8,22)};
    (24,4) => {vzeroall!(8,19)};
    (24,3) => {vzeroall!(8,16)};
    (24,2) => {vzeroall!(8,13)};
    (24,1) => {vzeroall!(8,10)};

    (16,12) => {vzeroall!(8,31)};
    (16,11) => {vzeroall!(8,29)};
    (16,10) => {vzeroall!(8,27)};
    (16,9) => {vzeroall!(8,25)};
    (16,8) => {vzeroall!(8,23)};
    (16,7) => {vzeroall!(8,21)};
    (16,6) => {vzeroall!(8,19)};
    (16,5) => {vzeroall!(8,17)};
    (16,4) => {vzeroall!(8,15)};
    (16,3) => {vzeroall!(8,13)};
    (16,2) => {vzeroall!(8,11)};
    (16,1) => {vzeroall!(8,9)};

    (8,12) => {vzeroall!(20,31)};
    (8,11) => {vzeroall!(20,30)};
    (8,10) => {vzeroall!(20,29)};
    (8,9) => {vzeroall!(20,28)};
    (8,8) => {vzeroall!(20,27)};
    (8,7) => {vzeroall!(20,26)};
    (8,6) => {vzeroall!(20,25)};
    (8,5) => {vzeroall!(20,24)};
    (8,4) => {vzeroall!(20,23)};
    (8,3) => {vzeroall!(20,22)};
    (8,2) => {vzeroall!(20,21)};
    (8,1) => {vzeroall!(20,20)};
}

macro_rules! asm_alpha_scale {
    (24, 8) => {asm_alpha_scale_0!(8,31)};
    (24, 7) => {asm_alpha_scale_0!(8,28)};
    (24, 6) => {asm_alpha_scale_0!(8,25)};
    (24, 5) => {asm_alpha_scale_0!(8,22)};
    (24, 4) => {asm_alpha_scale_0!(8,19)};
    (24, 3) => {asm_alpha_scale_0!(8,16)};
    (24, 2) => {asm_alpha_scale_0!(8,13)};
    (24, 1) => {asm_alpha_scale_0!(8,10)};

    (16, 12) => {asm_alpha_scale_0!(8,31)};
    (16, 11) => {asm_alpha_scale_0!(8,29)};
    (16, 10) => {asm_alpha_scale_0!(8,27)};
    (16, 9) => {asm_alpha_scale_0!(8,25)};
    (16, 8) => {asm_alpha_scale_0!(8,23)};
    (16, 7) => {asm_alpha_scale_0!(8,21)};
    (16, 6) => {asm_alpha_scale_0!(8,19)};
    (16, 5) => {asm_alpha_scale_0!(8,17)};
    (16, 4) => {asm_alpha_scale_0!(8,15)};
    (16, 3) => {asm_alpha_scale_0!(8,13)};
    (16, 2) => {asm_alpha_scale_0!(8,11)};
    (16, 1) => {asm_alpha_scale_0!(8,9)};

    (8, 12) => {asm_alpha_scale_0!(20,31)};
    (8, 11) => {asm_alpha_scale_0!(20,30)};
    (8, 10) => {asm_alpha_scale_0!(20,29)};
    (8, 9) => {asm_alpha_scale_0!(20,28)};
    (8, 8) => {asm_alpha_scale_0!(20,27)};
    (8, 7) => {asm_alpha_scale_0!(20,26)};
    (8, 6) => {asm_alpha_scale_0!(20,25)};
    (8, 5) => {asm_alpha_scale_0!(20,24)};
    (8, 4) => {asm_alpha_scale_0!(20,23)};
    (8, 3) => {asm_alpha_scale_0!(20,22)};
    (8, 2) => {asm_alpha_scale_0!(20,21)};
    (8, 1) => {asm_alpha_scale_0!(20,20)};
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
    (S,12) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n"
    };
    (S,11) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n"
    };
    (S,10) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n"
    };
    (S,9) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n"
    };
    (S,8) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n"
    };
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

macro_rules! c_reg_3x8 {
    (0,0) => { 8 }; (1,0) => { 9 }; (2,0) => { 10 };
    (0,1) => { 11 }; (1,1) => { 12 }; (2,1) => { 13 };
    (0,2) => { 14 }; (1,2) => { 15 }; (2,2) => { 16 };
    (0,3) => { 17 }; (1,3) => { 18 }; (2,3) => { 19 };
    (0,4) => { 20 }; (1,4) => { 21 }; (2,4) => { 22 };
    (0,5) => { 23 }; (1,5) => { 24 }; (2,5) => { 25 };
    (0,6) => { 26 }; (1,6) => { 27 }; (2,6) => { 28 };
    (0,7) => { 29 }; (1,7) => { 30 }; (2,7) => { 31 };
}

macro_rules! c_reg_2x12 {
    (0,0) => { 8 }; (1,0) => { 9 };
    (0,1) => { 10 }; (1,1) => { 11 };
    (0,2) => { 12 }; (1,2) => { 13 };
    (0,3) => { 14 }; (1,3) => { 15 };
    (0,4) => { 16 }; (1,4) => { 17 };
    (0,5) => { 18 }; (1,5) => { 19 };
    (0,6) => { 20 }; (1,6) => { 21 };
    (0,7) => { 22 }; (1,7) => { 23 };
    (0,8) => { 24 }; (1,8) => { 25 };
    (0,9) => { 26 }; (1,9) => { 27 };
    (0,10) => { 28 }; (1,10) => { 29 };
    (0,11) => { 30 }; (1,11) => { 31 };
}

macro_rules! c_reg_1x12 {
    (0,0) => { 20 };
    (0,1) => { 21 };
    (0,2) => { 22 };
    (0,3) => { 23 };
    (0,4) => { 24 };
    (0,5) => { 25 };
    (0,6) => { 26 };
    (0,7) => { 27 };
    (0,8) => { 28 };
    (0,9) => { 29 };
    (0,10) => { 30 };
    (0,11) => { 31 };
}

macro_rules! acc_3x8 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_3x8!(0,$ni), c_reg_3x8!(1,$ni), c_reg_3x8!(2,$ni))
    };
}

macro_rules! store_3x8 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_3x8!(0,$ni), c_reg_3x8!(1,$ni), c_reg_3x8!(2,$ni))
    };
}

macro_rules! acc_2x12 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_2x12!(0,$ni), c_reg_2x12!(1,$ni))
    };
}

macro_rules! store_2x12 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_2x12!(0,$ni), c_reg_2x12!(1,$ni))
    };
}

macro_rules! acc_1x12 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_1x12!(0,$ni))
    };
}

macro_rules! store_1x12 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_1x12!(0,$ni))
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
    (S, 0, $r:expr) => {
        concat!(
            vbroadcast!(), " ({bx}),%zmm", $r, "\n",
        )
    };
    (S, 1, $r:expr) => {
        concat!(
            vbroadcast!(), " ({bx},{x2},1),%zmm", $r, "\n",
        )
    };
    (S, 2, $r:expr) => {
        concat!(
            vbroadcast!(), " ({bx},{x2},2),%zmm", $r, "\n",
        )
    };
    (S, 3, $r:expr) => {
        concat!(
            "prefetcht0 64({x3}) \n",
            vbroadcast!(), " ({x3}),%zmm", $r, "\n",
        )
    };
    (S, 4, $r:expr) => {
        concat!(
            vbroadcast!(), " ({x3},{x2},1),%zmm", $r, "\n",
        )
    };
    (S, 5, $r:expr) => {
        concat!(
            vbroadcast!(), " ({x3},{x2},2),%zmm", $r, "\n",
        )
    };
    (S, 6, $r:expr) => {
        concat!(
            "prefetcht0 64({x4}) \n",
            vbroadcast!(), " ({x4}),%zmm", $r, "\n",
        )
    };
    (S, 7, $r:expr) => {
        concat!(
            vbroadcast!(), " ({x4},{x2},1),%zmm", $r, "\n",
        )
    };
    (S, 8, $r:expr) => {
        concat!(
            vbroadcast!(), " ({x4},{x2},2),%zmm", $r, "\n",
        )
    };
    (S, 9, $r:expr) => {
        concat!(
            vbroadcast!(), " ({x5}),%zmm", $r, "\n",
        )
    };
    (S, 10, $r:expr) => {
        concat!(
            vbroadcast!(), " ({x5},{x2},1),%zmm", $r, "\n",
        )
    };
    (S, 11, $r:expr) => {
        concat!(
            vbroadcast!(), " ({x5},{x2},2),%zmm", $r, "\n",
        )
    };
    (B, $N:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", $N, "*8({bx}), %zmm", $r, "\n",
        )
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
    ($ni:tt) => {
        concat!(
            vfmadd!(0, b_num_3x8!($ni), c_reg_3x8!(0,$ni)),
            vfmadd!(1, b_num_3x8!($ni), c_reg_3x8!(1,$ni)),
            vfmadd!(2, b_num_3x8!($ni), c_reg_3x8!(2,$ni)),
        )
    };
}

macro_rules! fmadd_2v {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, b_num_2x12!($ni), c_reg_2x12!(0,$ni)),
            vfmadd!(1, b_num_2x12!($ni), c_reg_2x12!(1,$ni)),
        )
    };
}

macro_rules! fmadd_1v {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, b_num_1x12!($ni), c_reg_1x12!(0,$ni)),
        )
    };
}


macro_rules! b_num_3x8 {
    (0) => {3};
    (1) => {4};
    (2) => {5};
    (3) => {6};
    (4) => {7};
    (5) => {3};
    (6) => {4};
    (7) => {5};
}

macro_rules! b_num_2x12 {
    (0) => {2};
    (1) => {3};
    (2) => {4};
    (3) => {5};
    (4) => {6};
    (5) => {7};
    (6) => {2};
    (7) => {3};
    (8) => {4};
    (9) => {5};
    (10) => {6};
    (11) => {7};
}

macro_rules! b_num_1x12 {
    (0) => {1};
    (1) => {2};
    (2) => {3};
    (3) => {4};
    (4) => {5};
    (5) => {6};
    (6) => {7};
    (7) => {8};
    (8) => {9};
    (9) => {10};
    (10) => {11};
    (11) => {12};
}

// ***************************** 3x8 ******************************* //
macro_rules! step_3x8 {
    (8, B, B) => {
        concat!(
            load_a!(24, B),
            "addq $192, {ax} \n",
            load_b!(B, 0, 3),
            fmadd_3v!(0),
            load_b!(B, 1, 4),
            fmadd_3v!(1),
            "prefetcht0 384({ax}) \n",
            "prefetcht0 64({bx}) \n",
            load_b!(B, 2, 5),
            fmadd_3v!(2),
            load_b!(B, 3, 6),
            fmadd_3v!(3),
            "prefetcht0 448({ax}) \n",
            load_b!(B, 4, 7),
            fmadd_3v!(4),
            load_b!(B, 5, 3),
            fmadd_3v!(5),
            "prefetcht0 512({ax}) \n",
            load_b!(B, 6, 4),
            fmadd_3v!(6),
            load_b!(B, 7, 5),
            fmadd_3v!(7),
            "addq $64, {bx} \n",
        )
        
    };
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(24, $a_layout),
                inc_a!($a_layout, 24),
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, b_num_3x8!(n)),
                    fmadd_3v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 2x12 ******************************* //
macro_rules! step_2x12 {
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(16, $a_layout),
                inc_a!($a_layout, 16),
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, b_num_2x12!(n)),
                    fmadd_2v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 1x12 ******************************* //
macro_rules! step_1x12 {
    ($nr:tt, $a_layout:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(8, $a_layout),
                inc_a!($a_layout, 8),
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, b_num_1x12!(n)),
                    fmadd_1v!(n),
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
            _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
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
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize,
            f: F,
        ) {
            mask_ptr!($is_partial, m, x);
            let mask_ptr = (&x) as *const u8;
            let (k_i, k_l) = (k / 4, k % 4);
            let mut dim_arr = [d_arr[0]*8, d_arr[1]*8, d_arr[3]*TC_SIZE, k_i, k_l];
            let mut cf = c;
            let mut c_buf = [0f64;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, $mr);
                dim_arr[2] = $mr*TC_SIZE;
                cf = c_buf.as_mut_ptr();
            }
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
                asm_c_load!($nr),
                // scale by alpha
                asm_alpha_scale!($mr, $nr),

                load_beta!(),

                load_mask_ptr_asm!($is_partial),				
                // 6 -> BETAZERO
                "je 6f",
                cum_seq!($acc_macro,$nr,$is_partial),

                // 6 -> BETAZERO
                "6:",
                cum_seq!($store_macro,$nr,$is_partial),
                
                // 7 -> DDONE
                "7:",
                // "vzeroupper",
                ax = inout(reg) a => _,
                bx = inout(reg) b => _,
                cx = inout(reg) cf => _,
                dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                alphax = inout(reg) alpha => _,
                betax = inout(reg) beta => _,
                maskx = inout(reg) mask_ptr => _,
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
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize, n: usize,
            f: F,
        ) {
            mask_ptr!($is_partial, m, x);
            let mask_ptr = (&x) as *const u8;
            let (k_i, k_l) = (k / 4, k % 4);
            let mut dim_arr = [d_arr[0]*8, d_arr[1]*8, d_arr[3]*TC_SIZE, k_i, k_l];
            let mut cf = c;
            let mut c_buf = [0f64;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, $mr);
                dim_arr[2] = $mr*TC_SIZE;
                cf = c_buf.as_mut_ptr();
            }
            let _ = 'blk: {
                seq!(ni in 1..$nr {
                    if ni == n {
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
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),

                            load_beta!(),

                            load_mask_ptr_asm!($is_partial),				
                            // 6 -> BETAZERO
                            "je 6f",
                            cum_seq!($acc_macro,ni,$is_partial),

                            // 6 -> BETAZERO
                            "6:",
                            cum_seq!($store_macro,ni,$is_partial),
                            
                            // 7 -> DDONE
                            "7:",
                            // "vzeroupper",
                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) cf => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            betax = inout(reg) beta => _,
                            maskx = inout(reg) mask_ptr => _,
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

def_ukernel!(step_3x8, acc_3x8, store_3x8, 24, 8, B, B, M, ukernel_3_bb_partial);
def_ukernel!(step_2x12, acc_2x12, store_2x12, 16, 8, B, B, M, ukernel_2_bb_partial);
def_ukernel!(step_1x12, acc_1x12, store_1x12, 8, 8, B, B, M, ukernel_1_bb_partial);

def_ukernel!(step_3x8, acc_3x8, store_3x8, 24, 8, B, S, C, ukernel_bs);

def_ukernel!(step_3x8, acc_3x8, store_3x8, 24, 8, B, S, M, ukernel_3_bs_partial);
def_ukernel!(step_2x12, acc_2x12, store_2x12, 16, 8, B, S, M, ukernel_2_bs_partial);
def_ukernel!(step_1x12, acc_1x12, store_1x12, 8, 8, B, S, M, ukernel_1_bs_partial);


def_ukernelxn!(step_3x8, acc_3x8, store_3x8, 24, 8, B, B, C, ukernel_n_bb);

def_ukernelxn!(step_3x8, acc_3x8, store_3x8, 24, 8, B, B, M, ukernel_3xn_bb_partial);
def_ukernelxn!(step_2x12, acc_2x12, store_2x12, 16, 8, B, B, M, ukernel_2xn_bb_partial);
def_ukernelxn!(step_1x12, acc_1x12, store_1x12, 8, 8, B, B, M, ukernel_1xn_bb_partial);

def_ukernelxn!(step_3x8, acc_3x8, store_3x8, 24, 8, B, S, C, ukernel_n_bs);

def_ukernelxn!(step_3x8, acc_3x8, store_3x8, 24, 8, B, S, M, ukernel_3xn_bs_partial);
def_ukernelxn!(step_2x12, acc_2x12, store_2x12, 16, 8, B, S, M, ukernel_2xn_bs_partial);
def_ukernelxn!(step_1x12, acc_1x12, store_1x12, 8, 8, B, S, M, ukernel_1xn_bs_partial);



// based on l1 prefetching scheme is from openblas impl for skylax
// see: httpd://github.com/OpenMathLib/OpenBLAS/pull/2300
// this is adapted to our ukernel of 3x8
// seems to stem from high bandwith of l1 cache (compared to other uarch e.g. haswell
// where the same l1 prefetching does not benefit as much)
pub(crate) unsafe fn ukernel_bb<F: UnaryFnC, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const TA, beta: *const TB,
    k: usize,
    d_arr: [usize; 4],
    a_pft1_offset: usize,
    f: F,
) {
    let k_l0 = k % 8;
    let k_l = if k_l0 == 0 {8} else {k_l0};
    let k_i = (k - k_l) / 4;
    let mut dim_arr = [d_arr[3]*TC_SIZE, k_i, k_l, a_pft1_offset];
    let mut cf = c;
    let mut c_buf = [0f64; 24 * 8];
    let c_cs = d_arr[3];
    if BUF {
        load_buf(c, d_arr[2], c_cs, &mut c_buf, 24, 8, 24);
        dim_arr[0] = 24*TC_SIZE;
        cf = c_buf.as_mut_ptr();
    }
    asm!(
        asm_vzeroall!(24,8),
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
        step_3x8!(8, B, B),

        "movq $64*4, {x4}",
        // divisiblity by 4
        "testq $3, {x0}",
        "cmovz {x1},{x4}",

        step_3x8!(8, B, B),

        "prefetcht1 ({x2})",

        "subq $64*3, {x2}",
        "addq {x4}, {x2}",

        step_3x8!(8, B, B),

        "prefetcht1 ({x5})",
        "addq $64, {x5}",

        "testq $63, {x0}",
        "cmovz {cx},{x2}",

        step_3x8!(8, B, B),

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
        step_3x8!(8, B, B),

        "add {x1}, {x2}",
        "dec {x0}",
        "jne 4b",
        "5:",
        "mov ({dim_arrx}),{x0}",
        "lea ({x0}, {x0}, 2), {x3}",
        "lea ({cx}, {x3},), {x1}",
        "lea ({x1}, {x3},), {x2}",
        // scale by alpha
        asm_alpha_scale!(24, 8),
        load_beta!(),

        // 6 -> BETAZERO
        "je 6f",
        cum_seq!(acc_3x8,8,C),

        // 6 -> BETAZERO
        "6:",
        cum_seq!(store_3x8,8,C),

        "7:",
        ax = inout(reg) a => _, 
        bx = inout(reg) b => _, 
        cx = inout(reg) cf => _,
        dim_arrx = inout(reg) dim_arr.as_ptr() => _, 
        alphax = inout(reg) alpha => _, 
        betax = inout(reg) beta => _, 
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
        for j in 0..8 {
            f.call(cf.add(j*24), 24);
        }
        store_buf(c, d_arr[2], c_cs, &c_buf, 24, 8, 24);
    } else {
        for j in 0..8 {
            f.call(cf.add(j*c_cs), 24);
        }
    }
}