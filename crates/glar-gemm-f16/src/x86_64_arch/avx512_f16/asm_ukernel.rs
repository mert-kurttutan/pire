use seq_macro::seq;
use std::arch::asm;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use crate::UnaryFnC;
use half::f16;
use glar_base::{
    load_buf, store_buf, c_mem, def_ukernel_avx512, cum_seq,
    load_a_avx512, storep_avx512, acc_p_avx512,
};

type TS = TC;

const ZERO: f16 = f16::ZERO;

const ZERO_SCALAR: f16 = f16::ZERO;
const ONE_SCALAR: f16 = f16::ONE;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "vfmadd231ph ", $m0, ",%zmm0,%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr,2) => {
        concat!(
            "vmovdqu16 ", $m0, ", %zmm1 {{%k1}}", "\n",
            "vfmadd231ph %zmm1,%zmm0,%zmm", $r1, "\n",
        )
    };
    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vaddph ", $m0, ",%zmm", $r1, ",%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr, 1) => {
        concat!(
            "vaddph ", $m0, ",%zmm", $r1, ",%zmm", $r1, "{{%k1}}\n",
        )
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                #(
                    "vpxorq %zmm",r,",%zmm",r,",%zmm",r,"\n",
                )*
            )
        })
    }
}

macro_rules! vbroadcast {
    () => {
        "vpbroadcastw"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $m2:expr, $r3:expr) => {
        concat!(
            "vfmadd231ph ", $m2, "{{1to32}}", ", %zmm", $r1,", %zmm", $r3, "\n",
        ) 
    };
}

macro_rules! bd {
    (B, $i:tt) => {
        concat!($i, "*2({bx})")
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
    (S, 8) => {
        "0({x4}, {x2}, 2)"
    };
    (S, 9) => {
        "0({x5})"
    };
    (S, 10) => {
        "0({x5}, {x2})"
    };
    (S, 11) => {
        "0({x5}, {x2}, 2)"
    };
    (S, 12) => {
        "0({x5})"
    };
    (S, 13) => {
        "0({x5}, {x2})"
    };
    (S, 14) => {
        "0({x5}, {x2}, 2)"
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovaps ", $m0, ",%zmm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %zmm", $r1, ", ", $m0, "\n",
        )
    };
    (P, $r1:expr, $m0:expr) => {
        concat!(
            "vmovdqu16 %zmm", $r1, ", ", $m0, " {{%k1}}\n",
        )
    };
}


macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                vbroadcast!(), " ({alphax}),%zmm1", "\n",
                #(
                    "vmulph %zmm1, %zmm", r, ",%zmm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %zmm0\n",
            "vxorps %ymm1,%ymm1,%ymm1\n",
            "vcomish %xmm1,%xmm0\n",
            // 6 -> BETAZERO
            "je 6f",

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


macro_rules! init_ab {
    (B) => {
        concat!(
            "/* {x5} */", "\n",
            "/* {x4} */", "\n",
            "/* {x3} */", "\n",
            "/* {x2} */", "\n",
            "/* {x1} */", "\n",
            "mov 24({dim_arrx}),{x0}", "\n",
        )
    };
    (S) => {
        concat!(
            // mov cs_b to reg
            "mov ({dim_arrx}), {x1}", "\n",
            "mov 8({dim_arrx}), {x2}", "\n",
            "lea ({x2}, {x2}, 2), {x5}", "\n",
            "lea ({bx}, {x5}, 1), {x3}", "\n",
            "lea ({x3}, {x5}, 1), {x4}", "\n",
            "lea ({x4}, {x5}, 1), {x5}", "\n",
            "lea ({x5}, {x5}, 1), {x5}", "\n",

            "mov 24({dim_arrx}),{x0}", "\n",
        )
    };
}

macro_rules! vzero_kernel {

    () => {vzeroall!(2,31)};
}

macro_rules! alpha_scale {
    (3,9) => {alpha_scale_0!(5,31)};
    (3,8) => {alpha_scale_0!(5,28)};
    (3,7) => {alpha_scale_0!(5,25)};
    (3,6) => {alpha_scale_0!(5,22)};
    (3,5) => {alpha_scale_0!(5,19)};
    (3,4) => {alpha_scale_0!(5,16)};
    (3,3) => {alpha_scale_0!(5,13)};
    (3,2) => {alpha_scale_0!(5,10)};
    (3,1) => {alpha_scale_0!(5,7)};

    (2,15) => {alpha_scale_0!(2,31)};
    (2,14) => {alpha_scale_0!(2,29)};
    (2,13) => {alpha_scale_0!(2,27)};
    (2,12) => {alpha_scale_0!(2,25)};
    (2,11) => {alpha_scale_0!(2,23)};
    (2,10) => {alpha_scale_0!(2,21)};
    (2,9) => {alpha_scale_0!(2,19)};
    (2,8) => {alpha_scale_0!(2,17)};
    (2,7) => {alpha_scale_0!(2,15)};
    (2,6) => {alpha_scale_0!(2,13)};
    (2,5) => {alpha_scale_0!(2,11)};
    (2,4) => {alpha_scale_0!(2,9)};
    (2,3) => {alpha_scale_0!(2,7)};
    (2,2) => {alpha_scale_0!(2,5)};
    (2,1) => {alpha_scale_0!(2,3)};

    (1,15) => {alpha_scale_0!(17,31)};
    (1,14) => {alpha_scale_0!(17,30)};
    (1,13) => {alpha_scale_0!(17,29)};
    (1,12) => {alpha_scale_0!(17,28)};
    (1,11) => {alpha_scale_0!(17,27)};
    (1,10) => {alpha_scale_0!(17,26)};
    (1,9) => {alpha_scale_0!(17,25)};
    (1,8) => {alpha_scale_0!(17,24)};
    (1,7) => {alpha_scale_0!(17,23)};
    (1,6) => {alpha_scale_0!(17,22)};
    (1,5) => {alpha_scale_0!(17,21)};
    (1,4) => {alpha_scale_0!(17,20)};
    (1,3) => {alpha_scale_0!(17,19)};
    (1,2) => {alpha_scale_0!(17,18)};
    (1,1) => {alpha_scale_0!(17,17)};
}

macro_rules! inc_b {
    (S,15) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n add {x1},{x5} \n"
    };
    (S,14) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n add {x1},{x5} \n"
    };
    (S,13) => {
        "add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n add {x1},{x5} \n"
    };
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
            "add $2*", $nr, ", {bx}", "\n",
        )
    };
}

macro_rules! c_reg_2x15 {
    (0,0) => { 2 }; (1,0) => { 3 };
    (0,1) => { 4 }; (1,1) => { 5 };
    (0,2) => { 6 }; (1,2) => { 7 };
    (0,3) => { 8 }; (1,3) => { 9 };
    (0,4) => { 10 }; (1,4) => { 11 };
    (0,5) => { 12 }; (1,5) => { 13 };
    (0,6) => { 14 }; (1,6) => { 15 };
    (0,7) => { 16 }; (1,7) => { 17 };
    (0,8) => { 18 }; (1,8) => { 19 };
    (0,9) => { 20 }; (1,9) => { 21 };
    (0,10) => { 22 }; (1,10) => { 23 };
    (0,11) => { 24 }; (1,11) => { 25 };
    (0,12) => { 26 }; (1,12) => { 27 };
    (0,13) => { 28 }; (1,13) => { 29 };
    (0,14) => { 30 }; (1,14) => { 31 };
}

macro_rules! c_reg_1x15 {
    (0,0) => { 17 };
    (0,1) => { 18 };
    (0,2) => { 19 };
    (0,3) => { 20 };
    (0,4) => { 21 };
    (0,5) => { 22 };
    (0,6) => { 23 };
    (0,7) => { 24 };
    (0,8) => { 25 };
    (0,9) => { 26 };
    (0,10) => { 27 };
    (0,11) => { 28 };
    (0,12) => { 29 };
    (0,13) => { 30 };
    (0,14) => { 31 };
}

macro_rules! acc_2x15 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx512!($layout, c_mem!($ni), $q, c_reg_2x15!(0,$ni), c_reg_2x15!(1,$ni))
    };
}

macro_rules! store_2x15 {
    ($ni:tt, $layout:tt) => {
        storep_avx512!($layout, c_mem!($ni), c_reg_2x15!(0,$ni), c_reg_2x15!(1,$ni))
    };
}

macro_rules! acc_1x15 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx512!($layout, c_mem!($ni), $q, c_reg_1x15!(0,$ni))
    };
}

macro_rules! store_1x15 {
    ($ni:tt, $layout:tt) => {
        storep_avx512!($layout, c_mem!($ni), c_reg_1x15!(0,$ni))
    };
}

macro_rules! fmadd_2v {
    ($ni:tt, $m:expr) => {
        concat!(
            vfmadd!(0, $m, c_reg_2x15!(0,$ni)),
            vfmadd!(1, $m, c_reg_2x15!(1,$ni)),
        )
    };
}

macro_rules! fmadd_1v {
    ($ni:tt, $m:expr) => {
        concat!(
            vfmadd!(0, $m, c_reg_1x15!(0,$ni)),
        )
    };
}

// // ***************************** 96x9 ******************************* //
// macro_rules! step_96x9 {
// 	(9, B) => {
// 		concat!(
// 			load_a_avx512!(96, B),
// 			"addq $192, {ax} \n",
// 			fmadd_3v!(0, bd!(B, 0)),
// 			fmadd_3v!(1, bd!(B, 1)),
// 			"prefetcht0 384({ax}) \n",
// 			"prefetcht0 32({bx}) \n",
// 			fmadd_3v!(2, bd!(B, 2)),
// 			fmadd_3v!(3, bd!(B, 3)),
// 			"prefetcht0 448({ax}) \n",
// 			fmadd_3v!(4, bd!(B, 4)),
// 			fmadd_3v!(5, bd!(B, 5)),
// 			"prefetcht0 512({ax}) \n",
// 			fmadd_3v!(6, bd!(B, 6)),
// 			fmadd_3v!(7, bd!(B, 7)),
// 			fmadd_3v!(8, bd!(B, 8)),
// 			"addq $18, {bx} \n",
// 		)
        
// 	};
// 	($nr:tt, $a_layout:tt, $b_layout:tt) => {
// 		seq!(n in 0..$nr {
// 			concat!(
// 				load_a_avx512!($mr, $a_layout),
// 				inc_a!($a_layout,96),
// 				"prefetcht0 64({bx}) \n",
// 				#(
// 					fmadd_3v!(n, bd!($b_layout, n)),
// 				)*
// 				inc_b!($b_layout,$nr), 
// 			)
// 		})
// 	};
// }

// ***************************** 2x15 ******************************* //
macro_rules! step_2x15 {
    (15, B) => {
        concat!(
            load_a_avx512!(2),
            "addq $128, {ax} \n",
            fmadd_2v!(0, bd!(B, 0)),
            fmadd_2v!(1, bd!(B, 1)),
            "prefetcht0 256({ax}) \n",
            fmadd_2v!(2, bd!(B, 2)),
            "prefetcht0 64({bx}) \n",
            fmadd_2v!(3, bd!(B, 3)),
            fmadd_2v!(4, bd!(B, 4)),
            fmadd_2v!(5, bd!(B, 5)),
            fmadd_2v!(6, bd!(B, 6)),
            fmadd_2v!(7, bd!(B, 7)),
            fmadd_2v!(8, bd!(B, 8)),
            fmadd_2v!(9, bd!(B, 9)),
            fmadd_2v!(10, bd!(B, 10)),
            "prefetcht0 320({ax}) \n",
            fmadd_2v!(11, bd!(B, 11)),
            fmadd_2v!(12, bd!(B, 12)),
            fmadd_2v!(13, bd!(B, 13)),
            fmadd_2v!(14, bd!(B, 14)),
            "addq $30, {bx} \n",
        )
        
    };
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx512!(2),
                "addq $128, {ax} \n",
                "prefetcht0 64({bx}) \n",
                #(
                    fmadd_2v!(n, bd!($b_layout, n)),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}
// ***************************** 1x15 ******************************* //
macro_rules! step_1x15 {
    (15, B) => {
        concat!(
            load_a_avx512!(1),
            "addq $64, {ax} \n",
            fmadd_1v!(0, bd!(B, 0)),
            fmadd_1v!(1, bd!(B, 1)),
            "prefetcht0 256({ax}) \n",
            fmadd_1v!(2, bd!(B, 2)),
            "prefetcht0 64({bx}) \n",
            fmadd_1v!(3, bd!(B, 3)),
            fmadd_1v!(4, bd!(B, 4)),
            fmadd_1v!(5, bd!(B, 5)),
            fmadd_1v!(6, bd!(B, 6)),
            fmadd_1v!(7, bd!(B, 7)),
            fmadd_1v!(8, bd!(B, 8)),
            fmadd_1v!(9, bd!(B, 9)),
            fmadd_1v!(10, bd!(B, 10)),
            "prefetcht0 320({ax}) \n",
            fmadd_1v!(11, bd!(B, 11)),
            fmadd_1v!(12, bd!(B, 12)),
            fmadd_1v!(13, bd!(B, 13)),
            fmadd_1v!(14, bd!(B, 14)),
            "addq $30, {bx} \n",
        )
        
    };
    ($nr:tt, $b_layout:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx512!(1),
                "addq $64, {ax} \n",
                "prefetcht0 64({bx}) \n",
                #(
                    fmadd_1v!(n, bd!($b_layout, n)),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! mask_ptr {
    (P, $m:expr, $nm:ident, $mask_ptr:ident) => {
        let $nm = if $m % VS == 0 && $m > 0 { 0xFFFFFFFF } else { (1_u32 << ($m % VS)) - 1 };
        let $mask_ptr = (&$nm) as *const u32;
    };
    (C, $m:expr, $nm:ident, $mask_ptr:ident) => {
        let $nm = 0xFFFFFFFF_u32;
        let $mask_ptr = (&$nm) as *const u32;
    };
}

macro_rules! load_mask {
    (P) => {
        "kmovd ({maskx}), %k1"
    };
    (C) => {
        "/* {maskx} */"
    }
}

def_ukernel_avx512!(1, step_2x15, acc_2x15, store_2x15, 2, 15, 15, 16, B, P, ukernel_2_bbp);
def_ukernel_avx512!(1, step_1x15, acc_1x15, store_1x15, 1, 15, 15, 16, B, P, ukernel_1_bbp);

def_ukernel_avx512!(1, step_2x15, acc_2x15, store_2x15, 2, 15, 15, 16, S, C, ukernel_bsc);

def_ukernel_avx512!(1, step_2x15, acc_2x15, store_2x15, 2, 15, 15, 16, S, P, ukernel_2_bsp);
def_ukernel_avx512!(1, step_1x15, acc_1x15, store_1x15, 1, 15, 15, 16, S, P, ukernel_1_bsp);


def_ukernel_avx512!(1, step_2x15, acc_2x15, store_2x15, 2, 15, 1, 15, B, C, ukernel_n_bbc);

def_ukernel_avx512!(1, step_2x15, acc_2x15, store_2x15, 2, 15, 1, 15, B, P, ukernel_2xn_bbp);
def_ukernel_avx512!(1, step_1x15, acc_1x15, store_1x15, 1, 15, 1, 15, B, P, ukernel_1xn_bbp);

def_ukernel_avx512!(1, step_2x15, acc_2x15, store_2x15, 2, 15, 1, 15, S, C, ukernel_n_bsc);

def_ukernel_avx512!(1, step_2x15, acc_2x15, store_2x15, 2, 15, 1, 15, S, P, ukernel_2xn_bsp);
def_ukernel_avx512!(1, step_1x15, acc_1x15, store_1x15, 1, 15, 1, 15, S, P, ukernel_1xn_bsp);



pub(crate) unsafe fn ukernel_bbc<F: UnaryFnC, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const TA, beta: *const TB,
    k: usize,
    d_arr: [usize; 3], c_cs: usize,
    a_pft1_offset: usize, _n: usize,
    f: F,
) {
    let k_l0 = k % 16;
    let k_l = if k_l0 == 0 {16} else {k_l0};
    let k_i = (k - k_l) / 4;
    let mut dim_arr = [c_cs*TC_SIZE, k_i, k_l, a_pft1_offset];
    let mut cf = c;
    let mut c_buf = [f16::ZERO; 64 * 15];
    if BUF {
        load_buf(c, d_arr[2], c_cs, &mut c_buf, 64, 15, 64);
        dim_arr[0] = 64*TC_SIZE;
        cf = c_buf.as_mut_ptr();
    }
    asm!(
        vzero_kernel!(),
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
        step_2x15!(15, B),

        "movq $64*4, {x4}",
        // divisiblity by 4
        "testq $3, {x0}",
        "cmovz {x1},{x4}",

        step_2x15!(15, B),

        "prefetcht1 ({x2})",

        "subq $64*3, {x2}",
        "addq {x4}, {x2}",

        step_2x15!(15, B),

        "prefetcht1 ({x5})",
        "addq $64, {x5}",

        "testq $63, {x0}",
        "cmovz {cx},{x2}",

        step_2x15!(15, B),

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
        step_2x15!(15, B),

        "add {x1}, {x2}",
        "dec {x0}",
        "jne 4b",
        "5:",
        "mov ({dim_arrx}),{x0}",
        "lea ({x0}, {x0}, 2), {x4}",
        "lea ({cx}, {x4},), {x1}",
        "lea ({x1}, {x4},), {x2}",
        "lea ({x2}, {x4},), {x3}",
        "lea ({x3}, {x4},), {x4}",
        // scale by alpha
        alpha_scale!(2, 15),
        load_beta!(),

        cum_seq!(acc_2x15,15,C,2),

        // 6 -> BETAZERO
        "6:",
        cum_seq!(store_2x15,15,C),

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
        for j in 0..15 {
            f.call(cf.add(j*64), 64);
        }
        store_buf(c, d_arr[2], c_cs, &c_buf, 64, 15, 64);
    } else {
        for j in 0..15 {
            f.call(cf.add(j*c_cs), 64);
        }
    }
}
