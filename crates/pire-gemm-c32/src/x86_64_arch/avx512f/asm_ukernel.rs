use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_avx512,
    init_ab,
    def_ukernel_avx512_2, init_ab_2,
    mem,
    acc_3, store_3, acc_2, store_2, acc_1, store_1,
};

type TS = TC;

const ZERO_SCALAR: TC = TC::ZERO;
const ONE_SCALAR: TC = TC::ONE;

macro_rules! vs {
    () => { "0x40" };
}

macro_rules! v_i {
    ($m:tt, $i:tt) => { concat!($i, "*0x40+" , $m) };
}
macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "vaddps ", $m0, ",%zmm", $r1, ",%zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr,1) => {
        concat!(
            "vmovupd ", $m0, ", %zmm2 {{%k1}}", "\n",
            "vaddps %zmm2, %zmm", $r1, ", %zmm", $r1, "\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "vmovupd ", $m0, ", %zmm3", "\n",
            complex_mul!(3, 4, 0),
            "vaddps %zmm3, %zmm", $r1, ", %zmm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr,2) => {
        concat!(
            "vmovupd ", $m0, ", %zmm3 {{%k1}}", "\n",
            complex_mul!(3, 4, 0),
            "vaddps %zmm3, %zmm", $r1, ", %zmm", $r1, "\n",
        ) 
    };
}

macro_rules! c_load {
    () => {
        concat!(
            permute_complex!(),
            "mov 16({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x3}\n",
            "lea ({cx}, {x3},), {x1}\n",
            "lea ({x1}, {x3},), {x2}\n",
            "lea ({x2}, {x3},), {x3}\n",
        )
    };
}

macro_rules! c_load_2 {
    () => {
        concat!(
            permute_complex!(),
            "mov ({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x3}\n",
            "lea ({cx}, {x3},), {x1}\n",
            "lea ({x1}, {x3},), {x2}\n",
            "lea ({x2}, {x3},), {x3}\n",
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

macro_rules! vbroadcast {
    () => { "vbroadcastss" };
}

macro_rules! vfmadd {
    ($r1:expr, $b1:expr, $b2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vfmadd231ps ", "%zmm", $b1, ", %zmm", $r1,", %zmm", $r3, "\n",
            "vfmadd231ps ", "%zmm", $b2, ", %zmm", $r1,", %zmm", $r4, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovaps ", mem!($m0, concat!("0x40*", $r1)), ", %zmm", $r1, "\n",
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
            "vmovupd %zmm", $r1, ", ", $m0, " {{%k1}}\n",
        )
    };
}


macro_rules! complex_mul {
    ($r0:tt, $rt:tt, $rs:tt) => {
        concat!(
            "vpermilps $0xb1, %zmm", $r0, ", %zmm", $rt, "\n",
            "vmulps %zmm1, %zmm", $r0, ", %zmm", $r0, "\n",
            "vmulps %zmm2, %zmm", $rt, ", %zmm", $rt, "\n",
            "vfmaddsub231ps %zmm0, %zmm", $r0, ", %zmm", $rt, "\n",
            "vmovaps %zmm", $rt, ", %zmm", $r0, "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    () => {
        concat!(
            vbroadcast!(), " ({alphax}), %zmm1 \n",
            vbroadcast!(), " 4({alphax}), %zmm2 \n",

            complex_mul!(8,9,0),
            complex_mul!(10,11,0),
            complex_mul!(12,13,0),
            complex_mul!(14,15,0),
            complex_mul!(16,17,0),
            complex_mul!(18,19,0),
            complex_mul!(20,21,0),
            complex_mul!(22,23,0),
            complex_mul!(24,25,0),
            complex_mul!(26,27,0),
            complex_mul!(28,29,0),
            complex_mul!(30,31,0),
        )
    };
}

macro_rules! v_to_c {
    ($r0:tt, $r1:tt, $rs:tt) => {
        concat!(
            "vpermilps $0xb1, %zmm", $r1, ", %zmm", $r1, "\n",
            // "vfmadd231ps %zmm", $r1, ", %zmm", $rs, ", %zmm", $r0, "\n",
            "vfmaddsub231ps %zmm0, %zmm", $r0, ", %zmm", $r1, "\n",
            "vmovaps %zmm", $r1, ", %zmm", $r0, "\n",
        )
    }
}

macro_rules! permute_complex {
    () => {
        concat!(
            // permute even and odd elements
            // "vbroadcastsd ({alternate}), %zmm0 \n",
            "mov $0x3f800000, {ax:e} \n",
            "movd {ax:e}, %xmm0 \n",
            "vbroadcastss %xmm0, %zmm0 \n",
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
}

macro_rules! load_beta {
    () => {
        concat!(
            "vbroadcastss ({betax}), %zmm1\n",
            "vbroadcastss 4({betax}), %zmm2\n",
        )
    }
}

macro_rules! vzero_kernel {
    () => {vzeroall!(8,31)};
}

macro_rules! alpha_scale {
    () => {alpha_scale_0!()};
}

macro_rules! inc_b {
    (S,$nr:tt) => { "add {x1},{bx} \n add {x1},{x3} \n" };
    (B,$nr:tt) => {
        concat!(
            "add $8*", $nr, ", {bx}", "\n",
        )
    };
    ($nr:tt) => {
        concat!(
            "add $8*", $nr, ", {bx}", "\n",
        )
    };
}

macro_rules! load_b {
    (S, 0, $r1:expr, $r2:expr) => {
        concat!(
            "prefetcht0 64({bx},{x1},8) \n",
            vbroadcast!(), " ({bx}),%zmm", $r1, "\n",
            vbroadcast!(), " 4({bx}),%zmm", $r2, "\n",
        )
    };
    (S, 1, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx},{x2},1),%zmm", $r1, "\n",
            vbroadcast!(), " 4({bx},{x2},1),%zmm", $r2, "\n",
        )
    };
    (S, 2, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx},{x2},2),%zmm", $r1, "\n",
            vbroadcast!(), " 4({bx},{x2},2),%zmm", $r2, "\n",
        )
    };
    (S, 3, $r1:expr, $r2:expr) => {
        concat!(
            "prefetcht0 64({x3},{x1},8) \n",
            vbroadcast!(), " ({x3}),%zmm", $r1, "\n",
            vbroadcast!(), " 4({x3}),%zmm", $r2, "\n",
        )
    };
    (S, 4, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({x3},{x2},1),%zmm", $r1, "\n",
            vbroadcast!(), " 4({x3},{x2},1),%zmm", $r2, "\n",
        )
    };
    (S, 5, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({x3},{x2},2),%zmm", $r1, "\n",
            vbroadcast!(), " 4({x3},{x2},2),%zmm", $r2, "\n",
        )
    };
    (B, $N:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ", $N, "*8({bx}), %zmm", $r1, "\n",
            vbroadcast!(), " ", $N, "*8+4({bx}), %zmm", $r2, "\n",
        )
    };
}

macro_rules! fmadd_3 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_3!($ni,0), br_3!($ni,1), cr!(0,$ni), dr!(0,$ni)),
            vfmadd!(1, br_3!($ni,0), br_3!($ni,1), cr!(1,$ni), dr!(1,$ni)),
            vfmadd!(2, br_3!($ni,0), br_3!($ni,1), cr!(2,$ni), dr!(2,$ni)),
        )
    };
}

macro_rules! fmadd_2 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_2!($ni,0), br_2!($ni,1), cr!(0,$ni), dr!(0,$ni)),
            vfmadd!(1, br_2!($ni,0), br_2!($ni,1), cr!(1,$ni), dr!(1,$ni)),
        )
    };
}

macro_rules! fmadd_1 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_1!($ni,0), br_1!($ni,1), cr!(0,$ni), dr!(0,$ni)),
        )
    };
}

macro_rules! br_3 {
    (0,0) => {3}; (0,1) => {4};
    (1,0) => {5}; (1,1) => {6};
    (2,0) => {7}; (2,1) => {3};
    (3,0) => {4}; (3,1) => {5};
}

macro_rules! br_2 {
    (0,0) => {2}; (0,1) => {3};
    (1,0) => {4}; (1,1) => {5};
    (2,0) => {6}; (2,1) => {7};
    (3,0) => {2}; (3,1) => {3};
    (4,0) => {4}; (4,1) => {5};
    (5,0) => {6}; (5,1) => {7};
}

macro_rules! br_1 {
    (0,0) => {1}; (0,1) => {2};
    (1,0) => {3}; (1,1) => {4};
    (2,0) => {5}; (2,1) => {6};
    (3,0) => {7}; (3,1) => {1};
    (4,0) => {2}; (4,1) => {3};
    (5,0) => {4}; (5,1) => {5};
}

macro_rules! cr {
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

macro_rules! dr {
    (0,0) => { 9 };
    (1,0) => { 11 };
    (2,0) => { 13 };
    (0,1) => { 15 };
    (1,1) => { 17 };
    (2,1) => { 19 };
    (0,2) => { 21 };
    (1,2) => { 23 };
    (2,2) => { 25 };
    (0,3) => { 27 };
    (1,3) => { 29 };
    (2,3) => { 31 };
}

// ***************************** 3 ******************************* //
macro_rules! step_3 {
    (B,4) => {
        concat!(
            load_b!(B, 0, br_3!(0,0), br_3!(0,1)),
            fmadd_3!(0),
            load_b!(B, 1, br_3!(1,0), br_3!(1,1)),
            fmadd_3!(1),
            "prefetcht0 384({ax}) \n",
            "prefetcht0 64({bx}) \n",
            load_b!(B, 2, br_3!(2,0), br_3!(2,1)),
            fmadd_3!(2),
            load_b!(B, 3, br_3!(3,0), br_3!(3,1)),
            fmadd_3!(3),
            "prefetcht0 448({ax}) \n",
        )
    };
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, br_3!(n,0), br_3!(n,1)),
                    fmadd_3!(n),
                )*
            )
        })
    };
}

// ***************************** 2 ******************************* //
macro_rules! step_2 {
    (B,6) => {
        concat!(
            load_b!(B, 0, br_2!(0,0), br_2!(0,1)),
            fmadd_2!(0),
            "prefetcht0 320({ax}) \n",
            load_b!(B, 1, br_2!(1,0), br_2!(1,1)),
            fmadd_2!(1),
            "prefetcht0 64({bx}) \n",
            load_b!(B, 2, br_2!(2,0), br_2!(2,1)),
            fmadd_2!(2),
            "prefetcht0 384({ax}) \n",
            load_b!(B, 3, br_2!(3,0), br_2!(3,1)),
            fmadd_2!(3),
            load_b!(B, 4, br_2!(4,0), br_2!(4,1)),
            fmadd_2!(4),
            load_b!(B, 5, br_2!(5,0), br_2!(5,1)),
            fmadd_2!(5),
        )
    };
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, br_2!(n,0), br_2!(n,1)),
                    fmadd_2!(n),
                )*
            )
        })
    };
}

// ***************************** 1 ******************************* //
macro_rules! step_1 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                "prefetcht0 64({bx}) \n",
                #(
                    load_b!($b_layout, n, br_1!(n,0), br_1!(n,1)),
                    fmadd_1!(n),
                )*
            )
        })
    };
}

macro_rules! mask_ptr {
    (P, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let $nm = if $m % VS == 0 && $m > 0 { 0xFF } else { (1_u8 << ($m % VS)) - 1 };
        let $mask_ptr = (&$nm) as *const u8;
    };
    (C, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let $nm = 0xFF_u8;
        let $mask_ptr = (&$nm) as *const u8;
    };
}

macro_rules! load_mask {
    (P) => { "kmovw ({maskx}), %k1" };
    (C) => { "/* {maskx} */" }
}

def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 4, B, P, ukernel_3_bbp);
def_ukernel_avx512!(1, step_2, acc_2, store_2, 2, 4, B, P, ukernel_2_bbp);
def_ukernel_avx512!(1, step_1, acc_1, store_1, 1, 4, B, P, ukernel_1_bbp);

def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 4, S, C, ukernel_bsc);

def_ukernel_avx512!(1, step_3, acc_3, store_3, 3, 4, S, P, ukernel_3_bsp);
def_ukernel_avx512!(1, step_2, acc_2, store_2, 2, 4, S, P, ukernel_2_bsp);
def_ukernel_avx512!(1, step_1, acc_1, store_1, 1, 4, S, P, ukernel_1_bsp);

def_ukernel_avx512_2!(1, step_3, acc_3, store_3, 3, 4, 4, 32);

