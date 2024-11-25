use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    c_mem, prefetch_0, def_ukernel_avx, 
    init_ab_avx, acc_p_avx, storep_avx, load_a_avx,
    def_ukernel_avx_2, init_ab_2,
};

type TS = TC;

const ZERO_SCALAR: TC = TC::ZERO;
const ONE_SCALAR: TC = TC::ONE;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "vaddpd ", $m0, ",%ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr,1) => {
        concat!(
            "vmaskmovpd ", $m0, ", %ymm2", ", %ymm5",  "\n",
            "vaddpd %ymm5, %ymm", $r1, ", %ymm", $r1, "\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "vmovupd ", $m0, ", %ymm5", "\n",
            complex_mul!(5, 7),
            "vaddpd %ymm5, %ymm", $r1, ", %ymm", $r1, "\n",
        ) 
    };
    (P, $m0:expr, $r1:expr,2) => {
        concat!(
            "vmaskmovpd ", $m0, ", %ymm2", ", %ymm5",  "\n",
            complex_mul!(5, 7),
            "vaddpd %ymm5, %ymm", $r1, ", %ymm", $r1, "\n",
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
            concat!(#("vpxor %ymm",r,",%ymm",r,",%ymm",r,"\n",)*)
        })
    }
}

macro_rules! vbroadcast {
    () => {
        "vbroadcastsd"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $b1:expr, $b2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vfmadd231pd ", "%ymm",$b1, ", %ymm", $r1,", %ymm", $r3, "\n",
            "vfmadd231pd ", "%ymm",$b2, ", %ymm", $r1,", %ymm", $r4, "\n",
        ) 
    };
    ($r1:expr, $b1:expr, $r3:expr) => {
        concat!(
            "vfmadd231pd ", "%ymm",$b1, ", %ymm", $r1,", %ymm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovapd ", $m0, ",%ymm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovupd %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (P, $r1:expr, $m0:expr) => {
        concat!(
            "vmaskmovpd %ymm", $r1, ", %ymm2, ", $m0,  "\n",
        )
    };
}

macro_rules! complex_mul {
    ($r0:tt, $rt:tt) => {
        concat!(
            "vpermilpd $0b101, %ymm", $r0, ", %ymm", $rt, "\n",
            "vmulpd %ymm0, %ymm", $r0, ", %ymm", $r0, "\n",
            "vmulpd %ymm1, %ymm", $rt, ", %ymm", $rt, "\n",
            "vaddsubpd %ymm", $rt, ", %ymm", $r0, ", %ymm", $r0, "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    () => {
        concat!(

            "vbroadcastsd ({alphax}), %ymm0 \n",
            "vbroadcastsd 8({alphax}), %ymm1 \n",
            
            complex_mul!(4, 5),
            complex_mul!(6, 7),
            complex_mul!(8, 9),
            complex_mul!(10, 11),
            complex_mul!(12, 13),
            complex_mul!(14, 15),
        )
    }
}


macro_rules! v_to_c {
    ($r0:tt, $r1:tt) => {
        concat!(
            "vpermilpd $0b101, %ymm", $r1, ", %ymm", $r1, "\n",
            "vaddsubpd %ymm", $r1, ", %ymm", $r0, ", %ymm", $r0, "\n",
        )
    }
}

macro_rules! permute_complex {
    () => {
        concat!(
            // permute even and odd elements
            v_to_c!(4, 5),
            v_to_c!(6, 7),
            v_to_c!(8, 9),
            v_to_c!(10, 11),
            v_to_c!(12, 13),
            v_to_c!(14, 15),
        )
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "vbroadcastsd ({betax}), %ymm0\n",
            "vbroadcastsd 8({betax}), %ymm1\n",
        )
    }
}


macro_rules! vzero_kernel {
    () => {vzeroall!(4,15)};
}	

macro_rules! inc_b {
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
        ""
    };
}

macro_rules! inc_a_k_unroll {
    ($X:tt, $K:tt) => {
        concat!(
            "add $32*", $K, "*", $X, ",{ax}", "\n",
        )
    };
}

macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => {
        ""
    };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $16*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}


macro_rules! alpha_scale {
    () => {
        alpha_scale_0!()
    };
}

macro_rules! c_reg_3x2 {
    (0,0) => { 4 };
    (1,0) => { 6 };
    (2,0) => { 8 };
    (0,1) => { 10 };
    (1,1) => { 12 };
    (2,1) => { 14 };
}
macro_rules! c_reg_2x3 {
    (0,0) => { 4 };
    (1,0) => { 6 };
    (0,1) => { 8 };
    (1,1) => { 10 };
    (0,2) => { 12 };
    (1,2) => { 14 };
}

macro_rules! c_reg_1x3 {
    (0,0) => { 4 };
    (0,1) => { 6 };
    (0,2) => { 8 };
}

macro_rules! acc_3x2 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx!($layout, c_mem!($ni), $q, c_reg_3x2!(0,$ni), c_reg_3x2!(1,$ni), c_reg_3x2!(2,$ni))
    };
}

macro_rules! store_3x2 {
    ($ni:tt, $layout:tt) => {
        storep_avx!($layout, c_mem!($ni), c_reg_3x2!(0,$ni), c_reg_3x2!(1,$ni), c_reg_3x2!(2,$ni))
    };
}

macro_rules! acc_2x3 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx!($layout, c_mem!($ni), $q, c_reg_2x3!(0,$ni), c_reg_2x3!(1,$ni))
    };
}

macro_rules! store_2x3 {
    ($ni:tt, $layout:tt) => {
        storep_avx!($layout, c_mem!($ni), c_reg_2x3!(0,$ni), c_reg_2x3!(1,$ni))
    };
}

macro_rules! acc_1x3 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx!($layout, c_mem!($ni), $q, c_reg_1x3!(0,$ni))
    };
}

macro_rules! store_1x3 {
    ($ni:tt, $layout:tt) => {
        storep_avx!($layout, c_mem!($ni), c_reg_1x3!(0,$ni))
    };
}

macro_rules! load_b {
    (S, 0, $K:tt, $X:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx}),%ymm", $r1, "\n",
        )
    };
    (S, 1, $K:tt, $X:tt, $r1:expr, $r2:expr) => {
        concat!(
            "prefetcht0 64({bx},{x2},1) \n",
            vbroadcast!(), " ({bx},{x2},1),%ymm", $r1, "\n",
        )
    };
    (S, 2, $K:tt, $X:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({bx},{x2},2),%ymm", $r1, "\n",
        )
    };
    (B, $N:tt, $K:tt, $X:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*16+", $N, "*16({bx}), %ymm", $r1, "\n",
            vbroadcast!(), " ", $K, "*", $X, "*16+", $N, "*16+8({bx}), %ymm", $r2, "\n",
        )
    };
}

macro_rules! load_b1 {
    (S, 0, $K:tt, $X:tt, $r1:expr) => {
        concat!(
            vbroadcast!(), " ({bx}),%ymm", $r1, "\n",
        )
    };
    (S, 1, $K:tt, $X:tt, $r1:expr) => {
        concat!(
            "prefetcht0 64({bx},{x2},1) \n",
            vbroadcast!(), " ({bx},{x2},1),%ymm", $r1, "\n",
        )
    };
    (B, $N:tt, $K:tt, $X:tt, $r1:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*16+", $N, "*16({bx}), %ymm", $r1, "\n",
        )
    };
}


macro_rules! load_b2 {
    (S, 0, $K:tt, $X:tt, $r1:expr) => {
        concat!(
            vbroadcast!(), " 8({bx}),%ymm", $r1, "\n",
        )
    };
    (S, 1, $K:tt, $X:tt, $r1:expr) => {
        concat!(
            "prefetcht0 64({bx},{x2},1) \n",
            vbroadcast!(), " 8({bx},{x2},1),%ymm", $r1, "\n",
        )
    };
    (B, $N:tt, $K:tt, $X:tt, $r1:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*16+", $N, "*16+8({bx}), %ymm", $r1, "\n",
        )
    };
}

macro_rules! fmadd_3v {
    (0,0) => {
        concat!(
            vfmadd!(0, 3, 4),
            vfmadd!(1, 3, 6),
            vfmadd!(2, 3, 8),
        )
    };
    (0,1) => {
        concat!(
            vfmadd!(0, 3, 5),
            vfmadd!(1, 3, 7),
            vfmadd!(2, 3, 9),
        )
    };
    (1,0) => {
        concat!(
            vfmadd!(0, 3, 10),
            vfmadd!(1, 3, 12),
            vfmadd!(2, 3, 14),
        )
    };
    (1,1) => {
        concat!(
            vfmadd!(0, 3, 11),
            vfmadd!(1, 3, 13),
            vfmadd!(2, 3, 15),
        )
    };
}

macro_rules! fmadd_2v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 3, 4, 5),
            vfmadd!(1, 2, 3, 6, 7),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 3, 8, 9),
            vfmadd!(1, 2, 3, 10, 11),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 2, 3, 12, 13),
            vfmadd!(1, 2, 3, 14, 15),
        )
    };
}

macro_rules! fmadd_1v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 3, 4, 5),
        )

    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 3, 6, 7),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 2, 3, 8, 9),
        )
    };
}

// ***************************** 3x2 ******************************* //
macro_rules! step_3x2 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx!(3, $K),
                #(
                    load_b1!($b_layout, n, $K, $nr, 3),
                    fmadd_3v!(n,0),
                    load_b2!($b_layout, n, $K, $nr, 3),
                    fmadd_3v!(n,1),
                )*
                inc_b!($b_layout,$nr),
            )
        })
    };
}

// ***************************** 2x3 ******************************* //
macro_rules! step_2x3 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx!(2, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, 2, 3),
                    fmadd_2v!(n),
                )*
                inc_b!($b_layout,$nr),
            )
        })
    };
}

// ***************************** 1x3 ******************************* //
macro_rules! step_1x3 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx!(1, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, 2, 3),
                    fmadd_1v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

#[inline(always)]
fn mask_and_offset(m: usize) -> ([u64;8], usize) {
    let mask: [u64; 8] = [
        u64::MAX, u64::MAX, u64::MAX, u64::MAX,
        0, 0, 0, 0,
    ];
    let mask_offset = if m % VS == 0 { 0 } else { VS - (m %VS)};
    // mask offset for 128 bit part of ymm register
    // since c64 is 128 bit wide
    (mask, mask_offset*2)
}



macro_rules! mask_ptr {
    (P, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let (mask, mask_offset) = mask_and_offset($m);
        let $nm = mask.as_ptr().add(mask_offset);
        let $mask_ptr = $nm;
    };
    (C, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let mask = [0xFFFFFFFF_u64];
        let $nm = mask.as_ptr();
        let $mask_ptr = $nm;
    };
}

macro_rules! load_mask {
    (P) => {
        "vmovdqu ({maskx}), %ymm2"
    };
    (C) => {
        "/* {maskx} */"
    }
}

def_ukernel_avx!(1, step_3x2, acc_3x2, store_3x2, 3, 2, B, P, ukernel_3_bbp);
def_ukernel_avx!(1, step_2x3, acc_2x3, store_2x3, 2, 2, B, P, ukernel_2_bbp);
def_ukernel_avx!(1, step_1x3, acc_1x3, store_1x3, 1, 2, B, P, ukernel_1_bbp);

def_ukernel_avx!(1, step_3x2, acc_3x2, store_3x2, 3, 2, S, C, ukernel_bsc);

def_ukernel_avx!(1, step_3x2, acc_3x2, store_3x2, 3, 2, S, P, ukernel_3_bsp);
def_ukernel_avx!(1, step_2x3, acc_2x3, store_2x3, 2, 2, S, P, ukernel_2_bsp);
def_ukernel_avx!(1, step_1x3, acc_1x3, store_1x3, 1, 2, S, P, ukernel_1_bsp);

def_ukernel_avx_2!(1, step_3x2, acc_3x2, store_3x2, 3, 2, 4, 64);
