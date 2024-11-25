use seq_macro::seq;
use super::VS;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    c_mem, def_ukernel_avx, 
    c_reg_2x4, c_reg_1x4,
    b_num_2x4, b_num_1x4,
    load_a_avx, storep_avx, acc_p_avx, 
    // def_ukernel_avx_2,
    // prefetch_0,
};

type TS = f32;


const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r:expr, 1) => {
        concat!(
            "vpaddd ", $m0, ", %ymm", $r, ", %ymm", $r, "\n",
        ) 
    };
    (C, $m0:expr, $r:expr, 2) => {
        concat!(
            "vcvtdq2ps %ymm", $r, ",%ymm", $r, "\n",
            "vcvtdq2ps ", $m0, ",%ymm2", "\n",
            "vfmadd231ps %ymm2,%ymm0,%ymm", $r, "\n",
            "vcvtps2dq %ymm", $r, ",%ymm", $r, "\n",
        ) 
    };
    (P, $m0:expr, $r:expr, 1) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vpaddd %ymm2, %ymm", $r, ", %ymm", $r, "\n",
        ) 
    };

    (P, $m0:expr, $r:expr, 2) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vcvtdq2ps %ymm2", ",%ymm2", "\n",
            "vcvtdq2ps %ymm", $r, ",%ymm", $r, "\n",
            "vfmadd231ps %ymm2,%ymm0,%ymm", $r, "\n",
            "vcvtps2dq %ymm", $r, ",%ymm", $r, "\n",
        ) 
    };
}

macro_rules! c_load {
    () => {
        concat!(
            "mov 16({dim_arrx}),{x0}\n",
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
        "vbroadcastss"
    };
}
macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vpmaddubsw %ymm", $r1, ", %ymm", $r2, ", %ymm", $r4, "\n",
            "vpmaddwd %ymm", $r4, ", %ymm15", ", %ymm", $r4, "\n",
            "vpaddd %ymm", $r4, ", %ymm", $r3, ", %ymm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "vmovaps ", $m0, ",%ymm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (P, $r1:expr, $m0:expr) => {
        concat!(
            "vmaskmovps %ymm", $r1, ", %ymm1, ", $m0,  "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "vbroadcastss ({alphax}),%ymm1", "\n",
                #(
                    "vcvtdq2ps %ymm", r, ",%ymm", r, "\n",
                    "vmulps %ymm1, %ymm", r, ",%ymm", r, "\n",
                    "vcvtps2dq %ymm", r, ",%ymm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %ymm0\n",
            "vxorps %ymm3,%ymm3,%ymm3\n",
            "vucomiss %xmm3,%xmm0\n",
        )
    }
}

macro_rules! init_ab_avx {
    (B) => {
        concat!(
            // move 2 1_i16 to xmm15
            "mov $0x10001, {x3:e}", "\n",
            "movd {x3:e}, %xmm15", "\n",
            "vbroadcastss %xmm15, %ymm15", "\n",
            "/* {x3} */", "\n",
            "/* {x2} */", "\n",
            "/* {x1} */", "\n",
            "mov 24({dim_arrx}),{x0}", "\n",
        )
    };
    (S) => {
        ""
    };
}

// macro_rules! init_ab_2 {
//     (B) => {
//         concat!(
//             // move 2 1_i16 to xmm15
//             "mov $0x10001, {x3:e}", "\n",
//             "movd {x3:e}, %xmm15", "\n",
//             "vbroadcastss %xmm15, %ymm15", "\n",
//             "/* {x3} */", "\n",
//             "/* {x2} */", "\n",
//             "/* {x1} */", "\n",
//             "mov 8({dim_arrx}),{x0}", "\n",
//         )
//     };
//     (S) => {
//         ""
//     };
// }

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
            "add $4*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}

macro_rules! acc_2x4 {
    ($ni:tt, $layout:tt, $b:tt) => {
        acc_p_avx!($layout, c_mem!($ni), $b, c_reg_2x4!(0,$ni), c_reg_2x4!(1,$ni))
    };
}

macro_rules! store_2x4 {
    ($ni:tt, $layout:tt) => {
        storep_avx!($layout, c_mem!($ni), c_reg_2x4!(0,$ni), c_reg_2x4!(1,$ni))
    };
}

macro_rules! acc_1x4 {
    ($ni:tt, $layout:tt, $b:tt) => {
        acc_p_avx!($layout, c_mem!($ni), $b, c_reg_1x4!(0,$ni))
    };
}

macro_rules! store_1x4 {
    ($ni:tt, $layout:tt) => {
        storep_avx!($layout, c_mem!($ni), c_reg_1x4!(0,$ni))
    };
}

macro_rules! load_b {
    (B, $N:tt, $K:tt, $X:tt, $r:expr) => {
        concat!(
            "vbroadcastss ", $K, "*", $X, "*4+", $N, "*4({bx}), %ymm", $r, "\n",
        )
    };
}

macro_rules! fmadd_2v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 4, 12),
            vfmadd!(1, 2, 5, 13),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 3, 6, 14),
            vfmadd!(1, 3, 7, 12),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 2, 8, 13),
            vfmadd!(1, 2, 9, 14),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 3, 10, 12),
            vfmadd!(1, 3, 11, 13),
        )
    };
}

macro_rules! fmadd_1v {
    (0) => {
        concat!(
            vfmadd!(0, 1, 7, 11),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 8, 12),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 3, 9, 13),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 4, 10, 14),
        )
    };
}

macro_rules! vzero_kernel {
    () => { vzeroall!(4,11) };
}

macro_rules! alpha_scale {
    () => { alpha_scale_0!(4,11) };
}

// ***************************** 2x4 ******************************* //
macro_rules! step_2x4 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx!(2, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_2x4!(n)),
                    fmadd_2v!(n),
                )*
            )
        })
    };
}

// ***************************** 1x4 ******************************* //
macro_rules! step_1x4 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a_avx!(1, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_1x4!(n)),
                    fmadd_1v!(n),
                )*
            )
        })
    };
}

#[inline(always)]
fn mask_and_offset(m: usize) -> ([u32;16], usize) {
    let mask: [u32; 16] = [
        u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let mask_offset = if m % VS == 0 { 0 } else { VS - (m %VS)};

    (mask, mask_offset)
}



macro_rules! mask_ptr {
    (P, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let (mask, mask_offset) = mask_and_offset($m);
        let $nm = mask.as_ptr().add(mask_offset);
        let $mask_ptr = $nm;
    };
    (C, $m:tt, $nm:ident, $mask_ptr:ident) => {
        let mask = [0xFFFF_u32];
        let $nm = mask.as_ptr();
        let $mask_ptr = $nm;
    };
}

macro_rules! load_mask {
    (P) => {
        "vmovdqu ({maskx}), %ymm1"
    };
    (C) => {
        "/* {maskx} */"
    }
}

def_ukernel_avx!(4, step_2x4, acc_2x4, store_2x4, 2, 4, B, P, ukernel_2_bbp);
def_ukernel_avx!(4, step_1x4, acc_1x4, store_1x4, 1, 4, B, P, ukernel_1_bbp);

// def_ukernel_avx_2!(4, step_2x4, acc_2x4, store_2x4, 2, 4, 16, 32);
def_ukernel_avx!(4,step_2x4, acc_2x4, store_2x4, 2, 4, B, C, ukernel_bbc);