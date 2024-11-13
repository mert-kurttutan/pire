use seq_macro::seq;
use std::arch::asm;
use crate::{TA, TB, TC, UnaryFnC, TC_SIZE};
use super::VS;
use glar_base::{
    load_buf, store_buf, c_mem, prefetch_0, def_ukernel_sse, mem,
    c_reg_2x4, c_reg_1x4,
    b_num_2x4, b_num_1x4, dim_to_reg_avx,
    cum_seq,
};
type TS = f32;

const ZERO: i32 = 0;

const ZERO_SCALAR: f32 = 0.0;
const ONE_SCALAR: f32 = 1.0;
macro_rules! beta_fmadd {
    (C, $m0:expr, $r:expr, 1) => {
        concat!(
            "movups ", $m0, ", %xmm2", "\n",
            "paddd ", "%xmm2", ", %xmm", $r, "\n",
            // "paddd ", $m0, ", %xmm", $r, "\n",
        ) 
    };
    (C, $m0:expr, $r:expr, 2) => {
        concat!(
            "cvtdq2ps %xmm", $r, ",%xmm", $r, "\n",
            "movups ", $m0, ",%xmm2", "\n",
            "cvtdq2ps %xmm2", ",%xmm2", "\n",
            "mulps %xmm0, %xmm2", "\n",
            "addps %xmm2, %xmm", $r, "\n",
            "cvtps2dq %xmm", $r, ",%xmm", $r, "\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("xorps %xmm",r,",%xmm",r,"\n",)*)
        })
    }
}

macro_rules! vbroadcast {
    () => {
        "movss"
    };
}
macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "movups %xmm", $r2, ", %xmm", $r4, "\n",
            "pmaddubsw %xmm", $r1, ", %xmm", $r4, "\n",
            "pmaddwd ", "%xmm15", ", %xmm", $r4, "\n",
            "paddd %xmm", $r4, ", %xmm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "movaps ", $m0, ",%xmm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "movups %xmm", $r1, ", ", $m0,  "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                // jmp to 8 if alpha is equal to one
                "cmp $0x3f800000, {alphax} \n", // 0x3f800000 is 1 in float
                "je 8f \n",
                "movss ({alphax}),%xmm1", "\n",
                "shufps $0,%xmm1,%xmm1", "\n",
                #(
                    "cvtdq2ps %xmm", r, ",%xmm", r, "\n",
                    "mulps %xmm1, %xmm", r, "\n",
                    "cvtps2dq %xmm", r, ",%xmm", r, "\n",
                )*
                "8:", "\n",
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            vbroadcast!(), " ({betax}), %xmm0\n",
            "shufps $0,%xmm0,%xmm0\n",
            "xorps %xmm3,%xmm3\n",
            "ucomiss %xmm3,%xmm0\n",
        )
    }
}

macro_rules! acc_p {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $b:tt) => {
        concat!(
            beta_fmadd!(C, $m0, $r1, $b),
            beta_fmadd!($layout, mem!($m0, "0x10"), $r2, $b),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $b:tt) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1, $b),
        )
    };
}



macro_rules! loadp {
    (2, $m0:expr) => {
        concat!(
            loadp_unit!($m0, 0),
            loadp_unit!(mem!($m0, "0x10"), 1),
        )
    };
    (1, $m0:expr) => {
        concat!(
            loadp_unit!($m0, 0),
        )
    };
}
macro_rules! storep {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            storep_unit!(C, $r1, $m0),
            storep_unit!($layout, $r2, mem!($m0, "0x10")),
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


macro_rules! init_ab_avx {
    (B) => {
        concat!(
            // move 2 1_i16 to xmm15
            "mov $0x10001, {x3:e}", "\n",
            "movd {x3:e}, %xmm15", "\n",
            "shufps $0,%xmm15,%xmm15\n",
            // "/* {x4} */", "\n",
            "/* {x3} */", "\n",
            "/* {x2} */", "\n",
            "/* {x1} */", "\n",
            "mov 24({dim_arrx}),{x0}", "\n",
        )
    };
    (B) => {
        ""
    };
}

macro_rules! inc_a_k_unroll {
    ($X:tt, $K:tt) => {
        concat!(
            "add $16*", $K, "*", $X, ",{ax}", "\n",
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
        acc_p!($layout, c_mem!($ni), c_reg_2x4!(0,$ni), c_reg_2x4!(1,$ni), $b)
    };
}

macro_rules! store_2x4 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_2x4!(0,$ni), c_reg_2x4!(1,$ni))
    };
}

macro_rules! acc_1x4 {
    ($ni:tt, $layout:tt, $b:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_1x4!(0,$ni), $b)
    };
}

macro_rules! store_1x4 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_1x4!(0,$ni))
    };
}

macro_rules! load_b {
    (B, $N:tt, $K:tt, $X:tt, $r:expr) => {
        concat!(
            "movss ", $K, "*", $X, "*4+", $N, "*4({bx}), %xmm", $r, "\n",
            "shufps $0, %xmm", $r, ", %xmm", $r, "\n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt, $K:tt) => {
        loadp!($mr, concat!($mr,"*16*",$K,"({ax})"))
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
            vfmadd!(1, 3, 7, 15),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 2, 8, 12),
            vfmadd!(1, 2, 9, 13),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 3, 10, 14),
            vfmadd!(1, 3, 11, 15),
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
    ($mr:tt,$nr:tt) => { dim_to_reg_avx!(alpha_scale_0, $mr, $nr) };
}

// ***************************** 2x4 ******************************* //
macro_rules! step_2x4 {
    ($nr:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2, $K),
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
                load_a!(1, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_1x4!(n)),
                    fmadd_1v!(n),
                )*
            )
        })
    };
}

def_ukernel_sse!(4, step_2x4, acc_2x4, store_2x4, 2, 4, 4, 5, B, C, ukernel_2_bbp);
def_ukernel_sse!(4, step_1x4, acc_1x4, store_1x4, 1, 4, 4, 5, B, C, ukernel_1_bbp);


def_ukernel_sse!(4, step_2x4, acc_2x4, store_2x4, 2, 4, 1, 4, B, C, ukernel_n_bbc);

def_ukernel_sse!(4, step_2x4, acc_2x4, store_2x4, 2, 4, 1, 4, B, C, ukernel_2xn_bbp);
def_ukernel_sse!(4, step_1x4, acc_1x4, store_1x4, 1, 4, 1, 4, B, C, ukernel_1xn_bbp);



pub(crate) unsafe fn ukernel_bbc<F: UnaryFnC, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const f32, beta: *const f32,
    k: usize,
    d_arr: [usize; 3], c_cs: usize,
    a_pft1_offset: usize, _n: usize,
    f: F,
) {
    let k_l0 = k % 32;
    let k_l = if k_l0 == 0 {8} else {k_l0 / 4};
    let k_i = (k - k_l*4) / 16;
    let mut dim_arr = [c_cs*TC_SIZE, k_i, k_l, a_pft1_offset];
    let mut cf = c;
    let mut c_buf = [0i32; 8 * 4];
    let one_i16 = [1i16;8];
    if BUF {
        load_buf(c, d_arr[2], c_cs, &mut c_buf, 8, 4, 8);
        dim_arr[0] = 8*TC_SIZE;
        cf = c_buf.as_mut_ptr();
    }
    asm!(
        vzero_kernel!(),
        "movaps ({one_i16_ptr}), %xmm15",
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
        prefetch_0!(128, "({bx})"),
        step_2x4!(4, B, 0),

        "movq $64*4, {x4}",
        // divisiblity by 4
        "testq $3, {x0}",
        "cmovz {x1},{x4}",

        step_2x4!(4, B, 1),

        "prefetcht1 ({x2})",

        "subq $64*3, {x2}",
        "addq {x4}, {x2}",

        step_2x4!(4, B, 2),

        "prefetcht1 ({x5})",
        "addq $16, {x5}",

        "testq $63, {x0}",
        "cmovz {cx},{x2}",

        step_2x4!(4, B, 3),

        inc_a_k_unroll!(2, 4),
        inc_b_k_unroll!(B, 4, 4),

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
        "prefetcht0 60({x2})",
        step_2x4!(4, B, 0),
        inc_a_k_unroll!(2, 1),
        inc_b_k_unroll!(B, 4, 1),

        "add {x1}, {x2}",
        "dec {x0}",
        "jne 4b",
        "5:",
        "mov ({dim_arrx}),{x0}",
        "lea ({x0}, {x0}, 2), {x3}",
        "lea ({cx}, {x3},), {x1}",
        // "lea ({x1}, {x3},), {x2}",
        // scale by alpha
        alpha_scale!(2, 4),
        load_beta!(),

        // 6 -> BETAZERO
        "je 6f",

        // check if beta is equal to 1
        "cmp $0x3f800000, {betax} \n", // 0x3f800000 is 1 in float
        "je 9f",

        cum_seq!(acc_2x4,4,C,2),
        "jmp 6f",

        "9:",
        // 9 -> BETA ONE
        cum_seq!(acc_2x4,4,C,1),

        // 6 -> BETAZERO
        "6:",
        cum_seq!(store_2x4,4,C),
        ax = inout(reg) a => _, 
        bx = inout(reg) b => _, 
        cx = inout(reg) cf => _,
        dim_arrx = inout(reg) dim_arr.as_ptr() => _, 
        alphax = inout(reg) alpha => _, 
        betax = inout(reg) beta => _,
        one_i16_ptr = in(reg) &one_i16,
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
            f.call(cf.add(j*8), 8);
        }
        store_buf(c, d_arr[2], c_cs, &c_buf, 8, 4, 8);
    } else {
        for j in 0..4 {
            f.call(cf.add(j*c_cs), 8);
        }
    }
}