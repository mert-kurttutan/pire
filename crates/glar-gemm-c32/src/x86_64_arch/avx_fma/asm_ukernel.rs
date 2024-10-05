use seq_macro::seq;
use std::arch::asm;
use super::VS;
use crate::{TA, TB, TC};
use crate::MyFn;
use glar_base::{load_buf, store_buf, c_mem};

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            // "vfmadd231ps ", $m0, ",%ymm0,%ymm", $r1, "\n",
            "vaddps ", $m0, ",%ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "vmaskmovpd ", $m0, ", %ymm1", ", %ymm2",  "\n",
            // "vfmadd231ps %ymm2, %ymm0,%ymm", $r1, "\n",
            "vaddps %ymm2, %ymm", $r1, ", %ymm", $r1, "\n",
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
    ($r1:expr, $b1:expr, $b2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vfmadd231ps ", "%ymm",$b1, ", %ymm", $r1,", %ymm", $r3, "\n",
            "vfmadd231ps ", "%ymm",$b2, ", %ymm", $r1,", %ymm", $r4, "\n",
        ) 
    };
    ($r1:expr, $b1:expr, $r3:expr) => {
        concat!(
            "vfmadd231ps ", "%ymm",$b1, ", %ymm", $r1,", %ymm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovp!($layout), $m0, ",%ymm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (B, $r1:expr, $m0:expr) => {
        concat!(
            "vmovaps %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
            "vmaskmovps %ymm", $r1, ", %ymm1, ", $m0,  "\n",
        )
    };
}

macro_rules! complex_mul {
    ($r0:tt, $rt:tt) => {
        concat!(
            "vpermilps $0xb1, %ymm", $r0, ", %ymm", $rt, "\n",
            "vmulps %ymm0, %ymm", $r0, ", %ymm", $r0, "\n",
            "vmulps %ymm1, %ymm", $rt, ", %ymm", $rt, "\n",
            "vaddsubps %ymm", $rt, ", %ymm", $r0, ", %ymm", $r0, "\n",
        )
    };
}

macro_rules! asm_alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        concat!(
            // "vpxor %xmm0, %xmm0, %xmm0", "\n",
            // "vucomiss 4({alphax}), %xmm0", "\n",
            // "je 9f", "\n",
            "vbroadcastss ({alphax}), %ymm0 \n",
            "vbroadcastss 4({alphax}), %ymm1 \n",
        
            complex_mul!(4, 5),
            complex_mul!(6, 7),
            complex_mul!(8, 9),
            complex_mul!(10, 11),
            complex_mul!(12, 13),
            complex_mul!(14, 15),
            // "9:", "\n",
        )
    }
}

macro_rules! v_to_c {
    ($r0:tt, $r1:tt) => {
        concat!(
            "vpermilps $0xb1, %ymm", $r1, ", %ymm", $r1, "\n",
            "vaddsubps %ymm", $r1, ", %ymm", $r0, ", %ymm", $r0, "\n",
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

macro_rules! mem {
    ($m0:tt, $b0:tt) => {
        concat!($b0, "+", $m0)
    };
}

macro_rules! acc_p {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!(C, mem!($m0, "0x20"), $r2),
            beta_fmadd!($layout, mem!($m0, "0x40"), $r3),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!($layout, mem!($m0, "0x20"), $r2),
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
            loadp_unit!($layout, mem!($m0, "0x20"), 1),
            loadp_unit!($layout, mem!($m0, "0x40"), 2),
        )
    };
    (8, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x20"), 1),
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
            storep_unit!(C, $r2, mem!($m0, "0x20")),
            storep_unit!($layout, $r3, mem!($m0, "0x40")),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            storep_unit!(C, $r1, $m0),
            storep_unit!($layout, $r2, mem!($m0, "0x20")),
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
            "lea ({x2}, {x2}, 2), {x3}", "\n",
            "lea ({bx}, {x3}, 1), {x3}", "\n",

            "mov 24({dim_arrx}),{x0}", "\n",
            "test {x0},{x0}", "\n",
        )
    };
}


macro_rules! asm_c_load {
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
    (12,2) => {vzeroall!(4,15)};
    (12,1) => {vzeroall!(4,9)};
    (8,3) => {vzeroall!(4,15)};
    (8,2) => {vzeroall!(4,15)};
    (8,1) => {vzeroall!(4,15)};

    (4,3) => {vzeroall!(4,15)};
    (4,2) => {vzeroall!(4,15)};
    (4,1) => {vzeroall!(4,15)};
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
    (C, $X:tt, $K:tt) => {
        ""
    };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $8*", $K, "*", $X, ",{ax}", "\n",
        )
    };
}

macro_rules! inc_b_k_unroll {
    (S, $X:tt, $K:tt) => {
        ""
    };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $8*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}


macro_rules! asm_alpha_scale {
    (12, 2) => {
        asm_alpha_scale_0!(4,15)
    };
    (12, 1) => {
        asm_alpha_scale_0!(4,9)
    };
    (8, 3) => {
        asm_alpha_scale_0!(4,15)
    };
    (8, 2) => {
        asm_alpha_scale_0!(4,11)
    };
    (8, 1) => {
        asm_alpha_scale_0!(4,7)
    };

    (4, 3) => {
        asm_alpha_scale_0!(4,15)
    };
    (4, 2) => {
        asm_alpha_scale_0!(4,11)
    };
    (4, 1) => {
        asm_alpha_scale_0!(4,7)
    };
}

macro_rules! c_reg_12x2 {
    (0,0) => { 4 };
    (1,0) => { 6 };
    (2,0) => { 8 };
    (0,1) => { 10 };
    (1,1) => { 12 };
    (2,1) => { 14 };
}

macro_rules! c_reg_8x3 {
    (0,0) => { 4 };
    (1,0) => { 6 };
    (0,1) => { 8 };
    (1,1) => { 10 };
    (0,2) => { 12 };
    (1,2) => { 14 };
}

macro_rules! c_reg_4x3 {
    (0,0) => { 4 };
    (0,1) => { 6 };
    (0,2) => { 8 };
}

macro_rules! acc_12x2 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_12x2!(0,$ni), c_reg_12x2!(1,$ni), c_reg_12x2!(2,$ni))
    };
}

macro_rules! store_12x2 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_12x2!(0,$ni), c_reg_12x2!(1,$ni), c_reg_12x2!(2,$ni))
    };
}

macro_rules! acc_8x3 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_8x3!(0,$ni), c_reg_8x3!(1,$ni))
    };
}

macro_rules! store_8x3 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_8x3!(0,$ni), c_reg_8x3!(1,$ni))
    };
}

macro_rules! acc_4x3 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_4x3!(0,$ni))
    };
}

macro_rules! store_4x3 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_4x3!(0,$ni))
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
            vbroadcast!(), " ", $K, "*", $X, "*8+", $N, "*8({bx}), %ymm", $r1, "\n",
            vbroadcast!(), " ", $K, "*", $X, "*8+", $N, "*8+4({bx}), %ymm", $r2, "\n",
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
            vbroadcast!(), " ", $K, "*", $X, "*8+", $N, "*8({bx}), %ymm", $r1, "\n",
        )
    };
}


macro_rules! load_b2 {
    (S, 0, $K:tt, $X:tt, $r1:expr) => {
        concat!(
            vbroadcast!(), " 4({bx}),%ymm", $r1, "\n",
        )
    };
    (S, 1, $K:tt, $X:tt, $r1:expr) => {
        concat!(
            "prefetcht0 64({bx},{x2},1) \n",
            vbroadcast!(), " 4({bx},{x2},1),%ymm", $r1, "\n",
        )
    };
    (B, $N:tt, $K:tt, $X:tt, $r1:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*8+", $N, "*8+4({bx}), %ymm", $r1, "\n",
        )
    };
}

macro_rules! load_a {
    ($mr:tt, B, $K:tt) => {
        loadp!($mr, B, concat!($mr,"*8*",$K,"({ax})"))
    };
    ($mr:tt, C, $K:tt) => {
        loadp!($mr, C, "0({ax})")
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

// ***************************** 12x2 ******************************* //
macro_rules! step_12x2 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(12, $a_layout, $K),
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

// ***************************** 8x3 ******************************* //
macro_rules! step_8x3 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(8, $a_layout, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, 2, 3),
                    fmadd_2v!(n),
                )*
                inc_b!($b_layout,$nr),
            )
        })
    };
}

// ***************************** 4x3 ******************************* //
macro_rules! step_4x3 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(4, $a_layout, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, 2, 3),
                    fmadd_1v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! prefetch_0 {
    ($dist:tt, $reg:tt, $k_i:tt) => {
        concat!(
            "prefetcht0 ", $dist, "+", $k_i, "*64(", $reg, ")", "\n"
        )
    };
}

macro_rules! prefetch_c {
    (12, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(11+j*$ldc) as *const i8, 3);
        });
    };
    (8, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
        });
    };
    (4, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(1+j*$ldc) as *const i8, 3);
        });
    }
}

#[inline(always)]
fn mask_and_offset(m: usize) -> ([u64;8], usize) {
    let mask: [u64; 8] = [
        u64::MAX, u64::MAX, u64::MAX, u64::MAX,
        0, 0, 0, 0,
    ];
    let mask_offset = if m % VS == 0 { 0 } else { VS - (m %VS)};

    (mask, mask_offset)
}



macro_rules! mask_ptr {
    (M, $m:tt, $nm:ident) => {
        let (mask, mask_offset) = mask_and_offset($m);
        let $nm = mask.as_ptr().add(mask_offset);
    };
    (C, $m:tt, $nm:ident) => {
        let mask = [0xFFFFFFFF_u64];
        let $nm = mask.as_ptr();
    };
}

macro_rules! load_mask_ptr_asm {
    (M) => {
        "vmovdqu ({maskx}), %ymm1"
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
            let mask_ptr = x;
            let k_iter = (k-0) / 4;
            let k_left = k % 4+0;
            let mut dim_arr = [d_arr[0]*8, d_arr[1]*8, d_arr[3]*8, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, $mr);
                dim_arr[2] = $mr*8;
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
                prefetch_0!(128, "{bx}", 0),
                $step_macro!($nr, $a_layout, $b_layout, 0),
                $step_macro!($nr, $a_layout, $b_layout, 1),
                $step_macro!($nr, $a_layout, $b_layout, 2),
                $step_macro!($nr, $a_layout, $b_layout, 3),

                inc_a_k_unroll!($a_layout, $mr, 4),
                inc_b_k_unroll!($b_layout, $nr, 4),
        
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
                $step_macro!($nr, $a_layout, $b_layout, 0),
                inc_a_k_unroll!($a_layout, $mr, 1),
                inc_b_k_unroll!($b_layout, $nr, 1),
                "dec {x0}",
        
                // 4 -> KLEFT
                "jne 4b",
        
                // 5 -> POSTACCUM
                "5:",

                permute_complex!(),
                asm_c_load!($nr),
                // scale by alpha
                asm_alpha_scale!($mr, $nr),

                load_mask_ptr_asm!($is_partial),				

                cum_seq!($acc_macro,$nr,$is_partial),

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
                x0 = out(reg) _,
                x1 = out(reg) _,
                x2 = out(reg) _,
                x3 = out(reg) _,
                out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
                out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
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
            // beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize, n: usize,
            f: F,
        ) {
            mask_ptr!($is_partial, m, x);
            let mask_ptr = x;
            let k_iter = k / 4;
            let k_left = k % 4;
            let mut dim_arr = [d_arr[0]*8, d_arr[1]*8, d_arr[3]*8, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, $mr);
                dim_arr[2] = $mr*8;
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
                            prefetch_0!(128, "{bx}", 0),
                            $step_macro!(ni, $a_layout, $b_layout, 0),
                            $step_macro!(ni, $a_layout, $b_layout, 1),
                            $step_macro!(ni, $a_layout, $b_layout, 2),
                            $step_macro!(ni, $a_layout, $b_layout, 3),
            
                            inc_a_k_unroll!($a_layout, $mr, 4),
                            inc_b_k_unroll!($b_layout, ni, 4),
                
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
                            $step_macro!(ni, $a_layout, $b_layout, 0),
                            inc_a_k_unroll!($a_layout, $mr, 1),
                            inc_b_k_unroll!($b_layout, ni, 1),

                            "dec {x0}",
                
                            // 4 -> KLEFT
                            "jne 4b",
                
                            // 5 -> POSTACCUM
                            "5:",
                            permute_complex!(),
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),


                            load_mask_ptr_asm!($is_partial),				

                            cum_seq!($acc_macro,ni,$is_partial),

                            cum_seq!($store_macro,ni,$is_partial),
                            
                            // 7 -> DDONE
                            "7:",
                            // "vzeroupper",
                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) cf => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            // betax = inout(reg) beta => _,
                            maskx = inout(reg) mask_ptr => _,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                            out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                            out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
                            out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
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

// def_ukernel!(step_8x3, acc_8x3, store_8x3, 8, 3, B, B, C, ukernel_8x3_bb);
// // def_ukernel!(step_4x3, acc_4x3, store_4x3, 8, 4, B, B, C, 4, ukernel_16x8_bb);


// def_ukernel!(step_8x3, acc_8x3, store_8x3, 8, 3, B, B, M, ukernel_8x3_bb_partial);
// def_ukernel!(step_4x3, acc_4x3, store_4x3, 4, 3, B, B, M, ukernel_4x3_bb_partial);

// def_ukernel!(step_8x3, acc_8x3, store_8x3, 8, 3, B, S, C, ukernel_8x3_bs);


// def_ukernel!(step_8x3, acc_8x3, store_8x3, 8, 3, B, S, M, ukernel_8x3_bs_partial);
// def_ukernel!(step_4x3, acc_4x3, store_4x3, 4, 3, B, S, M, ukernel_4x3_bs_partial);



// def_ukernel!(step_12x2, acc_12x2, store_12x2, 12, 2, B, B, C, ukernel_12x2_bb);

def_ukernel!(step_12x2, acc_12x2, store_12x2, 12, 2, B, B, M, ukernel_12x2_bb_partial);
def_ukernel!(step_8x3, acc_8x3, store_8x3, 8, 2, B, B, M, ukernel_8x2_bb_partial);
def_ukernel!(step_4x3, acc_4x3, store_4x3, 4, 2, B, B, M, ukernel_4x2_bb_partial);

def_ukernel!(step_12x2, acc_12x2, store_12x2, 12, 2, B, S, C, ukernel_12x2_bs);

def_ukernel!(step_12x2, acc_12x2, store_12x2, 12, 2, B, S, M, ukernel_12x2_bs_partial);
def_ukernel!(step_8x3, acc_8x3, store_8x3, 8, 2, B, S, M, ukernel_8x2_bs_partial);
def_ukernel!(step_4x3, acc_4x3, store_4x3, 4, 2, B, S, M, ukernel_4x2_bs_partial);

def_ukernelxn!(step_12x2, acc_12x2, store_12x2, 12, 2, B, B, C, ukernel_12xn_bb);
// def_ukernelxn!(step_8x3, acc_8x3, store_8x3, 8, 3, B, B, C, ukernel_8xn_bb);
// def_ukernelxn!(step_4x3, acc_4x3, store_4x3, 8, 4, B, B, C, 4, ukernel_8xn_bb);

def_ukernelxn!(step_12x2, acc_12x2, store_12x2, 12, 2, B, B, M, ukernel_12xn_bb_partial);
def_ukernelxn!(step_8x3, acc_8x3, store_8x3, 8, 3, B, B, M, ukernel_8xn_bb_partial);
def_ukernelxn!(step_4x3, acc_4x3, store_4x3, 4, 3, B, B, M, ukernel_4xn_bb_partial);

def_ukernelxn!(step_12x2, acc_12x2, store_12x2, 12, 2, B, S, C, ukernel_12xn_bs);
// def_ukernelxn!(step_8x3, acc_8x3, store_8x3, 8, 3, B, S, C, ukernel_8xn_bs);

def_ukernelxn!(step_12x2, acc_12x2, store_12x2, 12, 2, B, S, M, ukernel_12xn_bs_partial);
def_ukernelxn!(step_8x3, acc_8x3, store_8x3, 8, 3, B, S, M, ukernel_8xn_bs_partial);
def_ukernelxn!(step_4x3, acc_4x3, store_4x3, 4, 3, B, S, M, ukernel_4xn_bs_partial);


// based on l1 prefetching scheme is from openblas impl for skylax
// see: https://github.com/OpenMathLib/OpenBLAS/pull/2300
// this is adapted to our ukernel of 12x2
// seems to stem from high bandwith of l1 cache (compared to other uarch e.g. haswell
// where the same l1 prefetching does not benefit as much)
pub(crate) unsafe fn ukernel_12x2_bb<F: MyFn, const BUF: bool>(
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
    let mut cf = c;
    let mut c_buf = [TC::ZERO; 12 * 2];
    let c_cs = d_arr[3];
    if BUF {
        load_buf(c, d_arr[2], c_cs, &mut c_buf, 12, 2, 12);
        dim_arr[2] = 12*8;
        cf = c_buf.as_mut_ptr();
    }
    asm!(
        asm_vzeroall!(12,2),
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
        prefetch_0!(128, "{bx}", 0),
        step_12x2!(2, B, B, 0),

        "movq $64*4, {x4}",
        // divisiblity by 4
        "testq $3, {x0}",
        "cmovz {x1},{x4}",

        step_12x2!(2, B, B, 1),

        "prefetcht1 ({x2})",

        "subq $64*3, {x2}",
        "addq {x4}, {x2}",

        step_12x2!(2, B, B, 2),

        "prefetcht1 ({x5})",
        "addq $16, {x5}",

        "testq $63, {x0}",
        "cmovz {cx},{x2}",

        step_12x2!(2, B, B, 3),

        inc_a_k_unroll!(B, 12, 4),
        inc_b_k_unroll!(B, 2, 4),

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
        "prefetcht0 92({x2})",
        step_12x2!(2, B, B, 0),
        inc_a_k_unroll!(B, 12, 1),
        inc_b_k_unroll!(B, 2, 1),

        "add {x1}, {x2}",
        "dec {x0}",
        "jne 4b",
        "5:",
        "mov ({dim_arrx}),{x0}",
        permute_complex!(),
        // scale by alpha
        asm_alpha_scale!(12, 2),

        cum_seq!(acc_12x2,2,C),
        cum_seq!(store_12x2,2,C),

        "7:",
        ax = inout(reg) a => _, 
        bx = inout(reg) b => _, 
        cx = inout(reg) cf => _,
        dim_arrx = inout(reg) dim_arr.as_ptr() => _, 
        alphax = inout(reg) alpha => _, 
        x0 = out(reg) _, 
        x1 = out(reg)_, 
        x2 = out(reg) _, 
        // x3 = out(reg) _, 
        x4 = out(reg) _,
        x5 = out(reg) _, 
        out("xmm0") _, out("xmm1") _,
        out("xmm2") _, out("xmm3") _, out("xmm4") _, out("xmm5") _, out("xmm6") _,
        out("xmm7") _, out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
        out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
        options(att_syntax)
    );
    if BUF {
        for j in 0..2 {
            f.call(cf.add(j*12), 12);
        }
        store_buf(c, d_arr[2], c_cs, &c_buf, 12, 2, 12);
    } else {
        for j in 0..2 {
            f.call(cf.add(j*c_cs), 12);
        }
    }
}