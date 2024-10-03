use seq_macro::seq;
use std::arch::asm;

use crate::MyFn;

use crate::{TA, TB, TC};
use super::VS;
use crate::{load_buf, store_buf};
use glar_base::c_mem;

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
    (M, $m0:expr, $r:expr, 1) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vpaddd %ymm2, %ymm", $r, ", %ymm", $r, "\n",
        ) 
    };

    (M, $m0:expr, $r:expr, 2) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vcvtdq2ps %ymm", $r, ",%ymm", $r, "\n",
            "vfmadd231ps %ymm2,%ymm0,%ymm", $r, "\n",
            "vcvtps2dq %ymm", $r, ",%ymm", $r, "\n",
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
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "vpmaddwd %ymm", $r1, ", %ymm", $r2, ", %ymm", $r4, "\n",
            "vpaddd %ymm", $r4, ", %ymm", $r3, ", %ymm", $r3, "\n",
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

macro_rules! asm_alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                // jmp to 8 if alpha is equal to onex
                "vbroadcastss ({alphax}),%ymm1", "\n",
                "vucomiss ({onex}), %xmm1 \n",
                "je 8f \n",
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

macro_rules! mem {
    ($m0:tt, $b0:tt) => {
        concat!($b0, "+", $m0)
    };
}

macro_rules! acc_p {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $b:tt) => {
        concat!(
            beta_fmadd!(C, $m0, $r1, $b),
            beta_fmadd!($layout, mem!($m0, "0x20"), $r2, $b),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $b:tt) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1, $b),
        )
    };
}


macro_rules! loadp {
    (16, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x20"), 1),
        )
    };
    (8, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
        )
    };
}

macro_rules! storep {
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
            "lea ({x2}, {x2}, 2), {x3}", "\n",
            "lea ({bx}, {x3}, 1), {x3}", "\n",
            "lea ({bx}, {x2}, 1), {x4}", "\n",
            "lea ({bx}, {x2}, 2), {x5}", "\n",

            "mov 24({dim_arrx}),{x0}", "\n",
            "test {x0},{x0}", "\n",
        )
    };
}


macro_rules! asm_c_load {
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
    (16,4) => {vzeroall!(4,11)};
    (16,3) => {vzeroall!(4,9)};
    (16,2) => {vzeroall!(4,7)};
    (16,1) => {vzeroall!(4,5)};

    (8,4) => {vzeroall!(5,8)};
    (8,3) => {vzeroall!(5,7)};
    (8,2) => {vzeroall!(5,6)};
    (8,1) => {vzeroall!(5,5)};
}

macro_rules! inc_a_k_unroll {
    (C, $X:tt, $K:tt) => {
        ""
    };
    (B, $X:tt, $K:tt) => {
        concat!(
            "add $4*", $K, "*", $X, ",{ax}", "\n",
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


macro_rules! asm_alpha_scale {
    (16, 4) => {asm_alpha_scale_0!(4,11)};
    (16, 3) => {asm_alpha_scale_0!(4,9)};
    (16, 2) => {asm_alpha_scale_0!(4,7)};
    (16, 1) => {asm_alpha_scale_0!(4,5)};

    (8, 4) => {asm_alpha_scale_0!(5,8)};
    (8, 3) => {asm_alpha_scale_0!(5,7)};
    (8, 2) => {asm_alpha_scale_0!(5,6)};
    (8, 1) => {asm_alpha_scale_0!(5,5)};
}

macro_rules! c_reg_16x4 {
    (0,0) => { 4 }; (1,0) => { 5 };
    (0,1) => { 6 }; (1,1) => { 7 };
    (0,2) => { 8 }; (1,2) => { 9 };
    (0,3) => { 10 }; (1,3) => { 11 };
}

macro_rules! c_reg_8x4 {
    (0,0) => { 5 };
    (0,1) => { 6 };
    (0,2) => { 7 };
    (0,3) => { 8 };
}

macro_rules! acc_16x4 {
    ($ni:tt, $layout:tt, $b:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_16x4!(0,$ni), c_reg_16x4!(1,$ni), $b)
    };
}

macro_rules! store_16x4 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_16x4!(0,$ni), c_reg_16x4!(1,$ni))
    };
}

macro_rules! acc_8x4 {
    ($ni:tt, $layout:tt, $b:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_8x4!(0,$ni), $b)
    };
}

macro_rules! store_8x4 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_8x4!(0,$ni))
    };
}

macro_rules! cum_seq {
    ($step_macro:tt, $nr:tt, $layout:tt, $b:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!(n, $layout, $b),)*)
        })
    };
    ($step_macro:tt, $nr:tt, $layout:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!(n, $layout),)*)
        })
    };
}

macro_rules! load_b {
    (B, $N:tt, $K:tt, $X:tt, $r:expr) => {
        concat!(
            vbroadcast!(), "  ", $K, "*", $X, "*4+", $N, "*4({bx}), %ymm", $r, "\n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt, B, $K:tt) => {
        loadp!($mr, B, concat!($mr,"*4*",$K,"({ax})"))
    };
    ($mr:tt, C, $K:tt) => {
        loadp!($mr, C, "0({ax})")
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
            vfmadd!(0, 1, 5, 9),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 6, 10),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 3, 7, 11),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 4, 8, 12),
        )
    };
}

macro_rules! b_num_16x4 {
    (0) => {2};
    (1) => {3};
    (2) => {2};
    (3) => {3};
}

macro_rules! b_num_8x4 {
    (0) => {1};
    (1) => {2};
    (2) => {3};
    (3) => {4};
}

// ***************************** 16x4 ******************************* //
macro_rules! step_16x4 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(16, $a_layout, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_16x4!(n)),
                    fmadd_2v!(n),
                )*
            )
        })
    };
}

// ***************************** 8x4 ******************************* //
macro_rules! step_8x4 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(8, $a_layout, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_8x4!(n)),
                    fmadd_1v!(n),
                )*
            )
        })
    };
}

macro_rules! prefetch_0 {
    ($dist:tt, $reg:tt) => {
        concat!(
            "prefetcht0 ", $dist, $reg, "\n"
        )
    };
}

macro_rules! prefetch_c {
    (16, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(15+j*$ldc) as *const i8, 3);
        });
    };
    (8, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(2+j*$ldc) as *const i8, 3);
        });
    }
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
    (M, $m:tt, $nm:ident) => {
        let (mask, mask_offset) = mask_and_offset($m);
        let $nm = mask.as_ptr().add(mask_offset);
    };
    (C, $m:tt, $nm:ident) => {
        let mask = [0xFFFF_u32];
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
            alpha: *const f32, beta: *const f32,
            k: usize,
            d_arr: [usize; 4],
            m: usize,
            f: F,
        ) {
            mask_ptr!($is_partial, m, x);
            let mask_ptr = x;
            let k = (k+1) / 2 *2;
            let k_iter = k / 8;
            let k_left = (k % 8) / 2;
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*4, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [0i32;$mr*$nr];
            let c_cs = d_arr[3];
            let one = 1_f32;
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, $mr);
                dim_arr[2] = $mr*4;
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
                prefetch_0!(128, "({bx})"),
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
                asm_c_load!($nr),
                // scale by alpha
                asm_alpha_scale!($mr, $nr),

                "8:",
                load_mask_ptr_asm!($is_partial),
                load_beta!(),

                // 6 -> BETAZERO
                "je 6f",

                // check if beta is equal to 1
                "vucomiss ({onex}), %xmm0",
                "je 9f",

                cum_seq!($acc_macro,$nr,$is_partial,2),
                "jmp 6f",

                "9:",
                // 9 -> BETA ONE
                cum_seq!($acc_macro,$nr,$is_partial,1),

                // 6 -> BETAZERO
                "6:",
                cum_seq!($store_macro,$nr,$is_partial),

                ax = inout(reg) a => _,
                bx = inout(reg) b => _,
                cx = inout(reg) cf => _,
                dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                alphax = inout(reg) alpha => _,
                betax = inout(reg) beta => _,
                maskx = inout(reg) mask_ptr => _,
                onex = inout(reg) &one => _,
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
            alpha: *const f32, beta: *const f32,
            k: usize,
            d_arr: [usize; 4],
            m: usize, n: usize,
            f: F,
        ) {
            mask_ptr!($is_partial, m, x);
            let mask_ptr = x;
            let k = (k+1) / 2 *2;
            let k_iter = k / 8;
            let k_left = (k % 8) / 2;
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*4, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [0i32;$mr*$nr];
            let c_cs = d_arr[3];
            let one = 1_f32;
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, $mr);
                dim_arr[2] = $mr*4;
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
                            prefetch_0!(128, "({bx})"),
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
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),

                            "8:",
                            load_mask_ptr_asm!($is_partial),
                            load_beta!(),

                            // 6 -> BETAZERO
                            "je 6f",

                            // check if beta is equal to 1
                            "vucomiss ({onex}), %xmm0",
                            "je 9f",

                            cum_seq!($acc_macro,ni,$is_partial,2),
                            "jmp 6f",

                            "9:",
                            // 9 -> BETA ONE
                            cum_seq!($acc_macro,ni,$is_partial,1),

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
                            onex = inout(reg) &one => _,
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

def_ukernel!(step_16x4, acc_16x4, store_16x4, 16, 4, B, B, C, ukernel_16x4_bb);
// def_ukernel!(step_8x4, acc_8x4, store_8x4, 8, 4, B, B, C, 4, ukernel_16x8_bb);

def_ukernel!(step_16x4, acc_16x4, store_16x4, 16, 4, B, B, M, ukernel_16x4_bb_partial);
def_ukernel!(step_8x4, acc_8x4, store_8x4, 8, 4, B, B, M, ukernel_8x4_bb_partial);


def_ukernelxn!(step_16x4, acc_16x4, store_16x4, 16, 4, B, B, C, ukernel_16xn_bb);
// def_ukernelxn!(step_16x4, acc_16x4, store_16x4, 16, 4, B, B, C, 4, ukernel_16xn_bb);
// def_ukernelxn!(step_8x4, acc_8x4, store_8x4, 8, 4, B, B, C, 4, ukernel_16xn_bb);

def_ukernelxn!(step_16x4, acc_16x4, store_16x4, 16, 4, B, B, M, ukernel_16xn_bb_partial);
def_ukernelxn!(step_8x4, acc_8x4, store_8x4, 8, 4, B, B, M, ukernel_8xn_bb_partial);
