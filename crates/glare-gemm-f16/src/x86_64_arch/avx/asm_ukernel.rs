use seq_macro::seq;
use std::arch::asm;
use half::f16;
use crate::{load_buf, store_buf};
use glare_base::c_mem;

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "vcvtph2ps ", $m0, ", %ymm2\n",
            "vmulps %ymm2,%ymm0,%ymm2", "\n",
            "vaddps %ymm2,%ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    // (M, $m0:expr, $r1:expr) => {
    //     concat!(
    //         "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
    //         "vfmadd231ps %ymm2, %ymm0,%ymm", $r1, "\n",
    //     ) 
    // };
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
            "vmulps %ymm", $r1, ", %ymm", $r2,", %ymm", $r4, "\n",
            "vaddps %ymm", $r4, ", %ymm", $r3, ", %ymm", $r3, "\n",
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
            // "vmovups %ymm", $r1, ", ", $m0,  "\n",
            "vcvtps2ph $0x00, %ymm", $r1, ", ", $m0, "\n",
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
                vbroadcast!(), " ({alphax}),%ymm1", "\n",
                #(
                    "vmulps %ymm1, %ymm", r, ",%ymm", r, "\n",
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
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!(C, mem!($m0, "0x10"), $r2),
            beta_fmadd!($layout, mem!($m0, "0x20"), $r3),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!($layout, mem!($m0, "0x10"), $r2),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1),
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
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            storep_unit!(C, $r1, $m0),
            storep_unit!(C, $r2, mem!($m0, "0x10")),
            storep_unit!($layout, $r3, mem!($m0, "0x20")),
        )
    };
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
    (16,4) => {vzeroall!(8,15)};
    (16,3) => {vzeroall!(8,13)};
    (16,2) => {vzeroall!(8,11)};
    (16,1) => {vzeroall!(8,9)};

    (8,4) => {vzeroall!(9,12)};
    (8,3) => {vzeroall!(9,11)};
    (8,2) => {vzeroall!(9,10)};
    (8,1) => {vzeroall!(9,9)};
}
// macro_rules! inc_a {
// 	(C) => {
//     	"add {x1}, {ax} \n"
// 	};
// 	(B) => {
//     	""
// 	};
// }

macro_rules! inc_b {
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
        ""
    };
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
    (16, 4) => {
        asm_alpha_scale_0!(8,15)
    };
    (16, 3) => {
        asm_alpha_scale_0!(8,13)
    };
    (16, 2) => {
        asm_alpha_scale_0!(8,11)
    };
    (16, 1) => {
        asm_alpha_scale_0!(8,9)
    };

    (8, 4) => {
        asm_alpha_scale_0!(9,12)
    };
    (8, 3) => {
        asm_alpha_scale_0!(9,11)
    };
    (8, 2) => {
        asm_alpha_scale_0!(9,10)
    };
    (8, 1) => {
        asm_alpha_scale_0!(9,9)
    };
}

macro_rules! c_reg_16x4 {
    (0,0) => { 8 };
    (1,0) => { 9 };
    (0,1) => { 10 };
    (1,1) => { 11 };
    (0,2) => { 12 };
    (1,2) => { 13 };
    (0,3) => { 14 };
    (1,3) => { 15 };
}

macro_rules! c_reg_8x4 {
    (0,0) => { 9 };
    (0,1) => { 10 };
    (0,2) => { 11 };
    (0,3) => { 12 };
}


macro_rules! acc_16x4 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_16x4!(0,$ni), c_reg_16x4!(1,$ni))
    };
}

macro_rules! store_16x4 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_16x4!(0,$ni), c_reg_16x4!(1,$ni))
    };
}

macro_rules! acc_8x4 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_8x4!(0,$ni))
    };
}

macro_rules! store_8x4 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_8x4!(0,$ni))
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
    (B, $N:tt, $K:tt, $X:tt, $r:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*4+", $N, "*4({bx}), %ymm", $r, "\n",
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
            vfmadd!(0, 2, 8, 4),
            vfmadd!(1, 2, 9, 5),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 3, 10, 6),
            vfmadd!(1, 3, 11, 7),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 2, 12, 4),
            vfmadd!(1, 2, 13, 5),
        )
    };
    (3) => {
        concat!(
            vfmadd!(0, 3, 14, 6),
            vfmadd!(1, 3, 15, 7),
        )
    };
}

macro_rules! fmadd_1v {
    (0) => {
        concat!(vfmadd!(0, 1, 9, 5))
    };
    (1) => {
        concat!(vfmadd!(0, 2, 10, 6))
    };
    (2) => {
        concat!(vfmadd!(0, 3, 11, 7))
    };
    (3) => {
        concat!(vfmadd!(0, 4, 12, 8))
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
                inc_b!($b_layout,$nr), 
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
    (24, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
        });
    };
    (16, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
        });
    };
    (8, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(4+j*$ldc) as *const i8, 3);
        });
    }
}

use crate::MyFn;

// #[inline(always)]
// fn mask_and_offset(m: usize) -> ([u32;16], usize) {
// 	let mask: [u32; 16] = [
// 		u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX,
// 		0, 0, 0, 0, 0, 0, 0, 0,
// 	];
// 	let mask_offset = if m % VS == 0 { 0 } else { VS - (m %VS)};

// 	(mask, mask_offset)
// }



// macro_rules! mask_ptr {
// 	(M, $m:tt, $nm:ident) => {
// 		let (mask, mask_offset) = mask_and_offset($m);
// 		let $nm = mask.as_ptr().add(mask_offset);
// 	};
// 	(C, $m:tt, $nm:ident) => {
// 		let mask = [0xFFFF_u32];
// 		let $nm = mask.as_ptr();
// 	};
// }

// macro_rules! load_mask_ptr_asm {
// 	(M) => {
// 		"vmovdqu ({maskx}), %ymm1"
// 	};
// 	(C) => {
// 		"/* {maskx} */"
// 	}
// }

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
            a: *const f32, b: *const f32, c: *mut f16,
            alpha: *const f32, beta: *const f32,
            k: usize,
            d_arr: [usize; 4],
            m: usize,
            f: F,
        ) {
            // mask_ptr!($is_partial, m, x);
            // let mask_ptr = x;
            let k_iter = k / 4;
            let k_left = k % 4;
            let mut dim_arr = [d_arr[0]*2, d_arr[1]*2, d_arr[3]*2, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [f16::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr);
                dim_arr[2] = m*2;
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
                asm_c_load!($nr),
                // scale by alpha
                asm_alpha_scale!($mr, $nr),

                load_beta!(),

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
                // maskx = inout(reg) mask_ptr => _,
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
            a: *const f32, b: *const f32, c: *mut f16,
            alpha: *const f32, beta: *const f32,
            k: usize,
            d_arr: [usize; 4],
            m: usize, n: usize,
            f: F,
        ) {
            // mask_ptr!($is_partial, m, x);
            // let mask_ptr = x;
            let k_iter = k / 4;
            let k_left = k % 4;
            let mut dim_arr = [d_arr[0]*2, d_arr[1]*2, d_arr[3]*2, k_iter, k_left];
            let mut cf = c;
            let mut c_buf = [f16::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n);
                dim_arr[2] = m*2;
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
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),

                            load_beta!(),

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
                            // maskx = inout(reg) mask_ptr => _,
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

// NOTE: BS ukernel for f16 is hard to implement since it requires loading single f16 in a strided fashion
// we can do this, it will require avx2, and som other issues, which I dont want to deal with
// Additiionally and more importantly, it wont be performant neough since it reqiures to convert additioanl
// computation, it wont benefit from vectorization since we load single f16 in strided layout.

// Dont use partial since partially (and efficiently at the same time) is hard instead copy to c buffer

def_ukernel!(step_16x4, acc_16x4, store_16x4, 16, 4, B, B, C, ukernel_16x4_bb);
def_ukernel!(step_8x4, acc_8x4, store_8x4, 8, 4, B, B, C, ukernel_8x4_bb);

// def_ukernel!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, B, f32, M, 4, ukernel_24x4_bb_partial);
// def_ukernel!(VER2, step_16x6, acc_16x6, store_16x6, 16, 4, B, B, f32, M, 4, ukernel_16x4_bb_partial);
// def_ukernel!(VER1, step_8x6, acc_8x6, store_8x6, 8, 4, B, B, f32, M, 4, ukernel_8x4_bb_partial);

// def_ukernel!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, S, f16, C, 4, ukernel_24x4_bs);

// def_ukernel!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, S, f16, M, 4, ukernel_24x4_bs_partial);
// def_ukernel!(VER2, step_16x6, acc_16x6, store_16x6, 16, 4, B, S, f16, M, 4, ukernel_16x4_bs_partial);
// def_ukernel!(VER1, step_8x6, acc_8x6, store_8x6, 8, 4, B, S, f16, M, 4, ukernel_8x4_bs_partial);


def_ukernelxn!(step_16x4, acc_16x4, store_16x4, 16, 4, B, B, C, ukernel_16xn_bb);
def_ukernelxn!(step_8x4, acc_8x4, store_8x4, 8, 4, B, B, C, ukernel_8xn_bb);

// def_ukernelxn!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, B, f32, M, 4, ukernel_24xn_bb_partial);
// def_ukernelxn!(VER2, step_16x6, acc_16x6, store_16x6, 16, 4, B, B, f32, M, 4, ukernel_16xn_bb_partial);
// def_ukernelxn!(VER1, step_8x6, acc_8x6, store_8x6, 8, 4, B, B, f32, M, 4, ukernel_8xn_bb_partial);

// def_ukernelxn!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, S, f16, C, 4, ukernel_24xn_bs);

// def_ukernelxn!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, S, f16, M, 4, ukernel_24xn_bs_partial);
// def_ukernelxn!(VER2, step_16x6, acc_16x6, store_16x6, 16, 4, B, S, f16, M, 4, ukernel_16xn_bs_partial);
// def_ukernelxn!(VER1, step_8x6, acc_8x6, store_8x6, 8, 4, B, S, f16, M, 4, ukernel_8xn_bs_partial);


