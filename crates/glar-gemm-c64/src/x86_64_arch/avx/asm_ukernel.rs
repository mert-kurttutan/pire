use seq_macro::seq;
use std::arch::asm;
use std::arch::x86_64::_mm_prefetch;
use super::VS;
use crate::{TA, TB, TC};
use crate::MyFn;
use glar_base::{load_buf, store_buf, c_mem, prefetch_0};

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "vaddpd ", $m0, ",%ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "vmaskmovpd ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vaddpd %ymm2, %ymm", $r1, ", %ymm", $r1, "\n",
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
    ($r1:expr, $b1:expr, $b2:expr, $r2:expr, $r3:expr, $r4:expr, $r5:expr) => {
        concat!(
            "vmulpd %ymm", $r1, ", %ymm", $b1,", %ymm", $r4, "\n",
            "vaddpd %ymm", $r4, ", %ymm", $r2, ", %ymm", $r2, "\n",

            "vmulpd %ymm", $r1, ", %ymm", $b2,", %ymm", $r5, "\n",
            "vaddpd %ymm", $r5, ", %ymm", $r3, ", %ymm", $r3, "\n",
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
            "vmovupd %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (B, $r1:expr, $m0:expr) => {
        concat!(
            "vmovapd %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
            "vmaskmovpd %ymm", $r1, ", %ymm1, ", $m0,  "\n",
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

macro_rules! asm_alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        concat!(
            "vbroadcastsd ({alphax}), %ymm0 \n",
            "vbroadcastsd 8({alphax}), %ymm1 \n",
            
            complex_mul!(4, 5),
            complex_mul!(6, 7),
            complex_mul!(8, 9),
            complex_mul!(10, 11),
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
        )
    }
}

macro_rules! mem {
    ($m0:tt, $b0:tt) => {
        concat!($b0, "+", $m0)
    };
}

macro_rules! acc_p {
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
    (4, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x20"), 1),
        )
    };
    (2, $layout:tt, $m0:expr) => {
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
    (4,2) => {vzeroall!(4,11)};
    (4,1) => {vzeroall!(4,7)};

    (2,2) => {vzeroall!(4,7)};
    (2,1) => {vzeroall!(4,5)};
}	

macro_rules! inc_b {
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
            "add $16*", $K, "*", $X, ", {bx}", "\n",
        )
    };
}


macro_rules! asm_alpha_scale {

    (4, 2) => {
        asm_alpha_scale_0!(4,11)
    };
    (4, 1) => {
        asm_alpha_scale_0!(4,7)
    };

    (2, 2) => {
        asm_alpha_scale_0!(4,7)
    };
    (2, 1) => {
        asm_alpha_scale_0!(4,5)
    };
}


macro_rules! c_reg_2x2 {
    (0,0) => { 4 };
    (1,0) => { 6 };
    (0,1) => { 8 };
    (1,1) => { 10 };
}

macro_rules! c_reg_1x2 {
    (0,0) => { 4 };
    (0,1) => { 6 };
}

macro_rules! acc_2x2 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_2x2!(0,$ni), c_reg_2x2!(1,$ni))
    };
}

macro_rules! store_2x2 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_2x2!(0,$ni), c_reg_2x2!(1,$ni))
    };
}

macro_rules! acc_1x2 {
    ($ni:tt, $layout:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_1x2!(0,$ni))
    };
}

macro_rules! store_1x2 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_1x2!(0,$ni))
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
    (B, $N:tt, $K:tt, $X:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*16+", $N, "*16({bx}), %ymm", $r1, "\n",
            vbroadcast!(), " ", $K, "*", $X, "*16+", $N, "*16+8({bx}), %ymm", $r2, "\n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt, B, $K:tt) => {
        loadp!($mr, B, concat!($mr,"*16*",$K,"({ax})"))
    };
    ($mr:tt, C, $K:tt) => {
        loadp!($mr, C, "0({ax})")
    };
}

macro_rules! fmadd_2v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 3, 4, 5, 12, 13),
            vfmadd!(1, 2, 3, 6, 7, 14, 15),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 3, 8, 9, 12, 13),
            vfmadd!(1, 2, 3, 10, 11, 14, 15),
        )
    };
}

macro_rules! fmadd_1v {
    (0) => {
        concat!(
            vfmadd!(0, 2, 3, 4, 5, 12, 13),
        )

    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 3, 6, 7, 14, 15),
        )
    };
}

// ***************************** 2x2 ******************************* //
macro_rules! step_2x2 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(4, $a_layout, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, 2, 3),
                    fmadd_2v!(n),
                )*
                inc_b!($b_layout,$nr),
            )
        })
    };
}

// ***************************** 1x2 ******************************* //
macro_rules! step_1x2 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2, $a_layout, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, 2, 3),
                    fmadd_1v!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! prefetch_c {
    (4, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(4+j*$ldc) as *const i8, 3);
        });
    };
    (2, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(4+j*$ldc) as *const i8, 3);
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
    // mask offset for 128 bit part of ymm register
    // since c64 is 128 bit wide
    (mask, mask_offset*2)
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
            // beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize,
            f: F,
        ) {
            mask_ptr!($is_partial, m, x);
            let mask_ptr = x;
            let (k_i, k_l) = (k / 4, k % 4);
            let mut dim_arr = [d_arr[0]*16, d_arr[1]*16, d_arr[3]*16, k_i, k_l];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, $mr);
                dim_arr[2] = $mr*16;
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
                prefetch_0!(384, "{bx}"),
                $step_macro!($nr, $a_layout, $b_layout, 0),
                $step_macro!($nr, $a_layout, $b_layout, 1),
                prefetch_0!(448, "{bx}"),
                $step_macro!($nr, $a_layout, $b_layout, 2),
                $step_macro!($nr, $a_layout, $b_layout, 3),
                prefetch_0!(512, "{bx}"),

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
            let (k_i, k_l) = (k / 4, k % 4);
            let mut dim_arr = [d_arr[0]*16, d_arr[1]*16, d_arr[3]*16, k_i, k_l];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, $mr);
                dim_arr[2] = $mr*16;
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
                            prefetch_0!(392, "{bx}"),
                            $step_macro!(ni, $a_layout, $b_layout, 0),
                            $step_macro!(ni, $a_layout, $b_layout, 1),
                            prefetch_0!(448, "{bx}"),
                            $step_macro!(ni, $a_layout, $b_layout, 2),
                            $step_macro!(ni, $a_layout, $b_layout, 3),
                            prefetch_0!(512, "{bx}"),
            
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

def_ukernel!(step_2x2, acc_2x2, store_2x2, 4, 2, B, B, C, ukernel_bb);
// def_ukernel!(step_1x2, acc_1x2, store_1x2, 8, 4, B, B, C, 4, ukernel_16x8_bb);


def_ukernel!(step_2x2, acc_2x2, store_2x2, 4, 2, B, B, M, ukernel_2_bb_partial);
def_ukernel!(step_1x2, acc_1x2, store_1x2, 2, 2, B, B, M, ukernel_1_bb_partial);

def_ukernel!(step_2x2, acc_2x2, store_2x2, 4, 2, B, S, C, ukernel_bs);


def_ukernel!(step_2x2, acc_2x2, store_2x2, 4, 2, B, S, M, ukernel_2_bs_partial);
def_ukernel!(step_1x2, acc_1x2, store_1x2, 2, 2, B, S, M, ukernel_1_bs_partial);

def_ukernelxn!(step_2x2, acc_2x2, store_2x2, 4, 2, B, B, C, ukernel_n_bb);
// def_ukernelxn!(step_1x2, acc_1x2, store_1x2, 8, 4, B, B, C, 4, ukernel_2xn_bb);

def_ukernelxn!(step_2x2, acc_2x2, store_2x2, 4, 2, B, B, M, ukernel_2xn_bb_partial);
def_ukernelxn!(step_1x2, acc_1x2, store_1x2, 2, 2, B, B, M, ukernel_1xn_bb_partial);

def_ukernelxn!(step_2x2, acc_2x2, store_2x2, 4, 2, B, S, C, ukernel_n_bs);

def_ukernelxn!(step_2x2, acc_2x2, store_2x2, 4, 2, B, S, M, ukernel_2xn_bs_partial);
def_ukernelxn!(step_1x2, acc_1x2, store_1x2, 2, 2, B, S, M, ukernel_1xn_bs_partial);


