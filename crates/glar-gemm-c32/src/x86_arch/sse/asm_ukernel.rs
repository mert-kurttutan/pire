use seq_macro::seq;
use std::arch::asm;
use std::arch::x86::_mm_prefetch;
use crate::{TA, TB, TC, TC_SIZE};
use crate::UnaryFnC;
use glar_base::{load_buf, store_buf, c_mem, prefetch_0};

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "movups ", $m0, ",%xmm2", "\n",
            "addps ", "%xmm2", ",%xmm", $r1, "\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "movups ", $m0, ", %xmm5", "\n",
            complex_mul!(5, 7),
            "addps %xmm5, %xmm", $r1, "\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("xorpd %xmm",r,",%xmm",r,"\n",)*)
        })
    }
}
macro_rules! vmovp {
    (B) => {
        "movaps "
    };
    ($layout:tt) => {
        "movups "
    };
}

macro_rules! vbroadcast {
    () => {
        "movss"
    };
}

macro_rules! vfmadd {
    ($r1:expr, $b1:expr, $r2:expr, $r4:expr) => {
        concat!(
            "movups %xmm", $b1, ", %xmm", $r4, "\n",
            "mulps %xmm", $r1, ", %xmm", $r4, "\n",
            "addps %xmm", $r4, ", %xmm", $r2, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovp!($layout), $m0, ",%xmm", $r1, "\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "movups %xmm", $r1, ", ", $m0,  "\n",
        )
    };
    (B, $r1:expr, $m0:expr) => {
        concat!(
            "movaps %xmm", $r1, ", ", $m0,  "\n",
        )
    };

}

macro_rules! complex_mul {
    ($r0:tt, $rt:tt) => {
        concat!(
            "movaps %xmm", $r0, ", %xmm", $rt, "\n",
            "shufps $0xb1, %xmm", $r0, ", %xmm", $rt, "\n",
            "mulps %xmm0, %xmm", $r0, "\n",
            "mulps %xmm1, %xmm", $rt, "\n",
            "addsubps %xmm", $rt, ", %xmm", $r0, "\n",
        )
    };
}

macro_rules! asm_alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        concat!(
            "mov 4({ptr_arrx}), {x1}\n",
            vbroadcast!(), " ({x1}),%xmm0", "\n",
            "shufps ", "$0, %xmm0, %xmm0", "\n",

            vbroadcast!(), " 4({x1}),%xmm1", "\n",
            "shufps ", "$0, %xmm1, %xmm1", "\n",
        
            complex_mul!(4, 5),
            complex_mul!(6, 7),
        )
    }
}

macro_rules! v_to_c {
    ($r0:tt, $r1:tt) => {
        concat!(
            "shufps $0xb1, %xmm", $r1, ", %xmm", $r1, "\n",
            "addsubps %xmm", $r1, ", %xmm", $r0, "\n",
        )
    }
}

macro_rules! permute_complex {
    () => {
        concat!(
            // permute even and odd elements
            v_to_c!(4, 5),
            v_to_c!(6, 7),
        )
    }
}

macro_rules! acc_p {
    ($layout:tt, $m0:expr, $q:tt, $r1:expr) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1, $q),
        )
    };
}


macro_rules! loadp {
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
            storep_unit!($layout, $r2, mem!($m0, "0x10")),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            storep_unit!($layout, $r1, $m0),
        )
    };
}

macro_rules! load_beta {
    () => {
        concat!(
            "mov 8({ptr_arrx}), {x1}\n",
            "movss ({x1}), %xmm0 \n",
            "shufps ", "$0, %xmm0, %xmm0", "\n",
            "movss 4({x1}), %xmm1 \n",
            "shufps ", "$0, %xmm1, %xmm1", "\n",
        )
    }
}

// only non contigous along m and n direction which is not changed frequently during iteration along k direction
/*

x1 -> cs_a
x2 -> cs_b
x3 -> ax + 3*cs_a
x4 -> cx + 3*cs_b

*/


macro_rules! asm_init_ab {
    ($KER:tt,B,B) => {
        concat!(
            "/* {x1} */", "\n",
            "mov 12({dim_arrx}),{x0}", "\n",
            "test {x0},{x0}", "\n",
        )
    };
    ($ker:tt,B,S) => {
        concat!(
            // mov cs_b to reg
            "mov ({dim_arrx}), {x1}", "\n",
            // "mov 8({dim_arrx}), {x2}", "\n",

            "mov 12({dim_arrx}),{x0}", "\n",
            "test {x0},{x0}", "\n",
        )
    };
}


macro_rules! asm_c_load {
    (2) => {
        concat!(
            "mov ({ptr_arrx}), {cx}", "\n",
            "mov 8({dim_arrx}),{x0}", "\n",
        )
    };
    (1) => {
        concat!(
            "mov ({ptr_arrx}), {cx}", "\n",
            "mov 8({dim_arrx}),{x0}", "\n",
        )
    };
}


macro_rules! asm_vzeroall {
    (2,2) => {vzeroall!(4,7)};
    (2,1) => {vzeroall!(4,5)};
}	

macro_rules! inc_b {
    (S,2) => {
        "add {x1},{cx} \n"
    };
    (S,1) => {
        "add {x1},{cx} \n"
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
            "add $8*", $K, "*", $X, ", {cx}", "\n",
        )
    };
}


macro_rules! asm_alpha_scale {
    (2, 2) => {
        asm_alpha_scale_0!(4,7)
    };
    (2, 1) => {
        asm_alpha_scale_0!(4,5)
    };
}

macro_rules! c_reg_1x2 {
    (0,0) => { 4 };
    (0,1) => { 6 };
}

macro_rules! acc_1x2 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p!($layout, c_mem!($ni), $q, c_reg_1x2!(0,$ni))
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

    ($step_macro:tt, $nr:tt, $layout:tt, $q:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!(n, $layout, $q),)*)
        })
    };
}


macro_rules! load_b {
    (S, 0, $K:tt, $X:tt, $r1:expr, $r2:expr) => {
        concat!(
            vbroadcast!(), " ({cx}),%xmm", $r1, "\n",
            "shufps $0, %xmm", $r1, ", %xmm", $r1, "\n",
        )
    };
    (S, 1, $K:tt, $X:tt, $r1:expr, $r2:expr) => {
        concat!(
            // "prefetcht0 64({cx},{x2},1) \n",
            // vbroadcast!(), " ({cx},{x2},1),%xmm", $r1, "\n",
            "shufps $0, %xmm", $r1, ", %xmm", $r1, "\n",
        )
    };
    (B, $N:tt, $K:tt, $X:tt, $r1:expr, 0) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*8+", $N, "*8({cx}), %xmm", $r1, "\n",
            "shufps $0, %xmm", $r1, ", %xmm", $r1, "\n",
        )
    };

    (B, $N:tt, $K:tt, $X:tt, $r1:expr, 1) => {
        concat!(
            vbroadcast!(), " ", $K, "*", $X, "*8+", $N, "*8+4({cx}), %xmm", $r1, "\n",
            "shufps $0, %xmm", $r1, ", %xmm", $r1, "\n",
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

macro_rules! fmadd_1v {
    (0, 0) => {
        concat!(
            vfmadd!(0, 1, 4, 2),
        )
    };
    (0, 1) => {
        concat!(
            vfmadd!(0, 1, 5, 3),
        )
    };
    (1, 0) => {
        concat!(
            vfmadd!(0, 1, 6, 2),
        )
    };
    (1, 1) => {
        concat!(
            vfmadd!(0, 1, 7, 3),
        )
    };
}

// ***************************** 1x2 ******************************* //
macro_rules! step_1x2 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2, $a_layout, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, 1, 0),
                    fmadd_1v!(n, 0),
                    load_b!($b_layout, n, $K, $nr, 1, 1),
                    fmadd_1v!(n, 1),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! prefetch_c {
    (2, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
        });
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
            alpha: *const TA, 
            beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize,
            f: F,
        ) {
            let (k_i, k_l) = (k / 4, k % 4);
            let beta_st = if *beta == TB::ZERO {
                0i32
            } else if *beta == TB::ONE {
                1i32
            } else {
                2i32
            };
            let mut dim_arr = [d_arr[0]*8, d_arr[1]*8, d_arr[3]*TC_SIZE, k_i, k_l, beta_st as usize];
            let mut ptr_arr = [c, alpha, beta];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF || m != $mr {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, $mr);
                dim_arr[2] = $mr*TC_SIZE;
                cf = c_buf.as_mut_ptr();
                ptr_arr[0] = cf;
            }
            prefetch_c!($mr,$nr,c,c_cs);
            asm!(
                asm_vzeroall!($mr,$nr),
        
                asm_init_ab!($mr,$a_layout,$b_layout),
                
                // 3 -> CONSIDKLEFT
                "je 3f",
                
                // 2 -> KITER
                "2:",
                prefetch_0!(128, "{cx}"),
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
                "mov 16({dim_arrx}),{x0}",
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

                "cmpw $0, 20({dim_arrx})",
                "je 6f",
        
                "cmpw $1, 20({dim_arrx})",
                "je 15f",

                load_beta!(),
                cum_seq!($acc_macro,$nr,$is_partial,2),
                "jmp 6f",

                "15:",
                cum_seq!($acc_macro,$nr,$is_partial,1),

                "6:",
                cum_seq!($store_macro,$nr,$is_partial),
                
                // 7 -> DDONE
                "7:",
                ax = inout(reg) a => _,
                cx = inout(reg) b => _,
                ptr_arrx = inout(reg) ptr_arr.as_ptr() => _,
                dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                x0 = out(reg) _,
                x1 = out(reg) _,
                out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                options(att_syntax)
            );
            if BUF || m != $mr {
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
            alpha: *const TA, 
            beta: *const TB,
            k: usize,
            d_arr: [usize; 4],
            m: usize, n: usize,
            f: F,
        ) {
            let (k_i, k_l) = (k / 4, k % 4);
            let beta_st = if *beta == TB::ZERO {
                0i32
            } else if *beta == TB::ONE {
                1i32
            } else {
                2i32
            };
            let mut dim_arr = [d_arr[0]*8, d_arr[1]*8, d_arr[3]*TC_SIZE, k_i, k_l, beta_st as usize];
            let mut ptr_arr = [c, alpha, beta];
            let mut cf = c;
            let mut c_buf = [TC::ZERO;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF || m != $mr {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, $mr);
                dim_arr[2] = $mr*TC_SIZE;
                cf = c_buf.as_mut_ptr();
                ptr_arr[0] = cf;
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
                            prefetch_0!(128, "{cx}"),
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
                            "mov 16({dim_arrx}),{x0}",
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

                            "cmpw $0, 20({dim_arrx})",
                            "je 6f",
                    
                            "cmpw $1, 20({dim_arrx})",
                            "je 15f",

                            load_beta!(),
                            cum_seq!($acc_macro,ni,$is_partial,2),	
                            "jmp 6f",	

                            "15:",
                            cum_seq!($acc_macro,ni,$is_partial,1),

                            "6:",
                            cum_seq!($store_macro,ni,$is_partial),
                            
                            // 7 -> DDONE
                            "7:",
                            ax = inout(reg) a => _,
                            cx = inout(reg) b => _,
                            ptr_arrx = inout(reg) ptr_arr.as_ptr() => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                            out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                            options(att_syntax)
                        );
                        break 'blk;
                    }
                });
            };
            if BUF || m != $mr {
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

def_ukernel!(step_1x2, acc_1x2, store_1x2, 2, 2, B, B, C, ukernel_bb);

def_ukernel!(step_1x2, acc_1x2, store_1x2, 2, 2, B, B, C, ukernel_1_bb_partial);

def_ukernel!(step_1x2, acc_1x2, store_1x2, 2, 2, B, S, C, ukernel_bs);


def_ukernel!(step_1x2, acc_1x2, store_1x2, 2, 2, B, S, C, ukernel_1_bs_partial);

def_ukernelxn!(step_1x2, acc_1x2, store_1x2, 2, 2, B, B, C, ukernel_n_bb);

def_ukernelxn!(step_1x2, acc_1x2, store_1x2, 2, 2, B, B, C, ukernel_1xn_bb_partial);

def_ukernelxn!(step_1x2, acc_1x2, store_1x2, 2, 2, B, S, C, ukernel_n_bs);

def_ukernelxn!(step_1x2, acc_1x2, store_1x2, 2, 2, B, S, C, ukernel_1xn_bs_partial);


