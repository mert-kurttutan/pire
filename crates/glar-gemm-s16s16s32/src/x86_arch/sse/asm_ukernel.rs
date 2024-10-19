use seq_macro::seq;
use std::arch::asm;
use std::arch::x86::_mm_prefetch;
use crate::MyFn;
use crate::{TA, TB, TC, TC_SIZE};
use glar_base::{load_buf, store_buf, c_mem, prefetch_0};

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
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "movups %xmm", $r2, ", %xmm", $r4, "\n",
            "pmaddwd %xmm", $r1, ", %xmm", $r4, "\n",
            "paddd %xmm", $r4, ", %xmm", $r3, "\n",
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

macro_rules! asm_alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                // jmp to 8 if alpha is equal to onex
                "mov 4({ptr_arrx}), {ax}\n",
                vbroadcast!(), " ({ax}),%xmm1", "\n",
                "shufps ", "$0, %xmm1, %xmm1", "\n",
                "mov ({ptr_arrx}), {ax}\n",
                "ucomiss ({ax}), %xmm1 \n",
                "je 8f \n",
                #(
                    "cvtdq2ps %xmm", r, ",%xmm", r, "\n",
                    "mulps %xmm1, %xmm", r, "\n",
                    "cvtps2dq %xmm", r, ",%xmm", r, "\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "mov 8({ptr_arrx}), {ax}\n",
            vbroadcast!(), " ({ax}), %xmm0\n",
            "shufps $0, %xmm0, %xmm0\n",
            "xorps %xmm3,%xmm3\n",
            "ucomiss %xmm3,%xmm0\n",
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
    (8, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x10"), 1),
        )
    };
    (4, $layout:tt, $m0:expr) => {
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
            "mov 12({dim_arrx}),{x0}", "\n",
            "test {x0},{x0}", "\n",
        )
    };
}


macro_rules! asm_c_load {
    (2) => {
        concat!(
            // "mov ({ptr_arrx}), {cx}", "\n",
            "mov 8({dim_arrx}),{x0}", "\n",
        )
    };
    (1) => {
        concat!(
            // "mov ({ptr_arrx}), {cx}", "\n",
            "mov 8({dim_arrx}),{x0}", "\n",
        )
    };
}


macro_rules! asm_vzeroall {
    (8,2) => {vzeroall!(4,7)};
    (8,1) => {vzeroall!(4,5)};

    (4,2) => {vzeroall!(4,5)};
    (4,1) => {vzeroall!(4,4)};
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
    (8, 2) => {asm_alpha_scale_0!(4,7)};
    (8, 1) => {asm_alpha_scale_0!(4,5)};

    (4, 2) => {asm_alpha_scale_0!(4,5)};
    (4, 1) => {asm_alpha_scale_0!(4,4)};
}

macro_rules! c_reg_2x2 {
    (0,0) => { 4 }; (1,0) => { 5 };
    (0,1) => { 6 }; (1,1) => { 7 };
}

macro_rules! c_reg_1x2 {
    (0,0) => { 4 };
    (0,1) => { 5 };
}

macro_rules! acc_2x2 {
    ($ni:tt, $layout:tt, $b:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_2x2!(0,$ni), c_reg_2x2!(1,$ni), $b)
    };
}

macro_rules! store_2x2 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_2x2!(0,$ni), c_reg_2x2!(1,$ni))
    };
}

macro_rules! acc_1x2 {
    ($ni:tt, $layout:tt, $b:tt) => {
        acc_p!($layout, c_mem!($ni), c_reg_1x2!(0,$ni), $b)
    };
}

macro_rules! store_1x2 {
    ($ni:tt, $layout:tt) => {
        storep!($layout, c_mem!($ni), c_reg_1x2!(0,$ni))
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
            vbroadcast!(), "  ", $K, "*", $X, "*4+", $N, "*4({bx}), %xmm", $r, "\n",
            "shufps $0, %xmm", $r, ", %xmm", $r, "\n",
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
            vfmadd!(0, 2, 4, 3),
            vfmadd!(1, 2, 5, 3),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 6, 3),
            vfmadd!(1, 2, 7, 3),
        )
    };
}

macro_rules! fmadd_1v {
    (0) => {
        concat!(
            vfmadd!(0, 1, 4, 6),
        )
    };
    (1) => {
        concat!(
            vfmadd!(0, 2, 5, 7),
        )
    };
}

macro_rules! b_num_2x2 {
    (0) => {2};
    (1) => {2};
}

macro_rules! b_num_1x2 {
    (0) => {1};
    (1) => {2};
}

// ***************************** 2x2 ******************************* //
macro_rules! step_2x2 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(8, $a_layout, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_2x2!(n)),
                    fmadd_2v!(n),
                )*
            )
        })
    };
}

// ***************************** 1x2 ******************************* //
macro_rules! step_1x2 {
    ($nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(4, $a_layout, $K),
                #(
                    load_b!($b_layout, n, $K, $nr, b_num_1x2!(n)),
                    fmadd_1v!(n),
                )*
            )
        })
    };
}

macro_rules! prefetch_c {
    (8, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(15+j*$ldc) as *const i8, 3);
        });
    };
    (4, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(2+j*$ldc) as *const i8, 3);
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
        pub(crate) unsafe fn $func_name<F: MyFn, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const f32, beta: *const f32,
            k: usize,
            d_arr: [usize; 4],
            m: usize,
            f: F,
        ) {
            let (k_i, k_l) = (k / 8, (k % 8) / 2);
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*TC_SIZE, k_i, k_l];
            let one = 1_f32;
            let ptr_arr = [&one, alpha, beta];
            let mut cf = c;
            let mut c_buf = [0i32;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF || m != $mr {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, $nr, $mr);
                dim_arr[2] = $mr*TC_SIZE;
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
                asm_c_load!($nr),
                // scale by alpha
                asm_alpha_scale!($mr, $nr),

                "8:",
                load_beta!(),

                // 6 -> BETAZERO
                "je 6f",

                // check if beta is equal to 1
                "mov ({ptr_arrx}), {ax}",
                "ucomiss ({ax}), %xmm0",
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
                ptr_arrx = inout(reg) ptr_arr.as_ptr() => _,
                dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                x0 = out(reg) _, 
                cx = inout(reg) cf => _,
                // x2 = out(reg) _,
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
        pub(crate) unsafe fn $func_name<F: MyFn, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const f32, beta: *const f32,
            k: usize,
            d_arr: [usize; 4],
            m: usize, n: usize,
            f: F,
        ) {
            let (k_i, k_l) = (k / 8, (k % 8) / 2);
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*TC_SIZE, k_i, k_l];
            let one = 1_f32;
            let ptr_arr = [&one, alpha, beta];
            let mut cf = c;
            let mut c_buf = [0i32;$mr*$nr];
            let c_cs = d_arr[3];
            if BUF || m != $mr {
                load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, $mr);
                dim_arr[2] = $mr*TC_SIZE;
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
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),

                            "8:",
                            load_beta!(),

                            // 6 -> BETAZERO
                            "je 6f",

                            // check if beta is equal to 1
                            "mov ({ptr_arrx}), {ax}",
                            "ucomiss ({ax}), %xmm0",
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
                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            ptr_arrx = inout(reg) ptr_arr.as_ptr() => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            x0 = out(reg) _,
                            cx = inout(reg) cf => _,
                            // onex = inout(reg) &one => _,
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

def_ukernel!(step_2x2, acc_2x2, store_2x2, 8, 2, B, B, C, ukernel_bb);

def_ukernel!(step_2x2, acc_2x2, store_2x2, 8, 2, B, B, C, ukernel_2_bb_partial);
def_ukernel!(step_1x2, acc_1x2, store_1x2, 4, 2, B, B, C, ukernel_1_bb_partial);


def_ukernelxn!(step_2x2, acc_2x2, store_2x2, 8, 2, B, B, C, ukernel_n_bb);

def_ukernelxn!(step_2x2, acc_2x2, store_2x2, 8, 2, B, B, C, ukernel_2xn_bb_partial);
def_ukernelxn!(step_1x2, acc_1x2, store_1x2, 4, 2, B, B, C, ukernel_1xn_bb_partial);
