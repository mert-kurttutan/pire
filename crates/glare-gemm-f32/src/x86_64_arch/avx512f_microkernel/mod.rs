pub mod asm_ukernel;

pub(crate) use asm_ukernel::*;

use seq_macro::seq;
use paste::paste;

use crate::{TA,TB,TC};

const VS: usize = 16;

use crate::MyFn;


macro_rules! def_kernel_bb {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        seq!( nr_left in 2..$NR { paste! {
            pub unsafe fn [<kernel_bb>]<F: MyFn, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                beta: *const TC,
                c: *mut TC, c_rs: usize, c_cs: usize,
                ap: *const TA, bp: *const TB,
                f: F,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let mut m_iter = (m / MR) as u64;
                let m_left = m % MR;
                let mut ap_cur = ap;
                let mut c_cur0 = c;
                
                
                let n_iter0 = (n / NR) as u64;
                let n_left = (n % NR) as u64;
                let ld_arr = [0, 0, c_rs, c_cs];
                
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut bp_cur = bp;
                    let mut c_cur1 = c_cur0;
                    while n_iter > 0 {
                        [<ukernel_$MR x $NR _bb>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, ld_arr, MR, NR, f);
                        n_iter -= 1;
                        bp_cur = bp_cur.add(NR*k);
                        c_cur1 = c_cur1.add(NR*c_cs);
                    }
                    if n_left == 1 {
                        [<ukernel_$MR x 1 _bb>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, ld_arr, MR, 1, f);
                    }
                    #(
                        else if n_left == nr_left {
                            [<ukernel_$MR x nr_left _bb>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, ld_arr, MR, nr_left, f);
                        }
                    )*
                    m_iter -= 1;
                    ap_cur = ap_cur.add(MR*k);
                    c_cur0 = c_cur0.add(MR*c_rs);
                }


                $(
                    if (m_left+VS-1) / VS *VS == $mr_left {
                        let mut n_iter = n_iter0;
                        let mut bp_cur = bp;
                        let mut c_cur1 = c_cur0;
                        while n_iter > 0 {
                            [<ukernel_$mr_left x $NR _bb_partial>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, ld_arr, m_left, NR, f);
                            n_iter -= 1;
                            bp_cur = bp_cur.add(NR*k);
                            c_cur1 = c_cur1.add(NR*c_cs);
                        }
                        if n_left == 1 {
                            [<ukernel_$mr_left x 1 _bb_partial>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, ld_arr, m_left, 1, f);
                        }
                        #(
                        else if n_left == nr_left {
                            [<ukernel_$mr_left x nr_left _bb_partial>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, ld_arr, m_left, nr_left, f);
                        }
                        )*
                    }
                )*
            }        
        }});
    };
}

def_kernel_bb!(48, 8, 48, 32, 16);


macro_rules! def_kernel_bb {
    ($MR:tt, $NR:tt) => {
        pub unsafe fn kernel_bb<F: MyFn, const STRIDED: bool>(
            m: usize, n: usize, k: usize,
            alpha: *const TA,
            beta: *const TC,
            c: *mut TC, c_rs: usize, c_cs: usize,
            ap: *const TA, bp: *const TB,
            f: F,
        ) {
            const MR: usize = $MR;
            const NR: usize = $NR;
            let mut d0_iter = m / MR;
            let mut ap0 = ap;
            let mut c0 = c;
            
            
            let d1_iter0 = n / NR;
            let ld_arr = [0, 0, c_rs, c_cs];
            
            while d0_iter > 0 {
                let mut d1_iter = d1_iter0;
                let mut bp0 = bp;
                let mut c1 = c0;
                while d1_iter > 0 { paste! {
                        [<ukernel_$MR x $NR _bb>]::<_, STRIDED>(ap0, bp0, c1, alpha, beta, k, ld_arr, MR, NR, f);
                    }
                    d1_iter -= 1;
                    bp0 = bp0.add(NR*k);
                    c1 = c1.add(NR*c_cs);
                }
                d0_iter -= 1;
                ap0 = ap0.add(MR*k);
                c0 = c0.add(MR*c_rs);
            }
        }
    };
}

// def_kernel_bb!(48, 8);

macro_rules! def_kernel_bs {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        seq!( nr_left in 2..$NR { paste! {
            pub unsafe fn [<kernel_bs _v0>]<F: MyFn, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                beta: *const TC,
                b: *const TB, b_rs: usize, b_cs: usize,
                c: *mut TC, c_rs: usize, c_cs: usize,
                ap_cur: *const TA,
                f: F,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let mut m_iter = (m / MR) as u64;
                let m_left = m % MR;
                let mut c_cur0 = c;
                let mut ap_cur = ap_cur;
                
                let n_iter0 = (n / NR) as u64;
                let n_left = (n % NR) as u64;
                let ld_arr = [b_rs, b_cs, c_rs, c_cs];
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut b_cur = b;
                    let mut c_cur1 = c_cur0;
                    while n_iter > 0 {
                        [<ukernel_$MR x $NR _bs>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, MR, NR, f);
                        n_iter -= 1;
                        b_cur = b_cur.add(NR*b_cs);
                        c_cur1 = c_cur1.add(NR*c_cs);
                    }
                    if n_left == 1 {
                        [<ukernel_$MR x1_bs>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, MR, 1, f);
                    }
                    #(
                        else if n_left == nr_left {
                            [<ukernel_$MR x~nr_left _bs>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, MR, nr_left, f);
                        }
                    )*
                    m_iter -= 1;
                    ap_cur = ap_cur.add(MR*k);
                    c_cur0 = c_cur0.add(MR*c_rs);
                }

                $(
                    if m_left > ($mr_left - VS) {
                        let mut n_iter = n_iter0;
                        let mut b_cur = b;
                        let mut c_cur1 = c_cur0;
                        while n_iter > 0 {
                            [<ukernel_$mr_left x $NR _bs_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, m_left, NR, f);
                            // [<ukernel_$mr_left x $NR _bs>]::<_, true>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, m_left, NR, f);
                            n_iter -= 1;
                            b_cur = b_cur.add(NR*b_cs);
                            c_cur1 = c_cur1.add(NR*c_cs);
                        }
                        if n_left == 1 {
                            [<ukernel_$mr_left x1_bs_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, m_left, 1, f);
                        }
                        #(
                        else if n_left == nr_left {
                            [<ukernel_$mr_left x~nr_left _bs_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, m_left, nr_left, f);
                        }
                        )*
                        return;
                    }
                )*
            }        
        }});
    };
}

def_kernel_bs!(48, 8, 48, 32, 16);

use super::pack_avx::packa_panel_48;
macro_rules! def_kernel_sb {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        seq!( nr_left in 2..$NR { paste! {
            pub unsafe fn [<kernel_sb_v0>]<F: MyFn, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                beta: *const TC,
                a: *const TB, a_rs: usize, a_cs: usize,
                b: *const TA,
                c: *mut TC, c_rs: usize, c_cs: usize,
                ap_buf: *mut TA,
                f: F,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let mut m_iter = (m / MR) as u64;
                let m_left = m % MR;
                let mut c_cur0 = c;
                let mut a_cur = a;
                
                let n_iter0 = (n / NR) as u64;
                let n_left = (n % NR) as u64;
                let ld_arr = [0, 0, c_rs, c_cs];
                let ap_cur = ap_buf;
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut b_cur = b;
                    let mut c_cur1 = c_cur0;
                    packa_panel_48(MR, k, a_cur, a_rs, a_cs, ap_cur);
                    while n_iter > 0 {
                        [<ukernel_$MR x $NR _bb>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, MR, NR, f);
                        n_iter -= 1;
                        b_cur = b_cur.add(NR*k);
                        c_cur1 = c_cur1.add(NR*c_cs);
                    }
                    if n_left == 1 {
                        [<ukernel_$MR x1_bb>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, MR, 1, f);
                    }
                    #(
                        else if n_left == nr_left {
                            [<ukernel_$MR x~nr_left _bb>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, MR, nr_left, f);
                        }
                    )*
                    m_iter -= 1;
                    a_cur = a_cur.add(MR*a_rs);
                    c_cur0 = c_cur0.add(MR*c_rs);
                }

                $(
                    if m_left > ($mr_left - VS) {
                        packa_panel_48(m_left, k, a_cur, a_rs, a_cs, ap_cur);
                        let mut n_iter = n_iter0;
                        let mut b_cur = b;
                        let mut c_cur1 = c_cur0;
                        while n_iter > 0 {
                            [<ukernel_$mr_left x $NR _bb_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, m_left, NR, f);
                            n_iter -= 1;
                            b_cur = b_cur.add(NR*k);
                            c_cur1 = c_cur1.add(NR*c_cs);
                        }
                        if n_left == 1 {
                            [<ukernel_$mr_left x1_bb_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, m_left, 1, f);
                        }
                        #(
                        else if n_left == nr_left {
                            [<ukernel_$mr_left x~nr_left _bb_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, ld_arr, m_left, nr_left, f);
                        }
                        )*
                        return;
                    }
                )*
            }        
        }});
    };
}

def_kernel_sb!(48, 8, 48, 32, 16);

pub(crate) unsafe fn kernel_bs<F: MyFn>(
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    b: *const TB, b_rs: usize, b_cs: usize,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap: *const TA,
    f: F,
) {  
    if c_rs == 1 {
        kernel_bs_v0::<_, false>(
            m, n, k,
            alpha, beta,
            b, b_rs, b_cs,
            c, c_rs, c_cs,
            ap,
            f
        );
    } else {
        kernel_bs_v0::<_, true>(
            m, n, k,
            alpha, beta,
            b, b_rs, b_cs,
            c, c_rs, c_cs,
            ap,
            f
        );
    }

}

pub(crate) unsafe fn kernel_sb<F: MyFn>(
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    a: *const TB, a_rs: usize, a_cs: usize,
    b: *const TB,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap_buf: *mut TA,
    f: F,
 ) { 
    if c_rs == 1 {
        kernel_sb_v0::<_, false>(
            m, n, k,
            alpha, beta,
            a, a_rs, a_cs,
            b,
            c, c_rs, c_cs,
            ap_buf,
            f
        );
    } else {
        kernel_sb_v0::<_, true>(
            m, n, k,
            alpha, beta,
            a, a_rs, a_cs,
            b,
            c, c_rs, c_cs,
            ap_buf,
            f
        );
    }
 } 

pub(crate) unsafe fn kernel<F: MyFn>(
   m: usize, n: usize, k: usize,
   alpha: *const TA, beta: *const TC,
   c: *mut TC,
   c_rs: usize, c_cs: usize,
   ap: *const TA, bp: *const TB,
   f: F,
) {
    if c_rs == 1 {
        kernel_bb::<_, false>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    } else {
        kernel_bb::<_, true>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
    }
}
