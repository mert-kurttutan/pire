pub mod asm_ukernel;
// pub mod new_asm_ukernel;
// pub(crate) mod axpy_kernel;
pub(crate) mod intrinsics_pack;

pub(crate) use asm_ukernel::*;
// pub(crate) use new_asm_ukernel::*;
pub(crate) use intrinsics_pack::{
    packa_panel_48,
    packa_panel_32,
    packb_panel_12,
    packb_panel_8,
};
// pub(crate) use axpy_kernel::*;

use half::f16;

use seq_macro::seq;
use paste::paste;

use crate::{TA,TB,TC};

use crate::MyFn;

const VS: usize = 16;


pub unsafe fn load_c_strided<const MR: usize, const NR: usize>(
    c: *const f16, ct: *mut f16,
    m: usize,
    c_rs: usize, c_cs: usize,
) {
    for i in 0..NR {
        for j in 0..m {
            *ct.add(MR*i+j) = *c.add(i*c_cs + j*c_rs);
        }
    }
}

pub unsafe fn store_c_strided<const MR: usize, const NR: usize>(
    c: *mut f16, ct: *const f16,
    m: usize,
    c_rs: usize, c_cs: usize,
) {
    for i in 0..NR {
        for j in 0..m {
            *c.add(i*c_cs + j*c_rs) = *ct.add(MR*i+j);
        }
    }
}

macro_rules! def_kernel_bb {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        seq!( nr_left in 2..$NR { paste! {
            pub unsafe fn [<kernel_$MR x $NR>]<F: MyFn>(
                m: usize, n: usize, k: usize,
                alpha: *const f32,
                beta: *const f32,
                c: *mut f16, ldc: usize,
                ap: *const f32, bp: *const f32,
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
                let ld_arr = [0, 0];
                
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut bp_cur = bp;
                    let mut c_cur1 = c_cur0;
                    while n_iter > 0 {
                        [<ukernel_$MR x $NR _bb>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr, f);
                        n_iter -= 1;
                        bp_cur = bp_cur.add(NR*k);
                        c_cur1 = c_cur1.add(NR*ldc);
                    }
                    if n_left == 1 {
                        [<ukernel_$MR x1_bb>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr, f);
                    }
                    #(
                        else if n_left == nr_left {
                            [<ukernel_$MR x~nr_left _bb>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr, f);
                        }
                    )*
                    m_iter -= 1;
                    ap_cur = ap_cur.add(MR*k);
                    c_cur0 = c_cur0.add(MR);
                }
                let x = if m_left % VS == 0 && m_left > 0 { 0xFFFF } else { (1_u16 << (m_left % VS)) - 1 };
                let mask_ptr = (&x) as *const u16;
                $(
                    if m_left > ($mr_left - VS) {
                        let mut n_iter = n_iter0;
                        let mut bp_cur = bp;
                        let mut c_cur1 = c_cur0;
                        while n_iter > 0 {
                            [<ukernel_$mr_left x $NR _bb_partial>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr, f);
                            n_iter -= 1;
                            bp_cur = bp_cur.add(NR*k);
                            c_cur1 = c_cur1.add(NR*ldc);
                        }
                        if n_left == 1 {
                            [<ukernel_$mr_left x1_bb_partial>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr, f);
                        }
                        #(
                            else if n_left == nr_left {
                                [<ukernel_$mr_left x~nr_left _bb_partial>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr, f);
                            }
                        )*
                        return;
                    }
                )*
            }        
        }});
    };
}

def_kernel_bb!(48, 8, 48, 32, 16);
def_kernel_bb!(32, 12, 32, 16);


macro_rules! def_kernel_bb_strided {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        seq!( nr_left in 2..$NR { paste! {
            pub unsafe fn [<kernel_$MR x $NR _strided>]<F: MyFn>(
                m: usize, n: usize, k: usize,
                alpha: *const f32,
                beta: *const f32,
                c: *mut f16, c_rs: usize, c_cs: usize,
                ap: *const f32, bp: *const f32,
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
                let ld_arr = [0, 0];
                let mut c_temp_buf = [f16::ZERO; MR*NR];
                let ct = c_temp_buf.as_mut_ptr();
                
                
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut bp_cur = bp;
                    let mut c_cur1 = c_cur0;
                    while n_iter > 0 {
                        load_c_strided::<MR,NR>(c_cur1, ct, MR, c_rs, c_cs);
                        [<ukernel_$MR x $NR _bb>](ap_cur, bp_cur, ct, alpha, beta, k, MR, ld_arr, f);
                        store_c_strided::<MR,NR>(c_cur1, ct, MR, c_rs, c_cs);
                        n_iter -= 1;
                        bp_cur = bp_cur.add(NR*k);
                        c_cur1 = c_cur1.add(NR*c_cs);
                    }
                    if n_left == 1 {
                        load_c_strided::<MR,1>(c_cur1, ct, MR, c_rs, c_cs);
                        [<ukernel_$MR x1_bb>](ap_cur, bp_cur, c_cur1, alpha, beta, k, MR, ld_arr, f);
                        store_c_strided::<MR,1>(c_cur1, ct, MR, c_rs, c_cs);
                    }
                    #(
                        else if n_left == nr_left {
                            load_c_strided::<MR,nr_left>(c_cur1, ct, MR, c_rs, c_cs);
                            [<ukernel_$MR x~nr_left _bb>](ap_cur, bp_cur, c_cur1, alpha, beta, k, MR, ld_arr, f);
                            store_c_strided::<MR,nr_left>(c_cur1, ct, MR, c_rs, c_cs);
                        }
                    )*
                    m_iter -= 1;
                    ap_cur = ap_cur.add(MR*k);
                    c_cur0 = c_cur0.add(MR*c_rs);
                }


                $(
                    if m_left > ($mr_left - VS) {
                        let mut n_iter = n_iter0;
                        let mut bp_cur = bp;
                        let mut c_cur1 = c_cur0;
                        const MR_LEFT: usize = $mr_left;
                        while n_iter > 0 {
                            load_c_strided::<MR_LEFT,NR>(c_cur1, ct, m_left, c_rs, c_cs);
                            [<ukernel_$mr_left x $NR _bb>](ap_cur, bp_cur, ct, alpha, beta, k, MR_LEFT, ld_arr, f);
                            store_c_strided::<MR_LEFT,NR>(c_cur1, ct, m_left, c_rs, c_cs);
                            n_iter -= 1;
                            bp_cur = bp_cur.add(NR*k);
                            c_cur1 = c_cur1.add(NR*c_cs);
                        }
                        if n_left == 1 {
                            load_c_strided::<MR_LEFT,1>(c_cur1, ct, m_left, c_rs, c_cs);
                            [<ukernel_$mr_left x1_bb>](ap_cur, bp_cur, ct, alpha, beta, k, MR_LEFT, ld_arr, f);
                            store_c_strided::<MR_LEFT,1>(c_cur1, ct, m_left, c_rs, c_cs);
                        }
                        #(
                        else if n_left == nr_left {
                            load_c_strided::<MR_LEFT,nr_left>(c_cur1, ct, m_left, c_rs, c_cs);
                            [<ukernel_$mr_left x~nr_left _bb>](ap_cur, bp_cur, ct, alpha, beta, k, MR_LEFT, ld_arr, f);
                            store_c_strided::<MR_LEFT,nr_left>(c_cur1, ct, m_left, c_rs, c_cs);
                        }
                        )*
                        return;
                    }
                )*
            }        
        }});
    };
}

def_kernel_bb_strided!(48, 8, 48, 32, 16);
def_kernel_bb_strided!(32, 12, 32, 16);

pub(crate) unsafe fn kernel<const MR: usize, const NR: usize, F: MyFn>(
   m: usize, n: usize, k: usize,
   alpha: *const f32, beta: *const f32,
   c: *mut f16,
   c_rs: usize, c_cs: usize,
   ap: *const f32, bp: *const f32,
   f: F,
) {
   if MR == 48 && NR == 8 {
        if c_rs == 1 {
            kernel_48x8(m, n, k, alpha, beta, c, c_cs, ap, bp, f)
        } else {
            kernel_48x8_strided(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
        }
        return;
   }
   if MR == 32 && NR == 12 {
        if c_rs == 1 {
            kernel_32x12(m, n, k, alpha, beta, c, c_cs, ap, bp, f)
        } else {
            kernel_32x12_strided(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
        }
        return;
    }
    panic!("Kernel for MR = {} and NR = {} not implemented", MR, NR);
}


pub(crate) unsafe fn packa_panel<const MR: usize>(
    m: usize, k: usize,
    a: *const f16, a_rs: usize, a_cs: usize,
    ap: *mut f32,
){
    if MR == 48 {
        packa_panel_48(m, k, a, a_rs, a_cs, ap);
        return;
    }
    if MR == 32 {
        packa_panel_32(m, k, a, a_rs, a_cs, ap);
        return;
    }
    panic!("Packing for MR = {} not implemented", MR);
}


pub(crate) unsafe fn packb_panel<const NR: usize>(
    m: usize, k: usize,
    a: *const f16, a_rs: usize, a_cs: usize,
    ap: *mut f32,
){
    if NR == 8 {
        packb_panel_8(m, k, a, a_rs, a_cs, ap);
        return;
    }
    if NR == 12 {
        packb_panel_12(m, k, a, a_rs, a_cs, ap);
        return;
    }
    panic!("Packing for MR = {} not implemented", NR);
}
