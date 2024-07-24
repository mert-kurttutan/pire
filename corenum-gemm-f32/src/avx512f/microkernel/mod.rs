pub mod asm_ukernel;
// pub mod new_asm_ukernel;
pub(crate) mod axpy_kernel;
pub(crate) mod intrinsics_pack;

pub(crate) use asm_ukernel::*;
// pub(crate) use new_asm_ukernel::*;
pub(crate) use intrinsics_pack::{
    packa_panel_48,
    packa_panel_32,
    packb_panel_12,
    packb_panel_8,
};
pub(crate) use axpy_kernel::*;

use seq_macro::seq;
use paste::paste;

use crate::{TA,TB,TC};

const VS: usize = 16;

#[target_feature(enable = "avx,fma")]
pub unsafe fn axpy(
   m: usize, n: usize,
   alpha: *const TA,
   a: *const TA, a_rs: usize, a_cs: usize,
   x: *const TB, incx: usize,
   beta: *const TC,
   y: *mut TC, incy: usize,
) {
   if a_cs == 1 && incx == 1 {
       axpy_d(m, n, alpha, a, a_rs, x, beta, y, incy);
       return;
   }
   if a_rs == 1 && incy == 1 {
       axpy_v(m, n, alpha, a, a_cs, x, incx, beta, y);
       return;
   }
   if a_cs == 1 {
       for i in 0..m {
           let a_cur = a.add(i*a_rs);
           let y_cur = y.add(i * incy);
           let mut acc = 0.0;
           for j in 0..n {
               let x_cur = x.add(j * incx);
               acc += *a_cur.add(j) * *x_cur;
           }
           *y_cur = *beta * *y_cur + *alpha * acc;
       }
       return;
   }
   if a_rs == 1 || true {
       for i in 0..m {
           let y_cur = y.add(i*incy);
           let mut acc = 0.0;
           for j in 0..n {
               let a_cur = a.add(j*a_cs);
               let x_cur = x.add(j*incx);
               acc += *a_cur.add(i) * *x_cur;
           }
           *y_cur = *beta * *y_cur + *alpha * acc;
       }
       return;
   }
}

#[target_feature(enable = "avx,fma")]
pub unsafe fn load_c_strided<const MR: usize, const NR: usize>(
    c: *const TC, ct: *mut TC,
    m: usize,
    c_rs: usize, c_cs: usize,
) {
    for i in 0..NR {
        for j in 0..m {
            *ct.add(MR*i+j) = *c.add(i*c_cs + j*c_rs);
        }
    }
}

#[target_feature(enable = "avx,fma")]
pub unsafe fn store_c_strided<const MR: usize, const NR: usize>(
    c: *mut TC, ct: *const TC,
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
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<kernel_$MR x $NR>](
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                beta: *const TC,
                c: *mut TC, ldc: usize,
                ap: *const TA, bp: *const TB,
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
                        [<ukernel_$MR x $NR _bb>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
                        n_iter -= 1;
                        bp_cur = bp_cur.add(NR*k);
                        c_cur1 = c_cur1.add(NR*ldc);
                    }
                    if n_left == 1 {
                        [<ukernel_$MR x1_bb>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
                    }
                    #(
                        else if n_left == nr_left {
                            [<ukernel_$MR x~nr_left _bb>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
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
                            [<ukernel_$mr_left x $NR _bb_partial>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr);
                            n_iter -= 1;
                            bp_cur = bp_cur.add(NR*k);
                            c_cur1 = c_cur1.add(NR*ldc);
                        }
                        if n_left == 1 {
                            [<ukernel_$mr_left x1_bb_partial>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr);
                        }
                        #(
                            else if n_left == nr_left {
                                [<ukernel_$mr_left x~nr_left _bb_partial>](ap_cur, bp_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr);
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
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<kernel_$MR x $NR _strided>](
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                beta: *const TC,
                c: *mut TC, c_rs: usize, c_cs: usize,
                ap: *const TA, bp: *const TB,
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
                let mut c_temp_buf = [0_f32; MR*NR];
                let ct = c_temp_buf.as_mut_ptr();
                
                
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut bp_cur = bp;
                    let mut c_cur1 = c_cur0;
                    while n_iter > 0 {
                        load_c_strided::<MR,NR>(c_cur1, ct, MR, c_rs, c_cs);
                        [<ukernel_$MR x $NR _bb>](ap_cur, bp_cur, ct, alpha, beta, k, MR, ld_arr);
                        store_c_strided::<MR,NR>(c_cur1, ct, MR, c_rs, c_cs);
                        n_iter -= 1;
                        bp_cur = bp_cur.add(NR*k);
                        c_cur1 = c_cur1.add(NR*c_cs);
                    }
                    if n_left == 1 {
                        load_c_strided::<MR,1>(c_cur1, ct, MR, c_rs, c_cs);
                        [<ukernel_$MR x1_bb>](ap_cur, bp_cur, c_cur1, alpha, beta, k, MR, ld_arr);
                        store_c_strided::<MR,1>(c_cur1, ct, MR, c_rs, c_cs);
                    }
                    #(
                        else if n_left == nr_left {
                            load_c_strided::<MR,nr_left>(c_cur1, ct, MR, c_rs, c_cs);
                            [<ukernel_$MR x~nr_left _bb>](ap_cur, bp_cur, c_cur1, alpha, beta, k, MR, ld_arr);
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
                            [<ukernel_$mr_left x $NR _bb>](ap_cur, bp_cur, ct, alpha, beta, k, MR_LEFT, ld_arr);
                            store_c_strided::<MR_LEFT,NR>(c_cur1, ct, m_left, c_rs, c_cs);
                            n_iter -= 1;
                            bp_cur = bp_cur.add(NR*k);
                            c_cur1 = c_cur1.add(NR*c_cs);
                        }
                        if n_left == 1 {
                            load_c_strided::<MR_LEFT,1>(c_cur1, ct, m_left, c_rs, c_cs);
                            [<ukernel_$mr_left x1_bb>](ap_cur, bp_cur, ct, alpha, beta, k, MR_LEFT, ld_arr);
                            store_c_strided::<MR_LEFT,1>(c_cur1, ct, m_left, c_rs, c_cs);
                        }
                        #(
                        else if n_left == nr_left {
                            load_c_strided::<MR_LEFT,nr_left>(c_cur1, ct, m_left, c_rs, c_cs);
                            [<ukernel_$mr_left x~nr_left _bb>](ap_cur, bp_cur, ct, alpha, beta, k, MR_LEFT, ld_arr);
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

macro_rules! def_kernel_bs {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        seq!( nr_left in 2..$NR { paste! {
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<kernel_bs _v0>](
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                beta: *const TC,
                b: *const TB, b_rs: usize, b_cs: usize,
                c: *mut TC, ldc: usize,
                ap_cur: *const TA,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let mut m_iter = (m / MR) as u64;
                let m_left = m % MR;
                let mut c_cur0 = c;
                let mut ap_cur = ap_cur;
                
                let n_iter0 = (n / NR) as u64;
                let n_left = (n % NR) as u64;
                let ld_arr = [b_rs*4, b_cs*4];
                // use blocking since rrc kernel is hard to implement to current macro choices
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut b_cur = b;
                    let mut c_cur1 = c_cur0;
                    while n_iter > 0 {
                        [<ukernel_$MR x $NR _bs>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
                        n_iter -= 1;
                        b_cur = b_cur.add(NR*b_cs);
                        c_cur1 = c_cur1.add(NR*ldc);
                    }
                    if n_left == 1 {
                        [<ukernel_$MR x1_bs>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
                    }
                    #(
                        else if n_left == nr_left {
                            [<ukernel_$MR x~nr_left _bs>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
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
                        let mut b_cur = b;
                        let mut c_cur1 = c_cur0;
                        while n_iter > 0 {
                            [<ukernel_$mr_left x $NR _bs_partial>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr);
                            n_iter -= 1;
                            b_cur = b_cur.add(NR*b_cs);
                            c_cur1 = c_cur1.add(NR*ldc);
                        }
                        if n_left == 1 {
                            [<ukernel_$mr_left x1_bs_partial>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr);
                        }
                        #(
                        else if n_left == nr_left {
                            [<ukernel_$mr_left x~nr_left _bs_partial>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr);
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

pub(crate) unsafe fn kernel_bs(
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    b: *const TB, b_rs: usize, b_cs: usize,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap: *const TA,
) {  
    if c_rs == 1 {
        kernel_bs_v0(
            m, n, k,
            alpha, beta,
            b, b_rs, b_cs,
            c, c_cs,
            ap
        );
        return;
    }

}

macro_rules! def_kernel_sb {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        seq!( nr_left in 2..$NR { paste! {
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<kernel_sb_v0>](
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                beta: *const TC,
                a: *const TB, a_rs: usize, a_cs: usize,
                b: *const TA,
                c: *mut TC, ldc: usize,
                ap_buf: *mut TA,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let mut m_iter = (m / MR) as u64;
                let m_left = m % MR;
                let mut c_cur0 = c;
                let mut a_cur = a;
                
                let n_iter0 = (n / NR) as u64;
                let n_left = (n % NR) as u64;
                let ld_arr = [0*4, 0*4];
                let ap_cur = ap_buf;
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut b_cur = b;
                    let mut c_cur1 = c_cur0;
                    [<packa_panel_ $MR>](MR, k, a_cur, a_rs, a_cs, ap_cur);
                    while n_iter > 0 {
                        [<ukernel_$MR x $NR _bb>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
                        n_iter -= 1;
                        b_cur = b_cur.add(NR*k);
                        c_cur1 = c_cur1.add(NR*ldc);
                    }
                    if n_left == 1 {
                        [<ukernel_$MR x1_bb>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
                    }
                    #(
                        else if n_left == nr_left {
                            [<ukernel_$MR x~nr_left _bb>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
                        }
                    )*
                    m_iter -= 1;
                    a_cur = a_cur.add(MR*a_rs);
                    c_cur0 = c_cur0.add(MR);
                }

                let x = if m_left % VS == 0 && m_left > 0 { 0xFFFF } else { (1_u16 << (m_left % VS)) - 1 };
                let mask_ptr = (&x) as *const u16;
                $(
                    if m_left > ($mr_left - VS) {
                        [<packa_panel_ $MR>](m_left, k, a_cur, a_rs, a_cs, ap_cur);
                        let mut n_iter = n_iter0;
                        let mut b_cur = b;
                        let mut c_cur1 = c_cur0;
                        while n_iter > 0 {
                            [<ukernel_$mr_left x $NR _bb_partial>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr);
                            n_iter -= 1;
                            b_cur = b_cur.add(NR*k);
                            c_cur1 = c_cur1.add(NR*ldc);
                        }
                        if n_left == 1 {
                            [<ukernel_$mr_left x1_bb_partial>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr);
                        }
                        #(
                        else if n_left == nr_left {
                            [<ukernel_$mr_left x~nr_left _bb_partial>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr, mask_ptr);
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

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn kernel_sb(
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    a: *const TB, a_rs: usize, a_cs: usize,
    b: *const TB,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap_buf: *mut TA,
 ) { 
    if c_rs == 1 {
        kernel_sb_v0(
            m, n, k,
            alpha, beta,
            a, a_rs, a_cs,
            b,
            c, c_cs,
            ap_buf
        );
        return;
    }
 } 


// Make sure compiler optimizes the const conditional below
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn kernel<const MR: usize, const NR: usize>(
   m: usize, n: usize, k: usize,
   alpha: *const TA, beta: *const TC,
   c: *mut TC,
   c_rs: usize, c_cs: usize,
   ap: *const TA, bp: *const TB,
) {
   if MR == 48 && NR == 8 {
        if c_rs == 1 {
            kernel_48x8(m, n, k, alpha, beta, c, c_cs, ap, bp)
        } else {
            kernel_48x8_strided(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp)
        }
        return;
   }
   if MR == 32 && NR == 12 {
        if c_rs == 1 {
            kernel_32x12(m, n, k, alpha, beta, c, c_cs, ap, bp)
        } else {
            kernel_32x12_strided(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp)
        }
        return;
    }
    panic!("Kernel for MR = {} and NR = {} not implemented", MR, NR);
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packa_panel<const MR: usize>(
    m: usize, k: usize,
    a: *const TB, a_rs: usize, a_cs: usize,
    ap: *mut TB,
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


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_panel<const NR: usize>(
    m: usize, k: usize,
    a: *const TB, a_rs: usize, a_cs: usize,
    ap: *mut TB,
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
