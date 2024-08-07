pub mod asm_ukernel;
// pub mod new_asm_ukernel;
pub(crate) mod axpy_kernel;
pub(crate) mod intrinsics_pack;

pub(crate) use asm_ukernel::*;
// pub(crate) use new_asm_ukernel::*;
pub(crate) use intrinsics_pack::{
    packa_panel_16,
    packb_panel_4,
};
pub(crate) use axpy_kernel::*;

use seq_macro::seq;
use paste::paste;

use crate::{TA,TB,TC};

const VS: usize = 8;

use crate::MyFn;

#[target_feature(enable = "avx,fma,avx2")]
pub unsafe fn axpy<F: MyFn>(
   m: usize, n: usize,
   alpha: *const f32,
   a: *const TA, a_rs: usize, a_cs: usize,
   x: *const TB, incx: usize,
   beta: *const f32,
   y: *mut TC, incy: usize,
   f: F,
) {
//    if a_cs == 1 && incx == 1 {
//        axpy_d(m, n, alpha, a, a_rs, x, beta, y, incy);
//        for i in 0..m {
//            f.call(y.add(i*incy), m);
//        }
//        return;
//    }
   if a_rs == 1 && incy == 1 {
    axpy_v(m, n, alpha, a, a_cs, x, incx, beta, y);
    // move this inside axpy_v, and benchmark
    f.call(y, m);
    return;
   }

   if a_cs == 1 {
       for i in 0..m {
           let a_cur = a.add(i*a_rs);
           let y_cur = y.add(i * incy);
           let mut acc = 0_i32;
           for j in 0..n {
               let x_cur = x.add(j * incx);
               acc += *a_cur.add(j) as i32 * *x_cur as i32;
           }
           acc = if *alpha == 1.0 {
                acc
            } else {
                (*alpha * acc as f32) as i32
            };
           if *beta == 0.0 {
                *y_cur = acc;
            } else if *beta == 1.0 {
                *y_cur = *y_cur + acc;
           } else {
                *y_cur = (*beta * *y_cur as f32 + acc as f32) as i32;
           }
           f.call(y_cur, 1);
       }
       return;
   }
   if a_rs == 1 {
       for i in 0..m {
           let y_cur = y.add(i*incy);
           let mut acc = 0_i32;
           for j in 0..n {
               let a_cur = a.add(j*a_cs);
               let x_cur = x.add(j*incx);
               acc += *a_cur.add(i) as i32* *x_cur as i32;
           }
           *y_cur = (*beta * *y_cur as f32+ *alpha * acc as f32) as i32;
            f.call(y_cur, 1);
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
            pub unsafe fn [<kernel_$MR x $NR>]<F: MyFn>(
                m: usize, n: usize, k: usize,
                alpha: *const f32,
                beta: *const f32,
                c: *mut TC, ldc: usize,
                ap: *const TA, bp: *const TB,
                f: F,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let mut m_iter = (m / MR) as u64;
                let m_left = m % MR;
                let mut ap_cur = ap;
                let mut c_cur0 = c;
                let k = (k+1) / 2 *2;
                
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
                let mask: [u32; 16] = [
                    u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ];
                let mask_offset = if m_left % VS == 0 { 0 } else { VS - (m_left %VS)};
                let mask_ptr = mask.as_ptr().add(mask_offset);
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

def_kernel_bb!(16, 4, 16, 8);


macro_rules! def_kernel_bb_strided {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        seq!( nr_left in 2..$NR { paste! {
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<kernel_$MR x $NR _strided>]<F: MyFn>(
                m: usize, n: usize, k: usize,
                alpha: *const f32,
                beta: *const f32,
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
                let ld_arr = [0, 0];
                let mut c_temp_buf = [0_i32; MR*NR];
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

def_kernel_bb_strided!(16, 4, 16, 8);

macro_rules! def_kernel_sb {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        seq!( nr_left in 2..$NR { paste! {
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<kernel_sb_v0>]<F: MyFn>(
                m: usize, n: usize, k: usize,
                alpha: *const f32,
                beta: *const f32,
                a: *const TB, a_rs: usize, a_cs: usize,
                b: *const TA,
                c: *mut TC, ldc: usize,
                ap_buf: *mut TA,
                f: F,
            ) {
                let k_eff = (k+1) / 2 *2;
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
                        [<ukernel_$MR x $NR _bb>](ap_cur, b_cur, c_cur1, alpha, beta, k_eff, ldc, ld_arr, f);
                        n_iter -= 1;
                        b_cur = b_cur.add(NR*k_eff);
                        c_cur1 = c_cur1.add(NR*ldc);
                    }
                    if n_left == 1 {
                        [<ukernel_$MR x1_bb>](ap_cur, b_cur, c_cur1, alpha, beta, k_eff, ldc, ld_arr, f);
                    }
                    #(
                        else if n_left == nr_left {
                            [<ukernel_$MR x~nr_left _bb>](ap_cur, b_cur, c_cur1, alpha, beta, k_eff, ldc, ld_arr, f);
                        }
                    )*
                    m_iter -= 1;
                    a_cur = a_cur.add(MR*a_rs);
                    c_cur0 = c_cur0.add(MR);
                }

                let mask: [u32; 16] = [
                    u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ];
                let mask_offset = if m_left % VS == 0 { 0 } else { VS - (m_left %VS)};
                let mask_ptr = mask.as_ptr().add(mask_offset);
                $(
                    if m_left > ($mr_left - VS) {
                        [<packa_panel_ $MR>](m_left, k, a_cur, a_rs, a_cs, ap_cur);
                        let mut n_iter = n_iter0;
                        let mut b_cur = b;
                        let mut c_cur1 = c_cur0;
                        while n_iter > 0 {
                            [<ukernel_$mr_left x $NR _bb_partial>](ap_cur, b_cur, c_cur1, alpha, beta, k_eff, ldc, ld_arr, mask_ptr, f);
                            n_iter -= 1;
                            b_cur = b_cur.add(NR*k_eff);
                            c_cur1 = c_cur1.add(NR*ldc);
                        }
                        if n_left == 1 {
                            [<ukernel_$mr_left x1_bb_partial>](ap_cur, b_cur, c_cur1, alpha, beta, k_eff, ldc, ld_arr, mask_ptr, f);
                        }
                        #(
                        else if n_left == nr_left {
                            [<ukernel_$mr_left x~nr_left _bb_partial>](ap_cur, b_cur, c_cur1, alpha, beta, k_eff, ldc, ld_arr, mask_ptr, f);
                        }
                        )*
                        return;
                    }
                )*
            }        
        }});
    };
}

def_kernel_sb!(16, 4, 16, 8);

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn kernel_sb<F: MyFn>(
    m: usize, n: usize, k: usize,
    alpha: *const f32,
    beta: *const f32,
    a: *const TB, a_rs: usize, a_cs: usize,
    b: *const TB,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap_buf: *mut TA,
    f: F,
 ) { 
    if c_rs == 1 {
        kernel_sb_v0(
            m, n, k,
            alpha, beta,
            a, a_rs, a_cs,
            b,
            c, c_cs,
            ap_buf,
            f
        );
        return;
    }
 } 

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn kernel<const MR: usize, const NR: usize, F: MyFn>(
   m: usize, n: usize, k: usize,
   alpha: *const f32, beta: *const f32,
   c: *mut TC,
   c_rs: usize, c_cs: usize,
   ap: *const TA, bp: *const TB,
   f: F,
) {
    if MR == 16 && NR == 4 {
        if c_rs == 1 {
            kernel_16x4(m, n, k, alpha, beta, c, c_cs, ap, bp, f)
        } else {
            kernel_16x4_strided(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
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
    if MR == 16 {
        packa_panel_16(m, k, a, a_rs, a_cs, ap);
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
    if NR == 4 {
        packb_panel_4(m, k, a, a_rs, a_cs, ap);
        return;
    }
    panic!("Packing for MR = {} not implemented", NR);
}
