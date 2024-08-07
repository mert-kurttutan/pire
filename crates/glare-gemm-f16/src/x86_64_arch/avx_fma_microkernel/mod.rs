pub mod asm_ukernel;
// pub mod new_asm_ukernel;
pub(crate) mod axpy_kernel;
pub(crate) mod intrinsics_pack;

pub(crate) use asm_ukernel::*;
// pub(crate) use new_asm_ukernel::*;
pub(crate) use intrinsics_pack::{
    packa_panel_24,
    packa_panel_16,
    packb_panel_6,
    packb_panel_4,
};
pub(crate) use axpy_kernel::*;

use seq_macro::seq;
use paste::paste;


const VS: usize = 8;

use crate::MyFn;

use half::f16;

#[target_feature(enable = "avx,fma")]
pub unsafe fn axpy<F: MyFn>(
   m: usize, n: usize,
   alpha: *const f32,
   a: *const f16, a_rs: usize, a_cs: usize,
   x: *const f16, incx: usize,
   beta: *const f32,
   y: *mut f16, incy: usize,
   f: F,
) {
   if a_cs == 1 && incx == 1 {
       axpy_d(m, n, alpha, a, a_rs, x, beta, y, incy);
       for i in 0..m {
           f.call(y.add(i*incy), m);
       }
       return;
   }
   if a_rs == 1 && incy == 1 {
    axpy_v(m, n, alpha, a, a_cs, x, incx, beta, y);
    // move this inside axpy_v, and benchmark
    f.call(y, m);
    return;
   }
   let beta_f16 = f16::from_f32(*beta);
    let alpha_f16 = f16::from_f32(*alpha);
   if a_cs == 1 {
       for i in 0..m {
           let a_cur = a.add(i*a_rs);
           let y_cur = y.add(i * incy);
           let mut acc = f16::ZERO;
           for j in 0..n {
               let x_cur = x.add(j * incx);
               acc += *a_cur.add(j) * *x_cur;
           }
           *y_cur = beta_f16 * *y_cur + alpha_f16 * acc;
           f.call(y_cur, 1);
       }
       return;
   }
   if a_rs == 1 {
       for i in 0..m {
           let y_cur = y.add(i*incy);
           let mut acc = f16::ZERO;
           for j in 0..n {
               let a_cur = a.add(j*a_cs);
               let x_cur = x.add(j*incx);
               acc += *a_cur.add(i) * *x_cur;
           }
           *y_cur = beta_f16 * *y_cur + alpha_f16 * acc;
            f.call(y_cur, 1);
       }
       return;
   }
}

pub unsafe fn load_c_contiguous<const MR: usize, const NR: usize>(
    c: *const f16, ct: *mut f16,
    m: usize,
    ldc: usize,
) {
    for i in 0..NR {
        std::ptr::copy_nonoverlapping(c.add(i*ldc), ct.add(MR*i), m);
    }
}

pub unsafe fn store_c_contiguous<const MR: usize, const NR: usize>(
    c: *mut f16, ct: *const f16,
    m: usize,
    ldc: usize,
) {
    for i in 0..NR {
        std::ptr::copy_nonoverlapping(ct.add(MR*i), c.add(i*ldc), m);
    }
}

#[target_feature(enable = "avx")]
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

#[target_feature(enable = "avx")]
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
            #[target_feature(enable = "avx,fma")]
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
                let mut c_temp_buf = [f16::ZERO; MR*NR];
                let ct = c_temp_buf.as_mut_ptr();

                $(
                    if m_left > ($mr_left - VS) {
                        let mut n_iter = n_iter0;
                        let mut bp_cur = bp;
                        let mut c_cur1 = c_cur0;
                        while n_iter > 0 {
                            load_c_contiguous::<$mr_left, NR>(c_cur1, ct, m_left, ldc);
                            [<ukernel_$mr_left x $NR _bb>](ap_cur, bp_cur, ct, alpha, beta, k, $mr_left, ld_arr, f);
                            store_c_contiguous::<$mr_left, NR>(c_cur1, ct, m_left, ldc);
                            n_iter -= 1;
                            bp_cur = bp_cur.add(NR*k);
                            c_cur1 = c_cur1.add(NR*ldc);
                        }
                        if n_left == 1 {
                            load_c_contiguous::<$mr_left, 1>(c_cur1, ct, m_left, ldc);
                            [<ukernel_$mr_left x1_bb>](ap_cur, bp_cur, ct, alpha, beta, k, $mr_left, ld_arr, f);
                            store_c_contiguous::<$mr_left, 1>(c_cur1, ct, m_left, ldc);
                        }
                        #(
                            else if n_left == nr_left {
                                load_c_contiguous::<$mr_left, nr_left>(c_cur1, ct, m_left, ldc);
                                [<ukernel_$mr_left x~nr_left _bb>](ap_cur, bp_cur, ct, alpha, beta, k, $mr_left, ld_arr, f);
                                store_c_contiguous::<$mr_left, nr_left>(c_cur1, ct, m_left, ldc);
                            }
                        )*
                        return;
                    }
                )*
            }        
        }});
    };
}

def_kernel_bb!(24, 4, 24, 16, 8);
def_kernel_bb!(16, 6, 16, 8);

macro_rules! def_kernel_bb_strided {
    ($MR:tt, $NR:tt, $($mr_left:tt),*) => {
        seq!( nr_left in 2..$NR { paste! {
            #[target_feature(enable = "avx,fma")]
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

def_kernel_bb_strided!(24, 4, 24, 16, 8);
def_kernel_bb_strided!(16, 6, 16, 8);


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn kernel<const MR: usize, const NR: usize, F: MyFn>(
   m: usize, n: usize, k: usize,
   alpha: *const f32, beta: *const f32,
   c: *mut f16,
   c_rs: usize, c_cs: usize,
   ap: *const f32, bp: *const f32,
   f: F,
) {
   if MR == 24 && NR == 4 {
        if c_rs == 1 {
            kernel_24x4(m, n, k, alpha, beta, c, c_cs, ap, bp, f)
        } else {
            kernel_24x4_strided(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
        }
        return;
   }
   if MR == 16 && NR == 6 {
        if c_rs == 1 {
            kernel_16x6(m, n, k, alpha, beta, c, c_cs, ap, bp, f)
        } else {
            kernel_16x6_strided(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
        }
        return;
    }
    panic!("Kernel for MR = {} and NR = {} not implemented", MR, NR);
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packa_panel<const MR: usize>(
    m: usize, k: usize,
    a: *const f16, a_rs: usize, a_cs: usize,
    ap: *mut f32,
){
    if MR == 24 {
        packa_panel_24(m, k, a, a_rs, a_cs, ap);
        return;
    }
    if MR == 16 {
        packa_panel_16(m, k, a, a_rs, a_cs, ap);
        return;
    }
    panic!("Packing for MR = {} not implemented", MR);
}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_panel<const NR: usize>(
    m: usize, k: usize,
    a: *const f16, a_rs: usize, a_cs: usize,
    ap: *mut f32,
){
    if NR == 4 {
        packb_panel_4(m, k, a, a_rs, a_cs, ap);
        return;
    }
    if NR == 6 {
        packb_panel_6(m, k, a, a_rs, a_cs, ap);
        return;
    }
    panic!("Packing for MR = {} not implemented", NR);
}
