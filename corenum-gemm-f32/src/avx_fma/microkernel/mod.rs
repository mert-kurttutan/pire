pub mod asm_ukernel;
// pub mod new_asm_ukernel;
pub(crate) mod axpy_kernel;
pub(crate) mod intrinsics_pack;
use seq_macro::seq;
use paste::paste;




pub(crate) use asm_ukernel::*;
// pub(crate) use new_asm_ukernel::*;
pub(crate) use intrinsics_pack::{
    pack_panel_24,
    pack_panel_16,
    pack_panel_6,
    pack_panel_4,
};
pub(crate) use axpy_kernel::*;



use crate::{TA,TB,TC};



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
pub unsafe fn load_c<const MR: usize, const NR: usize>(
    c: *const TC, ct: *mut TC,
    m: usize,
    ldc: usize,
) {
    for i in 0..NR {
        ct.add(MR*i).copy_from_nonoverlapping(c.add(i*ldc), m);
    }
}

#[target_feature(enable = "avx,fma")]
pub unsafe fn store_c<const MR: usize, const NR: usize>(
    c: *mut TC, ct: *const TC,
    m: usize,
    ldc: usize,
) {
    for i in 0..NR {
        c.add(i*ldc).copy_from_nonoverlapping(ct.add(i*MR), m);
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


// Maybe use macro_rules to make code more uniform ??
// At the moment it does not seem to be problematic, as the codebase scales, it might
// become cumbersome to manage it manually



macro_rules! def_milikernel {
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

                let mask: [u32; 16] = [
                    u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ];
                let mask_offset = if m_left % 8 == 0 { 0 } else { 8 - (m_left %8)};
                let mask_ptr = mask.as_ptr().add(mask_offset);
                $(
                    if m_left > ($mr_left - 8) {
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
def_milikernel!(24, 4, 24, 16, 8);
def_milikernel!(16, 6, 16, 8);



macro_rules! def_milikernel_strided {
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
                    if m_left > ($mr_left - 8) {
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

def_milikernel_strided!(24, 4, 24, 16, 8);
def_milikernel_strided!(16, 6, 16, 8);


// TODO: Check if implementing br_partial is worth the performance gain/code complexity tradeoff
macro_rules! def_milikernel_blocked {
    ($MR:tt, $NR:tt, $is_b_row:tt, $layout1:tt, $layout2:tt, $($mr_left:tt),*) => {
        seq!( nr_left in 2..$NR { paste! {
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<kernel_sup_m_$layout1>](
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                beta: *const TC,
                b: *const TB, ldb: usize,
                c: *mut TC, ldc: usize,
                ap_cur: *const TA,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let mut m_iter = (m / MR) as u64;
                let m_left = m % MR;
                let mut c_cur0 = c;
                let b_cs = if $is_b_row {1} else {ldb};
                let mut ap_cur = ap_cur;
                
                let n_iter0 = (n / NR) as u64;
                let n_left = (n % NR) as u64;
                let ld_arr = [0, ldb*4];
                // use blocking since rrc kernel is hard to implement to current macro choices
                while m_iter > 0 {
                    let mut n_iter = n_iter0;
                    let mut b_cur = b;
                    let mut c_cur1 = c_cur0;
                    while n_iter > 0 {
                        [<ukernel_$MR x $NR _$layout2>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
                        n_iter -= 1;
                        b_cur = b_cur.add(NR*b_cs);
                        c_cur1 = c_cur1.add(NR*ldc);
                    }
                    if n_left == 1 {
                        [<ukernel_$MR x1_$layout2>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
                    }
                    #(
                        else if n_left == nr_left {
                            [<ukernel_$MR x~nr_left _$layout2>](ap_cur, b_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
                        }
                    )*
                    m_iter -= 1;
                    ap_cur = ap_cur.add(MR*k);
                    c_cur0 = c_cur0.add(MR);
                }


                $(
                    if m_left > ($mr_left - 8) {
                        let mut n_iter = n_iter0;
                        let mut b_cur = b;
                        let mut c_cur1 = c_cur0;
                        const MR_LEFT: usize = $mr_left;
                        let mut c_temp_buf = [0_f32; MR_LEFT*NR];
                        let ct = c_temp_buf.as_mut_ptr();
                        while n_iter > 0 {
                            load_c::<MR_LEFT,NR>(c_cur1, ct, m_left, ldc);
                            [<ukernel_$mr_left x $NR _$layout2>](ap_cur, b_cur, ct, alpha, beta, k, MR_LEFT, ld_arr);
                            store_c::<MR_LEFT,NR>(c_cur1, ct, m_left, ldc);
                            n_iter -= 1;
                            b_cur = b_cur.add(NR*b_cs);
                            c_cur1 = c_cur1.add(NR*ldc);
                        }
                        if n_left == 1 {
                            load_c::<MR_LEFT,1>(c_cur1, ct, m_left, ldc);
                            [<ukernel_$mr_left x1_$layout2>](ap_cur, b_cur, ct, alpha, beta, k, MR_LEFT, ld_arr);
                            store_c::<MR_LEFT,1>(c_cur1, ct, m_left, ldc);
                        }
                        #(
                        else if n_left == nr_left {
                            load_c::<MR_LEFT,nr_left>(c_cur1, ct, m_left, ldc);
                            [<ukernel_$mr_left x~nr_left _$layout2>](ap_cur, b_cur, ct, alpha, beta, k, MR_LEFT, ld_arr);
                            store_c::<MR_LEFT,nr_left>(c_cur1, ct, m_left, ldc);
                        }
                        )*
                        return;
                    }
                )*
            }        
        }});
    };
}
def_milikernel_blocked!(24, 4, true, r, br, 24, 16, 8);
def_milikernel_blocked!(24, 4, false, c, bc, 24, 16, 8);

// def_milikernel_blocked!(16, 6, true, rc, br, 16, 8);

// def_milikernel_blocked!(16, 6, false, cc, bc, 16, 8);

macro_rules! def_milikernel_blocked2 {
    ($MR:tt, $NR:tt, $is_b_row:tt, $layout1:tt, $layout2:tt, $($nr_left:tt),*) => {
        seq!( mr_left in 2..$MR { paste! {
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<kernel_sup_n_$layout1>](
                m: usize, n: usize, k: usize,
                alpha: *const TA,
                beta: *const TC,
                a: *const TB, lda: usize,
                c: *mut TC, ldc: usize,
                bp: *const TA,
            ) {
                const MR: usize = $MR;
                const NR: usize = $NR;
                let m_iter0 = (m / MR) as u64;
                let m_left = m % MR;
                let mut c_cur0 = c;
                let a_rs = if $is_b_row { lda } else { 1 };
                let mut bp_cur = bp;
                let mut n_iter = (n / NR) as u64;
                let n_left = (n % NR) as u64;
                let ld_arr = [0, lda * 4];
                // use blocking since rrc kernel is hard to implement to current macro choices
                while n_iter > 0 {
                    let mut m_iter = m_iter0;
                    let mut a_cur = a;
                    let mut c_cur1 = c_cur0;
                    while m_iter > 0 {
                        [<ukernel_$NR x $MR _$layout2>](bp_cur, a_cur, c_cur1, alpha, beta, k, ldc, ld_arr);
                        m_iter -= 1;
                        a_cur = a_cur.add(MR * a_rs);
                        c_cur1 = c_cur1.add(MR);
                    }
                    if m_left == 1 {
                        [<ukernel_$NR x1_$layout2 _partial>](bp_cur, a_cur, c_cur1, alpha, beta, k, ldc, ld_arr, 8);
                    }
                    #(
                        else if m_left == mr_left {
                            [<ukernel_$NR x~mr_left _$layout2 _partial>](bp_cur, a_cur, c_cur1, alpha, beta, k, ldc, ld_arr, 8);
                        }
                    )*
                    n_iter -= 1;
                    bp_cur = bp_cur.add(NR*k);
                    c_cur0 = c_cur0.add(NR*ldc);
                }


                $(
                    if n_left > ($nr_left - 8) {
                        let mut m_iter = m_iter0;
                        let mut a_cur = a;
                        let mut c_cur1 = c_cur0;
                        const MR_LEFT: usize = $nr_left;
                        let mut c_temp_buf = [0_f32; MR_LEFT*NR];
                        let ct = c_temp_buf.as_mut_ptr();
                        let n_left = if n_left % 8 == 0 { 8 } else { n_left % 8 };
                        while m_iter > 0 {
                            // load_c::<MR_LEFT,NR>(c_cur1, ct, m_left, ldc);
                            [<ukernel_$nr_left x $MR _$layout2 _partial>](bp_cur, a_cur, c_cur1, alpha, beta, k, ldc, ld_arr, n_left as usize);
                            // store_c::<MR_LEFT,NR>(c_cur1, ct, m_left, ldc);
                            m_iter -= 1;
                            a_cur = a_cur.add(MR*a_rs);
                            c_cur1 = c_cur1.add(MR);
                        }
                        if m_left == 1 {
                            // load_c::<MR_LEFT,1>(c_cur1, ct, m_left, ldc);
                            [<ukernel_$nr_left x1_$layout2 _partial>](bp_cur, a_cur, c_cur1, alpha, beta, k, ldc, ld_arr, n_left as usize);
                            // store_c::<MR_LEFT,1>(c_cur1, ct, m_left, ldc);
                        }
                        #(
                        else if m_left == mr_left {
                            // load_c::<MR_LEFT,mr_left>(c_cur1, ct, m_left, ldc);
                            [<ukernel_$nr_left x~mr_left _$layout2 _partial>](bp_cur, a_cur, c_cur1, alpha, beta, k, ldc, ld_arr, n_left as usize);
                            // store_c::<MR_LEFT,mr_left>(c_cur1, ct, m_left, ldc);
                        }
                        )*
                        return;
                    }
                )*
            }        
        }});
    };
}

def_milikernel_blocked2!(4, 24, true, r, rb_t, 24, 16, 8);
def_milikernel_blocked2!(4, 24, false, c, cb_t, 24, 16, 8);



pub(crate) unsafe fn kernel_sup_m(
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    b: *const TB, b_rs: usize, b_cs: usize,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap: *const TA,
) {  
    if c_rs == 1 {
        if b_cs == 1 {
            kernel_sup_m_r(
                m, n, k,
                alpha, beta,
                b, b_rs,
                c, c_cs,
                ap
            );
        } else {
            kernel_sup_m_c(
                m, n, k,
                alpha, beta,
                b, b_cs,
                c, c_cs,
                ap
            );
        }
    } else {
        panic!("Strided C is not supported yet");
    }

}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn kernel_sup_n(
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    b: *const TB, b_rs: usize, b_cs: usize,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap: *const TA,
 ) { 
    if c_rs == 1 {
        if b_cs == 1 {
            kernel_sup_n_r(
                m, n, k,
                alpha, beta,
                b, b_rs,
                c, c_cs,
                ap
            );
        } else {
            kernel_sup_n_c(
                m, n, k,
                alpha, beta,
                b, b_cs,
                c, c_cs,
                ap
            );
        }
    } else {
        panic!("Strided C is not implemented");
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
   if MR == 24 && NR == 4 {
        if c_rs == 1 {
            kernel_24x4(m, n, k, alpha, beta, c, c_cs, ap, bp)
        } else {
            kernel_24x4_strided(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp)
        }
   } else if MR == 16 && NR == 6 {
    if c_rs == 1 {
        kernel_16x6(m, n, k, alpha, beta, c, c_cs, ap, bp)
    } else {
        kernel_16x6_strided(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp)
    }
    } else {
        unimplemented!()
    }
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn pack_panel<const MR: usize>(
    m: usize, k: usize,
    a: *const TB, a_rs: usize, a_cs: usize,
    ap: *mut TB,
){
    if MR == 24 {
        pack_panel_24(m, k, a, a_rs, a_cs, ap)
    } else if MR == 16 {
        pack_panel_16(m, k, a, a_rs, a_cs, ap)
    } else if MR == 6 {
        pack_panel_6(m, k, a, a_rs, a_cs, ap)
    } else if MR == 4 {
        pack_panel_4(m, k, a, a_rs, a_cs, ap)
    } else  {
        panic!("Packing for MR = {} not implemented", MR);
    }
}
