
#![allow(unused)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]


use libc::{c_float, c_int, c_schar, c_void, c_double, c_ushort};


/// Integer type
pub type gint_t = i64;
/// Matrix dimension type
pub type dim_t = gint_t;
/// Stride type
pub type inc_t = gint_t;


const BLIS_TRANS_SHIFT: usize = 3;
const BLIS_CONJ_SHIFT: usize = 4;
const BLIS_UPLO_SHIFT: usize = 5;
const BLIS_UPPER_SHIFT: usize = 5;
const BLIS_DIAG_SHIFT: usize = 6;
const BLIS_LOWER_SHIFT: usize = 7;


/// Conjugation enum
#[repr(C)]
pub enum conj_t {
   BLIS_NO_CONJUGATE = 0,
   BLIS_CONJUGATE = 1 << BLIS_CONJ_SHIFT,
}


pub use self::conj_t::*;


/// Transpose enum
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum trans_t {
   BLIS_NO_TRANSPOSE = 0,
   BLIS_TRANSPOSE = 1 << BLIS_TRANS_SHIFT,
   BLIS_CONJ_NO_TRANSPOSE = 1 << BLIS_CONJ_SHIFT,
   BLIS_CONJ_TRANSPOSE = 1 << BLIS_TRANS_SHIFT | 1 << BLIS_CONJ_SHIFT
}


#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::enum_variant_names)]
pub enum CBLAS_LAYOUT {
   CblasRowMajor = 101,
   CblasColMajor = 102,
}
pub use self::CBLAS_LAYOUT::*;



#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::enum_variant_names)]
pub enum CBLAS_TRANSPOSE {
   CblasNoTrans = 111,
   CblasTrans = 112,
   CblasConjTrans = 113,
}
pub use self::CBLAS_TRANSPOSE::*;


#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum CBLAS_OFFSET {
   CblasRowOffset = 171,
   CblasColOffset = 172,
   CblasFixOffset = 173,
}
pub use self::CBLAS_OFFSET::*;


/// Error enum.
///
/// Actual C enum has more variants.
#[repr(C)]
pub enum err_t {
   BLIS_SUCCESS = -1,
   BLIS_FAILURE = -2,
   #[doc(hidden)]
   __INCOMPLETE = -140,
}


pub use self::trans_t::*;

#[cfg(feature="mkl")]
#[allow(dead_code)]
extern "C" {
   #[allow(clippy::too_many_arguments)]
   pub fn cblas_gemm_s8u8s32(
       layout: CBLAS_LAYOUT,
       transa: CBLAS_TRANSPOSE,
       transb: CBLAS_TRANSPOSE,
       offsetc: CBLAS_OFFSET,
       m: c_int,
       n: c_int,
       k: c_int,
       alpha: c_float,
       a: *const c_void,
       lda: c_int,
       oa: c_schar,
       b: *const c_void,
       ldb: c_int,
       ob: c_schar,
       beta: c_float,
       c: *mut c_int,
       ldc: c_int,
       oc: *const c_int,
   );


   pub fn cblas_sgemv(
       layout: CBLAS_LAYOUT,
       trans: CBLAS_TRANSPOSE,
       m: c_int, n: c_int,
       alpha: c_float,
       a: *const c_float, lda: c_int,
       x: *const c_float, incx: c_int,
       beta: c_float,
       y: *mut c_float, incy: c_int,
   );


   pub fn cblas_dgemm(
       layout: CBLAS_LAYOUT,
       transa: CBLAS_TRANSPOSE,
       transb: CBLAS_TRANSPOSE,
       m: c_int, n: c_int, k: c_int,
       alpha: c_double,
       a: *const c_double, lda: c_int,
       b: *const c_double, ldb: c_int,
       beta: c_double,
       c: *mut c_double, ldc: c_int,
   );


   pub fn cblas_dgemv(
       layout: CBLAS_LAYOUT,
       trans: CBLAS_TRANSPOSE,
       m: c_int, n: c_int,
       alpha: c_double,
       a: *const c_double, lda: c_int,
       x: *const c_double, incx: c_int,
       beta: c_double,
       y: *mut c_double, incy: c_int,
   );

   pub fn cblas_sgemm(
       layout: CBLAS_LAYOUT,
       transa: CBLAS_TRANSPOSE,
       transb: CBLAS_TRANSPOSE,
       m: c_int,
       n: c_int,
       k: c_int,
       alpha: c_float,
       a: *const c_float,
       lda: c_int,
       b: *const c_float,
       ldb: c_int,
       beta: c_float,
       c: *mut c_float,
       ldc: c_int,
   );

   pub fn cblas_hgemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_ushort,
        a: *const c_ushort,
        lda: c_int,
        b: *const c_ushort,
        ldb: c_int,
        beta: c_ushort,
        c: *mut c_ushort,
        ldc: c_int,
    );


    pub fn cblas_sgemm_batch(
        layout: CBLAS_LAYOUT,
        transa: *const CBLAS_TRANSPOSE,
        transb: *const CBLAS_TRANSPOSE,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const c_float,
        a: *const(*const c_float),
        lda: *const c_int,
        b: *const(*const c_float),
        ldb: *const c_int,
        beta: *const c_float,
        c: *const(*mut c_float),
        ldc: *const c_int,
        group_count: c_int,
        group_size: *const c_int,
    );
}


#[cfg(feature="blis")]
#[allow(dead_code)]
extern "C" {
   pub fn bli_sgemm(
       // layout: CBLAS_LAYOUT,
       transa: trans_t,
       transb: trans_t,
       m: c_int, n: c_long, k: c_int,
       alpha: *const c_float,
       a: *const c_float, rsa: i32, csa: i32,
       b: *const c_float, rsb: i32, csb: i32,
       beta: *const c_float,
       c: *mut c_float, rsc: i32, csc: i32,
   );


   pub fn bli_dgemm(
       // layout: CBLAS_LAYOUT,
       transa: trans_t,
       transb: trans_t,
       m: c_int, n: c_int, k: c_int,
       alpha: *const c_double,
       a: *const c_double, rsa: i32, csa: i32,
       b: *const c_double, rsb: i32, csb: i32,
       beta: *const c_double,
       c: *mut c_double, rsc: i32, csc: i32,
   );

}

#[cfg(feature="blasfeo")]
#[allow(dead_code)]
extern "C" {
    pub fn blasfeo_cblas_sgemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
     );

}     


use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;


pub fn random_matrix<T>(m: usize, n: usize, arr: &mut [T], ld: usize)
where rand::distributions::Standard: rand::prelude::Distribution<T>,
{
   let mut x = StdRng::seed_from_u64(43);
   for j in 0..n {
       for i in 0..m {
           // arr[j * ld + i] = rand::random::<T>();
           arr[j * ld + i] = x.gen::<T>();
       }
   }
}


#[derive(Copy, Clone, Debug)]
pub enum GemmBackend {
   Blis,
   Mkl,
   Blasfeo,
   RustGemm,
   Corenum,
}

#[derive(Copy, Clone, Debug)]
pub enum BenchType {
    SGemm,
    SGemmBatched,
}

pub fn dispatch_gemm_f32(
   backend: GemmBackend,
   m: usize, n: usize, k: usize,
   alpha: f32,
   a: *const f32, a_rs: isize, a_cs: isize,
   b: *const f32, b_rs: isize, b_cs: isize,
   beta: f32,
   c: *mut f32, c_rs: isize, c_cs: isize,
) {
   match backend {
       GemmBackend::Blis => {
           unsafe {
            #[cfg(feature="blis")]
               bli_sgemm(
                   BLIS_NO_TRANSPOSE,
                   BLIS_NO_TRANSPOSE,
                   m as i32, n as i32, k as i32,
                   &alpha,
                   a, a_rs as i32, a_cs as i32,
                   b, b_rs as i32, b_cs as i32,
                   &beta,
                   c, c_rs as i32, c_cs as i32,
               );
           }
       }
       GemmBackend::Mkl => {
           unsafe {
            #[cfg(feature="mkl")]
            {
                let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs as usize, a_cs as usize, b_rs as usize, b_cs as usize, c_rs as usize, c_cs as usize);
                cblas_sgemm(
                    layout,
                    transa,
                    transb,
                    m as i32, n as i32, k as i32,
                    alpha,
                    a, lda as i32,
                    b, ldb as i32,
                    beta,
                    c, ldc as i32,
                );
            }
           }
       }
       GemmBackend::Blasfeo => {
           unsafe {
            #[cfg(feature="blasfeo")]
               blasfeo_cblas_sgemm(
                   CblasColMajor,
                   CblasNoTrans,
                   CblasNoTrans,
                   m as i32, n as i32, k as i32,
                   alpha,
                   a, a_rs as i32,
                   b, b_rs as i32,
                   beta,
                   c, c_rs as i32,
               );
           }
       }
       GemmBackend::RustGemm => {
           unsafe {
            #[cfg(feature="rustgemm")]
               gemm::gemm(
                   m, n, k,
                   c, c_cs as isize, c_rs as isize,
                   true,
                   a, a_cs as isize, a_rs as isize,
                   b, b_cs as isize, b_rs as isize,
                   alpha, beta,
                   false, false, false,
                   gemm::Parallelism::Rayon(1)
               );
              }
         }
         GemmBackend::Corenum => {
              unsafe {
                corenum_gemm_f32::corenum_sgemm(
                     m, n, k,
                     alpha,
                     a, a_rs as usize, a_cs as usize,
                     b, b_rs as usize, b_cs as usize,
                     beta,
                     c, c_rs as usize, c_cs as usize,
                );
            }
        }
    }
}

pub fn dispatch_gemm_batch_f32(
    backend: GemmBackend,
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: *const f32, a_rs: isize, a_cs: isize, stridea: isize,
    b: *const f32, b_rs: isize, b_cs: isize, strideb: isize,
    beta: f32,
    c: *mut f32, c_rs: isize, c_cs: isize, stridec: isize,
    batch_size: usize,
) {
    match backend {
        GemmBackend::Blis => {
            unsafe {
             #[cfg(feature="blis")]
                for i in 0..batch_size {
                    bli_sgemm(
                        BLIS_NO_TRANSPOSE,
                        BLIS_NO_TRANSPOSE,
                        m as i32, n as i32, k as i32,
                        &alpha,
                        a.offset(i as isize * stridea), a_rs as i32, a_cs as i32,
                        b.offset(i as isize * strideb), b_rs as i32, b_cs as i32,
                        &beta,
                        c.offset(i as isize * stridec), c_rs as i32, c_cs as i32,
                    );
                }
            }
        }
        GemmBackend::Mkl => {
            unsafe {
             #[cfg(feature="mkl")]
             {
                let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs as usize, a_cs as usize, b_rs as usize, b_cs as usize, c_rs as usize, c_cs as usize);
                let transa_vec = vec![transa; batch_size];
                let transb_vec = vec![transb; batch_size];
                let m_vec = vec![m as i32; batch_size];
                let n_vec = vec![n as i32; batch_size];
                let k_vec = vec![k as i32; batch_size];
                let alpha_vec = vec![alpha; batch_size];
                let lda_vec = vec![lda; batch_size];
                let ldb_vec = vec![ldb; batch_size];
                let beta_vec = vec![beta; batch_size];
                let ldc_vec = vec![ldc; batch_size];
                let a_vec = (0..batch_size).map(|i| a.offset(i as isize * stridea)).collect::<Vec<*const f32>>();
                let b_vec = (0..batch_size).map(|i| b.offset(i as isize * strideb)).collect::<Vec<*const f32>>();
                let c_vec = (0..batch_size).map(|i| c.offset(i as isize * stridec)).collect::<Vec<*mut f32>>();
                let stride_size_vec = [batch_size as i32; 1];

                cblas_sgemm_batch(
                    layout,
                    transa_vec.as_ptr(),
                    transb_vec.as_ptr(),
                    m_vec.as_ptr(),
                    n_vec.as_ptr(),
                    k_vec.as_ptr(),
                    alpha_vec.as_ptr(),
                    a_vec.as_ptr(),
                    lda_vec.as_ptr(),
                    b_vec.as_ptr(),
                    ldb_vec.as_ptr(),
                    beta_vec.as_ptr(),
                    c_vec.as_ptr(),
                    ldc_vec.as_ptr(),
                    1,
                    stride_size_vec.as_ptr(),
                );
            }
        }
        }
        GemmBackend::Blasfeo => {
            // unsafe {
            //  #[cfg(feature="blasfeo")]
            //     for i in 0..batch_size {
            //         blasfeo_cblas_sgemm(
            //             CblasColMajor,
            //             CblasNoTrans,
            //             CblasNoTrans,
            //             m as i32, n as i32, k as i32,
            //             alpha,
            //             a.offset(i as isize * stridea),
            //             a_rs as i32,
            //             b.offset(i as isize * strideb),
            //             b_rs as i32,
            //             beta,
            //             c.offset(i as isize * stridec),
            //             c_rs as i32,
            //         );
            //     }
            // }
            unsafe {
                #[cfg(feature="mkl")]
                {
                   let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs as usize, a_cs as usize, b_rs as usize, b_cs as usize, c_rs as usize, c_cs as usize);
                   for i in 0..batch_size {
                          cblas_sgemm(
                            layout,
                            transa,
                            transb,
                            m as i32, n as i32, k as i32,
                            alpha,
                            a.offset(i as isize * stridea), lda as i32,
                            b.offset(i as isize * strideb), ldb as i32,
                            beta,
                            c.offset(i as isize * stridec), ldc as i32,
                          );
                   }
               }
           }
        }
        GemmBackend::RustGemm => {
            unsafe {
             #[cfg(feature="rustgemm")]
                for i in 0..batch_size {
                    gemm::gemm(
                        m, n, k,
                        c.offset(i as isize * stridec), c_cs as isize, c_rs as isize,
                        true,
                        a.offset(i as isize * stridea), a_cs as isize, a_rs as isize,
                        b.offset(i as isize * strideb), b_cs as isize, b_rs as isize,
                        alpha, beta,
                        false, false, false,
                        gemm::Parallelism::Rayon(1)
                    );
                }
            }
        }
        GemmBackend::Corenum => {
            unsafe {
             #[cfg(feature="corenum")]
                for i in 0..batch_size {
                    corenum_gemm_f32::corenum_sgemm(
                        m, n, k,
                        alpha,
                        a.offset(i as isize * stridea), a_rs as usize, a_cs as usize,
                        b.offset(i as isize * strideb), b_rs as usize, b_cs as usize,
                        beta,
                        c.offset(i as isize * stridec), c_rs as usize, c_cs as usize,
                    );
                }
            }
        }
    }
}


pub fn max_abs_diff<T: Copy + std::ops::Sub + Into<f64> + std::fmt::Debug>(ap: &[T], bp: &[T]) -> f64
where f64: From<<T as std::ops::Sub>::Output>
{
   let mut diff = 0_f64;
   let len = ap.len();
   // println!("------------------------------");
   let mut diff_idx = 0;
   for i in 0..len {
       let a = ap[i];
       let b = bp[i];
       let cur_diff = <<T as std::ops::Sub>::Output as Into<f64>>::into(a-b).abs();
       // println!("cur_diff: {:?}, a: {:?}, b: {:?}", cur_diff, a, b);
       if cur_diff > diff {
        //    println!("i: {:?}, cur_diff: {:?}, a: {:?}, b: {:?}", i, cur_diff, a, b);
            diff_idx = i;
           diff = cur_diff;
       }
   }
//    println!("diff_idx: {:?}", diff_idx);
//    println!("ap[diff_idx]: {:?}, bp[diff_idx]: {:?}", ap[diff_idx], bp[diff_idx]);
   diff
   // let diff = ap.iter().zip(bp.iter()).map(|(a, b)| <<T as std::ops::Sub>::Output as Into<f64>>::into(*a-*b).abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
   // diff
}


pub fn my_gemm_ref_col(m: usize, n: usize, k: usize, a: &[f64], ld_a: usize, b: &[f64], ld_b: usize, c: &mut [f64], ld_c: usize) {
   for i in 0..m {
       for j in 0..n {
           for p in 0..k {
               c[j*ld_c + i] += a[p*ld_a + i] * b[j*ld_b + p];
           }
       }
   }
}


pub fn my_gemm_ref_row(m: usize, n: usize, k: usize, a: &[f64], ld_a: usize, b: &[f64], ld_b: usize, c: &mut [f64], ld_c: usize) {
   for i in 0..m {
       for j in 0..n {
           for p in 0..k {
               c[i*ld_c + j] += a[i*ld_a + p] * b[p*ld_b + j];
           }
       }
   }
}


// #[target_feature(enable = "neon")]
pub unsafe fn gemm_fallback_f32(
	m: usize, n: usize, k: usize,
	alpha: f32,
	a: *const f32, a_rs: usize, a_cs: usize,
	b: *const f32, b_rs: usize, b_cs: usize,
	beta: f32,
	c: *mut f32, c_rs: usize, c_cs: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut dx = 0.0;
            for p in 0..k {
                dx += *a.add(a_rs * i + a_cs * p) * *b.add(b_rs * p + b_cs * j);
            }
            *c.add(c_rs * i + c_cs * j ) = alpha * dx +  beta * *c.add(c_rs * i + c_cs * j );
        }
    }
}


fn stride_to_cblas(
    m: usize, n: usize, k: usize,
	a_rs: usize, a_cs: usize,
	b_rs: usize, b_cs: usize,
	c_rs: usize, c_cs: usize,
) -> (CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, c_int, c_int, c_int) {
    let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = if c_rs == 1 {
        (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs)
    } else if c_cs == 1 {
        (a_cs, a_rs, b_cs, b_rs, c_cs, c_rs)
    } else {
        panic!("Non Trivial Stride is not available for Cblas Api");
    };
    // c_rs == 1
    let ldc = c_cs as c_int;
    let (a_trans, b_trans, lda, ldb) = if a_rs == 1 && b_rs == 1 && a_cs == m && b_cs == k {
        (CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, a_cs as c_int, b_cs as c_int)
    } else if a_rs == 1 && b_cs == 1 && a_cs == m && b_rs == n {
        (CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans, a_cs as c_int, b_rs as c_int)
    } else if a_cs == 1 && b_rs == 1 && a_rs == k && b_cs == k {
        (CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans, a_rs as c_int, b_cs as c_int)
    } else if a_cs == 1 && b_cs == 1 && a_rs == k && b_rs == n {
        (CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasTrans, a_rs as c_int, b_rs as c_int)
    } else {
        panic!("Non Trivial Stride is not available for Cblas Api");
    };
    (CBLAS_LAYOUT::CblasColMajor, a_trans, b_trans, lda, ldb, ldc)
}


fn cblas_to_stride(
    layout: CBLAS_LAYOUT,
    transa: CBLAS_TRANSPOSE, transb: CBLAS_TRANSPOSE,
    lda: c_int, ldb: c_int, ldc: c_int,
) -> (usize, usize, usize, usize, usize, usize ) {
    if layout == CBLAS_LAYOUT::CblasColMajor {
        let (a_rs, a_cs) = if transa == CBLAS_TRANSPOSE::CblasNoTrans {
            (1, lda as usize)
        } else {
            (lda as usize, 1)
        };
        let (b_rs, b_cs) = if transb == CBLAS_TRANSPOSE::CblasNoTrans {
            (1, ldb as usize)
        } else {
            (ldb as usize, 1)
        };
        (a_rs, a_cs, b_rs, b_cs, 1, ldc as usize)
    } else {
        let (a_rs, a_cs) = if transa == CBLAS_TRANSPOSE::CblasNoTrans {
            (lda as usize, 1)
        } else {
            (1, lda as usize)
        };
        let (b_rs, b_cs) = if transb == CBLAS_TRANSPOSE::CblasNoTrans {
            (ldb as usize, 1)
        } else {
            (1, ldb as usize)
        };
        (a_rs, a_cs, b_rs, b_cs, ldc as usize, 1)
    }
}



pub unsafe fn check_gemm_f32(
	m: usize, n: usize, k: usize,
	alpha: f32,
	a: *const f32, a_rs: usize, a_cs: usize,
	b: *const f32, b_rs: usize, b_cs: usize,
	beta: f32,
	c: &[f32], c_rs: usize, c_cs: usize,
    c_ref: &mut [f32],
) -> f64 {
    #[cfg(feature="mkl")] {
        let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        cblas_sgemm(
            layout, transa, transb, m as c_int, n as c_int, k as c_int, alpha, a, lda, b, ldb, beta, c_ref.as_mut_ptr(), ldc
        );
        let diff = max_abs_diff(&c, &c_ref);
        return diff;
    }
    #[cfg(not(feature="mkl"))] {
        // calculate diff using fallback
        gemm_fallback_f32(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c_ref.as_mut_ptr(), c_rs, c_cs);
        let diff = max_abs_diff(&c, &c_ref);
        return diff;
    }

}


pub fn cblas_params_from_str(layout_str: &str, m: usize, n: usize, k: usize) ->(i32, i32, i32, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE) {
    if layout_str == "nn" {
        (m as i32, k as i32, m as i32, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans)
    } else if layout_str == "nt" {
        (m as i32, n as i32, m as i32, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans)
    } else if layout_str == "tn" {
        (k as i32, k as i32, m as i32, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans)
    } else if layout_str == "tt" {
        (k as i32, n as i32, m as i32, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasTrans)
    } else {
        panic!("Unsupported layout str");
    }
 }
 
 
pub fn blis_params_from_str(layout_str: &str, m: usize, n: usize, k: usize) ->(isize, isize, isize, isize, isize, isize) {
    if layout_str == "nn" {
        (1, m as isize, 1, k as isize, 1, m as isize)
    } else if layout_str == "nt" {
        (1, m as isize, n as isize, 1, 1, m as isize)
    } else if layout_str == "tn" {
        (k as isize, 1, 1, k as isize, 1, m as isize)
    } else if layout_str == "tt" {
        (k as isize, 1, n as isize, 1, 1, m as isize)
    } else {
        panic!("Unsupported layout str");
    }
 }

pub fn gemm_backend_from_str(backend_str: &str) -> GemmBackend {
    if backend_str == "mkl" {
        GemmBackend::Mkl
    } else if backend_str == "blis" {
        GemmBackend::Blis
    } else if backend_str == "blasfeo" {
        GemmBackend::Blasfeo
    } else if backend_str == "rustgemm" {
        GemmBackend::RustGemm
    } else if backend_str == "corenum" {
        GemmBackend::Corenum
    } else {
        panic!("Unsupported backend str");
    }
 }

pub fn bench_type_from_str(bench_type_str: &str) -> BenchType {
    if bench_type_str == "sgemm" {
        BenchType::SGemm
    } else if bench_type_str == "sgemm_batched" {
        BenchType::SGemmBatched
    } else {
        panic!("Unsupported bench type str");
    }
}

fn test_sgemm(
    m: usize, n: usize, k: usize,
    gemm_backend: GemmBackend, args: &Args,
    alpha: f32, beta: f32,
    a_rs: isize, a_cs: isize,
    b_rs: isize, b_cs: isize,
    c_rs: isize, c_cs: isize,
) -> f64 {
    let mut a = vec![0.0; m * k];
    let mut b = vec![0.0; k * n];
    let mut c = vec![0.0; m * n];
    random_matrix(m, k, &mut a, m);
    random_matrix(k, n, &mut b, k);
    let mut c_ref = vec![0.0; m * n];
    c_ref.copy_from_slice(&c);
    let start_time = std::time::Instant::now();
    dispatch_gemm_f32(
        gemm_backend,
        m, n, k,
        alpha,
        a.as_ptr(), a_rs, a_cs,
        b.as_ptr(), b_rs, b_cs,
        beta,
        c.as_mut_ptr(), c_rs, c_cs,
    );
    if args.check {
        let diff = unsafe {
            check_gemm_f32(
                m, n, k, 
                alpha, 
                a.as_ptr(), a_rs as usize, a_cs as usize, 
                b.as_ptr(), b_rs as usize, b_cs as usize, 
                beta, 
                &c, c_rs as usize, c_cs as usize, 
                &mut c_ref
            )
        };
        println!("diff: {}", diff);
        // println!("c: {:?}", c);
        // println!("c_ref: {:?}", c_ref);
    }

    let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;

    end_time
}

fn test_sgemm_batched(
    m: usize, n: usize, k: usize,
    gemm_backend: GemmBackend, args: &Args,
    alpha: f32, beta: f32,
    a_rs: isize, a_cs: isize,
    b_rs: isize, b_cs: isize,
    c_rs: isize, c_cs: isize,
    batch_size: usize,
) -> f64 {
    let mut a = vec![0.0; m * k * batch_size];
    let mut b = vec![0.0; k * n * batch_size];
    let mut c = vec![0.0; m * n * batch_size];
    let stridea = m * k;
    let strideb = k * n;
    let stridec = m * n;
    for i in 0..batch_size {
        random_matrix(m, k, &mut a[i * stridea..], m);
        random_matrix(k, n, &mut b[i * strideb..], k);
    }
    let mut c_ref = vec![0.0; m * n * batch_size];
    c_ref.copy_from_slice(&c);
    let start_time = std::time::Instant::now();
    dispatch_gemm_batch_f32(
        gemm_backend,
        m, n, k,
        alpha,
        a.as_ptr(), a_rs, a_cs, stridea as isize,
        b.as_ptr(), b_rs, b_cs, strideb as isize,
        beta,
        c.as_mut_ptr(), c_rs, c_cs, stridec as isize,
        batch_size,
    );
    if args.check {
        let diff = unsafe {
            check_gemm_f32(
                m, n, k, 
                alpha, 
                a.as_ptr(), a_rs as usize, a_cs as usize, 
                b.as_ptr(), b_rs as usize, b_cs as usize, 
                beta, 
                &c, c_rs as usize, c_cs as usize, 
                &mut c_ref
            )
        };
        println!("diff: {}", diff);
        // println!("c: {:?}", c);
        // println!("c_ref: {:?}", c_ref);
    }

    let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;

    end_time
}


use corenum_gemm_f32::{
    CorenumPar,
};
 
use clap::Parser;
 
 
/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// dim m
    #[arg(short, long, default_value_t = 200)]
    m: usize,

    /// dim n
    #[arg(short, long, default_value_t = 200)]
    n: usize,

    /// dim k
    #[arg(short, long, default_value_t = 200)]
    k: usize,

    /// batch dim
    #[arg(short, long, default_value_t = 5)]
    batch_dim: usize,

   // tranpose layout
   #[arg(short, long, default_value_t = String::from("nt"))]
   t_layout: String,

   #[arg(short, long, default_value_t = false)]
   check: bool,

   // gemm backend
    #[arg(short, long, default_value_t = String::from("corenum"))]
    backend: String,

    // bench type
    #[arg(short, long, default_value_t = String::from("sgemm"))]
    bench_type: String,
}
 
 
 fn main() {
    let n_repeats = 2;
    let mut total_time = 0.0;
 
    let mut best_time = f64::INFINITY;
    let beta = 1.0;
    let alpha = 1.0;
    let args = Args::parse();
    let m = args.m;
    let n = args.n;
    let k = args.k;
    let layout_str = &args.t_layout;

    let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = blis_params_from_str(layout_str, m, n, k);

    let gemm_backend = gemm_backend_from_str(&args.backend);
    let bench_type = bench_type_from_str(&args.bench_type);
    let batch_dim = args.batch_dim;
    let mut rep = 0;
    while rep < n_repeats {
        let end_time = match bench_type {
            BenchType::SGemm => test_sgemm(m, n, k, gemm_backend, &args, alpha, beta, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::SGemmBatched => test_sgemm_batched(m, n, k, gemm_backend, &args, alpha, beta, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, batch_dim),
        };
        total_time += end_time;

        println!("time: {}, total_time: {}", end_time, total_time);
        if best_time > end_time {
            best_time = end_time;
        }
        rep += 1;
    }
    let gflops = 2.0 * m as f64 * n as f64 * k as f64 / best_time / 1e9;
    println!("best_time: {}, GFLOPS: {}", best_time, gflops);
 }
 