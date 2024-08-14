
#![allow(non_camel_case_types)]

use glare_dev::{
    random_matrix_std, random_matrix_uniform, max_abs_diff
};

use libc::{c_float, c_int, c_schar, c_void, c_double, c_ushort, c_short};

use num_complex::{
    c32,
    Complex32
};
const BLIS_TRANS_SHIFT: usize = 3;
const BLIS_CONJ_SHIFT: usize = 4;
// const BLIS_UPLO_SHIFT: usize = 5;
// const BLIS_UPPER_SHIFT: usize = 5;
// const BLIS_DIAG_SHIFT: usize = 6;
// const BLIS_LOWER_SHIFT: usize = 7;


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


pub use self::trans_t::*;

#[cfg(feature="mkl")]
#[allow(dead_code)]
extern "C" {
   #[allow(clippy::too_many_arguments)]
   pub fn cblas_gemm_s16s16s32(
       layout: CBLAS_LAYOUT,
       transa: CBLAS_TRANSPOSE,
       transb: CBLAS_TRANSPOSE,
       offsetc: CBLAS_OFFSET,
       m: c_int, n: c_int, k: c_int,
       alpha: c_float,
       a: *const c_short, lda: c_int, oa: c_short,
       b: *const c_short, ldb: c_int, ob: c_short,
       beta: c_float,
       c: *mut c_int, ldc: c_int, oc: *const c_int,
   );

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

   pub fn cblas_cgemm(
    layout: CBLAS_LAYOUT,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    lda: c_int,
    b: *const c_void,
    ldb: c_int,
    beta: *const c_void,
    c: *mut c_void,
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
        a: *const *const c_float,
        lda: *const c_int,
        b: *const *const c_float,
        ldb: *const c_int,
        beta: *const c_float,
        c: *const *mut c_float,
        ldc: *const c_int,
        group_count: c_int,
        group_size: *const c_int,
    );
}


#[cfg(feature="blis")]
#[allow(dead_code)]
extern "C" {
    pub fn bli_cgemm(
        transa: trans_t,
        transb: trans_t,
        m: c_int, n: c_int, k: c_int,
        alpha: *const c_void,
        a: *const c_void, rsa: i32, csa: i32,
        b: *const c_void, rsb: i32, csb: i32,
        beta: *const c_void,
        c: *mut c_void, rsc: i32, csc: i32,
    );
   pub fn bli_sgemm(
       transa: trans_t,
       transb: trans_t,
       m: c_int, n: c_int, k: c_int,
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

#[derive(Copy, Clone, Debug)]
pub enum GemmBackend {
   Blis,
   Mkl,
   RustGemm,
   Corenum,
}

#[derive(Copy, Clone, Debug)]
pub enum BenchType {
    DGemm,
    SGemm,
    SGemmBatched,
    HGemm,
    CGemm,
    GemmS16S16S32,
    GemmS8U8S32,
}

pub unsafe fn dispatch_dgemm(
    backend: GemmBackend,
    m: usize, n: usize, k: usize,
    alpha: f64,
    a: *const f64, a_rs: isize, a_cs: isize,
    b: *const f64, b_rs: isize, b_cs: isize,
    beta: f64,
    c: *mut f64, c_rs: isize, c_cs: isize,
 ) {
    match backend {
        GemmBackend::Blis => {
             #[cfg(feature="blis")]
                bli_dgemm(
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
        GemmBackend::Mkl => {
             #[cfg(feature="mkl")]
             {
                 let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs as usize, a_cs as usize, b_rs as usize, b_cs as usize, c_rs as usize, c_cs as usize);
                 cblas_dgemm(
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
        GemmBackend::RustGemm => {
             #[cfg(feature="rustgemm")]
                gemm::gemm(
                    m, n, k,
                    c, c_cs as isize, c_rs as isize,
                    true,
                    a, a_cs as isize, a_rs as isize,
                    b, b_cs as isize, b_rs as isize,
                    alpha, beta,
                    false, false, false,
                    gemm::Parallelism::Rayon(0)
                 );
          }
          GemmBackend::Corenum => {
            glare_gemm_f64::glare_dgemm(
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
 

pub unsafe fn dispatch_sgemm(
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
       GemmBackend::Mkl => {
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
       GemmBackend::RustGemm => {
            #[cfg(feature="rustgemm")]
               gemm::gemm(
                   m, n, k,
                   c, c_cs as isize, c_rs as isize,
                   true,
                   a, a_cs as isize, a_rs as isize,
                   b, b_cs as isize, b_rs as isize,
                   alpha, beta,
                   false, false, false,
                   gemm::Parallelism::Rayon(0)
                );
         }
         GemmBackend::Corenum => {
            glare_gemm_f32::glare_sgemm(
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

pub unsafe fn dispatch_cgemm(
    backend: GemmBackend,
    m: usize, n: usize, k: usize,
    alpha: Complex32,
    a: *const Complex32, a_rs: isize, a_cs: isize,
    b: *const Complex32, b_rs: isize, b_cs: isize,
    beta: Complex32,
    c: *mut Complex32, c_rs: isize, c_cs: isize,
 ) {
    match backend {
        GemmBackend::Blis => {
             #[cfg(feature="blis")]
             {
                let a = a as *const c_void;
                let b = b as *const c_void;
                let c = c as *mut c_void;
                let alpha_ptr = &alpha as *const Complex32 as *const c_void;
                let beta_ptr = &beta as *const Complex32 as *const c_void;
                bli_cgemm(
                    BLIS_NO_TRANSPOSE,
                    BLIS_NO_TRANSPOSE,
                    m as i32, n as i32, k as i32,
                    alpha_ptr,
                    a, a_rs as i32, a_cs as i32,
                    b, b_rs as i32, b_cs as i32,
                    beta_ptr,
                    c, c_rs as i32, c_cs as i32,
                );
             }
        }
        GemmBackend::Mkl => {
             #[cfg(feature="mkl")]
             {
                let a = a as *const c_void;
                let b = b as *const c_void;
                let c = c as *mut c_void;
                let alpha_ptr = &alpha as *const Complex32 as *const c_void;
                let beta_ptr = &beta as *const Complex32 as *const c_void;
                 let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs as usize, a_cs as usize, b_rs as usize, b_cs as usize, c_rs as usize, c_cs as usize);
                 cblas_cgemm(
                     layout,
                     transa,
                     transb,
                     m as i32, n as i32, k as i32,
                     alpha_ptr,
                     a, lda as i32,
                     b, ldb as i32,
                     beta_ptr,
                     c, ldc as i32,
                 );
             }
        }
        GemmBackend::RustGemm => {
             #[cfg(feature="rustgemm")]
                gemm::gemm(
                    m, n, k,
                    c, c_cs as isize, c_rs as isize,
                    true,
                    a, a_cs as isize, a_rs as isize,
                    b, b_cs as isize, b_rs as isize,
                    alpha, beta,
                    false, false, false,
                    gemm::Parallelism::Rayon(0)
                 );
          }
          GemmBackend::Corenum => {
            panic!("Not implemented for glare");
         }
     }
 }
 


pub unsafe fn dispatch_gemm_batch_f32(
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
             #[cfg(feature="blis")]
             {
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
        GemmBackend::RustGemm => {
             #[cfg(feature="rustgemm")]
             {
                for i in 0..batch_size {
                    gemm::gemm(
                        m, n, k,
                        c.offset(i as isize * stridec), c_cs as isize, c_rs as isize,
                        true,
                        a.offset(i as isize * stridea), a_cs as isize, a_rs as isize,
                        b.offset(i as isize * strideb), b_cs as isize, b_rs as isize,
                        alpha, beta,
                        false, false, false,
                        gemm::Parallelism::Rayon(0)
                    );
                }
            }
        }
        GemmBackend::Corenum => {
             #[cfg(feature="glare")]
             {
                for i in 0..batch_size {
                    glare_gemm_f32::glare_sgemm(
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

pub unsafe fn dispatch_gemm_f16(
    backend: GemmBackend,
    m: usize, n: usize, k: usize,
    alpha: f16,
    a: *const f16, a_rs: isize, a_cs: isize,
    b: *const f16, b_rs: isize, b_cs: isize,
    beta: f16,
    c: *mut f16, c_rs: isize, c_cs: isize,
 ) {
    match backend {
        GemmBackend::Blis => {
             #[cfg(feature="blis")]
            panic!("f16 not supported in blis");
        }
        GemmBackend::Mkl => {
             #[cfg(feature="mkl")]
             {
                 let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs as usize, a_cs as usize, b_rs as usize, b_cs as usize, c_rs as usize, c_cs as usize);
                 let a = a as *const c_ushort;
                    let b = b as *const c_ushort;
                    let c = c as *mut c_ushort;
                    let alpha = alpha.to_bits();
                    let beta = beta.to_bits();
                 cblas_hgemm(
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
        GemmBackend::RustGemm => {
             #[cfg(feature="rustgemm")]
                gemm::gemm(
                    m, n, k,
                    c, c_cs as isize, c_rs as isize,
                    true,
                    a, a_cs as isize, a_rs as isize,
                    b, b_cs as isize, b_rs as isize,
                    alpha, beta,
                    false, false, false,
                    gemm::Parallelism::Rayon(0)
                 );
          }
          GemmBackend::Corenum => {
            glare_gemm_f16::glare_hgemm(
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

 pub unsafe fn dispatch_gemm_s16s16s32(
    backend: GemmBackend,
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: *const i16, a_rs: isize, a_cs: isize,
    b: *const i16, b_rs: isize, b_cs: isize,
    beta: f32,
    c: *mut i32, c_rs: isize, c_cs: isize,
 ) {
    match backend {
        GemmBackend::Blis => {
             #[cfg(feature="blis")]
                panic!("s16s16s32 is not supported in blis");
        }
        GemmBackend::Mkl => {
             #[cfg(feature="mkl")]
             {
                let oc_val = 0;
                let oc = &oc_val as *const c_int;
                 let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs as usize, a_cs as usize, b_rs as usize, b_cs as usize, c_rs as usize, c_cs as usize);
                 cblas_gemm_s16s16s32(
                     layout,
                     transa,
                     transb,
                     CblasFixOffset,
                     m as i32, n as i32, k as i32,
                     alpha,
                     a, lda as i32, 0,
                     b, ldb as i32, 0,
                     beta,
                     c, ldc as i32, oc,
                 );
             }
        }
        GemmBackend::RustGemm => {
             #[cfg(feature="rustgemm")]
                panic!("s16s16s32 is not supported in rustgemm");
          }
          GemmBackend::Corenum => {
             use glare_gemm_s16s16s32::glare_gemm_s16s16s32 as gemm_s16s16s32;
            gemm_s16s16s32(
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

 pub unsafe fn dispatch_gemm_s8u8s32(
    backend: GemmBackend,
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: *const i8, a_rs: isize, a_cs: isize,
    b: *const u8, b_rs: isize, b_cs: isize,
    beta: f32,
    c: *mut i32, c_rs: isize, c_cs: isize,
 ) {
    match backend {
        GemmBackend::Blis => {
             #[cfg(feature="blis")]
                panic!("s16s16s32 is not supported in blis");
        }
        GemmBackend::Mkl => {
             #[cfg(feature="mkl")]
             {
                let oc_val = 0;
                let oc = &oc_val as *const c_int;
                let a = a as *const c_void;
                let b = b as *const c_void;
                 let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs as usize, a_cs as usize, b_rs as usize, b_cs as usize, c_rs as usize, c_cs as usize);
                 cblas_gemm_s8u8s32(
                     layout,
                     transa,
                     transb,
                     CblasFixOffset,
                     m as i32, n as i32, k as i32,
                     alpha,
                     a, lda as i32, 0,
                     b, ldb as i32, 0,
                     beta,
                     c, ldc as i32, oc,
                 );
             }
        }
        GemmBackend::RustGemm => {
             #[cfg(feature="rustgemm")]
                panic!("s16s16s32 is not supported in rustgemm");
          }
          GemmBackend::Corenum => {
             use glare_gemm_s8u8s32::glare_gemm_s8u8s32 as gemm_s8u8s32;
            gemm_s8u8s32(
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


pub unsafe fn gemm_fallback_f64(
	m: usize, n: usize, k: usize,
	alpha: f64,
	a: *const f64, a_rs: usize, a_cs: usize,
	b: *const f64, b_rs: usize, b_cs: usize,
	beta: f64,
	c: *mut f64, c_rs: usize, c_cs: usize,
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
    let (a_rs, a_cs, b_rs, b_cs, _, c_cs) = if c_rs == 1 {
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

pub unsafe fn check_sgemm(
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
        let diff = max_abs_diff(&c, &c_ref, 1e-3);
        return diff;
    }
    #[cfg(not(feature="mkl"))] {
        // calculate diff using fallback
        gemm_fallback_f32(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c_ref.as_mut_ptr(), c_rs, c_cs);
        let diff = max_abs_diff(&c, &c_ref, 1e-3);
        return diff;
    }
}


pub unsafe fn check_gemm_s16s16s32(
	m: usize, n: usize, k: usize,
    alpha: f32,
    a: *const i16, a_rs: usize, a_cs: usize,
    b: *const i16, b_rs: usize, b_cs: usize,
    beta: f32,
    c: &[i32], c_rs: usize, c_cs: usize,
    c_ref: &mut [i32],
) -> f64 {
    #[cfg(feature="mkl")] {
        let oc_val = 0;
        let oc = &oc_val as *const c_int;
        let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        cblas_gemm_s16s16s32(
            layout, transa, transb, CblasFixOffset, m as c_int, n as c_int, k as c_int, alpha, a, lda, 0, b, ldb, 0, beta, c_ref.as_mut_ptr(), ldc, oc
        );
        let diff = max_abs_diff(&c, &c_ref, 1e-3);
        return diff;
    }
    // #[cfg(not(feature="mkl"))] {
    //     // calculate diff using fallback
    //     gemm_fallback_f32(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c_ref.as_mut_ptr(), c_rs, c_cs);
    //     let diff = max_abs_diff(&c, &c_ref, 1e-3);
    //     return diff;
    // }
}



pub unsafe fn check_gemm_s8u8s32(
	m: usize, n: usize, k: usize,
    alpha: f32,
    a: *const i8, a_rs: usize, a_cs: usize,
    b: *const u8, b_rs: usize, b_cs: usize,
    beta: f32,
    c: &[i32], c_rs: usize, c_cs: usize,
    c_ref: &mut [i32],
) -> f64 {
    #[cfg(feature="mkl")] {
        let oc_val = 0;
        let oc = &oc_val as *const c_int;
        let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        let a = a as *const c_void;
        let b = b as *const c_void;
        cblas_gemm_s8u8s32(
            layout, transa, transb, CblasFixOffset, m as c_int, n as c_int, k as c_int, alpha, a, lda, 0, b, ldb, 0, beta, c_ref.as_mut_ptr(), ldc, oc
        );
        let diff = max_abs_diff(&c, &c_ref, 1e-3);
        return diff;
    }
    // #[cfg(not(feature="mkl"))] {
    //     // calculate diff using fallback
    //     gemm_fallback_f32(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c_ref.as_mut_ptr(), c_rs, c_cs);
    //     let diff = max_abs_diff(&c, &c_ref, 1e-3);
    //     return diff;
    // }
}



pub unsafe fn check_cgemm(
	m: usize, n: usize, k: usize,
	alpha: Complex32,
	a: *const Complex32, a_rs: usize, a_cs: usize,
	b: *const Complex32, b_rs: usize, b_cs: usize,
	beta: Complex32,
	c: &[Complex32], c_rs: usize, c_cs: usize,
    c_ref: &mut [Complex32],
) -> f64 {
    #[cfg(feature="mkl")] {
        let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        let a = a as *const c_void;
        let b = b as *const c_void;
        let c_ref_ptr = c_ref.as_mut_ptr() as *mut c_void;
        let alpha_ptr = &alpha as *const Complex32 as *const c_void;
        let beta_ptr = &beta as *const Complex32 as *const c_void;
        cblas_cgemm(
            layout, transa, transb, m as c_int, n as c_int, k as c_int, alpha_ptr, a, lda, b, ldb, beta_ptr, c_ref_ptr, ldc
        );
        let diff = max_abs_diff(&c, &c_ref, 1e-3);
        return diff;
    }
    #[cfg(not(feature="mkl"))] {
        // calculate diff using fallback
        gemm_fallback_c32(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c_ref.as_mut_ptr(), c_rs, c_cs);
        let diff = max_abs_diff(&c, &c_ref, 1e-3);
        return diff;
    }
}

pub unsafe fn check_dgemm(
	m: usize, n: usize, k: usize,
	alpha: f64,
	a: *const f64, a_rs: usize, a_cs: usize,
	b: *const f64, b_rs: usize, b_cs: usize,
	beta: f64,
	c: &[f64], c_rs: usize, c_cs: usize,
    c_ref: &mut [f64],
) -> f64 {
    #[cfg(feature="mkl")] {
        let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        cblas_dgemm(
            layout, transa, transb, m as c_int, n as c_int, k as c_int, alpha, a, lda, b, ldb, beta, c_ref.as_mut_ptr(), ldc
        );
        let diff = max_abs_diff(&c, &c_ref, 1e-3);
        return diff;
    }
    #[cfg(not(feature="mkl"))] {
        // calculate diff using fallback
        gemm_fallback_f64(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c_ref.as_mut_ptr(), c_rs, c_cs);
        let diff = max_abs_diff(&c, &c_ref, 1e-3);
        return diff;
    }
}

pub unsafe fn check_hgemm(
	m: usize, n: usize, k: usize,
	alpha: f16,
	a: *const f16, a_rs: usize, a_cs: usize,
	b: *const f16, b_rs: usize, b_cs: usize,
	beta: f16,
	c: &[f16], c_rs: usize, c_cs: usize,
    c_ref: &mut [f16],
) -> f64 {
    #[cfg(feature="mkl")] {
        let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        let a = a as *const c_ushort;
        let b = b as *const c_ushort;
        let c_ref_ptr = c_ref.as_mut_ptr() as *mut c_ushort;
        let alpha = alpha.to_bits();
        let beta = beta.to_bits();
        cblas_hgemm(
            layout, transa, transb, m as c_int, n as c_int, k as c_int, alpha, a, lda, b, ldb, beta, c_ref_ptr, ldc
        );
        let diff = max_abs_diff(&c, &c_ref, 1e-1);
        return diff;
    }
    #[cfg(not(feature="mkl"))] {
        // calculate diff using fallback
        gemm_fallback_f16(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c_ref.as_mut_ptr(), c_rs, c_cs);
        let diff = max_abs_diff(&c, &c_ref, 1e-1);
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
        return GemmBackend::Mkl;
    } 
    if backend_str == "blis" {
        return GemmBackend::Blis;
    } 
    if backend_str == "rustgemm" {
        return GemmBackend::RustGemm;
    } 
    if backend_str == "glare" {
        return GemmBackend::Corenum;
    } 
    panic!("Unsupported backend str");
 }

pub fn bench_type_from_str(bench_type_str: &str) -> BenchType {
    if bench_type_str == "sgemm" {
        return BenchType::SGemm;
    } 
    if bench_type_str == "sgemm_batched" {
        return BenchType::SGemmBatched;
    } 
    if bench_type_str == "hgemm" {
        return BenchType::HGemm;
    } 
    if bench_type_str == "dgemm" {
        return BenchType::DGemm;
    }
    if bench_type_str == "cgemm" {
        return BenchType::CGemm;
    }
    if bench_type_str == "gemm_s16s16s32" {
        return BenchType::GemmS16S16S32;
    }
    if bench_type_str == "gemm_s8u8s32" {
        return BenchType::GemmS8U8S32;
    }
    panic!("Unsupported bench type str");
}


fn test_dgemm(
    m: usize, n: usize, k: usize,
    gemm_backend: GemmBackend, args: &Args,
    alpha: f32, beta: f32,
    a_rs: isize, a_cs: isize,
    b_rs: isize, b_cs: isize,
    c_rs: isize, c_cs: isize,
) -> f64 {
    let alpha = alpha as f64;
    let beta = beta as f64;
    let mut a = vec![0.0; m * k];
    let mut b = vec![0.0; k * n];
    let mut c = vec![0.0; m * n];
    random_matrix_uniform(m, k, &mut a, m);
    random_matrix_uniform(k, n, &mut b, k);
    let mut c_ref = vec![0.0; m * n];
    c_ref.copy_from_slice(&c);
    let start_time = std::time::Instant::now();
    unsafe {
        dispatch_dgemm(
            gemm_backend,
            m, n, k,
            alpha,
            a.as_ptr(), a_rs, a_cs,
            b.as_ptr(), b_rs, b_cs,
            beta,
            c.as_mut_ptr(), c_rs, c_cs,
        );
    }

    if args.check {
        let diff = unsafe {
            check_dgemm(
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
    }

    let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;

    end_time
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
    random_matrix_uniform(m, k, &mut a, m);
    random_matrix_uniform(k, n, &mut b, k);
    let mut c_ref = vec![0.0; m * n];
    c_ref.copy_from_slice(&c);
    let start_time = std::time::Instant::now();
    unsafe {
        dispatch_sgemm(
            gemm_backend,
            m, n, k,
            alpha,
            a.as_ptr(), a_rs, a_cs,
            b.as_ptr(), b_rs, b_cs,
            beta,
            c.as_mut_ptr(), c_rs, c_cs,
        );
    }
    if args.check {
        let diff = unsafe {
            check_sgemm(
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
    }

    let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;

    end_time
}

fn test_cgemm(
    m: usize, n: usize, k: usize,
    gemm_backend: GemmBackend, args: &Args,
    alpha: f32, beta: f32,
    a_rs: isize, a_cs: isize,
    b_rs: isize, b_cs: isize,
    c_rs: isize, c_cs: isize,
) -> f64 {
    let alpha = c32(alpha, 0.0);
    let beta = c32(beta, 0.0);
    let mut a = vec![Complex32::ZERO; m * k];
    let mut b = vec![Complex32::ZERO; k * n];
    let mut c = vec![Complex32::ZERO; m * n];
    random_matrix_std(m, k, &mut a, m);
    random_matrix_std(k, n, &mut b, k);
    let mut c_ref = vec![Complex32::ZERO; m * n];
    c_ref.copy_from_slice(&c);
    let start_time = std::time::Instant::now();
    unsafe {
        dispatch_cgemm(
            gemm_backend,
            m, n, k,
            alpha,
            a.as_ptr(), a_rs, a_cs,
            b.as_ptr(), b_rs, b_cs,
            beta,
            c.as_mut_ptr(), c_rs, c_cs,
        );
    }
    if args.check {
        let diff = unsafe {
            check_cgemm(
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
        random_matrix_uniform(m, k, &mut a[i * stridea..], m);
        random_matrix_uniform(k, n, &mut b[i * strideb..], k);
    }
    let mut c_ref = vec![0.0; m * n * batch_size];
    c_ref.copy_from_slice(&c);
    let start_time = std::time::Instant::now();
    unsafe {
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
    }

    if args.check {
        let diff = unsafe {
            check_sgemm(
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
    }

    let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;

    end_time
}


use half::f16;
fn test_hgemm(
    m: usize, n: usize, k: usize,
    gemm_backend: GemmBackend, args: &Args,
    alpha: f32, beta: f32,
    a_rs: isize, a_cs: isize,
    b_rs: isize, b_cs: isize,
    c_rs: isize, c_cs: isize,
) -> f64 {
    let alpha = f16::from_f32(alpha);
    let beta = f16::from_f32(beta);
    let mut a = vec![f16::ZERO; m * k];
    let mut b = vec![f16::ZERO; k * n];
    let mut c = vec![f16::ZERO; m * n];
    random_matrix_uniform(m, k, &mut a, m);
    random_matrix_uniform(k, n, &mut b, k);
    let mut c_ref = vec![f16::ZERO; m * n];
    c_ref.copy_from_slice(&c);
    let start_time = std::time::Instant::now();
    unsafe {
        dispatch_gemm_f16(
            gemm_backend,
            m, n, k,
            alpha,
            a.as_ptr(), a_rs, a_cs,
            b.as_ptr(), b_rs, b_cs,
            beta,
            c.as_mut_ptr(), c_rs, c_cs,
        );
    }

    if args.check {
        let diff = unsafe {
            check_hgemm(
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
    }

    let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;

    end_time
}


fn test_gemm_s16s16s32(
    m: usize, n: usize, k: usize,
    gemm_backend: GemmBackend, args: &Args,
    alpha: f32, beta: f32,
    a_rs: isize, a_cs: isize,
    b_rs: isize, b_cs: isize,
    c_rs: isize, c_cs: isize,
) -> f64 {
    let mut a = vec![0_i16; m * k];
    let mut b = vec![0_i16; k * n];
    let mut c = vec![0_i32; m * n];
    random_matrix_uniform(m, k, &mut a, m);
    random_matrix_uniform(k, n, &mut b, k);
    let mut c_ref = vec![0_i32; m * n];
    c_ref.copy_from_slice(&c);
    let start_time = std::time::Instant::now();
    unsafe {
        dispatch_gemm_s16s16s32(
            gemm_backend,
            m, n, k,
            alpha,
            a.as_ptr(), a_rs, a_cs,
            b.as_ptr(), b_rs, b_cs,
            beta,
            c.as_mut_ptr(), c_rs, c_cs,
        );
    }
    if args.check {
        let diff = unsafe {
            check_gemm_s16s16s32(
                m, n, k, 
                alpha, 
                a.as_ptr(), a_rs as usize, a_cs as usize, 
                b.as_ptr(), b_rs as usize, b_cs as usize, 
                beta, 
                &c, c_rs as usize, c_cs as usize, 
                &mut c_ref
            )
        };
        // println!("c: {:?}", c);
        // println!("c_ref: {:?}", c_ref);
        println!("diff: {}", diff);
    }

    let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;

    end_time
}




fn test_gemm_s8u8s32(
    m: usize, n: usize, k: usize,
    gemm_backend: GemmBackend, args: &Args,
    alpha: f32, beta: f32,
    a_rs: isize, a_cs: isize,
    b_rs: isize, b_cs: isize,
    c_rs: isize, c_cs: isize,
) -> f64 {
    let mut a = vec![0_i8; m * k];
    let mut b = vec![0_u8; k * n];
    let mut c = vec![0_i32; m * n];
    random_matrix_uniform(m, k, &mut a, m);
    random_matrix_uniform(k, n, &mut b, k);
    // println!("a: {:?}", a);
    let mut c_ref = vec![0_i32; m * n];
    c_ref.copy_from_slice(&c);
    let start_time = std::time::Instant::now();
    unsafe {
        dispatch_gemm_s8u8s32(
            gemm_backend,
            m, n, k,
            alpha,
            a.as_ptr(), a_rs, a_cs,
            b.as_ptr(), b_rs, b_cs,
            beta,
            c.as_mut_ptr(), c_rs, c_cs,
        );
    }
    if args.check {
        let diff = unsafe {
            check_gemm_s8u8s32(
                m, n, k, 
                alpha, 
                a.as_ptr(), a_rs as usize, a_cs as usize, 
                b.as_ptr(), b_rs as usize, b_cs as usize, 
                beta, 
                &c, c_rs as usize, c_cs as usize, 
                &mut c_ref
            )
        };
        // println!("c: {:?}", c);
        // println!("c_ref: {:?}", c_ref);
        println!("diff: {}", diff);
    }

    let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;

    end_time
}


 
use clap::Parser;
 
 
/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// number of repeats
    #[arg(short, long, default_value_t = 2)]
    n_repeats: usize,

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
    #[arg(short, long, default_value_t = String::from("glare"))]
    backend: String,

    // bench type
    #[arg(short, long, default_value_t = String::from("sgemm"))]
    bench_type: String,

}
 
 
 fn main() {
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
    let n_repeats = args.n_repeats;
    let mut rep = 0;
    while rep < n_repeats {
        let end_time = match bench_type {
            BenchType::DGemm => test_dgemm(m, n, k, gemm_backend, &args, alpha, beta, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::SGemm => test_sgemm(m, n, k, gemm_backend, &args, alpha, beta, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::SGemmBatched => test_sgemm_batched(m, n, k, gemm_backend, &args, alpha, beta, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, batch_dim),
            BenchType::HGemm => test_hgemm(m, n, k, gemm_backend, &args, alpha, beta, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::CGemm => test_cgemm(m, n, k, gemm_backend, &args, alpha, beta, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::GemmS16S16S32 => test_gemm_s16s16s32(m, n, k, gemm_backend, &args, alpha, beta, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::GemmS8U8S32 => test_gemm_s8u8s32(m, n, k, gemm_backend, &args, alpha, beta, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
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
 