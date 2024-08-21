use glare_dev::{
    random_matrix_std, random_matrix_uniform,
    // CBLAS_LAYOUT, 
    CBLAS_TRANSPOSE,
    check_gemm_f16, check_gemm_f32, check_gemm_f64, check_gemm_s16s16s32, check_gemm_s8u8s32, 
    //check_gemm_c32,
};
#[cfg(feature="mkl")]
use glare_dev::{
    stride_to_cblas, CBLAS_OFFSET::*,
};

#[cfg(feature="mkl")]
use libc::{c_int, c_void, c_ushort};
#[cfg(feature="blis")]
use glare_dev::BLIS_NO_TRANSPOSE;

use num_complex::{
    c32,
    Complex32
};

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
                glare_dev::bli_dgemm(
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
                 glare_dev::cblas_dgemm(
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
                glare_dev::bli_sgemm(
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
                glare_dev::cblas_sgemm(
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
                let a = a as *const libc::c_void;
                let b = b as *const libc::c_void;
                let c = c as *mut libc::c_void;
                let alpha_ptr = &alpha as *const Complex32 as *const libc::c_void;
                let beta_ptr = &beta as *const Complex32 as *const libc::c_void;
                glare_dev::bli_cgemm(
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
                 glare_dev::cblas_cgemm(
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
            // print all inputs
            println!("m, n, k: {}, {}, {}", m, n, k);
            println!("alpha: {}, {}", alpha.re, alpha.im);
            println!("beta: {}, {}", beta.re, beta.im);
            println!("a_rs, a_cs: {}, {}", a_rs, a_cs);
            println!("b_rs, b_cs: {}, {}", b_rs, b_cs);
            println!("c_rs, c_cs: {}, {}", c_rs, c_cs);
            println!("a: {:?}", a);
            println!("b: {:?}", b);
            println!("c: {:?}", c);
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
                    glare_dev::bli_sgemm(
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

                glare_dev::cblas_sgemm_batch(
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
                 glare_dev::cblas_hgemm(
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
                 glare_dev::cblas_gemm_s16s16s32(
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
                 glare_dev::cblas_gemm_s8u8s32(
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
            check_gemm_f64(
                m, n, k, 
                alpha, 
                a.as_ptr(), a_rs as usize, a_cs as usize, 
                b.as_ptr(), b_rs as usize, b_cs as usize, 
                beta, 
                &c, c_rs as usize, c_cs as usize, 
                &mut c_ref,
                1e-3,
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
            check_gemm_f32(
                m, n, k, 
                alpha, 
                a.as_ptr(), a_rs as usize, a_cs as usize, 
                b.as_ptr(), b_rs as usize, b_cs as usize, 
                beta, 
                &c, c_rs as usize, c_cs as usize, 
                &mut c_ref,
                1e-3,
            )
        };
        println!("diff: {}", diff);
    }

    let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;

    end_time
}

fn test_cgemm(
    m: usize, n: usize, k: usize,
    gemm_backend: GemmBackend, _args: &Args,
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
    // if args.check {
    //     let diff = unsafe {
    //         check_gemm_c32(
    //             m, n, k, 
    //             alpha, 
    //             a.as_ptr(), a_rs as usize, a_cs as usize, 
    //             b.as_ptr(), b_rs as usize, b_cs as usize, 
    //             beta, 
    //             &c, c_rs as usize, c_cs as usize, 
    //             &mut c_ref,
    //             1e-3
    //         )
    //     };
    //     println!("diff: {}", diff);
    // }

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
            check_gemm_f32(
                m, n, k, 
                alpha, 
                a.as_ptr(), a_rs as usize, a_cs as usize, 
                b.as_ptr(), b_rs as usize, b_cs as usize, 
                beta, 
                &c, c_rs as usize, c_cs as usize, 
                &mut c_ref,
                1e-3,
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
            check_gemm_f16(
                m, n, k, 
                alpha, 
                a.as_ptr(), a_rs as usize, a_cs as usize, 
                b.as_ptr(), b_rs as usize, b_cs as usize, 
                beta, 
                &c, c_rs as usize, c_cs as usize, 
                &mut c_ref,
                1e-1,
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
                &mut c_ref,
                1e-3
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
                &mut c_ref,
                1e-3,
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
 