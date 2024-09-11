use half::f16;
use num_complex::Complex;

type C32 = Complex<f32>;

#[cfg(feature = "mkl")]
use glare_dev::{stride_to_cblas, CBLAS_OFFSET::*};

#[cfg(feature = "blis")]
use glare_dev::BLIS_NO_TRANSPOSE;
#[cfg(feature = "mkl")]
use libc::{c_int, c_ushort, c_void};

use num_complex::{Complex32, Complex64};
#[derive(Copy, Clone, Debug)]
pub enum BenchType {
    DGemm,
    SGemm,
    SGemmBatched,
    HGemm,
    CGemm,
    ZGemm,
    GemmS16S16S32,
    GemmS8U8S32,
}

#[derive(Copy, Clone, Debug)]
pub enum GemmBackend {
    Blis,
    Mkl,
    RustGemm,
    Glare,
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
        return GemmBackend::Glare;
    }
    panic!("Unsupported backend str");
}

pub unsafe fn dispatch_dgemm(
    backend: GemmBackend,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    a_rs: isize,
    a_cs: isize,
    b: *const f64,
    b_rs: isize,
    b_cs: isize,
    beta: f64,
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
) {
    match backend {
        GemmBackend::Blis => {
            #[cfg(feature = "blis")]
            glare_dev::bli_dgemm(
                BLIS_NO_TRANSPOSE,
                BLIS_NO_TRANSPOSE,
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a,
                a_rs as i32,
                a_cs as i32,
                b,
                b_rs as i32,
                b_cs as i32,
                &beta,
                c,
                c_rs as i32,
                c_cs as i32,
            );
        }
        GemmBackend::Mkl => {
            #[cfg(feature = "mkl")]
            {
                let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(
                    m,
                    n,
                    k,
                    a_rs as usize,
                    a_cs as usize,
                    b_rs as usize,
                    b_cs as usize,
                    c_rs as usize,
                    c_cs as usize,
                );
                glare_dev::cblas_dgemm(
                    layout, transa, transb, m as i32, n as i32, k as i32, alpha, a, lda as i32, b,
                    ldb as i32, beta, c, ldc as i32,
                );
            }
        }
        GemmBackend::RustGemm => {
            #[cfg(feature = "rustgemm")]
            gemm::gemm(
                m,
                n,
                k,
                c,
                c_cs as isize,
                c_rs as isize,
                true,
                a,
                a_cs as isize,
                a_rs as isize,
                b,
                b_cs as isize,
                b_rs as isize,
                alpha,
                beta,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
        GemmBackend::Glare => {
            glare_gemm_f64::glare_dgemm(
                m,
                n,
                k,
                alpha,
                a,
                a_rs as usize,
                a_cs as usize,
                b,
                b_rs as usize,
                b_cs as usize,
                beta,
                c,
                c_rs as usize,
                c_cs as usize,
            );
        }
    }
}

pub unsafe fn dispatch_sgemm(
    backend: GemmBackend,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    a_rs: isize,
    a_cs: isize,
    b: *const f32,
    b_rs: isize,
    b_cs: isize,
    beta: f32,
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
) {
    match backend {
        GemmBackend::Blis => {
            #[cfg(feature = "blis")]
            glare_dev::bli_sgemm(
                BLIS_NO_TRANSPOSE,
                BLIS_NO_TRANSPOSE,
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a,
                a_rs as i32,
                a_cs as i32,
                b,
                b_rs as i32,
                b_cs as i32,
                &beta,
                c,
                c_rs as i32,
                c_cs as i32,
            );
        }
        GemmBackend::Mkl => {
            #[cfg(feature = "mkl")]
            {
                let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(
                    m,
                    n,
                    k,
                    a_rs as usize,
                    a_cs as usize,
                    b_rs as usize,
                    b_cs as usize,
                    c_rs as usize,
                    c_cs as usize,
                );
                glare_dev::cblas_sgemm(
                    layout, transa, transb, m as i32, n as i32, k as i32, alpha, a, lda as i32, b,
                    ldb as i32, beta, c, ldc as i32,
                );
            }
        }
        GemmBackend::RustGemm => {
            #[cfg(feature = "rustgemm")]
            gemm::gemm(
                m,
                n,
                k,
                c,
                c_cs as isize,
                c_rs as isize,
                true,
                a,
                a_cs as isize,
                a_rs as isize,
                b,
                b_cs as isize,
                b_rs as isize,
                alpha,
                beta,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
        GemmBackend::Glare => {
            glare_gemm_f32::glare_sgemm(
                m,
                n,
                k,
                alpha,
                a,
                a_rs as usize,
                a_cs as usize,
                b,
                b_rs as usize,
                b_cs as usize,
                beta,
                c,
                c_rs as usize,
                c_cs as usize,
            );
        }
    }
}

pub unsafe fn dispatch_cgemm(
    backend: GemmBackend,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: *const Complex32,
    a_rs: isize,
    a_cs: isize,
    b: *const Complex32,
    b_rs: isize,
    b_cs: isize,
    beta: Complex32,
    c: *mut Complex32,
    c_rs: isize,
    c_cs: isize,
) {
    match backend {
        GemmBackend::Blis => {
            #[cfg(feature = "blis")]
            {
                let a = a as *const libc::c_void;
                let b = b as *const libc::c_void;
                let c = c as *mut libc::c_void;
                let alpha_ptr = &alpha as *const Complex32 as *const libc::c_void;
                let beta_ptr = &beta as *const Complex32 as *const libc::c_void;
                glare_dev::bli_cgemm(
                    BLIS_NO_TRANSPOSE,
                    BLIS_NO_TRANSPOSE,
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha_ptr,
                    a,
                    a_rs as i32,
                    a_cs as i32,
                    b,
                    b_rs as i32,
                    b_cs as i32,
                    beta_ptr,
                    c,
                    c_rs as i32,
                    c_cs as i32,
                );
            }
        }
        GemmBackend::Mkl => {
            #[cfg(feature = "mkl")]
            {
                let a = a as *const c_void;
                let b = b as *const c_void;
                let c = c as *mut c_void;
                let alpha_ptr = &alpha as *const Complex32 as *const c_void;
                let beta_ptr = &beta as *const Complex32 as *const c_void;
                let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(
                    m,
                    n,
                    k,
                    a_rs as usize,
                    a_cs as usize,
                    b_rs as usize,
                    b_cs as usize,
                    c_rs as usize,
                    c_cs as usize,
                );
                glare_dev::cblas_cgemm(
                    layout, transa, transb, m as i32, n as i32, k as i32, alpha_ptr, a, lda as i32,
                    b, ldb as i32, beta_ptr, c, ldc as i32,
                );
            }
        }
        GemmBackend::RustGemm => {
            #[cfg(feature = "rustgemm")]
            gemm::gemm(
                m,
                n,
                k,
                c,
                c_cs as isize,
                c_rs as isize,
                true,
                a,
                a_cs as isize,
                a_rs as isize,
                b,
                b_cs as isize,
                b_rs as isize,
                alpha,
                beta,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
        GemmBackend::Glare => {
            glare_gemm_c32::glare_cgemm(
                m,
                n,
                k,
                alpha,
                a,
                a_rs as usize,
                a_cs as usize,
                b,
                b_rs as usize,
                b_cs as usize,
                beta,
                c,
                c_rs as usize,
                c_cs as usize,
            );
        }
    }
}

pub unsafe fn dispatch_zgemm(
    backend: GemmBackend,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: *const Complex64,
    a_rs: isize,
    a_cs: isize,
    b: *const Complex64,
    b_rs: isize,
    b_cs: isize,
    beta: Complex64,
    c: *mut Complex64,
    c_rs: isize,
    c_cs: isize,
) {
    match backend {
        GemmBackend::Blis => {
            #[cfg(feature = "blis")]
            {
                let a = a as *const libc::c_void;
                let b = b as *const libc::c_void;
                let c = c as *mut libc::c_void;
                let alpha_ptr = &alpha as *const Complex64 as *const libc::c_void;
                let beta_ptr = &beta as *const Complex64 as *const libc::c_void;
                glare_dev::bli_zgemm(
                    BLIS_NO_TRANSPOSE,
                    BLIS_NO_TRANSPOSE,
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha_ptr,
                    a,
                    a_rs as i32,
                    a_cs as i32,
                    b,
                    b_rs as i32,
                    b_cs as i32,
                    beta_ptr,
                    c,
                    c_rs as i32,
                    c_cs as i32,
                );
            }
        }
        GemmBackend::Mkl => {
            #[cfg(feature = "mkl")]
            {
                let a = a as *const c_void;
                let b = b as *const c_void;
                let c = c as *mut c_void;
                let alpha_ptr = &alpha as *const Complex64 as *const c_void;
                let beta_ptr = &beta as *const Complex64 as *const c_void;
                let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(
                    m,
                    n,
                    k,
                    a_rs as usize,
                    a_cs as usize,
                    b_rs as usize,
                    b_cs as usize,
                    c_rs as usize,
                    c_cs as usize,
                );
                glare_dev::cblas_zgemm(
                    layout, transa, transb, m as i32, n as i32, k as i32, alpha_ptr, a, lda as i32,
                    b, ldb as i32, beta_ptr, c, ldc as i32,
                );
            }
        }
        GemmBackend::RustGemm => {
            #[cfg(feature = "rustgemm")]
            gemm::gemm(
                m,
                n,
                k,
                c,
                c_cs as isize,
                c_rs as isize,
                true,
                a,
                a_cs as isize,
                a_rs as isize,
                b,
                b_cs as isize,
                b_rs as isize,
                alpha,
                beta,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
        GemmBackend::Glare => {
            glare_gemm_c64::glare_zgemm(
                m,
                n,
                k,
                alpha,
                a,
                a_rs as usize,
                a_cs as usize,
                b,
                b_rs as usize,
                b_cs as usize,
                beta,
                c,
                c_rs as usize,
                c_cs as usize,
            );
        }
    }
}

pub unsafe fn dispatch_gemm_batch_f32(
    backend: GemmBackend,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    a_rs: isize,
    a_cs: isize,
    stridea: isize,
    b: *const f32,
    b_rs: isize,
    b_cs: isize,
    strideb: isize,
    beta: f32,
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
    stridec: isize,
    batch_size: usize,
) {
    match backend {
        GemmBackend::Blis => {
            #[cfg(feature = "blis")]
            {
                for i in 0..batch_size {
                    glare_dev::bli_sgemm(
                        BLIS_NO_TRANSPOSE,
                        BLIS_NO_TRANSPOSE,
                        m as i32,
                        n as i32,
                        k as i32,
                        &alpha,
                        a.offset(i as isize * stridea),
                        a_rs as i32,
                        a_cs as i32,
                        b.offset(i as isize * strideb),
                        b_rs as i32,
                        b_cs as i32,
                        &beta,
                        c.offset(i as isize * stridec),
                        c_rs as i32,
                        c_cs as i32,
                    );
                }
            }
        }
        GemmBackend::Mkl => {
            #[cfg(feature = "mkl")]
            {
                let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(
                    m,
                    n,
                    k,
                    a_rs as usize,
                    a_cs as usize,
                    b_rs as usize,
                    b_cs as usize,
                    c_rs as usize,
                    c_cs as usize,
                );
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
                let a_vec = (0..batch_size)
                    .map(|i| a.offset(i as isize * stridea))
                    .collect::<Vec<*const f32>>();
                let b_vec = (0..batch_size)
                    .map(|i| b.offset(i as isize * strideb))
                    .collect::<Vec<*const f32>>();
                let c_vec = (0..batch_size)
                    .map(|i| c.offset(i as isize * stridec))
                    .collect::<Vec<*mut f32>>();
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
            #[cfg(feature = "rustgemm")]
            {
                for i in 0..batch_size {
                    gemm::gemm(
                        m,
                        n,
                        k,
                        c.offset(i as isize * stridec),
                        c_cs as isize,
                        c_rs as isize,
                        true,
                        a.offset(i as isize * stridea),
                        a_cs as isize,
                        a_rs as isize,
                        b.offset(i as isize * strideb),
                        b_cs as isize,
                        b_rs as isize,
                        alpha,
                        beta,
                        false,
                        false,
                        false,
                        gemm::Parallelism::Rayon(0),
                    );
                }
            }
        }
        GemmBackend::Glare => {
            for i in 0..batch_size {
                glare_gemm_f32::glare_sgemm(
                    m,
                    n,
                    k,
                    alpha,
                    a.offset(i as isize * stridea),
                    a_rs as usize,
                    a_cs as usize,
                    b.offset(i as isize * strideb),
                    b_rs as usize,
                    b_cs as usize,
                    beta,
                    c.offset(i as isize * stridec),
                    c_rs as usize,
                    c_cs as usize,
                );
            }
        }
    }
}

pub unsafe fn dispatch_hgemm(
    backend: GemmBackend,
    m: usize,
    n: usize,
    k: usize,
    alpha: f16,
    a: *const f16,
    a_rs: isize,
    a_cs: isize,
    b: *const f16,
    b_rs: isize,
    b_cs: isize,
    beta: f16,
    c: *mut f16,
    c_rs: isize,
    c_cs: isize,
) {
    match backend {
        GemmBackend::Blis => {
            #[cfg(feature = "blis")]
            panic!("f16 not supported in blis");
        }
        GemmBackend::Mkl => {
            #[cfg(feature = "mkl")]
            {
                let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(
                    m,
                    n,
                    k,
                    a_rs as usize,
                    a_cs as usize,
                    b_rs as usize,
                    b_cs as usize,
                    c_rs as usize,
                    c_cs as usize,
                );
                let a = a as *const c_ushort;
                let b = b as *const c_ushort;
                let c = c as *mut c_ushort;
                let alpha = alpha.to_bits();
                let beta = beta.to_bits();
                glare_dev::cblas_hgemm(
                    layout, transa, transb, m as i32, n as i32, k as i32, alpha, a, lda as i32, b,
                    ldb as i32, beta, c, ldc as i32,
                );
            }
        }
        GemmBackend::RustGemm => {
            #[cfg(feature = "rustgemm")]
            gemm::gemm(
                m,
                n,
                k,
                c,
                c_cs as isize,
                c_rs as isize,
                true,
                a,
                a_cs as isize,
                a_rs as isize,
                b,
                b_cs as isize,
                b_rs as isize,
                alpha,
                beta,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
        GemmBackend::Glare => {
            glare_gemm_f16::glare_hgemm(
                m,
                n,
                k,
                alpha,
                a,
                a_rs as usize,
                a_cs as usize,
                b,
                b_rs as usize,
                b_cs as usize,
                beta,
                c,
                c_rs as usize,
                c_cs as usize,
            );
        }
    }
}

pub unsafe fn dispatch_gemm_s16s16s32(
    backend: GemmBackend,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const i16,
    a_rs: isize,
    a_cs: isize,
    b: *const i16,
    b_rs: isize,
    b_cs: isize,
    beta: f32,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
) {
    match backend {
        GemmBackend::Blis => {
            #[cfg(feature = "blis")]
            panic!("s16s16s32 is not supported in blis");
        }
        GemmBackend::Mkl => {
            #[cfg(feature = "mkl")]
            {
                let oc_val = 0;
                let oc = &oc_val as *const c_int;
                let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(
                    m,
                    n,
                    k,
                    a_rs as usize,
                    a_cs as usize,
                    b_rs as usize,
                    b_cs as usize,
                    c_rs as usize,
                    c_cs as usize,
                );
                glare_dev::cblas_gemm_s16s16s32(
                    layout,
                    transa,
                    transb,
                    CblasFixOffset,
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha,
                    a,
                    lda as i32,
                    0,
                    b,
                    ldb as i32,
                    0,
                    beta,
                    c,
                    ldc as i32,
                    oc,
                );
            }
        }
        GemmBackend::RustGemm => {
            #[cfg(feature = "rustgemm")]
            panic!("s16s16s32 is not supported in rustgemm");
        }
        GemmBackend::Glare => {
            use glare_gemm_s16s16s32::glare_gemm_s16s16s32 as gemm_s16s16s32;
            gemm_s16s16s32(
                m,
                n,
                k,
                alpha,
                a,
                a_rs as usize,
                a_cs as usize,
                b,
                b_rs as usize,
                b_cs as usize,
                beta,
                c,
                c_rs as usize,
                c_cs as usize,
            );
        }
    }
}

pub unsafe fn dispatch_gemm_s8u8s32(
    backend: GemmBackend,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const i8,
    a_rs: isize,
    a_cs: isize,
    b: *const u8,
    b_rs: isize,
    b_cs: isize,
    beta: f32,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
) {
    match backend {
        GemmBackend::Blis => {
            #[cfg(feature = "blis")]
            panic!("s16s16s32 is not supported in blis");
        }
        GemmBackend::Mkl => {
            #[cfg(feature = "mkl")]
            {
                let oc_val = 0;
                let oc = &oc_val as *const c_int;
                let a = a as *const c_void;
                let b = b as *const c_void;
                let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(
                    m,
                    n,
                    k,
                    a_rs as usize,
                    a_cs as usize,
                    b_rs as usize,
                    b_cs as usize,
                    c_rs as usize,
                    c_cs as usize,
                );
                glare_dev::cblas_gemm_s8u8s32(
                    layout,
                    transa,
                    transb,
                    CblasFixOffset,
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha,
                    a,
                    lda as i32,
                    0,
                    b,
                    ldb as i32,
                    0,
                    beta,
                    c,
                    ldc as i32,
                    oc,
                );
            }
        }
        GemmBackend::RustGemm => {
            #[cfg(feature = "rustgemm")]
            panic!("s16s16s32 is not supported in rustgemm");
        }
        GemmBackend::Glare => {
            use glare_gemm_s8u8s32::glare_gemm_s8u8s32 as gemm_s8u8s32;
            gemm_s8u8s32(
                m,
                n,
                k,
                alpha,
                a,
                a_rs as usize,
                a_cs as usize,
                b,
                b_rs as usize,
                b_cs as usize,
                beta,
                c,
                c_rs as usize,
                c_cs as usize,
            );
        }
    }
}

pub trait AS: 'static {
    type ASType: Copy;
    type BSType: Copy;
}

impl AS for f32 {
    type ASType = f32;
    type BSType = f32;
}

impl AS for f64 {
    type ASType = f64;
    type BSType = f64;
}

impl AS for f16 {
    type ASType = f16;
    type BSType = f16;
}

impl AS for i16 {
    type ASType = f32;
    type BSType = f32;
}

impl AS for i8 {
    type ASType = f32;
    type BSType = f32;
}

impl AS for Complex32 {
    type ASType = Complex32;
    type BSType = Complex32;
}

impl AS for Complex64 {
    type ASType = Complex64;
    type BSType = Complex64;
}

use std::any::TypeId;

// more ergonomic to use in bench
pub unsafe fn dispatch_gemm<TA: AS, TB: 'static, TC: 'static>(
    backend: GemmBackend,
    m: usize,
    n: usize,
    k: usize,
    alpha: TA::ASType,
    a: *const TA,
    a_rs: isize,
    a_cs: isize,
    b: *const TB,
    b_rs: isize,
    b_cs: isize,
    beta: TA::BSType,
    c: *mut TC,
    c_rs: isize,
    c_cs: isize,
) {
    if TypeId::of::<TA>() == TypeId::of::<f32>() {
        dispatch_sgemm(
            backend,
            m,
            n,
            k,
            *(&alpha as *const TA::ASType as *const f32),
            a as *const f32,
            a_rs,
            a_cs,
            b as *const f32,
            b_rs,
            b_cs,
            *(&beta as *const TA::BSType as *const f32),
            c as *mut f32,
            c_rs,
            c_cs,
        )
    } else if TypeId::of::<TA>() == TypeId::of::<f64>() {
        dispatch_dgemm(
            backend,
            m,
            n,
            k,
            *(&alpha as *const TA::ASType as *const f64),
            a as *const f64,
            a_rs,
            a_cs,
            b as *const f64,
            b_rs,
            b_cs,
            *(&beta as *const TA::BSType as *const f64),
            c as *mut f64,
            c_rs,
            c_cs,
        )
    } else if TypeId::of::<TA>() == TypeId::of::<f16>() {
        dispatch_hgemm(
            backend,
            m,
            n,
            k,
            *(&alpha as *const TA::ASType as *const f16),
            a as *const f16,
            a_rs,
            a_cs,
            b as *const f16,
            b_rs,
            b_cs,
            *(&beta as *const TA::BSType as *const f16),
            c as *mut f16,
            c_rs,
            c_cs,
        )
    } else if TypeId::of::<TA>() == TypeId::of::<C32>() {
        dispatch_cgemm(
            backend,
            m,
            n,
            k,
            *(&alpha as *const TA::ASType as *const Complex32),
            a as *const Complex32,
            a_rs,
            a_cs,
            b as *const Complex32,
            b_rs,
            b_cs,
            *(&beta as *const TA::BSType as *const Complex32),
            c as *mut Complex32,
            c_rs,
            c_cs,
        )
    } else if TypeId::of::<TA>() == TypeId::of::<Complex64>() {
        dispatch_zgemm(
            backend,
            m,
            n,
            k,
            *(&alpha as *const TA::ASType as *const Complex64),
            a as *const Complex64,
            a_rs,
            a_cs,
            b as *const Complex64,
            b_rs,
            b_cs,
            *(&beta as *const TA::BSType as *const Complex64),
            c as *mut Complex64,
            c_rs,
            c_cs,
        )
    }
}
