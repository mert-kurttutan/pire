

#![allow(non_camel_case_types)]
#![allow(dead_code)]

#![allow(unused)]

use libc::{c_float, c_int, c_schar, c_void, c_double, c_ushort};

use half::f16;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::enum_variant_names)]
pub enum CBLAS_LAYOUT {
   CblasRowMajor = 101,
   CblasColMajor = 102,
}
pub use self::CBLAS_LAYOUT::*;


#[repr(C)]
pub struct cntx_t(i32);


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




pub fn max_abs_diff<T: Copy + std::ops::Sub + Into<f64> + std::fmt::Debug>(
    ap: &[T], bp: &[T], eps: f64
) -> f64
where f64: From<<T as std::ops::Sub>::Output>
{
   let mut diff = 0_f64;
   let len = ap.len();
   // println!("------------------------------");
   let mut diff_idx = 0;
   for i in 0..len {
       let a = ap[i];
       let b = bp[i];
       let cur_diff_abs: f64 = <<T as std::ops::Sub>::Output as Into<f64>>::into(a-b).abs().into();
       let cur_diff_rel: f64 = cur_diff_abs / b.into().abs();
       let cur_diff = cur_diff_abs.min(cur_diff_rel);
       ;
       if cur_diff > diff {
            diff_idx = i;
           diff = cur_diff;
       }
   }
   diff
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



pub unsafe fn gemm_fallback_f16(
	m: usize, n: usize, k: usize,
	alpha: f16,
	a: *const f16, a_rs: usize, a_cs: usize,
	b: *const f16, b_rs: usize, b_cs: usize,
	beta: f16,
	c: *mut f16, c_rs: usize, c_cs: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut dx = f16::ZERO;
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



pub unsafe fn check_gemm_f64(
	m: usize, n: usize, k: usize,
	alpha: f64,
	a: *const f64, a_rs: usize, a_cs: usize,
	b: *const f64, b_rs: usize, b_cs: usize,
	beta: f64,
	c: &[f64], c_rs: usize, c_cs: usize,
    c_ref: &mut [f64],
    eps: f64,
) -> f64 {
    #[cfg(feature="mkl")] {
        let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        cblas_dgemm(
            layout, transa, transb, m as c_int, n as c_int, k as c_int, alpha, a, lda, b, ldb, beta, c_ref.as_mut_ptr(), ldc
        );
        let diff = max_abs_diff(&c, &c_ref, eps);
        return diff;
    }
    #[cfg(not(feature="mkl"))] {
        // calculate diff using fallback
        gemm_fallback_f64(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c_ref.as_mut_ptr(), c_rs, c_cs);
        let diff = max_abs_diff(&c, &c_ref, eps);
        return diff;
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
    eps: f64,
) -> f64 {
    #[cfg(feature="mkl")] {
        let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        cblas_sgemm(
            layout, transa, transb, m as c_int, n as c_int, k as c_int, alpha, a, lda, b, ldb, beta, c_ref.as_mut_ptr(), ldc
        );
        let diff = max_abs_diff(&c, &c_ref, eps);
        return diff;
    }
    #[cfg(not(feature="mkl"))] {
        // calculate diff using fallback
        gemm_fallback_f32(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c_ref.as_mut_ptr(), c_rs, c_cs);
        let diff = max_abs_diff(&c, &c_ref, eps);
        return diff;
    }

}


pub unsafe fn check_gemm_f16(
	m: usize, n: usize, k: usize,
	alpha: f16,
	a: *const f16, a_rs: usize, a_cs: usize,
	b: *const f16, b_rs: usize, b_cs: usize,
	beta: f16,
	c: &[f16], c_rs: usize, c_cs: usize,
    c_ref: &mut [f16],
    eps: f64,
) -> f64 {
    #[cfg(feature="mkl")] {
        let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        let a = a as *const u16;
        let b = b as *const u16;
        let c_ref_ptr = c_ref.as_mut_ptr() as *mut u16;
        let alpha = alpha.to_bits();
        let beta = beta.to_bits();
        cblas_hgemm(
            layout, transa, transb, m as c_int, n as c_int, k as c_int, alpha, a, lda, b, ldb, beta, c_ref_ptr, ldc
        );
        let diff = max_abs_diff(&c, &c_ref, eps);
        return diff;
    }
    #[cfg(not(feature="mkl"))] {
        // calculate diff using fallback
        gemm_fallback_f16(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c_ref.as_mut_ptr(), c_rs, c_cs);
        let diff = max_abs_diff(&c, &c_ref, eps);
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
 