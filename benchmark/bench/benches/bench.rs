use criterion::{criterion_group, criterion_main, Criterion};
use glare_gemm_f32::*;
use std::{thread::available_parallelism, time::Duration};

#[cfg(feature="rustgemm")]
use gemm::*;

// use bench::*;


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

//    pub fn cblas_hgemm(
//         layout: CBLAS_LAYOUT,
//         transa: CBLAS_TRANSPOSE,
//         transb: CBLAS_TRANSPOSE,
//         m: c_int,
//         n: c_int,
//         k: c_int,
//         alpha: c_ushort,
//         a: *const c_ushort,
//         lda: c_int,
//         b: *const c_ushort,
//         ldb: c_int,
//         beta: c_ushort,
//         c: *mut c_ushort,
//         ldc: c_int,
//     );


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

// #[inline(never)]
// unsafe fn gemm_fallback<T>(
//     m: usize,
//     n: usize,
//     k: usize,
//     dst: *mut T,
//     dst_cs: isize,
//     dst_rs: isize,
//     read_dst: bool,
//     lhs: *const T,
//     lhs_cs: isize,
//     lhs_rs: isize,
//     rhs: *const T,
//     rhs_cs: isize,
//     rhs_rs: isize,
//     alpha: T,
//     beta: T,
// ) where
//     T: num_traits::Zero + Send + Sync,
//     for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
//     for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
// {
//     (0..m).for_each(|row| {
//         (0..n).for_each(|col| {
//             let mut accum = <T as num_traits::Zero>::zero();
//             for depth in 0..k {
//                 let lhs = &*lhs.wrapping_offset(row as isize * lhs_rs + depth as isize * lhs_cs);

//                 let rhs = &*rhs.wrapping_offset(depth as isize * rhs_rs + col as isize * rhs_cs);

//                 accum = &accum + &(lhs * rhs);
//             }
//             accum = &accum * &beta;

//             let dst = dst.wrapping_offset(row as isize * dst_rs + col as isize * dst_cs);
//             if read_dst {
//                 accum = &accum + &(&alpha * &*dst);
//             }
//             *dst = accum
//         });
//     });
//     return;
// }

const ALPHA: f32 = 1.0;
const BETA: f32 = 1.0;

#[derive(Clone, Copy, Debug)]
pub enum DimSize{
    Big,
    Small,
}
fn get_mnk(dim_triple: (DimSize, DimSize, DimSize), d0: usize, dt: usize) -> (usize, usize, usize) {
    let m = match dim_triple.0 {
        DimSize::Small => d0,
        DimSize::Big => dt,
    };
    let n = match dim_triple.1 {
        DimSize::Small => d0,
        DimSize::Big => dt,
    };
    let k = match dim_triple.2 {
        DimSize::Small => d0,
        DimSize::Big => dt,
    };
    (m, n, k)

}

fn benchmark_dim(bench_type: (DimSize, DimSize, DimSize), d0: usize, d1: usize, step: usize) -> Vec<(usize, usize, usize)> {
    let mut dims = vec![];
    let small_dim = 50;
    for i in (d0..d1).step_by(step) {
        let mnk = {
            let m = match bench_type.0 {
                DimSize::Big => i,
                DimSize::Small => small_dim,
            };
            let n = match bench_type.1 {
                DimSize::Big => i,
                DimSize::Small => small_dim,
            };
            let k = match bench_type.2 {
                DimSize::Big => i,
                DimSize::Small => small_dim,
            };
            (m, n, k)
        };
        dims.push(mnk);
    }
    dims
}

pub fn bench_blas_group<M: criterion::measurement::Measurement>(
    bench_c: &mut BenchmarkGroup<M>, 
    m: usize, n: usize, k: usize,
    a: *const f32, a_rs: usize, a_cs: usize,
    b: *const f32, b_rs: usize, b_cs: usize,
    c: *mut f32, c_rs: usize, c_cs: usize,
) {
    bench_c.bench_function(
        &format!(
            "f32-mkl-gemm-{}×{}×{}", m, n, k
        ),
        |x| {
            x.iter(|| unsafe {
                unsafe { 
                    cblas_sgemm(
                        CblasColMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m as i32, n as i32, k as i32,
                        ALPHA,
                        a, m as i32,
                        b, k as i32,
                        BETA,
                        c, m as i32,
                    );
                }
            })
        },
    );

    bench_c.bench_function(
        &format!(
            "f32-glare-gemm-{}×{}×{}", m, n, k
        ),
        |x| {
            x.iter(|| unsafe {

                unsafe {

                    glare_sgemm(
                        m, n, k,
                        ALPHA,
                        a, a_rs, a_cs, 
                        b, b_rs, b_cs,
                        BETA,
                        c, c_rs, c_cs,
                    );
                }
            })
        },
    );

    // bench_c.bench_function(
    //     &format!(
    //         "f32-blis-gemm-{}×{}×{}", m, n, k
    //     ),
    //     |x| {
    //         x.iter(|| unsafe {
    //             unsafe {
    //                 bli_sgemm(
    //                     BLIS_NO_TRANSPOSE,
    //                     BLIS_NO_TRANSPOSE,
    //                     m as i32, n as i32, k as i32,
    //                     &ALPHA,
    //                     a, a_rs as i32, a_cs as i32,
    //                     b, b_rs as i32, b_cs as i32,
    //                     &BETA,
    //                     c, c_rs as i32, c_cs as i32,
    //                 );
    //             }
    //         })
    //     },
    // );

    // bench_c.bench_function(
    //     &format!(
    //         "f32-rust-gemm-{}×{}×{}", m, n, k
    //     ),
    //     |x| {
    //         x.iter(|| unsafe {
    //             unsafe {
    //                 gemm(
    //                     m, n, k,
    //                     c, c_cs as isize, c_rs as isize,
    //                     true,
    //                     a, a_cs as isize, a_rs as isize,
    //                     b, b_cs as isize, b_rs as isize,
    //                     0.0_f32,
    //                     0.0_f32,
    //                     false, false, false,
    //                     gemm::Parallelism::Rayon(0),
    //                 )
    //             }
    //         })
    //     },
    // );
}


use criterion::BenchmarkId;

pub fn bench_blas_group3<M: criterion::measurement::Measurement>(
    bench_c: &mut BenchmarkGroup<M>, 
    dim_triple: (DimSize, DimSize, DimSize),
    d0: usize, dt: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
) {
    let (m, n, k) = get_mnk(dim_triple, d0, dt);
    bench_c.bench_with_input(
        BenchmarkId::new("f32-mkl-gemm", dt), &dt, 
        |bench_b, x| bench_b.iter(
            || unsafe {
                cblas_sgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m as i32, n as i32, k as i32,
                    ALPHA,
                    a, m as i32,
                    b, k as i32,
                    BETA,
                    c, m as i32,
                );
            }
        )
    );
    bench_c.bench_with_input(
        BenchmarkId::new("f32-glare-gemm", dt), &dt, 
        |bench_b, x| bench_b.iter(
            || unsafe {
                glare_sgemm(
                    m, n, k,
                    ALPHA,
                    a, 1, m,
                    b, 1, k,
                    BETA,
                    c, 1, m,
                );
            }
        )
    );

    #[cfg(feature="blis")]
    bench_c.bench_with_input(
        BenchmarkId::new("f32-blis-gemm", dt), &dt, 
        |bench_b, x| bench_b.iter(
            || unsafe {
                bli_sgemm(
                    BLIS_NO_TRANSPOSE,
                    BLIS_NO_TRANSPOSE,
                    m as i32, n as i32, k as i32,
                    &ALPHA,
                    a, 1, m as i32,
                    b, 1, k as i32,
                    &BETA,
                    c, 1, m as i32,
                );
            }
        )
    );

    #[cfg(feature="rustgemm")]
    bench_c.bench_with_input(
        BenchmarkId::new("f32-rust-gemm", dt), &dt, 
        |bench_b, x| bench_b.iter(
            || unsafe {
                gemm(
                    m, n, k,
                    c, m as isize, 1,
                    true,
                    a, m as isize, 1,
                    b, k as isize, 1,
                    0.0_f32,
                    0.0_f32,
                    false, false, false,
                    gemm::Parallelism::Rayon(0),
                )
            }
        )
    );

}

pub fn bench_blas_group2(
    bench_c: &mut Criterion, 
    m: usize, n: usize, k: usize,
    a: *const f32, a_rs: usize, a_cs: usize,
    b: *const f32, b_rs: usize, b_cs: usize,
    c: *mut f32, c_rs: usize, c_cs: usize,
) {
    bench_c.bench_function(
        &format!(
            "f32-mkl-gemm-{}×{}×{}", m, n, k
        ),
        |x| {
            x.iter(|| unsafe {
                unsafe { 
                    cblas_sgemm(
                        CblasColMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m as i32, n as i32, k as i32,
                        ALPHA,
                        a, m as i32,
                        b, k as i32,
                        BETA,
                        c, m as i32,
                    );
                }
            })
        },
    );

    bench_c.bench_function(
        &format!(
            "f32-glare-gemm-{}×{}×{}", m, n, k
        ),
        |x| {
            x.iter(|| unsafe {

                unsafe {

                    glare_sgemm(
                        m, n, k,
                        ALPHA,
                        a, a_rs, a_cs, 
                        b, b_rs, b_cs,
                        BETA,
                        c, c_rs, c_cs,
                    );
                }
            })
        },
    );
    #[cfg(feature="blis")]
    bench_c.bench_function(
        &format!(
            "f32-blis-gemm-{}×{}×{}", m, n, k
        ),
        |x| {
            x.iter(|| unsafe {
                unsafe {
                    bli_sgemm(
                        BLIS_NO_TRANSPOSE,
                        BLIS_NO_TRANSPOSE,
                        m as i32, n as i32, k as i32,
                        &ALPHA,
                        a, a_rs as i32, a_cs as i32,
                        b, b_rs as i32, b_cs as i32,
                        &BETA,
                        c, c_rs as i32, c_cs as i32,
                    );
                }
            })
        },
    );
    #[cfg(feature="rustgemm")]
    bench_c.bench_function(
        &format!(
            "f32-rust-gemm-{}×{}×{}", m, n, k
        ),
        |x| {
            x.iter(|| unsafe {
                unsafe {
                    gemm(
                        m, n, k,
                        c, c_cs as isize, c_rs as isize,
                        true,
                        a, a_cs as isize, a_rs as isize,
                        b, b_cs as isize, b_rs as isize,
                        0.0_f32,
                        0.0_f32,
                        false, false, false,
                        gemm::Parallelism::Rayon(0),
                    )
                }
            })
        },
    );
}


// pub fn criterion_benchmark(c: &mut Criterion) {
//     {
//         let mnk_vec = benchmark_dim((DimSize::Big, DimSize::Big, DimSize::Big), 1000, 5000, 400);
//         let mut group = c.benchmark_group("bbb");

//         for (m, n, k) in mnk_vec {
//             let a = vec![0.0_f32; m * k];
//             let b_vec = vec![0.0_f32; k * n];
//             let mut c_vec = vec![0.0_f32; m * n];
//             bench_blas_group(&mut group, *m, *n, *k, a.as_ptr(), *m, 1, b_vec.as_ptr(), *k, 1, c_vec.as_mut_ptr(), *m, 1);

//         }
    
//     }
// }



use criterion::BenchmarkGroup;
fn bench_bbb(c: &mut Criterion) {
    let mut group = c.benchmark_group("bbb");
    let dim_triple = (DimSize::Big, DimSize::Small, DimSize::Big);
    let m1 = 8000;
    let n1 = 8000;
    let k1 = 8000;
    let a = vec![0.0_f32; m1 * k1];
    let b_vec = vec![0.0_f32; k1 * n1];
    let mut c_vec = vec![0.0_f32; m1 * n1];
    let d0 = 1;
    let mnk_vec = vec![
        // 10, 100, 
        // 128,
        // 256,
        // 320, 640, 960, 2048,
        // 2400, 3200, 4000, 4800, 5600, 6400, 
        7200, 8000,
    ];
    for dt in mnk_vec {
        bench_blas_group3(&mut group, dim_triple, d0, dt, a.as_ptr(), b_vec.as_ptr(), c_vec.as_mut_ptr());
        // bench_blas_group2(c, *m, *n, *k, a.as_ptr(), 1, *m, b_vec.as_ptr(), 1, *k, c_vec.as_mut_ptr(), 1, *m);
        // group.bench_with_input(BenchmarkId::new("Recursive", i), i, 
        //     |b, i| b.iter(|| fibonacci_slow(*i)));
        // group.bench_with_input(BenchmarkId::new("Iterative", i), i, 
        //     |b, i| b.iter(|| fibonacci_fast(*i)));
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(10);
    targets = bench_bbb
);
criterion_main!(benches);
