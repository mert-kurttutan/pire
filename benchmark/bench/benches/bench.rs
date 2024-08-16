use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

use glare_dev::{
    // CBLAS_LAYOUT, 
    // CBLAS_TRANSPOSE,
    CBLAS_TRANSPOSE::*,
    // CBLAS_OFFSET::*,
    CBLAS_LAYOUT::CblasColMajor,
    cblas_sgemm, 
    // cblas_dgemm, 
    // cblas_cgemm, 
    // cblas_hgemm, 
    // cblas_sgemm_batch,
    // cblas_gemm_s16s16s32, 
    // cblas_gemm_s8u8s32,
    // check_gemm_f16, check_gemm_f32, check_gemm_f64, check_gemm_s16s16s32, check_gemm_s8u8s32, 
    //check_gemm_c32,
};

#[cfg(feature="blis")]
use glare_dev::BLIS_NO_TRANSPOSE;

// use libc::{c_int, c_void, c_ushort};

// use num_complex::{
//     c32,
//     Complex32
// };



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
            })
        },
    );

    bench_c.bench_function(
        &format!(
            "f32-glare-gemm-{}×{}×{}", m, n, k
        ),
        |x| {
            x.iter(|| unsafe {
                glare_gemm_f32::glare_sgemm(
                    m, n, k,
                    ALPHA,
                    a, a_rs, a_cs, 
                    b, b_rs, b_cs,
                    BETA,
                    c, c_rs, c_cs,
                );
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
    //                 glare_dev::bli_sgemm(
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
    #[cfg(feature="mkl")]
    bench_c.bench_with_input(
        BenchmarkId::new("f32-mkl-gemm", dt), &dt, 
        |bench_b, _x| bench_b.iter(
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
        |bench_b, _x| bench_b.iter(
            || unsafe {
                glare_gemm_f32::glare_sgemm(
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
        |bench_b, _x| bench_b.iter(
            || unsafe {
                glare_dev::bli_sgemm(
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
        |bench_b, _x| bench_b.iter(
            || unsafe {
                gemm::gemm(
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
    #[cfg(feature="mkl")]
    bench_c.bench_function(
        &format!(
            "f32-mkl-gemm-{}×{}×{}", m, n, k
        ),
        |x| {
            x.iter(|| unsafe {
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
        })
        },
    );

    bench_c.bench_function(
        &format!(
            "f32-glare-gemm-{}×{}×{}", m, n, k
        ),
        |x| {
            x.iter(|| unsafe {
                glare_gemm_f32::glare_sgemm(
                    m, n, k,
                    ALPHA,
                    a, a_rs, a_cs, 
                    b, b_rs, b_cs,
                    BETA,
                    c, c_rs, c_cs,
                );
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
                glare_dev::bli_sgemm(
                    BLIS_NO_TRANSPOSE,
                    BLIS_NO_TRANSPOSE,
                    m as i32, n as i32, k as i32,
                    &ALPHA,
                    a, a_rs as i32, a_cs as i32,
                    b, b_rs as i32, b_cs as i32,
                    &BETA,
                    c, c_rs as i32, c_cs as i32,
                );
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
                gemm::gemm(
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
    let dim_triple = (DimSize::Big, DimSize::Big, DimSize::Big);
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
        // 2400, 3200, 4000, 4800, 
        5600, 6400, 
        // 7200, 8000,
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
