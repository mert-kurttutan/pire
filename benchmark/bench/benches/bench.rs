use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

#[cfg(feature="blis")]
use glare_dev::BLIS_NO_TRANSPOSE;

// use num_complex::{
//     c32,
//     Complex32
// };

use bench::{
    dispatch_gemm,
    GemmBackend,
    AS,
};

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

use criterion::BenchmarkId;
use std::any::type_name;

pub fn bench_blas_group3<M: criterion::measurement::Measurement,TA:AS, TB:'static, TC:'static>(
    bench_c: &mut BenchmarkGroup<M>, 
    dim_triple: (DimSize, DimSize, DimSize),
    d0: usize, dt: usize,
    alpha: TA::ASType,
    a: *const TA,
    b: *const TB,
    beta: TA::BSType,
    c: *mut TC,
) {
    let (m, n, k) = get_mnk(dim_triple, d0, dt);
    let (a_rs, a_cs) = (1, m as isize);
    let (b_rs, b_cs) = (1, k as isize);
    let (c_rs, c_cs) = (1, m as isize);
    let type_name = type_name::<TA>();
    #[cfg(feature="mkl")]
    bench_c.bench_with_input(
        BenchmarkId::new(format!("{}-mkl-gemm", type_name), dt), &dt,
        |bench_b, _x| bench_b.iter(
            || unsafe {
                dispatch_gemm(
                    GemmBackend::Mkl,
                    m, n, k,
                    alpha,
                    a, a_rs, a_cs,
                    b, b_rs, b_cs,
                    beta,
                    c, c_rs, c_cs,
                );
            }
        )
    );
    bench_c.bench_with_input(
        BenchmarkId::new(format!("{}-glare-gemm", type_name), dt), &dt,
        |bench_b, _x| bench_b.iter(
            || unsafe {
                dispatch_gemm(
                    GemmBackend::Glare,
                    m, n, k,
                    alpha,
                    a, a_rs, a_cs,
                    b, b_rs, b_cs,
                    beta,
                    c, c_rs, c_cs,
                );
            }
        )
    );

    #[cfg(feature="blis")]
    bench_c.bench_with_input(
        BenchmarkId::new(format!("{}-blis-gemm", type_name), dt), &dt,
        |bench_b, _x| bench_b.iter(
            || unsafe {
                dispatch_gemm(
                    GemmBackend::Blis,
                    m, n, k,
                    alpha,
                    a, a_rs, a_cs,
                    b, b_rs, b_cs,
                    beta,
                    c, c_rs, c_cs,
                );
            }
        )
    );

    #[cfg(feature="rustgemm")]
    bench_c.bench_with_input(
        BenchmarkId::new(format!("{}-rust-gemm", type_name), dt), &dt,
        |bench_b, _x| bench_b.iter(
            || unsafe {
                dispatch_gemm(
                    GemmBackend::RustGemm,
                    m, n, k,
                    alpha,
                    a, a_rs, a_cs,
                    b, b_rs, b_cs,
                    beta,
                    c, c_rs, c_cs,
                );
            }
        )
    );

}

use criterion::BenchmarkGroup;
fn bench_bbb(c: &mut Criterion) {
    let mut group = c.benchmark_group("bbb");
    let dim_triple = (DimSize::Big, DimSize::Big, DimSize::Big);
    let m = 4800;
    let alpha = 1.0_f32;
    let beta = 1.0_f32;
    let a = vec![0.0_f32; m * m];
    let b_vec = vec![0.0_f32; m * m];
    let mut c_vec = vec![0.0_f32; m * m];
    let d0 = 1;
    let mnk_vec = vec![
        // 10, 100, 
        // 128,
        // 256,
        // 320, 640, 960, 2048,
        // 2400, 3200, 4000, 4800, 
        // 5600, 6400, 
        4800,
        // 7200, 8000,
    ];
    for dt in mnk_vec {
        bench_blas_group3(
            &mut group, dim_triple, d0, dt, 
            alpha,
            a.as_ptr(), 
            b_vec.as_ptr(),
            beta, 
            c_vec.as_mut_ptr()
        );
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
