use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

use half::f16;
use num_complex::Complex;

use bench::{dispatch_gemm, GemmBackend, AS};

trait BenchmarkType {
    type TA;
    type TB;
    type TC;
    type TS;

    const M: usize;
    const TA_ONE: Self::TA;
    const TB_ONE: Self::TB;
    const TC_ONE: Self::TC;
    const SCALAR_ONE: Self::TS;
    const SCALAR_ZERO: Self::TS;
}

struct BenchmarkI8;
impl BenchmarkType for BenchmarkI8 {
    type TA = i8;
    type TB = i8;
    type TC = i32;
    type TS = f32;

    const M: usize = 6000;
    const TA_ONE: Self::TA = 1;
    const TB_ONE: Self::TB = 1;
    const TC_ONE: Self::TC = 1;
    const SCALAR_ONE: Self::TS = 1.0;
    const SCALAR_ZERO: Self::TS = 0.0;
}

struct BenchmarkI16;
impl BenchmarkType for BenchmarkI16 {
    type TA = i16;
    type TB = i16;
    type TC = i32;
    type TS = f32;

    const M: usize = 5400;
    const TA_ONE: Self::TA = 1;
    const TB_ONE: Self::TB = 1;
    const TC_ONE: Self::TC = 1;
    const SCALAR_ONE: Self::TS = 1.0;
    const SCALAR_ZERO: Self::TS = 0.0;
}

struct BenchmarkF32;
impl BenchmarkType for BenchmarkF32 {
    type TA = f32;
    type TB = f32;
    type TC = f32;
    type TS = f32;

    const M: usize = 4800;
    const TA_ONE: Self::TA = 1.0;
    const TB_ONE: Self::TB = 1.0;
    const TC_ONE: Self::TC = 1.0;
    const SCALAR_ONE: Self::TS = 1.0;
    const SCALAR_ZERO: Self::TS = 0.0;
}

struct BenchmarkF64;
impl BenchmarkType for BenchmarkF64 {
    type TA = f64;
    type TB = f64;
    type TC = f64;
    type TS = f64;

    const M: usize = 3600;
    const TA_ONE: Self::TA = 1.0;
    const TB_ONE: Self::TB = 1.0;
    const TC_ONE: Self::TC = 1.0;
    const SCALAR_ONE: Self::TS = 1.0;
    const SCALAR_ZERO: Self::TS = 0.0;
}

struct BenchmarkC32;
impl BenchmarkType for BenchmarkC32 {
    type TA = Complex<f32>;
    type TB = Complex<f32>;
    type TC = Complex<f32>;
    type TS = Complex<f32>;

    const M: usize = 3200;
    const TA_ONE: Self::TA = Complex::new(1.0, 0.0);
    const TB_ONE: Self::TB = Complex::new(1.0, 0.0);
    const TC_ONE: Self::TC = Complex::new(1.0, 0.0);
    const SCALAR_ONE: Self::TS = Complex::new(1.0, 0.0);
    const SCALAR_ZERO: Self::TS = Complex::new(0.0, 0.0);
}

struct BenchmarkC64;
impl BenchmarkType for BenchmarkC64 {
    type TA = Complex<f64>;
    type TB = Complex<f64>;
    type TC = Complex<f64>;
    type TS = Complex<f64>;

    const M: usize = 2400;
    const TA_ONE: Self::TA = Complex::new(1.0, 0.0);
    const TB_ONE: Self::TB = Complex::new(1.0, 0.0);
    const TC_ONE: Self::TC = Complex::new(1.0, 0.0);
    const SCALAR_ONE: Self::TS = Complex::new(1.0, 0.0);
    const SCALAR_ZERO: Self::TS = Complex::new(0.0, 0.0);
}

struct BenchmarkF16;
impl BenchmarkType for BenchmarkF16 {
    type TA = f16;
    type TB = f16;
    type TC = f16;
    type TS = f16;

    const M: usize = 4800;
    const TA_ONE: Self::TA = f16::ONE;
    const TB_ONE: Self::TB = f16::ONE;
    const TC_ONE: Self::TC = f16::ONE;
    const SCALAR_ONE: Self::TS = f16::ONE;
    const SCALAR_ZERO: Self::TS = f16::ZERO;
}

#[derive(Clone, Copy, Debug)]
pub enum DimSize {
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

pub fn bench_blas_group3<M: criterion::measurement::Measurement, TA: AS, TB: 'static, TC: 'static>(
    bench_c: &mut BenchmarkGroup<M>,
    dim_triple: (DimSize, DimSize, DimSize),
    d0: usize,
    dt: usize,
    alpha: TA::ASType,
    a: *const TA,
    b: *const TB,
    beta: TA::BSType,
    c: *mut TC,
) {
    let (m, n, k) = get_mnk(dim_triple, d0, dt);
    let n = 51864;
    let m = 1;
    let k = 384;
    let (a_rs, a_cs) = (k as isize, 1);
    let (b_rs, b_cs) = (1, k as isize);
    let (c_rs, c_cs) = (1, m as isize);
    let type_name = type_name::<TA>();
    #[cfg(feature = "pire")]
    bench_c.bench_with_input(BenchmarkId::new(format!("{}-pire-gemm", type_name), dt), &dt, |bench_b, _x| {
        bench_b.iter(|| unsafe {
            dispatch_gemm(GemmBackend::Pire, m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c, c_rs, c_cs);
        })
    });
    #[cfg(feature = "mkl")]
    bench_c.bench_with_input(BenchmarkId::new(format!("{}-mkl-gemm", type_name), dt), &dt, |bench_b, _x| {
        bench_b.iter(|| unsafe {
            dispatch_gemm(GemmBackend::Mkl, m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c, c_rs, c_cs);
        })
    });

    #[cfg(feature = "blis")]
    bench_c.bench_with_input(BenchmarkId::new(format!("{}-blis-gemm", type_name), dt), &dt, |bench_b, _x| {
        bench_b.iter(|| unsafe {
            dispatch_gemm(GemmBackend::Blis, m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c, c_rs, c_cs);
        })
    });
    #[cfg(feature = "openblas")]
    bench_c.bench_with_input(BenchmarkId::new(format!("{}-mkl-gemm", type_name), dt), &dt, |bench_b, _x| {
        bench_b.iter(|| unsafe {
            dispatch_gemm(GemmBackend::OpenBLAS, m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c, c_rs, c_cs);
        })
    });

    #[cfg(feature = "rustgemm")]
    bench_c.bench_with_input(BenchmarkId::new(format!("{}-rust-gemm", type_name), dt), &dt, |bench_b, _x| {
        bench_b.iter(|| unsafe {
            dispatch_gemm(GemmBackend::RustGemm, m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c, c_rs, c_cs);
        })
    });
}

#[cfg(not(any(
    feature = "bench-f64",
    feature = "bench-i8",
    feature = "bench-i16",
    feature = "bench-c32",
    feature = "bench-c64",
    feature = "bench-f16"
)))]
type MainBenchmarkType = BenchmarkF32;
#[cfg(feature = "bench-f64")]
type MainBenchmarkType = BenchmarkF64;
#[cfg(feature = "bench-i8")]
type MainBenchmarkType = BenchmarkI8;
#[cfg(feature = "bench-i16")]
type MainBenchmarkType = BenchmarkI16;
#[cfg(feature = "bench-c32")]
type MainBenchmarkType = BenchmarkC32;
#[cfg(feature = "bench-c64")]
type MainBenchmarkType = BenchmarkC64;
#[cfg(feature = "bench-f16")]
type MainBenchmarkType = BenchmarkF16;

use criterion::BenchmarkGroup;
fn bench_bbb(c: &mut Criterion) {
    let mut group = c.benchmark_group("bbb");
    let dim_triple = (DimSize::Big, DimSize::Big, DimSize::Big);
    let m = MainBenchmarkType::M;
    let alpha = MainBenchmarkType::SCALAR_ONE;
    let beta = MainBenchmarkType::SCALAR_ZERO;
    let mut a = vec![MainBenchmarkType::TA_ONE; m * m];
    let mut b_vec = vec![MainBenchmarkType::TB_ONE; m * m];
    let mut c_vec = vec![MainBenchmarkType::TC_ONE; m * m];
    use pire_dev::random_matrix_uniform;
    random_matrix_uniform(&mut a);
    random_matrix_uniform(&mut b_vec);
    random_matrix_uniform(&mut c_vec);
    let d0 = 1;
    let mnk_vec = vec![
        // 10, 100,
        // 128,
        // 256,
        // 320, 640, 960, 2048,
        // 2400, 3200, 4000, 4800,
        // 5600, 6400,
        m,
        // 7200, 8000,
    ];
    for dt in mnk_vec {
        bench_blas_group3(&mut group, dim_triple, d0, dt, alpha, a.as_ptr(), b_vec.as_ptr(), beta, c_vec.as_mut_ptr());
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
