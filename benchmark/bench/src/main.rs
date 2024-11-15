use glar_dev::{
    check_gemm_c32,
    check_gemm_c64,
    check_gemm_f16,
    check_gemm_f32,
    check_gemm_f64,
    check_gemm_s16s16s32,
    check_gemm_s8u8s32,
    random_matrix_uniform,
    // CBLAS_LAYOUT,
    CBLAS_TRANSPOSE,
};
use half::f16;

use bench::{
    dispatch_cgemm, dispatch_dgemm, dispatch_gemm_batch_f32, dispatch_gemm_s16s16s32, dispatch_gemm_s8u8s32,
    dispatch_hgemm, dispatch_sgemm, dispatch_zgemm, gemm_backend_from_str, BenchType, GemmBackend,
};

unsafe fn unary_fn<T>(_: *mut T, _: usize) {}

use num_complex::{c32, c64, Complex32, Complex64};

pub fn cblas_params_from_str(
    layout_str: &str,
    m: usize,
    n: usize,
    k: usize,
) -> (i32, i32, i32, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE) {
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

pub fn blis_params_from_str(
    layout_str: &str,
    m: usize,
    n: usize,
    k: usize,
) -> (isize, isize, isize, isize, isize, isize) {
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
    if bench_type_str == "zgemm" {
        return BenchType::ZGemm;
    }
    if bench_type_str == "gemm_s16s16s32" {
        return BenchType::GemmS16S16S32;
    }
    if bench_type_str == "gemm_s8u8s32" {
        return BenchType::GemmS8U8S32;
    }
    panic!("Unsupported bench type str");
}

fn test_benchmark(mut f: impl FnMut(), n_repeats: usize) -> Vec<f64> {
    let mut best_time = f64::INFINITY;
    let mut rep = 0;
    let mut times = Vec::new();
    while rep < n_repeats {
        let start_time = std::time::Instant::now();
        f();
        let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;
        if best_time > end_time {
            best_time = end_time;
        }
        times.push(end_time);
        rep += 1;
    }
    times
}

fn test_dgemm(
    m: usize,
    n: usize,
    k: usize,
    gemm_backend: GemmBackend,
    args: &Args,
    alpha: f32,
    beta: f32,
    a_rs: isize,
    a_cs: isize,
    b_rs: isize,
    b_cs: isize,
    c_rs: isize,
    c_cs: isize,
) -> Vec<f64> {
    let alpha = alpha as f64;
    let beta = beta as f64;
    let mut a = vec![0.0; m * k];
    let mut b = vec![0.0; k * n];
    let mut c = vec![0.0; m * n];
    random_matrix_uniform(&mut a);
    random_matrix_uniform(&mut b);
    random_matrix_uniform(&mut c);
    let mut c_0 = c.clone();

    if args.check {
        let diff = unsafe {
            check_gemm_f64(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                a_rs as usize,
                a_cs as usize,
                b.as_ptr(),
                b_rs as usize,
                b_cs as usize,
                beta,
                &c,
                c_rs as usize,
                c_cs as usize,
                &mut c_0,
                unary_fn::<f64>,
                1e-3,
            )
        };
        println!("diff: {}", diff);
        vec![0.0]
    } else {
        test_benchmark(
            || unsafe {
                dispatch_dgemm(
                    gemm_backend,
                    m,
                    n,
                    k,
                    alpha,
                    a.as_ptr(),
                    a_rs,
                    a_cs,
                    b.as_ptr(),
                    b_rs,
                    b_cs,
                    beta,
                    c.as_mut_ptr(),
                    c_rs,
                    c_cs,
                );
            },
            args.n_repeats,
        )
    }
}

fn test_sgemm(
    m: usize,
    n: usize,
    k: usize,
    gemm_backend: GemmBackend,
    args: &Args,
    alpha: f32,
    beta: f32,
    a_rs: isize,
    a_cs: isize,
    b_rs: isize,
    b_cs: isize,
    c_rs: isize,
    c_cs: isize,
) -> Vec<f64> {
    let mut a = vec![0.0; m * k];
    let mut b = vec![0.0; k * n];
    let mut c = vec![0.0; m * n];
    random_matrix_uniform(&mut a);
    random_matrix_uniform(&mut b);
    random_matrix_uniform(&mut c);
    let mut c_0 = c.clone();

    if args.check {
        let diff = unsafe {
            check_gemm_f32(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                a_rs as usize,
                a_cs as usize,
                b.as_ptr(),
                b_rs as usize,
                b_cs as usize,
                beta,
                &c,
                c_rs as usize,
                c_cs as usize,
                &mut c_0,
                unary_fn::<f32>,
                1e-3,
            )
        };
        println!("diff: {}", diff);
        vec![0.0]
    } else {
        test_benchmark(
            || unsafe {
                dispatch_sgemm(
                    gemm_backend,
                    m,
                    n,
                    k,
                    alpha,
                    a.as_ptr(),
                    a_rs,
                    a_cs,
                    b.as_ptr(),
                    b_rs,
                    b_cs,
                    beta,
                    c.as_mut_ptr(),
                    c_rs,
                    c_cs,
                );
            },
            args.n_repeats,
        )
    }
}

fn test_cgemm(
    m: usize,
    n: usize,
    k: usize,
    gemm_backend: GemmBackend,
    args: &Args,
    alpha: f32,
    beta: f32,
    a_rs: isize,
    a_cs: isize,
    b_rs: isize,
    b_cs: isize,
    c_rs: isize,
    c_cs: isize,
) -> Vec<f64> {
    let alpha = c32(alpha, 0.0);
    let beta = c32(beta, 0.0);
    let mut a = vec![Complex32::ONE; m * k];
    let mut b = vec![Complex32::ONE; k * n];
    let mut c = vec![Complex32::ONE; m * n];
    random_matrix_uniform(&mut a);
    random_matrix_uniform(&mut b);
    random_matrix_uniform(&mut c);
    let mut c_0 = c.clone();

    if args.check {
        let diff = unsafe {
            check_gemm_c32(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                a_rs as usize,
                a_cs as usize,
                b.as_ptr(),
                b_rs as usize,
                b_cs as usize,
                beta,
                &c,
                c_rs as usize,
                c_cs as usize,
                &mut c_0,
                unary_fn::<Complex32>,
                1e-3,
            )
        };
        println!("diff: {}", diff);
        vec![0.0]
    } else {
        test_benchmark(
            || unsafe {
                dispatch_cgemm(
                    gemm_backend,
                    m,
                    n,
                    k,
                    alpha,
                    a.as_ptr(),
                    a_rs,
                    a_cs,
                    b.as_ptr(),
                    b_rs,
                    b_cs,
                    beta,
                    c.as_mut_ptr(),
                    c_rs,
                    c_cs,
                );
            },
            args.n_repeats,
        )
    }
}

fn test_zgemm(
    m: usize,
    n: usize,
    k: usize,
    gemm_backend: GemmBackend,
    args: &Args,
    alpha: f32,
    beta: f32,
    a_rs: isize,
    a_cs: isize,
    b_rs: isize,
    b_cs: isize,
    c_rs: isize,
    c_cs: isize,
) -> Vec<f64> {
    let alpha = c64(alpha as f64, 0.0);
    let beta = c64(beta as f64, 0.0);
    let mut a = vec![Complex64::ONE; m * k];
    let mut b = vec![Complex64::ONE; k * n];
    let mut c = vec![Complex64::ONE; m * n];
    random_matrix_uniform(&mut a);
    random_matrix_uniform(&mut b);
    random_matrix_uniform(&mut c);
    let mut c_0 = c.clone();

    if args.check {
        let diff = unsafe {
            check_gemm_c64(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                a_rs as usize,
                a_cs as usize,
                b.as_ptr(),
                b_rs as usize,
                b_cs as usize,
                beta,
                &c,
                c_rs as usize,
                c_cs as usize,
                &mut c_0,
                unary_fn::<Complex64>,
                1e-3,
            )
        };
        println!("diff: {}", diff);
        vec![0.0]
    } else {
        test_benchmark(
            || unsafe {
                dispatch_zgemm(
                    gemm_backend,
                    m,
                    n,
                    k,
                    alpha,
                    a.as_ptr(),
                    a_rs,
                    a_cs,
                    b.as_ptr(),
                    b_rs,
                    b_cs,
                    beta,
                    c.as_mut_ptr(),
                    c_rs,
                    c_cs,
                );
            },
            args.n_repeats,
        )
    }
}

fn test_sgemm_batched(
    m: usize,
    n: usize,
    k: usize,
    gemm_backend: GemmBackend,
    args: &Args,
    alpha: f32,
    beta: f32,
    a_rs: isize,
    a_cs: isize,
    b_rs: isize,
    b_cs: isize,
    c_rs: isize,
    c_cs: isize,
) -> Vec<f64> {
    let batch_size = args.batch_dim;
    let mut a = vec![0.0; m * k * batch_size];
    let mut b = vec![0.0; k * n * batch_size];
    let mut c = vec![0.0; m * n * batch_size];
    let stridea = m * k;
    let strideb = k * n;
    let stridec = m * n;
    random_matrix_uniform(&mut a);
    random_matrix_uniform(&mut b);
    random_matrix_uniform(&mut c);
    let mut c_0 = c.clone();

    if args.check {
        let diff = unsafe {
            check_gemm_f32(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                a_rs as usize,
                a_cs as usize,
                b.as_ptr(),
                b_rs as usize,
                b_cs as usize,
                beta,
                &c,
                c_rs as usize,
                c_cs as usize,
                &mut c_0,
                unary_fn::<f32>,
                1e-3,
            )
        };
        println!("diff: {}", diff);
        vec![0.0]
    } else {
        test_benchmark(
            || unsafe {
                dispatch_gemm_batch_f32(
                    gemm_backend,
                    m,
                    n,
                    k,
                    alpha,
                    a.as_ptr(),
                    a_rs,
                    a_cs,
                    stridea as isize,
                    b.as_ptr(),
                    b_rs,
                    b_cs,
                    strideb as isize,
                    beta,
                    c.as_mut_ptr(),
                    c_rs,
                    c_cs,
                    stridec as isize,
                    batch_size,
                );
            },
            args.n_repeats,
        )
    }
}

fn test_hgemm(
    m: usize,
    n: usize,
    k: usize,
    gemm_backend: GemmBackend,
    args: &Args,
    alpha: f32,
    beta: f32,
    a_rs: isize,
    a_cs: isize,
    b_rs: isize,
    b_cs: isize,
    c_rs: isize,
    c_cs: isize,
) -> Vec<f64> {
    let alpha = f16::from_f32(alpha);
    let beta = f16::from_f32(beta);
    let mut a = vec![f16::ONE; m * k];
    let mut b = vec![f16::ONE; k * n];
    let mut c = vec![f16::ONE; m * n];
    random_matrix_uniform(&mut a);
    random_matrix_uniform(&mut b);
    random_matrix_uniform(&mut c);
    let mut c_0 = c.clone();

    if args.check {
        let diff = unsafe {
            check_gemm_f16(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                a_rs as usize,
                a_cs as usize,
                b.as_ptr(),
                b_rs as usize,
                b_cs as usize,
                beta,
                &c,
                c_rs as usize,
                c_cs as usize,
                &mut c_0,
                unary_fn::<f16>,
                1e-1,
            )
        };
        println!("diff: {}", diff);
        vec![0.0]
    } else {
        test_benchmark(
            || unsafe {
                dispatch_hgemm(
                    gemm_backend,
                    m,
                    n,
                    k,
                    alpha,
                    a.as_ptr(),
                    a_rs,
                    a_cs,
                    b.as_ptr(),
                    b_rs,
                    b_cs,
                    beta,
                    c.as_mut_ptr(),
                    c_rs,
                    c_cs,
                );
            },
            args.n_repeats,
        )
    }
}

fn test_gemm_s16s16s32(
    m: usize,
    n: usize,
    k: usize,
    gemm_backend: GemmBackend,
    args: &Args,
    alpha: f32,
    beta: f32,
    a_rs: isize,
    a_cs: isize,
    b_rs: isize,
    b_cs: isize,
    c_rs: isize,
    c_cs: isize,
) -> Vec<f64> {
    let mut a = vec![0_i16; m * k];
    let mut b = vec![0_i16; k * n];
    let mut c = vec![0_i32; m * n];
    random_matrix_uniform(&mut a);
    random_matrix_uniform(&mut b);
    random_matrix_uniform(&mut c);
    let mut c_0 = c.clone();

    if args.check {
        let diff = unsafe {
            check_gemm_s16s16s32(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                a_rs as usize,
                a_cs as usize,
                b.as_ptr(),
                b_rs as usize,
                b_cs as usize,
                beta,
                &c,
                c_rs as usize,
                c_cs as usize,
                &mut c_0,
                unary_fn::<i32>,
                1e-3,
            )
        };
        println!("diff: {}", diff);
        vec![0.0]
    } else {
        test_benchmark(
            || unsafe {
                dispatch_gemm_s16s16s32(
                    gemm_backend,
                    m,
                    n,
                    k,
                    alpha,
                    a.as_ptr(),
                    a_rs,
                    a_cs,
                    b.as_ptr(),
                    b_rs,
                    b_cs,
                    beta,
                    c.as_mut_ptr(),
                    c_rs,
                    c_cs,
                );
            },
            args.n_repeats,
        )
    }
}

fn test_gemm_s8u8s32(
    m: usize,
    n: usize,
    k: usize,
    gemm_backend: GemmBackend,
    args: &Args,
    alpha: f32,
    beta: f32,
    a_rs: isize,
    a_cs: isize,
    b_rs: isize,
    b_cs: isize,
    c_rs: isize,
    c_cs: isize,
) -> Vec<f64> {
    let mut a = vec![0_i8; m * k];
    let mut b = vec![0_u8; k * n];
    let mut c = vec![0_i32; m * n];
    random_matrix_uniform(&mut a);
    random_matrix_uniform(&mut b);
    random_matrix_uniform(&mut c);
    let mut c_0 = c.clone();

    if args.check {
        let diff = unsafe {
            check_gemm_s8u8s32(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                a_rs as usize,
                a_cs as usize,
                b.as_ptr(),
                b_rs as usize,
                b_cs as usize,
                beta,
                &c,
                c_rs as usize,
                c_cs as usize,
                &mut c_0,
                unary_fn::<i32>,
                1e-3,
            )
        };
        println!("diff: {}", diff);
        vec![0.0]
    } else {
        test_benchmark(
            || unsafe {
                dispatch_gemm_s8u8s32(
                    gemm_backend,
                    m,
                    n,
                    k,
                    alpha,
                    a.as_ptr(),
                    a_rs,
                    a_cs,
                    b.as_ptr(),
                    b_rs,
                    b_cs,
                    beta,
                    c.as_mut_ptr(),
                    c_rs,
                    c_cs,
                );
            },
            args.n_repeats,
        )
    }
}

use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// number of repeats
    #[arg(long, default_value_t = 5)]
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
    #[arg(long, default_value_t = 5)]
    batch_dim: usize,

    // tranpose layout
    #[arg(short, long, default_value_t = String::from("nt"))]
    t_layout: String,

    #[arg(short, long, default_value_t = false)]
    check: bool,

    // gemm backend
    #[arg(long, default_value_t = String::from("glar"))]
    backend: String,

    // bench type
    #[arg(long, default_value_t = String::from("sgemm"))]
    bench_type: String,

    // alpha
    #[arg(short, long, default_value_t = 1.0)]
    alpha: f32,

    // beta
    #[arg(long, default_value_t = 1.0)]
    beta: f32,
}
use serde::{Deserialize, Serialize};

use bench::BenchmarkConfig;

#[derive(Serialize, Deserialize, Debug)]
pub enum DimStrategy {
    Big(Vec<usize>),
    SmallM(usize, Vec<usize>),
    SmallN(usize, Vec<usize>),
    SmallK(usize, Vec<usize>),
}

impl DimStrategy {
    pub fn mnk_idx(&self, i: usize) -> (usize, usize, usize) {
        match self {
            DimStrategy::Big(v) => (v[i], v[i], v[i]),
            DimStrategy::SmallM(m, v) => (*m, v[i], v[i]),
            DimStrategy::SmallN(n, v) => (v[i], *n, v[i]),
            DimStrategy::SmallK(k, v) => (v[i], v[i], *k),
        }
    }

    pub fn dim_len(&self) -> usize {
        match self {
            DimStrategy::Big(v) => v.len(),
            DimStrategy::SmallM(_, v) => v.len(),
            DimStrategy::SmallN(_, v) => v.len(),
            DimStrategy::SmallK(_, v) => v.len(),
        }
    }
}

use bench::get_benchmark_config;

#[derive(Serialize, Deserialize, Debug)]
pub struct BenchmarkResult {
    pub bench_config: BenchmarkConfig,
    pub dim_strategy: DimStrategy,
    pub layout: String,
    pub times: Vec<Vec<f64>>,
    pub implementation: String,
}

use std::fs;
use std::path::Path;

const PROJECT_DIR: &str = core::env!("CARGO_MANIFEST_DIR");

const BENCHMARK_FOLDER: &str = "benchmark_results";

fn get_bench_filename() -> String {
    // look for the name of files inside the benchmark folder
    // filename should be benchmark_results_n.json
    // where n is the number of files inside the folder
    let benchmark_folder_path = Path::new(PROJECT_DIR).join(BENCHMARK_FOLDER);
    let files = fs::read_dir(benchmark_folder_path.clone()).unwrap();
    let num_files = files.count();
    format!("benchmark_result_{}.json", num_files)
}

fn prepare_num_threads() {
    let default_num_threads = std::thread::available_parallelism().unwrap().get().to_string();
    let n_threads_str = std::env::var("NUM_THREADS").unwrap_or(default_num_threads);
    std::env::set_var("NUM_THREADS", n_threads_str.clone());
    std::env::set_var("GLAR_NUM_THREADS", n_threads_str.clone());
    std::env::set_var("BLIS_NUM_THREADS", n_threads_str.clone());
    std::env::set_var("OPENBLAS_NUM_THREADS", n_threads_str.clone());
    std::env::set_var("MKL_NUM_THREADS", n_threads_str.clone());
    std::env::set_var("OMP_NUM_THREADS", n_threads_str.clone());
}

fn prepare_benchmark_file() -> std::path::PathBuf {
    let benchmark_folder_path = Path::new(PROJECT_DIR).join(BENCHMARK_FOLDER);
    fs::create_dir_all(benchmark_folder_path.clone()).unwrap();
    let benchmark_result_path = benchmark_folder_path.join(get_bench_filename());
    benchmark_result_path
}

// fn main() {
//     prepare_num_threads();
//     let dim_strategy = DimStrategy::Big(vec![4800]);
//     let hw = get_benchmark_config();
//     let args = Args::parse();
//     let alpha = args.alpha;
//     let beta = args.beta;
//     let layout_str = &args.t_layout;
//     let gemm_backend = gemm_backend_from_str(&args.backend);
//     let bench_type = bench_type_from_str(&args.bench_type);

//     let test_func = match bench_type {
//         BenchType::DGemm => test_dgemm,
//         BenchType::SGemm => test_sgemm,
//         BenchType::SGemmBatched => test_sgemm_batched,
//         BenchType::HGemm => test_hgemm,
//         BenchType::CGemm => test_cgemm,
//         BenchType::ZGemm => test_zgemm,
//         BenchType::GemmS16S16S32 => test_gemm_s16s16s32,
//         BenchType::GemmS8U8S32 => test_gemm_s8u8s32,
//     };
//     let dim_len = dim_strategy.dim_len();
//     for i in 0..dim_len {
//         let (m, n, k) = dim_strategy.mnk_idx(i);
//         let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = blis_params_from_str(layout_str, m, n, k);
//         let end_time = test_func(m, n, k, gemm_backend, &args, alpha, beta.into(), a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
//         println!("m: {}, n: {}, k: {}, time: {:?}", m, n, k, end_time);
//     }
// }

static LONG_DIMS: [usize; 12] = [1, 2, 4, 13, 27, 128, 256, 512, 1024, 2400, 4800, 9600];
const SMALL_DIM: usize = 79;

fn main() {
    prepare_num_threads();
    let benchmark_result_path = prepare_benchmark_file();
    let dim_strategy = DimStrategy::SmallM(SMALL_DIM, LONG_DIMS.to_vec());
    let hw = get_benchmark_config();
    let args = Args::parse();
    let alpha = args.alpha;
    let beta = args.beta;
    let layout_str = &args.t_layout;
    let gemm_backend = gemm_backend_from_str(&args.backend);
    let bench_type = bench_type_from_str(&args.bench_type);

    let mut benchmark_result = BenchmarkResult {
        bench_config: hw,
        dim_strategy: dim_strategy,
        layout: args.t_layout.clone(),
        times: vec![],
        implementation: args.backend.clone(),
    };
    let j = serde_json::to_string(&benchmark_result).unwrap();
    std::fs::write(benchmark_result_path.clone(), j).unwrap();

    let test_func = match bench_type {
        BenchType::DGemm => test_dgemm,
        BenchType::SGemm => test_sgemm,
        BenchType::SGemmBatched => test_sgemm_batched,
        BenchType::HGemm => test_hgemm,
        BenchType::CGemm => test_cgemm,
        BenchType::ZGemm => test_zgemm,
        BenchType::GemmS16S16S32 => test_gemm_s16s16s32,
        BenchType::GemmS8U8S32 => test_gemm_s8u8s32,
    };
    let dim_len = benchmark_result.dim_strategy.dim_len();
    for i in 0..dim_len {
        let (m, n, k) = benchmark_result.dim_strategy.mnk_idx(i);
        let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = blis_params_from_str(layout_str, m, n, k);
        let end_time = test_func(m, n, k, gemm_backend, &args, alpha, beta.into(), a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        benchmark_result.times.push(end_time);
        let j = serde_json::to_string(&benchmark_result).unwrap();
        std::fs::write(benchmark_result_path.clone(), j).unwrap();
    }
}
