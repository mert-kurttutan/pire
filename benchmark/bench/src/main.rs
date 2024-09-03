use glare_dev::{
    random_matrix_std, random_matrix_uniform,
    // CBLAS_LAYOUT, 
    CBLAS_TRANSPOSE,
    check_gemm_f16, check_gemm_f32, check_gemm_f64, check_gemm_s16s16s32, check_gemm_s8u8s32, 
    check_gemm_c32,
    check_gemm_c64,
};
use half::f16;

use bench::{
    dispatch_sgemm, dispatch_dgemm, dispatch_cgemm, dispatch_gemm_batch_f32, dispatch_hgemm,
    dispatch_gemm_s16s16s32, dispatch_gemm_s8u8s32,
    dispatch_zgemm,
    GemmBackend, gemm_backend_from_str, BenchType,
};


use num_complex::{
    c32,
    Complex32,
    c64,
    Complex64,
};

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
    random_matrix_std(m, n, &mut c, m);
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
            check_gemm_c32(
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
        println!("diff: {}", diff);
        // println!("c: {:?}", &c[..10]);
        // println!("c_ref: {:?}", &c_ref[..10]);
    }

    let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;

    end_time
}


fn test_zgemm(
    m: usize, n: usize, k: usize,
    gemm_backend: GemmBackend, args: &Args,
    alpha: f32, beta: f64,
    a_rs: isize, a_cs: isize,
    b_rs: isize, b_cs: isize,
    c_rs: isize, c_cs: isize,
) -> f64 {
    let alpha = c64(alpha as f64, 0.0);
    let beta = c64(beta, 0.0);
    let mut a = vec![Complex64::ZERO; m * k];
    let mut b = vec![Complex64::ZERO; k * n];
    let mut c = vec![Complex64::ZERO; m * n];
    random_matrix_std(m, k, &mut a, m);
    random_matrix_std(k, n, &mut b, k);
    random_matrix_std(m, n, &mut c, m);
    let mut c_ref = vec![Complex64::ZERO; m * n];
    c_ref.copy_from_slice(&c);
    let start_time = std::time::Instant::now();
    unsafe {
        dispatch_zgemm(
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
            check_gemm_c64(
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
        // println!("c    : {:?}", &c[..8]);
        // println!("c_ref: {:?}", &c_ref[..8]);
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
        dispatch_hgemm(
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
    random_matrix_uniform(m, n, &mut c, m);
    let mut c_ref = vec![0_i32; m * n];
    c_ref.copy_from_slice(&c);
    let start_time = std::time::Instant::now();
    // let beta = beta * 13.4;
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
        // println!("c    : {:?}", c);
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

    // alpha
    #[arg(short, long, default_value_t = 1.0)]
    alpha: f32,

    // beta
    #[arg(short, long, default_value_t = 1.0)]
    beta: f32,
}

 
 fn main() {
    let mut total_time = 0.0;
 
    let mut best_time = f64::INFINITY;
    // let beta = 1.0;
    // let alpha = 1.0;
    let args = Args::parse();
    let alpha = args.alpha;
    let beta = args.beta;
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
            BenchType::DGemm => test_dgemm(m, n, k, gemm_backend, &args, alpha, beta.into(), a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::SGemm => test_sgemm(m, n, k, gemm_backend, &args, alpha, beta.into(), a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::SGemmBatched => test_sgemm_batched(m, n, k, gemm_backend, &args, alpha, beta.into(), a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, batch_dim),
            BenchType::HGemm => test_hgemm(m, n, k, gemm_backend, &args, alpha, beta.into(), a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::CGemm => test_cgemm(m, n, k, gemm_backend, &args, alpha, beta.into(), a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::ZGemm => test_zgemm(m, n, k, gemm_backend, &args, alpha, beta.into(), a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::GemmS16S16S32 => test_gemm_s16s16s32(m, n, k, gemm_backend, &args, alpha, beta.into(), a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
            BenchType::GemmS8U8S32 => test_gemm_s8u8s32(m, n, k, gemm_backend, &args, alpha, beta.into(), a_rs, a_cs, b_rs, b_cs, c_rs, c_cs),
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
 