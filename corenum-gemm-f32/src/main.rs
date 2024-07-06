
#![allow(unused)]

use corenum_gemm_f32::{
    CorenumPar,
    // BLAS_LAYOUT,
    // gelu_f32,
};
use corenum_dev::{
    random_matrix, blis_params_from_str,
};
 
#[cfg(feature="blis")]
use corenum_dev::{
    BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE
};
 
use clap::Parser;
 
 
/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// dim m
    #[arg(short, long, default_value_t = 200)]
    m: usize,


    /// dim n
    #[arg(short, long, default_value_t = 200)]
    n: usize,


    /// dim k
    #[arg(short, long, default_value_t = 200)]
    k: usize,

   // tranpose layout
   #[arg(short, long, default_value_t = String::from("nt"))]
   t_layout: String,


   #[arg(short, long, default_value_t = false)]
   check: bool,
}
 
 
 fn main() {
    let n_repeats = 2;
    let mut total_time = 0.0;
    let size_limit = 51;
    let mut i = 50;
    const N_THREAD: usize = 2;
 
    // println!("-> Main Thread on CPU {}", unsafe { libc::sched_getcpu() });
    let mut best_time = f64::INFINITY;
    let beta = 1.0;
    let alpha = 1.0;
    let args = Args::parse();
    let m = args.m;
    let n = args.n;
    let k = args.k;
    let layout_str = &args.t_layout;

    let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = blis_params_from_str(layout_str, m, n, k);

  
    while i < size_limit {
        let mut rep = 0;
        while rep < n_repeats {
            let mut a = vec![0.0; m * k];
            let mut b = vec![0.0; k * n];
            let mut c = vec![0.0; m * n];
            // let mut c = avec![0.0; m*n];
            random_matrix(m, k, &mut a, m);
            random_matrix(k, n, &mut b, k);
            // println!("a: {:?}", a);
            // println!("b: {:?}", b);
            let mut c_ref = vec![0.0; m * n];
            c_ref.copy_from_slice(&c);
            let start_time = std::time::Instant::now();
            #[cfg(feature="corenum")]
            unsafe {
                let d_par = CorenumPar::new(
                    1, 1, 1, 1, 1, 1
                );
                corenum_gemm_f32::corenum_sgemm(
                    m, n, k,
                    alpha,
                    a.as_ptr(), a_rs as usize, a_cs as usize,
                    b.as_ptr(), b_rs as usize, b_cs as usize,
                    beta,
                    c.as_mut_ptr(), c_rs as usize, c_cs as usize,
                    &d_par,
                );
            }
            #[cfg(feature="mkl")]
            unsafe {
                use corenum_dev::CblasColMajor;
                let (lda, ldb, ldc, a_layout, b_layout) = corenum_dev::cblas_params_from_str(layout_str, m, n, k);
                corenum_dev::cblas_sgemm(
                    CblasColMajor,
                    a_layout,
                    b_layout,
                    m as i32, n as i32, k as i32,
                    alpha,
                    a.as_ptr(), lda as i32,
                    b.as_ptr(), ldb as i32,
                    beta,
                    c.as_mut_ptr(), ldc as i32,
                );
            }

            #[cfg(feature="blis")]
            unsafe {
                corenum_dev::bli_sgemm(
                    BLIS_NO_TRANSPOSE,
                    BLIS_NO_TRANSPOSE,
                    m as i32, n as i32, k as i32,
                    &alpha,
                    a.as_ptr(), a_rs, a_cs,
                    b.as_ptr(), b_rs, b_cs,
                    &beta,
                    c.as_mut_ptr(), c_rs, c_cs,
                );
            }
 
            #[cfg(feature="blasfeo")]
            unsafe {
                use corenum_dev::CblasColMajor;
                let (lda, ldb, ldc, a_layout, b_layout) = corenum_dev::cblas_params_from_str(layout_str, m, n, k);
                corenum_dev::blasfeo_cblas_sgemm(
                    CblasColMajor,
                    a_layout,
                    b_layout,
                    m as i32, n as i32, k as i32,
                    alpha,
                    a.as_ptr(), lda as i32,
                    b.as_ptr(), ldb as i32,
                    beta,
                    c.as_mut_ptr(), ldc as i32,
                );
            }
 
            #[cfg(feature="rustgemm")]
            unsafe {
                gemm(
                    m, n, k,
                    c.as_mut_ptr(), 1 as isize, m as isize,
                    true,
                    a.as_ptr(), m as isize, 1 as isize,
                    b.as_ptr(), 1 as isize, n as isize,
                    alpha, beta,
                    false, false, false,
                    gemm::Parallelism::Rayon(1)
                );
            }
            if args.check {
                let diff = unsafe {
                    corenum_dev::check_gemm_f32(
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
                // println!("c: {:?}", c);
                // println!("c_ref: {:?}", c_ref);
            }
 
            let end_time = start_time.elapsed().as_nanos() as f64 / 1e9;
            total_time += end_time;
 
 
            println!("time: {}, total_time: {}", end_time, total_time);
            if best_time > end_time {
                best_time = end_time;
            }
            rep += 1;
        }
        // let diff_max = 0;
        // println!("dim: {}, GFLOPS: {}", dim, gflops / best_time);
        // println!("------------------------------------------");
        i += 1;
    }
    let gflops = 2.0 * m as f64 * n as f64 * k as f64 / best_time / 1e9;
    println!("best_time: {}, GFLOPS: {}", best_time, gflops);
 }
 
 
 
 
 
 