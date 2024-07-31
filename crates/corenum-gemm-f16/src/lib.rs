#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_arch;

#[cfg(target_arch = "aarch64")]
pub(crate) mod armv8;

pub(crate) mod reference;

pub(crate) type TA = u16;
pub(crate) type TB = u16;
pub(crate) type TC = u16;

#[cfg(target_arch = "x86_64")]
use corenum_base::{
	hw_avx,
	hw_avx512f,
	hw_fma,
	hw_model,
    hw_avx512f16,
};

use corenum_base::{
    GemmGotoPackaPackb,
	GemmSmallM,
	GemmSmallN,
	GemmCache,
	Gemv,
	corenum_gemv,
	corenum_gemm,
	StridedMatrix,
	GemmArray,
	StridedMatrixMut,
	GemmOut,
};
pub use corenum_base::CorenumPar;


use std::thread;
use corenum_base::RUNTIME_HW_CONFIG;

pub unsafe fn corenum_gemm_f16f16f16<
A: GemmArray<u16,X=u16> + GemmArray<f32,X=u16>, 
B: GemmArray<u16,X=u16> + GemmArray<f32,X=u16>,
C: GemmOut<X=u16,Y=u16>,
>(
	m: usize, n: usize, k: usize,
	alpha: f32,
	a: A,
	b: B,
	beta: C::X,
	c: C,
){	
	let par = CorenumPar::default();
	#[cfg(target_arch = "x86_64")]
	{
        if hw_avx512f16() {
            let hw_config = x86_64_arch::Avx512Dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, 4800, 192, 512, x86_64_arch::Avx512Features::Avx512F16);
            corenum_gemm(
                &hw_config, m, n, k, 1_u16, a, b, beta, c, &par
            );
            return;
        } 
        if hw_avx() {
            let hw_config = x86_64_arch::AvxDispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, 4800, 192, 512, x86_64_arch::AvxFeatures::AvxFma);
            corenum_gemm(
                &hw_config, m, n, k, 1_f32, a, b, beta, c, &par
            );
        }
	}

	#[cfg(target_arch="aarch64")]
	{
		const MR: usize = 24;
		const NR: usize = 4;
		let hw_config = armv8::AvxFma::<MR,NR>{
			goto_mc: 4800, goto_nc: 192, goto_kc: 512,
			is_l1_shared: false, is_l2_shared: false, is_l3_shared: true
		};
		corenum_gemm(
			&hw_config, m, n, k, alpha, a, b, beta, c, &par
		);
		return;
	}
}



pub unsafe fn corenum_hgemm(
	m: usize, n: usize, k: usize,
	alpha: f32,
	a: *const u16, a_rs: usize, a_cs: usize,
	b: *const u16, b_rs: usize, b_cs: usize,
	beta: u16,
	c: *mut u16, c_rs: usize, c_cs: usize,
) {
	// do not exchange if transa && transb
	let (m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b) = if c_cs == 1 && c_rs != 1 {
    	(n, m, b_rs, b_cs, a_rs, a_cs, c_cs, c_rs, b, a)
	} else {
    	(m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b)
	};
	let a = StridedMatrix::new(a, a_rs, a_cs);
	let b = StridedMatrix::new(b, b_rs, b_cs);
	let c = StridedMatrixMut::new(c, c_rs, c_cs);
	corenum_gemm_f16f16f16(m, n, k, alpha, a, b, beta, c);
}



#[cfg(test)]
mod tests {
	use super::*;
	use corenum_dev::{
    	random_matrix,
    	check_gemm_f32,
	};

	const EPS: f64 = 2e-2;

	// static M_ARR: [usize; 32] = [1, 2, 3, 16, 32, 24, 37, 38, 17, 32, 48, 64, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 958, 959, 960, 950, 951, 943, 944];
	static M_ARR: [usize; 32] = [1, 2, 3, 16, 32, 24, 37, 38, 17, 32, 48, 64, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 458, 459, 460, 450, 451, 443, 444];
	static N_ARR: [usize; 28] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 64, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 658, 659, 660];
	static K_ARR: [usize; 10] = [1, 8, 16, 64, 128, 129, 130, 131, 132, 509];
	static ALPHA_ARR: [f32; 1] = [1.0];
	static BETA_ARR: [f32; 3] = [1.0, 0.0, 3.1415];
	enum Layout {
    	NN,
    	NT,
    	TN,
    	TT,
	}

	fn dispatch_strides(layout: &Layout, m: usize, n: usize, k: usize) -> (usize, usize, usize, usize, usize, usize) {
    	match layout {
        	Layout::NN => (1, m, 1, k, 1, m),
        	Layout::NT => (1, m, n, 1, 1, m),
        	Layout::TN => (k, 1, 1, k, 1, m),
        	Layout::TT => (k, 1, n, 1, 1, m),
    	}
	}
	fn test_gemm(layout: &Layout) {
    	for m in M_ARR {
        	for n in N_ARR {
            	let mut c = vec![0.0; m * n];
            	let mut c_ref = vec![0.0; m * n];
            	for k in K_ARR {
                	let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = dispatch_strides(&layout, m, n, k);
                	let mut a = vec![0.0; m * k];
                	let mut b = vec![0.0; k * n];
                	for alpha in ALPHA_ARR {
                    	for beta in BETA_ARR {
                        	random_matrix(m, k, &mut a, m);
                        	random_matrix(k, n, &mut b, k);
                        	random_matrix(m, n, &mut c, m);
                        	c_ref.copy_from_slice(&c);
                        	unsafe {
                            	corenum_sgemm(
                                	m, n, k,
                                	alpha,
                                	a.as_ptr(), a_rs, a_cs,
                                	b.as_ptr(), b_rs, b_cs,
                                	beta,
                                	c.as_mut_ptr(), c_rs, c_cs,
                            	);
                        	}
                        	let diff_max = unsafe { 
								check_gemm_f32(
									m, n, k,
									alpha,
									a.as_ptr(), a_rs, a_cs,
									b.as_ptr(), b_rs, b_cs,
									beta,
									&mut c, c_rs, c_cs,
									&mut c_ref,
								)
							};
                        	// if diff_max >= EPS {
                            // 	println!("a: {:?}", a);
                            // 	println!("b: {:?}", b);
                            // 	println!("c: {:?}", c);
                            // 	println!("c_ref: {:?}", c_ref);
                        	// }
                        	assert!(diff_max < EPS, "diff_max: {}, m: {}, n: {}, k: {}, alpha: {}, beta: {}", diff_max, m, n, k, alpha, beta);
                    	}
                	}
            	}
        	}
    	}
	}

	fn test_gemm_ap(layout: &Layout) {
    	for m in M_ARR {
        	for n in N_ARR {
            	let mut c = vec![0.0; m * n];
            	let mut c_ref = vec![0.0; m * n];
            	for k in K_ARR {
                	let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = dispatch_strides(&layout, m, n, k);
                	let mut a = vec![0.0; m * k];
                	let mut b = vec![0.0; k * n];
					let mut ap = vec![0_f32; (m+100)*k+512];
					let ap_offset = ap.as_ptr().align_offset(512);
					let ap_mut_ptr = unsafe {ap.as_mut_ptr().add(ap_offset)};
					let ap_ptr = ap_mut_ptr as *const f32;
                	for alpha in ALPHA_ARR {
                    	for beta in ALPHA_ARR {
                        	random_matrix(m, k, &mut a, m);
                        	random_matrix(k, n, &mut b, k);
                        	random_matrix(m, n, &mut c, m);
                        	c_ref.copy_from_slice(&c);
							unsafe {
								packa_f32(m, k, a.as_ptr(), a_rs, a_cs, ap_mut_ptr);
							}
							let ap_matrix = corenum_base::PackedMatrix{
								data_ptr: ap_ptr,
								mc: 4800,
								kc: 512,
								mr: 48,
								k,
								m,
								rs: a_rs,
								cs: a_cs,
							};
							let b_matrix = StridedMatrix{
								data_ptr: b.as_ptr(),
								rs: b_rs, cs: b_cs,
							};
							let c_matrix = StridedMatrixMut{
								data_ptr: c.as_mut_ptr(),
								rs: c_rs, cs: c_cs,
							};
                        	unsafe {
                            	corenum_gemm_f32f32f32(
                                	m, n, k,
                                	alpha,
                                	ap_matrix,
                                	b_matrix,
                                	beta,
                                	c_matrix,
                            	);
                        	}
                        	let diff_max = unsafe { 
								check_gemm_f32(
									m, n, k,
									alpha,
									a.as_ptr(), a_rs, a_cs,
									b.as_ptr(), b_rs, b_cs,
									beta,
									&mut c, c_rs, c_cs,
									&mut c_ref,
								)
							};
                        	// if diff_max >= EPS {
                            // 	println!("a: {:?}", a);
                            // 	println!("b: {:?}", b);
                            // 	println!("c:     {:?}", c);
                            // 	println!("c_ref: {:?}", c_ref);
                        	// }
                        	assert!(diff_max < EPS, "diff_max: {}, m: {}, n: {}, k: {}, alpha: {}, beta: {}", diff_max, m, n, k, alpha, beta);
                    	}
                	}
            	}
        	}
    	}
	}
	// #[test]
	// fn test_nn_col_ap() {
    // 	test_gemm_ap(&Layout::NN);
	// }
	#[test]
	fn test_nn_col() {
    	test_gemm(&Layout::NN);
	}

	#[test]
	fn test_nt_col() {
    	test_gemm(&Layout::NT);
	}

	#[test]
	fn test_tn_col() {
    	test_gemm(&Layout::TN);
	}

	#[test]
	fn test_tt_col() {
    	test_gemm(&Layout::TT);
	}

}
