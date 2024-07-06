pub(crate) mod haswell;
pub(crate) mod reference;

pub(crate) type TA = f32;
pub(crate) type TB = f32;
pub(crate) type TC = f32;


use corenum_base::StridedMatrix;
use corenum_base::{
	HWConfig, RUNTIME_HW_CONFIG,
    GemmGotoPackaPackb,
	GemmSmallM,
	GemmSmallN,
	Gemv,
	corenum_gemv,
};
pub use corenum_base::CorenumPar;
use haswell::{
	HaswellGemm,
	// HaswellGemv,
};

// use reference::{
// 	// ReferenceGemm,
// 	ReferenceGemv,
// };

// use haswell::microkernel::gelu;

// pub unsafe fn gelu_f32(a: *mut f32, n: usize) {
// 	use std::arch::x86_64::*;
// 	const VS: usize = 8;
// 	let n_iter = n / VS;
// 	let n_left = n % VS;
// 	let mut p = 0;
// 	while p < n_iter {
//     	let a_vec = _mm256_loadu_ps(a.add(p*VS));
//     	let a_vec = gelu(a_vec);
//     	_mm256_storeu_ps(a.add(p*VS), a_vec);
//     	p += 1;
// 	}
// 	let mut leftover_vec = [0.0; VS];
// 	let a = a.add(n_iter*VS);
// 	// load leftover to leftover_vec
// 	std::ptr::copy_nonoverlapping(a, leftover_vec.as_mut_ptr(), n_left);
// 	let a_vec = _mm256_loadu_ps(leftover_vec.as_ptr());
// 	let a_vec = gelu(a_vec);
// 	_mm256_storeu_ps(leftover_vec.as_mut_ptr(), a_vec);
// 	// store leftover_vec back to a
// 	std::ptr::copy_nonoverlapping(leftover_vec.as_ptr(), a, n_left);
// }

use haswell::Identity;

pub unsafe fn corenum_gemv_f32f32f32<
A: Copy+GemmArray<T=f32,U=f32> + SupN, 
B: Copy+GemmArray<T=f32,U=f32> + SupM,
>(
	m: usize, n: usize,
	alpha: TA,
	a: A,
	b: B,
	beta: TC,
	c: *mut TC, inc_y: usize,
	par: &CorenumPar,
){	
	match *RUNTIME_HW_CONFIG {
		HWConfig::Haswell => {
			corenum_gemv::<TC, A, B, HaswellGemm>(
				m, n, alpha, a, b, beta, c, inc_y, par
			);
		}
		HWConfig::Reference => {
			// corenum_gemm::<TA, TB, TC, InputA, InputB, ReferenceGemm>(
			// 	m, n, k, alpha, a, b, beta, c, c_rs, c_cs, par
			// );
		}
	}
}

pub unsafe fn corenum_sgemv(
	m: usize, n: usize,
	alpha: TA,
	a: *const TA, a_rs: usize, a_cs: usize,
	x: *const TB, incx: usize,
	beta: TC,
	y: *mut TC, incy: usize,
	par: &CorenumPar,
) {
	let a = StridedMatrix{
		data_ptr: a,
		rs: a_rs,
		cs: a_cs,
	};
	let x = StridedMatrix{
		data_ptr: x,
		rs: incx,
		cs: 1,
	};
	corenum_gemv_f32f32f32(m, n, alpha, a, x, beta, y, incy, par);

}

pub unsafe fn corenum_sdot(
	n: usize,
	alpha: TA,
	x: *const TA, incx: usize,
	y: *const TB, incy: usize,
	beta: TC,
	res: *mut TC,
	par: &CorenumPar
) {
	corenum_sgemv(1, n, alpha, x, 1, incx, y, incy, beta, res, 1, par);
}

use corenum_base::corenum_gemm;


use haswell::{
	SupM, SupN,
};

use corenum_base::{
	GemmArray,
	StridedMatrixMut,
};


pub unsafe fn corenum_gemm_f32f32f32<
A: Copy+GemmArray<T=f32,U=f32> + SupN + Send + Sync+'static, 
B: Copy+GemmArray<T=f32,U=f32> + SupM + Send + Sync+'static,
>(
	m: usize, n: usize, k: usize,
	alpha: TA,
	a: A,
	b: B,
	beta: TC,
	c: StridedMatrixMut<f32>,
	par: &CorenumPar,
){	
	match *RUNTIME_HW_CONFIG {
		HWConfig::Haswell => {
			corenum_gemm::<TC, A, B, Identity, HaswellGemm>(
				m, n, k, alpha, a, b, beta, c, par
			);
		}
		HWConfig::Reference => {
			// corenum_gemm::<TA, TB, TC, InputA, InputB, ReferenceGemm>(
			// 	m, n, k, alpha, a, b, beta, c, c_rs, c_cs, par
			// );
		}
	}
}

pub unsafe fn packa_f32(
	m: usize, k: usize,
	a: *const TA,
	a_rs: usize, a_cs: usize,
	ap: *mut TA,
) {
	match *RUNTIME_HW_CONFIG {
		HWConfig::Haswell => {
			let mut ap = ap;
			for i in (0..m).step_by(4800) {
				let mc_len = if m >= (i + 4800) {4800} else {m - i};
				let mc_len_eff = (mc_len + 23) / 24 * 24;
				for p in (0..k).step_by(192) {
					let kc_len = if k >= (p + 192) {192} else {k - p};
					haswell::pack_panel::<24>(mc_len, kc_len, a.add(i*a_rs+p*a_cs), a_rs, a_cs, ap);
					ap = ap.add(mc_len_eff*kc_len);	
				}
			}
		}
		HWConfig::Reference => {
			// ReferencePackA::packa(m, k, a)
			unimplemented!()
		}
	}
}

pub unsafe fn corenum_sgemm(
	m: usize, n: usize, k: usize,
	alpha: TA,
	a: *const TA, a_rs: usize, a_cs: usize,
	b: *const TB, b_rs: usize, b_cs: usize,
	beta: TC,
	c: *mut TC, c_rs: usize, c_cs: usize,
	par: &CorenumPar,
) {
	// do not exchange if transa && transb
	let (m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b) = if c_cs == 1 && c_rs != 1 {
    	(n, m, b_rs, b_cs, a_rs, a_cs, c_cs, c_rs, b, a)
	} else {
    	(m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b)
	};
	let a = StridedMatrix{
		data_ptr: a,
		rs: a_rs,
		cs: a_cs,
	};
	let b = StridedMatrix{
		data_ptr: b,
		rs: b_rs,
		cs: b_cs,
	};
	let c = StridedMatrixMut{
		data_ptr: c,
		rs: c_rs,
		cs: c_cs,
	};
	corenum_gemm_f32f32f32(m, n, k, alpha, a, b, beta, c, par);

}




#[cfg(test)]
mod tests {
	use super::*;
	use corenum_dev::{
    	random_matrix,
    	check_gemm_f32,
	};

	const EPS: f64 = 2e-2;

	const M_ARR: [usize; 21] = [1, 2, 3, 17, 64, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144];
	const N_ARR: [usize; 21] = M_ARR;
	const K_ARR: [usize; 10] = [1, 8, 16, 64, 128, 129, 130, 131, 132, 509];
	const ALPHA_ARR: [f32; 4] = [1.0, 0.0, -1.0, 3.1415];
	const BETA_ARR: [f32; 4] = ALPHA_ARR;
	// const ALPHA_ARR: [f32; 1] = [1.0];
	// const BETA_ARR: [f32; 1] = ALPHA_ARR;
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
    	let d_par = CorenumPar::new(
        	4, 1, 1, 4, 1, 1
    	);
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
                                	&d_par,
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
                        	if diff_max >= EPS {
                            	println!("a: {:?}", a);
                            	println!("b: {:?}", b);
                            	println!("c: {:?}", c);
                            	println!("c_ref: {:?}", c_ref);
                        	}
                        	assert!(diff_max < EPS, "diff_max: {}, m: {}, n: {}, k: {}, alpha: {}, beta: {}", diff_max, m, n, k, alpha, beta);
                    	}
                	}
            	}
        	}
    	}
	}

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
