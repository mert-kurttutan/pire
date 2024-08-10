#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_arch;

#[cfg(target_arch = "aarch64")]
pub(crate) mod armv8;

// pub(crate) mod reference;

pub(crate) type TA = f32;
pub(crate) type TB = f32;
pub(crate) type TC = f32;

#[derive(Copy, Clone)]
pub(crate) struct NullFn;

pub(crate) trait MyFn: Copy + std::marker::Sync {
	fn call(self, c: *mut TC, m: usize);
}

impl MyFn for NullFn{
	#[inline(always)]
	fn call(self, _c: *mut TC, _m: usize) {}
}

impl MyFn for fn(*mut TC, m: usize){
	#[inline(always)]
	fn call(self, c: *mut TC, m: usize) {
		self(c, m);
	}
}

#[cfg(target_arch = "x86_64")]
use x86_64_arch::X86_64dispatcher;

// use reference::RefGemm;

use glare_base::{
	GemmCache,
	StridedMatrix,
	get_cache_params,
	is_simd_f32,
	Array,
	GlarePar,
	RUNTIME_HW_CONFIG,
};

#[inline(always)]
fn get_mcnckc() -> (usize, usize, usize) {
	if (*RUNTIME_HW_CONFIG).cpu_ft.avx512f {
		return (4800, 192, 512);
	}
	if (*RUNTIME_HW_CONFIG).cpu_ft.avx && (*RUNTIME_HW_CONFIG).cpu_ft.fma {
		return (4800, 320, 192);
	}
	// reference cache params
	get_cache_params()
}

pub(crate) unsafe fn glare_sgemm_generic<
F: MyFn,
>(
	m: usize, n: usize, k: usize,
	alpha: TA,
	a: Array<TA>,
	b: Array<TB>,
	beta: TC,
	c: Array<TC>,
	f: F,
) 
{
	let par = GlarePar::default();
	let (mc, nc, kc) = get_mcnckc();
	if is_simd_f32() {
		let x86_64_features = (*RUNTIME_HW_CONFIG).cpu_ft;
		let hw_config = X86_64dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, x86_64_features, NullFn{});
		x86_64_arch::glare_gemm(&hw_config, m, n, k, alpha, a, b, beta, c, &par);
		return;
	}

	// if none of the optimized paths are available, use reference implementation
	// let hw_config = RefGemm::new(&*RUNTIME_HW_CONFIG, mc, nc, kc);
	// glare_gemm(&hw_config, m, n, k, alpha, a, b, beta, c, &par);

}

pub unsafe fn glare_sgemm(
	m: usize, n: usize, k: usize,
	alpha: TA,
	a: *const TA, a_rs: usize, a_cs: usize,
	b: *const TB, b_rs: usize, b_cs: usize,
	beta: TC,
	c: *mut TC, c_rs: usize, c_cs: usize,
) {
	// transpose if c is row strided i.e. c_cs == 1 and c_rs != 1
	let (m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b) = if c_cs == 1 && c_rs != 1 {
    	(n, m, b_rs, b_cs, a_rs, a_cs, c_cs, c_rs, b, a)
	} else {
    	(m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b)
	};
	let a = StridedMatrix::new(a, a_rs, a_cs);
	let a = Array::StridedMatrix(a);
	let b = StridedMatrix::new(b, b_rs, b_cs);
	let b = Array::StridedMatrix(b);
	let c = StridedMatrix::new(c, c_rs, c_cs);
	let c = Array::StridedMatrix(c);
	let null_fn = NullFn{};
	glare_sgemm_generic(m, n, k, alpha, a, b, beta, c, null_fn);
}

pub unsafe fn glare_sgemm_fused(
	m: usize, n: usize, k: usize,
	alpha: TA,
	a: *const TA, a_rs: usize, a_cs: usize,
	b: *const TB, b_rs: usize, b_cs: usize,
	beta: TC,
	c: *mut TC, c_rs: usize, c_cs: usize,
	unary: fn(*mut TC, usize),
) {
	// transpose if c is row strided i.e. c_cs == 1 and c_rs != 1
	let (m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b) = if c_cs == 1 && c_rs != 1 {
    	(n, m, b_rs, b_cs, a_rs, a_cs, c_cs, c_rs, b, a)
	} else {
    	(m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b)
	};
	let a = StridedMatrix::new(a, a_rs, a_cs);
	let a = Array::StridedMatrix(a);
	let b = StridedMatrix::new(b, b_rs, b_cs);
	let b = Array::StridedMatrix(b);
	let c = StridedMatrix::new(c, c_rs, c_cs);
	let c = Array::StridedMatrix(c);
	glare_sgemm_generic(m, n, k, alpha, a, b, beta, c, unary);
}

pub unsafe fn glare_sgemv(
	m: usize, n: usize,
	alpha: TA,
	a: *const TA, a_rs: usize, a_cs: usize,
	x: *const TB, incx: usize,
	beta: TC,
	y: *mut TC, incy: usize,
) {
	glare_sgemm(
		m, 1, n,
		alpha,
		a, a_rs, a_cs,
		x, 1, incx,
		beta,
		y, 1, incy,
	)	
}
pub unsafe fn glare_sdot(
	n: usize,
	alpha: TA,
	x: *const TA, incx: usize,
	y: *const TB, incy: usize,
	beta: TC,
	res: *mut TC,
) {
	glare_sgemm(
		1, 1, n,
		alpha,
		x, incx, 1,
		y, incy, 1,
		beta,
		res, 1, 1,
	)
}

pub unsafe fn packa_f32(
	m: usize, k: usize,
	a: *const TA,
	a_rs: usize, a_cs: usize,
	ap: *mut TA,
) {
	let align_offset = ap.align_offset(256);
	let ap = ap.add(align_offset);
	if m == 1 || k == 1 {
		for i in 0..m {
			for j in 0..k {
				*ap.add(i*k+j) = *a.add(i*a_rs+j*a_cs);
			}
		}
		return;
	}

	#[cfg(target_arch = "x86_64")]
	{
		// let avx = hw_avx();
		// let fma = hw_fma();
		// let model = hw_model();
		// let avx512f = hw_avx512f();
		// if avx512f {
		// 	match model {
		// 		_ => {
		// 			const MC: usize = 4800;
		// 			const MR: usize = 48;
		// 			const KC: usize = 512;
		// 			for i in (0..m).step_by(MC) {
		// 				let mc_len = if m >= (i + MC) {MC} else {m - i};
		// 				let mc_len_eff = (mc_len + MR-1) / MR * MR;
		// 				for p in (0..k).step_by(KC) {
		// 					let kc_len = if k >= (p + KC) {KC} else {k - p};
		// 					// avx512f::packa_panel::<MR>(mc_len, kc_len, a.add(i*a_rs+p*a_cs), a_rs, a_cs, ap);
		// 					ap = ap.add(mc_len_eff*kc_len);	
		// 				}
		// 			}
		// 		}
		// 	}
		// 	return;
		
		// }
		// if avx && fma {
		// 	match model {
		// 		_ => {
		// 			const MC: usize = 4800;
		// 			const MR: usize = 24;
		// 			const KC: usize = 192;
		// 			for i in (0..m).step_by(MC) {
		// 				let mc_len = if m >= (i + MC) {MC} else {m - i};
		// 				let mc_len_eff = (mc_len + MR-1) / MR * MR;
		// 				for p in (0..k).step_by(KC) {
		// 					let kc_len = if k >= (p + KC) {KC} else {k - p};
		// 					// avx_fma::packa_panel::<MR>(mc_len, kc_len, a.add(i*a_rs+p*a_cs), a_rs, a_cs, ap);
		// 					ap = ap.add(mc_len_eff*kc_len);	
		// 				}
		// 			}
		// 		}
		// 	}
		// 	return;
		// }
	}

}


#[cfg(test)]
mod tests {
	use super::*;
	use glare_dev::{
    	random_matrix_uniform,
    	check_gemm_f32,
	};

	const EPS: f64 = 2e-2;

	// static M_ARR: [usize; 32] = [1, 2, 3, 16, 32, 24, 37, 38, 17, 32, 48, 64, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 958, 959, 960, 950, 951, 943, 944];
	static M_ARR: [usize; 32] = [1, 2, 3, 16, 32, 24, 37, 38, 17, 32, 48, 64, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 458, 459, 460, 450, 451, 443, 444];
	static N_ARR: [usize; 28] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 64, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 658, 659, 660];
	static K_ARR: [usize; 10] = [1, 8, 16, 64, 128, 129, 130, 131, 132, 509];
	static ALPHA_ARR: [f32; 2] = [1.0, 2.17];
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
                        	random_matrix_uniform(m, k, &mut a, m);
                        	random_matrix_uniform(k, n, &mut b, k);
                        	random_matrix_uniform(m, n, &mut c, m);
                        	c_ref.copy_from_slice(&c);
                        	unsafe {
                            	glare_sgemm(
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
									EPS,
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
                        	random_matrix_uniform(m, k, &mut a, m);
                        	random_matrix_uniform(k, n, &mut b, k);
                        	random_matrix_uniform(m, n, &mut c, m);
                        	c_ref.copy_from_slice(&c);
							unsafe {
								packa_f32(m, k, a.as_ptr(), a_rs, a_cs, ap_mut_ptr);
							}
							let ap_matrix = glare_base::PackedMatrix{
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
                        	// unsafe {
                            // 	glare_gemm_f32f32f32(
                            //     	m, n, k,
                            //     	alpha,
                            //     	ap_matrix,
                            //     	b_matrix,
                            //     	beta,
                            //     	c_matrix,
                            // 	);
                        	// }
                        	let diff_max = unsafe { 
								check_gemm_f32(
									m, n, k,
									alpha,
									a.as_ptr(), a_rs, a_cs,
									b.as_ptr(), b_rs, b_cs,
									beta,
									&mut c, c_rs, c_cs,
									&mut c_ref,
									EPS,
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
