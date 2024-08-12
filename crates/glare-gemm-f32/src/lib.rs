#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_arch;

#[cfg(target_arch = "aarch64")]
pub(crate) mod armv8;

pub(crate) mod reference;

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

use reference::RefGemm;

use glare_base::{
	GemmCache,
	StridedMatrix,
	StridedMatrixMut,
	Array,
	ArrayMut,
	GlarePar,
	RUNTIME_HW_CONFIG,
	get_cache_params,
	has_f32_compute,
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
	c: ArrayMut<TC>,
	f: F,
) 
{
	let par = GlarePar::default();
	let (mc, nc, kc) = get_mcnckc();
	if has_f32_compute() {
		let x86_64_features = (*RUNTIME_HW_CONFIG).cpu_ft;
		let hw_config = X86_64dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, x86_64_features, f);
		x86_64_arch::glare_gemm(&hw_config, m, n, k, alpha, a, b, beta, c, &par);
		return;
	}
	// if none of the optimized paths are available, use reference implementation
	let hw_config = RefGemm::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, f);
	reference::glare_gemm(&hw_config, m, n, k, alpha, a, b, beta, c, &par);

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
	let c = StridedMatrixMut::new(c, c_rs, c_cs);
	let c = ArrayMut::StridedMatrix(c);
	let null_fn = NullFn{};
	glare_sgemm_generic(m, n, k, alpha, a, b, beta, c, null_fn);
}

#[cfg(feature = "fuse")]
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
	let c = StridedMatrixMut::new(c, c_rs, c_cs);
	let c = ArrayMut::StridedMatrix(c);
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
) -> Array<TA> {
	let align_offset = ap.align_offset(256);
	let mut ap = ap.add(align_offset);
	let ap0 = ap;
	if m == 1 || k == 1 {
		for j in 0..k {
			for i in 0..m {
				*ap.add(j*m+i) = *a.add(i*a_rs+j*a_cs);
			}
		}
		return Array::StridedMatrix(StridedMatrix{
			data_ptr: ap0 as *const f32,
			rs: 1,
			cs: m,
		});
	}
	let par = GlarePar::default();
	let (mc, nc, kc) = get_mcnckc();
	let x86_64_features = (*RUNTIME_HW_CONFIG).cpu_ft;
	let hw_config = X86_64dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, x86_64_features, NullFn{});
	// if none of the optimized paths are available, use reference implementation
	let hw_config_ref = RefGemm::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn{});


	#[cfg(target_arch = "x86_64")]
	{
		let vs = if has_f32_compute() {hw_config.vs} else {hw_config_ref.vs};
		for i in (0..m).step_by(mc) {
			let mc_len = if m >= (i + mc) {mc} else {m - i};
			let mc_len_eff = (mc_len + vs-1) / vs * vs;
			for p in (0..k).step_by(kc) {
				let kc_len = if k >= (p + kc) {kc} else {k - p};
				let a_cur = a.add(i*a_rs+p*a_cs);
				if has_f32_compute() {
					hw_config.packa_fn(a_cur, ap, mc_len, kc_len, a_rs, a_cs);
				} else {
					hw_config_ref.packa_fn(a_cur, ap, mc_len, kc_len, a_rs, a_cs);
				}
				ap = ap.add(mc_len_eff*kc_len);	
			}
		}
		return Array::PackedMatrix(glare_base::PackedMatrix{
			data_ptr: ap0 as *const f32,
			mc: mc,
			kc: kc,
			k,
			m,
		});
	}
}

pub unsafe fn packb_f32(
	n: usize, k: usize,
	b: *const TB,
	b_rs: usize, b_cs: usize,
	bp: *mut TB,
) -> Array<TB> {
	let align_offset = bp.align_offset(512);
	let mut bp = bp.add(align_offset);
	let bp0 = bp;
	if n == 1 || k == 1 {
		for i in 0..n {
			for j in 0..k {
				*bp.add(i*k+j) = *b.add(i*b_cs+j*b_rs);
			}
		}
		return Array::StridedMatrix(StridedMatrix{
			data_ptr: bp0 as *const f32,
			rs: 1,
			cs: k,
		});
	}
	let (mc, nc, kc) = get_mcnckc();
	let hw_config_ref = RefGemm::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn{});

	#[cfg(target_arch = "x86_64")]
	{
		let (mc, nc, kc) = get_mcnckc();
		let x86_64_features = (*RUNTIME_HW_CONFIG).cpu_ft;
		let hw_config = X86_64dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, x86_64_features, NullFn{});
		for i in (0..n).step_by(nc) {
			let nc_len = if n >= (i + nc) {nc} else {n - i};
			let nc_len_eff = nc_len; // (nc_len + nr-1) / nr * nr;
			for p in (0..k).step_by(kc) {
				let kc_len = if k >= (p + kc) {kc} else {k - p};
				let b_cur = b.add(i*b_cs+p*b_rs);
				if has_f32_compute() {
					hw_config.packb_fn(b_cur, bp, nc_len, kc_len, b_rs, b_cs);
				} else {
					hw_config_ref.packb_fn(b_cur, bp, nc_len, kc_len, b_rs, b_cs);
				}
				bp = bp.add(nc_len_eff*kc_len);	
			}
		}
		return Array::PackedMatrix(glare_base::PackedMatrix{
			data_ptr: bp0 as *const f32,
			mc: nc,
			kc: kc,
			k,
			m: n,
		});
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
	static M_ARR: [usize; 33] = [1, 2, 3, 16, 32, 24, 37, 38, 17, 32, 48, 64, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 458, 459, 460, 450, 451, 443, 444, 4879];
	static N_ARR: [usize; 28] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 64, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 658, 659, 660];
	static K_ARR: [usize; 10] = [1, 8, 16, 64, 128, 129, 130, 131, 132, 509];
	static ALPHA_ARR: [f32; 1] = [1.0];
	static BETA_ARR: [f32; 1] = [1.0];
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
							let ap_matrix = unsafe {packa_f32(m, k, a.as_ptr(), a_rs, a_cs, ap_mut_ptr)};
							let b_matrix = StridedMatrix{
								data_ptr: b.as_ptr(),
								rs: b_rs, cs: b_cs,
							};
							let b_matrix = Array::StridedMatrix(b_matrix);
							let c_matrix = StridedMatrixMut{
								data_ptr: c.as_mut_ptr(),
								rs: c_rs, cs: c_cs,
							};
							let c_matrix = ArrayMut::StridedMatrix(c_matrix);
                        	unsafe {
                            	glare_sgemm_generic(
                                	m, n, k,
                                	alpha,
                                	ap_matrix,
                                	b_matrix,
                                	beta,
                                	c_matrix,
									NullFn{},
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
	fn test_gemm_bp(layout: &Layout) {
    	for m in M_ARR {
        	for n in N_ARR {
            	let mut c = vec![0.0; m * n];
            	let mut c_ref = vec![0.0; m * n];
            	for k in K_ARR {
                	let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = dispatch_strides(&layout, m, n, k);
                	let mut a = vec![0.0; m * k];
                	let mut b = vec![0.0; k * n];
					let mut bp = vec![0_f32; (n+100)*k+512];
					let bp_offset = bp.as_ptr().align_offset(512);
					let bp_mut_ptr = unsafe {bp.as_mut_ptr().add(bp_offset)};
					let bp_ptr = bp_mut_ptr as *const f32;
                	for alpha in ALPHA_ARR {
                    	for beta in ALPHA_ARR {
                        	random_matrix_uniform(m, k, &mut a, m);
                        	random_matrix_uniform(k, n, &mut b, k);
                        	random_matrix_uniform(m, n, &mut c, m);
                        	c_ref.copy_from_slice(&c);
							let a_matrix = StridedMatrix{
								data_ptr: a.as_ptr(),
								rs: a_rs, cs: a_cs,
							};
							let a_matrix = Array::StridedMatrix(a_matrix);
							let bp_matrix = unsafe {packb_f32(n, k, b.as_ptr(), b_rs, b_cs, bp_mut_ptr)};

							let c_matrix = StridedMatrixMut{
								data_ptr: c.as_mut_ptr(),
								rs: c_rs, cs: c_cs,
							};
							let c_matrix = ArrayMut::StridedMatrix(c_matrix);
                        	unsafe {
                            	glare_sgemm_generic(
                                	m, n, k,
                                	alpha,
                                	a_matrix,
                                	bp_matrix,
                                	beta,
                                	c_matrix,
									NullFn{},
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
	#[test]
	fn test_nn_col_ap() {
    	test_gemm_ap(&Layout::NN);
	}
	#[test]
	fn test_nt_col_ap() {
    	test_gemm_ap(&Layout::NN);
	}
	#[test]
	fn test_tn_col_ap() {
    	test_gemm_ap(&Layout::NN);
	}
	#[test]
	fn test_tt_col_ap() {
    	test_gemm_ap(&Layout::NN);
	}
	#[test]
	fn test_nn_col_bp() {
    	test_gemm_bp(&Layout::NN);
	}
	#[test]
	fn test_nt_col_bp() {
    	test_gemm_bp(&Layout::NT);
	}
	#[test]
	fn test_tn_col_bp() {
    	test_gemm_bp(&Layout::TN);
	}
	#[test]
	fn test_tt_col_bp() {
    	test_gemm_bp(&Layout::TT);
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
