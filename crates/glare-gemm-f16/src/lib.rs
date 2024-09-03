#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_arch;

#[cfg(target_arch = "aarch64")]
pub(crate) mod armv8;

pub(crate) mod reference;

pub(crate) type TA = f16;
pub(crate) type TB = f16;
pub(crate) type TC = f16;

pub use half::f16;

// #[cfg(target_arch = "x86_64")]
// use x86_64_arch::{
// 	F32Dispatcher, F16Dispatcher,
// };

use glare_base::{
	GemmCache,
	Array,
	ArrayMut,
	GlarePar,
	RUNTIME_HW_CONFIG,
	get_cache_params,
	has_f32_compute,
	has_f16_compute,
};

use reference::RefGemm;

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


#[inline(always)]
fn get_mcnckc() -> (usize, usize, usize) {
	if (*RUNTIME_HW_CONFIG).cpu_ft.avx512f {
		match (*RUNTIME_HW_CONFIG).hw_model {
			_ => {
				return (4800, 192, 512);
			}
		}
	}
	if (*RUNTIME_HW_CONFIG).cpu_ft.avx && (*RUNTIME_HW_CONFIG).cpu_ft.fma {
		match (*RUNTIME_HW_CONFIG).hw_model {
			_ => {
				return (4800, 320, 192);
			}
		}
	}
	if (*RUNTIME_HW_CONFIG).cpu_ft.avx {
		match (*RUNTIME_HW_CONFIG).hw_model {
			_ => {
				return (4800, 320, 192);
			}
		}
	}
	// reference cache params
	get_cache_params()
}

pub(crate) unsafe fn glare_hgemm_generic<
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
	if has_f16_compute() {
		let hw_config = x86_64_arch::F16Dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, f);
		x86_64_arch::glare_gemm_native(&hw_config, m, n, k, alpha, a, b, beta, c, &par);
		return;
	}
	if has_f32_compute() {
		let hw_config = x86_64_arch::F32Dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, f);
		x86_64_arch::glare_gemm(&hw_config, m, n, k, alpha.to_f32(), a, b, beta.to_f32(), c, &par);
		return;
	}

	// if none of the optimized paths are available, use reference implementation
	let hw_config = RefGemm::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, f);
	reference::glare_gemm(&hw_config, m, n, k, alpha.to_f32(), a, b, beta.to_f32(), c, &par);
}



pub unsafe fn glare_hgemm(
	m: usize, n: usize, k: usize,
	alpha: f16,
	a: *const f16, a_rs: usize, a_cs: usize,
	b: *const f16, b_rs: usize, b_cs: usize,
	beta: f16,
	c: *mut f16, c_rs: usize, c_cs: usize,
) {
	// do not exchange if transa && transb
	let (m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b) = if c_cs == 1 && c_rs != 1 {
    	(n, m, b_rs, b_cs, a_rs, a_cs, c_cs, c_rs, b, a)
	} else {
    	(m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b)
	};
	let a = Array::strided_matrix(a, a_rs, a_cs);
	let b = Array::strided_matrix(b, b_rs, b_cs);
	let c = ArrayMut::strided_matrix(c, c_rs, c_cs);
	let null_fn = NullFn{};
	glare_hgemm_generic(m, n, k, alpha, a, b, beta, c, null_fn);
}

#[cfg(feature = "fuse")]
pub unsafe fn glare_hgemm_fused(
	m: usize, n: usize, k: usize,
	alpha: f16,
	a: *const f16, a_rs: usize, a_cs: usize,
	b: *const f16, b_rs: usize, b_cs: usize,
	beta: f16,
	c: *mut f16, c_rs: usize, c_cs: usize,
	unary: fn(*mut f16, usize),
) {
	// transpose if c is row strided i.e. c_cs == 1 and c_rs != 1
	let (m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b) = if c_cs == 1 && c_rs != 1 {
    	(n, m, b_rs, b_cs, a_rs, a_cs, c_cs, c_rs, b, a)
	} else {
    	(m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b)
	};
	let a = Array::strided_matrix(a, a_rs, a_cs);
	let b = Array::strided_matrix(b, b_rs, b_cs);
	let c = ArrayMut::strided_matrix(c, c_rs, c_cs);
	glare_hgemm_generic(m, n, k, alpha, a, b, beta, c, unary);
}


pub unsafe fn packa_f16(
	m: usize, k: usize,
	a: *const f16,
	a_rs: usize, a_cs: usize,
	ap: *mut f16,
) -> Array<f16> {
	let align_offset = ap.align_offset(256);
	let mut ap = ap.add(align_offset);
	let ap0 = ap;
	if m == 1 {
		for j in 0..k {
			*ap.add(j) = *a.add(j*a_cs);
		}
		return Array::strided_matrix(ap0, 1, m);
	}
	let (mc, nc, kc) = get_mcnckc();
	let hw_config = x86_64_arch::F32Dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn{});
	let hw_config_f16 = x86_64_arch::F16Dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn{});
	// if none of the optimized paths are available, use reference implementation
	let hw_config_ref = RefGemm::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn{});
	// let vs = if has_f32_compute() {hw_config.vs} else {hw_config_ref.vs};
	let vs = {
		if has_f16_compute() {
			hw_config_f16.vs
		} else if has_f32_compute() {
			hw_config.vs
		} else {
			hw_config_ref.vs
		}
	};

	#[cfg(target_arch = "x86_64")]
	{
		for p in (0..k).step_by(kc) {
			let kc_len = if k >= (p + kc) {kc} else {k - p};
			for i in (0..m).step_by(mc) {
				let mc_len = if m >= (i + mc) {mc} else {m - i};
				let mc_len_eff = (mc_len + vs-1) / vs * vs;
				let a_cur = a.add(i*a_rs+p*a_cs);
				if has_f16_compute() {
					hw_config_f16.packa_fn(a_cur, ap, mc_len, kc_len, a_rs, a_cs);
				} else if has_f32_compute() {
					hw_config.packa_fnsame(a_cur, ap, mc_len, kc_len, a_rs, a_cs);
				} else {
					hw_config_ref.packa_fnsame(a_cur, ap, mc_len, kc_len, a_rs, a_cs);
				}
				ap = ap.add(mc_len_eff*kc_len);	
			}
		}
		return Array::packed_matrix(ap0, mc, kc, m, k);
	}
}

pub unsafe fn packb_f16(
	n: usize, k: usize,
	b: *const f16,
	b_rs: usize, b_cs: usize,
	bp: *mut f16,
) -> Array<f16>{
	let align_offset = bp.align_offset(512);
	let mut bp = bp.add(align_offset);
	let bp0 = bp;
	if n == 1 {
		for j in 0..k {
			*bp.add(j) = *b.add(j*b_rs);
		}
		return Array::strided_matrix(bp0, 1, k);
	}
	let (mc, nc, kc) = get_mcnckc();
	let hw_config_ref = RefGemm::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn{});
	let hw_config = x86_64_arch::F32Dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn{});
	let hw_config_f16 = x86_64_arch::F16Dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn{});

	#[cfg(target_arch = "x86_64")]
	{
		for p in (0..k).step_by(kc) {
			let kc_len = if k >= (p + kc) {kc} else {k - p};
			for i in (0..n).step_by(nc) {
				let nc_len = if n >= (i + nc) {nc} else {n - i};
				let nc_len_eff = nc_len; // (nc_len + nr-1) / nr * nr;
				let b_cur = b.add(i*b_cs+p*b_rs);
				if has_f16_compute() {
					hw_config_f16.packb_fn(b_cur, bp, nc_len, kc_len, b_rs, b_cs);
				} else if has_f32_compute() {
					hw_config.packb_fnsame(b_cur, bp, nc_len, kc_len, b_rs, b_cs);
				} else {
					hw_config_ref.packb_fnsame(b_cur, bp, nc_len, kc_len, b_rs, b_cs);
				}
				bp = bp.add(nc_len_eff*kc_len);	 
			}
		}
		return Array::packed_matrix(bp0, nc, kc, n, k);
	}
}




#[cfg(test)]
mod tests {
	use super::*;
	use glare_dev::{
    	random_matrix_uniform,
    	check_gemm_f16,
		generate_m_dims, generate_n_dims, generate_k_dims,
		ABLayout, layout_to_strides,
	};

	const EPS: f64 = 4e-1;

	static ALPHA_ARR: [f32; 2] = [1.0, 3.1415];
	static BETA_ARR: [f32; 1] = [1.0];

	fn test_gemm(layout: &ABLayout, is_a_packed: bool, is_b_packed: bool) {
		let (mc, nc, kc) = get_mcnckc();
		let (mr, nr, kr) = (48, 8, 8);
		let m_dims = generate_m_dims(mc, mr);
		let n_dims = generate_n_dims(nc, nr);
		let k_dims = generate_k_dims(kc, kr);
    	for m in m_dims.iter() {
			let m = *m;
        	for n in n_dims.iter() {
				let n = *n;
            	let mut c = vec![f16::ZERO; m * n];
            	let mut c_ref = vec![f16::ZERO; m * n];
            	for k in k_dims.iter() {
					let k = *k;
                	let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = layout_to_strides(&layout, m, n, k);
                	let mut a = vec![f16::ZERO; m * k];
                	let mut b = vec![f16::ZERO; k * n];
					random_matrix_uniform(m, k, &mut a, m);
					random_matrix_uniform(k, n, &mut b, k);
					let ap_size = if is_a_packed { (m+100)*k+512 } else {1024};
					let mut ap = vec![f16::ZERO; ap_size];
					let ap_offset = ap.as_ptr().align_offset(512);
					let ap_mut_ptr = unsafe {ap.as_mut_ptr().add(ap_offset)};
					let a_matrix = if is_a_packed {
						unsafe {packa_f16(m, k, a.as_ptr(), a_rs, a_cs, ap_mut_ptr)}
					} else {
						unsafe{Array::strided_matrix(a.as_ptr(), a_rs, a_cs)}
					};
					let bp_size = if is_b_packed { (n+100)*k+512 } else {1024};
					let mut bp = vec![f16::ZERO; bp_size];
					let bp_offset = bp.as_ptr().align_offset(512);
					let bp_mut_ptr = unsafe {bp.as_mut_ptr().add(bp_offset)};
					let b_matrix = if is_b_packed {
						unsafe {packb_f16(n, k, b.as_ptr(), b_rs, b_cs, bp_mut_ptr)}
					} else {
						unsafe{Array::strided_matrix(b.as_ptr(), b_rs, b_cs)}
					};
                	for alpha in ALPHA_ARR {
                    	for beta in ALPHA_ARR {
							let alpha = f16::from_f32(alpha);
							let beta = f16::from_f32(beta);
                        	random_matrix_uniform(m, n, &mut c, m);
                        	c_ref.copy_from_slice(&c);
							let c_matrix = unsafe {
								ArrayMut::strided_matrix(c.as_mut_ptr(), c_rs, c_cs)
							};
                        	unsafe {
                            	glare_hgemm_generic(
                                	m, n, k,
                                	alpha,
                                	a_matrix,
                                	b_matrix,
                                	beta,
                                	c_matrix,
									NullFn{},
                            	);
                        	}
                        	let diff_max = unsafe { 
								check_gemm_f16(
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
    	test_gemm(&ABLayout::NN, true, false);
	}
	#[test]
	fn test_nt_col_ap() {
    	test_gemm(&ABLayout::NT, true, false);
	}
	#[test]
	fn test_tn_col_ap() {
    	test_gemm(&ABLayout::TN, true, false);
	}
	#[test]
	fn test_tt_col_ap() {
    	test_gemm(&ABLayout::TT, true, false);
	}
	#[test]
	fn test_nn_col_bp() {
    	test_gemm(&ABLayout::NN, false, true);
	}
	#[test]
	fn test_nt_col_bp() {
    	test_gemm(&ABLayout::NT, false, true);
	}
	#[test]
	fn test_tn_col_bp() {
    	test_gemm(&ABLayout::TN, false, true);
	}
	#[test]
	fn test_tt_col_bp() {
    	test_gemm(&ABLayout::TT, false, true);
	}
	#[test]
	fn test_nn_col() {
    	test_gemm(&ABLayout::NN, false, false);
	}
	#[test]
	fn test_nt_col() {
    	test_gemm(&ABLayout::NT, false, false);
	}
	#[test]
	fn test_tn_col() {
    	test_gemm(&ABLayout::TN, false, false);
	}
	#[test]
	fn test_tt_col() {
    	test_gemm(&ABLayout::TT, false, false);
	}
	#[test]
	fn test_nn_col_apbp() {
		test_gemm(&ABLayout::NN, true, true);
	}
	#[test]
	fn test_nt_col_apbp() {
		test_gemm(&ABLayout::NT, true, true);
	}
	#[test]
	fn test_tn_col_apbp() {
		test_gemm(&ABLayout::TN, true, true);
	}
	#[test]
	fn test_tt_col_apbp() {
		test_gemm(&ABLayout::TT, true, true);
	}

}
