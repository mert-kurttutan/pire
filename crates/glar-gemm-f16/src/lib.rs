#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_arch;

pub(crate) mod reference;

pub(crate) type TA = f16;
pub(crate) type TB = f16;
pub(crate) type TC = f16;

pub use half::f16;

use glar_base::{
    ap_size, bp_size, has_f16_compute, has_f16f32_compute, Array, ArrayMut, GemmCache, GlarPar, RUNTIME_HW_CONFIG,
};

use reference::RefGemm;

#[derive(Copy, Clone)]
pub(crate) struct NullFn;

pub(crate) trait MyFn: Copy + std::marker::Sync {
    unsafe fn call(self, c: *mut TC, m: usize);
}

impl MyFn for NullFn {
    #[inline(always)]
    unsafe fn call(self, _c: *mut TC, _m: usize) {}
}

impl MyFn for unsafe fn(*mut TC, m: usize) {
    #[inline(always)]
    unsafe fn call(self, c: *mut TC, m: usize) {
        self(c, m);
    }
}

use glar_base::{load_buf, store_buf};

pub(crate) unsafe fn glar_hgemm_generic<F: MyFn>(
    m: usize,
    n: usize,
    k: usize,
    alpha: TA,
    a: Array<TA>,
    b: Array<TB>,
    beta: TC,
    c: ArrayMut<TC>,
    f: F,
) {
    let par = GlarPar::default(m, n);
    if has_f16_compute() {
        let hw_config = x86_64_arch::F16Dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, f);
        x86_64_arch::glar_gemm_native(&hw_config, m, n, k, alpha, a, b, beta, c, &par);
        return;
    }
    if has_f16f32_compute() {
        let hw_config = x86_64_arch::F32Dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, f);
        x86_64_arch::glar_gemm(&hw_config, m, n, k, alpha.to_f32(), a, b, beta.to_f32(), c, &par);
        return;
    }

    // if none of the optimized paths are available, use reference implementation
    let hw_config = RefGemm::from_hw_cfg(&*RUNTIME_HW_CONFIG, f);
    reference::glar_gemm(&hw_config, m, n, k, alpha.to_f32(), a, b, beta.to_f32(), c, &par);
}

pub unsafe fn glar_hgemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f16,
    a: *const f16,
    a_rs: usize,
    a_cs: usize,
    b: *const f16,
    b_rs: usize,
    b_cs: usize,
    beta: f16,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
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
    let null_fn = NullFn {};
    glar_hgemm_generic(m, n, k, alpha, a, b, beta, c, null_fn);
}

#[cfg(feature = "fuse")]
pub unsafe fn glar_hgemm_fused(
    m: usize,
    n: usize,
    k: usize,
    alpha: f16,
    a: *const f16,
    a_rs: usize,
    a_cs: usize,
    b: *const f16,
    b_rs: usize,
    b_cs: usize,
    beta: f16,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
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
    glar_hgemm_generic(m, n, k, alpha, a, b, beta, c, unary);
}

// block idx for packa and packb is s.t.
// m dim for block idx is contiguous and n dim is contiguous
// this is to ensure that indexing for parallelization over these dims are easy  (otherwise ranges would have to be in the same mc, nc range)
// this is not an issue since we do not parallelize over k dim (think about this when we parallelize over k dim in the future, which is only beneficial only
// in the special case of very large k and small m, n
pub unsafe fn packa_f16(m: usize, k: usize, a: *const TA, a_rs: usize, a_cs: usize, ap: *mut TA) -> Array<TA> {
    assert_eq!(ap.align_offset(glar_base::AB_ALIGN), 0);
    if m == 1 {
        for j in 0..k {
            *ap.add(j) = *a.add(j * a_cs);
        }
        return Array::strided_matrix(ap, 1, m);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_f16_compute() {
            return x86_64_arch::packa_full_f16(m, k, a, a_rs, a_cs, ap);
        }
        if has_f16f32_compute() {
            return x86_64_arch::packa_full_f32(m, k, a, a_rs, a_cs, ap);
        }
    }

    #[cfg(target_arch = "x86")]
    {
        if has_f32_compute() {
            return x86_arch::packa_full(m, k, a, a_rs, a_cs, ap);
        }
    }
    reference::packa_full(m, k, a, a_rs, a_cs, ap)
}

pub unsafe fn packb_f16(n: usize, k: usize, b: *const TB, b_rs: usize, b_cs: usize, bp: *mut TB) -> Array<TB> {
    assert_eq!(bp.align_offset(glar_base::AB_ALIGN), 0);
    if n == 1 {
        for j in 0..k {
            *bp.add(j) = *b.add(j * b_rs);
        }
        return Array::strided_matrix(bp, 1, k);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_f16_compute() {
            return x86_64_arch::packb_full_f16(n, k, b, b_rs, b_cs, bp);
        }
        if has_f16f32_compute() {
            return x86_64_arch::packb_full_f32(n, k, b, b_rs, b_cs, bp);
        }
    }
    #[cfg(target_arch = "x86")]
    {
        if has_f32_compute() {
            return x86_arch::packb_full(n, k, b, b_rs, b_cs, bp);
        }
    }
    reference::packb_full(n, k, b, b_rs, b_cs, bp)
}

pub unsafe fn packa_f16_with_ref(m: usize, k: usize, a: &[TA], a_rs: usize, a_cs: usize, ap: &mut [TA]) -> Array<TA> {
    let pack_size = ap_size::<TA>(m, k);
    let ap_align_offset = ap.as_ptr().align_offset(glar_base::AB_ALIGN);
    // safety check
    assert!(ap.len() >= pack_size);
    let ap = &mut ap[ap_align_offset..];
    unsafe { packa_f16(m, k, a.as_ptr(), a_rs, a_cs, ap.as_mut_ptr()) }
}

pub unsafe fn packb_f16_with_ref(n: usize, k: usize, b: &[TB], b_rs: usize, b_cs: usize, bp: &mut [TB]) -> Array<TB> {
    let pack_size = bp_size::<TB>(n, k);
    let bp_align_offset = bp.as_ptr().align_offset(glar_base::AB_ALIGN);
    // safety check
    assert!(bp.len() >= pack_size);
    let bp = &mut bp[bp_align_offset..];
    unsafe { packb_f16(n, k, b.as_ptr(), b_rs, b_cs, bp.as_mut_ptr()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glar_base::{get_cache_params, matrix_size};
    use glar_dev::{
        check_gemm_f16, generate_k_dims, generate_m_dims, generate_n_dims, layout_to_strides, random_matrix_uniform,
        ABLayout,
    };

    #[inline(always)]
    #[allow(unreachable_code)]
    pub(crate) fn get_mcnckc() -> (usize, usize, usize) {
        #[cfg(target_arch = "x86_64")]
        {
            return x86_64_arch::get_mcnckc();
        }
        get_cache_params()
    }

    unsafe fn my_unary(c: *mut TC, m: usize) {
        for i in 0..m {
            *c.add(i) *= f16::from_f32(2.0);
        }
    }

    // fn my_unary(_c: *mut TC, _m: usize) {}

    const EPS: f64 = 4e-1;

    // static ALPHA_ARR: [f32; 2] = [1.0, 3.1415];
    // static BETA_ARR: [f32; 3] = [1.0, 3.1415, 0.0];
    static ALPHA_ARR: [f32; 1] = [1.0];
    static BETA_ARR: [f32; 1] = [1.0];

    fn test_gemm(layout: &ABLayout, is_a_packed: bool, is_b_packed: bool) {
        let (mc, nc, kc) = get_mcnckc();
        let (mr, nr, kr) = (48, 8, 8);
        let m_dims = generate_m_dims(mc, mr);
        let n_dims = generate_n_dims(nc, nr);
        let k_dims = generate_k_dims(kc, kr);
        let unary_fn: unsafe fn(*mut TC, usize) = my_unary;
        for m in m_dims.iter() {
            let m = *m;
            let (c_rs, c_cs) = (1, m);
            for n in n_dims.iter() {
                let n = *n;
                let c_size = matrix_size(c_rs, c_cs, m, n);
                let mut c = vec![f16::ZERO; c_size];
                let mut c_ref = vec![f16::ZERO; c_size];
                for k in k_dims.iter() {
                    let k = *k;
                    let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = layout_to_strides(&layout, m, n, k);
                    let mut a = vec![f16::ZERO; m * k];
                    let mut b = vec![f16::ZERO; k * n];
                    random_matrix_uniform(m, k, &mut a, m);
                    random_matrix_uniform(k, n, &mut b, k);
                    let ap_size = if is_a_packed { ap_size::<TA>(m, k) } else { 0 };
                    let mut ap = vec![TA::ZERO; ap_size];
                    let a_matrix = if is_a_packed {
                        unsafe { packa_f16_with_ref(m, k, &a, a_rs, a_cs, &mut ap) }
                    } else {
                        Array::strided_matrix(a.as_ptr(), a_rs, a_cs)
                    };
                    let bp_size = if is_b_packed { bp_size::<TB>(n, k) } else { 0 };
                    let mut bp = vec![TB::ZERO; bp_size];
                    let b_matrix = if is_b_packed {
                        unsafe { packb_f16_with_ref(n, k, &b, b_rs, b_cs, &mut bp) }
                    } else {
                        Array::strided_matrix(b.as_ptr(), b_rs, b_cs)
                    };
                    for alpha in ALPHA_ARR {
                        for beta in BETA_ARR {
                            let alpha = f16::from_f32(alpha);
                            let beta = f16::from_f32(beta);
                            random_matrix_uniform(m, n, &mut c, m);
                            c_ref.copy_from_slice(&c);
                            let c_matrix = ArrayMut::strided_matrix(c.as_mut_ptr(), c_rs, c_cs);
                            unsafe {
                                glar_hgemm_generic(m, n, k, alpha, a_matrix, b_matrix, beta, c_matrix, unary_fn);
                            }
                            let diff_max = unsafe {
                                check_gemm_f16(
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
                                    &mut c,
                                    c_rs,
                                    c_cs,
                                    &mut c_ref,
                                    unary_fn,
                                    EPS,
                                )
                            };
                            // if diff_max >= EPS {
                            // 	println!("a: {:?}", a);
                            // 	println!("b: {:?}", b);
                            // 	println!("c:     {:?}", c);
                            // 	println!("c_ref: {:?}", c_ref);
                            // }
                            assert!(
                                diff_max < EPS,
                                "diff_max: {}, m: {}, n: {}, k: {}, alpha: {}, beta: {}",
                                diff_max,
                                m,
                                n,
                                k,
                                alpha,
                                beta
                            );
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
