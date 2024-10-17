#[cfg(target_arch = "aarch64")]
pub(crate) mod arm64;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_arch;
#[cfg(target_arch = "x86")]
pub(crate) mod x86_arch;

#[cfg(target_arch = "x86_64")]
use x86_64_arch::{glar_gemm, packa_full, packb_full, KernelDispatcher};

#[cfg(target_arch = "x86")]
use x86_arch::{glar_gemm, packa_full, packb_full, KernelDispatcher};

#[cfg(target_arch = "aarch64")]
use arm64::{glar_gemm, packa_full, packb_full, KernelDispatcher};

pub(crate) mod reference;

pub(crate) type TA = i8;
pub(crate) type TB = u8;
pub(crate) type TC = i32;

use glar_base::{has_i8i32_compute, Array, ArrayMut, GemmCache, GlarPar, RUNTIME_HW_CONFIG};

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

pub(crate) unsafe fn glar_gemm_s8u8s32_generic<F: MyFn>(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: Array<TA>,
    b: Array<TB>,
    beta: f32,
    c: ArrayMut<TC>,
    f: F,
) {
    let par = GlarPar::default(m, n);
    if has_i8i32_compute() {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
        {
            let hw_config = KernelDispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, f);
            glar_gemm(&hw_config, m, n, k, alpha, a, b, beta, c, &par);
            return;
        }
    }
    // if none of the optimized paths are available, use reference implementation
    let hw_config = RefGemm::from_hw_cfg(&*RUNTIME_HW_CONFIG, f);
    reference::glar_gemm(&hw_config, m, n, k, alpha, a, b, beta, c, &par);
}

pub unsafe fn glar_gemm_s8u8s32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const TA,
    a_rs: usize,
    a_cs: usize,
    b: *const TB,
    b_rs: usize,
    b_cs: usize,
    beta: f32,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
) {
    // transpose if c is row strided i.e. c_cs == 1 and c_rs != 1
    // let (m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b) = if c_cs == 1 && c_rs != 1 {
    // 	(n, m, b_rs, b_cs, a_rs, a_cs, c_cs, c_rs, b, a)
    // } else {
    // 	(m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b)
    // };
    let a = Array::strided_matrix(a, a_rs, a_cs);
    let b = Array::strided_matrix(b, b_rs, b_cs);
    let c = ArrayMut::strided_matrix(c, c_rs, c_cs);
    let null_fn = NullFn {};
    glar_gemm_s8u8s32_generic(m, n, k, alpha, a, b, beta, c, null_fn);
}

#[cfg(feature = "fuse")]
pub unsafe fn glar_gemm_s8u8s32_fused(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const TA,
    a_rs: usize,
    a_cs: usize,
    b: *const TB,
    b_rs: usize,
    b_cs: usize,
    beta: f32,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    unary: fn(*mut TC, usize),
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
    glar_gemm_s8u8s32_generic(m, n, k, alpha, a, b, beta, c, unary);
}

pub fn ap_size(m: usize, k: usize) -> usize {
    let mv: usize;
    let kv: usize;
    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
    {
        let hw_config = KernelDispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, NullFn {});
        mv = hw_config.mv();
        kv = hw_config.kv();
    }

    let m_rounded = (m + mv - 1) / mv * mv;
    let k_rounded = (k + kv - 1) / kv * kv;
    m_rounded * k_rounded + glar_base::AB_ALIGN / std::mem::size_of::<TA>()
}

pub fn bp_size(n: usize, k: usize) -> usize {
    let nv: usize;
    let kv: usize;
    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
    {
        let hw_config = KernelDispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, NullFn {});
        nv = hw_config.nv();
        kv = hw_config.kv();
    }

    let n_rounded = (n + nv - 1) / nv * nv;
    let k_rounded = (k + kv - 1) / kv * kv;
    n_rounded * k_rounded + glar_base::AB_ALIGN / std::mem::size_of::<TB>()
}

// block idx for packa and packb is s.t.
// m dim for block idx is contiguous and n dim is contiguous
// this is to ensure that indexing for parallelization over these dims are easy  (otherwise ranges would have to be in the same mc, nc range)
// this is not an issue since we do not parallelize over k dim (think about this when we parallelize over k dim in the future, which is only beneficial only
// in the special case of very large k and small m, n
pub unsafe fn packa_i8(m: usize, k: usize, a: *const TA, a_rs: usize, a_cs: usize, ap: *mut TA) -> Array<TA> {
    assert_eq!(ap.align_offset(glar_base::AB_ALIGN), 0);
    if m == 1 {
        for j in 0..k {
            *ap.add(j) = *a.add(j * a_cs);
        }
        return Array::strided_matrix(ap, 1, m);
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
    {
        if has_i8i32_compute() {
            return packa_full(m, k, a, a_rs, a_cs, ap);
        }
    }
    reference::packa_full(m, k, a, a_rs, a_cs, ap)
}

pub unsafe fn packb_u8(n: usize, k: usize, b: *const TB, b_rs: usize, b_cs: usize, bp: *mut TB) -> Array<TB> {
    assert_eq!(bp.align_offset(glar_base::AB_ALIGN), 0);
    if n == 1 {
        for j in 0..k {
            *bp.add(j) = *b.add(j * b_rs);
        }
        return Array::strided_matrix(bp, 1, k);
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
    {
        if has_i8i32_compute() {
            return packb_full(n, k, b, b_rs, b_cs, bp);
        }
    }
    reference::packb_full(n, k, b, b_rs, b_cs, bp)
}

pub unsafe fn packa_i8_with_ref(m: usize, k: usize, a: &[TA], a_rs: usize, a_cs: usize, ap: &mut [TA]) -> Array<TA> {
    let pack_size = ap_size(m, k);
    let ap_align_offset = ap.as_ptr().align_offset(glar_base::AB_ALIGN);
    // safety check
    assert!(ap.len() >= pack_size);
    let ap = &mut ap[ap_align_offset..];
    unsafe { packa_i8(m, k, a.as_ptr(), a_rs, a_cs, ap.as_mut_ptr()) }
}

pub unsafe fn packb_u8_with_ref(n: usize, k: usize, b: &[TB], b_rs: usize, b_cs: usize, bp: &mut [TB]) -> Array<TB> {
    let pack_size = bp_size(n, k);
    let bp_align_offset = bp.as_ptr().align_offset(glar_base::AB_ALIGN);
    // safety check
    assert!(bp.len() >= pack_size);
    let bp = &mut bp[bp_align_offset..];
    unsafe { packb_u8(n, k, b.as_ptr(), b_rs, b_cs, bp.as_mut_ptr()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glar_base::{get_cache_params, matrix_size};
    use glar_dev::{
        check_gemm_s8u8s32, generate_k_dims, generate_m_dims, generate_n_dims, layout_to_strides,
        random_matrix_uniform, ABLayout,
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
            *c.add(i) *= 2;
        }
    }

    // fn my_unary(_c: *mut TC, _m: usize) {}

    const EPS: f64 = 1e-1;

    // static ALPHA_ARR: [f32; 2] = [1.0, 3.0];
    // static BETA_ARR: [f32; 3] = [1.0, 3.0, 0.0];
    static ALPHA_ARR: [f32; 1] = [2.0];
    static BETA_ARR: [f32; 1] = [3.0];

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
                let mut c = vec![0i32; c_size];
                let mut c_ref = vec![0i32; c_size];
                for k in k_dims.iter() {
                    let k = *k;
                    let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = layout_to_strides(&layout, m, n, k);
                    let mut a = vec![0i8; m * k];
                    let mut b = vec![0u8; k * n];
                    random_matrix_uniform(m, k, &mut a, m);
                    random_matrix_uniform(k, n, &mut b, k);
                    let ap_size_val = if is_a_packed { ap_size(m, k) } else { 0 };
                    let mut ap = vec![0i8; ap_size_val];
                    let a_matrix = if is_a_packed {
                        unsafe { packa_i8_with_ref(m, k, &a, a_rs, a_cs, &mut ap) }
                    } else {
                        Array::strided_matrix(a.as_ptr(), a_rs, a_cs)
                    };
                    let bp_size_val = if is_b_packed { bp_size(n, k) } else { 0 };
                    let mut bp = vec![0u8; bp_size_val];
                    let b_matrix = if is_b_packed {
                        unsafe { packb_u8_with_ref(n, k, &b, b_rs, b_cs, &mut bp) }
                    } else {
                        Array::strided_matrix(b.as_ptr(), b_rs, b_cs)
                    };
                    for alpha in ALPHA_ARR {
                        for beta in BETA_ARR {
                            random_matrix_uniform(m, n, &mut c, m);
                            c_ref.copy_from_slice(&c);
                            let c_matrix = ArrayMut::strided_matrix(c.as_mut_ptr(), c_rs, c_cs);
                            unsafe {
                                glar_gemm_s8u8s32_generic(m, n, k, alpha, a_matrix, b_matrix, beta, c_matrix, unary_fn);
                            }
                            let diff_max = unsafe {
                                check_gemm_s8u8s32(
                                    m,
                                    n,
                                    k,
                                    alpha as f32,
                                    a.as_ptr(),
                                    a_rs,
                                    a_cs,
                                    b.as_ptr(),
                                    b_rs,
                                    b_cs,
                                    beta as f32,
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
