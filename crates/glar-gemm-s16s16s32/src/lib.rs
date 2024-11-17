#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_arch;
#[cfg(target_arch = "x86")]
pub(crate) mod x86_arch;

#[cfg(target_arch = "x86_64")]
use x86_64_arch::{
    get_mcnckc_simd, glar_gemm, packa_fn_simd, packb_fn_simd, round_k_simd, round_m_simd, KernelDispatcher,
};

#[cfg(target_arch = "x86")]
use x86_arch::{
    get_mcnckc_simd, glar_gemm, packa_fn_simd, packb_fn_simd, round_k_simd, round_m_simd, KernelDispatcher,
};

pub(crate) mod reference;

pub(crate) type TA = i16;
pub(crate) type TB = i16;
pub(crate) type TC = i32;
#[allow(unused)]
const TC_SIZE: usize = core::mem::size_of::<TC>();

use glar_base::{
    get_cache_params, has_i16i32_compute, Array, ArrayMut, GemmCache, GlarPar, IdentityFn, UnaryFn, AB_ALIGN,
};

use reference::{packa_fn_ref, packb_fn_ref, round_k_ref, round_m_ref, RefGemm};

use core::mem::size_of;

pub(crate) trait UnaryFnC: UnaryFn<TC> {}
impl<F: UnaryFn<TC>> UnaryFnC for F {}

pub(crate) unsafe fn glar_gemm_s16s16s32_fused<F: UnaryFnC>(
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
    if has_i16i32_compute() {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let hw_config = KernelDispatcher::new(f);
            glar_gemm(&hw_config, m, n, k, alpha, a, b, beta, c, &par);
            return;
        }
    }
    // if none of the optimized paths are available, use reference implementation
    let hw_config = RefGemm::new(f);
    reference::glar_gemm(&hw_config, m, n, k, alpha, a, b, beta, c, &par);
}

pub unsafe fn glar_gemm_s16s16s32(
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
    let (m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b) = if c_cs == 1 && c_rs != 1 {
        (n, m, b_rs, b_cs, a_rs, a_cs, c_cs, c_rs, b, a)
    } else {
        (m, n, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, a, b)
    };
    let a = Array::strided_matrix(a, a_rs, a_cs);
    let b = Array::strided_matrix(b, b_rs, b_cs);
    let c = ArrayMut::strided_matrix(c, c_rs, c_cs);
    let identity_fn = IdentityFn {};
    glar_gemm_s16s16s32_fused(m, n, k, alpha, a, b, beta, c, identity_fn);
}

#[cfg(feature = "fuse")]
pub unsafe fn glar_gemm_s16s16s32_fn_ptr(
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
    unary: unsafe fn(*mut TC, usize),
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
    glar_gemm_s16s16s32_fused(m, n, k, alpha, a, b, beta, c, unary);
}
fn dispatch_round_m() -> fn(usize) -> usize {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if has_i16i32_compute() {
            return round_m_simd;
        }
    }
    round_m_ref
}
fn dispatch_round_k() -> fn(usize) -> usize {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if has_i16i32_compute() {
            return round_k_simd;
        }
    }
    round_k_ref
}

fn dispatch_pack_a() -> unsafe fn(*const TA, *mut TA, usize, usize, usize, usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if has_i16i32_compute() {
            return packa_fn_simd;
        }
    }
    packa_fn_ref
}

fn dispatch_pack_b() -> unsafe fn(*const TB, *mut TB, usize, usize, usize, usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if has_i16i32_compute() {
            return packb_fn_simd;
        }
    }
    packb_fn_ref
}

fn dispatch_get_mcnckc() -> (usize, usize, usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if has_i16i32_compute() {
            return get_mcnckc_simd();
        }
    }
    get_cache_params()
}

glar_base::packing_api!(TA, TB);

#[cfg(test)]
mod tests {
    use super::*;
    use glar_base::{get_cache_params, matrix_size};
    use glar_dev::{
        check_gemm_s16s16s32, generate_k_dims, generate_m_dims, generate_n_dims, layout_to_strides,
        random_matrix_uniform, ABLayout,
    };

    #[inline(always)]
    #[allow(unreachable_code)]
    pub(crate) fn get_mcnckc() -> (usize, usize, usize) {
        #[cfg(target_arch = "x86_64")]
        {
            return x86_64_arch::get_mcnckc_simd();
        }
        get_cache_params()
    }

    unsafe fn unary_fn_test(c: *mut TC, m: usize) {
        for i in 0..m {
            *c.add(i) *= 2;
        }
    }

    const EPS: f64 = 1e-1;

    static ALPHA_ARR: [f32; 1] = [2.0];
    static BETA_ARR: [f32; 1] = [3.0];

    fn test_gemm(layout: &ABLayout, is_a_packed: bool, is_b_packed: bool) {
        let a_stride_scale = 1;
        let b_stride_scale = 1;
        let c_stride_scale = 2;
        let (mc, nc, kc) = get_mcnckc();
        let (mr, nr, kr) = (48, 8, 8);
        let m_dims = generate_m_dims(mc, mr);
        let n_dims = generate_n_dims(nc, nr);
        let k_dims = generate_k_dims(kc, kr);
        let unary_fn: unsafe fn(*mut TC, usize) = unary_fn_test;
        let m_max = *m_dims.iter().max().unwrap();
        let n_max = *n_dims.iter().max().unwrap();
        let k_max = *k_dims.iter().max().unwrap();
        let a_size = matrix_size(m_max, k_max) * a_stride_scale;
        let b_size = matrix_size(k_max, n_max) * b_stride_scale;
        let c_size = matrix_size(m_max, n_max) * c_stride_scale;
        let mut a = vec![0i16; a_size];
        let mut b = vec![0i16; b_size];
        random_matrix_uniform(&mut a);
        random_matrix_uniform(&mut b);
        let mut c = vec![0i32; c_size];
        let mut c_ref = vec![0i32; c_size];

        let ap_size = if is_a_packed { a_size_packed(m_max, k_max) } else { 0 };
        let mut ap = vec![0i16; ap_size];

        let bp_size = if is_b_packed { b_size_packed(n_max, k_max) } else { 0 };
        let mut bp = vec![0i16; bp_size];
        for &m in &m_dims {
            for &n in &n_dims {
                for &k in &k_dims {
                    let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = layout_to_strides(&layout, m, n, k);
                    let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = (
                        a_rs * a_stride_scale,
                        a_cs * a_stride_scale,
                        b_rs * b_stride_scale,
                        b_cs * b_stride_scale,
                        c_rs * c_stride_scale,
                        c_cs * c_stride_scale,
                    );
                    let a_matrix = if is_a_packed {
                        unsafe { pack_a(m, k, &a, a_rs, a_cs, &mut ap) }
                    } else {
                        Array::strided_matrix(a.as_ptr(), a_rs, a_cs)
                    };
                    let b_matrix = if is_b_packed {
                        unsafe { pack_b(n, k, &b, b_rs, b_cs, &mut bp) }
                    } else {
                        Array::strided_matrix(b.as_ptr(), b_rs, b_cs)
                    };
                    for alpha in ALPHA_ARR {
                        for beta in BETA_ARR {
                            random_matrix_uniform(&mut c);
                            c_ref.copy_from_slice(&c);
                            let c_matrix = ArrayMut::strided_matrix(c.as_mut_ptr(), c_rs, c_cs);
                            unsafe {
                                glar_gemm_s16s16s32_fused(m, n, k, alpha, a_matrix, b_matrix, beta, c_matrix, unary_fn);
                            }
                            let diff_max = unsafe {
                                check_gemm_s16s16s32(
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
