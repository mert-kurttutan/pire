#[cfg(target_arch = "aarch64")]
pub(crate) mod arm64;
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

#[cfg(target_arch = "aarch64")]
use arm64::{get_mcnckc_simd, glar_gemm, packa_fn_simd, packb_fn_simd, round_k_simd, round_m_simd, KernelDispatcher};

pub(crate) mod reference;

use core::mem::size_of;

pub(crate) type TA = i8;
pub(crate) type TB = u8;
pub(crate) type TC = i32;
#[allow(unused)]
const TC_SIZE: usize = size_of::<TC>();

use reference::{packa_fn_ref, packb_fn_ref, round_k_ref, round_m_ref, RefGemm};

use glar_base::{
    get_cache_params, has_i8i32_compute, Array, ArrayMut, GemmCache, GlarPar, IdentityFn, UnaryFn, AB_ALIGN,
};

pub(crate) trait UnaryFnC: UnaryFn<TC> {}
impl<F: UnaryFn<TC>> UnaryFnC for F {}

pub(crate) unsafe fn glar_gemm_s8u8s32_generic<F: UnaryFnC>(
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
            let hw_config = KernelDispatcher::new(f);
            glar_gemm(&hw_config, m, n, k, alpha, a, b, beta, c, &par);
            return;
        }
    }
    // if none of the optimized paths are available, use reference implementation
    let hw_config = RefGemm::new(f);
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
    let null_fn = IdentityFn {};
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

fn dispatch_round_m() -> fn(usize) -> usize {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
    {
        if has_i8i32_compute() {
            return round_m_simd;
        }
    }
    round_m_ref
}
fn dispatch_round_k() -> fn(usize) -> usize {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
    {
        if has_i8i32_compute() {
            return round_k_simd;
        }
    }
    round_k_ref
}

fn dispatch_packa() -> unsafe fn(*const TA, *mut TA, usize, usize, usize, usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
    {
        if has_i8i32_compute() {
            return packa_fn_simd;
        }
    }
    packa_fn_ref
}

fn dispatch_packb() -> unsafe fn(*const TB, *mut TB, usize, usize, usize, usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
    {
        if has_i8i32_compute() {
            return packb_fn_simd;
        }
    }
    packb_fn_ref
}

fn dispatch_get_mcnckc() -> (usize, usize, usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
    {
        if has_i8i32_compute() {
            return get_mcnckc_simd();
        }
    }
    get_cache_params()
}

fn a_size_packed(m: usize, k: usize) -> usize {
    let round_m_fn = dispatch_round_m();
    let round_k_fn = dispatch_round_k();
    let m_round = round_m_fn(m);
    let k_round = round_k_fn(k);
    return m_round * k_round + AB_ALIGN / size_of::<TA>();
}

fn b_size_packed(n: usize, k: usize) -> usize {
    let round_k_fn = dispatch_round_k();
    let k_round = round_k_fn(k);
    return n * k_round + AB_ALIGN / size_of::<TB>();
}

// block idx for packa and packb is s.t.
// m dim for block idx is contiguous and n dim is contiguous
// this is to ensure that indexing for parallelization over these dims are easy  (otherwise ranges would have to be in the same mc, nc range)
// this is not an issue since we do not parallelize over k dim (think about this when we parallelize over k dim in the future, which is only beneficial only
// in the special case of very large k and small m, n
pub unsafe fn packa_i8_unchecked(m: usize, k: usize, a: *const TA, a_rs: usize, a_cs: usize, ap: *mut TA) -> Array<TA> {
    assert_eq!(ap.align_offset(AB_ALIGN), 0);
    if m == 1 {
        for j in 0..k {
            *ap.add(j) = *a.add(j * a_cs);
        }
        return Array::strided_matrix(ap, 1, m);
    }
    let pack_fn = dispatch_packa();
    let round_m_fn = dispatch_round_m();
    let round_k_fn = dispatch_round_k();

    let (mc, _, kc) = dispatch_get_mcnckc();
    let mut ap_cur = ap;
    for p in (0..k).step_by(kc) {
        let kc_len = kc.min(k - p);
        let kc_len_eff = round_k_fn(kc_len);
        for i in (0..m).step_by(mc) {
            let mc_len = mc.min(m - i);
            let mc_len_eff = round_m_fn(mc_len);
            let a_cur = a.add(i * a_rs + p * a_cs);
            pack_fn(a_cur, ap_cur, mc_len, kc_len, a_rs, a_cs);
            ap_cur = ap_cur.add(mc_len_eff * kc_len_eff);
        }
    }
    return Array::packed_matrix(ap, m, k);
}

pub unsafe fn packb_u8_unchecked(n: usize, k: usize, b: *const TB, b_rs: usize, b_cs: usize, bp: *mut TB) -> Array<TB> {
    assert_eq!(bp.align_offset(AB_ALIGN), 0);
    if n == 1 {
        for j in 0..k {
            *bp.add(j) = *b.add(j * b_rs);
        }
        return Array::strided_matrix(bp, 1, k);
    }
    let pack_fn = dispatch_packb();
    let round_k_fn = dispatch_round_k();

    let (_, nc, kc) = dispatch_get_mcnckc();
    let mut bp_cur = bp;
    for p in (0..k).step_by(kc) {
        let kc_len = kc.min(k - p);
        let kc_len_eff = round_k_fn(kc_len);
        for i in (0..n).step_by(nc) {
            let nc_len = nc.min(n - i);
            let b_cur = b.add(i * b_cs + p * b_rs);
            pack_fn(b_cur, bp_cur, nc_len, kc_len, b_rs, b_cs);
            bp_cur = bp_cur.add(nc_len * kc_len_eff);
        }
    }
    return Array::packed_matrix(bp, n, k);
}

pub unsafe fn packa_i8(m: usize, k: usize, a: &[TA], a_rs: usize, a_cs: usize, ap: &mut [TA]) -> Array<TA> {
    let pack_size = a_size_packed(m, k);
    let ap_ptr = ap.as_mut_ptr() as *mut f32;
    let ap_align_offset = ap_ptr.align_offset(AB_ALIGN);
    // safety check
    assert!(ap.len() >= pack_size);
    let ap_ptr_o = ap_ptr.add(ap_align_offset);
    unsafe { packa_i8_unchecked(m, k, a.as_ptr(), a_rs, a_cs, ap_ptr_o as *mut TA) }
}

pub unsafe fn packb_u8(n: usize, k: usize, b: &[TB], b_rs: usize, b_cs: usize, bp: &mut [TB]) -> Array<TB> {
    let pack_size = b_size_packed(n, k);
    let bp_ptr = bp.as_mut_ptr() as *mut f32;
    let bp_align_offset = bp_ptr.align_offset(AB_ALIGN);
    // safety check
    assert!(bp.len() >= pack_size);
    let bp_ptr_o = bp_ptr.add(bp_align_offset);
    unsafe { packb_u8_unchecked(n, k, b.as_ptr(), b_rs, b_cs, bp_ptr_o as *mut TB) }
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
            return x86_64_arch::get_mcnckc_simd();
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
        let a_stride_scale = 1;
        let b_stride_scale = 1;
        let c_stride_scale = 2;
        let (mc, nc, kc) = get_mcnckc();
        let (mr, nr, kr) = (48, 8, 8);
        let m_dims = generate_m_dims(mc, mr);
        let n_dims = generate_n_dims(nc, nr);
        let k_dims = generate_k_dims(kc, kr);
        let unary_fn: unsafe fn(*mut TC, usize) = my_unary;
        let m_max = *m_dims.iter().max().unwrap();
        let n_max = *n_dims.iter().max().unwrap();
        let k_max = *k_dims.iter().max().unwrap();
        let a_size = matrix_size(m_max, k_max) * a_stride_scale;
        let b_size = matrix_size(k_max, n_max) * b_stride_scale;
        let c_size = matrix_size(m_max, n_max) * c_stride_scale;
        let mut a = vec![0i8; a_size];
        let mut b = vec![0u8; b_size];
        random_matrix_uniform(&mut a);
        random_matrix_uniform(&mut b);
        let mut c = vec![0i32; c_size];
        let mut c_ref = vec![0i32; c_size];

        let ap_size = if is_a_packed { a_size_packed(m_max, k_max) } else { 0 };
        let mut ap = vec![0i8; ap_size];

        let bp_size = if is_b_packed { b_size_packed(n_max, k_max) } else { 0 };
        let mut bp = vec![0u8; bp_size];
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
                        unsafe { packa_i8(m, k, &a, a_rs, a_cs, &mut ap) }
                    } else {
                        Array::strided_matrix(a.as_ptr(), a_rs, a_cs)
                    };
                    let b_matrix = if is_b_packed {
                        unsafe { packb_u8(n, k, &b, b_rs, b_cs, &mut bp) }
                    } else {
                        Array::strided_matrix(b.as_ptr(), b_rs, b_cs)
                    };
                    for alpha in ALPHA_ARR {
                        for beta in BETA_ARR {
                            random_matrix_uniform(&mut c);
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
