#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_arch;

pub(crate) mod reference;

pub(crate) type TA = i8;
pub(crate) type TB = u8;
pub(crate) type TC = i32;

#[derive(Copy, Clone)]
pub(crate) struct NullFn;

pub(crate) trait MyFn: Copy + std::marker::Sync {
    fn call(self, c: *mut TC, m: usize);
}

use glare_base::{
    ap_size_int, bp_size_int, get_cache_params, has_i8i32_compute, Array, ArrayMut, GemmCache,
    GlarePar, HWModel, StridedMatrix, StridedMatrixMut, RUNTIME_HW_CONFIG,
};

use reference::RefGemm;

impl MyFn for NullFn {
    #[inline(always)]
    fn call(self, _c: *mut TC, _m: usize) {}
}

impl MyFn for fn(*mut TC, m: usize) {
    #[inline(always)]
    fn call(self, c: *mut TC, m: usize) {
        self(c, m);
    }
}

#[inline(always)]
fn get_mcnckc() -> (usize, usize, usize) {
    // let mc = std::env::var("GLARE_MC").unwrap_or("4800".to_string()).parse::<usize>().unwrap();
    // let nc = std::env::var("GLARE_NC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // let kc = std::env::var("GLARE_KC").unwrap_or("512".to_string()).parse::<usize>().unwrap();
    // return (mc, nc, kc);
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        HWModel::Skylake => (4800, 192, 512),
        HWModel::Haswell | HWModel::Broadwell => (4800, 320, 192),
        _ => get_cache_params(),
    };
    (mc, nc, kc)
}

#[cfg(target_arch = "x86_64")]
use x86_64_arch::X86_64dispatcher;

pub(crate) unsafe fn glare_gemm_s8u8s32_generic<F: MyFn>(
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
    let par = GlarePar::default();
    let (mc, nc, kc) = get_mcnckc();
    if has_i8i32_compute() {
        let hw_config = X86_64dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, f);
        x86_64_arch::glare_gemm(&hw_config, m, n, k, alpha, a, b, beta, c, &par);
        return;
    }
    // if none of the optimized paths are available, use reference implementation
    let hw_config = RefGemm::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, f);
    reference::glare_gemm(&hw_config, m, n, k, alpha, a, b, beta, c, &par);
}

pub unsafe fn glare_gemm_s8u8s32(
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
    let a = StridedMatrix::new(a, a_rs, a_cs);
    let a = Array::StridedMatrix(a);
    let b = StridedMatrix::new(b, b_rs, b_cs);
    let b = Array::StridedMatrix(b);
    let c = StridedMatrixMut::new(c, c_rs, c_cs);
    let c = ArrayMut::StridedMatrix(c);
    let null_fn = NullFn {};
    glare_gemm_s8u8s32_generic(m, n, k, alpha, a, b, beta, c, null_fn);
}

#[cfg(feature = "fuse")]
pub unsafe fn glare_gemm_s8u8s32_fused(
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
    let a = StridedMatrix::new(a, a_rs, a_cs);
    let a = Array::StridedMatrix(a);
    let b = StridedMatrix::new(b, b_rs, b_cs);
    let b = Array::StridedMatrix(b);
    let c = StridedMatrixMut::new(c, c_rs, c_cs);
    let c = ArrayMut::StridedMatrix(c);
    glare_gemm_s8u8s32_generic(m, n, k, alpha, a, b, beta, c, unary);
}

pub unsafe fn glare_gemv_s8u8s32(
    m: usize,
    n: usize,
    alpha: f32,
    a: *const TA,
    a_rs: usize,
    a_cs: usize,
    x: *const TB,
    incx: usize,
    beta: f32,
    y: *mut TC,
    incy: usize,
) {
    glare_gemm_s8u8s32(m, 1, n, alpha, a, a_rs, a_cs, x, 1, incx, beta, y, 1, incy)
}
pub unsafe fn glare_dot_s8u8s32(
    n: usize,
    alpha: f32,
    x: *const TA,
    incx: usize,
    y: *const TB,
    incy: usize,
    beta: f32,
    res: *mut TC,
) {
    glare_gemm_s8u8s32(1, 1, n, alpha, x, incx, 1, y, incy, 1, beta, res, 1, 1)
}

// block idx for packa and packb is s.t.
// m dim for block idx is contiguous and n dim is contiguous
// this is to ensure that indexing for parallelization over these dims are easy  (otherwise ranges would have to be in the same mc, nc range)
// this is not an issue since we do not parallelize over k dim (think about this when we parallelize over k dim in the future, which is only beneficial only
// in the special case of very large k and small m, n
pub unsafe fn packa_i8(
    m: usize,
    k: usize,
    a: *const TA,
    a_rs: usize,
    a_cs: usize,
    ap: *mut TA,
) -> Array<TA> {
    assert_eq!(ap.align_offset(glare_base::AB_ALIGN), 0);
    let mut ap = ap;
    if m == 1 {
        for j in 0..k {
            *ap.add(j) = *a.add(j * a_cs);
        }
        return Array::strided_matrix(ap, 1, m);
    }
    let (mc, nc, kc) = get_mcnckc();
    let hw_config = X86_64dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn {});
    // if none of the optimized paths are available, use reference implementation
    let hw_config_ref = RefGemm::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn {});

    #[cfg(target_arch = "x86_64")]
    {
        let ap0 = ap;
        let vs = if has_i8i32_compute() { hw_config.vs } else { hw_config_ref.vs };
        for p in (0..k).step_by(kc) {
            let kc_len = if k >= (p + kc) { kc } else { k - p };
            for i in (0..m).step_by(mc) {
                let mc_len = if m >= (i + mc) { mc } else { m - i };
                let mc_len_eff = (mc_len + vs - 1) / vs * vs;
                let a_cur = a.add(i * a_rs + p * a_cs);
                let kc_len_eff = if has_i8i32_compute() {
                    hw_config.packa_fn(a_cur, ap, mc_len, kc_len, a_rs, a_cs);
                    hw_config.round_up(kc_len)
                } else {
                    hw_config_ref.packa_fn(a_cur, ap, mc_len, kc_len, a_rs, a_cs);
                    hw_config_ref.round_up(kc_len)
                };
                ap = ap.add(mc_len_eff * kc_len_eff);
            }
        }
        return Array::packed_matrix(ap0, m, k);
    }
}

pub unsafe fn packb_u8(
    n: usize,
    k: usize,
    b: *const TB,
    b_rs: usize,
    b_cs: usize,
    bp: *mut TB,
) -> Array<TB> {
    assert_eq!(bp.align_offset(glare_base::AB_ALIGN), 0);
    let mut bp = bp;
    if n == 1 {
        for j in 0..k {
            *bp.add(j) = *b.add(j * b_rs);
        }
        return Array::strided_matrix(bp, 1, k);
    }
    let (mc, nc, kc) = get_mcnckc();
    let hw_config_ref = RefGemm::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn {});
    let hw_config = X86_64dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, mc, nc, kc, NullFn {});

    #[cfg(target_arch = "x86_64")]
    {
        let bp0 = bp;
        for p in (0..k).step_by(kc) {
            let kc_len = if k >= (p + kc) { kc } else { k - p };
            for i in (0..n).step_by(nc) {
                let nc_len = if n >= (i + nc) { nc } else { n - i };
                let nc_len_eff = nc_len;
                let b_cur = b.add(i * b_cs + p * b_rs);
                let kc_len_eff = if has_i8i32_compute() {
                    hw_config.packb_fn(b_cur, bp, nc_len, kc_len, b_rs, b_cs);
                    hw_config.round_up(kc_len)
                } else {
                    hw_config_ref.packb_fn(b_cur, bp, nc_len, kc_len, b_rs, b_cs);
                    hw_config_ref.round_up(kc_len)
                };
                bp = bp.add(nc_len_eff * kc_len_eff);
            }
        }
        return Array::packed_matrix(bp0, n, k);
    }
}

pub unsafe fn packa_i8_with_ref(
    m: usize,
    k: usize,
    a: &[TA],
    a_rs: usize,
    a_cs: usize,
    ap: &mut [TA],
) -> Array<TA> {
    let pack_size = ap_size_int::<TA, TC>(m, k);
    let ap_align_offset = ap.as_ptr().align_offset(glare_base::AB_ALIGN);
    // safety check
    assert!(ap.len() >= pack_size);
    let ap = &mut ap[ap_align_offset..];
    unsafe { packa_i8(m, k, a.as_ptr(), a_rs, a_cs, ap.as_mut_ptr()) }
}

pub unsafe fn packb_u8_with_ref(
    n: usize,
    k: usize,
    b: &[TB],
    b_rs: usize,
    b_cs: usize,
    bp: &mut [TB],
) -> Array<TB> {
    let pack_size = bp_size_int::<TB, TC>(n, k);
    let bp_align_offset = bp.as_ptr().align_offset(glare_base::AB_ALIGN);
    // safety check
    assert!(bp.len() >= pack_size);
    let bp = &mut bp[bp_align_offset..];
    unsafe { packb_u8(n, k, b.as_ptr(), b_rs, b_cs, bp.as_mut_ptr()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glare_dev::{
        check_gemm_s8u8s32, generate_k_dims, generate_m_dims, generate_n_dims, layout_to_strides,
        random_matrix_uniform, ABLayout,
    };

    const EPS: f64 = 1e-2;

    static ALPHA_ARR: [f32; 2] = [1.0, 3.0];
    static BETA_ARR: [f32; 3] = [1.0, 3.0, 0.0];

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
                let mut c = vec![0i32; m * n];
                let mut c_ref = vec![0i32; m * n];
                for k in k_dims.iter() {
                    let k = *k;
                    let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = layout_to_strides(&layout, m, n, k);
                    let mut a = vec![0i8; m * k];
                    let mut b = vec![0u8; k * n];
                    random_matrix_uniform(m, k, &mut a, m);
                    random_matrix_uniform(k, n, &mut b, k);
                    let ap_size = if is_a_packed { ap_size_int::<TA, TC>(m, k) } else { 0 };
                    let mut ap = vec![0i8; ap_size];
                    let a_matrix = if is_a_packed {
                        unsafe { packa_i8_with_ref(m, k, &a, a_rs, a_cs, &mut ap) }
                    } else {
                        Array::strided_matrix(a.as_ptr(), a_rs, a_cs)
                    };
                    let bp_size = if is_b_packed { bp_size_int::<TB, TC>(n, k) } else { 0 };
                    let mut bp = vec![0u8; bp_size];
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
                                glare_gemm_s8u8s32_generic(
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    a_matrix,
                                    b_matrix,
                                    beta,
                                    c_matrix,
                                    NullFn {},
                                );
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
