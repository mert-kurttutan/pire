pub(crate) mod avx;
pub(crate) mod avx512_f16;
pub(crate) mod avx512f;
pub(crate) mod avx_fma;
pub(crate) mod pack_avx;
// pub(crate) mod pack_f16_avx;

use crate::f16;
// use avx_fma::axpy;

const AVX_FMA_MR: usize = 24; // register block size
const AVX_FMA_NR: usize = 4; // register block size

const AVX512F_MR: usize = 48; // register block size
const AVX512F_NR: usize = 8; // register block size

const AVX512_F16_MR: usize = 64; // register block size
const AVX512_F16_NR: usize = 15; // register block size

use glar_base::{
    acquire, def_glar_gemm, def_pa, extend, get_apbp_barrier, get_cache_params, get_mem_pool_size_goto,
    get_mem_pool_size_small_m, get_mem_pool_size_small_n, has_f16_compute, is_mixed, run_small_m, run_small_n,
    split_c_range, split_range, Array, ArrayMut, GemmPool, GlarPar, GlarThreadConfig, HWModel, PArray, PArrayMixed,
    PoolSize, PtrData, AB_ALIGN, PACK_POOL, RUNTIME_HW_CONFIG,
};

use core::mem::size_of;

use crate::{GemmCache, IdentityFn, UnaryFnC, TA, TB, TC};

#[inline(always)]
pub(crate) fn get_mcnckc() -> (usize, usize, usize) {
    // let mc = std::env::var("GLAR_MC").unwrap_or("4800".to_string()).parse::<usize>().unwrap();
    // let nc = std::env::var("GLAR_NC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // let kc = std::env::var("GLAR_KC").unwrap_or("768".to_string()).parse::<usize>().unwrap();
    // return (mc, nc, kc);
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        HWModel::Skylake => (4800, 384, 1024),
        HWModel::Haswell => (4800, 320, 192),
        _ => get_cache_params(),
    };
    (mc, nc, kc)
}

#[inline(always)]
pub(crate) fn get_mcnckc_f16() -> (usize, usize, usize) {
    // let mc = std::env::var("GLAR_MC").unwrap_or("4800".to_string()).parse::<usize>().unwrap();
    // let nc = std::env::var("GLAR_NC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // let kc = std::env::var("GLAR_KC").unwrap_or("768".to_string()).parse::<usize>().unwrap();
    // return (mc, nc, kc);
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        HWModel::Skylake => (4800, 384, 1024),
        HWModel::Haswell => (4800, 320, 192),
        _ => get_cache_params(),
    };
    (mc, nc, kc)
}

pub(crate) unsafe fn packa_full_f32(
    m: usize,
    k: usize,
    a: *const TA,
    a_rs: usize,
    a_cs: usize,
    ap: *mut TA,
) -> Array<TA> {
    let (mc, _, kc) = get_mcnckc();
    assert_eq!(ap.align_offset(glar_base::AB_ALIGN), 0);
    let hw_config = KernelDispatcherF32::new(IdentityFn {});
    let mut ap_cur = ap;
    let vs = hw_config.vs;
    for p in (0..k).step_by(kc) {
        let kc_len = kc.min(k - p);
        for i in (0..m).step_by(mc) {
            let mc_len = mc.min(m - i);
            let mc_len_eff = (mc_len + vs - 1) / vs * vs;
            let a_cur = a.add(i * a_rs + p * a_cs);
            hw_config.packa_fnsame(a_cur, ap_cur, mc_len, kc_len, a_rs, a_cs);
            ap_cur = ap_cur.add(mc_len_eff * kc_len);
        }
    }
    return Array::packed_matrix(ap, m, k);
}

pub(crate) unsafe fn packb_full_f32(
    n: usize,
    k: usize,
    b: *const TB,
    b_rs: usize,
    b_cs: usize,
    bp: *mut TB,
) -> Array<TB> {
    let (_, nc, kc) = get_mcnckc();
    assert_eq!(bp.align_offset(glar_base::AB_ALIGN), 0);
    let hw_config = KernelDispatcherF32::new(IdentityFn {});
    let mut bp_cur = bp;
    for p in (0..k).step_by(kc) {
        let kc_len = kc.min(k - p);
        for i in (0..n).step_by(nc) {
            let nc_len = nc.min(n - i);
            let nc_len_eff = nc_len;
            let b_cur = b.add(i * b_cs + p * b_rs);
            hw_config.packb_fnsame(b_cur, bp_cur, nc_len, kc_len, b_rs, b_cs);
            bp_cur = bp_cur.add(nc_len_eff * kc_len);
        }
    }
    return Array::packed_matrix(bp, n, k);
}
pub(crate) unsafe fn packa_full(m: usize, k: usize, a: *const TA, a_rs: usize, a_cs: usize, ap: *mut f16) -> Array<TB> {
    let (mc, _, kc) = get_mcnckc();
    assert_eq!(ap.align_offset(glar_base::AB_ALIGN), 0);
    let hw_config = KernelDispatcher::new(IdentityFn {});
    let mut ap_cur = ap;
    let vs = hw_config.vs;
    for p in (0..k).step_by(kc) {
        let kc_len = kc.min(k - p);
        for i in (0..m).step_by(mc) {
            let mc_len = mc.min(m - i);
            let mc_len_eff = (mc_len + vs - 1) / vs * vs;
            let a_cur = a.add(i * a_rs + p * a_cs);
            hw_config.packa_fn(a_cur, ap_cur, mc_len, kc_len, a_rs, a_cs);
            ap_cur = ap_cur.add(mc_len_eff * kc_len);
        }
    }
    return Array::packed_matrix(ap, m, k);
}

pub(crate) unsafe fn packb_full(n: usize, k: usize, b: *const TB, b_rs: usize, b_cs: usize, bp: *mut TB) -> Array<TB> {
    let (_, nc, kc) = get_mcnckc();
    assert_eq!(bp.align_offset(glar_base::AB_ALIGN), 0);
    let hw_config = KernelDispatcher::new(IdentityFn {});
    let mut bp_cur = bp;
    for p in (0..k).step_by(kc) {
        let kc_len = kc.min(k - p);
        for i in (0..n).step_by(nc) {
            let nc_len = nc.min(n - i);
            let nc_len_eff = nc_len;
            let b_cur = b.add(i * b_cs + p * b_rs);
            hw_config.packb_fn(b_cur, bp_cur, nc_len, kc_len, b_rs, b_cs);
            bp_cur = bp_cur.add(nc_len_eff * kc_len);
        }
    }
    return Array::packed_matrix(bp, n, k);
}

pub(crate) fn ap_size_0(m: usize, k: usize) -> usize {
    if has_f16_compute() {
        let hw_config = KernelDispatcher::new(IdentityFn {});
        let m_rounded = hw_config.round_m(m);
        let k_rounded = hw_config.round_k(k);
        m_rounded * k_rounded + AB_ALIGN / size_of::<TA>()
    } else {
        let hw_config = KernelDispatcherF32::new(IdentityFn {});
        let m_rounded = hw_config.round_m(m);
        let k_rounded = hw_config.round_k(k);
        m_rounded * k_rounded + AB_ALIGN / size_of::<TA>()
    }
}

pub(crate) fn bp_size_0(n: usize, k: usize) -> usize {
    if has_f16_compute() {
        let hw_config = KernelDispatcher::new(IdentityFn {});
        let k_rounded = hw_config.round_k(k);
        n * k_rounded + AB_ALIGN / size_of::<TB>()
    } else {
        let hw_config = KernelDispatcherF32::new(IdentityFn {});
        let k_rounded = hw_config.round_k(k);
        n * k_rounded + AB_ALIGN / size_of::<TB>()
    }
}

pub(crate) enum RegDim {
    Reg48x8,
    Reg24x4,
    Reg16x4,
}

pub(crate) struct KernelDispatcherF32<T: UnaryFnC = IdentityFn> {
    mc: usize,
    nc: usize,
    kc: usize,
    mr: usize,
    nr: usize,
    pub(crate) vs: usize,
    pub(crate) reg_dim: RegDim,
    // TODO: Cech jr parallelism is beneificial for perf
    // is_l1_shared: bool,
    is_l2_shared: bool,
    is_l3_shared: bool,
    // features: CpuFeatures,
    func: T,
}

impl<F: UnaryFnC> KernelDispatcherF32<F> {
    pub(crate) fn new(f: F) -> Self {
        let hw_config = &*RUNTIME_HW_CONFIG;
        let (mc, nc, kc) = get_mcnckc();
        let features = hw_config.cpu_ft();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (mr, nr, reg_dim) = if features.avx512f {
            (AVX512F_MR, AVX512F_NR, RegDim::Reg48x8)
        } else if features.avx && features.fma {
            (AVX_FMA_MR, AVX_FMA_NR, RegDim::Reg24x4)
        } else {
            (16, 4, RegDim::Reg16x4)
        };
        let vs = if features.avx512f { 16 } else { 8 };
        Self {
            mc,
            nc,
            kc,
            mr,
            nr,
            vs,
            reg_dim,
            // is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            // features,
            func: f,
        }
    }

    pub(crate) fn is_compute_native(&self) -> bool {
        false
    }

    pub(crate) unsafe fn packa_fn(&self, x: *const f16, y: *mut f32, m: usize, k: usize, rs: usize, cs: usize) {
        match self.reg_dim {
            RegDim::Reg48x8 => pack_avx::packa_panel_48(m, k, x, rs, cs, y, self.vs),
            RegDim::Reg24x4 => pack_avx::packa_panel_24(m, k, x, rs, cs, y, self.vs),
            RegDim::Reg16x4 => pack_avx::packa_panel_16(m, k, x, rs, cs, y, self.vs),
        }
    }

    pub(crate) unsafe fn packb_fn(&self, x: *const f16, y: *mut f32, n: usize, k: usize, rs: usize, cs: usize) {
        match self.reg_dim {
            RegDim::Reg48x8 => pack_avx::packb_panel_8(n, k, x, cs, rs, y),
            RegDim::Reg24x4 | RegDim::Reg16x4 => pack_avx::packb_panel_4(n, k, x, cs, rs, y),
        }
    }

    pub(crate) unsafe fn packa_fnsame(&self, x: *const f16, y: *mut f16, m: usize, k: usize, rs: usize, cs: usize) {
        match self.reg_dim {
            RegDim::Reg48x8 => pack_avx::packa_panel_48_same(m, k, x, rs, cs, y, self.vs),
            RegDim::Reg24x4 => pack_avx::packa_panel_24_same(m, k, x, rs, cs, y, self.vs),
            RegDim::Reg16x4 => pack_avx::packa_panel_16_same(m, k, x, rs, cs, y, self.vs),
        }
    }

    pub(crate) unsafe fn packb_fnsame(&self, x: *const f16, y: *mut f16, n: usize, k: usize, rs: usize, cs: usize) {
        match self.reg_dim {
            RegDim::Reg48x8 => pack_avx::packb_panel_8_same(n, k, x, cs, rs, y),
            RegDim::Reg24x4 | RegDim::Reg16x4 => pack_avx::packb_panel_4_same(n, k, x, cs, rs, y),
        }
    }

    #[target_feature(enable = "avx,f16c")]
    unsafe fn cvt_mixed(&self, x: *const f16, y: *mut f32, m: usize) {
        use core::arch::x86_64::*;
        let m_iter = m / 8;
        let m_rem = m % 8;
        for i in 0..m_iter {
            let x_ptr = x.add(i * 8);
            let y_ptr = y.add(i * 8);
            let v_f32 = _mm256_cvtph_ps(_mm_loadu_si128(x_ptr as *const __m128i));
            _mm256_storeu_ps(y_ptr, v_f32);
        }
        for i in 0..m_rem {
            let x_ptr = x.add(m_iter * 8 + i);
            let y_ptr = y.add(m_iter * 8 + i);
            *y_ptr = (*x_ptr).to_f32();
        }
    }

    pub(crate) fn round_k(&self, k: usize) -> usize {
        k
    }

    pub(crate) fn round_m(&self, m: usize) -> usize {
        (m + self.vs - 1) / self.vs * self.vs
    }
}

pub(crate) struct KernelDispatcher<T: UnaryFnC = IdentityFn> {
    mc: usize,
    nc: usize,
    kc: usize,
    mr: usize,
    nr: usize,
    pub(crate) vs: usize,
    // is_l1_shared: bool,
    is_l2_shared: bool,
    is_l3_shared: bool,
    func: T,
}

impl<F: UnaryFnC> KernelDispatcher<F> {
    pub(crate) fn new(f: F) -> Self {
        let hw_config = &*RUNTIME_HW_CONFIG;
        let (mc, nc, kc) = get_mcnckc_f16();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let mr = AVX512_F16_MR;
        let nr = AVX512_F16_NR;
        let vs = 32;
        Self {
            mc,
            nc,
            kc,
            mr,
            nr,
            vs,
            // is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            func: f,
        }
    }

    pub(crate) fn is_compute_native(&self) -> bool {
        true
    }

    pub(crate) unsafe fn packa_fn(&self, x: *const f16, y: *mut f16, m: usize, k: usize, rs: usize, cs: usize) {
        pack_avx::packa_panel_64_same(m, k, x, rs, cs, y, self.vs);
    }

    pub(crate) unsafe fn packb_fn(&self, x: *const f16, y: *mut f16, n: usize, k: usize, rs: usize, cs: usize) {
        pack_avx::packb_panel_15_same(n, k, x, cs, rs, y);
    }

    pub(crate) fn round_k(&self, k: usize) -> usize {
        k
    }

    pub(crate) fn round_m(&self, m: usize) -> usize {
        (m + self.vs - 1) / self.vs * self.vs
    }
}

impl<T: UnaryFnC> GemmCache for KernelDispatcherF32<T> {
    fn mr(&self) -> usize {
        self.mr
    }
    fn nr(&self) -> usize {
        self.nr
    }
    fn get_kc_eff(&self) -> usize {
        self.kc
    }
    fn get_mc_eff(&self, par: usize) -> usize {
        if self.is_l3_shared {
            (self.mc / (self.mr * par)) * self.mr
        } else {
            self.mc
        }
    }
    fn get_nc_eff(&self, par: usize) -> usize {
        if self.is_l2_shared {
            (self.nc / (self.nr * par)) * self.nr
        } else {
            self.nc
        }
    }
}

impl<T: UnaryFnC> GemmCache for KernelDispatcher<T> {
    fn mr(&self) -> usize {
        self.mr
    }
    fn nr(&self) -> usize {
        self.nr
    }
    fn get_kc_eff(&self) -> usize {
        self.kc
    }
    fn get_mc_eff(&self, par: usize) -> usize {
        if self.is_l3_shared {
            (self.mc / (self.mr * par)) * self.mr
        } else {
            self.mc
        }
    }
    fn get_nc_eff(&self, par: usize) -> usize {
        if self.is_l2_shared {
            (self.nc / (self.nr * par)) * self.nr
        } else {
            self.nc
        }
    }
}

unsafe fn kernel<F: UnaryFnC>(
    hw_cfg: &KernelDispatcherF32<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
    ap: *const f32,
    bp: *const f32,
    kc_last: bool,
) {
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Reg48x8 => avx512f::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg24x4 => avx_fma::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg16x4 => avx::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg48x8 => avx512f::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg24x4 => avx_fma::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg16x4 => avx::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
        }
    }
}

#[allow(unused)]
unsafe fn kernel_m<F: UnaryFnC>(
    hw_cfg: &KernelDispatcherF32<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    b: *const f16,
    b_rs: usize,
    b_cs: usize,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
    ap: *const f32,
    kc_last: bool,
) {
}

#[allow(unused)]
unsafe fn kernel_n<F: UnaryFnC>(
    hw_cfg: &KernelDispatcherF32<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    a: *const f16,
    a_rs: usize,
    a_cs: usize,
    ap: *mut f32,
    b: *const f32,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
    kc_last: bool,
) {
}

unsafe fn glar_gemv<F: UnaryFnC>(
    hw_cfg: &KernelDispatcherF32<F>,
    m: usize,
    n: usize,
    alpha: *const f32,
    a: Array<TA>,
    x: Array<TB>,
    beta: *const f32,
    y: ArrayMut<TC>,
) {
    let x_ptr = x.src();
    let inc_x = x.rs();
    let y_ptr = y.src();
    let incy = y.rs();

    match hw_cfg.reg_dim {
        RegDim::Reg48x8 | RegDim::Reg24x4 => {
            avx_fma::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
        RegDim::Reg16x4 => {
            avx::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
    }
}

def_glar_gemm!(
    KernelDispatcherF32,
    f16,
    f32,
    f16,
    f32,
    f16,
    f32,
    f32,
    PackArrTypeAM,
    PackArrTypeBM,
    1_f32,
    glar_gemm_f32,
    gemm_mt,
    gemm_goto_serial,
    kernel,
    gemm_small_m_serial,
    kernel_m,
    gemm_small_n_serial,
    kernel_n,
    glar_gemv,
    glar_gemv,
    packa_f32,
    packb_f32,
    false,
    false,
    into_pack_array2,
    T,
);

unsafe fn kernel_native<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f16,
    beta: *const f16,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
    ap: *const f16,
    bp: *const f16,
    kc_last: bool,
) {
    if kc_last {
        avx512_f16::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func);
    } else {
        let null_fn = IdentityFn {};
        avx512_f16::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn);
    }
}

unsafe fn kernel_m_native<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f16,
    beta: *const f16,
    b: *const f16,
    b_rs: usize,
    b_cs: usize,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
    ap: *const f16,
    kc_last: bool,
) {
    if kc_last {
        avx512_f16::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
    } else {
        let null_fn = IdentityFn {};
        avx512_f16::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn);
    }
}

unsafe fn kernel_n_native<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f16,
    beta: *const f16,
    a: *const f16,
    a_rs: usize,
    a_cs: usize,
    ap: *mut f16,
    b: *const f16,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
    kc_last: bool,
) {
    if kc_last {
        avx512_f16::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
    } else {
        let null_fn = IdentityFn {};
        avx512_f16::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn);
    }
}

unsafe fn glar_gemv_native<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    alpha: *const f16,
    a: Array<f16>,
    x: Array<f16>,
    beta: *const f16,
    y: ArrayMut<f16>,
) {
    let x_ptr = x.src();
    let inc_x = x.rs();
    let y_ptr = y.src();
    let incy = y.rs();
    let beta_val = (*beta).to_f32();
    let beta_t = &beta_val as *const f32;

    let alhpa_val = (*alpha).to_f32();
    let alpha_t = &alhpa_val as *const f32;
    // use compute_f32 until we have f16 axpy
    avx_fma::axpy(m, n, alpha_t, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta_t, y_ptr, incy, hw_cfg.func);
    return;
}

def_glar_gemm!(
    KernelDispatcher,
    f16,
    f16,
    f16,
    f16,
    f16,
    f16,
    f16,
    PackArrTypeA,
    PackArrTypeB,
    f16::ONE,
    glar_gemm,
    gemm_mt_native,
    gemm_goto_serial_native,
    kernel_native,
    gemm_small_m_serial_native,
    kernel_m_native,
    gemm_small_n_serial_native,
    kernel_n_native,
    glar_gemv_native,
    glar_gemv_native,
    packa,
    packb,
    true,
    true,
    into_pack_array,
    F,
);
