pub(crate) mod avx;
pub(crate) mod avx512_f16;
pub(crate) mod avx512f;
pub(crate) mod avx_fma;
pub(crate) mod pack_avx;

use crate::f16;

const AVX512_F16_VS: usize = 32;
const AVX512F_VS: usize = 16;
const AVX_VS: usize = 8;

const AVX512F_MR: usize = 48;
const AVXFMA_MR: usize = 24;
const AVX_MR: usize = 16;
const AVX512_F16_MR: usize = 64;

const AVX512F_NR: usize = 8;
const AVXFMA_NR: usize = 4;
const AVX_NR: usize = 4;
const AVX512_F16_NR: usize = 15;

use pire_base::{
    acquire, def_pa, def_pire_gemm, extend, get_apbp_barrier, get_cache_params, get_mem_pool_size_goto,
    get_mem_pool_size_small_m, get_mem_pool_size_small_n, has_f16_compute, is_mixed, run_small_m, run_small_n,
    split_c_range, split_range, Array, ArrayMut, GemmPool, HWModel, PArray, PArrayMixed, PirePar, PireThreadConfig,
    PoolSize, PtrData, PACK_POOL, RUNTIME_HW_CONFIG,
};

use crate::{GemmCache, IdentityFn, UnaryFnC, TA, TB, TC};

#[inline(always)]
pub(crate) fn get_mcnckc_simd_f32() -> (usize, usize, usize) {
    let features = (*RUNTIME_HW_CONFIG).cpu_ft();
    let (mr, nr) = if features.avx512f {
        (AVX512F_MR, AVX512F_NR)
    } else if features.avx && features.fma {
        (AVXFMA_MR, AVXFMA_NR)
    } else {
        (AVX_MR, AVX_NR)
    };
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        HWModel::Skylake => (4800, 384, 1024),
        HWModel::Haswell => (4800, 320, 192),
        _ => get_cache_params(),
    };
    (mc / mr * mr, nc / nr * nr, kc)
}

#[inline(always)]
pub(crate) fn get_mcnckc_simd_f16() -> (usize, usize, usize) {
    let mr = AVX512_F16_MR;
    let nr = AVX512_F16_NR;
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        HWModel::Skylake => (4800, 360, 1024),
        HWModel::Haswell => (4800, 320, 192),
        _ => get_cache_params(),
    };
    (mc / mr * mr, nc / nr * nr, kc)
}

#[inline(always)]
pub(crate) fn get_mcnckc_simd() -> (usize, usize, usize) {
    if has_f16_compute() {
        get_mcnckc_simd_f16()
    } else {
        get_mcnckc_simd_f32()
    }
}

pub(crate) unsafe fn packa_fn_simd_f32(x: *const TA, y: *mut f32, m: usize, k: usize, rs: usize, cs: usize) {
    let hw_config = &*RUNTIME_HW_CONFIG;
    if hw_config.cpu_ft.avx512f {
        pack_avx::packa_panel_48(m, k, x, rs, cs, y, AVX512F_VS);
    } else if hw_config.cpu_ft.avx && hw_config.cpu_ft.fma {
        pack_avx::packa_panel_24(m, k, x, rs, cs, y, AVX_VS);
    } else {
        pack_avx::packa_panel_16(m, k, x, rs, cs, y, AVX_VS);
    }
}
pub(crate) unsafe fn packb_fn_simd_f32(x: *const TB, y: *mut f32, n: usize, k: usize, rs: usize, cs: usize) {
    if (*RUNTIME_HW_CONFIG).cpu_ft.avx512f {
        pack_avx::packb_panel_8(n, k, x, cs, rs, y);
    } else {
        pack_avx::packb_panel_4(n, k, x, cs, rs, y);
    }
}

pub(crate) unsafe fn packa_fn_simd_f16(x: *const TA, y: *mut f16, m: usize, k: usize, rs: usize, cs: usize) {
    pack_avx::packa_panel_64_same(m, k, x, rs, cs, y, AVX512_F16_VS);
}
pub(crate) unsafe fn packb_fn_simd_f16(x: *const TB, y: *mut f16, n: usize, k: usize, rs: usize, cs: usize) {
    pack_avx::packb_panel_15_same(n, k, x, cs, rs, y);
}

pub(crate) unsafe fn packa_fn_simd(x: *const TA, y: *mut f16, m: usize, k: usize, rs: usize, cs: usize) {
    if has_f16_compute() {
        packa_fn_simd_f16(x, y, m, k, rs, cs);
    } else {
        let hw_config = &*RUNTIME_HW_CONFIG;
        if hw_config.cpu_ft.avx512f {
            pack_avx::packa_panel_48_same(m, k, x, rs, cs, y, AVX512F_VS);
        } else if hw_config.cpu_ft.avx && hw_config.cpu_ft.fma {
            pack_avx::packa_panel_24_same(m, k, x, rs, cs, y, AVX_VS);
        } else {
            pack_avx::packa_panel_16_same(m, k, x, rs, cs, y, AVX_VS);
        }
    }
}
pub(crate) unsafe fn packb_fn_simd(x: *const TB, y: *mut f16, n: usize, k: usize, rs: usize, cs: usize) {
    if has_f16_compute() {
        pack_avx::packb_panel_15_same(n, k, x, cs, rs, y);
    } else {
        let hw_config = &*RUNTIME_HW_CONFIG;
        if hw_config.cpu_ft.avx512f {
            pack_avx::packb_panel_8_same(n, k, x, cs, rs, y);
        } else {
            pack_avx::packb_panel_4_same(n, k, x, cs, rs, y);
        }
    }
}

pub(crate) fn round_m_simd(m: usize) -> usize {
    let vs = if has_f16_compute() {
        AVX512_F16_VS
    } else {
        let hw_config = &*RUNTIME_HW_CONFIG;
        if hw_config.cpu_ft.avx512f {
            AVX512F_VS
        } else {
            AVX_VS
        }
    };
    (m + vs - 1) / vs * vs
}

pub(crate) fn round_k_simd(k: usize) -> usize {
    k
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
        let (mc, nc, kc) = get_mcnckc_simd_f32();
        let features = hw_config.cpu_ft();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (mr, nr, reg_dim) = if features.avx512f {
            (AVX512F_MR, AVX512F_NR, RegDim::Reg48x8)
        } else if features.avx && features.fma {
            (AVXFMA_MR, AVXFMA_NR, RegDim::Reg24x4)
        } else {
            (AVX_MR, AVX_NR, RegDim::Reg16x4)
        };
        let vs = if features.avx512f { AVX512F_VS } else { AVX_VS };
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
        let (mc, nc, kc) = get_mcnckc_simd_f16();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let mr = AVX512_F16_MR;
        let nr = AVX512_F16_NR;
        let vs = AVX512_F16_VS;
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

unsafe fn pire_gemv<F: UnaryFnC>(
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

#[allow(unused)]
unsafe fn kernel_mn<F: UnaryFnC>(
    hw_cfg: &KernelDispatcherF32<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    a: *const TB,
    a_rs: usize,
    a_cs: usize,
    b: *const TB,
    b_rs: usize,
    b_cs: usize,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    kc_last: bool,
) {
    return;
    // if kc_last {
    //     match hw_cfg.reg_dim {
    //         RegDim::Avx512f => avx512f::kernel_ss(m, n, k, alpha, beta, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, c_cs, hw_cfg.func),
    //         RegDim::AvxFma => return,
    //         RegDim::Avx => return,
    //         RegDim::Sse => return,
    //     }
    // } else {
    //     let null_fn = IdentityFn {};
    //     match hw_cfg.reg_dim {
    //         RegDim::Avx512f => avx512f::kernel_ss(m, n, k, alpha, beta, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, c_cs, null_fn),
    //         RegDim::AvxFma => return,
    //         RegDim::Avx => return,
    //         RegDim::Sse => return,
    //     }
    // }
}


def_pire_gemm!(
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
    pire_gemm_f32,
    gemm_mt,
    gemm_goto_serial,
    kernel,
    gemm_small_m_serial,
    kernel_m,
    gemm_small_n_serial,
    kernel_n,
    gemm_small_mn_serial,
    kernel_mn,
    pire_gemv,
    pire_gemv,
    packa0,
    packb0,
    packa_fn_simd_f32,
    packb_fn_simd_f32,
    false,
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

#[allow(unused)]
unsafe fn kernel_mn_native<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f16,
    beta: *const f16,
    a: *const TB,
    a_rs: usize,
    a_cs: usize,
    b: *const TB,
    b_rs: usize,
    b_cs: usize,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    kc_last: bool,
) {
    return;
    // if kc_last {
    //     match hw_cfg.reg_dim {
    //         RegDim::Avx512f => avx512f::kernel_ss(m, n, k, alpha, beta, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, c_cs, hw_cfg.func),
    //         RegDim::AvxFma => return,
    //         RegDim::Avx => return,
    //         RegDim::Sse => return,
    //     }
    // } else {
    //     let null_fn = IdentityFn {};
    //     match hw_cfg.reg_dim {
    //         RegDim::Avx512f => avx512f::kernel_ss(m, n, k, alpha, beta, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, c_cs, null_fn),
    //         RegDim::AvxFma => return,
    //         RegDim::Avx => return,
    //         RegDim::Sse => return,
    //     }
    // }
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

unsafe fn pire_gemv_native<F: UnaryFnC>(
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

def_pire_gemm!(
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
    pire_gemm,
    gemm_mt_native,
    gemm_goto_serial_native,
    kernel_native,
    gemm_small_m_serial_native,
    kernel_m_native,
    gemm_small_n_serial_native,
    kernel_n_native,
    gemm_small_mn_serial_native,
    kernel_mn_native,
    pire_gemv_native,
    pire_gemv_native,
    packa0f16,
    packb0f16,
    packa_fn_simd_f16,
    packb_fn_simd_f16,
    false,
    false,
    false,
    into_pack_array,
    F,
);
