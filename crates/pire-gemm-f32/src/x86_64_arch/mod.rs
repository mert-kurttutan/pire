pub(crate) mod avx;
pub(crate) mod avx512f;
pub(crate) mod avx_fma;
pub(crate) mod pack_avx;
pub(crate) mod pack_sse;
pub(crate) mod sse;

use pire_base::{
    acquire, def_pa, def_pire_gemm, extend, get_apbp_barrier, get_cache_params, get_mem_pool_size_goto,
    get_mem_pool_size_small_m, get_mem_pool_size_small_n, is_mixed, run_small_m, run_small_n, split_c_range,
    split_range, Array, ArrayMut, GemmPool, HWModel, PArray, PirePar, PireThreadConfig, PoolSize, PtrData, PACK_POOL,
    RUNTIME_HW_CONFIG,
};

use crate::{GemmCache, IdentityFn, UnaryFnC, TA, TB, TC};

const AVX512F_VS: usize = 16;
const AVX_VS: usize = 8;
const SSE_VS: usize = 4;

const AVX512F_MR: usize = 48;
const AVXFMA_MR: usize = 24;
const AVX_MR: usize = 16;
const SSE_MR: usize = 8;

const AVX512F_NR: usize = 8;
const AVXFMA_NR: usize = 4;
const AVX_NR: usize = 4;
const SSE_NR: usize = 4;

#[inline(always)]
pub(crate) fn get_mcnckc_simd() -> (usize, usize, usize) {
    let features = (*RUNTIME_HW_CONFIG).cpu_ft();
    let (mr, nr) = if features.avx512f {
        (AVX512F_MR, AVX512F_NR)
    } else if features.avx && features.fma {
        (AVXFMA_MR, AVXFMA_NR)
    } else if features.avx {
        (AVX_MR, AVX_NR)
    } else {
        (SSE_MR, SSE_NR)
    };
    // let mc = std::env::var("PIRE_MC").unwrap_or("4800".to_string()).parse::<usize>().unwrap();
    // let nc = std::env::var("PIRE_NC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // let kc = std::env::var("PIRE_KC").unwrap_or("768".to_string()).parse::<usize>().unwrap();
    // return (mc, nc, kc);
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        HWModel::Skylake => (4800, 384, 1024),
        HWModel::Haswell => (4800, 320, 192),
        _ => get_cache_params(),
    };
    (mc / mr * mr, nc / nr * nr, kc)
}

pub(crate) unsafe fn packa_fn_simd(x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
    let cpu_ft = (*RUNTIME_HW_CONFIG).cpu_ft;
    if cpu_ft.avx512f {
        pack_avx::packa_panel_48(m, k, x, rs, cs, y, AVX512F_VS);
    } else if cpu_ft.avx && cpu_ft.fma {
        pack_avx::packa_panel_24(m, k, x, rs, cs, y, AVX_VS);
    } else if cpu_ft.avx {
        pack_avx::packa_panel_16(m, k, x, rs, cs, y, AVX_VS);
    } else {
        pack_sse::packa_panel_8(m, k, x, rs, cs, y, SSE_VS);
    }
}

pub(crate) unsafe fn packb_fn_simd(x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
    if (*RUNTIME_HW_CONFIG).cpu_ft.avx512f {
        pack_avx::packb_panel_8(n, k, x, cs, rs, y);
    } else if (*RUNTIME_HW_CONFIG).cpu_ft.avx {
        pack_avx::packb_panel_4(n, k, x, cs, rs, y);
    } else {
        pack_sse::packb_panel_4(n, k, x, cs, rs, y);
    }
}

pub(crate) fn round_m_simd(m: usize) -> usize {
    let hw_config = &*RUNTIME_HW_CONFIG;
    let vs = if hw_config.cpu_ft.avx512f {
        AVX512F_VS
    } else if hw_config.cpu_ft.avx {
        AVX_VS
    } else {
        SSE_VS
    };
    (m + vs - 1) / vs * vs
}

pub(crate) fn round_k_simd(k: usize) -> usize {
    k
}

pub(crate) enum RegDim {
    Avx512f,
    AvxFma,
    Avx,
    Sse,
}

pub(crate) struct KernelDispatcher<T: UnaryFnC = IdentityFn> {
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

impl<F: UnaryFnC> KernelDispatcher<F> {
    pub(crate) fn new(f: F) -> Self {
        let hw_config = &*RUNTIME_HW_CONFIG;
        let (mc, nc, kc) = get_mcnckc_simd();
        let features = hw_config.cpu_ft();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();

        let (mr, nr, reg_dim) = if features.avx512f {
            (AVX512F_MR, AVX512F_NR, RegDim::Avx512f)
        } else if features.avx && features.fma {
            (AVXFMA_MR, AVXFMA_NR, RegDim::AvxFma)
        } else if features.avx {
            (AVX_MR, AVX_NR, RegDim::Avx)
        } else {
            (SSE_MR, SSE_NR, RegDim::Sse)
        };
        let vs = if features.avx512f {
            AVX512F_VS
        } else if features.avx {
            AVX_VS
        } else {
            SSE_VS
        };
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
        true
    }

    pub(crate) fn round_k(&self, k: usize) -> usize {
        k
    }

    pub(crate) fn round_m(&self, m: usize) -> usize {
        (m + self.vs - 1) / self.vs * self.vs
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
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    beta: *const TC,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap: *const TA,
    bp: *const TB,
    kc_last: bool,
) {
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Avx512f => avx512f::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::AvxFma => avx_fma::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Avx => avx::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Sse => sse::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Avx512f => avx512f::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::AvxFma => avx_fma::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Avx => avx::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Sse => sse::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
        }
    }
}

unsafe fn kernel_m<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    beta: *const TC,
    b: *const TB,
    b_rs: usize,
    b_cs: usize,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap: *const TA,
    kc_last: bool,
) {
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Avx512f => avx512f::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::AvxFma => avx_fma::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::Avx => avx::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::Sse => sse::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Avx512f => avx512f::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn),
            RegDim::AvxFma => avx_fma::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn),
            RegDim::Avx => avx::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn),
            RegDim::Sse => sse::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn),
        }
    }
}

unsafe fn kernel_n<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    beta: *const TC,
    a: *const TA,
    a_rs: usize,
    a_cs: usize,
    ap: *mut TA,
    b: *const TB,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    kc_last: bool,
) {
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Avx512f => {
                avx512f::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func)
            }
            RegDim::AvxFma => {
                avx_fma::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func)
            }
            RegDim::Avx => avx::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::Sse => sse::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Avx512f => avx512f::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::AvxFma => avx_fma::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::Avx => avx::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::Sse => sse::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
        }
    }
}

unsafe fn kernel_mn<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    beta: *const TC,
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
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Avx512f => avx512f::kernel_ss(m, n, k, alpha, beta, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, c_cs, hw_cfg.func),
            RegDim::AvxFma => return,
            RegDim::Avx => return,
            RegDim::Sse => return,
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Avx512f => avx512f::kernel_ss(m, n, k, alpha, beta, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, c_cs, null_fn),
            RegDim::AvxFma => return,
            RegDim::Avx => return,
            RegDim::Sse => return,
        }
    }
}


unsafe fn pire_gemv<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
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
        RegDim::Avx512f | RegDim::AvxFma => {
            avx_fma::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
        RegDim::Avx => avx::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func),
        RegDim::Sse => sse::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func),
    }
}

def_pire_gemm!(
    KernelDispatcher,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    PackArrTypeA,
    PackArrTypeB,
    1_f32,
    pire_gemm,
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
    packa_fn_simd,
    packb_fn_simd,
    true,
    true,
    false,
    into_pack_array,
    F,
);
