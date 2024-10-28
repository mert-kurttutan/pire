pub(crate) mod avx2;
pub(crate) mod avx512_vnni;
pub(crate) mod avx512bw;
pub(crate) mod pack_avx;
pub(crate) mod pack_sse;
pub(crate) mod sse;

use glar_base::{
    acquire, def_glar_gemm, def_pa, extend, get_apbp_barrier, get_cache_params, get_mem_pool_size_goto,
    get_mem_pool_size_small_m, get_mem_pool_size_small_n, is_mixed, run_small_m, run_small_n, split_c_range,
    split_range, Array, ArrayMut, GemmPool, GlarPar, GlarThreadConfig, HWModel, PArray, PoolSize, PtrData, PACK_POOL,
    RUNTIME_HW_CONFIG,
};

use crate::{GemmCache, IdentityFn, UnaryFnC, TA, TB, TC};

const AVX512F_VS: usize = 16;
const AVX_VS: usize = 8;
const SSE_VS: usize = 4;

const AVX512_VNNI_MR: usize = 48;
const AVX512BW_MR: usize = 32;
const AVX2_MR: usize = 16;
const SSE_MR: usize = 8;

const AVX512_VNNI_NR: usize = 8;
const AVX512BW_NR: usize = 8;
const AVX2_NR: usize = 4;
const SSE_NR: usize = 4;

#[inline(always)]
pub(crate) fn get_mcnckc_simd() -> (usize, usize, usize) {
    let features = (*RUNTIME_HW_CONFIG).cpu_ft();
    let (mr, nr) = if features.avx512_vnni {
        (AVX512_VNNI_MR, AVX512_VNNI_NR)
    } else if features.avx512bw {
        (AVX512BW_MR, AVX512BW_NR)
    } else if features.avx2 {
        (AVX2_MR, AVX2_NR)
    } else {
        (SSE_MR, SSE_NR)
    };
    // let mc = std::env::var("GLAR_MC").unwrap_or("4800".to_string()).parse::<usize>().unwrap();
    // let nc = std::env::var("GLAR_NC").unwrap_or("320".to_string()).parse::<usize>().unwrap();
    // let kc = std::env::var("GLAR_KC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // return (mc, nc, kc);
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        HWModel::Skylake => (4800, 192, 512),
        HWModel::Haswell => (4800, 320, 576),
        _ => get_cache_params(),
    };
    (mc / mr * mr, nc / nr * nr, kc)
}

pub(crate) unsafe fn packa_fn_simd(x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
    let features = (*RUNTIME_HW_CONFIG).cpu_ft();
    if features.avx512_vnni {
        pack_avx::packa_panel_48(m, k, x, rs, cs, y, AVX512F_VS);
    } else if features.avx512bw {
        pack_avx::packa_panel_32(m, k, x, rs, cs, y, AVX512F_VS);
    } else if features.avx2 {
        pack_avx::packa_panel_16(m, k, x, rs, cs, y, AVX_VS);
    } else {
        pack_sse::packa_panel_8(m, k, x, rs, cs, y, SSE_VS);
    }
}

pub(crate) unsafe fn packb_fn_simd(x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
    let features = (*RUNTIME_HW_CONFIG).cpu_ft();
    if features.avx512_vnni || features.avx512bw {
        pack_avx::packb_panel_8(n, k, x, cs, rs, y);
    } else if features.avx2 {
        pack_avx::packb_panel_4(n, k, x, cs, rs, y);
    } else {
        pack_sse::packb_panel_4(n, k, x, cs, rs, y);
    }
}

pub(crate) fn round_m_simd(m: usize) -> usize {
    let hw_config = &*RUNTIME_HW_CONFIG;
    let vs = if hw_config.cpu_ft.avx512_vnni || hw_config.cpu_ft.avx512bw {
        AVX512F_VS
    } else if hw_config.cpu_ft.avx2 {
        AVX_VS
    } else {
        SSE_VS
    };
    (m + vs - 1) / vs * vs
}

pub(crate) fn round_k_simd(k: usize) -> usize {
    (k + 1) / 2 * 2
}

pub(crate) enum RegDim {
    Avx512VNNI,
    Avx512BW,
    Avx2,
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
        let (mr, nr, reg_dim) = if features.avx512_vnni {
            (AVX512_VNNI_MR, AVX512_VNNI_NR, RegDim::Avx512VNNI)
        } else if features.avx512bw {
            (AVX512BW_MR, AVX512BW_NR, RegDim::Avx512BW)
        } else if features.avx2 {
            (AVX2_MR, AVX2_NR, RegDim::Avx2)
        } else {
            (SSE_MR, SSE_NR, RegDim::Sse)
        };
        let vs = if features.avx512bw || features.avx512_vnni {
            AVX512F_VS
        } else if features.avx2 {
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
        (k + 1) / 2 * 2
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
    alpha: *const f32,
    beta: *const f32,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap: *const TA,
    bp: *const TB,
    kc_last: bool,
) {
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Avx512VNNI => avx512_vnni::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Avx512BW => avx512bw::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Avx2 => avx2::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Sse => sse::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Avx512VNNI => avx512_vnni::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Avx512BW => avx512bw::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Avx2 => avx2::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Sse => sse::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
        }
    }
}

#[allow(unused)]
unsafe fn kernel_m<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    b: *const TB,
    b_rs: usize,
    b_cs: usize,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap: *const TA,
    kc_last: bool,
) {
    panic!("Not implemented");
}

unsafe fn kernel_n<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
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
            RegDim::Avx512VNNI => {
                avx512_vnni::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func)
            }
            RegDim::Avx512BW => {
                avx512bw::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func)
            }
            RegDim::Avx2 => avx2::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::Sse => sse::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Avx512VNNI => {
                avx512_vnni::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn)
            }
            RegDim::Avx512BW => avx512bw::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::Avx2 => avx2::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::Sse => sse::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
        }
    }
}

unsafe fn glar_gemv<F: UnaryFnC>(
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
        RegDim::Avx512VNNI | RegDim::Avx512BW | RegDim::Avx2 => {
            avx2::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
        RegDim::Sse => sse::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func),
    }
    return;
}

def_glar_gemm!(
    KernelDispatcher,
    i16,
    i16,
    i16,
    i16,
    i32,
    f32,
    f32,
    PackArrTypeA,
    PackArrTypeB,
    1_f32,
    glar_gemm,
    gemm_mt,
    gemm_goto_serial,
    kernel,
    gemm_small_m_serial,
    kernel_m,
    gemm_small_n_serial,
    kernel_n,
    glar_gemv,
    glar_gemv,
    packa0,
    packb0,
    packa_fn_simd,
    packb_fn_simd,
    false,
    true,
    into_pack_array,
    F,
);
