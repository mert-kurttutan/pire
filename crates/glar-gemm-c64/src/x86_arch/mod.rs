pub(crate) mod pack_sse;
pub(crate) mod sse;

use glar_base::{
    acquire, def_glar_gemm, def_pa, extend, get_apbp_barrier, get_mem_pool_size_goto, get_mem_pool_size_small_m,
    get_mem_pool_size_small_n, is_mixed, run_small_m, run_small_n, split_c_range, split_range, Array, ArrayMut,
    GemmPool, GlarPar, GlarThreadConfig, PArray, PoolSize, PtrData, PACK_POOL, RUNTIME_HW_CONFIG,
};

use crate::{GemmCache, IdentityFn, UnaryFnC, TA, TB, TC};

const VS: usize = 1;
const MR: usize = 1;
const NR: usize = 2;

#[inline(always)]
pub(crate) fn get_mcnckc_simd() -> (usize, usize, usize) {
    // let mc = std::env::var("GLAR_MC").unwrap_or("4800".to_string()).parse::<usize>().unwrap();
    // let nc = std::env::var("GLAR_NC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // let kc = std::env::var("GLAR_KC").unwrap_or("768".to_string()).parse::<usize>().unwrap();
    // return (mc, nc, kc);
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        _ => (4800, 320, 192),
    };
    (mc, nc, kc)
}

pub(crate) unsafe fn packa_fn_simd(x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
    pack_sse::packa_panel_1(m, k, x, rs, cs, y, 1);
}

pub(crate) unsafe fn packb_fn_simd(x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
    pack_sse::packb_panel_2(n, k, x, cs, rs, y);
}

pub(crate) fn round_m_simd(m: usize) -> usize {
    let vs = VS;
    (m + vs - 1) / vs * vs
}

pub(crate) fn round_k_simd(k: usize) -> usize {
    k
}

pub(crate) enum RegDim {
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
    func: T,
}

impl<F: UnaryFnC> KernelDispatcher<F> {
    pub(crate) fn new(f: F) -> Self {
        let hw_config = &*RUNTIME_HW_CONFIG;
        let (mc, nc, kc) = get_mcnckc_simd();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();

        let (mr, nr, reg_dim) = (MR, NR, RegDim::Sse);
        let vs = VS;
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
            RegDim::Sse => sse::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
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
            RegDim::Sse => sse::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
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
            RegDim::Sse => sse::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Sse => sse::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
        }
    }
}

unsafe fn glar_gemv<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    alpha: *const TA,
    a: Array<TA>,
    x: Array<TB>,
    beta: *const TC,
    y: ArrayMut<TC>,
) {
    let x_ptr = x.src();
    let inc_x = x.rs();
    let y_ptr = y.src();
    let incy = y.rs();
    match hw_cfg.reg_dim {
        RegDim::Sse => sse::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func),
    }
}

def_glar_gemm!(
    KernelDispatcher,
    TA,
    TA,
    TB,
    TB,
    TC,
    TA,
    TC,
    PackArrTypeA,
    PackArrTypeB,
    TC::ONE,
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
    packa,
    packb,
    packa_fn_simd,
    packb_fn_simd,
    false,
    false,
    into_pack_array,
    F,
);
