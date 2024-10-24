pub(crate) mod neon;
pub(crate) mod pack_neon;
pub(crate) mod pack_sve;
pub(crate) mod sve;

use glar_base::{
    acquire, def_glar_gemm, def_pa, extend, get_apbp_barrier, get_mem_pool_size_goto, get_mem_pool_size_small_m,
    get_mem_pool_size_small_n, is_mixed, run_small_m, run_small_n, split_c_range, split_range, Array, ArrayMut,
    GemmPool, GlarPar, GlarThreadConfig, PArray, PoolSize, PtrData, PACK_POOL, RUNTIME_HW_CONFIG,
};

use crate::{GemmCache, IdentityFn, UnaryFnC, TA, TB, TC};

const NEON_VS: usize = 4;
const NEON_MR: usize = 12;

const NEON_NR: usize = 2;
const SVE_NR: usize = 8;

#[inline(always)]
pub(crate) fn get_mcnckc_simd() -> (usize, usize, usize) {
    // let mc = std::env::var("GLAR_MC").unwrap_or("4800".to_string()).parse::<usize>().unwrap();
    // let nc = std::env::var("GLAR_NC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // let kc = std::env::var("GLAR_KC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // return (mc, nc, kc);
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        _ => (4800, 192, 384),
    };
    (mc, nc, kc)
}

pub(crate) unsafe fn packa_fn_simd(x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
    let features = (*RUNTIME_HW_CONFIG).cpu_ft();
    if features.sve && features.fcma {
        pack_sve::packa_panel(m, k, x, rs, cs, y, 12, 8);
    } else {
        pack_neon::packa_panel_12(m, k, x, rs, cs, y, NEON_VS);
    }
}

pub(crate) unsafe fn packb_fn_simd(x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
    let features = (*RUNTIME_HW_CONFIG).cpu_ft();
    if features.sve && features.fcma {
        pack_sve::packb_panel_8(n, k, x, cs, rs, y);
    } else {
        pack_neon::packb_panel_2(n, k, x, cs, rs, y);
    }
}

pub(crate) fn round_m_simd(m: usize) -> usize {
    let features = (*RUNTIME_HW_CONFIG).cpu_ft();
    let vs = if features.sve { 12 } else { NEON_VS };
    (m + vs - 1) / vs * vs
}

pub(crate) fn round_k_simd(k: usize) -> usize {
    k
}

pub(crate) enum RegDim {
    Reg12x2,
    RegMrx8,
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
        let features = hw_config.cpu_ft();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();

        let (mr, nr, reg_dim) = if features.sve && features.fcma {
            (12, SVE_NR, RegDim::RegMrx8)
        } else {
            (NEON_MR, NEON_NR, RegDim::Reg12x2)
        };
        let vs = if features.sve { 12 } else { NEON_VS };
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
            RegDim::Reg12x2 => neon::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::RegMrx8 => {
                sve::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.mr, hw_cfg.nr, hw_cfg.func)
            }
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg12x2 => neon::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::RegMrx8 => sve::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.mr, hw_cfg.nr, null_fn),
        }
    }
}

#[allow(unused)]
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
            RegDim::Reg12x2 => neon::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::RegMrx8 => sve::kernel_sb(
                m,
                n,
                k,
                alpha,
                beta,
                a,
                a_rs,
                a_cs,
                b,
                c,
                c_rs,
                c_cs,
                ap,
                hw_cfg.mr,
                hw_cfg.nr,
                hw_cfg.func,
            ),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg12x2 => neon::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::RegMrx8 => {
                sve::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.mr, hw_cfg.nr, null_fn)
            }
        }
    }
}

unsafe fn glar_gemv<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    alpha: *const TC,
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
        RegDim::Reg12x2 => {
            neon::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
        RegDim::RegMrx8 => {
            sve::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
    }
}

def_glar_gemm!(
    KernelDispatcher,
    TA,
    TC,
    TC,
    TC,
    TC,
    TC,
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
    true,
    into_pack_array,
    F,
);
