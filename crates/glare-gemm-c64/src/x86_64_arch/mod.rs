pub(crate) mod avx;
pub(crate) mod avx512f;
pub(crate) mod avx_fma;
pub(crate) mod pack_avx;

use glare_base::{
    acquire, def_glare_gemm, def_pa, extend, get_apbp_barrier, get_mem_pool_size_goto,
    get_mem_pool_size_small_m, get_mem_pool_size_small_n, is_mixed, run_small_m, run_small_n,
    split_c_range, split_range, Array, ArrayMut, GemmPool, GlarePar, GlareThreadConfig, HWConfig,
    PArray, PoolSize, PtrData, PACK_POOL,
};

use crate::{GemmCache, MyFn, NullFn, TA, TB, TC};

pub(crate) enum RegDim {
    Reg8x7,
    Reg4x3,
    Reg4x2,
}

pub(crate) struct X86_64dispatcher<T: MyFn = NullFn> {
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

impl<F: MyFn> X86_64dispatcher<F> {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, f: F) -> Self {
        let features = hw_config.cpu_ft();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (mr, nr, reg_dim) = if features.avx512f {
            (8, 7, RegDim::Reg8x7)
        } else if features.avx && features.fma {
            (4, 3, RegDim::Reg4x3)
        } else {
            (4, 2, RegDim::Reg4x2)
        };
        let vs = if features.avx512f { 4 } else { 2 };
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

    pub(crate) unsafe fn packa_fn(
        &self,
        x: *const TA,
        y: *mut TA,
        m: usize,
        k: usize,
        rs: usize,
        cs: usize,
    ) {
        match self.reg_dim {
            RegDim::Reg8x7 => pack_avx::packa_panel_8(m, k, x, rs, cs, y, self.vs),
            RegDim::Reg4x3 | RegDim::Reg4x2 => pack_avx::packa_panel_4(m, k, x, rs, cs, y, self.vs),
        }
    }

    pub(crate) unsafe fn packb_fn(
        &self,
        x: *const TB,
        y: *mut TB,
        n: usize,
        k: usize,
        rs: usize,
        cs: usize,
    ) {
        match self.reg_dim {
            RegDim::Reg8x7 => pack_avx::packb_panel_7(n, k, x, cs, rs, y),
            RegDim::Reg4x3 => pack_avx::packb_panel_3(n, k, x, cs, rs, y),
            RegDim::Reg4x2 => pack_avx::packb_panel_2(n, k, x, cs, rs, y),
        }
    }

    pub(crate) fn is_compute_native(&self) -> bool {
        true
    }

    pub(crate) fn round_up(&self, k: usize) -> usize {
        k
    }
}

impl<T: MyFn> GemmCache for X86_64dispatcher<T> {
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

unsafe fn kernel<F: MyFn>(
    hw_cfg: &X86_64dispatcher<F>,
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
    kc_first: bool,
) {
    if kc_first {
        avx::scale_c(m, n, beta, c, c_rs, c_cs);
    }
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Reg8x7 => avx512f::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg4x3 => {
                avx_fma::kernel_4x3(m, n, k, alpha, c, c_rs, c_cs, ap, bp, hw_cfg.func)
            }
            RegDim::Reg4x2 => avx::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, hw_cfg.func),
        }
    } else {
        let null_fn = NullFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg8x7 => avx512f::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg4x3 => avx_fma::kernel_4x3(m, n, k, alpha, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg4x2 => avx::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, null_fn),
        }
    }
}

unsafe fn kernel_m<F: MyFn>(
    hw_cfg: &X86_64dispatcher<F>,
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
    kc_first: bool,
) {
    if kc_first {
        avx::scale_c(m, n, beta, c, c_rs, c_cs);
    }
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Reg8x7 => {
                avx512f::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func)
            }
            RegDim::Reg4x3 => avx_fma::kernel_4x3_bs(
                m,
                n,
                k,
                alpha,
                b,
                b_rs,
                b_cs,
                c,
                c_rs,
                c_cs,
                ap,
                hw_cfg.func,
            ),
            RegDim::Reg4x2 => {
                avx::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func)
            }
        }
    } else {
        let null_fn = NullFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg8x7 => {
                avx512f::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn)
            }
            RegDim::Reg4x3 => {
                avx_fma::kernel_4x3_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn)
            }
            RegDim::Reg4x2 => {
                avx::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn)
            }
        }
    }
}

unsafe fn kernel_n<F: MyFn>(
    hw_cfg: &X86_64dispatcher<F>,
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
    kc_first: bool,
) {
    if kc_first {
        avx::scale_c(m, n, beta, c, c_rs, c_cs);
    }
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Reg8x7 => {
                avx512f::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func)
            }
            RegDim::Reg4x3 => avx_fma::kernel_4x3_sb(
                m,
                n,
                k,
                alpha,
                a,
                a_rs,
                a_cs,
                b,
                c,
                c_rs,
                c_cs,
                ap,
                hw_cfg.func,
            ),
            RegDim::Reg4x2 => {
                avx::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func)
            }
        }
    } else {
        let null_fn = NullFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg8x7 => {
                avx512f::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn)
            }
            RegDim::Reg4x3 => {
                avx_fma::kernel_4x3_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn)
            }
            RegDim::Reg4x2 => {
                avx::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn)
            }
        }
    }
}

unsafe fn glare_gemv<F: MyFn>(
    hw_cfg: &X86_64dispatcher<F>,
    m: usize,
    n: usize,
    alpha: *const TA,
    a: Array<TA>,
    x: Array<TB>,
    beta: *const TC,
    y: ArrayMut<TC>,
) {
    let x_ptr = x.data_ptr();
    let inc_x = x.rs();
    let y_ptr = y.data_ptr();
    let incy = y.rs();
    match hw_cfg.reg_dim {
        RegDim::Reg8x7 | RegDim::Reg4x3 => avx_fma::axpy(
            m,
            n,
            alpha,
            a.data_ptr(),
            a.rs(),
            a.cs(),
            x_ptr,
            inc_x,
            beta,
            y_ptr,
            incy,
            hw_cfg.func,
        ),
        RegDim::Reg4x2 => avx::axpy(
            m,
            n,
            alpha,
            a.data_ptr(),
            a.rs(),
            a.cs(),
            x_ptr,
            inc_x,
            beta,
            y_ptr,
            incy,
            hw_cfg.func,
        ),
    }
}

def_glare_gemm!(
    X86_64dispatcher,
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
    glare_gemm,
    gemm_mt,
    gemm_goto_serial,
    kernel,
    gemm_small_m_serial,
    kernel_m,
    gemm_small_n_serial,
    kernel_n,
    glare_gemv,
    glare_gemv,
    packa,
    packb,
    false,
    false,
    into_pack_array,
    F,
);
