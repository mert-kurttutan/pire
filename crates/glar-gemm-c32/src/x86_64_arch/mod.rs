pub(crate) mod avx;
pub(crate) mod avx512f;
pub(crate) mod avx_fma;
pub(crate) mod pack_avx;
pub(crate) mod pack_sse;
pub(crate) mod sse;

use glar_base::{
    acquire, def_glar_gemm, def_pa, extend, get_apbp_barrier, get_cache_params, get_mem_pool_size_goto,
    get_mem_pool_size_small_m, get_mem_pool_size_small_n, is_mixed, run_small_m, run_small_n, split_c_range,
    split_range, Array, ArrayMut, GemmPool, GlarPar, GlarThreadConfig, HWConfig, HWModel, PArray, PoolSize, PtrData,
    PACK_POOL, RUNTIME_HW_CONFIG,
};

use crate::{GemmCache, MyFn, NullFn, TA, TB, TC};

#[inline(always)]
pub(crate) fn get_mcnckc() -> (usize, usize, usize) {
    // let mc = std::env::var("GLAR_MC").unwrap_or("2400".to_string()).parse::<usize>().unwrap();
    // let nc = std::env::var("GLAR_NC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // let kc = std::env::var("GLAR_KC").unwrap_or("512".to_string()).parse::<usize>().unwrap();
    // return (mc, nc, kc);
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        HWModel::Skylake => (2400, 192, 320),
        HWModel::Haswell => (2400, 192, 192),
        _ => get_cache_params(),
    };
    (mc, nc, kc)
}

pub(crate) unsafe fn packa_full(m: usize, k: usize, a: *const TA, a_rs: usize, a_cs: usize, ap: *mut TA) -> Array<TA> {
    let (mc, _, kc) = get_mcnckc();
    assert_eq!(ap.align_offset(glar_base::AB_ALIGN), 0);
    let hw_config = X86_64dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, NullFn {});
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
    let hw_config = X86_64dispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, NullFn {});
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

pub(crate) enum RegDim {
    Reg24x4,
    Reg12x2,
    Reg8x2,
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
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, f: F) -> Self {
        let (mc, nc, kc) = get_mcnckc();
        let features = hw_config.cpu_ft();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (mr, nr, reg_dim) = if features.avx512f {
            (24, 4, RegDim::Reg24x4)
        } else if features.avx && features.fma {
            (12, 2, RegDim::Reg12x2)
        } else if features.avx {
            (8, 2, RegDim::Reg8x2)
        } else {
            (4, 2, RegDim::Reg4x2)
        };
        let vs = if features.avx512f {
            8
        } else if features.avx {
            4
        } else {
            2
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

    pub(crate) unsafe fn packa_fn(&self, x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
        match self.reg_dim {
            RegDim::Reg24x4 => pack_avx::packa_panel_24(m, k, x, rs, cs, y, self.vs),
            RegDim::Reg12x2 => pack_avx::packa_panel_12(m, k, x, rs, cs, y, self.vs),
            RegDim::Reg8x2 => pack_avx::packa_panel_8(m, k, x, rs, cs, y, self.vs),
            RegDim::Reg4x2 => pack_sse::packa_panel_4(m, k, x, rs, cs, y, self.vs),
        }
    }

    pub(crate) unsafe fn packb_fn(&self, x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
        match self.reg_dim {
            RegDim::Reg24x4 => pack_avx::packb_panel_4(n, k, x, cs, rs, y),
            RegDim::Reg12x2 => pack_avx::packb_panel_2(n, k, x, cs, rs, y),
            RegDim::Reg8x2 => pack_avx::packb_panel_2(n, k, x, cs, rs, y),
            RegDim::Reg4x2 => pack_sse::packb_panel_2(n, k, x, cs, rs, y),
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
        match hw_cfg.reg_dim {
            RegDim::Reg24x4 | RegDim::Reg12x2 | RegDim::Reg8x2 => avx::scale_c(m, n, beta, c, c_rs, c_cs),
            RegDim::Reg4x2 => sse::scale_c(m, n, beta, c, c_rs, c_cs),
        }
    }
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Reg24x4 => avx512f::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg12x2 => avx_fma::kernel_12x2(m, n, k, alpha, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg8x2 => avx::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg4x2 => sse::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, hw_cfg.func),
        }
    } else {
        let null_fn = NullFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg24x4 => avx512f::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg12x2 => avx_fma::kernel_12x2(m, n, k, alpha, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg8x2 => avx::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg4x2 => sse::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, null_fn),
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
    // avx is common to all kernels
    if kc_first {
        match hw_cfg.reg_dim {
            RegDim::Reg24x4 | RegDim::Reg12x2 | RegDim::Reg8x2 => avx::scale_c(m, n, beta, c, c_rs, c_cs),
            RegDim::Reg4x2 => sse::scale_c(m, n, beta, c, c_rs, c_cs),
        }
    }
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Reg24x4 => avx512f::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::Reg12x2 => avx_fma::kernel_12x2_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::Reg8x2 => avx::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::Reg4x2 => sse::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func),
        }
    } else {
        let null_fn = NullFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg24x4 => avx512f::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn),
            RegDim::Reg12x2 => avx_fma::kernel_12x2_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn),
            RegDim::Reg8x2 => avx::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn),
            RegDim::Reg4x2 => sse::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, null_fn),
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
        match hw_cfg.reg_dim {
            RegDim::Reg24x4 | RegDim::Reg12x2 | RegDim::Reg8x2 => avx::scale_c(m, n, beta, c, c_rs, c_cs),
            RegDim::Reg4x2 => sse::scale_c(m, n, beta, c, c_rs, c_cs),
        }
    }
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Reg24x4 => avx512f::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::Reg12x2 => {
                avx_fma::kernel_12x2_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func)
            }
            RegDim::Reg8x2 => avx::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::Reg4x2 => sse::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
        }
    } else {
        let null_fn = NullFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg24x4 => avx512f::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::Reg12x2 => avx_fma::kernel_12x2_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::Reg8x2 => avx::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::Reg4x2 => sse::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
        }
    }
}

unsafe fn glar_gemv<F: MyFn>(
    hw_cfg: &X86_64dispatcher<F>,
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
        RegDim::Reg24x4 | RegDim::Reg12x2 => {
            avx_fma::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
        RegDim::Reg8x2 => avx::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func),
        RegDim::Reg4x2 => sse::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func),
    }
}

def_glar_gemm!(
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
    false,
    false,
    into_pack_array,
    F,
);
