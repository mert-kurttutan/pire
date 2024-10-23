pub(crate) mod avx2;
pub(crate) mod avx512_vnni;
pub(crate) mod avx512bw;
pub(crate) mod pack_avx;
pub(crate) mod pack_sse;
pub(crate) mod sse;

use glar_base::{
    acquire, def_glar_gemm, def_pa, extend, get_apbp_barrier, get_cache_params, get_mem_pool_size_goto,
    get_mem_pool_size_small_m, get_mem_pool_size_small_n, is_mixed, run_small_m, run_small_n, split_c_range,
    split_range, Array, ArrayMut, GemmPool, GlarPar, GlarThreadConfig, HWModel, PArray, PoolSize, PtrData, AB_ALIGN,
    PACK_POOL, RUNTIME_HW_CONFIG,
};

use core::mem::size_of;

use crate::{GemmCache, IdentityFn, UnaryFnC, TA, TB, TC};

#[inline(always)]
pub(crate) fn get_mcnckc() -> (usize, usize, usize) {
    // let mc = std::env::var("GLAR_MC").unwrap_or("5400".to_string()).parse::<usize>().unwrap();
    // let nc = std::env::var("GLAR_NC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // let kc = std::env::var("GLAR_KC").unwrap_or("512".to_string()).parse::<usize>().unwrap();
    // return (mc, nc, kc);
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        HWModel::Skylake => (4800, 192, 512),
        HWModel::Haswell => (4800, 320, 768),
        _ => get_cache_params(),
    };
    (mc, nc, kc)
}

pub(crate) unsafe fn packa_full(m: usize, k: usize, a: *const TA, a_rs: usize, a_cs: usize, ap: *mut TA) -> Array<TA> {
    let (mc, _, kc) = get_mcnckc();
    assert_eq!(ap.align_offset(glar_base::AB_ALIGN), 0);
    let hw_config = KernelDispatcher::new(IdentityFn {});
    let mut ap_cur = ap;
    let vs = hw_config.vs;
    for p in (0..k).step_by(kc) {
        let kc_len = kc.min(k - p);
        let kc_len_eff = hw_config.round_k(kc_len);
        for i in (0..m).step_by(mc) {
            let mc_len = mc.min(m - i);
            let mc_len_eff = (mc_len + vs - 1) / vs * vs;
            let a_cur = a.add(i * a_rs + p * a_cs);
            packa_fn(a_cur, ap_cur, mc_len, kc_len, a_rs, a_cs);
            ap_cur = ap_cur.add(mc_len_eff * kc_len_eff);
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
        let kc_len_eff = hw_config.round_k(kc_len);
        for i in (0..n).step_by(nc) {
            let nc_len = nc.min(n - i);
            let nc_len_eff = nc_len;
            let b_cur = b.add(i * b_cs + p * b_rs);
            packb_fn(b_cur, bp_cur, nc_len, kc_len, b_rs, b_cs);
            bp_cur = bp_cur.add(nc_len_eff * kc_len_eff);
        }
    }
    return Array::packed_matrix(bp, n, k);
}

pub(crate) fn ap_size(m: usize, k: usize) -> usize {
    let hw_config = KernelDispatcher::new(IdentityFn {});
    let m_rounded = hw_config.round_m(m);
    let k_rounded = hw_config.round_k(k);
    m_rounded * k_rounded + AB_ALIGN / size_of::<TA>()
}

pub(crate) fn bp_size(n: usize, k: usize) -> usize {
    let hw_config = KernelDispatcher::new(IdentityFn {});
    let k_rounded = hw_config.round_k(k);
    n * k_rounded + AB_ALIGN / size_of::<TB>()
}

pub(crate) unsafe fn packa_fn(x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
    let hw_config = &*RUNTIME_HW_CONFIG;
    if hw_config.cpu_ft.avx512_vnni {
        pack_avx::packa_panel_48(m, k, x, rs, cs, y, 16);
    } else if hw_config.cpu_ft.avx512bw {
        pack_avx::packa_panel_32(m, k, x, rs, cs, y, 16);
    } else if hw_config.cpu_ft.avx2 {
        pack_avx::packa_panel_16(m, k, x, rs, cs, y, 8);
    } else {
        pack_sse::packa_panel_8(m, k, x, rs, cs, y, 4);
    }
}

pub(crate) unsafe fn packb_fn(x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
    if (*RUNTIME_HW_CONFIG).cpu_ft.avx512_vnni {
        pack_avx::packb_panel_8(n, k, x, cs, rs, y);
    } else if (*RUNTIME_HW_CONFIG).cpu_ft.avx512bw {
        pack_avx::packb_panel_8(n, k, x, cs, rs, y);
    } else if (*RUNTIME_HW_CONFIG).cpu_ft.avx2 {
        pack_avx::packb_panel_4(n, k, x, cs, rs, y);
    } else {
        pack_sse::packb_panel_4(n, k, x, cs, rs, y);
    }
}

pub(crate) enum RegDim {
    Reg48x8,
    Reg32x8,
    Reg16x4,
    Reg8x4,
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
        let (mc, nc, kc) = get_mcnckc();
        let features = hw_config.cpu_ft();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (mr, nr, reg_dim) = if features.avx512_vnni {
            (48, 8, RegDim::Reg48x8)
        } else if features.avx512bw {
            (32, 8, RegDim::Reg32x8)
        } else if features.avx2 {
            (16, 4, RegDim::Reg16x4)
        } else {
            (8, 4, RegDim::Reg8x4)
        };
        let vs = if features.avx512bw || features.avx512_vnni {
            16
        } else if features.avx2 {
            8
        } else {
            4
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

    // pub(crate) unsafe fn packa_fn(&self, x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
    //     match self.reg_dim {
    //         RegDim::Reg48x8 => pack_avx::packa_panel_48(m, k, x, rs, cs, y, self.vs),
    //         RegDim::Reg32x8 => pack_avx::packa_panel_32(m, k, x, rs, cs, y, self.vs),
    //         RegDim::Reg16x4 => pack_avx::packa_panel_16(m, k, x, rs, cs, y, self.vs),
    //         RegDim::Reg8x4 => pack_sse::packa_panel_8(m, k, x, rs, cs, y, self.vs),
    //     }
    // }

    // pub(crate) unsafe fn packb_fn(&self, x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
    //     match self.reg_dim {
    //         RegDim::Reg48x8 => pack_avx::packb_panel_8(n, k, x, cs, rs, y),
    //         RegDim::Reg32x8 => pack_avx::packb_panel_8(n, k, x, cs, rs, y),
    //         RegDim::Reg16x4 => pack_avx::packb_panel_4(n, k, x, cs, rs, y),
    //         RegDim::Reg8x4 => pack_sse::packb_panel_4(n, k, x, cs, rs, y),
    //     }
    // }

    pub(crate) fn is_compute_native(&self) -> bool {
        true
    }

    pub(crate) fn round_k(&self, k: usize) -> usize {
        (k + 3) / 4 * 4
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
            RegDim::Reg48x8 => avx512_vnni::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg32x8 => avx512bw::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg16x4 => avx2::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg8x4 => sse::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg48x8 => avx512_vnni::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg32x8 => avx512bw::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg16x4 => avx2::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg8x4 => sse::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
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
            RegDim::Reg48x8 => {
                avx512_vnni::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func)
            }
            RegDim::Reg32x8 => {
                avx512bw::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func)
            }
            RegDim::Reg16x4 => avx2::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::Reg8x4 => sse::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
        }
    } else {
        let null_fn = IdentityFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg48x8 => {
                avx512_vnni::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn)
            }
            RegDim::Reg32x8 => avx512bw::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::Reg16x4 => avx2::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::Reg8x4 => sse::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
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
        RegDim::Reg48x8 | RegDim::Reg32x8 | RegDim::Reg16x4 => {
            avx2::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
        RegDim::Reg8x4 => sse::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func),
    }
    return;
}

unsafe fn glar_gemv2<F: UnaryFnC>(
    hw_cfg: &KernelDispatcher<F>,
    m: usize,
    n: usize,
    alpha: *const f32,
    a: Array<TB>,
    x: Array<TA>,
    beta: *const f32,
    y: ArrayMut<TC>,
) {
    let x_ptr = x.src();
    let inc_x = x.rs();
    let y_ptr = y.src();
    let incy = y.rs();
    match hw_cfg.reg_dim {
        RegDim::Reg48x8 | RegDim::Reg32x8 | RegDim::Reg16x4 => {
            avx2::axpy2(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
        RegDim::Reg8x4 => {
            sse::axpy2(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
    }
    return;
}

def_glar_gemm!(
    KernelDispatcher,
    i8,
    i8,
    u8,
    u8,
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
    glar_gemv2,
    packa0,
    packb0,
    packa_fn,
    packb_fn,
    false,
    true,
    into_pack_array,
    F,
);
