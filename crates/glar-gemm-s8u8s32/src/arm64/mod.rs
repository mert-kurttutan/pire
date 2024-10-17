// pub(crate) mod neon;
// pub(crate) mod pack_neon;
pub(crate) mod pack_sve;
pub(crate) mod sve;

use glar_base::{
    acquire, def_glar_gemm, def_pa, extend, get_apbp_barrier, get_mem_pool_size_goto, get_mem_pool_size_small_m,
    get_mem_pool_size_small_n, is_mixed, run_small_m, run_small_n, split_c_range, split_range, Array, ArrayMut,
    GemmPool, GlarPar, GlarThreadConfig, HWConfig, PArray, PoolSize, PtrData, PACK_POOL, RUNTIME_HW_CONFIG,
};

use crate::{GemmCache, MyFn, NullFn, TA, TB, TC};

#[inline(always)]
pub(crate) fn get_mcnckc() -> (usize, usize, usize) {
    // let mc = std::env::var("GLAR_MC").unwrap_or("4800".to_string()).parse::<usize>().unwrap();
    // let nc = std::env::var("GLAR_NC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // let kc = std::env::var("GLAR_KC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // return (mc, nc, kc);
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        _ => (4800, 192, 384),
    };
    (mc, nc, kc)
}

pub(crate) unsafe fn packa_full(m: usize, k: usize, a: *const TA, a_rs: usize, a_cs: usize, ap: *mut TA) -> Array<TA> {
    let (mc, _, kc) = get_mcnckc();
    assert_eq!(ap.align_offset(glar_base::AB_ALIGN), 0);
    let hw_config = KernelDispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, NullFn {});
    let mut ap_cur = ap;
    let vs = hw_config.vs;
    for p in (0..k).step_by(kc) {
        let kc_len = kc.min(k - p);
        let kc_len_eff = hw_config.round_up(kc_len);
        for i in (0..m).step_by(mc) {
            let mc_len = mc.min(m - i);
            let mc_len_eff = (mc_len + vs - 1) / vs * vs;
            let a_cur = a.add(i * a_rs + p * a_cs);
            hw_config.packa_fn(a_cur, ap_cur, mc_len, kc_len, a_rs, a_cs);
            ap_cur = ap_cur.add(mc_len_eff * kc_len_eff);
        }
    }
    return Array::packed_matrix(ap, m, k);
}

pub(crate) unsafe fn packb_full(n: usize, k: usize, b: *const TB, b_rs: usize, b_cs: usize, bp: *mut TB) -> Array<TB> {
    let (_, nc, kc) = get_mcnckc();
    assert_eq!(bp.align_offset(glar_base::AB_ALIGN), 0);
    let hw_config = KernelDispatcher::from_hw_cfg(&*RUNTIME_HW_CONFIG, NullFn {});
    let mut bp_cur = bp;
    for p in (0..k).step_by(kc) {
        let kc_len = kc.min(k - p);
        let kc_len_eff = hw_config.round_up(kc_len);
        for i in (0..n).step_by(nc) {
            let nc_len = nc.min(n - i);
            let nc_len_eff = nc_len;
            let b_cur = b.add(i * b_cs + p * b_rs);
            hw_config.packb_fn(b_cur, bp_cur, nc_len, kc_len, b_rs, b_cs);
            bp_cur = bp_cur.add(nc_len_eff * kc_len_eff);
        }
    }
    return Array::packed_matrix(bp, n, k);
}

pub(crate) enum RegDim {
    // Reg24x4,
    RegMrx8,
}

pub(crate) struct KernelDispatcher<T: MyFn = NullFn> {
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

impl<F: MyFn> KernelDispatcher<F> {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, f: F) -> Self {
        let (mc, nc, kc) = get_mcnckc();
        let features = hw_config.cpu_ft();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();

        // let (mr, nr, reg_dim) = if features.sve {
        //     (24, 8, RegDim::RegMrx8)
        // } else {
        //     (24, 4, RegDim::Reg24x4)
        // };
        let (mr, nr, reg_dim) = (16, 12, RegDim::RegMrx8);
        let vs = if features.sve { 16 } else { 8 };
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
            // RegDim::Reg24x4 => pack_neon::packa_panel_24(m, k, x, rs, cs, y, self.vs),
            RegDim::RegMrx8 => pack_sve::packa_panel(m, k, x, rs, cs, y, self.vs, self.mr),
        }
    }

    pub(crate) unsafe fn packb_fn(&self, x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
        match self.reg_dim {
            // RegDim::Reg24x4 => pack_neon::packb_panel_4(n, k, x, cs, rs, y),
            RegDim::RegMrx8 => pack_sve::packb_panel_12(n, k, x, cs, rs, y),
        }
    }

    pub(crate) fn is_compute_native(&self) -> bool {
        true
    }

    pub(crate) fn round_up(&self, k: usize) -> usize {
        (k + 7) / 8 * 8
    }

    pub(crate) fn mv(&self) -> usize {
        self.mr
    }

    pub(crate) fn nv(&self) -> usize {
        1
    }

    pub(crate) fn kv(&self) -> usize {
        8
    }
}

impl<T: MyFn> GemmCache for KernelDispatcher<T> {
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
    _kc_first: bool,
) {
    let mr = hw_cfg.mr;
    let nr = hw_cfg.nr;
    if kc_last {
        match hw_cfg.reg_dim {
            // RegDim::Reg24x4 => neon::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::RegMrx8 => sve::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, mr, nr, hw_cfg.func),
        }
    } else {
        let null_fn = NullFn {};
        match hw_cfg.reg_dim {
            // RegDim::Reg24x4 => neon::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::RegMrx8 => sve::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, mr, nr, null_fn),
        }
    }
}

#[allow(unused)]
unsafe fn kernel_m<F: MyFn>(
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
    _kc_first: bool,
) {
}

unsafe fn kernel_n<F: MyFn>(
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
    _kc_first: bool,
) {
    let mr = hw_cfg.mr;
    let nr = hw_cfg.nr;
    if kc_last {
        match hw_cfg.reg_dim {
            // RegDim::Reg24x4 => neon::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func),
            RegDim::RegMrx8 => {
                sve::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, mr, nr, hw_cfg.func)
            }
        }
    } else {
        let null_fn = NullFn {};
        match hw_cfg.reg_dim {
            // RegDim::Reg24x4 => neon::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, null_fn),
            RegDim::RegMrx8 => {
                sve::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, mr, nr, null_fn)
            }
        }
    }
}

unsafe fn glar_gemv<F: MyFn>(
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
        // RegDim::Reg24x4 => {
        //     neon::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        // }
        RegDim::RegMrx8 => {
            sve::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
    }
}

unsafe fn glar_gemv2<F: MyFn>(
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
        // RegDim::Reg24x4 => {
        //     neon::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        // }
        RegDim::RegMrx8 => {
            sve::axpy2(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
    }
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
    packa,
    packb,
    false,
    false,
    into_pack_array,
    F,
);
