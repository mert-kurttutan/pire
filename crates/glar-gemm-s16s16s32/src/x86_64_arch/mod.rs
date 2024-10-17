pub(crate) mod avx2;
pub(crate) mod avx512_vnni;
pub(crate) mod avx512bw;
pub(crate) mod pack_avx;
pub(crate) mod pack_sse;
pub(crate) mod sse;

const AVX2_MR: usize = 16; // register block size
const AVX2_NR: usize = 4; // register block size

const AVX512BW_MR: usize = 32; // register block size
const AVX512BW_NR: usize = 8; // register block size

const VS: usize = 8; // vector size in float, __m256

use glar_base::{
    acquire, def_glar_gemm, def_pa, extend, get_apbp_barrier, get_cache_params, get_mem_pool_size_goto,
    get_mem_pool_size_small_m, get_mem_pool_size_small_n, is_mixed, run_small_m, run_small_n, split_c_range,
    split_range, Array, ArrayMut, GemmPool, GlarPar, GlarThreadConfig, HWConfig, HWModel, PArray, PoolSize, PtrData,
    PACK_POOL, RUNTIME_HW_CONFIG,
};

use crate::{GemmCache, MyFn, NullFn, TA, TB, TC};

#[inline(always)]
pub(crate) fn get_mcnckc() -> (usize, usize, usize) {
    // let mc = std::env::var("GLAR_MC").unwrap_or("4800".to_string()).parse::<usize>().unwrap();
    // let nc = std::env::var("GLAR_NC").unwrap_or("320".to_string()).parse::<usize>().unwrap();
    // let kc = std::env::var("GLAR_KC").unwrap_or("192".to_string()).parse::<usize>().unwrap();
    // return (mc, nc, kc);
    let (mc, nc, kc) = match (*RUNTIME_HW_CONFIG).hw_model {
        HWModel::Skylake => (4800, 192, 512),
        HWModel::Haswell => (4800, 320, 576),
        _ => get_cache_params(),
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
    Reg48x8,
    Reg32x8,
    Reg16x4,
    Reg8x4,
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
    // features: CpuFeatures,
    func: T,
}

impl<F: MyFn> KernelDispatcher<F> {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, f: F) -> Self {
        let (mc, nc, kc) = get_mcnckc();
        let features = hw_config.cpu_ft();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (mr, nr, reg_dim) = if features.avx512_vnni {
            (48, 8, RegDim::Reg48x8)
        } else if features.avx512bw {
            (AVX512BW_MR, AVX512BW_NR, RegDim::Reg32x8)
        } else if features.avx2 {
            (AVX2_MR, AVX2_NR, RegDim::Reg16x4)
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

    pub(crate) unsafe fn packa_fn(&self, x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
        match self.reg_dim {
            RegDim::Reg48x8 => pack_avx::packa_panel_48(m, k, x, rs, cs, y, self.vs),
            RegDim::Reg32x8 => pack_avx::packa_panel_32(m, k, x, rs, cs, y, self.vs),
            RegDim::Reg16x4 => pack_avx::packa_panel_16(m, k, x, rs, cs, y, self.vs),
            RegDim::Reg8x4 => pack_sse::packa_panel_8(m, k, x, rs, cs, y, self.vs),
        }
    }

    pub(crate) unsafe fn packb_fn(&self, x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
        match self.reg_dim {
            RegDim::Reg48x8 => pack_avx::packb_panel_8(n, k, x, cs, rs, y),
            RegDim::Reg32x8 => pack_avx::packb_panel_8(n, k, x, cs, rs, y),
            RegDim::Reg16x4 => pack_avx::packb_panel_4(n, k, x, cs, rs, y),
            RegDim::Reg8x4 => pack_sse::packb_panel_4(n, k, x, cs, rs, y),
        }
    }

    pub(crate) fn is_compute_native(&self) -> bool {
        true
    }

    pub(crate) fn round_up(&self, k: usize) -> usize {
        (k + 1) / 2 * 2
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
    if kc_last {
        match hw_cfg.reg_dim {
            RegDim::Reg48x8 => avx512_vnni::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg32x8 => avx512bw::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg16x4 => avx2::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
            RegDim::Reg8x4 => sse::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func),
        }
    } else {
        let null_fn = NullFn {};
        match hw_cfg.reg_dim {
            RegDim::Reg48x8 => avx512_vnni::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg32x8 => avx512bw::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg16x4 => avx2::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
            RegDim::Reg8x4 => sse::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, null_fn),
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
    kc_first: bool,
) {
    panic!("Not implemented");
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
        let null_fn = NullFn {};
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
        RegDim::Reg48x8 | RegDim::Reg32x8 | RegDim::Reg16x4 => {
            avx2::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func)
        }
        RegDim::Reg8x4 => sse::axpy(m, n, alpha, a.src(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func),
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
    packa,
    packb,
    false,
    true,
    into_pack_array,
    F,
);
