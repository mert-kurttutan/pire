pub(crate) mod avx_fma_microkernel;
pub(crate) mod avx512f_microkernel;
pub(crate) mod avx_microkernel;
pub(crate) mod pack_avx;


use glare_base::split_c_range;
use glare_base::split_range;

use glare_base::def_glare_gemm;
use glare_base::is_mixed;

use glare_base::{
    GlarePar, GlareThreadConfig,
   CpuFeatures,
   HWConfig,
   HWModel,
   Array,
   PoolSize,
   ArrayMut,
    PArray,
    get_mem_pool_size_goto,
    get_mem_pool_size_small_m,
    get_mem_pool_size_small_n,
    run_small_m, run_small_n,
    get_apbp_barrier,
    extend, acquire,
    PACK_POOL,
    GemmPool,
};


use crate::{
   TA, TB, TC,
   GemmCache,
   MyFn, NullFn
};

pub(crate) struct X86_64dispatcher<
T: MyFn = NullFn
> {
    mc: usize,
    nc: usize,
    kc: usize,
    mr: usize,
    nr: usize,
    // TODO: Cech jr parallelism is beneificial for perf
    // is_l1_shared: bool,
    is_l2_shared: bool,
    is_l3_shared: bool,
    func: T,
    features: CpuFeatures,
    pub(crate) vs: usize,
}

impl<F: MyFn> X86_64dispatcher<F> {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, f: F) -> Self {
        let features = hw_config.cpu_ft();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (mr, nr) = if features.avx512f {
            (24, 8)
        } else if features.avx && features.fma {
            match hw_config.hw_model {
                HWModel::Broadwell => (8, 6),
                _ => (12, 4),
            }
        } else {
            (8, 4)
        };
        let vs = if features.avx512f {
            8 
        } else {
            4
        };
        Self {
            mc: mc,
            nc: nc,
            kc: kc,
            // is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            func: f,
            features,
            mr,
            nr,
            vs,
        }
    }

    pub(crate) unsafe fn packa_fn(&self, x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
        if self.mr == 24 {
            pack_avx::packa_panel_24(m, k, x, rs, cs, y, self.vs);
            return;
        } 
        if self.mr == 8 {
            pack_avx::packa_panel_8(m, k, x, rs, cs, y, self.vs);
            return;
        }
        if self.mr == 12 {
            pack_avx::packa_panel_12(m, k, x, rs, cs, y, self.vs);
            return;
        }
    }

    pub(crate) unsafe fn packb_fn(&self, x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
        if self.nr == 8 {
            pack_avx::packb_panel_8(n, k, x, cs, rs, y);
            return;
        }
        if self.nr == 6 {
            pack_avx::packb_panel_6(n, k, x, cs, rs, y);
            return;
        }
        if self.nr == 4 {
            pack_avx::packb_panel_4(n, k, x, cs, rs, y);
            return;
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
    fn get_kc_eff(&self) -> usize {self.kc}
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

unsafe fn kernel<F:MyFn>(
    hw_cfg: &X86_64dispatcher<F>,
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    c: *mut TC,
    c_rs: usize, c_cs: usize,
    ap: *const TA, bp: *const TB,
    _kc_last: bool, _kc_first: bool,
) {
 if hw_cfg.features.avx512f {
     avx512f_microkernel::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func);
     return;
 }
 if hw_cfg.features.avx && hw_cfg.features.fma {
    if hw_cfg.mr == 8 && hw_cfg.nr == 6 {
        avx_fma_microkernel::kernel_8x6(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func);
        return;
    }
    if hw_cfg.mr == 12 && hw_cfg.nr == 4 {
        avx_fma_microkernel::kernel_12x4(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func);
        return;
    }
 }
 if hw_cfg.features.avx {
    avx_microkernel::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func);
    return;
}
}

unsafe fn kernel_m<F:MyFn>(
    hw_cfg: &X86_64dispatcher<F>,
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    b: *const TB, b_rs: usize, b_cs: usize,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap: *const TA,
    _kc_last: bool, _kc_first: bool,
) {
    if hw_cfg.features.avx512f {
        avx512f_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
    if hw_cfg.features.avx && hw_cfg.features.fma {
        if hw_cfg.mr == 8 && hw_cfg.nr == 6 {
            avx_fma_microkernel::kernel_8x6_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
            return;
        }
        if hw_cfg.mr == 12 && hw_cfg.nr == 4 {
            avx_fma_microkernel::kernel_12x4_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
            return;
        }
    }
    if hw_cfg.features.avx {
        avx_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
}


unsafe fn kernel_n<F:MyFn>(
    hw_cfg: &X86_64dispatcher<F>,
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    a: *const TA, a_rs: usize, a_cs: usize,
    ap: *mut TA,
    b: *const TB,
    c: *mut TC, c_rs: usize, c_cs: usize,
    _kc_last: bool, _kc_first: bool,
) {
    if hw_cfg.features.avx512f {
        avx512f_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
    if hw_cfg.features.avx && hw_cfg.features.fma {
        if hw_cfg.mr == 8 && hw_cfg.nr == 6 {
            avx_fma_microkernel::kernel_8x6_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
            return;
        }
        if hw_cfg.mr == 12 && hw_cfg.nr == 4 {
            avx_fma_microkernel::kernel_12x4_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
            return;
        }
    }
    if hw_cfg.features.avx {
        avx_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
}

unsafe fn glare_gemv<F:MyFn>(
    hw_cfg: &X86_64dispatcher<F>,
    m: usize, n: usize,
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
    if hw_cfg.features.avx512f || (hw_cfg.features.avx && hw_cfg.features.fma) {
        avx_fma_microkernel::axpy(m, n, alpha, a.data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func);
        return;
    }
    if hw_cfg.features.avx || hw_cfg.features.avx512f {
        avx_microkernel::axpy(m, n, alpha, a.data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, hw_cfg.func);
        return;
    }
}

type F64Pack = PArray<f64>;

def_glare_gemm!(
    X86_64dispatcher,
    f64,f64,f64,f64,f64,f64,f64,
    F64Pack, F64Pack,
    1_f64,
    glare_gemm, gemm_mt,
    gemm_goto_serial, kernel,
    gemm_small_m_serial, kernel_m,
    gemm_small_n_serial, kernel_n,
    glare_gemv, glare_gemv,
    packa, packb,
    true, true,
    into_pack_array, F,
);
