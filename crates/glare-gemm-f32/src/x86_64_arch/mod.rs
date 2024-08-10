pub(crate) mod avx_fma_microkernel;
pub(crate) mod avx512f_microkernel;
pub(crate) mod avx_microkernel;
pub(crate) mod pack_avx;

const AVX_GOTO_MR: usize = 24; // register block size
const AVX_GOTO_NR: usize = 4; // register block size

const AVX_FMA_GOTO_MR: usize = 24; // register block size
const AVX_FMA_GOTO_NR: usize = 4; // register block size

const AVX512F_GOTO_MR: usize = 48; // register block size
const AVX512F_GOTO_NR: usize = 8; // register block size


const VS: usize = 8; // vector size in float, __m256

use glare_base::split_c_range;
use glare_base::split_range;
use glare_base::def_glare_gemm;

use glare_base::{
    GlarePar, GlareThreadConfig,
   CpuFeatures,
   HWConfig,
   Array,
   ArrayMut,
    PArray,
    get_mem_pool_size_goto,
    get_mem_pool_size_small_m,
    get_mem_pool_size_small_n,
    run_small_m, run_small_n,
    get_ap_bp, get_apbp_barrier,
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
    goto_mc: usize,
    goto_nc: usize,
    goto_kc: usize,
    goto_mr: usize,
    goto_nr: usize,
    // TODO: Cech jr parallelism is beneificial for perf
    // is_l1_shared: bool,
    is_l2_shared: bool,
    is_l3_shared: bool,
    func: T,
    features: CpuFeatures,
}

impl<F: MyFn> X86_64dispatcher<F> {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, features: CpuFeatures, f: F) -> Self {
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (goto_mr, goto_nr) = if features.avx512f {
            (AVX512F_GOTO_MR, AVX512F_GOTO_NR)
        } else if features.avx && features.fma {
            (AVX_FMA_GOTO_MR, AVX_FMA_GOTO_NR)
        } else {
            (AVX_GOTO_MR, AVX_GOTO_NR)
        };
        Self {
            goto_mc: mc,
            goto_nc: nc,
            goto_kc: kc,
            // is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            func: f,
            features,
            goto_mr,
            goto_nr,
        }
    }

    unsafe fn packa_fn(self: &Self, x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
        if self.features.avx512f {
            pack_avx::packa_panel_48(m, k, x, rs, cs, y);
            return;
        } 
        if self.features.avx && self.features.fma {
            pack_avx::packa_panel_24(m, k, x, rs, cs, y);
            return;
        }
        if self.features.avx {
            pack_avx::packa_panel_24(m, k, x, rs, cs, y);
            return;
        }
    }

    unsafe fn packb_fn(self: &Self, x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
        if self.features.avx512f {
            pack_avx::packb_panel_8(n, k, x, cs, rs, y);
            return;
        }
        if self.features.avx && self.features.fma {
            pack_avx::packb_panel_4(n, k, x, cs, rs, y);
            return;
        }
        if self.features.avx {
            pack_avx::packb_panel_4(n, k, x, cs, rs, y);
            return;
        }
    }
}

impl<
T: MyFn,
AP, BP,
> GemmCache<AP,BP> for X86_64dispatcher<T> {
    // const CACHELINE_PAD: usize = 256;
    fn mr(&self) -> usize {
        self.goto_mr
    }
    fn nr(&self) -> usize {
        self.goto_nr
    }
    fn get_kc_eff(&self) -> usize {self.goto_kc}
    fn get_mc_eff(&self, par: usize) -> usize {
        if self.is_l3_shared {
            (self.goto_mc / (self.goto_mr * par)) * self.goto_mr
        } else {
            self.goto_mc
        }
    }
    fn get_nc_eff(&self, par: usize) -> usize {
        if self.is_l2_shared {
            (self.goto_nc / (self.goto_nr * par)) * self.goto_nr
        } else {
            self.goto_nc
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
    _kc_last: bool
) {
 if hw_cfg.features.avx512f {
     avx512f_microkernel::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func);
     return;
 }
 if hw_cfg.features.avx && hw_cfg.features.fma {
     avx_fma_microkernel::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func);
     return;
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
) {
    if hw_cfg.features.avx512f {
        avx512f_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
    if hw_cfg.features.avx && hw_cfg.features.fma {
        avx_fma_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
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
) {
    if hw_cfg.features.avx512f {
        avx512f_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
    if hw_cfg.features.avx && hw_cfg.features.fma {
        avx_fma_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
    if hw_cfg.features.avx {
        avx_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
}

unsafe fn glare_gemv<F:MyFn>(
    hw_cfg: &X86_64dispatcher<F>,
    m: usize, n: usize,
    alpha: *const f32,
    a: Array<TA>,
    x: Array<TB>,
    beta: *const f32,
    y: ArrayMut<TC>,
) {
    let x_ptr = x.data_ptr();
    let inc_x = x.rs();
    let y_ptr   = y.data_ptr();
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


def_glare_gemm!(
    X86_64dispatcher,
    f32,f32,f32,f32,f32,f32,f32,
    1_f32,
    glare_gemm, gemm_mt,
    gemm_goto_serial, kernel,
    gemm_small_m_serial, kernel_m,
    gemm_small_n_serial, kernel_n,
    glare_gemv,
    packa, packb,
    true, true,
);
