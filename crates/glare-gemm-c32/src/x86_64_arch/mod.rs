pub(crate) mod avx;
pub(crate) mod avx512f;
pub(crate) mod avx_fma;
pub(crate) mod pack_avx;

use glare_base::{
    acquire, def_glare_gemm, def_pa, extend, get_apbp_barrier, get_mem_pool_size_goto,
    get_mem_pool_size_small_m, get_mem_pool_size_small_n, is_mixed, run_small_m, run_small_n,
    split_c_range, split_range, Array, ArrayMut, CpuFeatures, GemmPool, GlarePar,
    GlareThreadConfig, HWConfig, PArray, PoolSize, PtrData, PACK_POOL,
};

use crate::{GemmCache, MyFn, NullFn, TA, TB, TC};

pub(crate) struct X86_64dispatcher<T: MyFn = NullFn> {
    mc: usize,
    nc: usize,
    kc: usize,
    pub(crate) mr: usize,
    pub(crate) nr: usize,
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
            (16, 7)
        } else if features.avx && features.fma {
            (8, 3)
        } else {
            (8, 2)
        };
        let vs = if features.avx512f { 8 } else { 4 };
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

    pub(crate) unsafe fn packa_fn(
        &self,
        x: *const TA,
        y: *mut TA,
        m: usize,
        k: usize,
        rs: usize,
        cs: usize,
    ) {
        if self.mr == 24 {
            pack_avx::packa_panel_24(m, k, x, rs, cs, y, self.vs);
            return;
        }
        if self.mr == 16 {
            pack_avx::packa_panel_16(m, k, x, rs, cs, y, self.vs);
            return;
        }
        if self.mr == 8 {
            pack_avx::packa_panel_8(m, k, x, rs, cs, y, self.vs);
            return;
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
        if self.nr == 4 {
            pack_avx::packb_panel_4(n, k, x, cs, rs, y);
            return;
        }
        if self.nr == 7 {
            pack_avx::packb_panel_7(n, k, x, cs, rs, y);
            return;
        }
        if self.nr == 3 {
            pack_avx::packb_panel_3(n, k, x, cs, rs, y);
            return;
        }
        if self.nr == 2 {
            pack_avx::packb_panel_2(n, k, x, cs, rs, y);
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
    _kc_last: bool,
    kc_first: bool,
) {
    if hw_cfg.features.avx512f {
        if kc_first {
            avx::scale_c(m, n, beta, c, c_rs, c_cs);
        }
        avx512f::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, hw_cfg.func);
        return;
    }
    if hw_cfg.features.avx && hw_cfg.features.fma {
        if kc_first {
            avx::scale_c(m, n, beta, c, c_rs, c_cs);
        }
        if hw_cfg.mr == 8 && hw_cfg.nr == 3 {
            avx_fma::kernel_8x3(m, n, k, alpha, c, c_rs, c_cs, ap, bp, hw_cfg.func);
            return;
        }
    }

    if hw_cfg.features.avx {
        if kc_first {
            avx::scale_c(m, n, beta, c, c_rs, c_cs);
        }
        avx::kernel(m, n, k, alpha, c, c_rs, c_cs, ap, bp, hw_cfg.func);
        return;
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
    _kc_last: bool,
    kc_first: bool,
) {
    if hw_cfg.features.avx512f {
        if kc_first {
            avx::scale_c(m, n, beta, c, c_rs, c_cs);
        }
        avx512f::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
    if hw_cfg.features.avx && hw_cfg.features.fma {
        if kc_first {
            avx::scale_c(m, n, beta, c, c_rs, c_cs);
        }
        if hw_cfg.mr == 8 && hw_cfg.nr == 3 {
            avx_fma::kernel_8x3_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
            return;
        }
    }
    if hw_cfg.features.avx {
        if kc_first {
            avx::scale_c(m, n, beta, c, c_rs, c_cs);
        }
        avx::kernel_bs(m, n, k, alpha, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
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
    _kc_last: bool,
    kc_first: bool,
) {
    if hw_cfg.features.avx512f {
        if kc_first {
            avx::scale_c(m, n, beta, c, c_rs, c_cs);
        }
        avx512f::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
    if hw_cfg.features.avx && hw_cfg.features.fma {
        if kc_first {
            avx::scale_c(m, n, beta, c, c_rs, c_cs);
        }
        if hw_cfg.mr == 8 && hw_cfg.nr == 3 {
            avx_fma::kernel_8x3_sb(
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
            );
            return;
        }
    }
    if hw_cfg.features.avx {
        if kc_first {
            avx::scale_c(m, n, beta, c, c_rs, c_cs);
        }
        avx::kernel_sb(m, n, k, alpha, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
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
    if hw_cfg.features.avx512f || (hw_cfg.features.avx && hw_cfg.features.fma) {
        avx_fma::axpy(
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
        );
        return;
    }
    if hw_cfg.features.avx || hw_cfg.features.avx512f {
        avx::axpy(
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
        );
        return;
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
