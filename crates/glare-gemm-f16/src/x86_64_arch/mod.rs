pub(crate) mod avx_fma_microkernel;
pub(crate) mod avx512f_microkernel;
pub(crate) mod avx512_f16_microkernel;
pub(crate) mod pack_avx;
// pub(crate) mod pack_f16_avx;


use crate::f16;
// use avx_fma_microkernel::axpy;

const AVX_FMA_GOTO_MR: usize = 24; // register block size
const AVX_FMA_GOTO_NR: usize = 4; // register block size

const AVX512F_GOTO_MR: usize = 48; // register block size
const AVX512F_GOTO_NR: usize = 8; // register block size

const AVX512_F16_GOTO_MR: usize = 64; // register block size
const AVX512_F16_GOTO_NR: usize = 15; // register block size


use glare_base::split_c_range;
use glare_base::split_range;

use glare_base::def_glare_gemm;
use glare_base::is_mixed;

use glare_base::{
    GlarePar, GlareThreadConfig,
   CpuFeatures,
   HWConfig,
   Array,
   ArrayMut,
    PArray,
    PArrayMixed,
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

pub(crate) struct F32Dispatcher<
T: MyFn = NullFn
> {
    mc: usize,
    nc: usize,
    kc: usize,
    pub(crate) mr: usize,
    pub(crate) nr: usize,
    is_l1_shared: bool,
    is_l2_shared: bool,
    is_l3_shared: bool,
    func: T,
    features: CpuFeatures,
    pub(crate) vs: usize,
}


impl<F: MyFn> F32Dispatcher<F>{
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, features: CpuFeatures, f: F) -> Self {
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        
        let (mr, nr) = if features.avx512f {
            (AVX512F_GOTO_MR, AVX512F_GOTO_NR)
        } else if features.avx && features.fma {
            (AVX_FMA_GOTO_MR, AVX_FMA_GOTO_NR)
        } else {
            (AVX_FMA_GOTO_MR, AVX_FMA_GOTO_NR)
        };
        let vs = if features.avx512f {
            16
        } else {
            8
        };
        Self {
            mc: mc,
            nc: nc,
            kc: kc,
            is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            func: f,
            features: features,
            mr,
            nr,
            vs,
        }
    }

    pub(crate) fn is_compute_native(&self) -> bool {
        false
    }

    pub(crate) unsafe fn packa_fn(&self, x: *const f16, y: *mut f32, m: usize, k: usize, rs: usize, cs: usize) {
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

    pub(crate) unsafe fn packb_fn(&self, x: *const f16, y: *mut f32, n: usize, k: usize, rs: usize, cs: usize) {
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

    pub(crate) unsafe fn packa_fnsame(&self, x: *const f16, y: *mut f16, m: usize, k: usize, rs: usize, cs: usize) {
        if self.features.avx512f {
            pack_avx::packa_panel_48_same(m, k, x, rs, cs, y);
            return;
        } 
        if self.features.avx && self.features.fma {
            pack_avx::packa_panel_24_same(m, k, x, rs, cs, y);
            return;
        }
        if self.features.avx {
            pack_avx::packa_panel_24_same(m, k, x, rs, cs, y);
            return;
        }
    }

    pub(crate) unsafe fn packb_fnsame(&self, x: *const f16, y: *mut f16, n: usize, k: usize, rs: usize, cs: usize) {
        if self.features.avx512f {
            pack_avx::packb_panel_8_same(n, k, x, cs, rs, y);
            return;
        }
        if self.features.avx && self.features.fma {
            pack_avx::packb_panel_4_same(n, k, x, cs, rs, y);
            return;
        }
        if self.features.avx {
            pack_avx::packb_panel_4_same(n, k, x, cs, rs, y);
            return;
        }
    }

    unsafe fn cvt_mixed(&self, x: *const f16, y: *mut f32, m: usize) {
        use std::arch::x86_64::*;
        let m_iter = m / 8;
        let m_rem = m % 8;
        for i in 0..m_iter {
            let x_ptr = x.add(i * 8);
            let y_ptr = y.add(i * 8);
            let v_f32 = _mm256_cvtph_ps(_mm_loadu_si128(x_ptr as *const __m128i));
            _mm256_storeu_ps(y_ptr, v_f32);
        }
        for i in 0..m_rem {
            let x_ptr = x.add(m_iter * 8 + i);
            let y_ptr = y.add(m_iter * 8 + i);
            *y_ptr = (*x_ptr).to_f32();
        }
    }

    pub(crate) fn round_up(&self, k: usize) -> usize {
        k
    }
}

pub(crate) struct F16Dispatcher<
T: MyFn = NullFn
> {
    mc: usize,
    nc: usize,
    kc: usize,
    mr: usize,
    nr: usize,
    is_l1_shared: bool,
    is_l2_shared: bool,
    is_l3_shared: bool,
    func: T,
    features: CpuFeatures,
    pub(crate) vs: usize,
}

impl<F: MyFn> F16Dispatcher<F>{
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, features: CpuFeatures, f: F) -> Self {
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let mr = AVX512_F16_GOTO_MR;
        let nr = AVX512_F16_GOTO_NR;
        let vs = 32;
        Self {
            mc: mc,
            nc: nc,
            kc: kc,
            is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            func: f,
            mr,
            nr,
            features: features,
            vs: vs,
        }
    }


    pub(crate) fn is_compute_native(&self) -> bool {
        true
    }

    pub(crate) unsafe fn packa_fn(&self, x: *const f16, y: *mut f16, m: usize, k: usize, rs: usize, cs: usize) {
        if self.features.avx512f16 {
            pack_avx::packa_panel_64_same(m, k, x, rs, cs, y);
            return;
        } 
    }

    pub(crate) unsafe fn packb_fn(&self, x: *const f16, y: *mut f16, n: usize, k: usize, rs: usize, cs: usize) {
        if self.features.avx512f16 {
            pack_avx::packb_panel_15_same(n, k, x, cs, rs, y);
            return;
        }
    }

    pub(crate) fn round_up(&self, k: usize) -> usize {
        k
    }

}

impl<
T: MyFn,
AP, BP,
> GemmCache<AP,BP> for F32Dispatcher<T> {
    // const CACHELINE_PAD: usize = 256;
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


impl<
T: MyFn,
AP, BP,
> GemmCache<AP,BP> for F16Dispatcher<T> {
    // const CACHELINE_PAD: usize = 256;
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
    hw_cfg: &F32Dispatcher<F>,
    m: usize, n: usize, k: usize,
    alpha: *const f32,
    beta: *const f32,
    c: *mut f16,
    c_rs: usize, c_cs: usize,
    ap: *const f32, bp: *const f32,
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
}

#[allow(unused)]
unsafe fn kernel_m<F:MyFn>(
    hw_cfg: &F32Dispatcher<F>,
    m: usize, n: usize, k: usize,
    alpha: *const f32,
    beta: *const f32,
    b: *const f16, b_rs: usize, b_cs: usize,
    c: *mut f16, c_rs: usize, c_cs: usize,
    ap: *const f32,
) {
    // if hw_cfg.features.avx512f {
    //     avx512f_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
    //     return;
    // }
    // if hw_cfg.features.avx && hw_cfg.features.fma {
    //     avx_fma_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
    //     return;
    // }
}

#[allow(unused)]
unsafe fn kernel_n<F:MyFn>(
    hw_cfg: &F32Dispatcher<F>,
    m: usize, n: usize, k: usize,
    alpha: *const f32,
    beta: *const f32,
    a: *const f16, a_rs: usize, a_cs: usize,
    ap: *mut f32,
    b: *const f32,
    c: *mut f16, c_rs: usize, c_cs: usize,
) {
    // if hw_cfg.features.avx512f {
    //     avx512f_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
    //     return;
    // }
    // if hw_cfg.features.avx && hw_cfg.features.fma {
    //     avx_fma_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
    //     return;
    // }
}

unsafe fn glare_gemv<F:MyFn>(
    hw_cfg: &F32Dispatcher<F>,
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
}
type F16Pack = PArrayMixed<f16,f32>;

def_glare_gemm!(
    F32Dispatcher,
    f16,f32,f16,f32,f16,f32,f32,
    F16Pack, F16Pack,
    1_f32,
    glare_gemm, gemm_mt,
    gemm_goto_serial, kernel,
    gemm_small_m_serial, kernel_m,
    gemm_small_n_serial, kernel_n,
    glare_gemv, glare_gemv,
    packa, packb,
    false, false,
    into_pack_array2, T,
);



unsafe fn kernel_native<F:MyFn>(
    hw_cfg: &F16Dispatcher<F>,
    m: usize, n: usize, k: usize,
    alpha: *const f16,
    beta: *const f16,
    c: *mut f16,
    c_rs: usize, c_cs: usize,
    ap: *const f16, bp: *const f16,
    _kc_last: bool
) {
 if hw_cfg.features.avx512f {
     avx512_f16_microkernel::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, hw_cfg.func);
     return;
 }
}

#[allow(unused)]
unsafe fn kernel_m_native<F:MyFn>(
    hw_cfg: &F16Dispatcher<F>,
    m: usize, n: usize, k: usize,
    alpha: *const f16,
    beta: *const f16,
    b: *const f16, b_rs: usize, b_cs: usize,
    c: *mut f16, c_rs: usize, c_cs: usize,
    ap: *const f16,
) {
    if hw_cfg.features.avx512f {
        avx512_f16_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
}

#[allow(unused)]
unsafe fn kernel_n_native<F:MyFn>(
    hw_cfg: &F16Dispatcher<F>,
    m: usize, n: usize, k: usize,
    alpha: *const f16,
    beta: *const f16,
    a: *const f16, a_rs: usize, a_cs: usize,
    ap: *mut f16,
    b: *const f16,
    c: *mut f16, c_rs: usize, c_cs: usize,
) {
    if hw_cfg.features.avx512f {
        avx512_f16_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, hw_cfg.func);
        return;
    }
}

unsafe fn glare_gemv_native<F:MyFn>(
    hw_cfg: &F16Dispatcher<F>,
    m: usize, n: usize,
    alpha: *const f16,
    a: Array<f16>,
    x: Array<f16>,
    beta: *const f16,
    y: ArrayMut<f16>,
) {
    let x_ptr = x.data_ptr();
    let inc_x = x.rs();
    let y_ptr   = y.data_ptr();
    let incy = y.rs();
    let beta_val = (*beta).to_f32();
    let beta_t = &beta_val as *const f32;

    let alhpa_val = (*alpha).to_f32();
    let alpha_t = &alhpa_val as *const f32;
    // use compute_f32 until we have f16 axpy
    if hw_cfg.features.avx512f || (hw_cfg.features.avx && hw_cfg.features.fma) {
        avx_fma_microkernel::axpy(m, n, alpha_t, a.data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta_t, y_ptr, incy, hw_cfg.func);
        return;
    }
}


type F16Pack0 = PArray<f16>;

def_glare_gemm!(
    F16Dispatcher,
    f16,f16,f16,f16,f16,f16,f16,
    F16Pack0, F16Pack0,
    f16::ONE,
    glare_gemm_native, gemm_mt_native,
    gemm_goto_serial_native, kernel_native,
    gemm_small_m_serial_native, kernel_m_native,
    gemm_small_n_serial_native, kernel_n_native,
    glare_gemv_native, glare_gemv_native,
    packa_native, packb_native,
    true, true,
    into_pack_array, F,
);
