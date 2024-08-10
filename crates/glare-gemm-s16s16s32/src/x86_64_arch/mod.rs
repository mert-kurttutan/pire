pub(crate) mod avx_fma_microkernel;
// pub(crate) mod avx512f_microkernel;
pub(crate) mod pack_avx;

use glare_base::GemmArray;
use glare_base::GemmOut;
use glare_base::CpuFeatures;

const AVX_FMA_GOTO_MR: usize = 16; // register block size
const AVX_FMA_GOTO_NR: usize = 4; // register block size

const AVX512F_GOTO_MR: usize = 48; // register block size
const AVX512F_GOTO_NR: usize = 8; // register block size


const VS: usize = 8; // vector size in float, __m256

use std::marker::Sync;

use glare_base::{
   GemmPackA, GemmPackB
};


use crate::{
   GemmGotoPackaPackb, GemmSmallM, GemmSmallN, Gemv, TA, TB, TC,
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

use glare_base::HWConfig;

impl<F: MyFn> X86_64dispatcher<F> {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, features: CpuFeatures, f: F) -> Self {
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (goto_mr, goto_nr) = if features.avx512f {
            (AVX512F_GOTO_MR, AVX512F_GOTO_NR)
        } else if features.avx && features.fma {
            (AVX_FMA_GOTO_MR, AVX_FMA_GOTO_NR)
        } else {
            (AVX_FMA_GOTO_MR, AVX_FMA_GOTO_NR)
        };
        Self {
            goto_mc: mc,
            goto_nc: nc,
            goto_kc: kc,
            // is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            func: f,
            features: features,
            goto_mr,
            goto_nr,
        }
    }
}

impl<
T: MyFn
> GemmPackA<TA,TA> for X86_64dispatcher<T> {
    unsafe fn packa_fn(self: &X86_64dispatcher<T>, x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
        if self.features.avx2{
            pack_avx::packa_panel_16(m, k, x, rs, cs, y);
            return;
        }
    }
}

impl<
T: MyFn
> GemmPackB<TA,TA> for X86_64dispatcher<T> {
    unsafe fn packb_fn(self: &X86_64dispatcher<T>, x: *const TA, y: *mut TA, n: usize, k: usize, rs: usize, cs: usize) {
        pack_avx::packb_panel_4(n, k, x, cs, rs, y);
    }
}


impl<
T: MyFn,
AP, BP,
A: GemmArray<AP>,
B: GemmArray<BP>,
> GemmCache<AP,BP,A,B> for X86_64dispatcher<T> {
    const IS_EFFICIENT: bool = false;
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

use glare_base::AccCoef;

impl<F: MyFn> AccCoef for X86_64dispatcher<F> {
    type AS = f32;
    type BS = f32;
}

impl<
A: GemmArray<TA,X=TA>, 
B: GemmArray<TB,X=TB>,
C: GemmOut<X=TC,Y=TC>,
F: MyFn + Sync,
> Gemv<TA,TB,A,B,C> for X86_64dispatcher<F>
{
   unsafe fn gemv_serial(
    self: &Self,
       m: usize, n: usize,
       alpha: *const f32,
       a: A,
       x: B,
       beta: *const f32,
       y: C,
   ) {
        let x_ptr = x.get_data_ptr();
        let inc_x = x.rs();
        let y_ptr   = y.data_ptr();
        let incy = y.rs();
        avx_fma_microkernel::axpy(m, n, alpha, a.get_data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, self.func);
   }
}


impl<
A: GemmArray<TA,X=TA>, 
B: GemmArray<TB,X=TB>,
C: GemmOut<X=TC,Y=TC>,
F: MyFn + Sync,
> GemmGotoPackaPackb<TA,TB,A,B,C> for X86_64dispatcher<F>
{
   const ONE: f32 = 1_f32;
   unsafe fn kernel(
       self: &Self,
       m: usize, n: usize, k: usize,
       alpha: *const f32,
       beta: *const f32,
       c: *mut TC,
       c_rs: usize, c_cs: usize,
       ap: *const TA, bp: *const TB,
       _kc_last: bool
   ) {
        avx_fma_microkernel::kernel(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, self.func);
   }
}

impl<
A: GemmArray<TA,X=TA>, 
B: GemmArray<TB,X=TB>,
C: GemmOut<X=TC,Y=TC>,
F: MyFn + Sync,
> GemmSmallM<TA,TB,A,B,C> for X86_64dispatcher<F>
{
    const ONE: f32 = 1_f32;
   #[allow(unused_variables)]
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const f32,
        beta: *const f32,
        b: *const TB, b_rs: usize, b_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        ap: *const TA,
   ) {
    panic!("This should not run since goto is more effiicent on s16s16s32 gemm");
   }
}



impl<
A: GemmArray<TA,X=TA>, 
B: GemmArray<TB,X=TB>,
C: GemmOut<X=TC,Y=TC>,
F: MyFn + Sync,
> GemmSmallN<TA,TB,A,B,C> for X86_64dispatcher<F>
{
    const ONE: f32 = 1.0;
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const f32,
        beta: *const f32,
        a: *const TA, a_rs: usize, a_cs: usize,
        ap: *mut TA,
        b: *const TB,
        c: *mut TC, c_rs: usize, c_cs: usize,
   ) {
        avx_fma_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, self.func);
   }
}
