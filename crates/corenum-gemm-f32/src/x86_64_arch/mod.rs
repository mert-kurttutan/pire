pub(crate) mod avx_fma_microkernel;
pub(crate) mod avx512f_microkernel;

use avx_fma_microkernel::axpy;

use corenum_base::GemmArray;
use corenum_base::GemmOut;

const AVX_FMA_GOTO_MR: usize = 24; // register block size
const AVX_FMA_GOTO_NR: usize = 4; // register block size

const AVX512F_GOTO_MR: usize = 48; // register block size
const AVX512F_GOTO_NR: usize = 8; // register block size


const VS: usize = 8; // vector size in float, __m256

use std::marker::Sync;

use corenum_base::{
   GemmPackA, GemmPackB
};


use crate::{
   GemmGotoPackaPackb, GemmSmallM, GemmSmallN, Gemv, TA, TB, TC,
   GemmCache,
   MyFn, NullFn
};

pub(crate) enum X86_64Features {
    AvxFma,
    Avx512f,
}

pub(crate) struct X86_64dispatcher<
T: MyFn = NullFn
> {
    goto_mc: usize,
    goto_nc: usize,
    goto_kc: usize,
    goto_mr: usize,
    goto_nr: usize,
    is_l1_shared: bool,
    is_l2_shared: bool,
    is_l3_shared: bool,
    func: T,
    features: X86_64Features,
}

use corenum_base::HWConfig;

impl<F: MyFn> X86_64dispatcher<F> {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, features: X86_64Features, f: F) -> Self {
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let goto_mr = match features {
            X86_64Features::AvxFma => AVX_FMA_GOTO_MR,
            X86_64Features::Avx512f => AVX512F_GOTO_MR,
        };
        let goto_nr = match features {
            X86_64Features::AvxFma => AVX_FMA_GOTO_NR,
            X86_64Features::Avx512f => AVX512F_GOTO_NR,
        };
        Self {
            goto_mc: mc,
            goto_nc: nc,
            goto_kc: kc,
            is_l1_shared,
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
    unsafe fn packa_fn(self: &X86_64dispatcher<T>, a: *const TA, ap: *mut TA, m: usize, k: usize, a_rs: usize, a_cs: usize) {
        match self.features {
            X86_64Features::Avx512f => {
                avx512f_microkernel::packa_panel::<AVX512F_GOTO_MR>(m, k, a, a_rs, a_cs, ap);
            }
            X86_64Features::AvxFma => {
                avx_fma_microkernel::packa_panel::<AVX_FMA_GOTO_MR>(m, k, a, a_rs, a_cs, ap);
            }
        }
    }
}

impl<
T: MyFn
> GemmPackB<TA,TA> for X86_64dispatcher<T> {
    unsafe fn packb_fn(self: &X86_64dispatcher<T>, b: *const TA, bp: *mut TA, n: usize, k: usize, b_rs: usize, b_cs: usize) {
        match self.features {
            X86_64Features::Avx512f => {
                avx512f_microkernel::packb_panel::<AVX512F_GOTO_NR>(n, k, b, b_cs, b_rs, bp);
            }
            X86_64Features::AvxFma => {
                avx_fma_microkernel::packb_panel::<AVX_FMA_GOTO_NR>(n, k, b, b_cs, b_rs, bp);
            }
        }
    }
}


impl<
T: MyFn,
AP, BP,
A: GemmArray<AP>,
B: GemmArray<BP>,
> GemmCache<AP,BP,A,B> for X86_64dispatcher<T> {
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
       alpha: *const TA,
       a: A,
       x: B,
       beta: *const C::X,
       y: C,
   ) {
        let x_ptr = x.get_data_ptr();
        let inc_x = x.rs();
        let y_ptr   = y.data_ptr();
        let incy = y.rs();
        axpy(m, n, alpha, a.get_data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, self.func);
   }
}


impl<
A: GemmArray<TA,X=TA>, 
B: GemmArray<TB,X=TB>,
C: GemmOut<X=TC,Y=TC>,
F: MyFn + Sync,
> GemmGotoPackaPackb<TA,TB,A,B,C> for X86_64dispatcher<F>
{
   const ONE: TC = 1.0;
   unsafe fn kernel(
       self: &Self,
       m: usize, n: usize, k: usize,
       alpha: *const TA,
       beta: *const TC,
       c: *mut TC,
       c_rs: usize, c_cs: usize,
       ap: *const TA, bp: *const TB,
       _kc_last: bool
   ) {
        match self.features {
            X86_64Features::Avx512f => {
                avx512f_microkernel::kernel::<AVX512F_GOTO_MR, AVX512F_GOTO_NR, _>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, self.func)
            }
            X86_64Features::AvxFma => {
                avx_fma_microkernel::kernel::<AVX_FMA_GOTO_MR, AVX_FMA_GOTO_NR, _>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, self.func);
            }
        }
   }
}

impl<
A: GemmArray<TA,X=TA>, 
B: GemmArray<TB,X=TB>,
C: GemmOut<X=TC,Y=TC>,
F: MyFn + Sync,
> GemmSmallM<TA,TB,A,B,C> for X86_64dispatcher<F>
{
    const ONE: TC = 1.0;
   
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        b: *const TB, b_rs: usize, b_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        ap: *const TA,
   ) {
    match self.features {
        X86_64Features::Avx512f => {
            avx512f_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, self.func)
        }
        X86_64Features::AvxFma => {
            avx_fma_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, self.func);
        }
    }
   }
}



impl<
A: GemmArray<TA,X=TA>, 
B: GemmArray<TB,X=TB>,
C: GemmOut<X=TC,Y=TC>,
F: MyFn + Sync,
> GemmSmallN<TA,TB,A,B,C> for X86_64dispatcher<F>
{
    const ONE: TC = 1.0;
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        a: *const TA, a_rs: usize, a_cs: usize,
        ap: *mut TA,
        b: *const TB,
        c: *mut TC, c_rs: usize, c_cs: usize,
   ) {
        match self.features {
            X86_64Features::Avx512f => {
                avx512f_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, self.func)
            }
            X86_64Features::AvxFma => {
                avx_fma_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap, self.func);
            }
        }
   }
}
