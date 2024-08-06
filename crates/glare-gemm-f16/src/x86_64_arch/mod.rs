pub(crate) mod avx_fma_microkernel;
pub(crate) mod avx512f_microkernel;

use crate::f16;
// use avx_fma_microkernel::axpy;

use glare_base::GemmArray;

const AVX_FMA_GOTO_MR: usize = 24; // register block size
const AVX_FMA_GOTO_NR: usize = 4; // register block size

const AVX512F_GOTO_MR: usize = 48; // register block size
const AVX512F_GOTO_NR: usize = 8; // register block size


const VS: usize = 8; // vector size in float, __m256

use glare_base::{
    GemmPackA, GemmPackB, HWConfig,
   GemmOut, F32Features, CpuFeatures
};


use crate::{
   GemmGotoPackaPackb, GemmSmallM, GemmSmallN, Gemv, TA, TB, TC,
   GemmCache, NullFn, MyFn
};


pub(crate) enum F16Features {
    Avx512F16,
    Avx512BF16,
}

pub(crate) struct F32Dispatcher<
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
    features: CpuFeatures,
}
use glare_base::AccCoef;

impl<F: MyFn> AccCoef for F32Dispatcher<F> {
    type AS = f32;
    type BS = f32;
}

impl<F: MyFn> AccCoef for F16Dispatcher<F> {
    type AS = f16;
    type BS = f16;
}


impl F32Dispatcher {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, features: CpuFeatures) -> Self {
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let goto_mr = match features.f32_ft {
            F32Features::AvxFma => AVX_FMA_GOTO_MR,
            _ => AVX512F_GOTO_MR,
        };
        let goto_nr = match features.f32_ft {
            F32Features::AvxFma => AVX_FMA_GOTO_NR,
            _ => AVX512F_GOTO_NR,
        };
        Self {
            goto_mc: mc,
            goto_nc: nc,
            goto_kc: kc,
            is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            func: NullFn,
            features: features,
            goto_mr,
            goto_nr,
        }
    }
}

pub(crate) struct F16Dispatcher<
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
    features: F16Features,
}

impl F16Dispatcher {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, features: F16Features) -> Self {
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let goto_mr = AVX512F_GOTO_MR;
        let goto_nr = AVX512F_GOTO_NR;
        Self {
            goto_mc: mc,
            goto_nc: nc,
            goto_kc: kc,
            is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            func: NullFn,
            goto_mr,
            goto_nr,
            features: features,
        }
    }
}

impl<
T: MyFn,
AP, BP,
A: GemmArray<AP>,
B: GemmArray<BP>,
> GemmCache<AP,BP,A,B> for F32Dispatcher<T> {
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
T: MyFn,
AP, BP,
A: GemmArray<AP>,
B: GemmArray<BP>,
> GemmCache<AP,BP,A,B> for F16Dispatcher<T> {
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
T: MyFn
> GemmPackA<f16,f32> for F32Dispatcher<T> {
    unsafe fn packa_fn(self: &F32Dispatcher<T>, a: *const f16, ap: *mut f32, m: usize, k: usize, a_rs: usize, a_cs: usize) {
        match self.features.f32_ft {
            F32Features::AvxFma | F32Features::Avx => {
                if self.features.f16c {
                    avx_fma_microkernel::packa_panel::<AVX_FMA_GOTO_MR>(m, k, a, a_rs, a_cs, ap);
                } else {
                    // avx_fma_microkernel::packa_panel::<AVX_FMA_GOTO_MR>(m, k, a, a_rs, a_cs, ap);
                }
            }
            F32Features::Avx512F => {
                avx512f_microkernel::packa_panel::<AVX512F_GOTO_MR>(m, k, a, a_rs, a_cs, ap);
            }
            _ => {
                // avx512f_microkernel::packa_panel::<AVX512F_GOTO_MR>(m, k, a, a_rs, a_cs, ap);
            }
        }
    }
}
impl<
T: MyFn
> GemmPackB<f16,f32> for F32Dispatcher<T> {
    unsafe fn packb_fn(self: &F32Dispatcher<T>, b: *const f16, bp: *mut f32, n: usize, k: usize, b_rs: usize, b_cs: usize) {
        // packb_panel::<GOTO_NR>(n, k, b, b_cs, b_rs, bp);
        match self.features.f32_ft {
            F32Features::AvxFma | F32Features::Avx => {
                if self.features.f16c {
                    avx_fma_microkernel::packb_panel::<AVX_FMA_GOTO_NR>(n, k, b, b_cs, b_rs, bp);
                } else {
                    //
                }
            }
            F32Features::Avx512F => {
                avx512f_microkernel::packb_panel::<AVX512F_GOTO_NR>(n, k, b, b_cs, b_rs, bp);
            }
            _ => {
                // avx512f_microkernel::packb_panel::<AVX512F_GOTO_NR>(n, k, b, b_cs, b_rs, bp);
            }

        }
    }
}

impl<
T: MyFn
> GemmPackA<f16,f16> for F16Dispatcher<T> {
    #[allow(unused_variables)]
    unsafe fn packa_fn(self: &F16Dispatcher<T>, a: *const f16, ap: *mut f16, m: usize, k: usize, a_rs: usize, a_cs: usize) {
    }
}

impl<
T: MyFn
> GemmPackB<f16,f16> for F16Dispatcher<T> {
    #[allow(unused_variables)]
    unsafe fn packb_fn(self: &F16Dispatcher<T>, b: *const f16, bp: *mut f16, n: usize, k: usize, b_rs: usize, b_cs: usize) {
    }
}


impl<
T: MyFn,
A: GemmArray<f32, X=f16>, 
B: GemmArray<f32, X=f16>,
C: GemmOut<X=f16,Y=f16>,
> Gemv<f32,f32,A,B,C> for F32Dispatcher<T>
{
    #[allow(unused_variables)]
   unsafe fn gemv_serial(
         self: &Self,
       m: usize, n: usize,
       alpha: *const f32,
       a: A,
       x: B,
       beta: *const Self::BS,
       y: C,
   ) {
        let x_ptr = x.get_data_ptr();
        let inc_x = x.rs();
        let y_ptr   = y.data_ptr();
        let incy = y.rs();
        // let beta_val = *beta;
        // let beta_t = beta_val.to_f32();
        // let beta = &beta_t as *const f32;
        match self.features.f32_ft {
            F32Features::Avx512F => {
                avx_fma_microkernel::axpy(m, n, alpha, a.get_data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, self.func);
            }
            F32Features::AvxFma => {
                avx_fma_microkernel::axpy(m, n, alpha, a.get_data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy, self.func);
            }
            _ => { panic!("Unsupported feature set for this kernel") }
        }
    }
}

impl<
T: MyFn,
A: GemmArray<f16, X=f16>, 
B: GemmArray<f16, X=f16>,
C: GemmOut<X=f16,Y=f16>,
> Gemv<TA,TB,A,B,C> for F16Dispatcher<T>
{
    #[allow(unused_variables)]
   unsafe fn gemv_serial(
    self: &Self,
       m: usize, n: usize,
       alpha: *const TA,
       a: A,
       x: B,
       beta: *const C::X,
       y: C,
   ) {
        // let x_ptr = x.get_data_ptr();
        // let inc_x = x.rs();
        // let y_ptr   = y.data_ptr();
        // let incy = y.rs();
        // axpy(m, n, alpha, a.get_data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy);
   }
}

impl<
T: MyFn,
A: GemmArray<f32,X=f16>,
B: GemmArray<f32,X=f16>,
C: GemmOut<X=f16,Y=f16>,
> GemmGotoPackaPackb<f32,f32,A,B,C> for F32Dispatcher<T>
where 
F32Dispatcher<T>: GemmPackA<f16, f32> + GemmPackB<f16, f32> 
{
   const ONE: f32 = 1_f32;
   unsafe fn kernel(
       self: &Self,
       m: usize, n: usize, k: usize,
       alpha: *const f32,
       beta: *const f32,
       c: *mut TC,
       c_rs: usize, c_cs: usize,
       ap: *const f32, bp: *const f32,
       _kc_last: bool
   ) {
        let my_func = self.func;
        // let beta_val = *beta;
        // let beta_t = beta_val.to_f32();
        // let beta = &beta_t as *const f32;
        match self.features.f32_ft {
            F32Features::AvxFma => {
                avx_fma_microkernel::kernel::<AVX_FMA_GOTO_MR, AVX_FMA_GOTO_NR, _>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, my_func);
            }
            F32Features::Avx512F => {
                avx512f_microkernel::kernel::<AVX512F_GOTO_MR, AVX512F_GOTO_NR, _>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, my_func)
            }
            _ => { panic!("Unsupported feature set for this kernel") }
        }
   }
}


impl<
T: MyFn,
A: GemmArray<f16, X=f16>, 
B: GemmArray<f16, X=f16>,
C: GemmOut<X=f16,Y=f16>,
> GemmGotoPackaPackb<f16,f16,A,B,C> for F16Dispatcher<T>
where 
F16Dispatcher<T>: GemmPackA<f16, f16> + GemmPackB<f16, f16> 
{
   const ONE: TC = f16::ONE;
   #[allow(unused_variables)]
   unsafe fn kernel(
       self: &Self,
       m: usize, n: usize, k: usize,
       alpha: *const f16,
       beta: *const TC,
       c: *mut TC,
       c_rs: usize, c_cs: usize,
       ap: *const f16, bp: *const f16,
       _kc_last: bool
   ) {
   }
}



impl<
T: MyFn,
A: GemmArray<f32, X = f16>, 
B: GemmArray<f32, X = f16>,
C: GemmOut<X=f16,Y=f16>,
> GemmSmallM<f32,f32,A,B,C> for F32Dispatcher<T>
where F32Dispatcher<T>: GemmPackA<f16, f32>
{
    const ONE: f32 = 1_f32;
    #[allow(unused_variables)]
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const f32,
        beta: *const f32,
        b: *const f16, b_rs: usize, b_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        ap: *const f32,
   ) {
   }
}



impl<
T: MyFn,
A: GemmArray<f32, X = f16>, 
B: GemmArray<f32, X = f16>,
C: GemmOut<X=f16,Y=f16>,
> GemmSmallN<f32,f32,A,B,C> for F32Dispatcher<T>
where 
F32Dispatcher<T>: GemmPackB<f16, f32>
{
    const ONE: f32 = 1_f32;
    #[allow(unused_variables)]
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const f32,
        beta: *const f32,
        a: *const f16, a_rs: usize, a_cs: usize,
        ap: *mut f32,
        b: *const f32,
        c: *mut TC, c_rs: usize, c_cs: usize,
   ) {
   }
}





impl<
T: MyFn,
A: GemmArray<f16, X = f16>, 
B: GemmArray<f16, X = f16>,
C: GemmOut<X=f16,Y=f16>,
> GemmSmallM<TA,TB,A,B,C> for F16Dispatcher<T>
where F16Dispatcher<T>: GemmPackA<A::X, TA>
{
    const ONE: TC = f16::ONE;
    #[allow(unused_variables)]
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const f16,
        beta: *const f16,
        b: *const f16, b_rs: usize, b_cs: usize,
        c: *mut f16, c_rs: usize, c_cs: usize,
        ap: *const f16,
   ) {
   }
}



impl<
T: MyFn,
A: GemmArray<f16, X = f16>, 
B: GemmArray<f16>,
C: GemmOut<X=f16,Y=f16>,
> GemmSmallN<TA,TB,A,B,C> for F16Dispatcher<T>
where 
F16Dispatcher<T>: GemmPackB<B::X, TB>
{
    const ONE: TC = f16::ONE;
    #[allow(unused_variables)]
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const f16,
        beta: *const f16,
        a: *const f16, a_rs: usize, a_cs: usize,
        ap: *mut f16,
        b: *const f16,
        c: *mut TC, c_rs: usize, c_cs: usize,
   ) {
   }
}
