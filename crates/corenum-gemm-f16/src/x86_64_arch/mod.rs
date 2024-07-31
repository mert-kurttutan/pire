// pub(crate) mod avx_fma_microkernel;
// pub(crate) mod avx512f_microkernel;

use corenum_base::GemmArrayP;
use corenum_base::PackedMatrix;
// use avx_fma_microkernel::axpy;

use corenum_base::GemmArray;

const AVX_FMA_GOTO_MR: usize = 24; // register block size
const AVX_FMA_GOTO_NR: usize = 4; // register block size

const AVX512F_GOTO_MR: usize = 48; // register block size
const AVX512F_GOTO_NR: usize = 8; // register block size


const VS: usize = 8; // vector size in float, __m256

use corenum_base::{
   StridedMatrix, GemmPackA, GemmPackB
};


use crate::{
   GemmGotoPackaPackb, GemmSmallM, GemmSmallN, Gemv, TA, TB, TC,
   GemmCache,
};

pub(crate) struct NullFn;

pub(crate) trait MyFn{}

impl MyFn for NullFn{}

impl MyFn for fn(*mut f32, m: usize){}

pub(crate) enum AvxFeatures {
    AvxFma,
    Avx,
}

pub(crate) enum Avx512Features {
    Avx512F16,
    Avx512BF16,
}

pub(crate) struct AvxDispatcher<
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
    features: AvxFeatures,
}

pub(crate) struct Avx512Dispatcher<
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
    features: Avx512Features,
}

impl Avx512Dispatcher {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, features: Avx512Features) -> Self {
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

use corenum_base::HWConfig;

impl AvxDispatcher {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, features: AvxFeatures) -> Self {
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let goto_mr = match features {
            AvxFeatures::AvxFma => AVX_FMA_GOTO_MR,
            AvxFeatures::Avx => AVX512F_GOTO_MR,
        };
        let goto_nr = match features {
            AvxFeatures::AvxFma => AVX_FMA_GOTO_NR,
            AvxFeatures::Avx => AVX512F_GOTO_NR,
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

impl<
T: MyFn
> GemmPackA<u16,f32> for AvxDispatcher<T> {
    unsafe fn packa_fn(self: &AvxDispatcher<T>, a: *const u16, ap: *mut f32, m: usize, k: usize, a_rs: usize, a_cs: usize) {
        match self.features {
            AvxFeatures::Avx => {
                // avx512f_microkernel::packa_panel::<AVX512F_GOTO_MR>(m, k, a, a_rs, a_cs, ap);
            }
            AvxFeatures::AvxFma => {
                // avx_fma_microkernel::packa_panel::<AVX_FMA_GOTO_MR>(m, k, a, a_rs, a_cs, ap);
            }
        }
    }
}

impl<
T: MyFn
> GemmPackA<u16,u16> for Avx512Dispatcher<T> {
    unsafe fn packa_fn(self: &Avx512Dispatcher<T>, a: *const u16, ap: *mut u16, m: usize, k: usize, a_rs: usize, a_cs: usize) {
    }
}

impl<
T: MyFn
> GemmPackB<u16,f32> for AvxDispatcher<T> {
    unsafe fn packb_fn(self: &AvxDispatcher<T>, b: *const u16, bp: *mut f32, n: usize, k: usize, b_rs: usize, b_cs: usize) {
        // packb_panel::<GOTO_NR>(n, k, b, b_cs, b_rs, bp);
        match self.features {
            AvxFeatures::Avx => {
                // avx512f_microkernel::packb_panel::<AVX512F_GOTO_NR>(n, k, b, b_cs, b_rs, bp);
            }
            AvxFeatures::AvxFma => {
                // avx_fma_microkernel::packb_panel::<AVX_FMA_GOTO_NR>(n, k, b, b_cs, b_rs, bp);
            }
        }
    }
}

impl<
T: MyFn
> GemmPackB<u16,u16> for Avx512Dispatcher<T> {
    unsafe fn packb_fn(self: &Avx512Dispatcher<T>, b: *const u16, bp: *mut u16, n: usize, k: usize, b_rs: usize, b_cs: usize) {
    }
}

impl<
T: MyFn,
AP, BP,
A: GemmArray<AP>,
B: GemmArray<BP>,
> GemmCache<AP,BP,A,B> for AvxDispatcher<T> {
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
> GemmCache<AP,BP,A,B> for Avx512Dispatcher<T> {
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
A: GemmArray<f32, X=u16>, 
B: GemmArray<f32, X=u16>,
C: GemmOut<X=u16,Y=u16>,
> Gemv<f32,f32,A,B,C> for AvxDispatcher<NullFn>
{
   unsafe fn gemv_serial(
         self: &Self,
       m: usize, n: usize,
       alpha: *const f32,
       a: A,
       x: B,
       beta: *const C::X,
       y: C,
   ) {
        let x_ptr = x.get_data_ptr();
        let inc_x = x.rs();
        let y_ptr   = y.data_ptr();
        let incy = y.rs();
        // axpy(m, n, alpha, a.get_data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy);
   }
}

impl<
A: GemmArray<u16, X=u16>, 
B: GemmArray<u16, X=u16>,
C: GemmOut<X=u16,Y=u16>,
> Gemv<TA,TB,A,B,C> for Avx512Dispatcher<NullFn>
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
        // axpy(m, n, alpha, a.get_data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy);
   }
}


use corenum_base::GemmOut;

impl<
A: GemmArray<f32, X=u16>, 
B: GemmArray<f32, X=u16>,
C: GemmOut<X=u16,Y=u16>,
> GemmGotoPackaPackb<f32,f32,A,B,C> for AvxDispatcher<NullFn>
where 
AvxDispatcher<NullFn>: GemmPackA<u16, f32> + GemmPackB<u16, f32> 
{
   const ONE: TC = 1_u16;
   unsafe fn kernel(
       self: &Self,
       m: usize, n: usize, k: usize,
       alpha: *const f32,
       beta: *const TC,
       c: *mut TC,
       c_rs: usize, c_cs: usize,
       ap: *const f32, bp: *const f32,
       _kc_last: bool
   ) {
        match self.features {
            AvxFeatures::Avx => {
                // avx512f_microkernel::kernel::<AVX512F_GOTO_MR, AVX512F_GOTO_NR>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp)
            }
            AvxFeatures::AvxFma => {
                // avx_fma_microkernel::kernel::<AVX_FMA_GOTO_MR, AVX_FMA_GOTO_NR>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp);
            }
        }
   }
}


impl<
A: GemmArray<u16, X=u16>, 
B: GemmArray<u16, X=u16>,
C: GemmOut<X=u16,Y=u16>,
> GemmGotoPackaPackb<u16,u16,A,B,C> for Avx512Dispatcher<NullFn>
where 
Avx512Dispatcher<NullFn>: GemmPackA<u16, u16> + GemmPackB<u16, u16> 
{
   const ONE: TC = 1_u16;
   unsafe fn kernel(
       self: &Self,
       m: usize, n: usize, k: usize,
       alpha: *const u16,
       beta: *const TC,
       c: *mut TC,
       c_rs: usize, c_cs: usize,
       ap: *const u16, bp: *const u16,
       _kc_last: bool
   ) {
   }
}



impl<
A: GemmArray<f32, X = u16>, 
B: GemmArray<f32, X = u16>,
C: GemmOut<X=u16,Y=u16>,
> GemmSmallM<f32,f32,A,B,C> for AvxDispatcher<NullFn>
where AvxDispatcher<NullFn>: GemmPackA<u16, f32>
{
    const ONE: TC = 1_u16;
   
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const f32,
        beta: *const TC,
        b: *const u16, b_rs: usize, b_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        ap: *const f32,
   ) {
    match self.features {
        AvxFeatures::Avx => {
            // avx512f_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap)
        }
        AvxFeatures::AvxFma => {
            // avx_fma_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap);
        }
    }
   }
}



impl<
A: GemmArray<f32, X = u16>, 
B: GemmArray<f32, X = u16>,
C: GemmOut<X=u16,Y=u16>,
> GemmSmallN<f32,f32,A,B,C> for AvxDispatcher<NullFn>
where 
AvxDispatcher<NullFn>: GemmPackB<u16, f32>
{
    const ONE: TC = 1_u16;
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const f32,
        beta: *const TC,
        a: *const u16, a_rs: usize, a_cs: usize,
        ap: *mut f32,
        b: *const f32,
        c: *mut TC, c_rs: usize, c_cs: usize,
   ) {
        match self.features {
            AvxFeatures::Avx => {
                // avx512f_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap)
            }
            AvxFeatures::AvxFma => {
                // avx_fma_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap);
            }
        }
   }
}





impl<
A: GemmArray<u16, X = u16>, 
B: GemmArray<u16, X = u16>,
C: GemmOut<X=u16,Y=u16>,
> GemmSmallM<TA,TB,A,B,C> for Avx512Dispatcher<NullFn>
where Avx512Dispatcher<NullFn>: GemmPackA<A::X, TA>
{
    const ONE: TC = 1_u16;
   
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const u16,
        beta: *const u16,
        b: *const u16, b_rs: usize, b_cs: usize,
        c: *mut u16, c_rs: usize, c_cs: usize,
        ap: *const u16,
   ) {
   }
}



impl<
A: GemmArray<u16, X = u16>, 
B: GemmArray<u16>,
C: GemmOut<X=u16,Y=u16>,
> GemmSmallN<TA,TB,A,B,C> for Avx512Dispatcher<NullFn>
where 
Avx512Dispatcher<NullFn>: GemmPackB<B::X, TB>
{
    const ONE: TC = 1_u16;
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const u16,
        beta: *const u16,
        a: *const u16, a_rs: usize, a_cs: usize,
        ap: *mut u16,
        b: *const u16,
        c: *mut TC, c_rs: usize, c_cs: usize,
   ) {
   }
}
