pub(crate) mod avx_fma_microkernel;
pub(crate) mod avx512f_microkernel;

use corenum_base::GemmArrayP;
use corenum_base::PackedMatrix;
use avx_fma_microkernel::axpy;

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

pub struct NullFn;

pub trait MyFn{}

impl MyFn for NullFn{}

impl MyFn for fn(*mut f32, m: usize){}

pub enum x86Backend {
    AvxFma,
    Avx512f,
}

pub struct x86_64<
T: MyFn = NullFn
> {
    pub goto_mc: usize,
    pub goto_nc: usize,
    pub goto_kc: usize,
    pub is_l1_shared: bool,
    pub is_l2_shared: bool,
    pub is_l3_shared: bool,
    pub func: T,
    pub backend: x86Backend,
}

use corenum_base::HWConfig;

impl x86_64 {
    pub fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, backend: x86Backend) -> Self {
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        Self {
            goto_mc: mc,
            goto_nc: nc,
            goto_kc: kc,
            is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            func: NullFn,
            backend: backend,
        }
    }
}


impl x86_64<fn(*mut f32, usize)> {
    pub fn from_hw_cfg_func(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, func: fn(*mut f32, m: usize)) -> Self {
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        Self {
            goto_mc: mc,
            goto_nc: nc,
            goto_kc: kc,
            is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            func: func,
            backend: x86Backend::AvxFma,
        }
    }
}

impl<
T: MyFn
> GemmPackA<TA,TA> for x86_64<T> {
    unsafe fn packa_fn(self: &x86_64<T>, a: *const TA, ap: *mut TA, m: usize, k: usize, a_rs: usize, a_cs: usize) {
        match self.backend {
            x86Backend::Avx512f => {
                avx512f_microkernel::packa_panel::<AVX512F_GOTO_MR>(m, k, a, a_rs, a_cs, ap);
            }
            x86Backend::AvxFma => {
                avx_fma_microkernel::packa_panel::<AVX_FMA_GOTO_MR>(m, k, a, a_rs, a_cs, ap);
            }
        }
    }
}

impl<
T: MyFn
> GemmPackB<TA,TA> for x86_64<T> {
    unsafe fn packb_fn(self: &x86_64<T>, b: *const TA, bp: *mut TA, n: usize, k: usize, b_rs: usize, b_cs: usize) {
        // packb_panel::<GOTO_NR>(n, k, b, b_cs, b_rs, bp);
        match self.backend {
            x86Backend::Avx512f => {
                avx512f_microkernel::packb_panel::<AVX512F_GOTO_NR>(n, k, b, b_cs, b_rs, bp);
            }
            x86Backend::AvxFma => {
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
> GemmCache<AP,BP,A,B> for x86_64<T> {
    // const CACHELINE_PAD: usize = 256;
    fn mr(&self) -> usize {
        match self.backend {
            x86Backend::Avx512f => AVX512F_GOTO_MR,
            x86Backend::AvxFma => AVX_FMA_GOTO_MR,
        }
    }
    fn nr(&self) -> usize {
        match self.backend {
            x86Backend::Avx512f => AVX512F_GOTO_NR,
            x86Backend::AvxFma => AVX_FMA_GOTO_NR,
        }
    }
    fn get_kc_eff(&self) -> usize {self.goto_kc}
    fn get_mc_eff(&self, par: usize) -> usize {
        if self.is_l3_shared {
            self.goto_mc / par
        } else {
            self.goto_mc
        }
    }
    fn get_nc_eff(&self, par: usize) -> usize {
        if self.is_l2_shared {
            self.goto_nc / par
        } else {
            self.goto_nc
        }
    }
}

impl<
A: GemmArray<f32, X=f32>, 
B: GemmArray<f32, X=f32>,
C: GemmOut<X=f32,Y=f32>,
> Gemv<TA,TB,A,B,C> for x86_64<NullFn>
{
   unsafe fn gemv_serial(
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
        axpy(m, n, alpha, a.get_data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy);
   }
}



use corenum_base::GemmOut;

impl<
A: GemmArray<f32>, 
B: GemmArray<f32>,
C: GemmOut<X=f32,Y=f32>,
> GemmGotoPackaPackb<TA,TB,A,B,C> for x86_64<NullFn>
where 
x86_64<NullFn>: GemmPackA<A::X, TA> + GemmPackB<B::X, TB>
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
        match self.backend {
            x86Backend::Avx512f => {
                avx512f_microkernel::kernel::<AVX512F_GOTO_MR, AVX512F_GOTO_NR>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp)
            }
            x86Backend::AvxFma => {
                avx_fma_microkernel::kernel::<AVX_FMA_GOTO_MR, AVX_FMA_GOTO_NR>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp);
            }
        }
   }
}


impl<
A: GemmArray<f32>, 
B: GemmArray<f32, X = f32>,
C: GemmOut<X=f32,Y=f32>,
> GemmSmallM<TA,TB,A,B,C> for x86_64<NullFn>
where x86_64<NullFn>: GemmPackA<A::X, TA>
{
    const ONE: TC = 1.0;
   
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        b: *const f32, b_rs: usize, b_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        ap: *const TA,
   ) {
    match self.backend {
        x86Backend::Avx512f => {
            avx512f_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap)
        }
        x86Backend::AvxFma => {
            avx_fma_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap);
        }
    }
   }
}



impl<
A: GemmArray<f32, X = f32>, 
B: GemmArray<f32>,
C: GemmOut<X=f32,Y=f32>,
> GemmSmallN<TA,TB,A,B,C> for x86_64<NullFn>
where 
x86_64<NullFn>: GemmPackB<B::X, TB>
{
    const ONE: TC = 1.0;
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        a: *const f32, a_rs: usize, a_cs: usize,
        ap: *mut f32,
        b: *const f32,
        c: *mut TC, c_rs: usize, c_cs: usize,
   ) {
        match self.backend {
            x86Backend::Avx512f => {
                avx512f_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap)
            }
            x86Backend::AvxFma => {
                avx_fma_microkernel::kernel_sb(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap);
            }
        }
   }
}

impl<
A: GemmArray<f32, X=f32>, 
B: GemmArray<f32, X=f32>,
C: GemmOut<X=f32,Y=f32>,
> Gemv<TA,TB,A,B,C> for x86_64<UFn>
{
    
   unsafe fn gemv_serial(
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
        axpy(m, n, alpha, a.get_data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy);
   }
}

type UFn = fn(*mut f32, usize);

impl<
A: GemmArray<f32>, 
B: GemmArray<f32>,
C: GemmOut<X=f32,Y=f32>,
> GemmGotoPackaPackb<TA,TB,A,B,C> for x86_64<UFn>
where 
x86_64<UFn>: GemmPackA<A::X, TA> + GemmPackB<B::X, TB>
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
       kc_last: bool
   ) {
        if kc_last {
            // kernel_fuse::<GOTO_MR, GOTO_NR>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, self.func);
        } else {
            avx_fma_microkernel::kernel::<AVX_FMA_GOTO_MR, AVX_FMA_GOTO_NR>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp);
        }
   }

}


impl<
A: GemmArray<f32>, 
B: GemmArray<f32, X = f32>,
C: GemmOut<X=f32,Y=f32>,
> GemmSmallM<TA,TB,A,B,C> for x86_64<UFn>
where x86_64<UFn>: GemmPackA<A::X, TA>
{
    const ONE: TC = 1.0;
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        b: *const f32, b_rs: usize, b_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        ap: *const TA,
   ) {
    avx_fma_microkernel::kernel_bs(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap);
   }
}



impl<
A: GemmArray<f32, X = f32>, 
B: GemmArray<f32>,
C: GemmOut<X=f32,Y=f32>,
> GemmSmallN<TA,TB,A,B,C> for x86_64<UFn>
where 
x86_64<UFn>: GemmPackB<B::X, TB>
{
    const ONE: TC = 1.0;
   unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        a: *const f32, a_rs: usize, a_cs: usize,
        ap: *mut f32,
        b: *const f32,
        c: *mut TC, c_rs: usize, c_cs: usize,
   ) {
    avx_fma_microkernel::kernel_sb(
            m, n, k, alpha, beta, 
            a, a_rs, a_cs, 
            b,
            c, c_rs, c_cs,
            ap
        );
   }
}
