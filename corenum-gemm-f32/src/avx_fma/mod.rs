
pub(crate) mod microkernel;


use corenum_base::GemmArrayP;
use corenum_base::PackedMatrix;
use corenum_base::StridedMatrixMut;
pub(crate) use microkernel::{
   pack_panel,
   kernel,
   kernel_sup_m,
   kernel_sup_n,
   axpy,
};


const VS: usize = 8; // vector size in float, __m256

use corenum_base::{
   StridedMatrix, GemmPack
};


use crate::{
   GemmGotoPackaPackb, GemmSmallM, GemmSmallN, Gemv, TA, TB, TC,
   GemmCache,
};


// pub const GOTO_MC: usize = env_or!("CORENUM_SGEMM_MC", 4800);
// // could be either 320 or 320 based experiments on comet lake local machine
// pub const GOTO_NC: usize = env_or!("CORENUM_SGEMM_NC", 320);
// // KC should be nice multiple of 4 (multiple of 64 is OK)
// // on comet lake it gives optimal perf when KC == 192 or 256
// pub const GOTO_KC: usize = env_or!("CORENUM_SGEMM_KC", 192);
// pub const GOTO_MR: usize = env_or!("CORENUM_SGEMM_MR", 24);
// pub const GOTO_NR: usize = env_or!("CORENUM_SGEMM_NR", 4);


// 24x4 is better than 16x6 for haswell/coffe lake etc


// pub const SA_NC: usize = env_or!("CORENUM_SGEMM_NC_SA", 168);
// pub const SA_KC: usize = env_or!("CORENUM_SGEMM_KC_SA", 256);
// pub const SA_MR: usize = env_or!("CORENUM_SGEMM_MR_SA", 16);
// pub const SA_NR: usize = env_or!("CORENUM_SGEMM_NR_SA", 6);

// pub const SN_MC: usize = env_or!("CORENUM_SGEMM_NC_SA", 4800);
// pub const SN_KC: usize = env_or!("CORENUM_SGEMM_KC_SA", 192);
// pub const SN_MR: usize = env_or!("CORENUM_SGEMM_MR_SA", 24);
// pub const SN_NR: usize = env_or!("CORENUM_SGEMM_NR_SA", 4);

pub struct AvxFma<
const GOTO_MR: usize,
const GOTO_NR: usize,
> {
    pub goto_mc: usize,
    pub goto_nc: usize,
    pub goto_kc: usize,
    pub is_l1_shared: bool,
    pub is_l2_shared: bool,
    pub is_l3_shared: bool,
}



impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
A: GemmArray<f32, X=f32> + Axpy, 
B: GemmArray<f32, X=f32>,
C: GemmOut<X=f32,Y=f32>,
> Gemv<TA,TB,A,B,C> for AvxFma<GOTO_MR,GOTO_NR>
{
    #[target_feature(enable = "avx,fma")]
   unsafe fn gemv_serial(
       m: usize, n: usize,
       alpha: *const TA,
       a: A,
       x: B,
       beta: *const C::X,
       y: C,
   ) {
        let x_ptr = x.get_data_ptr();
        let inc_x = x.get_rs();
        let y_ptr   = y.data_ptr();
        let incy = y.rs();
        A::axpy(m, n, alpha, a, x_ptr, inc_x, beta, y_ptr, incy)

   }
}



// func signature rule
// 1 -> m, n, k params if exists
// 2 -> pointer a and lda
// 3 -> pointer b and ldb
// 4 -> pointer c and ldc
// 5 -> pointer packed a and b
// 6 -> m related blocking params
// 7 -> n related blocking params


impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
> GemmPack<TA,TA> for AvxFma<GOTO_MR,GOTO_NR> {
    #[target_feature(enable = "avx,fma")]
    unsafe fn packa_fn(a: *const TA, ap: *mut TA, m: usize, k: usize, a_rs: usize, a_cs: usize) {
        pack_panel::<GOTO_MR>(m, k, a, a_rs, a_cs, ap);
    }

    #[target_feature(enable = "avx,fma")]
    unsafe fn packb_fn(b: *const TA, bp: *mut TA, n: usize, k: usize, b_rs: usize, b_cs: usize) {
        pack_panel::<GOTO_NR>(n, k, b, b_rs, b_cs, bp);
    }
}

impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
> GemmPack<u16,TA> for AvxFma<GOTO_MR,GOTO_NR> {
    #[target_feature(enable = "avx,fma")]
    unsafe fn packa_fn(a: *const u16, ap: *mut TA, m: usize, k: usize, a_rs: usize, a_cs: usize) {
        // pack_panel::<GOTO_MR>(m, k, a, a_rs, a_cs, ap);
    }

    #[target_feature(enable = "avx,fma")]
    unsafe fn packb_fn(b: *const u16, bp: *mut TA, n: usize, k: usize, b_rs: usize, b_cs: usize) {
        // pack_panel::<GOTO_NR>(n, k, b, b_rs, b_cs, bp);
    }
}

impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
AP, BP,
A: GemmArray<AP>,
B: GemmArray<BP>,
> GemmCache<AP,BP,A,B> for AvxFma<GOTO_MR,GOTO_NR> {
    const CACHELINE_PAD: usize = 256;
    const MR: usize = GOTO_MR;
    const NR: usize = GOTO_NR;
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




pub struct Identity {}

use corenum_base::UnaryOp;

impl UnaryOp<f32,f32> for Identity {
    const IS_IDENTITY: bool = true;
    #[inline(always)]
    unsafe fn apply_inplace(_x: *mut f32) {
        
    }

    #[inline(always)]
    unsafe fn map(_x: *const f32, _y: *mut f32) {
        
    }
}

use corenum_base::GemmOut;

impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
A: GemmArray<f32>, 
B: GemmArray<f32>,
C: GemmOut<X=f32,Y=f32>,
> GemmGotoPackaPackb<TA,TB,A,B,C> for AvxFma<GOTO_MR,GOTO_NR>
where 
AvxFma<GOTO_MR,GOTO_NR>: GemmPack<A::X, TA> + GemmPack<B::X, TB>
{
   const ONE: TC = 1.0;
   #[target_feature(enable = "avx,fma")]
   unsafe fn kernel(
       m: usize, n: usize, k: usize,
       alpha: *const TA,
       beta: *const TC,
       c: *mut TC,
       c_rs: usize, c_cs: usize,
       ap: *const TA, bp: *const TB,
       _kc_last: bool
   ) {
       kernel::<GOTO_MR, GOTO_NR>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp)
   }

}


pub trait SupM {
    unsafe fn kernel_sup_m(
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        b: Self, b_rs: usize, b_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        ap: *const TA,
    );
}

pub trait Axpy {
    unsafe fn axpy(
        m: usize, n: usize,
        alpha: *const TA,
        a: Self,
        x: *const TB, inc_x: usize,
        beta: *const TC,
        y: *mut TC, inc_y: usize,
    );
}

impl Axpy for StridedMatrix<f32>{
    #[target_feature(enable = "avx,fma")]
    unsafe fn axpy(
        m: usize, n: usize,
        alpha: *const TA,
        a: StridedMatrix<f32>,
        x: *const TB, inc_x: usize,
        beta: *const TC,
        y: *mut TC, inc_y: usize,
    ) {
        let a_ptr = a.data_ptr;
        axpy(m, n, alpha, a_ptr, a.rs, a.cs, x, inc_x, beta, y, inc_y);
    }
}

impl Axpy for PackedMatrix<f32>{
    #[target_feature(enable = "avx,fma")]
    unsafe fn axpy(
        m: usize, n: usize,
        alpha: *const TA,
        a: PackedMatrix<f32>,
        x: *const TB, inc_x: usize,
        beta: *const TC,
        y: *mut TC, inc_y: usize,
    ) {
        if m == 1 || n == 1 {
            let a_ptr = a.data_ptr;
            axpy(m, n, alpha, a_ptr, a.rs, a.cs, x, inc_x, beta, y, inc_y);
            return;
        }

        let mut mc = 0;
        let mc_end = m;
        let mc_eff = a.mc;
        let kc_eff = a.kc;
        let one = 1_f32;
        while mc < mc_end {
            let mc_len = mc_eff.min(mc_end - mc);
 
            let c_i = y.add(inc_y*mc);
            let mut kc = 0;
            let kc_end = n;
            // axpy(mc_len, n, alpha, a_ptr, a.rs, a.cs, x, inc_x, beta, c_i, inc_y);
            while kc < n {
                let kc_len = kc_eff.min(kc_end - kc);
                let beta_t = if kc == 0 { beta } else { &one as *const TC};
                let ap = a.packa_dispatch_hw::<AvxFma::<24,4>>(mc, kc, mc_len, kc_len, 0, false);
                let mut mr = 0;
                while mr < mc_len {
                    let mr_len = 24.min(mc_len - mr);
                    let c_i = c_i.add(mr*inc_y);
                    let a_cs = {
                        if mr_len > 16 {
                            24
                        } else if mr_len > 8  {
                            16
                        } else {
                            8
                        }
                    };
                    axpy(mr_len, kc_len, alpha, ap.add(mr*kc_len), 1, a_cs, x.add(inc_x*kc), inc_x, beta_t, c_i, inc_y);
                    mr += 24;
                }
                kc += kc_eff;
            }
            mc += mc_eff;
        }
    }
}

pub trait SupN {
    unsafe fn kernel_sup_n(
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        a: Self, a_rs: usize, a_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        bp: *const TB,
    );
}

impl SupM for StridedMatrix<f32>{
    #[target_feature(enable = "avx,fma")]
    unsafe fn kernel_sup_m(
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        b: StridedMatrix<f32>, b_rs: usize, b_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        ap: *const TA,
    ) {
        let b_ptr = b.data_ptr.add(b_rs*b.rs+b_cs*b.cs);
        kernel_sup_m(m, n, k, alpha, beta, b_ptr, b.rs, b.cs, c, c_rs, c_cs, ap);
    }

}

impl SupN for StridedMatrix<f32>{
    #[target_feature(enable = "avx,fma")]
    unsafe fn kernel_sup_n(
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        a: StridedMatrix<f32>, a_rs: usize, a_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        bp: *const TB,
    ) {
        let a_ptr = a.data_ptr.add(a_rs*a.rs+a_cs*a.cs);
        kernel_sup_n(m, n, k, alpha, beta, a_ptr, a.rs, a.cs, c, c_rs, c_cs, bp);
    }
}

impl SupN for PackedMatrix<f32>{
    #[target_feature(enable = "avx,fma")]
    unsafe fn kernel_sup_n(
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        a: PackedMatrix<f32>, a_rs: usize, a_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        bp: *const TB,
    ) {
        let a_ptr = a.data_ptr.add(a_rs*a.rs+a_cs*a.cs);
        kernel_sup_n(m, n, k, alpha, beta, a_ptr, a.rs, a.cs, c, c_rs, c_cs, bp);
    }
}



use corenum_base::GemmArray;

impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
A: GemmArray<f32>, 
B: GemmArray<f32> + SupM,
C: GemmOut<X=f32,Y=f32>,
> GemmSmallM<TA,TB,A,B,C> for AvxFma<GOTO_MR,GOTO_NR>
where AvxFma<GOTO_MR,GOTO_NR>: GemmPack<A::X, TA>
{
    const ONE: TC = 1.0;
   #[target_feature(enable = "avx,fma")]
   unsafe fn kernel(
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        b: B, b_rs: usize, b_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        ap: *const TA,
   ) {
    B::kernel_sup_m(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap);
   }
}



impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
A: GemmArray<f32>+SupN, 
B: GemmArray<f32>,
C: GemmOut<X=f32,Y=f32>,
> GemmSmallN<TA,TB,A,B,C> for AvxFma<GOTO_MR,GOTO_NR>
where 
AvxFma<GOTO_MR,GOTO_NR>: GemmPack<B::X, TB>
{
    const ONE: TC = 1.0;
   #[target_feature(enable = "avx,fma")]
   unsafe fn kernel(
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        a: A, a_rs: usize, a_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        bp: *const TB,
   ) {
    A::kernel_sup_n(m, n, k, alpha, beta, a, a_rs, a_cs, c, c_rs, c_cs, bp);
   }
}
