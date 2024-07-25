
pub(crate) mod microkernel;


use corenum_base::GemmArrayP;
use corenum_base::Tensor2D;
use corenum_base::PackedMatrix;
pub(crate) use microkernel::{
   packa_panel,
   packb_panel,
   kernel,
   kernel_bs,
   kernel_sb,
   axpy,
};


const VS: usize = 8; // vector size in float, __m256

use corenum_base::{
   StridedMatrix, GemmPackA, GemmPackB
};


use crate::{
   GemmGotoPackaPackb, GemmSmallM, GemmSmallN, Gemv, TA, TB, TC,
   GemmCache,
};

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
A: GemmArray<f32, X=f32>, 
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
        let inc_x = x.rs();
        let y_ptr   = y.data_ptr();
        let incy = y.rs();
        axpy(m, n, alpha, a.get_data_ptr(), a.rs(), a.cs(), x_ptr, inc_x, beta, y_ptr, incy);
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
> GemmPackA<TA,TA> for AvxFma<GOTO_MR,GOTO_NR> {
    #[target_feature(enable = "avx,fma")]
    unsafe fn packa_fn(a: *const TA, ap: *mut TA, m: usize, k: usize, a_rs: usize, a_cs: usize) {
        packa_panel::<GOTO_MR>(m, k, a, a_rs, a_cs, ap);
    }
}

impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
> GemmPackB<TA,TA> for AvxFma<GOTO_MR,GOTO_NR> {
    #[target_feature(enable = "avx,fma")]
    unsafe fn packb_fn(b: *const TA, bp: *mut TA, n: usize, k: usize, b_rs: usize, b_cs: usize) {
        packb_panel::<GOTO_NR>(n, k, b, b_cs, b_rs, bp);
    }
}


// impl<
// const GOTO_MR: usize,
// const GOTO_NR: usize,
// > GemmPackA<u16,TA> for AvxFma<GOTO_MR,GOTO_NR> {
//     #[target_feature(enable = "avx,fma")]
//     unsafe fn packa_fn(a: *const u16, ap: *mut TA, m: usize, k: usize, a_rs: usize, a_cs: usize) {
//         // pack_panel::<GOTO_MR>(m, k, a, a_rs, a_cs, ap);
//     }
// }
// impl<
// const GOTO_MR: usize,
// const GOTO_NR: usize,
// > GemmPackB<u16,TA> for AvxFma<GOTO_MR,GOTO_NR> {
//     #[target_feature(enable = "avx,fma")]
//     unsafe fn packb_fn(b: *const u16, bp: *mut TA, n: usize, k: usize, b_rs: usize, b_cs: usize) {
//         // pack_panel::<GOTO_NR>(n, k, b, b_rs, b_cs, bp);
//     }
// }

impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
AP, BP,
A: GemmArray<AP>,
B: GemmArray<BP>,
> GemmCache<AP,BP,A,B> for AvxFma<GOTO_MR,GOTO_NR> {
    // const CACHELINE_PAD: usize = 256;
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





use corenum_base::GemmOut;

impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
A: GemmArray<f32>, 
B: GemmArray<f32>,
C: GemmOut<X=f32,Y=f32>,
> GemmGotoPackaPackb<TA,TB,A,B,C> for AvxFma<GOTO_MR,GOTO_NR>
where 
AvxFma<GOTO_MR,GOTO_NR>: GemmPackA<A::X, TA> + GemmPackB<B::X, TB>
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

use corenum_base::GemmArray;

impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
A: GemmArray<f32>, 
B: GemmArray<f32, X = f32>,
C: GemmOut<X=f32,Y=f32>,
> GemmSmallM<TA,TB,A,B,C> for AvxFma<GOTO_MR,GOTO_NR>
where AvxFma<GOTO_MR,GOTO_NR>: GemmPackA<A::X, TA>
{
    const ONE: TC = 1.0;
   #[target_feature(enable = "avx,fma")]
   unsafe fn kernel(
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        b: B, //b_rs: usize, b_cs: usize,
        c: *mut TC, c_rs: usize, c_cs: usize,
        ap: *const TA,
   ) {
    let b_ptr = b.get_data_ptr();//.add(b_rs*b.get_rs()+b_cs*b.get_cs());
    kernel_bs(m, n, k, alpha, beta, b_ptr, b.rs(), b.cs(), c, c_rs, c_cs, ap);
    // B::kernel_sup_m(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap);
   }
}



impl<
const GOTO_MR: usize,
const GOTO_NR: usize,
A: GemmArray<f32, X = f32>, 
B: GemmArray<f32>,
C: GemmOut<X=f32,Y=f32>,
> GemmSmallN<TA,TB,A,B,C> for AvxFma<GOTO_MR,GOTO_NR>
where 
AvxFma<GOTO_MR,GOTO_NR>: GemmPackB<B::X, TB>
{
    const ONE: TC = 1.0;
   #[target_feature(enable = "avx,fma")]
   unsafe fn kernel(
        m: usize, n: usize, k: usize,
        alpha: *const TA,
        beta: *const TC,
        a: A::PackArray, b: *const f32,
        c: *mut TC, c_rs: usize, c_cs: usize,
        // ap_buf: *mut TA,
   ) {
        let a_ptr = a.get_data_ptr();
        let a_ptr_rs = a.rs();
        let a_ptr_cs = a.cs();
        let ap_buf = a.get_data_p_ptr();
        kernel_sb(
            m, n, k, alpha, beta, 
            a_ptr, a_ptr_rs, a_ptr_cs, 
            b,
            c, c_rs, c_cs,
            ap_buf
        );
   }
}
