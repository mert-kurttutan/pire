
use glare_base::{
    GemmPackA,
    GemmPackB,
    GemmCache,
    GemmArray,
    HWConfig,
    GemmOut,
    AccCoef,
};
 
 
 use crate::{
    TA,TB,TC,
    GemmGotoPackaPackb,
    GemmSmallM,
    GemmSmallN,
    Gemv,
 };

impl AccCoef for RefGemm {
    type AS = TA;
    type BS = TC;
}
 
 
pub struct RefGemm {
    mc: usize, nc: usize, kc: usize,
    mr: usize, nr: usize,
    is_l1_shared: bool,
    is_l2_shared: bool,
    is_l3_shared: bool,
}

impl RefGemm {
    pub fn new(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize) -> Self {
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (mr, nr) = (24, 4);
        Self { mc, nc, kc, mr, nr, is_l1_shared, is_l2_shared, is_l3_shared }
    }
}
 
unsafe fn packa_ref(a: *const TA, ap: *mut TA, m: usize, k: usize, a_rs: usize, a_cs: usize, mr: usize) {
    let mut a_cur = a;
    let mut ap_cur = ap;
    let mut i = 0;
    while i < m / mr {
        let mut j = 0;
        while j < k {
            for ix in 0..mr {
                *ap_cur.add(ix+j*mr) = *a_cur.add(ix*a_rs+j*a_cs);
            }
            j += 1;
        }
        i += 1;
        a_cur = a_cur.add(mr * a_rs);
        ap_cur = ap_cur.add(mr * k);
    }

    let mut j = 0;
    let mr_left = m % mr;
    while j < k {
        for ix in 0..mr_left {
            *ap_cur.add(ix+j*mr_left) = *a_cur.add(ix*a_rs+j*a_cs);
        }
        j += 1;
    }
}
impl GemmPackA<TA,TA> for RefGemm {
    unsafe fn packa_fn(self: &RefGemm, a: *const TA, ap: *mut TA, m: usize, k: usize, a_rs: usize, a_cs: usize) {
        packa_ref(a, ap, m, k, a_rs, a_cs, self.mr);
    }
}

impl GemmPackB<TB,TB> for RefGemm {
    unsafe fn packb_fn(self: &RefGemm, b: *const TB, bp: *mut TB, n: usize, k: usize, b_rs: usize, b_cs: usize) {
        let mut b_cur = b;
        let mut bp_cur = bp;
        let mut i = 0;
        while i < n / self.nr {
            let mut j = 0;
            while j < k {
                for ix in 0..self.nr {
                    *bp_cur.add(ix+j*self.nr) = *b_cur.add(ix*b_cs+j*b_rs);
                }
                j += 1;
            }
            i += 1;
            b_cur = b_cur.add(self.nr * b_cs);
            bp_cur = bp_cur.add(self.nr * k);
        }

        let mut j = 0;
        let n_left = n % self.nr;
        while j < k {
            for ix in 0..n_left {
                *bp_cur.add(ix+j*n_left) = *b_cur.add(ix*b_cs+j*b_rs);
            }
            j += 1;
        }
    }
}


impl<
AP, BP,
A: GemmArray<AP>,
B: GemmArray<BP>,
> GemmCache<AP,BP,A,B> for RefGemm {
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
A: GemmArray<TA,X=TA>, 
B: GemmArray<TB,X=TB>,
C: GemmOut<X=TC,Y=TC>,
// F: MyFn + Sync,
> GemmGotoPackaPackb<TA,TB,A,B,C> for RefGemm
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
        let mut i = 0;
        let mut acc = vec![0.0; self.mr * self.nr];

        while i < m {
            let mr_eff = if i + self.mr > m { m - i } else { self.mr };
            let mut j = 0;
            while j < n {
                let nr_eff = if j + self.nr > n { n - j } else { self.nr };
                let mut p = 0;
                while p < k {
                    let a_cur = ap.add(i * k + p * mr_eff);
                    let b_cur = bp.add(j * k + p * nr_eff);
                    let mut ii = 0;
                    while ii < mr_eff {
                        let mut jj = 0;
                        while jj < nr_eff {
                            acc[ii * nr_eff + jj] += *a_cur.add(ii) * *b_cur.add(jj);
                            jj += 1;
                        }
                        ii += 1;
                    }
                    p += 1;
                }
                // store c
                let mut ii = 0;
                while ii < mr_eff {
                    let mut jj = 0;
                    while jj < nr_eff {
                        *c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) = *c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) * *beta + acc[ii * nr_eff + jj] * *alpha;
                        acc[ii * nr_eff + jj] = 0.0;
                        jj += 1;
                    }
                    ii += 1;
                }
                j += self.nr;
            }

            i += self.mr;
        }
   }
}



impl<
A: GemmArray<TA,X=TA>, 
B: GemmArray<TB,X=TB>,
C: GemmOut<X=TC,Y=TC>,
// F: MyFn + Sync,
> GemmSmallM<TA,TB,A,B,C> for RefGemm
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
        let mut acc = vec![0.0; self.mr * self.nr];
        let mut i = 0;
        while i < m {
            let mr_eff = if i + self.mr > m { m - i } else { self.mr };
            let mut j = 0;
            while j < n {
                let nr_eff = if j + self.nr > n { n - j } else { self.nr };
                let mut p = 0;
                while p < k {
                    let a_cur = ap.add(i * k + p * mr_eff);
                    let b_cur = b.add(j * b_cs + p * b_rs);
                    let mut ii = 0;
                    while ii < mr_eff {
                        let mut jj = 0;
                        while jj < nr_eff {
                            acc[ii * nr_eff + jj] += *a_cur.add(ii) * *b_cur.add(jj*b_cs);
                            jj += 1;
                        }
                        ii += 1;
                    }
                    p += 1;
                }
                // store c
                let mut ii = 0;
                while ii < mr_eff {
                    let mut jj = 0;
                    while jj < nr_eff {
                        *c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) = *c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) * *beta + acc[ii * nr_eff + jj] * *alpha;
                        acc[ii * nr_eff + jj] = 0.0;
                        jj += 1;
                    }
                    ii += 1;
                }
                j += self.nr;
            }
            i += self.mr;
        }
   }
}

impl<
A: GemmArray<TA,X=TA>, 
B: GemmArray<TB,X=TB>,
C: GemmOut<X=TC,Y=TC>,
// F: MyFn + Sync,
> GemmSmallN<TA,TB,A,B,C> for RefGemm
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
        let mut acc = vec![0.0; self.mr * self.nr];
        let mut i = 0;
        while i < m {
            let mr_eff = if i + self.mr > m { m - i } else { self.mr };
            packa_ref(a.add(i * a_rs), ap, mr_eff, k, a_rs, a_cs, self.mr);
            let mut j = 0;
            while j < n {
                let nr_eff = if j + self.nr > n { n - j } else { self.nr };
                let mut p = 0;
                while p < k {
                    let a_cur = ap.add(p * mr_eff);
                    let b_cur = b.add(j * k + p * nr_eff);
                    let mut ii = 0;
                    while ii < mr_eff {
                        let mut jj = 0;
                        while jj < nr_eff {
                            acc[ii * nr_eff + jj] += *a_cur.add(ii) * *b_cur.add(jj);
                            jj += 1;
                        }
                        ii += 1;
                    }
                    p += 1;
                }
                // store c
                let mut ii = 0;
                while ii < mr_eff {
                    let mut jj = 0;
                    while jj < nr_eff {
                        *c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) = *c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) * *beta + acc[ii * nr_eff + jj] * *alpha;
                        acc[ii * nr_eff + jj] = 0.0;
                        jj += 1;
                    }
                    ii += 1;
                }
                j += self.nr;
            }
            i += self.mr;
        }   
   }
}


impl<
A: GemmArray<TA,X=TA>, 
B: GemmArray<TB,X=TB>,
C: GemmOut<X=TC,Y=TC>,
// F: MyFn + Sync,
> Gemv<TA,TB,A,B,C> for RefGemm
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
        let mut i = 0;
        let a_rs = a.rs();
        let a_cs = a.cs();
        let x_ptr = x.get_data_ptr();
        let inc_x = x.rs();
        let y_ptr   = y.data_ptr();
        let incy = y.rs();
        let a_ptr = a.get_data_ptr();

        while i < m {
            let mut j = 0;
            let mut acc = 0.0;
            while j < n {
                acc += *a_ptr.add(i * a_rs + j * a_cs) * *x_ptr.add(j * inc_x);
                j += 1;
            }
            *y_ptr.add(i * incy) = *y_ptr.add(i * incy) * *beta + acc * *alpha;
            i += 1;
        }
   }
}
