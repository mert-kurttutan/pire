use glare_base::split_c_range;
use glare_base::split_range;
use glare_base::def_glare_gemm;
use glare_base::is_mixed;

use glare_base::{
    GlarePar, GlareThreadConfig,
   HWConfig,
   Array,
   ArrayMut,
    PArray,
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

unsafe fn packb_ref(b: *const TB, bp: *mut TB, n: usize, k: usize, b_rs: usize, b_cs: usize, nr: usize) {
    let mut b_cur = b;
    let mut bp_cur = bp;
    let mut i = 0;
    while i < n / nr {
        let mut j = 0;
        while j < k {
            for ix in 0..nr {
                *bp_cur.add(ix+j*nr) = *b_cur.add(ix*b_cs+j*b_rs);
            }
            j += 1;
        }
        i += 1;
        b_cur = b_cur.add(nr * b_cs);
        bp_cur = bp_cur.add(nr * k);
    }

    let mut j = 0;
    let n_left = n % nr;
    while j < k {
        for ix in 0..n_left {
            *bp_cur.add(ix+j*n_left) = *b_cur.add(ix*b_cs+j*b_rs);
        }
        j += 1;
    }
}




pub(crate) struct RefGemm<
T: MyFn = NullFn
> {
    mc: usize,
    nc: usize,
    kc: usize,
    mr: usize,
    nr: usize,
    // TODO: Cech jr parallelism is beneificial for perf
    // is_l1_shared: bool,
    is_l2_shared: bool,
    is_l3_shared: bool,
    func: T,
    pub(crate) vs: usize,
}

impl<F: MyFn> RefGemm<F> {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, f: F) -> Self {
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (mr, nr) = (24, 4);
        Self { 
            mc, nc, kc, mr, nr, 
            // is_l1_shared, 
            is_l2_shared, is_l3_shared, 
            func: f,
            vs: 1,
        }
    }

    pub(crate) unsafe fn packa_fn(self: &Self, x: *const TA, y: *mut TA, m: usize, k: usize, rs: usize, cs: usize) {
        packa_ref(x, y, m, k, rs, cs, self.mr);
    }

    pub(crate) unsafe fn packb_fn(self: &Self, x: *const TB, y: *mut TB, n: usize, k: usize, rs: usize, cs: usize) {
        packb_ref(x, y, n, k, rs, cs, self.nr);
    }

    pub(crate) fn is_compute_native(&self) -> bool {
        true
    }
    pub(crate) fn round_up(&self, k: usize) -> usize {
        k
    }
}

impl<
T: MyFn,
AP, BP,
> GemmCache<AP,BP> for RefGemm<T> {
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
    hw_cfg: &RefGemm<F>,
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    c: *mut TC,
    c_rs: usize, c_cs: usize,
    ap: *const TA, bp: *const TB,
    kc_last: bool, _kc_first: bool,
) {
    let mut i = 0;
    let mut acc = vec![TC::ZERO; hw_cfg.mr * hw_cfg.nr];

    while i < m {
        let mr_eff = if i + hw_cfg.mr > m { m - i } else { hw_cfg.mr };
        let mut j = 0;
        while j < n {
            let nr_eff = if j + hw_cfg.nr > n { n - j } else { hw_cfg.nr };
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
                    let c_cur = c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs);
                    *c_cur = *c_cur * *beta + acc[ii * nr_eff + jj] * *alpha;
                    if kc_last {
                        hw_cfg.func.call(c_cur, 1);
                    }
                    acc[ii * nr_eff + jj] = TC::ZERO;
                    jj += 1;
                }
                ii += 1;
            }
            j += hw_cfg.nr;
        }

        i += hw_cfg.mr;
    }
}

#[allow(unused)]
unsafe fn kernel_m<F:MyFn>(
    hw_cfg: &RefGemm<F>,
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    b: *const TB, b_rs: usize, b_cs: usize,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap: *const TA,
    kc_last: bool, _kc_first: bool,
) {
}


#[allow(unused)]
unsafe fn kernel_n<F:MyFn>(
    hw_cfg: &RefGemm<F>,
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    a: *const TA, a_rs: usize, a_cs: usize,
    ap: *mut TA,
    b: *const TB,
    c: *mut TC, c_rs: usize, c_cs: usize,
    kc_last: bool, _kc_first: bool,
) {  
}

unsafe fn glare_gemv<F:MyFn>(
    _hw_cfg: &RefGemm<F>,
    m: usize, n: usize,
    alpha: *const TA,
    a: Array<TA>,
    x: Array<TB>,
    beta: *const TC,
    y: ArrayMut<TC>,
) {
    let mut i = 0;
    let a_rs = a.rs();
    let a_cs = a.cs();
    let x_ptr = x.data_ptr();
    let inc_x = x.rs();
    let y_ptr   = y.data_ptr();
    let incy = y.rs();
    let a_ptr = a.data_ptr();

    while i < m {
        let mut j = 0;
        let mut acc = TC::ZERO;
        while j < n {
            acc += *a_ptr.add(i * a_rs + j * a_cs) * *x_ptr.add(j * inc_x);
            j += 1;
        }
        *y_ptr.add(i * incy) = *y_ptr.add(i * incy) * *beta + acc * *alpha;
        i += 1;
    }
}

type C32Pack = PArray<TA>;

def_glare_gemm!(
    RefGemm,
    TA,TA,TB,TB,TC,TA,TC,
    C32Pack, C32Pack,
    TC::ONE,
    glare_gemm, gemm_mt,
    gemm_goto_serial, kernel,
    gemm_small_m_serial, kernel_m,
    gemm_small_n_serial, kernel_n,
    glare_gemv, glare_gemv,
    packa, packb,
    false, false,
    into_pack_array, F,
);
