unsafe fn packa_ref(a: *const i16, ap: *mut i16, m: usize, k: usize, a_rs: usize, a_cs: usize, mr: usize) {
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

unsafe fn packb_ref(b: *const i16, bp: *mut i16, n: usize, k: usize, b_rs: usize, b_cs: usize, nr: usize) {
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


const VS: usize = 8; // vector size in float, __m256

use glare_base::split_c_range;
use glare_base::split_range;
use glare_base::def_glare_gemm;

use glare_base::{
    GlarePar, GlareThreadConfig,
   CpuFeatures,
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
}

impl<F: MyFn> RefGemm<F> {
    pub(crate) fn from_hw_cfg(hw_config: &HWConfig, mc: usize, nc: usize, kc: usize, f: F) -> Self {
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (mr, nr) = (24, 4);
        Self { 
            mc, nc, kc, mr, nr, 
            // is_l1_shared, 
            is_l2_shared, is_l3_shared, 
            func: f 
        }
    }

    unsafe fn packa_fn(&self, x: *const i16, y: *mut i16, m: usize, k: usize, rs: usize, cs: usize) {
        packa_ref(x, y, m, k, rs, cs, self.mr);
    }

    unsafe fn packb_fn(&self, x: *const i16, y: *mut i16, n: usize, k: usize, rs: usize, cs: usize) {
        packb_ref(x, y, n, k, rs, cs, self.nr);
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
    alpha: *const f32,
    beta: *const f32,
    c: *mut i32,
    c_rs: usize, c_cs: usize,
    ap: *const i16, bp: *const i16,
    _kc_last: bool
) {
    let mut i = 0;
    let mut acc = vec![0_i32; hw_cfg.mr * hw_cfg.nr];

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
                        acc[ii * nr_eff + jj] += (*a_cur.add(ii) as i32) * (*b_cur.add(jj) as i32);
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
                    *c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) = ((*c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) as f32) * *beta + acc[ii * nr_eff + jj] as f32 * *alpha) as i32;
                    acc[ii * nr_eff + jj] = 0;
                    jj += 1;
                }
                ii += 1;
            }
            j += hw_cfg.nr;
        }

        i += hw_cfg.mr;
    }
}

unsafe fn kernel_m<F:MyFn>(
    hw_cfg: &RefGemm<F>,
    m: usize, n: usize, k: usize,
    alpha: *const f32,
    beta: *const f32,
    b: *const i16, b_rs: usize, b_cs: usize,
    c: *mut i32, c_rs: usize, c_cs: usize,
    ap: *const i16,
) {
    let mut acc = vec![0_i32; hw_cfg.mr * hw_cfg.nr];
    let mut i = 0;
    while i < m {
        let mr_eff = if i + hw_cfg.mr > m { m - i } else { hw_cfg.mr };
        let mut j = 0;
        while j < n {
            let nr_eff = if j + hw_cfg.nr > n { n - j } else { hw_cfg.nr };
            let mut p = 0;
            while p < k {
                let a_cur = ap.add(i * k + p * mr_eff);
                let b_cur = b.add(j * b_cs + p * b_rs);
                let mut ii = 0;
                while ii < mr_eff {
                    let mut jj = 0;
                    while jj < nr_eff {
                        acc[ii * nr_eff + jj] += (*a_cur.add(ii) as i32) * (*b_cur.add(jj*b_cs) as i32);
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
                    *c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) = ((*c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) as f32) * *beta + acc[ii * nr_eff + jj] as f32 * *alpha) as i32;
                    acc[ii * nr_eff + jj] = 0;
                    jj += 1;
                }
                ii += 1;
            }
            j += hw_cfg.nr;
        }
        i += hw_cfg.mr;
    }
}


unsafe fn kernel_n<F:MyFn>(
    hw_cfg: &RefGemm<F>,
    m: usize, n: usize, k: usize,
    alpha: *const f32,
    beta: *const f32,
    a: *const i16, a_rs: usize, a_cs: usize,
    ap: *mut i16,
    b: *const i16,
    c: *mut i32, c_rs: usize, c_cs: usize,
) {
    let mut acc = vec![0_i32; hw_cfg.mr * hw_cfg.nr];
    let mut i = 0;
    while i < m {
        let mr_eff = if i + hw_cfg.mr > m { m - i } else { hw_cfg.mr };
        packa_ref(a.add(i * a_rs), ap, mr_eff, k, a_rs, a_cs, hw_cfg.mr);
        let mut j = 0;
        while j < n {
            let nr_eff = if j + hw_cfg.nr > n { n - j } else { hw_cfg.nr };
            let mut p = 0;
            while p < k {
                let a_cur = ap.add(p * mr_eff);
                let b_cur = b.add(j * k + p * nr_eff);
                let mut ii = 0;
                while ii < mr_eff {
                    let mut jj = 0;
                    while jj < nr_eff {
                        acc[ii * nr_eff + jj] += (*a_cur.add(ii) as i32) * (*b_cur.add(jj) as i32);
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
                    *c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) = ((*c.add(i * c_rs + j * c_cs + ii * c_rs + jj * c_cs) as f32) * *beta + acc[ii * nr_eff + jj] as f32 * *alpha) as i32;
                    acc[ii * nr_eff + jj] = 0;
                    jj += 1;
                }
                ii += 1;
            }
            j += hw_cfg.nr;
        }
        i += hw_cfg.mr;
    }   
}

unsafe fn glare_gemv<F:MyFn>(
    hw_cfg: &RefGemm<F>,
    m: usize, n: usize,
    alpha: *const f32,
    a: Array<i16>,
    x: Array<i16>,
    beta: *const f32,
    y: ArrayMut<i32>,
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
        let mut acc = 0_i32;
        while j < n {
            acc += (*a_ptr.add(i * a_rs + j * a_cs) as i32) * (*x_ptr.add(j * inc_x) as i32);
            j += 1;
        }
        *y_ptr.add(i * incy) = ((*y_ptr.add(i * incy) as f32) * *beta + acc as f32* *alpha) as i32;
        i += 1;
    }
}


def_glare_gemm!(
    RefGemm,
    i16,i16,i16,i16,i32,f32,f32,
    1_f32,
    glare_gemm, gemm_mt,
    gemm_goto_serial, kernel,
    gemm_small_m_serial, kernel_m,
    gemm_small_n_serial, kernel_n,
    glare_gemv,
    packa, packb,
    true, true,
);
