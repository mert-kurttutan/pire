use pire_base::{
    acquire, def_pa, def_pire_gemm, extend, get_apbp_barrier, get_cache_params, get_mem_pool_size_goto,
    get_mem_pool_size_small_m, get_mem_pool_size_small_n, is_mixed, run_small_m, run_small_n, split_c_range,
    split_range, Array, ArrayMut, GemmPool, PArrayMixed, PirePar, PireThreadConfig, PoolSize, PtrData, PACK_POOL,
    RUNTIME_HW_CONFIG,
};

use half::f16;

const MR: usize = 24;
const NR: usize = 4;

use crate::{GemmCache, IdentityFn, UnaryFnC};

unsafe fn packa_fn(a: *const f16, ap: *mut f32, m: usize, k: usize, a_rs: usize, a_cs: usize) {
    let mut a_cur = a;
    let mut ap_cur = ap;
    let mr = MR;
    let mut i = 0;
    while i < m / mr {
        let mut j = 0;
        while j < k {
            for ix in 0..mr {
                *ap_cur.add(ix + j * mr) = (*a_cur.add(ix * a_rs + j * a_cs)).to_f32();
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
            *ap_cur.add(ix + j * mr_left) = (*a_cur.add(ix * a_rs + j * a_cs)).to_f32();
        }
        j += 1;
    }
}

unsafe fn packb_fn(b: *const f16, bp: *mut f32, n: usize, k: usize, b_rs: usize, b_cs: usize) {
    let mut b_cur = b;
    let mut bp_cur = bp;
    let nr = NR;
    let mut i = 0;
    while i < n / nr {
        let mut j = 0;
        while j < k {
            for ix in 0..nr {
                *bp_cur.add(ix + j * nr) = (*b_cur.add(ix * b_cs + j * b_rs)).to_f32();
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
            *bp_cur.add(ix + j * n_left) = (*b_cur.add(ix * b_cs + j * b_rs)).to_f32();
        }
        j += 1;
    }
}

pub(crate) unsafe fn packa_fn_ref(a: *const f16, ap: *mut f16, m: usize, k: usize, a_rs: usize, a_cs: usize) {
    let mut a_cur = a;
    let mut ap_cur = ap;
    let mut i = 0;
    let mr = MR;
    while i < m / mr {
        let mut j = 0;
        while j < k {
            for ix in 0..mr {
                *ap_cur.add(ix + j * mr) = *a_cur.add(ix * a_rs + j * a_cs);
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
            *ap_cur.add(ix + j * mr_left) = *a_cur.add(ix * a_rs + j * a_cs);
        }
        j += 1;
    }
}

pub(crate) unsafe fn packb_fn_ref(b: *const f16, bp: *mut f16, n: usize, k: usize, b_rs: usize, b_cs: usize) {
    let mut b_cur = b;
    let mut bp_cur = bp;
    let mut i = 0;
    let nr = NR;
    while i < n / nr {
        let mut j = 0;
        while j < k {
            for ix in 0..nr {
                *bp_cur.add(ix + j * nr) = *b_cur.add(ix * b_cs + j * b_rs);
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
            *bp_cur.add(ix + j * n_left) = *b_cur.add(ix * b_cs + j * b_rs);
        }
        j += 1;
    }
}

pub(crate) fn round_m_ref(m: usize) -> usize {
    m
}

pub(crate) fn round_k_ref(k: usize) -> usize {
    k
}

pub(crate) struct RefGemm<T: UnaryFnC = IdentityFn> {
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

impl<F: UnaryFnC> RefGemm<F> {
    pub(crate) fn new(f: F) -> Self {
        let hw_config = &*RUNTIME_HW_CONFIG;
        let (mc, nc, kc) = get_cache_params();
        let (_, is_l2_shared, is_l3_shared) = hw_config.get_cache_info();
        let (mr, nr) = (MR, NR);
        Self {
            mc,
            nc,
            kc,
            mr,
            nr,
            // is_l1_shared,
            is_l2_shared,
            is_l3_shared,
            func: f,
        }
    }

    unsafe fn cvt_mixed(&self, x: *const f16, y: *mut f32, m: usize) {
        let mut i = 0;
        while i < m {
            *y.add(i) = (*x.add(i)).to_f32();
            i += 1;
        }
    }
    pub(crate) fn is_compute_native(&self) -> bool {
        false
    }

    pub(crate) fn round_k(&self, k: usize) -> usize {
        k
    }
    pub(crate) fn round_m(&self, m: usize) -> usize {
        m
    }
}

impl<T: UnaryFnC> GemmCache for RefGemm<T> {
    fn mr(&self) -> usize {
        self.mr
    }
    fn get_kc_eff(&self) -> usize {
        self.kc
    }
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

unsafe fn kernel<F: UnaryFnC>(
    hw_cfg: &RefGemm<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
    ap: *const f32,
    bp: *const f32,
    kc_last: bool,
) {
    let mut i = 0;
    let mut acc = vec![0.0; hw_cfg.mr * hw_cfg.nr];

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
                    *c_cur = f16::from_f32((*c_cur).to_f32() * *beta + acc[ii * nr_eff + jj] * *alpha);
                    if kc_last {
                        hw_cfg.func.call(c_cur, 1);
                    }
                    acc[ii * nr_eff + jj] = 0.0;
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
unsafe fn kernel_m<F: UnaryFnC>(
    hw_cfg: &RefGemm<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    b: *const f16,
    b_rs: usize,
    b_cs: usize,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
    ap: *const f32,
    kc_last: bool,
) {
}

#[allow(unused)]
unsafe fn kernel_n<F: UnaryFnC>(
    hw_cfg: &RefGemm<F>,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    beta: *const f32,
    a: *const f16,
    a_rs: usize,
    a_cs: usize,
    ap: *mut f32,
    b: *const f32,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
    kc_last: bool,
) {
}

unsafe fn pire_gemv<F: UnaryFnC>(
    hw_cfg: &RefGemm<F>,
    m: usize,
    n: usize,
    alpha: *const f32,
    a: Array<f16>,
    x: Array<f16>,
    beta: *const f32,
    y: ArrayMut<f16>,
) {
    let mut i = 0;
    let a_rs = a.rs();
    let a_cs = a.cs();
    let x_ptr = x.src();
    let inc_x = x.rs();
    let y_ptr = y.src();
    let incy = y.rs();
    let a_ptr = a.src();

    while i < m {
        let mut j = 0;
        let mut acc = 0.0;
        while j < n {
            acc += (*a_ptr.add(i * a_rs + j * a_cs)).to_f32() * (*x_ptr.add(j * inc_x)).to_f32();
            j += 1;
        }
        *y_ptr.add(i * incy) = f16::from_f32((*y_ptr.add(i * incy)).to_f32() * *beta + acc * *alpha);

        hw_cfg.func.call(y_ptr.add(i * incy), 1);
        i += 1;
    }
}

def_pire_gemm!(
    RefGemm,
    f16,
    f32,
    f16,
    f32,
    f16,
    f32,
    f32,
    PackArrTypeA,
    PackArrTypeB,
    1_f32,
    pire_gemm,
    gemm_mt,
    gemm_goto_serial,
    kernel,
    gemm_small_m_serial,
    kernel_m,
    gemm_small_n_serial,
    kernel_n,
    pire_gemv,
    pire_gemv,
    packa0,
    packb0,
    packa_fn,
    packb_fn,
    false,
    false,
    into_pack_array2,
    T,
);
