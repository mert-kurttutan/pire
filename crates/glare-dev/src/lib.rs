#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(unused)]

use libc::{c_double, c_float, c_int, c_schar, c_short, c_ushort, c_void};

use num_complex::{c32, c64, Complex32, Complex64};

use half::f16;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::enum_variant_names)]
pub enum CBLAS_LAYOUT {
    CblasRowMajor = 101,
    CblasColMajor = 102,
}
pub use self::CBLAS_LAYOUT::*;

#[repr(C)]
pub struct cntx_t(i32);

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::enum_variant_names)]
pub enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113,
}
pub use self::CBLAS_TRANSPOSE::*;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum CBLAS_OFFSET {
    CblasRowOffset = 171,
    CblasColOffset = 172,
    CblasFixOffset = 173,
}
pub use self::CBLAS_OFFSET::*;

#[cfg(feature = "mkl")]
#[allow(dead_code)]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn cblas_gemm_s8u8s32(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        offsetc: CBLAS_OFFSET,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_void,
        lda: c_int,
        oa: c_schar,
        b: *const c_void,
        ldb: c_int,
        ob: c_schar,
        beta: c_float,
        c: *mut c_int,
        ldc: c_int,
        oc: *const c_int,
    );

    pub fn cblas_sgemv(
        layout: CBLAS_LAYOUT,
        trans: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        x: *const c_float,
        incx: c_int,
        beta: c_float,
        y: *mut c_float,
        incy: c_int,
    );

    pub fn cblas_dgemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        b: *const c_double,
        ldb: c_int,
        beta: c_double,
        c: *mut c_double,
        ldc: c_int,
    );

    pub fn cblas_dgemv(
        layout: CBLAS_LAYOUT,
        trans: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        x: *const c_double,
        incx: c_int,
        beta: c_double,
        y: *mut c_double,
        incy: c_int,
    );

    pub fn cblas_sgemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );

    pub fn cblas_hgemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_ushort,
        a: *const c_ushort,
        lda: c_int,
        b: *const c_ushort,
        ldb: c_int,
        beta: c_ushort,
        c: *mut c_ushort,
        ldc: c_int,
    );

    pub fn cblas_cgemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        lda: c_int,
        b: *const c_void,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        ldc: c_int,
    );

    pub fn cblas_zgemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        lda: c_int,
        b: *const c_void,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        ldc: c_int,
    );

    #[allow(clippy::too_many_arguments)]
    pub fn cblas_gemm_s16s16s32(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        offsetc: CBLAS_OFFSET,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_short,
        lda: c_int,
        oa: c_short,
        b: *const c_short,
        ldb: c_int,
        ob: c_short,
        beta: c_float,
        c: *mut c_int,
        ldc: c_int,
        oc: *const c_int,
    );

    pub fn cblas_sgemm_batch(
        layout: CBLAS_LAYOUT,
        transa: *const CBLAS_TRANSPOSE,
        transb: *const CBLAS_TRANSPOSE,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const c_float,
        a: *const *const c_float,
        lda: *const c_int,
        b: *const *const c_float,
        ldb: *const c_int,
        beta: *const c_float,
        c: *const *mut c_float,
        ldc: *const c_int,
        group_count: c_int,
        group_size: *const c_int,
    );

}

const BLIS_TRANS_SHIFT: usize = 3;
const BLIS_CONJ_SHIFT: usize = 4;
const BLIS_UPLO_SHIFT: usize = 5;
const BLIS_UPPER_SHIFT: usize = 5;
const BLIS_DIAG_SHIFT: usize = 6;
const BLIS_LOWER_SHIFT: usize = 7;

/// Conjugation enum
#[repr(C)]
pub enum conj_t {
    BLIS_NO_CONJUGATE = 0,
    BLIS_CONJUGATE = 1 << BLIS_CONJ_SHIFT,
}

pub use self::conj_t::*;
/// Transpose enum
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum trans_t {
    BLIS_NO_TRANSPOSE = 0,
    BLIS_TRANSPOSE = 1 << BLIS_TRANS_SHIFT,
    BLIS_CONJ_NO_TRANSPOSE = 1 << BLIS_CONJ_SHIFT,
    BLIS_CONJ_TRANSPOSE = 1 << BLIS_TRANS_SHIFT | 1 << BLIS_CONJ_SHIFT,
}
pub use self::trans_t::*;
#[cfg(feature = "blis")]
extern "C" {
    pub fn bli_cgemm(
        transa: trans_t,
        transb: trans_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        rsa: i32,
        csa: i32,
        b: *const c_void,
        rsb: i32,
        csb: i32,
        beta: *const c_void,
        c: *mut c_void,
        rsc: i32,
        csc: i32,
    );
    pub fn bli_zgemm(
        transa: trans_t,
        transb: trans_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        rsa: i32,
        csa: i32,
        b: *const c_void,
        rsb: i32,
        csb: i32,
        beta: *const c_void,
        c: *mut c_void,
        rsc: i32,
        csc: i32,
    );
    pub fn bli_sgemm(
        transa: trans_t,
        transb: trans_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_float,
        a: *const c_float,
        rsa: i32,
        csa: i32,
        b: *const c_float,
        rsb: i32,
        csb: i32,
        beta: *const c_float,
        c: *mut c_float,
        rsc: i32,
        csc: i32,
    );

    pub fn bli_dgemm(
        // layout: CBLAS_LAYOUT,
        transa: trans_t,
        transb: trans_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_double,
        a: *const c_double,
        rsa: i32,
        csa: i32,
        b: *const c_double,
        rsb: i32,
        csb: i32,
        beta: *const c_double,
        c: *mut c_double,
        rsc: i32,
        csc: i32,
    );

}

pub enum ABLayout {
    NN,
    NT,
    TN,
    TT,
}

pub fn layout_to_strides(
    layout: &ABLayout,
    m: usize,
    n: usize,
    k: usize,
) -> (usize, usize, usize, usize, usize, usize) {
    match layout {
        ABLayout::NN => (1, m, 1, k, 1, m),
        ABLayout::NT => (1, m, n, 1, 1, m),
        ABLayout::TN => (k, 1, 1, k, 1, m),
        ABLayout::TT => (k, 1, n, 1, 1, m),
    }
}

use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub trait Bound {
    type X: rand::distributions::uniform::SampleUniform;
    fn min_value() -> Self::X;
    fn max_value() -> Self::X;
    fn my_sample(dist: &Uniform<Self::X>, rng: &mut StdRng) -> Self;
}

impl Bound for f32 {
    type X = f32;
    fn min_value() -> Self {
        -2.0
    }
    fn max_value() -> Self {
        2.0
    }
    fn my_sample(dist: &Uniform<Self>, rng: &mut StdRng) -> Self {
        dist.sample(rng)
    }
}

impl Bound for f64 {
    type X = f64;
    fn min_value() -> Self {
        -10.0
    }
    fn max_value() -> Self {
        10.0
    }
    fn my_sample(dist: &Uniform<Self>, rng: &mut StdRng) -> Self {
        dist.sample(rng)
    }
}

impl Bound for i16 {
    type X = i16;
    fn min_value() -> Self {
        -10
    }
    fn max_value() -> Self {
        10
    }

    fn my_sample(dist: &Uniform<Self>, rng: &mut StdRng) -> Self {
        dist.sample(rng)
    }
}

impl Bound for i8 {
    type X = i8;
    fn min_value() -> Self {
        -10
    }
    fn max_value() -> Self {
        10
    }

    fn my_sample(dist: &Uniform<Self>, rng: &mut StdRng) -> Self {
        dist.sample(rng)
    }
}

impl Bound for u8 {
    type X = u8;
    fn min_value() -> Self {
        10
    }
    fn max_value() -> Self {
        20
    }

    fn my_sample(dist: &Uniform<Self>, rng: &mut StdRng) -> Self {
        dist.sample(rng)
    }
}

impl Bound for i32 {
    type X = i32;
    fn min_value() -> Self {
        -10
    }
    fn max_value() -> Self {
        10
    }
    fn my_sample(dist: &Uniform<Self>, rng: &mut StdRng) -> Self {
        dist.sample(rng)
    }
}

impl Bound for f16 {
    type X = f16;
    fn min_value() -> Self {
        f16::from_f32(-1.0)
    }
    fn max_value() -> Self {
        f16::from_f32(1.0)
    }
    fn my_sample(dist: &Uniform<Self>, rng: &mut StdRng) -> Self {
        dist.sample(rng)
    }
}

impl Bound for Complex<f32> {
    type X = f32;
    fn min_value() -> f32 {
        -1.0
    }
    fn max_value() -> f32 {
        1.0
    }
    fn my_sample(dist: &Uniform<f32>, rng: &mut StdRng) -> Self {
        // dist.sample(rng)
        let x = dist.sample(rng);
        let y = dist.sample(rng);
        Complex::new(x, y)
    }
}

impl Bound for Complex<f64> {
    type X = f64;
    fn min_value() -> f64 {
        -1.0
    }
    fn max_value() -> f64 {
        1.0
    }
    fn my_sample(dist: &Uniform<f64>, rng: &mut StdRng) -> Self {
        // dist.sample(rng)
        let x = dist.sample(rng);
        let y = dist.sample(rng);
        Complex::new(x, y)
    }
}

pub fn random_matrix_std<T>(m: usize, n: usize, arr: &mut [T], ld: usize)
where
    rand::distributions::Standard: rand::prelude::Distribution<T>,
{
    let mut x = StdRng::seed_from_u64(43);
    for j in 0..n {
        for i in 0..m {
            arr[j * ld + i] = x.gen::<T>();
        }
    }
}

pub fn random_matrix_uniform<T>(m: usize, n: usize, arr: &mut [T], ld: usize)
where
    T: Bound,
    T::X: rand::distributions::uniform::SampleUniform,
{
    let t0 = T::min_value();
    let t1 = T::max_value();
    let mut x = StdRng::seed_from_u64(43);
    let un_dist = Uniform::new(t0, t1);
    for j in 0..n {
        for i in 0..m {
            // arr[j * ld + i] = un_dist.sample(&mut x);
            arr[j * ld + i] = T::my_sample(&un_dist, &mut x);
        }
    }
}

pub trait Diff {
    fn diff(&self, other: &Self) -> f64;
}

impl Diff for f32 {
    fn diff(&self, other: &Self) -> f64 {
        let diff_abs = (self - other).abs();
        let diff_rel = diff_abs / self.abs();
        diff_abs.min(diff_rel) as f64
    }
}

impl Diff for f64 {
    fn diff(&self, other: &Self) -> f64 {
        let diff_abs = (self - other).abs();
        let diff_rel = diff_abs / self.abs();
        diff_abs.min(diff_rel) as f64
    }
}

impl Diff for i16 {
    fn diff(&self, other: &Self) -> f64 {
        let diff_abs = (*self - *other).abs() as f64;
        diff_abs
    }
}

impl Diff for i8 {
    fn diff(&self, other: &Self) -> f64 {
        let diff_abs = (*self as i16 - *other as i16).abs() as f64;
        diff_abs
    }
}

impl Diff for u8 {
    fn diff(&self, other: &Self) -> f64 {
        let diff_abs = (*self as i16 - *other as i16).abs() as f64;
        diff_abs
    }
}

impl Diff for i32 {
    fn diff(&self, other: &Self) -> f64 {
        let diff_abs = (*self - *other).abs() as f64;
        diff_abs
    }
}

impl Diff for f16 {
    fn diff(&self, other: &Self) -> f64 {
        let x = self.to_f32();
        let y = other.to_f32();
        let diff_abs = (x - y).abs();
        let diff_rel = diff_abs / x.abs();
        diff_abs.min(diff_rel) as f64
    }
}

use num_complex::Complex;

impl Diff for Complex<f32> {
    fn diff(&self, other: &Self) -> f64 {
        let diff_re = self.re.diff(&other.re);
        let diff_im = self.im.diff(&other.im);
        diff_re.max(diff_im)
    }
}

impl Diff for Complex<f64> {
    fn diff(&self, other: &Self) -> f64 {
        let diff_re = self.re.diff(&other.re);
        let diff_im = self.im.diff(&other.im);
        diff_re.max(diff_im)
    }
}

pub fn max_abs_diff<T: Copy + std::fmt::Debug>(ap: &[T], bp: &[T], eps: f64) -> f64
where
    T: Diff,
{
    let mut diff = 0_f64;
    let len = ap.len();
    // println!("------------------------------");
    let mut diff_idx = 0;
    for i in 0..len {
        let a = ap[i];
        let b = bp[i];
        let cur_diff: f64 = a.diff(&b);
        if cur_diff > diff {
            // println!("diff: {:?} {:?}", a, b);
            // println!("idx: {:?}", i);
            diff_idx = i;
            diff = cur_diff;
        }
    }
    diff
}

pub unsafe fn gemm_fallback_f64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    a_rs: usize,
    a_cs: usize,
    b: *const f64,
    b_rs: usize,
    b_cs: usize,
    beta: f64,
    c: *mut f64,
    c_rs: usize,
    c_cs: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut dx = 0.0;
            for p in 0..k {
                dx += *a.add(a_rs * i + a_cs * p) * *b.add(b_rs * p + b_cs * j);
            }
            *c.add(c_rs * i + c_cs * j) = alpha * dx + beta * *c.add(c_rs * i + c_cs * j);
        }
    }
}

pub unsafe fn gemm_fallback_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    a_rs: usize,
    a_cs: usize,
    b: *const f32,
    b_rs: usize,
    b_cs: usize,
    beta: f32,
    c: *mut f32,
    c_rs: usize,
    c_cs: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut dx = 0.0;
            for p in 0..k {
                dx += *a.add(a_rs * i + a_cs * p) * *b.add(b_rs * p + b_cs * j);
            }
            *c.add(c_rs * i + c_cs * j) = alpha * dx + beta * *c.add(c_rs * i + c_cs * j);
        }
    }
}

pub unsafe fn gemm_fallback_s16s16s32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const i16,
    a_rs: usize,
    a_cs: usize,
    b: *const i16,
    b_rs: usize,
    b_cs: usize,
    beta: f32,
    c: *mut i32,
    c_rs: usize,
    c_cs: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut dx = 0i32;
            for p in 0..k {
                dx += *a.add(a_rs * i + a_cs * p) as i32 * *b.add(b_rs * p + b_cs * j) as i32;
            }
            *c.add(c_rs * i + c_cs * j) =
                (alpha * dx as f32 + beta * *c.add(c_rs * i + c_cs * j) as f32) as i32;
        }
    }
}

pub unsafe fn gemm_fallback_s8u8s32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const i8,
    a_rs: usize,
    a_cs: usize,
    b: *const u8,
    b_rs: usize,
    b_cs: usize,
    beta: f32,
    c: *mut i32,
    c_rs: usize,
    c_cs: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut dx = 0i32;
            for p in 0..k {
                dx += *a.add(a_rs * i + a_cs * p) as i32 * *b.add(b_rs * p + b_cs * j) as i32;
            }
            *c.add(c_rs * i + c_cs * j) =
                (alpha * dx as f32 + beta * *c.add(c_rs * i + c_cs * j) as f32) as i32;
        }
    }
}

pub unsafe fn gemm_fallback_c32(
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: *const Complex32,
    a_rs: usize,
    a_cs: usize,
    b: *const Complex32,
    b_rs: usize,
    b_cs: usize,
    beta: Complex32,
    c: *mut Complex32,
    c_rs: usize,
    c_cs: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut dx = Complex32::ZERO;
            for p in 0..k {
                dx += *a.add(a_rs * i + a_cs * p) * *b.add(b_rs * p + b_cs * j);
            }
            *c.add(c_rs * i + c_cs * j) = alpha * dx + beta * *c.add(c_rs * i + c_cs * j);
        }
    }
}

pub unsafe fn gemm_fallback_c64(
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: *const Complex64,
    a_rs: usize,
    a_cs: usize,
    b: *const Complex64,
    b_rs: usize,
    b_cs: usize,
    beta: Complex64,
    c: *mut Complex64,
    c_rs: usize,
    c_cs: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut dx = Complex64::ZERO;
            for p in 0..k {
                dx += *a.add(a_rs * i + a_cs * p) * *b.add(b_rs * p + b_cs * j);
            }
            *c.add(c_rs * i + c_cs * j) = alpha * dx + beta * *c.add(c_rs * i + c_cs * j);
        }
    }
}

pub unsafe fn gemm_fallback_f16(
    m: usize,
    n: usize,
    k: usize,
    alpha: f16,
    a: *const f16,
    a_rs: usize,
    a_cs: usize,
    b: *const f16,
    b_rs: usize,
    b_cs: usize,
    beta: f16,
    c: *mut f16,
    c_rs: usize,
    c_cs: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut dx = f16::ZERO;
            for p in 0..k {
                dx += *a.add(a_rs * i + a_cs * p) * *b.add(b_rs * p + b_cs * j);
            }
            *c.add(c_rs * i + c_cs * j) = alpha * dx + beta * *c.add(c_rs * i + c_cs * j);
        }
    }
}

pub fn stride_to_cblas(
    m: usize,
    n: usize,
    k: usize,
    a_rs: usize,
    a_cs: usize,
    b_rs: usize,
    b_cs: usize,
    c_rs: usize,
    c_cs: usize,
) -> (CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, c_int, c_int, c_int) {
    let (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs) = if c_rs == 1 {
        (a_rs, a_cs, b_rs, b_cs, c_rs, c_cs)
    } else if c_cs == 1 {
        (a_cs, a_rs, b_cs, b_rs, c_cs, c_rs)
    } else {
        panic!("Non Trivial Stride is not available for Cblas Api");
    };
    // c_rs == 1
    let ldc = c_cs as c_int;
    let (a_trans, b_trans, lda, ldb) = if a_rs == 1 && b_rs == 1 && a_cs == m && b_cs == k {
        (CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, a_cs as c_int, b_cs as c_int)
    } else if a_rs == 1 && b_cs == 1 && a_cs == m && b_rs == n {
        (CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans, a_cs as c_int, b_rs as c_int)
    } else if a_cs == 1 && b_rs == 1 && a_rs == k && b_cs == k {
        (CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans, a_rs as c_int, b_cs as c_int)
    } else if a_cs == 1 && b_cs == 1 && a_rs == k && b_rs == n {
        (CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasTrans, a_rs as c_int, b_rs as c_int)
    } else {
        panic!("Non Trivial Stride is not available for Cblas Api");
    };
    (CBLAS_LAYOUT::CblasColMajor, a_trans, b_trans, lda, ldb, ldc)
}

fn cblas_to_stride(
    layout: CBLAS_LAYOUT,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    lda: c_int,
    ldb: c_int,
    ldc: c_int,
) -> (usize, usize, usize, usize, usize, usize) {
    if layout == CBLAS_LAYOUT::CblasColMajor {
        let (a_rs, a_cs) = if transa == CBLAS_TRANSPOSE::CblasNoTrans {
            (1, lda as usize)
        } else {
            (lda as usize, 1)
        };
        let (b_rs, b_cs) = if transb == CBLAS_TRANSPOSE::CblasNoTrans {
            (1, ldb as usize)
        } else {
            (ldb as usize, 1)
        };
        (a_rs, a_cs, b_rs, b_cs, 1, ldc as usize)
    } else {
        let (a_rs, a_cs) = if transa == CBLAS_TRANSPOSE::CblasNoTrans {
            (lda as usize, 1)
        } else {
            (1, lda as usize)
        };
        let (b_rs, b_cs) = if transb == CBLAS_TRANSPOSE::CblasNoTrans {
            (ldb as usize, 1)
        } else {
            (1, ldb as usize)
        };
        (a_rs, a_cs, b_rs, b_cs, ldc as usize, 1)
    }
}

pub unsafe fn check_gemm_s16s16s32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const i16,
    a_rs: usize,
    a_cs: usize,
    b: *const i16,
    b_rs: usize,
    b_cs: usize,
    beta: f32,
    c: &[i32],
    c_rs: usize,
    c_cs: usize,
    c_ref: &mut [i32],
    unary: unsafe fn(*mut i32, m: usize),
    eps: f64,
) -> f64 {
    #[cfg(feature = "mkl")]
    {
        let oc_val = 0;
        let oc = &oc_val as *const c_int;
        let (layout, transa, transb, lda, ldb, ldc) =
            stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        cblas_gemm_s16s16s32(
            layout,
            transa,
            transb,
            CblasFixOffset,
            m as c_int,
            n as c_int,
            k as c_int,
            alpha,
            a,
            lda,
            0,
            b,
            ldb,
            0,
            beta,
            c_ref.as_mut_ptr(),
            ldc,
            oc,
        );
    }
    #[cfg(not(feature = "mkl"))]
    {
        // calculate diff using fallback
        gemm_fallback_s16s16s32(
            m,
            n,
            k,
            alpha,
            a,
            a_rs,
            a_cs,
            b,
            b_rs,
            b_cs,
            beta,
            c_ref.as_mut_ptr(),
            c_rs,
            c_cs,
        );
    }

    let c_ref_ptr = c_ref.as_mut_ptr();
    if c_rs == 1 {
        for j in 0..n {
            unary(c_ref_ptr.add(j * c_cs), m);
        }
    } else if c_cs == 1 {
        for i in 0..m {
            unary(c_ref_ptr.add(i * c_rs), n);
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                unary(c_ref_ptr.add(i * c_rs + j * c_cs), 1);
            }
        }
    }

    let diff = max_abs_diff(&c, &c_ref, eps);
    return diff;
}

pub unsafe fn check_gemm_s8u8s32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const i8,
    a_rs: usize,
    a_cs: usize,
    b: *const u8,
    b_rs: usize,
    b_cs: usize,
    beta: f32,
    c: &[i32],
    c_rs: usize,
    c_cs: usize,
    c_ref: &mut [i32],
    unary: unsafe fn(*mut i32, m: usize),
    eps: f64,
) -> f64 {
    #[cfg(feature = "mkl")]
    {
        let oc_val = 0;
        let oc = &oc_val as *const c_int;
        let (layout, transa, transb, lda, ldb, ldc) =
            stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        let a = a as *const c_void;
        let b = b as *const c_void;
        cblas_gemm_s8u8s32(
            layout,
            transa,
            transb,
            CblasFixOffset,
            m as c_int,
            n as c_int,
            k as c_int,
            alpha,
            a,
            lda,
            0,
            b,
            ldb,
            0,
            beta,
            c_ref.as_mut_ptr(),
            ldc,
            oc,
        );
    }
    #[cfg(not(feature = "mkl"))]
    {
        // calculate diff using fallback
        gemm_fallback_s8u8s32(
            m,
            n,
            k,
            alpha,
            a,
            a_rs,
            a_cs,
            b,
            b_rs,
            b_cs,
            beta,
            c_ref.as_mut_ptr(),
            c_rs,
            c_cs,
        );
    }

    let c_ref_ptr = c_ref.as_mut_ptr();
    if c_rs == 1 {
        for j in 0..n {
            unary(c_ref_ptr.add(j * c_cs), m);
        }
    } else if c_cs == 1 {
        for i in 0..m {
            unary(c_ref_ptr.add(i * c_rs), n);
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                unary(c_ref_ptr.add(i * c_rs + j * c_cs), 1);
            }
        }
    }

    let diff = max_abs_diff(&c, &c_ref, eps);
    return diff;
}

pub unsafe fn check_gemm_f16(
    m: usize,
    n: usize,
    k: usize,
    alpha: f16,
    a: *const f16,
    a_rs: usize,
    a_cs: usize,
    b: *const f16,
    b_rs: usize,
    b_cs: usize,
    beta: f16,
    c: &[f16],
    c_rs: usize,
    c_cs: usize,
    c_ref: &mut [f16],
    unary: unsafe fn(*mut f16, m: usize),
    eps: f64,
) -> f64 {
    #[cfg(feature = "mkl")]
    {
        let (layout, transa, transb, lda, ldb, ldc) =
            stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        let a = a as *const c_ushort;
        let b = b as *const c_ushort;
        let c_ref_ptr = c_ref.as_mut_ptr() as *mut c_ushort;
        let alpha = alpha.to_bits();
        let beta = beta.to_bits();
        cblas_hgemm(
            layout, transa, transb, m as c_int, n as c_int, k as c_int, alpha, a, lda, b, ldb,
            beta, c_ref_ptr, ldc,
        );
    }
    #[cfg(not(feature = "mkl"))]
    {
        // calculate diff using fallback
        gemm_fallback_f16(
            m,
            n,
            k,
            alpha,
            a,
            a_rs,
            a_cs,
            b,
            b_rs,
            b_cs,
            beta,
            c_ref.as_mut_ptr(),
            c_rs,
            c_cs,
        );
    }

    let c_ref_ptr = c_ref.as_mut_ptr();
    if c_rs == 1 {
        for j in 0..n {
            unary(c_ref_ptr.add(j * c_cs), m);
        }
    } else if c_cs == 1 {
        for i in 0..m {
            unary(c_ref_ptr.add(i * c_rs), n);
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                unary(c_ref_ptr.add(i * c_rs + j * c_cs), 1);
            }
        }
    }

    let diff = max_abs_diff(&c, &c_ref, eps);
    return diff;
}

pub unsafe fn check_gemm_f64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    a_rs: usize,
    a_cs: usize,
    b: *const f64,
    b_rs: usize,
    b_cs: usize,
    beta: f64,
    c: &[f64],
    c_rs: usize,
    c_cs: usize,
    c_ref: &mut [f64],
    unary: unsafe fn(*mut f64, m: usize),
    eps: f64,
) -> f64 {
    #[cfg(feature = "mkl")]
    {
        let (layout, transa, transb, lda, ldb, ldc) =
            stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        cblas_dgemm(
            layout,
            transa,
            transb,
            m as c_int,
            n as c_int,
            k as c_int,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c_ref.as_mut_ptr(),
            ldc,
        );
    }
    #[cfg(not(feature = "mkl"))]
    {
        // calculate diff using fallback
        gemm_fallback_f64(
            m,
            n,
            k,
            alpha,
            a,
            a_rs,
            a_cs,
            b,
            b_rs,
            b_cs,
            beta,
            c_ref.as_mut_ptr(),
            c_rs,
            c_cs,
        );
    }

    let c_ref_ptr = c_ref.as_mut_ptr();
    if c_rs == 1 {
        for j in 0..n {
            unary(c_ref_ptr.add(j * c_cs), m);
        }
    } else if c_cs == 1 {
        for i in 0..m {
            unary(c_ref_ptr.add(i * c_rs), n);
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                unary(c_ref_ptr.add(i * c_rs + j * c_cs), 1);
            }
        }
    }

    let diff = max_abs_diff(&c, &c_ref, eps);
    return diff;
}

pub unsafe fn check_gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    a_rs: usize,
    a_cs: usize,
    b: *const f32,
    b_rs: usize,
    b_cs: usize,
    beta: f32,
    c: &[f32],
    c_rs: usize,
    c_cs: usize,
    c_ref: &mut [f32],
    unary: unsafe fn(*mut f32, m: usize),
    eps: f64,
) -> f64 {
    #[cfg(feature = "mkl")]
    {
        let (layout, transa, transb, lda, ldb, ldc) =
            stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        cblas_sgemm(
            layout,
            transa,
            transb,
            m as c_int,
            n as c_int,
            k as c_int,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c_ref.as_mut_ptr(),
            ldc,
        );
    }
    #[cfg(not(feature = "mkl"))]
    {
        // calculate diff using fallback
        gemm_fallback_f32(
            m,
            n,
            k,
            alpha,
            a,
            a_rs,
            a_cs,
            b,
            b_rs,
            b_cs,
            beta,
            c_ref.as_mut_ptr(),
            c_rs,
            c_cs,
        );
    }
    let c_ref_ptr = c_ref.as_mut_ptr();
    if c_rs == 1 {
        for j in 0..n {
            unary(c_ref_ptr.add(j * c_cs), m);
        }
    } else if c_cs == 1 {
        for i in 0..m {
            unary(c_ref_ptr.add(i * c_rs), n);
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                unary(c_ref_ptr.add(i * c_rs + j * c_cs), 1);
            }
        }
    }

    let diff = max_abs_diff(&c, &c_ref, eps);
    return diff;
}

// pub unsafe fn check_gemm_f16(
// 	m: usize, n: usize, k: usize,
// 	alpha: f16,
// 	a: *const f16, a_rs: usize, a_cs: usize,
// 	b: *const f16, b_rs: usize, b_cs: usize,
// 	beta: f16,
// 	c: &[f16], c_rs: usize, c_cs: usize,
//     c_ref: &mut [f16],
//     eps: f64,
// ) -> f64 {
//     #[cfg(feature="mkl")] {
//         let (layout, transa, transb, lda, ldb, ldc) = stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
//         let a = a as *const u16;
//         let b = b as *const u16;
//         let c_ref_ptr = c_ref.as_mut_ptr() as *mut u16;
//         let alpha = alpha.to_bits();
//         let beta = beta.to_bits();
//         cblas_hgemm(
//             layout, transa, transb, m as c_int, n as c_int, k as c_int, alpha, a, lda, b, ldb, beta, c_ref_ptr, ldc
//         );
//         let diff = max_abs_diff(&c, &c_ref, eps);
//         return diff;
//     }
//     #[cfg(not(feature="mkl"))] {
//         // calculate diff using fallback
//         gemm_fallback_f16(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c_ref.as_mut_ptr(), c_rs, c_cs);
//         let diff = max_abs_diff(&c, &c_ref, eps);
//         return diff;
//     }

// }

pub unsafe fn check_gemm_c32(
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: *const Complex32,
    a_rs: usize,
    a_cs: usize,
    b: *const Complex32,
    b_rs: usize,
    b_cs: usize,
    beta: Complex32,
    c: &[Complex32],
    c_rs: usize,
    c_cs: usize,
    c_ref: &mut [Complex32],
    unary: unsafe fn(*mut Complex32, m: usize),
    eps: f64,
) -> f64 {
    #[cfg(feature = "mkl")]
    {
        let (layout, transa, transb, lda, ldb, ldc) =
            stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        let a = a as *const c_void;
        let b = b as *const c_void;
        let c_ref_ptr = c_ref.as_mut_ptr() as *mut c_void;
        let alpha_ptr = &alpha as *const Complex32 as *const c_void;
        let beta_ptr = &beta as *const Complex32 as *const c_void;
        cblas_cgemm(
            layout, transa, transb, m as c_int, n as c_int, k as c_int, alpha_ptr, a, lda, b, ldb,
            beta_ptr, c_ref_ptr, ldc,
        );
    }
    #[cfg(not(feature = "mkl"))]
    {
        // calculate diff using fallback
        gemm_fallback_c32(
            m,
            n,
            k,
            alpha,
            a,
            a_rs,
            a_cs,
            b,
            b_rs,
            b_cs,
            beta,
            c_ref.as_mut_ptr(),
            c_rs,
            c_cs,
        );
    }
    let c_ref_ptr = c_ref.as_mut_ptr();
    if c_rs == 1 {
        for j in 0..n {
            unary(c_ref_ptr.add(j * c_cs), m);
        }
    } else if c_cs == 1 {
        for i in 0..m {
            unary(c_ref_ptr.add(i * c_rs), n);
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                unary(c_ref_ptr.add(i * c_rs + j * c_cs), 1);
            }
        }
    }

    let diff = max_abs_diff(&c, &c_ref, eps);
    return diff;
}

pub unsafe fn check_gemm_c64(
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: *const Complex64,
    a_rs: usize,
    a_cs: usize,
    b: *const Complex64,
    b_rs: usize,
    b_cs: usize,
    beta: Complex64,
    c: &[Complex64],
    c_rs: usize,
    c_cs: usize,
    c_ref: &mut [Complex64],
    unary: unsafe fn(*mut Complex64, m: usize),
    eps: f64,
) -> f64 {
    #[cfg(feature = "mkl")]
    {
        let (layout, transa, transb, lda, ldb, ldc) =
            stride_to_cblas(m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs);
        let a = a as *const c_void;
        let b = b as *const c_void;
        let c_ref_ptr = c_ref.as_mut_ptr() as *mut c_void;
        let alpha_ptr = &alpha as *const Complex64 as *const c_void;
        let beta_ptr = &beta as *const Complex64 as *const c_void;
        cblas_zgemm(
            layout, transa, transb, m as c_int, n as c_int, k as c_int, alpha_ptr, a, lda, b, ldb,
            beta_ptr, c_ref_ptr, ldc,
        );
    }
    #[cfg(not(feature = "mkl"))]
    {
        // calculate diff using fallback
        gemm_fallback_c64(
            m,
            n,
            k,
            alpha,
            a,
            a_rs,
            a_cs,
            b,
            b_rs,
            b_cs,
            beta,
            c_ref.as_mut_ptr(),
            c_rs,
            c_cs,
        );
    }

    let c_ref_ptr = c_ref.as_mut_ptr();
    if c_rs == 1 {
        for j in 0..n {
            unary(c_ref_ptr.add(j * c_cs), m);
        }
    } else if c_cs == 1 {
        for i in 0..m {
            unary(c_ref_ptr.add(i * c_rs), n);
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                unary(c_ref_ptr.add(i * c_rs + j * c_cs), 1);
            }
        }
    }

    let diff = max_abs_diff(&c, &c_ref, eps);
    return diff;
}

pub fn cblas_params_from_str(
    layout_str: &str,
    m: usize,
    n: usize,
    k: usize,
) -> (i32, i32, i32, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE) {
    if layout_str == "nn" {
        (m as i32, k as i32, m as i32, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans)
    } else if layout_str == "nt" {
        (m as i32, n as i32, m as i32, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans)
    } else if layout_str == "tn" {
        (k as i32, k as i32, m as i32, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans)
    } else if layout_str == "tt" {
        (k as i32, n as i32, m as i32, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasTrans)
    } else {
        panic!("Unsupported layout str");
    }
}

pub fn generate_m_dims(mc: usize, mr: usize) -> Vec<usize> {
    // return vec![1, 47, 101, 1201];
    let mut a_dims = vec![];
    for m in 1..mr {
        a_dims.push(m);
        a_dims.push(m + 100);
        // a_dims.push(m+mc);
    }
    a_dims.push(mc + 29);
    a_dims
}

pub fn generate_n_dims(nc: usize, nr: usize) -> Vec<usize> {
    // return vec![1, 17, 47, 101, 901];
    let mut a_dims = vec![];
    for n in 1..nr {
        a_dims.push(n);
        a_dims.push(n + 200);
        a_dims.push(n + nc);
    }
    a_dims
}
// kr does not really exist, it is to have the same patter as other dims, also
// it might be also be thought of as being tested against k_unrolling parameter
pub fn generate_k_dims(kc: usize, kr: usize) -> Vec<usize> {
    // return vec![1, 17, 47, 101, 901]
    let mut a_dims = vec![];
    let kr = 8;
    for k in 1..kr {
        a_dims.push(k);
        a_dims.push(k + 50);
        a_dims.push(k + kc);
    }
    a_dims
}
