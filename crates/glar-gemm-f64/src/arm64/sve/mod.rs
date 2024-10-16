#[rustfmt::skip]
pub mod asm_ukernel;
pub(crate) mod axpy_kernel;

pub(crate) use asm_ukernel::*;
pub(crate) use axpy_kernel::*;

// use paste::paste;

use crate::{TA, TB, TC};

// const VS: usize = 8;

use crate::MyFn;

#[target_feature(enable = "neon")]
pub unsafe fn axpy<F: MyFn>(
    m: usize,
    n: usize,
    alpha: *const TA,
    a: *const TA,
    a_rs: usize,
    a_cs: usize,
    x: *const TB,
    incx: usize,
    beta: *const TC,
    y: *mut TC,
    incy: usize,
    f: F,
) {
    if a_cs == 1 && incx == 1 {
        axpy_d(m, n, alpha, a, a_rs, x, beta, y, incy);
        for i in 0..m {
            f.call(y.add(i * incy), 1);
        }
        return;
    }
    if a_rs == 1 && incy == 1 {
        axpy_v(m, n, alpha, a, a_cs, x, incx, beta, y);
        f.call(y, m);
        return;
    }

    for i in 0..m {
        let y_cur = y.add(i * incy);
        let mut acc = 0.0;
        for j in 0..n {
            let a_cur = a.add(i * a_rs + j * a_cs);
            let x_cur = x.add(j * incx);
            acc += *a_cur * *x_cur;
        }
        *y_cur = *beta * *y_cur + *alpha * acc;
        f.call(y_cur, 1);
    }
}

pub unsafe fn kernel_bb<F: MyFn, const STRIDED: bool>(
    m: usize, n: usize, k: usize,
    alpha: *const TA,
    beta: *const TC,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap: *const TA, bp: *const TB,
    mr: usize, nr: usize,
    f: F,
) {
    let m_rounded = m / mr * mr;
    let n_rounded = n / nr * nr;
    let m_left = m % mr;
    let n_left = n % nr;

    let d_arr = [0, 0, c_rs, c_cs];

    let mut m_i = 0;
    while m_i < m_rounded {
        let c_cur0 = c.add(m_i * c_rs);
        let ap_cur = ap.add(m_i * k);
        let mut n_i = 0;
        while n_i < n_rounded {
            let bp_cur = bp.add(n_i * k);
            let c_cur1 = c_cur0.add(n_i * c_cs);
            ukernel_12x8_bb::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, mr, f);
            n_i += nr;
        }
        if n_left != 0 {
            let bp_cur = bp.add(n_i * k);
            let c_cur1 = c_cur0.add(n_i * c_cs);
            ukernel_12xn_bb::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, mr, n_left, f);
        }
        m_i += mr;
    }

    if m_left != 0 {
        let c_cur0 = c.add(m_i * c_rs);
        let ap_cur = ap.add(m_i * k);
        let mut n_i = 0;
        while n_i < n_rounded {
            let bp_cur = bp.add(n_i * k);
            let c_cur1 = c_cur0.add(n_i * c_cs);
            ukernel_12x8_bb_partial::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, m_left, f);
            n_i += nr;
        }
        if n_left != 0 {
            let bp_cur = bp.add(n_i * k);
            let c_cur1 = c_cur0.add(n_i * c_cs);
            ukernel_12xn_bb_partial::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, m_left, n_left, f);
        }
    }
}

use super::pack_sve::packa_panel;

pub unsafe fn kernel_sb_v0<F: MyFn, const STRIDED: bool>(
    m: usize, n: usize, k: usize,
    alpha: *const TA, beta: *const TC,
    a: *const TA, a_rs: usize, a_cs: usize,
    bp: *const TB,
    c: *mut TC, c_rs: usize, c_cs: usize,
    ap: *mut TA,
    mr: usize, nr: usize,
    f: F,
) {
    let k_eff = k;
    let m_rounded = m / mr * mr;
    let n_rounded = n / nr * nr;
    let m_left = m % mr;
    let n_left = n % nr;

    let d_arr = [0, 0, c_rs, c_cs];

    let mut m_i = 0;
    while m_i < m_rounded {
        let c_cur0 = c.add(m_i * c_rs);
        let a_cur = a.add(m_i * a_rs);
        packa_panel(mr, k, a_cur, a_rs, a_cs, ap, mr, mr);
        let mut n_i = 0;
        while n_i < n_rounded {
            let bp_cur = bp.add(n_i * k_eff);
            let c_cur1 = c_cur0.add(n_i * c_cs);
            ukernel_12x8_bb::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, mr, f);
            n_i += nr;
        }
        if n_left != 0 {
            let bp_cur = bp.add(n_i * k_eff);
            let c_cur1 = c_cur0.add(n_i * c_cs);
            ukernel_12xn_bb::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, mr, n_left, f);
        }
        m_i += mr;
    }

    if m_left != 0 {
        let c_cur0 = c.add(m_i * c_rs);
        let a_cur = a.add(m_i * a_rs);
        packa_panel(m_left, k, a_cur, a_rs, a_cs, ap, mr, mr);
        let mut n_i = 0;
        while n_i < n_rounded {
            let bp_cur = bp.add(n_i * k_eff);
            let c_cur1 = c_cur0.add(n_i * c_cs);
            ukernel_12x8_bb_partial::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, m_left, f);
            n_i += nr;
        }
        if n_left != 0 {
            let bp_cur = bp.add(n_i * k_eff);
            let c_cur1 = c_cur0.add(n_i * c_cs);
            ukernel_12xn_bb_partial::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, m_left, n_left, f);
        }
    }
}

// #[target_feature(enable = "neon")]
pub(crate) unsafe fn kernel_sb<F: MyFn>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    beta: *const TC,
    a: *const TB,
    a_rs: usize,
    a_cs: usize,
    b: *const TB,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap_buf: *mut TA,
    mr: usize, nr: usize,
    f: F,
) {
    if c_rs == 1 {
        kernel_sb_v0::<_, false>(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, mr, nr, f);
    } else {
        kernel_sb_v0::<_, true>(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, mr, nr, f);
    }
}

pub(crate) unsafe fn kernel<F: MyFn>(
    m: usize,
    n: usize,
    k: usize,
    alpha: *const TA,
    beta: *const TC,
    c: *mut TC,
    c_rs: usize,
    c_cs: usize,
    ap: *const TA,
    bp: *const TB,
    mr: usize, nr: usize,
    f: F,
) {
    if c_rs == 1 {
        kernel_bb::<_, false>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, mr, nr, f)
    } else {
        kernel_bb::<_, true>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, mr, nr, f)
    }
}
