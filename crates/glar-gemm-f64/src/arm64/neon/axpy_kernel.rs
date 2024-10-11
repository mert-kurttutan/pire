use std::arch::aarch64::*;
use seq_macro::seq;

use crate::{TA, TB, TC};

const VS: usize = 2;


#[inline(always)]
unsafe fn store_v(dst: *mut TC, src: float64x2_t) {
    vst1q_f64(dst, src);
}

#[inline(always)]
unsafe fn load_v(src: *const TC) -> float64x2_t {
    vld1q_f64(src)
}

#[inline(always)]
unsafe fn zero_v() -> float64x2_t {
    vdupq_n_f64(0.0)
}

#[inline(always)]
unsafe fn scale(v: float64x2_t, alpha: TC) -> float64x2_t {
    vmulq_n_f64(v, alpha)
}


pub(crate) struct AxpyKernel {
    pub(crate) acc: [float64x2_t; 2],
}

impl AxpyKernel {
    #[inline(always)]
    pub(crate) unsafe fn new() -> Self {
        AxpyKernel {
            acc: [zero_v(); 2],
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn set_zero(&mut self) {
        self.acc = [zero_v(); 2];
    }

    #[inline(always)]
    pub(crate) unsafe fn set_zero_s(&mut self) {
        self.acc[0] = zero_v();
    }

    #[inline(always)]
    pub(crate) unsafe fn accumulate(&mut self, i: usize, lda: usize, a_cur: *const TC, x: TC) {
        let a_v_0 = load_v(a_cur.add(i * lda));
        let a_v_1 = load_v(a_cur.add(i * lda + VS));

        self.acc[0] = vfmaq_n_f64(self.acc[0], a_v_0, x);
        self.acc[1] = vfmaq_n_f64(self.acc[1], a_v_1, x);
    }

    #[inline(always)]
    pub(crate) unsafe fn accumulate_s(&mut self, i: usize, lda: usize, a_cur: *const TC, x: TC) {
        let a_v_0 = load_v(a_cur.add(i * lda));

        self.acc[0] = vfmaq_n_f64(self.acc[0], a_v_0, x);
    }

    #[inline(always)]
    pub(crate) unsafe fn scale(&mut self, alpha: TC) {
        if alpha != 1.0 {
            self.acc[0] = scale(self.acc[0], alpha);
            self.acc[1] = scale(self.acc[1], alpha);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn scale_s(&mut self, alpha: TC) {
        if alpha != 1.0 {
            self.acc[0] = scale(self.acc[0], alpha);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn beta_zero(&self, y_cur: *mut TC) {
        store_v(y_cur, self.acc[0]);
        store_v(y_cur.add(VS), self.acc[1]);
    }

    #[inline(always)]
    pub(crate) unsafe fn beta_zero_s(&self, y_cur: *mut TC) {
        store_v(y_cur, self.acc[0]);
    }

    #[inline(always)]
    pub(crate) unsafe fn beta_one(&self, y_cur: *mut TC) {
        let y_v_0 = load_v(y_cur);
        let y_v_0 = vaddq_f64(y_v_0, self.acc[0]);
        store_v(y_cur, y_v_0);

        let y_v_1 = load_v(y_cur.add(VS));
        let y_v_1 = vaddq_f64(y_v_1, self.acc[1]);
        store_v(y_cur.add(VS), y_v_1);
    }

    #[inline(always)]
    pub(crate) unsafe fn beta_one_s(&self, y_cur: *mut TC) {
        let y_v_0 = load_v(y_cur);
        let y_v_0 = vaddq_f64(y_v_0, self.acc[0]);
        store_v(y_cur, y_v_0);
    }

    #[inline(always)]
    pub(crate) unsafe fn beta(&self, y_cur: *mut TC, beta_t: TC) {
        let mut y_v_0 = load_v(y_cur);
        y_v_0 = scale(y_v_0, beta_t);
        y_v_0 = vaddq_f64(y_v_0, self.acc[0]);
        store_v(y_cur, y_v_0);

        let mut y_v_1 = load_v(y_cur.add(VS));
        y_v_1 = scale(y_v_1, beta_t);
        y_v_1 = vaddq_f64(y_v_1, self.acc[1]);
        store_v(y_cur.add(VS), y_v_1);
    }

    #[inline(always)]
    pub(crate) unsafe fn beta_s(&self, y_cur: *mut TC, beta_t: TC) {
        let mut y_v_0 = load_v(y_cur);
        y_v_0 = scale(y_v_0, beta_t);
        y_v_0 = vaddq_f64(y_v_0, self.acc[0]);
        store_v(y_cur, y_v_0);
    }

    #[inline(always)]
    pub(crate) unsafe fn kernel_scalar(
        &self,
        k: usize,
        mi: usize, m: usize, lda: usize, 
        alpha_t: TC, beta_t: TC,
        a_cur: *const TC, x_v_ptr: *const TC,
        y: *mut TC,
    ) {
        let mut mi = mi;
        while mi < m {
            let y_cur = y.add(mi);
            let a_cur = a_cur.add(mi);
            let mut acc = 0.0;

            for j in 0..k {
                acc += *a_cur.add(j*lda) * *x_v_ptr.add(j);
            }

            acc *= alpha_t;

            if beta_t == 0.0 {
                *y_cur = acc;
            } else if beta_t == 1.0 {
                *y_cur = *y_cur + acc;
            } else {
                *y_cur = *y_cur * beta_t + acc;
            }

            mi += 1;
        }
    }
}


#[target_feature(enable = "neon")]
pub(crate) unsafe fn axpy_v(
    m: usize,
    n: usize,
    alpha: *const TA,
    a: *const TA,
    lda: usize,
    x: *const TB,
    incx: usize,
    beta: *const TC,
    y: *mut TC,
) {
    const K_UNROLL: usize = VS;
    const M_UNROLL: usize = 2;
    let mut axpy_ker = AxpyKernel::new();
    let mut beta_t = *beta;
    let alpha_t = *alpha;
    let mut ni = 0;
    let n_lane = n / K_UNROLL * K_UNROLL;
    let m_lane_unroll = m / (M_UNROLL * VS) * (M_UNROLL * VS);
    let m_lane = m / VS * VS;
    while ni < n_lane {
        let a_cur = a.add(ni * lda);
        let x_cur = x.add(ni * incx);
        let x_v = [*x_cur, *x_cur.add(incx), *x_cur.add(incx * 2), *x_cur.add(incx * 3)];

        let mut mi = 0;
        while mi < m_lane_unroll {
            let y_cur = y.add(mi);
            let a_cur = a_cur.add(mi);
            // set acc to zero
            axpy_ker.set_zero();
            seq!(j in 0..2 {
                axpy_ker.accumulate(j, lda, a_cur, x_v[j]);
            });
            axpy_ker.scale(alpha_t);

            if beta_t == 0.0 {
                axpy_ker.beta_zero(y_cur);
            } else if beta_t == 1.0 {
                axpy_ker.beta_one(y_cur);
            } else {
                axpy_ker.beta(y_cur, beta_t);
            }

            mi += M_UNROLL * VS;
        }

        while mi < m_lane {
            let y_cur = y.add(mi);
            let a_cur = a_cur.add(mi);
            // set acc to zero
            axpy_ker.set_zero_s();
            seq!(j in 0..2 {
                axpy_ker.accumulate_s(j, lda, a_cur, x_v[j]);
            });
            axpy_ker.scale_s(alpha_t);

            if beta_t == 0.0 {
                axpy_ker.beta_zero_s(y_cur);
            } else if beta_t == 1.0 {
                axpy_ker.beta_one_s(y_cur);
            } else {
                axpy_ker.beta_s(y_cur, beta_t);
            }

            mi += VS;
        }

        axpy_ker.kernel_scalar(
            K_UNROLL, mi, m, lda, alpha_t, beta_t, a_cur, x_v.as_ptr(), y
        );

        beta_t = 1.0;
        ni += K_UNROLL;
    }

    while ni < n {
        let a_cur = a.add(ni * lda);
        let x_cur = x.add(ni * incx);
        let x_cur_0 = *x_cur;

        let mut mi = 0;
        while mi < m_lane_unroll {
            let y_cur = y.add(mi);
            let a_cur = a_cur.add(mi);
            // set acc to zero
            axpy_ker.set_zero();
            axpy_ker.accumulate(0, lda, a_cur, x_cur_0);
            axpy_ker.scale(alpha_t);

            if beta_t == 0.0 {
                axpy_ker.beta_zero(y_cur);
            } else if beta_t == 1.0 {
                axpy_ker.beta_one(y_cur);
            } else {
                axpy_ker.beta(y_cur, beta_t);
            }

            mi += M_UNROLL * VS;
        }

        while mi < m_lane {
            let y_cur = y.add(mi);
            let a_cur = a_cur.add(mi);
            // set acc to zero
            axpy_ker.set_zero_s();
            axpy_ker.accumulate_s(0, lda, a_cur, x_cur_0);
            axpy_ker.scale_s(alpha_t);

            if beta_t == 0.0 {
                axpy_ker.beta_zero_s(y_cur);
            } else if beta_t == 1.0 {
                axpy_ker.beta_one_s(y_cur);
            } else {
                axpy_ker.beta_s(y_cur, beta_t);
            }

            mi += VS;
        }

        // scalar impl
        axpy_ker.kernel_scalar(
            1, mi, m, lda, alpha_t, beta_t, a_cur, x_cur, y
        );

        beta_t = 1.0;
        ni += 1;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn axpy_d(
    m: usize,
    n: usize,
    alpha: *const TA,
    a: *const TA,
    lda: usize,
    x: *const TB,
    beta: *const TC,
    y: *mut TC,
    incy: usize,
) {
    let alpha_t = *alpha;
    let beta_t = *beta;
    const K_UNROLL: usize = 1;
    const M_UNROLL: usize = 2;
    let m_unroll = m / M_UNROLL * M_UNROLL;
    let n_vec = n / VS * VS;
    
    // accumulate the vectorized part to scalar f64
    let mut acc_arr = [0.0; VS];
    let mut i = 0;
    while i < m_unroll {
        let y_cur = y.add(i * incy);
        let a_cur = a.add(i * lda);
        let mut acc_s = [0.0; M_UNROLL];

        let mut acc_v = [zero_v(); M_UNROLL*K_UNROLL];

        let mut j = 0;
        while j < n_vec {
            let a_cur = a_cur.add(j);
            let x_cur = x.add(j);
            let x_v_0 = load_v(x_cur);

            seq!(p in 0..2 {
                let a_p = load_v(a_cur.add(p * lda));
                acc_v[p] = vfmaq_f64(acc_v[p], a_p, x_v_0);
            });

            j += VS * K_UNROLL;
        }

        acc_s[0] = 0.0;
        acc_s[1] = 0.0;
        while j < n {
            let a_cur = a_cur.add(j);
            let x_cur = x.add(j);
            acc_s[0] += *a_cur * *x_cur;
            acc_s[1] += *a_cur.add(lda) * *x_cur;
            j += 1;
        }

        seq!(p in 0..2 {
            vst1q_f64(acc_arr.as_mut_ptr(), acc_v[p]);
            acc_s[p] += acc_arr[0] + acc_arr[1];
            acc_s[p] *= alpha_t;
        });

        if beta_t == 0.0 {
            seq!(p in 0..2 {
                *y_cur.add(incy * p) = acc_s[p];
            });
        } else if beta_t == 1.0 {
            seq!(p in 0..2 {
                *y_cur.add(incy * p) = *y_cur.add(incy * p) + acc_s[p];
            });
        } else {
            seq!(p in 0..2 {
                *y_cur.add(incy * p) = *y_cur.add(incy * p) * beta_t + acc_s[p];
            });
        }

        i += M_UNROLL;
    }


    while i < m {
        let y_cur = y.add(i * incy);
        let a_cur = a.add(i * lda);

        let mut acc_v = [zero_v(); 1*K_UNROLL];

        let mut j = 0;
        while j < n_vec {
            let a_cur = a_cur.add(j);
            let x_cur = x.add(j);
            let x_v_0 = load_v(x_cur);

            let a_p = load_v(a_cur);
            acc_v[0] = vfmaq_f64(acc_v[0], a_p, x_v_0);

            j += VS * K_UNROLL;
        }

        let mut acc_s = 0.0;
        while j < n {
            let a_cur = a_cur.add(j);
            let x_cur = x.add(j);
            acc_s += *a_cur * *x_cur;
            j += 1;
        }

        vst1q_f64(acc_arr.as_mut_ptr(), acc_v[0]);
        acc_s += acc_arr[0] + acc_arr[1];
        acc_s *= alpha_t;

        if beta_t == 0.0 {
            *y_cur = acc_s;
        } else if beta_t == 1.0 {
            *y_cur = *y_cur + acc_s;
        } else {
            *y_cur = *y_cur * beta_t + acc_s;
        }

        i += 1;
    }
}




