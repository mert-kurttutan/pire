use std::arch::aarch64::*;
use seq_macro::seq;
use std::arch::asm;

use crate::{TA, TB, TC};

const VS: usize = 8;

#[allow(non_camel_case_types)]
pub(crate) type float16x8_t = uint16x8_t;

#[target_feature(enable = "neon,fp16")]
#[inline]
unsafe fn store_v(dst: *mut TC, src: float16x8_t) {
    vst1q_u16(dst as *mut u16, src);
}

#[target_feature(enable = "neon,fp16")]
#[inline]
unsafe fn load_v(src: *const TC) -> float16x8_t {
    vld1q_u16(src as *const u16)
}

#[inline(always)]
unsafe fn zero_v() -> float16x8_t {
    let mut x0: float16x8_t;
    asm!(
        "dup {x0:v}.8h, wzr",
        x0 = out (vreg) x0,
        // out("wzr") _,
    );
    x0
}

#[target_feature(enable = "neon,fp16")]
#[inline]
unsafe fn scale(mut v: float16x8_t, alpha: TC) -> float16x8_t {
    asm!(
        "ldr {x2:h}, [{alphax}]",
        "fmul {x1:v}.8h, {x1:v}.8h, {x2}.8h",
        alphax = in(reg) &alpha,
        x1 = inout(vreg) v,
        x2 = out(vreg) _,
    );
    v
}

#[target_feature(enable = "neon,fp16")]
#[inline]
unsafe fn fma(a: float16x8_t, b: float16x8_t, mut c: float16x8_t) -> float16x8_t {
    asm!(
        "fmla {cx:v}.8h, {ax:v}.8h, {bx:v}.8h",
        ax = in(vreg) a,
        bx = in(vreg) b,
        cx = inout(vreg) c,
    );
    c
}

#[target_feature(enable = "neon,fp16")]
#[inline]
unsafe fn fma_by_scalar(a: float16x8_t, b: &TC, mut c: float16x8_t) -> float16x8_t {
    asm!(
        "ldr {x2:h}, [{bx}]",
        "fmla {cx:v}.8h, {ax:v}.8h, {x2}.h[0]",
        ax = in(vreg) a,
        bx = in(reg) b,
        cx = inout(vreg) c,
        x2 = out(vreg) _,
    );
    c
}
#[target_feature(enable = "neon,fp16")]
#[inline]
unsafe fn add_inplace(mut a: float16x8_t, b: float16x8_t) -> float16x8_t {
    asm!(
        "fadd {ax:v}.8h, {ax:v}.8h, {bx:v}.8h",
        ax = inout(vreg) a,
        bx = in(vreg) b,
    );

    a
}


pub(crate) struct AxpyKernel {
    pub(crate) acc: [float16x8_t; 2],
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

        self.acc[0] = fma_by_scalar(a_v_0, &x, self.acc[0]);
        self.acc[1] = fma_by_scalar(a_v_1, &x, self.acc[1]);
    }

    #[inline(always)]
    pub(crate) unsafe fn accumulate_s(&mut self, i: usize, lda: usize, a_cur: *const TC, x: TC) {
        let a_v_0 = load_v(a_cur.add(i * lda));

        self.acc[0] = fma_by_scalar(a_v_0, &x, self.acc[0]);
    }

    #[inline(always)]
    pub(crate) unsafe fn scale(&mut self, alpha: TC) {
        if alpha != TC::ONE {
            self.acc[0] = scale(self.acc[0], alpha);
            self.acc[1] = scale(self.acc[1], alpha);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn scale_s(&mut self, alpha: TC) {
        if alpha != TC::ONE {
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
        let mut y_v_0 = load_v(y_cur);
        y_v_0 = add_inplace(y_v_0, self.acc[0]);
        store_v(y_cur, y_v_0);

        let mut y_v_1 = load_v(y_cur.add(VS));
        y_v_1 = add_inplace(y_v_1, self.acc[1]);
        store_v(y_cur.add(VS), y_v_1);
    }

    #[inline(always)]
    pub(crate) unsafe fn beta_one_s(&self, y_cur: *mut TC) {
        let mut y_v_0 = load_v(y_cur);
        y_v_0 = add_inplace(y_v_0, self.acc[0]);
        store_v(y_cur, y_v_0);
    }

    #[inline(always)]
    pub(crate) unsafe fn beta(&self, y_cur: *mut TC, beta_t: TC) {
        let mut y_v_0 = load_v(y_cur);
        y_v_0 = scale(y_v_0, beta_t);
        y_v_0 = add_inplace(y_v_0, self.acc[0]);
        store_v(y_cur, y_v_0);

        let mut y_v_1 = load_v(y_cur.add(VS));
        y_v_1 = scale(y_v_1, beta_t);
        y_v_1 = add_inplace(y_v_1, self.acc[1]);
        store_v(y_cur.add(VS), y_v_1);
    }

    #[inline(always)]
    pub(crate) unsafe fn beta_s(&self, y_cur: *mut TC, beta_t: TC) {
        let mut y_v_0 = load_v(y_cur);
        y_v_0 = scale(y_v_0, beta_t);
        y_v_0 = add_inplace(y_v_0, self.acc[0]);
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
            let mut acc = TC::ZERO;

            for j in 0..k {
                acc += *a_cur.add(j*lda) * *x_v_ptr.add(j);
            }

            acc *= alpha_t;

            if beta_t == TC::ZERO {
                *y_cur = acc;
            } else if beta_t == TC::ONE {
                *y_cur = *y_cur + acc;
            } else {
                *y_cur = *y_cur * beta_t + acc;
            }

            mi += 1;
        }
    }
}


#[target_feature(enable = "neon,fp16")]
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
        let x_v = [
            *x_cur, *x_cur.add(incx), *x_cur.add(incx * 2), *x_cur.add(incx * 3),
            *x_cur.add(incx * 4), *x_cur.add(incx * 5), *x_cur.add(incx * 6), *x_cur.add(incx * 7),
        ];

        let mut mi = 0;
        while mi < m_lane_unroll {
            let y_cur = y.add(mi);
            let a_cur = a_cur.add(mi);
            // set acc to zero
            axpy_ker.set_zero();
            seq!(j in 0..8 {
                axpy_ker.accumulate(j, lda, a_cur, x_v[j]);
            });
            axpy_ker.scale(alpha_t);

            if beta_t == TC::ZERO {
                axpy_ker.beta_zero(y_cur);
            } else if beta_t == TC::ONE {
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
            seq!(j in 0..8 {
                axpy_ker.accumulate_s(j, lda, a_cur, x_v[j]);
            });
            axpy_ker.scale_s(alpha_t);

            if beta_t == TC::ZERO {
                axpy_ker.beta_zero_s(y_cur);
            } else if beta_t == TC::ONE {
                axpy_ker.beta_one_s(y_cur);
            } else {
                axpy_ker.beta_s(y_cur, beta_t);
            }

            mi += VS;
        }

        axpy_ker.kernel_scalar(
            K_UNROLL, mi, m, lda, alpha_t, beta_t, a_cur, x_v.as_ptr(), y
        );

        beta_t = TC::ONE;
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

            if beta_t == TC::ZERO {
                axpy_ker.beta_zero(y_cur);
            } else if beta_t == TC::ONE {
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

            if beta_t == TC::ZERO {
                axpy_ker.beta_zero_s(y_cur);
            } else if beta_t == TC::ONE {
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

        beta_t = TC::ONE;
        ni += 1;
    }
}


#[target_feature(enable = "neon,fp16")]
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
    
    // accumulate the vectorized part to scalar f32
    let mut acc_arr = [TC::ZERO; VS];
    let mut i = 0;
    while i < m_unroll {
        let y_cur = y.add(i * incy);
        let a_cur = a.add(i * lda);
        let mut acc_s = [TC::ZERO; M_UNROLL];

        let mut acc_v = [zero_v(); M_UNROLL*K_UNROLL];

        let mut j = 0;
        while j < n_vec {
            let a_cur = a_cur.add(j);
            let x_cur = x.add(j);
            let x_v_0 = load_v(x_cur);

            seq!(p in 0..2 {
                let a_p = load_v(a_cur.add(p * lda));
                acc_v[p] = fma(a_p, x_v_0, acc_v[p]);
            });

            j += VS * K_UNROLL;
        }

        acc_s[0] = TC::ZERO;
        acc_s[1] = TC::ZERO;
        while j < n {
            let a_cur = a_cur.add(j);
            let x_cur = x.add(j);
            acc_s[0] += *a_cur * *x_cur;
            acc_s[1] += *a_cur.add(lda) * *x_cur;
            j += 1;
        }

        seq!(p in 0..2 {
            store_v(acc_arr.as_mut_ptr(), acc_v[p]);
            acc_s[p] += acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3] + acc_arr[4] + acc_arr[5] + acc_arr[6] + acc_arr[7];
            acc_s[p] *= alpha_t;
        });

        if beta_t == TC::ZERO {
            seq!(p in 0..2 {
                *y_cur.add(incy * p) = acc_s[p];
            });
        } else if beta_t == TC::ONE {
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
            acc_v[0] = fma(a_p, x_v_0, acc_v[0]);

            j += VS * K_UNROLL;
        }

        let mut acc_s = TC::ZERO;
        while j < n {
            let a_cur = a_cur.add(j);
            let x_cur = x.add(j);
            acc_s += *a_cur * *x_cur;
            j += 1;
        }

        store_v(acc_arr.as_mut_ptr(), acc_v[0]);
        acc_s += acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3] + acc_arr[4] + acc_arr[5] + acc_arr[6] + acc_arr[7];
        acc_s *= alpha_t;

        if beta_t == TC::ZERO {
            *y_cur = acc_s;
        } else if beta_t == TC::ONE {
            *y_cur = *y_cur + acc_s;
        } else {
            *y_cur = *y_cur * beta_t + acc_s;
        }

        i += 1;
    }
}

