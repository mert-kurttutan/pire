use seq_macro::seq;
use std::ptr::copy_nonoverlapping;
use crate::{TA,TB};


use paste::paste;



#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn storeu_ps<const M: usize>(
    src: __m256, dst: *mut f32
) {
    let mut temp_arr = [0.0; 8];
    _mm256_storeu_ps(temp_arr.as_mut_ptr(), src);
    copy_nonoverlapping(temp_arr.as_ptr(), dst, M);
}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_c_kx6(
    k_iter: usize, k_left: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 5;
    const M1: usize = 4;
    const M2: usize = 1;
    while k_i < k_iter {
        let a0 = _mm256_loadu_ps(b);
        let a1 = _mm256_loadu_ps(b.add(ldb));
        let a2 = _mm256_loadu_ps(b.add(ldb*2));
        let a3 = _mm256_loadu_ps(b.add(ldb*3));

        // transpose
        let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
        let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
        let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

        let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        storeu_ps::<M1>(x0, bp);
        storeu_ps::<M1>(x0_h, bp.add(M*4));

        let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        storeu_ps::<M1>(x1, bp.add(M));
        storeu_ps::<M1>(x1_h, bp.add(M+M*4));

        let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        storeu_ps::<M1>(x2, bp.add(2*M));
        storeu_ps::<M1>(x2_h, bp.add(2*M+M*4));

        let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        storeu_ps::<M1>(x3, bp.add(3*M));
        storeu_ps::<M1>(x3_h, bp.add(3*M+M*4));



        let a0 = _mm256_loadu_ps(b.add(ldb*4));

        // transpose
        let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));

        let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        storeu_ps::<M2>(x0, bp);
        storeu_ps::<M2>(x0_h, bp.add(M*4));

        let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        storeu_ps::<M2>(x1, bp.add(M));
        storeu_ps::<M2>(x1_h, bp.add(M+M*4));

        let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        storeu_ps::<M2>(x2, bp.add(2*M));
        storeu_ps::<M2>(x2_h, bp.add(2*M+M*4));

        let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        storeu_ps::<M2>(x3, bp.add(3*M));
        storeu_ps::<M2>(x3_h, bp.add(3*M+M*4));

        b = b.add(8);
        bp = bp.add(M2*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        copy_packed::<1>(b.add(ldb*2), bp.add(2));
        copy_packed::<1>(b.add(ldb*3), bp.add(3));
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_c_kx5(
    k_iter: usize, k_left: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 5;
    const M1: usize = 4;
    const M2: usize = 1;
    while k_i < k_iter {
        let a0 = _mm256_loadu_ps(b);
        let a1 = _mm256_loadu_ps(b.add(ldb));
        let a2 = _mm256_loadu_ps(b.add(ldb*2));
        let a3 = _mm256_loadu_ps(b.add(ldb*3));

        // transpose
        let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
        let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
        let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

        let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        storeu_ps::<M1>(x0, bp);
        storeu_ps::<M1>(x0_h, bp.add(M*4));

        let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        storeu_ps::<M1>(x1, bp.add(M));
        storeu_ps::<M1>(x1_h, bp.add(M+M*4));

        let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        storeu_ps::<M1>(x2, bp.add(2*M));
        storeu_ps::<M1>(x2_h, bp.add(2*M+M*4));

        let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        storeu_ps::<M1>(x3, bp.add(3*M));
        storeu_ps::<M1>(x3_h, bp.add(3*M+M*4));



        let a0 = _mm256_loadu_ps(b.add(ldb*4));

        // transpose
        let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));

        let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        storeu_ps::<M2>(x0, bp);
        storeu_ps::<M2>(x0_h, bp.add(M*4));

        let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        storeu_ps::<M2>(x1, bp.add(M));
        storeu_ps::<M2>(x1_h, bp.add(M+M*4));

        let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        storeu_ps::<M2>(x2, bp.add(2*M));
        storeu_ps::<M2>(x2_h, bp.add(2*M+M*4));

        let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        storeu_ps::<M2>(x3, bp.add(3*M));
        storeu_ps::<M2>(x3_h, bp.add(3*M+M*4));

        b = b.add(8);
        bp = bp.add(M2*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        copy_packed::<1>(b.add(ldb*2), bp.add(2));
        copy_packed::<1>(b.add(ldb*3), bp.add(3));
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}



#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_c_kx4(
    k_iter: usize, k_left: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 4;

    while k_i < k_iter {
        let a0 = _mm256_loadu_ps(b);
        let a1 = _mm256_loadu_ps(b.add(ldb));
        let a2 = _mm256_loadu_ps(b.add(ldb*2));
        let a3 = _mm256_loadu_ps(b.add(ldb*3));

        // transpose
        let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
        let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
        let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

        let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        storeu_ps::<M>(x0, bp);
        storeu_ps::<M>(x0_h, bp.add(M*4));

        let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        storeu_ps::<M>(x1, bp.add(M));
        storeu_ps::<M>(x1_h, bp.add(M+M*4));

        let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        storeu_ps::<M>(x2, bp.add(2*M));
        storeu_ps::<M>(x2_h, bp.add(2*M+M*4));

        let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        storeu_ps::<M>(x3, bp.add(3*M));
        storeu_ps::<M>(x3_h, bp.add(3*M+M*4));

        b = b.add(8);
        bp = bp.add(M*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        copy_packed::<1>(b.add(ldb*2), bp.add(2));
        copy_packed::<1>(b.add(ldb*3), bp.add(3));
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_c_kx3(
    k_iter: usize, k_left: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 3;
    let a3 = _mm256_setzero_ps();
    while k_i < k_iter {
        let a0 = _mm256_loadu_ps(b);
        let a1 = _mm256_loadu_ps(b.add(ldb));
        let a2 = _mm256_loadu_ps(b.add(ldb*2));

        // transpose
        let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
        let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
        let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

        let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        storeu_ps::<M>(x0, bp);
        storeu_ps::<M>(x0_h, bp.add(M*4));

        let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        storeu_ps::<M>(x1, bp.add(M));
        storeu_ps::<M>(x1_h, bp.add(M+M*4));

        let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        storeu_ps::<M>(x2, bp.add(2*M));
        storeu_ps::<M>(x2_h, bp.add(2*M+M*4));

        let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        storeu_ps::<M>(x3, bp.add(3*M));
        storeu_ps::<M>(x3_h, bp.add(3*M+M*4));

        b = b.add(8);
        bp = bp.add(M*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        copy_packed::<1>(b.add(ldb*2), bp.add(2));
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_c_kx2(
    k_iter: usize, k_left: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    let t2 = _mm256_setzero_pd();
    let t3 = _mm256_setzero_pd();
    const M: usize = 2;
    while k_i < k_iter {
        let a0 = _mm256_loadu_ps(b);
        let a1 = _mm256_loadu_ps(b.add(ldb));

        // transpose
        let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));

        let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        storeu_ps::<M>(x0, bp);
        storeu_ps::<M>(x0_h, bp.add(M*4));

        let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        storeu_ps::<M>(x1, bp.add(M));
        storeu_ps::<M>(x1_h, bp.add(M+M*4));

        let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        storeu_ps::<M>(x2, bp.add(2*M));
        storeu_ps::<M>(x2_h, bp.add(2*M+M*4));

        let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        storeu_ps::<M>(x3, bp.add(3*M));
        storeu_ps::<M>(x3_h, bp.add(3*M+M*4));


        b = b.add(8);
        bp = bp.add(M*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<1>(b, bp);
        copy_packed::<1>(b.add(ldb), bp.add(1));
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}



#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_c_kx1(
    k_iter: usize, k_left: usize,
    b: *const TB, _ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    let t2 = _mm256_setzero_pd();
    let t3 = _mm256_setzero_pd();
    let a1 = _mm256_setzero_ps();
    const M: usize = 1;
    while k_i < k_iter {
        let a0 = _mm256_loadu_ps(b);

        // transpose
        let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
        let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));

        let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
        let x0_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x0, 1));

        storeu_ps::<M>(x0, bp);
        storeu_ps::<M>(x0_h, bp.add(M*4));

        let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
        let x1_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x1, 1));
        storeu_ps::<M>(x1, bp.add(M));
        storeu_ps::<M>(x1_h, bp.add(M+M*4));

        let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
        let x2_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x2, 1));
        storeu_ps::<M>(x2, bp.add(2*M));
        storeu_ps::<M>(x2_h, bp.add(2*M+M*4));

        let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));
        let x3_h = _mm256_castps128_ps256(_mm256_extractf128_ps(x3, 1));
        storeu_ps::<M>(x3, bp.add(3*M));
        storeu_ps::<M>(x3_h, bp.add(3*M+M*4));


        b = b.add(8);
        bp = bp.add(M*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<1>(b, bp);
        b = b.add(1);
        bp = bp.add(M);
        k_i += 1;
    }
}



#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_r_kx6(
    k_iter: usize, k_left: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 6;
    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb*2), bp.add(M*2));
        copy_packed::<M>(b.add(ldb*3), bp.add(M*3));
        copy_packed::<M>(b.add(ldb*4), bp.add(M*4));
        copy_packed::<M>(b.add(ldb*5), bp.add(M*5));
        copy_packed::<M>(b.add(ldb*6), bp.add(M*6));
        copy_packed::<M>(b.add(ldb*7), bp.add(M*7));
        b = b.add(8*ldb);
        bp = bp.add(M*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}



#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_r_kx5(
    k_iter: usize, k_left: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 5;
    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb*2), bp.add(M*2));
        copy_packed::<M>(b.add(ldb*3), bp.add(M*3));
        copy_packed::<M>(b.add(ldb*4), bp.add(M*4));
        copy_packed::<M>(b.add(ldb*5), bp.add(M*5));
        copy_packed::<M>(b.add(ldb*6), bp.add(M*6));
        copy_packed::<M>(b.add(ldb*7), bp.add(M*7));
        b = b.add(8*ldb);
        bp = bp.add(M*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}



#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_r_kx4(
    k_iter: usize, k_left: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 4;
    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb*2), bp.add(M*2));
        copy_packed::<M>(b.add(ldb*3), bp.add(M*3));
        copy_packed::<M>(b.add(ldb*4), bp.add(M*4));
        copy_packed::<M>(b.add(ldb*5), bp.add(M*5));
        copy_packed::<M>(b.add(ldb*6), bp.add(M*6));
        copy_packed::<M>(b.add(ldb*7), bp.add(M*7));
        b = b.add(8*ldb);
        bp = bp.add(M*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_r_kx3(
    k_iter: usize, k_left: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 3;

    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb*2), bp.add(M*2));
        copy_packed::<M>(b.add(ldb*3), bp.add(M*3));
        copy_packed::<M>(b.add(ldb*4), bp.add(M*4));
        copy_packed::<M>(b.add(ldb*5), bp.add(M*5));
        copy_packed::<M>(b.add(ldb*6), bp.add(M*6));
        copy_packed::<M>(b.add(ldb*7), bp.add(M*7));
        b = b.add(8*ldb);
        bp = bp.add(M*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_r_kx2(
    k_iter: usize, k_left: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 2;

    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb*2), bp.add(M*2));
        copy_packed::<M>(b.add(ldb*3), bp.add(M*3));
        copy_packed::<M>(b.add(ldb*4), bp.add(M*4));
        copy_packed::<M>(b.add(ldb*5), bp.add(M*5));
        copy_packed::<M>(b.add(ldb*6), bp.add(M*6));
        copy_packed::<M>(b.add(ldb*7), bp.add(M*7));
        b = b.add(8*ldb);
        bp = bp.add(M*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packb_r_kx1(
    k_iter: usize, k_left: usize,
    b: *const TB, ldb: usize,
    bp: *mut TB,
) {
    let mut b = b;
    let mut bp = bp;

    let mut k_i = 0;
    const M: usize = 1;

    while k_i < k_iter {
        copy_packed::<M>(b, bp);
        copy_packed::<M>(b.add(ldb), bp.add(M));
        copy_packed::<M>(b.add(ldb*2), bp.add(M*2));
        copy_packed::<M>(b.add(ldb*3), bp.add(M*3));
        copy_packed::<M>(b.add(ldb*4), bp.add(M*4));
        copy_packed::<M>(b.add(ldb*5), bp.add(M*5));
        copy_packed::<M>(b.add(ldb*6), bp.add(M*6));
        copy_packed::<M>(b.add(ldb*7), bp.add(M*7));
        b = b.add(8*ldb);
        bp = bp.add(M*8);
        k_i += 1;
    }

    k_i = 0;

    while k_i <  k_left {
        copy_packed::<M>(b, bp);
        b = b.add(ldb);
        bp = bp.add(M);
        k_i += 1;
    }
}



macro_rules! def_packb {
   ($nr:tt) => {
       seq!(NL in 1..$nr {
           paste! {
            #[target_feature(enable = "avx,fma")]
            pub(crate) unsafe fn [<pack_panel_$nr>](
                   n: usize, k: usize,
                   b: *const TB, b_rs: usize, b_cs: usize,
                   bp: *mut TB,
               ) {
                   let k_iter = k / 8;
                   let k_left = k % 8;
                   let mut bp = bp;
                   let mut b = b;
                   const NR: usize = $nr;
                   const NR_LAST_STEP: usize = $nr;
                   let mut n_idx = 0;
                   if b_cs == 1 {
                       let ldb = b_rs;
                       while n_idx + NR_LAST_STEP <= n {
                           [<packb_r_kx$nr>](k_iter, k_left, b, ldb, bp);
                           n_idx += NR;
                           bp = bp.add(k*NR);
                           b = b.add(NR);
                       }
                       let n_left = n - n_idx;
                       #(
                           if n_left == NL {
                               [<packb_r_kx~NL>](k_iter, k_left, b, ldb, bp);
                               return;
                           }
                       )*
                   } else if b_rs == 1 {
                       let ldb = b_cs;
                       while n_idx + NR_LAST_STEP <= n {
                           [<packb_c_kx$nr>](k_iter, k_left, b, ldb, bp);
                           n_idx += NR;
                           bp = bp.add(k*NR);
                           b =  b.add(ldb*NR);
                       }
                       let n_left = n - n_idx;
                       #(
                           if n_left == NL {
                               [<packb_c_kx~NL>](k_iter, k_left, b, ldb, bp);
                               return;
                           }
                       )*
                   }
               }   
           }
       });
   };
}


def_packb!(4);

def_packb!(6);


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packa_scalar_k<const MR: usize>(
    m_left: usize, k: usize,
    a: *const TA, a_rs: usize, a_cs: usize,
    ap: *mut TA,
) {
    for i in 0..m_left  {
        for j in 0..k {
            *ap.add(j*MR+i) = *a.add(j*a_cs + i*a_rs);
        }
    }
}

use std::arch::x86_64::*;

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn pack_t<const MR: usize>(
    a: *const TA, lda: usize,
    ap: *mut TB,
) {
    let a0 = _mm256_loadu_ps(a);
    let a1 = _mm256_loadu_ps(a.add(lda));
    let a2 = _mm256_loadu_ps(a.add(lda*2));
    let a3 = _mm256_loadu_ps(a.add(lda*3));

    // transpose
    let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
    let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
    let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
    let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

    let x0 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
    let x1 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
    let x2 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
    let x3 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));

    let a0 = _mm256_loadu_ps(a.add(lda*4));
    let a1 = _mm256_loadu_ps(a.add(lda*5));
    let a2 = _mm256_loadu_ps(a.add(lda*6));
    let a3 = _mm256_loadu_ps(a.add(lda*7));

    // transpose
    let t0 = _mm256_castps_pd(_mm256_unpacklo_ps(a0, a1));
    let t1 = _mm256_castps_pd(_mm256_unpackhi_ps(a0, a1));
    let t2 = _mm256_castps_pd(_mm256_unpacklo_ps(a2, a3));
    let t3 = _mm256_castps_pd(_mm256_unpackhi_ps(a2, a3));

    let x4 = _mm256_castpd_ps(_mm256_unpacklo_pd(t0, t2));
    let x5 = _mm256_castpd_ps(_mm256_unpackhi_pd(t0, t2));
    let x6 = _mm256_castpd_ps(_mm256_unpacklo_pd(t1, t3));
    let x7 = _mm256_castpd_ps(_mm256_unpackhi_pd(t1, t3));

    // exchange hi of x0 and lo of x4
    let x0_t = _mm256_permute2f128_ps(x0, x4, 0b0010_0000);
    let x4_t = _mm256_permute2f128_ps(x0, x4, 0b0011_0001);
    // exchange hi of x1 and lo of x5
    let x1_t = _mm256_permute2f128_ps(x1, x5, 0b0010_0000);
    let x5_t = _mm256_permute2f128_ps(x1, x5, 0b0011_0001);
    // exchange hi of x2 and lo of x6
    let x2_t = _mm256_permute2f128_ps(x2, x6, 0b0010_0000);
    let x6_t = _mm256_permute2f128_ps(x2, x6, 0b0011_0001);
    // exchange hi of x3 and lo of x7
    let x3_t = _mm256_permute2f128_ps(x3, x7, 0b0010_0000);
    let x7_t = _mm256_permute2f128_ps(x3, x7, 0b0011_0001);

    _mm256_store_ps(ap, x0_t);
    _mm256_store_ps(ap.add(MR), x1_t);
    _mm256_store_ps(ap.add(MR*2), x2_t);
    _mm256_store_ps(ap.add(MR*3), x3_t);
    _mm256_store_ps(ap.add(MR*4), x4_t);
    _mm256_store_ps(ap.add(MR*5), x5_t);
    _mm256_store_ps(ap.add(MR*6), x6_t);
    _mm256_store_ps(ap.add(MR*7), x7_t); 
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packa_r_kx24(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        pack_t::<24>(a, lda, ap);

        pack_t::<24>(a.add(8*lda), lda, ap.add(8));

        pack_t::<24>(a.add(16*lda), lda, ap.add(16));

        ap = ap.add(192);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<1>(a, ap);
        copy_packed::<1>(a.add(lda), ap.add(1));
        copy_packed::<1>(a.add(lda*2), ap.add(2));
        copy_packed::<1>(a.add(lda*3), ap.add(3));
        copy_packed::<1>(a.add(lda*4), ap.add(4));
        copy_packed::<1>(a.add(lda*5), ap.add(5));
        copy_packed::<1>(a.add(lda*6), ap.add(6));
        copy_packed::<1>(a.add(lda*7), ap.add(7));
        copy_packed::<1>(a.add(lda*8), ap.add(8));
        copy_packed::<1>(a.add(lda*9), ap.add(9));
        copy_packed::<1>(a.add(lda*10), ap.add(10));
        copy_packed::<1>(a.add(lda*11), ap.add(11));
        copy_packed::<1>(a.add(lda*12), ap.add(12));
        copy_packed::<1>(a.add(lda*13), ap.add(13));
        copy_packed::<1>(a.add(lda*14), ap.add(14));
        copy_packed::<1>(a.add(lda*15), ap.add(15));
        copy_packed::<1>(a.add(lda*16), ap.add(16));
        copy_packed::<1>(a.add(lda*17), ap.add(17));
        copy_packed::<1>(a.add(lda*18), ap.add(18));
        copy_packed::<1>(a.add(lda*19), ap.add(19));
        copy_packed::<1>(a.add(lda*20), ap.add(20));
        copy_packed::<1>(a.add(lda*21), ap.add(21));
        copy_packed::<1>(a.add(lda*22), ap.add(22));
        copy_packed::<1>(a.add(lda*23), ap.add(23));

        ap = ap.add(24);
        a = a.add(1);
        k_i += 1;
    }
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packa_r_kx16(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        pack_t::<16>(a, lda, ap);

        pack_t::<16>(a.add(8*lda), lda, ap.add(8));

        ap = ap.add(128);
        a = a.add(8);
        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<1>(a, ap);
        copy_packed::<1>(a.add(lda), ap.add(1));
        copy_packed::<1>(a.add(lda*2), ap.add(2));
        copy_packed::<1>(a.add(lda*3), ap.add(3));
        copy_packed::<1>(a.add(lda*4), ap.add(4));
        copy_packed::<1>(a.add(lda*5), ap.add(5));
        copy_packed::<1>(a.add(lda*6), ap.add(6));
        copy_packed::<1>(a.add(lda*7), ap.add(7));
        copy_packed::<1>(a.add(lda*8), ap.add(8));
        copy_packed::<1>(a.add(lda*9), ap.add(9));
        copy_packed::<1>(a.add(lda*10), ap.add(10));
        copy_packed::<1>(a.add(lda*11), ap.add(11));
        copy_packed::<1>(a.add(lda*12), ap.add(12));
        copy_packed::<1>(a.add(lda*13), ap.add(13));
        copy_packed::<1>(a.add(lda*14), ap.add(14));
        copy_packed::<1>(a.add(lda*15), ap.add(15));

        ap = ap.add(16);
        a = a.add(1);
        k_i += 1;
    }
}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packa_c_kx24(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        // use vector intrinscs
        let a0 = _mm256_loadu_ps(a);
        let a1 = _mm256_loadu_ps(a.add(8));
        let a2 = _mm256_loadu_ps(a.add(16));
        _mm256_store_ps(ap, a0);
        _mm256_store_ps(ap.add(8), a1);
        _mm256_store_ps(ap.add(16), a2);

        let a0 = _mm256_loadu_ps(a.add(lda));
        let a1 = _mm256_loadu_ps(a.add(lda+8));
        let a2 = _mm256_loadu_ps(a.add(lda+16));
        _mm256_store_ps(ap.add(24), a0);
        _mm256_store_ps(ap.add(32), a1);
        _mm256_store_ps(ap.add(40), a2);

        let a0 = _mm256_loadu_ps(a.add(lda*2));
        let a1 = _mm256_loadu_ps(a.add(lda*2+8));
        let a2 = _mm256_loadu_ps(a.add(lda*2+16));
        _mm256_store_ps(ap.add(48), a0);
        _mm256_store_ps(ap.add(56), a1);
        _mm256_store_ps(ap.add(64), a2);

        let a0 = _mm256_loadu_ps(a.add(lda*3));
        let a1 = _mm256_loadu_ps(a.add(lda*3+8));
        let a2 = _mm256_loadu_ps(a.add(lda*3+16));
        _mm256_store_ps(ap.add(72), a0);
        _mm256_store_ps(ap.add(80), a1);
        _mm256_store_ps(ap.add(88), a2);

        let a0 = _mm256_loadu_ps(a.add(lda*4));
        let a1 = _mm256_loadu_ps(a.add(lda*4+8));
        let a2 = _mm256_loadu_ps(a.add(lda*4+16));
        _mm256_store_ps(ap.add(96), a0);
        _mm256_store_ps(ap.add(104), a1);
        _mm256_store_ps(ap.add(112), a2);

        let a0 = _mm256_loadu_ps(a.add(lda*5));
        let a1 = _mm256_loadu_ps(a.add(lda*5+8));
        let a2 = _mm256_loadu_ps(a.add(lda*5+16));
        _mm256_store_ps(ap.add(120), a0);
        _mm256_store_ps(ap.add(128), a1);
        _mm256_store_ps(ap.add(136), a2);

        let a0 = _mm256_loadu_ps(a.add(lda*6));
        let a1 = _mm256_loadu_ps(a.add(lda*6+8));
        let a2 = _mm256_loadu_ps(a.add(lda*6+16));
        _mm256_store_ps(ap.add(144), a0);
        _mm256_store_ps(ap.add(152), a1);
        _mm256_store_ps(ap.add(160), a2);

        let a0 = _mm256_loadu_ps(a.add(lda*7));
        let a1 = _mm256_loadu_ps(a.add(lda*7+8));
        let a2 = _mm256_loadu_ps(a.add(lda*7+16));
        _mm256_store_ps(ap.add(168), a0);
        _mm256_store_ps(ap.add(176), a1);
        _mm256_store_ps(ap.add(184), a2);

        ap = ap.add(192);
        a = a.add(8*lda);

        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        let a0 = _mm256_loadu_ps(a);
        let a1 = _mm256_loadu_ps(a.add(8));
        let a2 = _mm256_loadu_ps(a.add(16));
        _mm256_store_ps(ap, a0);
        _mm256_store_ps(ap.add(8), a1);
        _mm256_store_ps(ap.add(16), a2);

        ap = ap.add(24);
        a = a.add(lda);
        k_i += 1;
    }

}

#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packa_c_kx16(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        // use vector intrinscs
        let a0 = _mm256_loadu_ps(a);
        let a1 = _mm256_loadu_ps(a.add(8));
        _mm256_store_ps(ap, a0);
        _mm256_store_ps(ap.add(8), a1);

        let a0 = _mm256_loadu_ps(a.add(lda));
        let a1 = _mm256_loadu_ps(a.add(lda+8));
        _mm256_store_ps(ap.add(16), a0);
        _mm256_store_ps(ap.add(24), a1);

        let a0 = _mm256_loadu_ps(a.add(lda*2));
        let a1 = _mm256_loadu_ps(a.add(lda*2+8));
        _mm256_store_ps(ap.add(32), a0);
        _mm256_store_ps(ap.add(40), a1);

        let a0 = _mm256_loadu_ps(a.add(lda*3));
        let a1 = _mm256_loadu_ps(a.add(lda*3+8));
        _mm256_store_ps(ap.add(48), a0);
        _mm256_store_ps(ap.add(56), a1);

        let a0 = _mm256_loadu_ps(a.add(lda*4));
        let a1 = _mm256_loadu_ps(a.add(lda*4+8));
        _mm256_store_ps(ap.add(64), a0);
        _mm256_store_ps(ap.add(72), a1);

        let a0 = _mm256_loadu_ps(a.add(lda*5));
        let a1 = _mm256_loadu_ps(a.add(lda*5+8));
        _mm256_store_ps(ap.add(80), a0);
        _mm256_store_ps(ap.add(88), a1);

        let a0 = _mm256_loadu_ps(a.add(lda*6));
        let a1 = _mm256_loadu_ps(a.add(lda*6+8));
        _mm256_store_ps(ap.add(96), a0);
        _mm256_store_ps(ap.add(104), a1);

        let a0 = _mm256_loadu_ps(a.add(lda*7));
        let a1 = _mm256_loadu_ps(a.add(lda*7+8));
        _mm256_store_ps(ap.add(112), a0);
        _mm256_store_ps(ap.add(120), a1);

        ap = ap.add(128);
        a = a.add(8*lda);

        k_i += 1;
    }
    k_i = 0;
    while k_i < k_left {
        let a0 = _mm256_loadu_ps(a);
        let a1 = _mm256_loadu_ps(a.add(8));
        _mm256_store_ps(ap, a0);
        _mm256_store_ps(ap.add(8), a1);

        ap = ap.add(16);
        a = a.add(lda);
        k_i += 1;
    }
}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packa_c_kx8(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        // use vector intrinscs
        let a0 = _mm256_loadu_ps(a);
        _mm256_store_ps(ap, a0);

        let a0 = _mm256_loadu_ps(a.add(lda));
        _mm256_store_ps(ap.add(8), a0);

        let a0 = _mm256_loadu_ps(a.add(lda*2));
        _mm256_store_ps(ap.add(16), a0);

        let a0 = _mm256_loadu_ps(a.add(lda*3));
        _mm256_store_ps(ap.add(24), a0);

        let a0 = _mm256_loadu_ps(a.add(lda*4));
        _mm256_store_ps(ap.add(32), a0);

        let a0 = _mm256_loadu_ps(a.add(lda*5));
        _mm256_store_ps(ap.add(40), a0);

        let a0 = _mm256_loadu_ps(a.add(lda*6));
        _mm256_store_ps(ap.add(48), a0);

        let a0 = _mm256_loadu_ps(a.add(lda*7));
        _mm256_store_ps(ap.add(56), a0);

        ap = ap.add(64);
        a = a.add(8*lda);

        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        let a0 = _mm256_loadu_ps(a);
        _mm256_store_ps(ap, a0);

        ap = ap.add(8);
        a = a.add(lda);
        k_i += 1;
    }

}



#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn copy_packed<const M: usize>(a: *const f32, b: *mut f32) {
    std::ptr::copy_nonoverlapping(a, b, M);
}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn packa_c_k<const M: usize>(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        // use vector intrinscs
        copy_packed::<M>(a, ap);
        
        copy_packed::<M>(a.add(lda), ap.add(8));

        copy_packed::<M>(a.add(lda*2), ap.add(16));

        copy_packed::<M>(a.add(lda*3), ap.add(24));

        copy_packed::<M>(a.add(lda*4), ap.add(32));

        copy_packed::<M>(a.add(lda*5), ap.add(40));

        copy_packed::<M>(a.add(lda*6), ap.add(48));

        copy_packed::<M>(a.add(lda*7), ap.add(56));


        ap = ap.add(64);
        a = a.add(8*lda);

        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left {
        copy_packed::<M>(a, ap);

        ap = ap.add(8);
        a = a.add(lda);
        k_i += 1;
    }

}





#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn pack_panel_24(
   m: usize, k: usize,
   a: *const TA, a_rs: usize, a_cs: usize,
   ap: *mut TA,
) {
   let mut ap = ap;
   let mut a = a;
   const MR: usize = 24;
   const MR_LAST_STEP: usize = 24;
   let mut m_idx = 0;
   if a_cs == 1 {
       let lda = a_rs;
       let k_iter = k / 8;
       let k_left = k % 8;
       while m_idx + MR_LAST_STEP <= m {
           packa_r_kx24(k_iter, k_left, a, lda, ap);
           m_idx += MR;
           ap = ap.add(k * MR);
           a = a.add(MR*lda);


       }
       let m_left = m - m_idx;
       if m_left > 16 {
            packa_scalar_k::<24>(
                m_left, k,
                a, a_rs, a_cs,
                ap
            );
           return;
       }
       if m_left > 8 {
        packa_scalar_k::<16>(
            m_left, k,
            a, a_rs, a_cs,
            ap
        );
       return;
        }
        if m_left > 0 {
            packa_scalar_k::<8>(
                m_left, k,
                a, a_rs, a_cs,
                ap
            );
           return;
        }
   } else if a_rs == 1 {
       let lda = a_cs;
       let k_iter = k / 8;
       let k_left = k % 8;
       while m_idx + MR_LAST_STEP <= m {
           packa_c_kx24(k_iter, k_left, a, lda, ap);
           m_idx += MR;
           ap = ap.add(k * MR);
           a = a.add(MR);
       }
       let m_left = m - m_idx;
       if m_left > 16 {
           let k_iter = (k - 1) / 8;
           let k_left = (k - 1) % 8;
           packa_c_kx24(k_iter, k_left, a, lda, ap);
           ap.add((k-1)*24).copy_from_nonoverlapping(a.add(lda*(k-1)), m_left);
           return;
       }
       if m_left > 8 {
           let k_iter = (k - 1) / 8;
           let k_left = (k - 1) % 8;
           packa_c_kx16(k_iter, k_left, a, lda, ap);
           ap.add((k-1)*16).copy_from_nonoverlapping(a.add(lda*(k-1)), m_left);
           return;
       }
       if m_left > 4 {
           let k_iter = (k - 1) / 8;
           let k_left = (k - 1) % 8;
           packa_c_kx8(k_iter, k_left, a, lda, ap);
           ap.add((k-1)*8).copy_from_nonoverlapping(a.add(lda*(k-1)), m_left);
           return;
       }
       if m_left > 2 {
           let k_iter = (k - 1) / 8;
           let k_left = (k - 1) % 8;
           packa_c_k::<4>(k_iter, k_left, a, lda, ap);
           ap.add((k-1)*8).copy_from_nonoverlapping(a.add(lda*(k-1)), m_left);
           return;
       }
       if m_left > 1 {
            packa_c_k::<2>(k_iter, k_left, a, lda, ap);
           return;
       }
       if m_left > 0 {
            packa_c_k::<1>(k_iter, k_left, a, lda, ap);
           return;
       }
   }
}


#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn pack_panel_16(
    m: usize, k: usize,
    a: *const TA, a_rs: usize, a_cs: usize,
    ap: *mut TA,
 ) {
    let mut ap = ap;
    let mut a = a;
    const MR: usize = 16;
    const MR_LAST_STEP: usize = 16;
    let mut m_idx = 0;
    if a_cs == 1 {
        let lda = a_rs;
        let k_iter = k / 8;
        let k_left = k % 8;
        while m_idx + MR_LAST_STEP <= m {
            packa_r_kx16(k_iter, k_left, a, lda, ap);
            m_idx += MR;
            ap = ap.add(k * MR);
            a = a.add(MR*lda);
 
        }
        let m_left = m - m_idx;
        if m_left > 8 {
         packa_scalar_k::<16>(
             m_left, k,
             a, a_rs, a_cs,
             ap
         );
        return;
         }
         if m_left > 0 {
             packa_scalar_k::<8>(
                 m_left, k,
                 a, a_rs, a_cs,
                 ap
             );
            return;
         }
    } else if a_rs == 1 {
        let lda = a_cs;
        let k_iter = k / 8;
        let k_left = k % 8;
        while m_idx + MR_LAST_STEP <= m {
            packa_c_kx16(k_iter, k_left, a, lda, ap);
            m_idx += MR;
            ap = ap.add(k * MR);
            a = a.add(MR);
        }
        let m_left = m - m_idx;
        if m_left > 8 {
            let k_iter = (k - 1) / 8;
            let k_left = (k - 1) % 8;
            packa_c_kx16(k_iter, k_left, a, lda, ap);
            ap.add((k-1)*16).copy_from_nonoverlapping(a.add(lda*(k-1)), m_left);
            return;
        }
        if m_left > 4 {
            let k_iter = (k - 1) / 8;
            let k_left = (k - 1) % 8;
            packa_c_kx8(k_iter, k_left, a, lda, ap);
            ap.add((k-1)*8).copy_from_nonoverlapping(a.add(lda*(k-1)), m_left);
            return;
        }
        if m_left > 2 {
            let k_iter = (k - 1) / 8;
            let k_left = (k - 1) % 8;
            packa_c_k::<4>(k_iter, k_left, a, lda, ap);
            ap.add((k-1)*8).copy_from_nonoverlapping(a.add(lda*(k-1)), m_left);
            return;
        }
        if m_left > 1 {
            packa_c_k::<2>(k_iter, k_left, a, lda, ap);
            return;
        }
        if m_left > 0 {
            packa_c_k::<1>(k_iter, k_left, a, lda, ap);
            return;
        }
    }
 }
 
