use seq_macro::seq;
use crate::{TA,TB};


use paste::paste;

use std::arch::x86_64::*;


#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn pack_scalar_k(
    m_left: usize, k: usize,
    a: *const TA, a_rs: usize, a_cs: usize,
    ap: *mut TA, vs: usize
) {
    let mr = (m_left + vs - 1) / vs * vs;
    for i in 0..m_left  {
        for j in 0..k/2 {
            *ap.add(j*2*mr+i*2) = *a.add(2*j*a_cs + i*a_rs);
            *ap.add(j*2*mr+i*2+1) = *a.add((2*j+1)*a_cs + i*a_rs);
        }
    }
    if k % 2 != 0 {
        for i in 0..m_left {
            *ap.add(k/2*2*mr+i*2) = *a.add(2*(k/2)*a_cs + i*a_rs);
            *ap.add(k/2*2*mr+i*2+1) = 0;
        }
    }
    // for i in m_left..mr {
    //     for j in 0..k/2 {
    //         *ap.add(j*2*mr+i*2) = 0;
    //         *ap.add(j*2*mr+i*2+1) = 0;
    //     }
    // }
}

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave_t<const M: usize>(
    a: *const TA, ap: *mut TA, lda: usize
) {
    if M == 1 {
        let mut t0 = [0_i16; 2];
        t0[0] = *a;
        t0[1] = *a.add(1);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 2);
        return;
    }
    if M == 2 {
        let mut t0 = [0_i16; 4];
        t0[0] = *a;
        t0[1] = *a.add(1);
        t0[2] = *a.add(lda);
        t0[3] = *a.add(lda+1);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 4);
        return;
    }
    if M == 3 {
        let mut t0 = [0_i16; 6];
        t0[0] = *a;
        t0[1] = *a.add(1);
        t0[2] = *a.add(lda);
        t0[3] = *a.add(lda+1);
        t0[4] = *a.add(lda*2);
        t0[5] = *a.add(lda*2+1);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 6);
        return;
    }
    if M == 4 {
        let mut t0 = [0_i16; 8];
        t0[0] = *a;
        t0[1] = *a.add(1);
        t0[2] = *a.add(lda);
        t0[3] = *a.add(lda+1);
        t0[4] = *a.add(lda*2);
        t0[5] = *a.add(lda*2+1);
        t0[6] = *a.add(lda*3);
        t0[7] = *a.add(lda*3+1);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 8);
        return;
    }

    if M == 8 {
        let mut t0 = [0_i16; 16];
        t0[0] = *a;
        t0[1] = *a.add(1);
        t0[2] = *a.add(lda);
        t0[3] = *a.add(lda+1);
        t0[4] = *a.add(lda*2);
        t0[5] = *a.add(lda*2+1);
        t0[6] = *a.add(lda*3);
        t0[7] = *a.add(lda*3+1);
        t0[8] = *a.add(lda*4);
        t0[9] = *a.add(lda*4+1);
        t0[10] = *a.add(lda*5);
        t0[11] = *a.add(lda*5+1);
        t0[12] = *a.add(lda*6);
        t0[13] = *a.add(lda*6+1);
        t0[14] = *a.add(lda*7);
        t0[15] = *a.add(lda*7+1);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 16);
        return;
    }

    if M == 16 {
        let mut t0 = [0_i16; 32];
        t0[0] = *a;
        t0[1] = *a.add(1);
        t0[2] = *a.add(lda);
        t0[3] = *a.add(lda+1);
        t0[4] = *a.add(lda*2);
        t0[5] = *a.add(lda*2+1);
        t0[6] = *a.add(lda*3);
        t0[7] = *a.add(lda*3+1);
        t0[8] = *a.add(lda*4);
        t0[9] = *a.add(lda*4+1);
        t0[10] = *a.add(lda*5);
        t0[11] = *a.add(lda*5+1);
        t0[12] = *a.add(lda*6);
        t0[13] = *a.add(lda*6+1);
        t0[14] = *a.add(lda*7);
        t0[15] = *a.add(lda*7+1);
        t0[16] = *a.add(lda*8);
        t0[17] = *a.add(lda*8+1);
        t0[18] = *a.add(lda*9);
        t0[19] = *a.add(lda*9+1);
        t0[20] = *a.add(lda*10);
        t0[21] = *a.add(lda*10+1);
        t0[22] = *a.add(lda*11);
        t0[23] = *a.add(lda*11+1);
        t0[24] = *a.add(lda*12);
        t0[25] = *a.add(lda*12+1);
        t0[26] = *a.add(lda*13);
        t0[27] = *a.add(lda*13+1);
        t0[28] = *a.add(lda*14);
        t0[29] = *a.add(lda*14+1);
        t0[30] = *a.add(lda*15);
        t0[31] = *a.add(lda*15+1);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 32);
        return;
    }

    if M == 32 {
        let mut t0 = [0_i16; 64];
        seq!(i in 0..32 {
            t0[2*i] = *a.add(lda*i);
            t0[2*i+1] = *a.add(lda*i+1);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 64);
    }
}

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave<const M: usize>(
    a: *const TA, ap: *mut TA, lda: usize
) {
    if M == 1 {
        let mut t0 = [0_i16; 2];
        t0[0] = *a;
        t0[1] = *a.add(lda);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 2);
        return;
    }
    if M == 2 {
        let mut t0 = [0_i16; 4];
        t0[0] = *a;
        t0[1] = *a.add(lda);
        t0[2] = *a.add(1);
        t0[3] = *a.add(lda+1);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 4);
        return;
    }
    if M == 3 {
        let mut t0 = [0_i16; 6];
        t0[0] = *a;
        t0[1] = *a.add(lda);
        t0[2] = *a.add(1);
        t0[3] = *a.add(lda+1);
        t0[4] = *a.add(2);
        t0[5] = *a.add(lda+2);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 6);
        return;
    }
    if M == 4 {
        let mut t0 = [0_i16; 8];
        t0[0] = *a;
        t0[1] = *a.add(lda);
        t0[2] = *a.add(1);
        t0[3] = *a.add(lda+1);
        t0[4] = *a.add(2);
        t0[5] = *a.add(lda+2);
        t0[6] = *a.add(3);
        t0[7] = *a.add(lda+3);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 8);
        return;
    }

    if M == 8 {
        let mut t0 = [0_i16; 16];
        t0[0] = *a;
        t0[1] = *a.add(lda);
        t0[2] = *a.add(1);
        t0[3] = *a.add(lda+1);
        t0[4] = *a.add(2);
        t0[5] = *a.add(lda+2);
        t0[6] = *a.add(3);
        t0[7] = *a.add(lda+3);
        t0[8] = *a.add(4);
        t0[9] = *a.add(lda+4);
        t0[10] = *a.add(5);
        t0[11] = *a.add(lda+5);
        t0[12] = *a.add(6);
        t0[13] = *a.add(lda+6);
        t0[14] = *a.add(7);
        t0[15] = *a.add(lda+7);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 16);
        return;
    }
}

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave_left<const M: usize>(
    a: *const TA, ap: *mut TA
) {
    if M == 1 {
        let mut t0 = [0_i16; 2];
        t0[0] = *a;
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 2);
        return;
    }
    if M == 2 {
        let mut t0 = [0_i16; 4];
        t0[0] = *a;
        t0[2] = *a.add(1);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 4);
        return;
    }
    if M == 3 {
        let mut t0 = [0_i16; 6];
        t0[0] = *a;
        t0[2] = *a.add(1);
        t0[4] = *a.add(2);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 6);
        return;
    }
    if M == 4 {
        let mut t0 = [0_i16; 8];
        t0[0] = *a;
        t0[2] = *a.add(1);
        t0[4] = *a.add(2);
        t0[6] = *a.add(3);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 8);
        return;
    }

    if M == 8 {
        let mut t0 = [0_i16; 16];
        t0[0] = *a;
        t0[2] = *a.add(1);
        t0[4] = *a.add(2);
        t0[6] = *a.add(3);
        t0[8] = *a.add(4);
        t0[10] = *a.add(5);
        t0[12] = *a.add(6);
        t0[14] = *a.add(7);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 16);
        return;
    }
}


#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn interleave_left_t<const M: usize>(
    a: *const TA, ap: *mut TA, lda: usize
) {
    if M == 1 {
        let mut t0 = [0_i16; 2];
        t0[0] = *a;
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 2);
        return;
    }
    if M == 2 {
        let mut t0 = [0_i16; 4];
        t0[0] = *a;
        t0[2] = *a.add(lda);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 4);
        return;
    }
    if M == 3 {
        let mut t0 = [0_i16; 6];
        t0[0] = *a;
        t0[2] = *a.add(lda);
        t0[4] = *a.add(2*lda);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 6);
        return;
    }
    if M == 4 {
        let mut t0 = [0_i16; 8];
        t0[0] = *a;
        t0[2] = *a.add(lda);
        t0[4] = *a.add(2*lda);
        t0[6] = *a.add(3*lda);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 8);
        return;
    }

    if M == 8 {
        let mut t0 = [0_i16; 16];
        t0[0] = *a;
        t0[2] = *a.add(lda);
        t0[4] = *a.add(2*lda);
        t0[6] = *a.add(3*lda);
        t0[8] = *a.add(4*lda);
        t0[10] = *a.add(5*lda);
        t0[12] = *a.add(6*lda);
        t0[14] = *a.add(7*lda);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 16);
        return;
    }

    if M == 16 {
        let mut t0 = [0_i16; 32];
        t0[0] = *a;
        t0[2] = *a.add(lda);
        t0[4] = *a.add(2*lda);
        t0[6] = *a.add(3*lda);
        t0[8] = *a.add(4*lda);
        t0[10] = *a.add(5*lda);
        t0[12] = *a.add(6*lda);
        t0[14] = *a.add(7*lda);
        t0[16] = *a.add(8*lda);
        t0[18] = *a.add(9*lda);
        t0[20] = *a.add(10*lda);
        t0[22] = *a.add(11*lda);
        t0[24] = *a.add(12*lda);
        t0[26] = *a.add(13*lda);
        t0[28] = *a.add(14*lda);
        t0[30] = *a.add(15*lda);
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 32);
        return;
    }

    if M == 32 {
        let mut t0 = [0_i16; 64];
        seq!(i in 0..32 {
            t0[2*i] = *a.add(lda*i);
        });
        std::ptr::copy_nonoverlapping(t0.as_ptr(), ap, 64);
    }
}

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn pack_k_v0<const M: usize, const MR: usize>(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        // use vector intrinscs
        seq!(i in 0..4 {
            interleave::<M>(a.add(lda*2*i), ap.add(MR*2*i), lda);
        });

        ap = ap.add(MR*8);
        a = a.add(8*lda);

        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left / 2 {
        interleave::<M>(a, ap, lda);
        ap = ap.add(MR*2);
        a = a.add(lda*2);
        k_i += 1;
    }

    if k_left % 2 != 0 {
        interleave_left::<M>(a, ap);
    }
}


#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn pack_k_v1<const M: usize, const MR: usize>(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    while k_i < k_iter {
        // use vector intrinscs
        seq!(i in 0..4 {
            interleave_t::<M>(a.add(2*i), ap.add(MR*2*i), lda);
        });

        ap = ap.add(MR*8);
        a = a.add(8);

        k_i += 1;
    }

    k_i = 0;

    while k_i < k_left / 2 {
        interleave_t::<M>(a, ap, lda);
        ap = ap.add(MR*2);
        a = a.add(2);
        k_i += 1;
    }

    if k_left % 2 != 0 {
        interleave_left_t::<M>(a, ap, lda);
    }
}

#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn pack_kx16_v0(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 16;
    while k_i < k_iter {
        // use vector intrinscs
        seq!(i in 0..4 {
            let a0 = _mm256_loadu_si256(a.add(lda*2*i) as *const __m256i);
            let b0 = _mm256_loadu_si256(a.add(lda*(2*i+1)) as *const __m256i);
            let t0 = _mm256_unpacklo_epi16(a0, b0);
            let t1 = _mm256_unpackhi_epi16(a0, b0);
            let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
            let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
            _mm256_storeu_si256(ap.add(MR*2*i) as *mut __m256i, a0);
            _mm256_storeu_si256(ap.add(MR*2*i+16) as *mut __m256i, b0);
        });

        ap = ap.add(MR*8);
        a = a.add(8*lda);

        k_i += 1;
    }
    k_i = 0;
    while k_i < k_left / 2 {
        let a0 = _mm256_loadu_si256(a as *const __m256i);
        let b0 = _mm256_loadu_si256(a.add(lda) as *const __m256i);
        let t0 = _mm256_unpacklo_epi16(a0, b0);
        let t1 = _mm256_unpackhi_epi16(a0, b0);
        let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
        let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
        _mm256_storeu_si256(ap as *mut __m256i, a0);
        _mm256_storeu_si256(ap.add(16) as *mut __m256i, b0);

        ap = ap.add(MR*2);
        a = a.add(lda*2);
        k_i += 1;
    }
    if k_left % 2 != 0 {
        let a0 = _mm256_loadu_si256(a as *const __m256i);
        let b0 = _mm256_setzero_si256();
        let t0 = _mm256_unpacklo_epi16(a0, b0);
        let t1 = _mm256_unpackhi_epi16(a0, b0);
        let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
        let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
        _mm256_storeu_si256(ap as *mut __m256i, a0);
        _mm256_storeu_si256(ap.add(16) as *mut __m256i, b0);
    }
}



#[target_feature(enable = "avx,avx2")]
pub(crate) unsafe fn pack_kx32_v0(
    k_iter: usize, k_left: usize,
    a: *const TA, lda: usize,
    ap: *mut TA,
) {
    let mut k_i = 0;
    let mut a = a;
    let mut ap = ap;
    const MR: usize = 32;
    while k_i < k_iter {
        // use vector intrinscs
        seq!(i in 0..4 {
            let a0 = _mm256_loadu_si256(a.add(lda*2*i) as *const __m256i);
            let b0 = _mm256_loadu_si256(a.add(lda*(2*i+1)) as *const __m256i);
            let t0 = _mm256_unpacklo_epi16(a0, b0);
            let t1 = _mm256_unpackhi_epi16(a0, b0);
            let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
            let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
            _mm256_storeu_si256(ap.add(MR*2*i) as *mut __m256i, a0);
            _mm256_storeu_si256(ap.add(MR*2*i+16) as *mut __m256i, b0);
        });

        seq!(i in 0..4 {
            let a0 = _mm256_loadu_si256(a.add(lda*2*i+16) as *const __m256i);
            let b0 = _mm256_loadu_si256(a.add(lda*(2*i+1)+16) as *const __m256i);
            let t0 = _mm256_unpacklo_epi16(a0, b0);
            let t1 = _mm256_unpackhi_epi16(a0, b0);
            let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
            let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
            _mm256_storeu_si256(ap.add(MR*2*i+32) as *mut __m256i, a0);
            _mm256_storeu_si256(ap.add(MR*2*i+48) as *mut __m256i, b0);
        });

        ap = ap.add(MR*8);
        a = a.add(8*lda);

        k_i += 1;
    }
    k_i = 0;
    while k_i < k_left / 2 {
        let a0 = _mm256_loadu_si256(a as *const __m256i);
        let b0 = _mm256_loadu_si256(a.add(lda) as *const __m256i);
        let t0 = _mm256_unpacklo_epi16(a0, b0);
        let t1 = _mm256_unpackhi_epi16(a0, b0);
        let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
        let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
        _mm256_storeu_si256(ap as *mut __m256i, a0);
        _mm256_storeu_si256(ap.add(16) as *mut __m256i, b0);


        let a0 = _mm256_loadu_si256(a.add(16) as *const __m256i);
        let b0 = _mm256_loadu_si256(a.add(lda+16) as *const __m256i);
        let t0 = _mm256_unpacklo_epi16(a0, b0);
        let t1 = _mm256_unpackhi_epi16(a0, b0);
        let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
        let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
        _mm256_storeu_si256(ap.add(32) as *mut __m256i, a0);
        _mm256_storeu_si256(ap.add(48) as *mut __m256i, b0);

        ap = ap.add(MR*2);
        a = a.add(lda*2);
        k_i += 1;
    }
    if k_left % 2 != 0 {
        let a0 = _mm256_loadu_si256(a as *const __m256i);
        let b0 = _mm256_setzero_si256();
        let t0 = _mm256_unpacklo_epi16(a0, b0);
        let t1 = _mm256_unpackhi_epi16(a0, b0);
        let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
        let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);

        _mm256_storeu_si256(ap as *mut __m256i, a0);
        _mm256_storeu_si256(ap.add(16) as *mut __m256i, b0);

        let a0 = _mm256_loadu_si256(a.add(16) as *const __m256i);
        let b0 = _mm256_setzero_si256();
        let t0 = _mm256_unpacklo_epi16(a0, b0);
        let t1 = _mm256_unpackhi_epi16(a0, b0);
        let a0 = _mm256_permute2f128_si256(t0, t1, 0b0010_0000);
        let b0 = _mm256_permute2f128_si256(t0, t1, 0b0011_0001);
        _mm256_storeu_si256(ap.add(32) as *mut __m256i, a0);
        _mm256_storeu_si256(ap.add(48) as *mut __m256i, b0);
    }
}


// #[target_feature(enable = "avx,avx2")]
// pub(crate) unsafe fn pack_kx16_v1(
//     k_iter: usize, k_left: usize,
//     a: *const TA, lda: usize,
//     ap: *mut TA,
// ) {
//     let mut k_i = 0;
//     let mut a = a;
//     let mut ap = ap;
//     const MR: usize = 16;
//     while k_i < k_iter {
//         // pack_t::<MR>(a, lda, ap);
//         // pack_t::<MR>(a.add(8*lda), lda, ap.add(8));

//         ap = ap.add(MR*8);
//         a = a.add(8);
//         k_i += 1;
//     }

//     k_i = 0;

//     while k_i < k_left {
//         seq!(i in 0..16 {
//             *ap.add(i) = *a.add(i*lda);
//         });

//         ap = ap.add(MR);
//         a = a.add(1);
//         k_i += 1;
//     }
// }


macro_rules! def_packb {
   ($nr:tt) => {
        paste! {
        #[target_feature(enable = "avx,avx2")]
        pub(crate) unsafe fn [<packb_panel_ $nr>](
                n: usize, k: usize,
                b: *const TB, b_rs: usize, b_cs: usize,
                bp: *mut TB,
            ) {
                let k_eff = (k+1) / 2 * 2;
                let k_iter = k / 8;
                let k_left = k % 8;
                let mut bp = bp;
                let mut b = b;
                const NR: usize = $nr;
                const NR_LAST_STEP: usize = $nr;
                let mut n_idx = 0;
                if b_rs == 1 {
                    let ldb = b_cs;
                    while n_idx + NR_LAST_STEP <= n {
                        pack_k_v0::<NR,NR>(k_iter, k_left, b, ldb, bp);
                        n_idx += NR;
                        bp = bp.add(k_eff*NR);
                        b = b.add(NR);
                    }
                    let n_left = n - n_idx;
                    seq!(NL in 1..$nr {
                        if n_left == NL {
                            // pack_k_v0::<NL,NL>(k_iter, k_left, b, ldb, bp);
                            pack_scalar_k(
                                NL, k,
                                b, b_rs, b_cs,
                                bp, 1
                            );
                            return;
                        }
                    });
                } else if b_cs == 1 {
                    let ldb = b_rs;
                    while n_idx + NR_LAST_STEP <= n {
                        pack_k_v1::<NR,NR>(k_iter, k_left, b, ldb, bp);
                        n_idx += NR;
                        bp = bp.add(k_eff*NR);
                        b =  b.add(NR*ldb);
                    }
                    let n_left = n - n_idx;
                    seq!(NL in 1..$nr {
                        if n_left == NL {
                            // pack_k_v1::<NL,NL>(k_iter, k_left, b, ldb, bp);
                            pack_scalar_k(
                                NL, k,
                                b, b_rs, b_cs,
                                bp, 1
                            );
                            return;
                        }
                    });
                }
            }   
        }
   };
}


def_packb!(4);
def_packb!(8);


macro_rules! def_packa {
    ($mr:tt) => {
        paste! {
            #[target_feature(enable = "avx,avx2")]
            pub(crate) unsafe fn [<packa_panel_ $mr>](
                m_left: usize, k: usize,
                a: *const TA, a_rs: usize, a_cs: usize,
                ap: *mut TA, vs: usize
            ) {
                let k_eff = (k+1) / 2 * 2;
                let mut ap = ap;
                let mut a = a;
                const MR: usize = $mr;
                const MR_LAST_STEP: usize = $mr;
                let mut m_idx = 0;
                if a_rs == 1 {
                    let lda = a_cs;
                    let k_iter = k / 8;
                    let k_left = k % 8;
                    while m_idx + MR_LAST_STEP <= m_left {
                        [<pack_kx$mr _v0>](k_iter, k_left, a, lda, ap);
                        m_idx += MR;
                        ap = ap.add(k_eff * MR);
                        a = a.add(MR);
                    }
                    let m_left = m_left - m_idx;
                    pack_scalar_k(
                        m_left, k,
                        a, a_rs, a_cs,
                        ap, vs
                    );

                } else if a_cs == 1 {
                    let lda = a_rs;
                    let k_iter = k / 8;
                    let k_left = k % 8;
                    while m_idx + MR_LAST_STEP <= m_left {
                        // [<pack_kx$mr _v1>](k_iter, k_left, a, lda, ap);
                        pack_k_v1::<$mr,$mr>(k_iter, k_left, a, lda, ap);
                        m_idx += MR;
                        ap = ap.add(k_eff * MR);
                        a = a.add(MR*lda);
                    }
                    let m_left = m_left - m_idx;
                    pack_scalar_k(
                        m_left, k,
                        a, a_rs, a_cs,
                        ap, vs
                    );
                }
            }
        }
    };
}

def_packa!(16);

def_packa!(32);

