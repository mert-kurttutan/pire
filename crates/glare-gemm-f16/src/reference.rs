
#![allow(unused)]

 
 
 use crate::{
    TA,TB,TC,
    GemmGotoPackaPackb,
    GemmSmallM,
    GemmSmallN,
    Gemv,
 };
 
// pub const GOTO_MC: usize = env_or!("CORENUM_SGEMM_MC", 4800);
// pub const GOTO_NC: usize = env_or!("CORENUM_SGEMM_NC", 320);
// pub const GOTO_KC: usize = env_or!("CORENUM_SGEMM_KC", 192);
// pub const GOTO_MR: usize = env_or!("CORENUM_SGEMM_MR", 24);
// pub const GOTO_NR: usize = env_or!("CORENUM_SGEMM_NR", 4);

// pub const SGEMM_M_TH: usize = env_or!("CORENUM_SGEMM_SPM_TH", 128);
// pub const SGEMM_N_TH: usize = env_or!("CORENUM_SGEMM_SPN_TH", 256);
// pub const SGEMM_K_TH: usize = env_or!("CORENUM_SGEMM_SPK_TH", 256);

// pub const SA_NC: usize = env_or!("CORENUM_SGEMM_NC_SA", 168);
// pub const SA_KC: usize = env_or!("CORENUM_SGEMM_KC_SA", 256);
// pub const SA_MR: usize = env_or!("CORENUM_SGEMM_MR_SA", 16);
// pub const SA_NR: usize = env_or!("CORENUM_SGEMM_NR_SA", 6);

pub struct ReferenceGemv {}
 

 
 pub struct ReferenceGemm {}
 
 
 
//  impl Gemv<TA,TB,TC> for ReferenceGemv
//  {
//     unsafe fn gemv_serial(
//         m: usize, n: usize,
//         alpha: *const TA,
//         a: *const TA, a_rs: usize, a_cs: usize,
//         x: *const TB, incx: usize,
//         beta: *const TC,
//         y: *mut TC, incy: usize,
//         t_id: usize, t_par: &CorenumPar
//     ) {
//          for i in 0..m {
//              let mut sum = 0.0;
//              for j in 0..n {
//                  sum += *a.add(a_rs*i + a_cs*j) * *x.add(j*incx);
//              }
//              if *beta == 0.0 {
//                  *y.add(i*incy) = sum * *alpha;
//              } else {
//                  *y.add(i*incy) = sum * *alpha + *beta * *y.add(i*incy);
//              }
//          }   
//      }
//  }
 
 
 
//  impl GemmSmallM<TA,TB,TC> for ReferenceGemm
//  {
//     const MC: usize = SGEMM_M_TH; const NC: usize = SA_NC; const KC: usize = SA_KC;
//     const MR: usize = SA_MR; const NR: usize = SA_NR;
//     const ONE: TC = 1.0;
//     unsafe fn packa(
//         m: usize, k: usize,
//         a: *const TA, a_rs: usize, a_cs: usize,
//         ap: *mut TA,
//     ) {
//      let mut i = 0;
//      let mut ap = ap;
//      let mut a = a;
//      while i < m {
//          let i_end = if m >= (i + SA_MR) {SA_MR} else {m - i}; 
//          for p in 0..k {
//              for i_i in 0..i_end {
//                  *ap.add(i_i + p * SA_MR) = *a.add(a_rs*i_i + a_cs*p)
//              }
//          }
//          ap = ap.add(k*SA_MR);
//          a = a.add(SA_MR*a_rs);
//          i += SA_MR
//      }
 
//     }
//     unsafe fn kernel(
//         m: usize, n: usize, k: usize,
//         alpha: *const TA,
//         beta: *const TC,
//         b: *const TB, b_rs: usize, b_cs: usize,
//         c: *mut TC, c_rs: usize, c_cs: usize,
//         ap: *mut TA,
//     ) {
//          let mut m_iter = 0;
//          let mut b = b;
//          let mut c0 = c;
//          let mut n_iter = 0;
//         while n_iter < n {
//             let mut ab_cum = [[0_f32; SA_MR]; SA_NR];
//             for p in 0..k {
//                 for j in 0..SA_NR {
//                     for i in 0..SA_MR {
//                         ab_cum[j][i] += *ap.add(p*SA_MR + i) * *b.add(p*b_rs + j*b_cs);
//                     }
//                 }
//             }
//             let m_end = if m >= m_iter + SA_MR { SA_MR } else {m - m_iter};
//             let n_end = if n >= n_iter + SA_NR { SA_NR } else {n - n_iter};
//             if *beta == 0.0 {
//                 for j in 0..n_end {
//                     for i in 0..m_end {
//                         *c0.add(i*c_rs + c_cs*j) = *alpha * ab_cum[j][i];
//                     }
//                 }
//             } else {
//                 for j in 0..n_end {
//                     for i in 0..m_end {
//                         *c0.add(i*c_rs + c_cs*j) = *alpha * ab_cum[j][i] + *beta * *c0.add(i*c_rs + c_cs*j);
//                     }
//                 }
//             }
//             n_iter += SA_NR;
//             c0 = c0.add(SA_NR*c_cs);
//             b = b.add(b_cs*SA_NR);
//         }
 
//     }
//  }
 
 
 
 
 
 
 
//  // func signature rule
//  // 1 -> m, n, k params if exists
//  // 2 -> pointer a and lda
//  // 3 -> pointer b and ldb
//  // 4 -> pointer c and ldc
//  // 5 -> pointer packed a and b
//  // 6 -> m related blocking params
//  // 7 -> n related blocking params
 
 
 
//  impl GemmGotoPackaPackb<TA,TB,TC> for ReferenceGemm
//  {
//     const MC: usize = GOTO_MC; const NC: usize = GOTO_NC; const KC: usize = GOTO_KC;
//     const MR: usize = GOTO_MR; const NR: usize = GOTO_NR;
//     const ONE: TC = 1.0;
//     fn is_packa(t_cfg: &CorenumThreadConfig) -> bool {
//         assert!(t_cfg.par.num_threads == 1, "Multithreading is not supported yet");
//         true
//     }
//     fn is_packb(t_cfg: &CorenumThreadConfig) -> bool {
//         assert!(t_cfg.par.num_threads == 1, "Multithreading is not supported yet");
//         true
//     }
//     unsafe fn packa(
//         m: usize, k: usize,
//         a: *const TA, a_rs: usize, a_cs: usize,
//         ap: *mut TA,
//     ) {
//      let mut i = 0;
//      let mut ap = ap;
//      let mut a = a;
//      while i < m {
//          let i_end = if m >= (i + GOTO_MR) {GOTO_MR} else {m - i}; 
//          for p in 0..k {
//              for i_i in 0..i_end {
//                  *ap.add(i_i + p * GOTO_MR) = *a.add(a_rs*i_i + a_cs*p)
//              }
//          }
//          ap = ap.add(k*GOTO_MR);
//          a = a.add(GOTO_MR*a_rs);
//          i += GOTO_MR
//      }
 
//     }
//     unsafe fn packb(
//         n: usize, k: usize,
//         b: *const TA, b_rs: usize, b_cs: usize,
//         bp: *mut TA,
//     ) {
//          let mut j = 0;
//          let mut bp = bp;
//          let mut b = b;
//          while j < n {
//              let j_end = if n >= (j + GOTO_NR) {GOTO_NR} else {n - j}; 
//              for p in 0..k {
//                  for j_i in 0..j_end {
//                      *bp.add(j_i + p * GOTO_NR) = *b.add(b_rs*p + b_cs*j_i)
//                  }
//              }
//              bp = bp.add(k*GOTO_NR);
//              b = b.add(GOTO_NR*b_cs);
//              j += GOTO_NR
//          }
 
//      }
//     unsafe fn kernel(
//         m: usize, n: usize, k: usize,
//         alpha: *const TA,
//         beta: *const TC,
//         c: *mut TC,
//         c_rs: usize, c_cs: usize,
//         ap: *const TA, bp: *const TB,
//     ) {
//          let mut m_iter = 0;
//          let mut ap = ap;
//          let mut c0 = c;
//          while m_iter < m {
//              let mut n_iter = 0;
//              let mut bp = bp;
//              let mut c1 = c0;
//              while n_iter < n {
//                  let mut ab_cum = [[0_f32; GOTO_MR]; GOTO_NR];
//                  for p in 0..k {
//                      for j in 0..GOTO_NR {
//                          for i in 0..GOTO_MR {
//                              ab_cum[j][i] += *ap.add(p*GOTO_MR + i) * *bp.add(p*GOTO_NR + j);
//                          }
//                      }
//                  }
//                  let m_end = if m >= m_iter + GOTO_MR { GOTO_NR } else {m - m_iter};
//                  let n_end = if n >= n_iter + GOTO_NR { GOTO_NR } else {n - n_iter};
//                  if *beta == 0.0 {
//                      for j in 0..n_end {
//                          for i in 0..m_end {
//                              *c1.add(i*c_rs + c_cs*j) = *alpha * ab_cum[j][i];
//                          }
//                      }
//                  } else {
//                      for j in 0..n_end {
//                          for i in 0..m_end {
//                              *c1.add(i*c_rs + c_cs*j) = *alpha * ab_cum[j][i] + *beta * *c1.add(i*c_rs + c_cs*j);
//                          }
//                      }
//                  }
//                  n_iter += GOTO_NR;
//                  c1 = c1.add(GOTO_NR*c_cs);
//                  bp = bp.add(k*GOTO_NR);
//              }
//              m_iter += GOTO_MR;
//              c0 = c0.add(GOTO_MR*c_rs);
//              ap = ap.add(k*GOTO_MR);
//          }
 
//     }
//  }
 
 