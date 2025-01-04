//! # This crate is only for internal use in the pire project
//! Nothing is expected to be used outside this module
//! No semver guarantees

use core::mem::size_of;
use once_cell::sync::Lazy;
use std::sync::{Barrier, Mutex, MutexGuard, RwLock, RwLockReadGuard};

pub mod range_rwlock;

#[derive(Copy, Clone)]
pub struct IdentityFn;

pub trait UnaryFn<T>: Copy + std::marker::Sync {
    unsafe fn call(self, c: *mut T, m: usize);
}

impl<T> UnaryFn<T> for IdentityFn {
    #[inline(always)]
    unsafe fn call(self, _c: *mut T, _m: usize) {}
}

impl<T> UnaryFn<T> for unsafe fn(*mut T, m: usize) {
    #[inline(always)]
    unsafe fn call(self, c: *mut T, m: usize) {
        self(c, m);
    }
}

#[inline(always)]
pub unsafe fn load_buf<T: Copy>(c: *const T, c_rs: usize, c_cs: usize, c_buf: &mut [T], m: usize, n: usize, mr: usize) {
    for j in 0..n {
        for i in 0..m {
            c_buf[i + j * mr] = *c.add(i * c_rs + j * c_cs);
        }
    }
}

#[inline(always)]
pub unsafe fn store_buf<T: Copy>(c: *mut T, c_rs: usize, c_cs: usize, c_buf: &[T], m: usize, n: usize, mr: usize) {
    for j in 0..n {
        for i in 0..m {
            *c.add(i * c_rs + j * c_cs) = c_buf[i + j * mr];
        }
    }
}

pub fn matrix_size(m: usize, n: usize) -> usize {
    n * m
}

use range_rwlock::{RangeLock, RangeLockReadGuard, RangeLockWriteGuard};

#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
pub struct CpuFeatures {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512f16: bool,
    // pub avx512bf16: bool,
    pub avx512bw: bool,
    pub avx512_vnni: bool,
    pub fma: bool,
    pub fma4: bool,
    pub f16c: bool,
}

#[cfg(target_arch = "x86")]
#[derive(Copy, Clone)]
pub struct CpuFeatures {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
}

#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
pub struct CpuFeatures {
    pub sve: bool,
    pub neon: bool,
    pub fp16: bool,
    pub f32mm: bool,
    pub fcma: bool,
    pub i8mm: bool,
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
#[derive(Copy, Clone)]
pub struct CpuFeatures {
    pub dummy: bool,
}

// padding in bytes
const CACHELINE_PAD: usize = 1024;

pub struct HWConfig {
    pub cpu_ft: CpuFeatures,
    pub hw_model: HWModel,
    is_l1_shared: bool,
    is_l2_shared: bool,
    is_l3_shared: bool,
    #[cfg(target_arch = "x86_64")]
    model_id: u8,
}

impl HWConfig {
    pub fn get_cache_info(&self) -> (bool, bool, bool) {
        (self.is_l1_shared, self.is_l2_shared, self.is_l3_shared)
    }
    pub fn hw_model(&self) -> HWModel {
        self.hw_model
    }

    pub fn cpu_ft(&self) -> CpuFeatures {
        self.cpu_ft
    }

    #[cfg(target_arch = "x86_64")]
    pub fn model_id(&self) -> u8 {
        self.model_id
    }
}

#[derive(Copy, Clone)]
pub enum HWModel {
    Reference,
    Haswell,
    Skylake,
}

const SKYLAKE: [u8; 13] = [78, 85, 94, 126, 140, 141, 167, 151, 154, 183, 186, 143, 207];

const HASWELL: [u8; 10] = [69, 70, 63, 42, 58, 165, 79, 86, 61, 71];

impl HWModel {
    pub fn from_hw(family_id: u8, model_id: u8, _cpu_ft: CpuFeatures) -> Self {
        if family_id == 6 {
            if SKYLAKE.contains(&model_id) {
                return HWModel::Skylake;
            }
            if HASWELL.contains(&model_id) {
                return HWModel::Haswell;
            }
        }
        // if model id is not in the list, default by looking at cpu features
        #[cfg(target_arch = "x86_64")]
        {
            if _cpu_ft.avx512f {
                return HWModel::Skylake;
            }
            if _cpu_ft.avx {
                return HWModel::Haswell;
            }
        }

        // default to reeference
        return HWModel::Reference;
    }
    pub fn get_cache_info(&self) -> (bool, bool, bool) {
        match self {
            HWModel::Reference => (false, false, true),
            HWModel::Haswell => (false, false, true),
            HWModel::Skylake => (false, false, true),
        }
    }
}

// Use family and model id instead of cache size parameters
// since the relation between optimal parameters (based on performance) and cache size parameters  can be non-trivial
// e.g. it might be cpu model dependent

#[inline]
fn detect_hw_config() -> HWConfig {
    #[cfg(target_arch = "x86_64")]
    {
        let cpuid = raw_cpuid::CpuId::new();
        let feature_info = cpuid.get_feature_info().unwrap();
        let extended_feature_info = cpuid.get_extended_feature_info().unwrap();
        let sse = feature_info.has_sse();
        let sse2 = feature_info.has_sse2();
        let sse3 = feature_info.has_sse3();
        let ssse3 = feature_info.has_ssse3();
        let avx = feature_info.has_avx();
        let fma = feature_info.has_fma();
        let avx2 = extended_feature_info.has_avx2();
        let avx512f16 = extended_feature_info.has_avx512_fp16();
        // let avx512bf16 = extended_feature_info.has_avx512_bf16();
        let avx512f = extended_feature_info.has_avx512f();
        let avx512bw = extended_feature_info.has_avx512bw();
        let avx512_vnni = extended_feature_info.has_avx512vnni();
        let f16c = feature_info.has_f16c();
        let extended_processor_info = cpuid.get_extended_processor_and_feature_identifiers().unwrap();
        let fma4 = extended_processor_info.has_fma4();
        let cpu_ft = CpuFeatures {
            sse,
            sse2,
            sse3,
            ssse3,
            avx,
            avx2,
            avx512f,
            avx512f16,
            avx512bw,
            avx512_vnni,
            fma,
            fma4,
            f16c,
        };
        let family_id = feature_info.family_id();
        let model_id = feature_info.model_id();
        let hw_model = HWModel::from_hw(family_id, model_id, cpu_ft);
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_model.get_cache_info();
        return HWConfig { cpu_ft, hw_model, is_l1_shared, is_l2_shared, is_l3_shared, model_id };
    }
    #[cfg(target_arch = "x86")]
    {
        let cpuid = raw_cpuid::CpuId::new();
        let feature_info = cpuid.get_feature_info().unwrap();
        let sse = feature_info.has_sse();
        let sse2 = feature_info.has_sse2();
        let sse3 = feature_info.has_sse3();
        let ssse3 = feature_info.has_ssse3();
        let cpu_ft = CpuFeatures { sse, sse2, sse3, ssse3 };
        let family_id = feature_info.family_id();
        let model_id = feature_info.model_id();
        let hw_model = HWModel::from_hw(family_id, model_id, cpu_ft);
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_model.get_cache_info();
        return HWConfig { cpu_ft, hw_model, is_l1_shared, is_l2_shared, is_l3_shared };
    }
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        let neon = is_aarch64_feature_detected!("neon");
        let sve = is_aarch64_feature_detected!("sve");
        let fp16 = is_aarch64_feature_detected!("fp16");
        let f32mm = is_aarch64_feature_detected!("f32mm");
        let fcma = is_aarch64_feature_detected!("fcma");
        let i8mm = is_aarch64_feature_detected!("i8mm");

        return HWConfig {
            cpu_ft: CpuFeatures { neon, sve, fp16, f32mm, fcma, i8mm },
            hw_model: HWModel::Reference,
            is_l1_shared: false,
            is_l2_shared: false,
            is_l3_shared: true,
        };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        return HWConfig {
            cpu_ft: CpuFeatures { dummy: false },
            hw_model: HWModel::Reference,
            is_l1_shared: false,
            is_l2_shared: false,
            is_l3_shared: true,
        };
    }
}

#[cfg(feature = "debug_cpu_features")]
#[allow(unused)]
fn apply_debug_cpu_features(cpu_ft: &mut CpuFeatures) {
    #[cfg(target_arch = "x86_64")]
    {
        let sse_turn_off = std::env::var("PIRE_SSE_OFF").is_ok();
        let sse2_turn_off = std::env::var("PIRE_SSE2_OFF").is_ok();
        let sse3_turn_off = std::env::var("PIRE_SSE3_OFF").is_ok();
        let ssse3_turn_off = std::env::var("PIRE_SSSE3_OFF").is_ok();
        let avx_turn_off = std::env::var("PIRE_AVX_OFF").is_ok();
        let avx2_turn_off = std::env::var("PIRE_AVX2_OFF").is_ok();
        let avx512f_turn_off = std::env::var("PIRE_AVX512F_OFF").is_ok();
        let avx512f16_turn_off = std::env::var("PIRE_AVX512F16_OFF").is_ok();
        let avx512bw_turn_off = std::env::var("PIRE_AVX512BW_OFF").is_ok();
        let avx512_vnni_turn_off = std::env::var("PIRE_AVX512_VNNI_OFF").is_ok();
        let fma_turn_off = std::env::var("PIRE_FMA_OFF").is_ok();
        let fma4_turn_off = std::env::var("PIRE_FMA4_OFF").is_ok();
        let f16c_turn_off = std::env::var("PIRE_F16C_OFF").is_ok();

        cpu_ft.sse = cpu_ft.sse && !sse_turn_off;
        cpu_ft.sse2 = cpu_ft.sse2 && !sse2_turn_off;
        cpu_ft.sse3 = cpu_ft.sse3 && !sse3_turn_off;
        cpu_ft.ssse3 = cpu_ft.ssse3 && !ssse3_turn_off;
        cpu_ft.avx = cpu_ft.avx && !avx_turn_off;
        cpu_ft.avx2 = cpu_ft.avx2 && !avx2_turn_off;
        cpu_ft.avx512f = cpu_ft.avx512f && !avx512f_turn_off;
        cpu_ft.avx512f16 = cpu_ft.avx512f16 && !avx512f16_turn_off;
        cpu_ft.avx512bw = cpu_ft.avx512bw && !avx512bw_turn_off;
        cpu_ft.avx512_vnni = cpu_ft.avx512_vnni && !avx512_vnni_turn_off;
        cpu_ft.fma = cpu_ft.fma && !fma_turn_off;
        cpu_ft.fma4 = cpu_ft.fma4 && !fma4_turn_off;
        cpu_ft.f16c = cpu_ft.f16c && !f16c_turn_off;
    }
    #[cfg(target_arch = "x86")]
    {
        let sse_turn_off = std::env::var("PIRE_SSE_OFF").is_ok();
        let sse2_turn_off = std::env::var("PIRE_SSE2_OFF").is_ok();
        let sse3_turn_off = std::env::var("PIRE_SSE3_OFF").is_ok();
        let ssse3_turn_off = std::env::var("PIRE_SSSE3_OFF").is_ok();

        cpu_ft.sse = cpu_ft.sse && !sse_turn_off;
        cpu_ft.sse2 = cpu_ft.sse2 && !sse2_turn_off;
        cpu_ft.sse3 = cpu_ft.sse3 && !sse3_turn_off;
        cpu_ft.ssse3 = cpu_ft.ssse3 && !ssse3_turn_off;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let neon_turn_off = std::env::var("PIRE_NEON_OFF").is_ok();
        let sve_turn_off = std::env::var("PIRE_SVE_OFF").is_ok();
        let fp16_turn_off = std::env::var("PIRE_FP16_OFF").is_ok();
        let f32mm_turn_off = std::env::var("PIRE_F32MM_OFF").is_ok();
        let fcma_turn_off = std::env::var("PIRE_FCMA_OFF").is_ok();
        let i8mm_turn_off = std::env::var("PIRE_I8MM_OFF").is_ok();

        cpu_ft.neon = cpu_ft.neon && !neon_turn_off;
        cpu_ft.sve = cpu_ft.sve && !sve_turn_off;
        cpu_ft.fp16 = cpu_ft.fp16 && !fp16_turn_off;
        cpu_ft.f32mm = cpu_ft.f32mm && !f32mm_turn_off;
        cpu_ft.fcma = cpu_ft.fcma && !fcma_turn_off;
        cpu_ft.i8mm = cpu_ft.i8mm && !i8mm_turn_off;
    }
}

#[cfg(not(feature = "debug_cpu_features"))]
pub static RUNTIME_HW_CONFIG: Lazy<HWConfig> = Lazy::new(|| detect_hw_config());
#[cfg(feature = "debug_cpu_features")]
pub static RUNTIME_HW_CONFIG: Lazy<HWConfig> = Lazy::new(|| {
    let mut hw_config = detect_hw_config();
    apply_debug_cpu_features(&mut hw_config.cpu_ft);
    hw_config
});

pub static PIRE_NUM_THREADS: Lazy<usize> = Lazy::new(|| {
    let n_core = std::thread::available_parallelism().unwrap().get();
    // PIRE_NUM_THREADS or the number of logical cores
    let x = std::env::var("PIRE_NUM_THREADS").unwrap_or(n_core.to_string());
    x.parse::<usize>().unwrap()
});
#[cfg(target_arch = "x86_64")]
pub(crate) mod cpu_features {
    use super::HWModel;
    use super::RUNTIME_HW_CONFIG;

    pub fn hw_model() -> HWModel {
        RUNTIME_HW_CONFIG.hw_model
    }

    pub fn has_f32_compute() -> bool {
        // RUNTIME_HW_CONFIG.cpu_ft.avx512f || RUNTIME_HW_CONFIG.cpu_ft.avx
        // dont use above since some avx512f also rely on avx instructions
        // (even though avx512f should imply), we are being super conservative here
        RUNTIME_HW_CONFIG.cpu_ft.avx || RUNTIME_HW_CONFIG.cpu_ft.sse
    }

    pub fn has_c32_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.avx || (RUNTIME_HW_CONFIG.cpu_ft.sse && RUNTIME_HW_CONFIG.cpu_ft.sse3)
    }

    pub fn has_f16f32_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.avx && RUNTIME_HW_CONFIG.cpu_ft.f16c
    }
    pub fn has_f64_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.avx || (RUNTIME_HW_CONFIG.cpu_ft.sse && RUNTIME_HW_CONFIG.cpu_ft.sse2)
    }

    pub fn has_c64_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.avx
            || (RUNTIME_HW_CONFIG.cpu_ft.sse && RUNTIME_HW_CONFIG.cpu_ft.sse2 && RUNTIME_HW_CONFIG.cpu_ft.sse3)
    }

    pub fn has_f16_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.avx512f16
            && RUNTIME_HW_CONFIG.cpu_ft.avx
            && RUNTIME_HW_CONFIG.cpu_ft.f16c
            && RUNTIME_HW_CONFIG.cpu_ft.fma
    }
    pub fn has_i16i32_compute() -> bool {
        (RUNTIME_HW_CONFIG.cpu_ft.avx2 && RUNTIME_HW_CONFIG.cpu_ft.avx)
            || (RUNTIME_HW_CONFIG.cpu_ft.sse && RUNTIME_HW_CONFIG.cpu_ft.sse2)
    }
    pub fn has_i8i32_compute() -> bool {
        (RUNTIME_HW_CONFIG.cpu_ft.avx2 && RUNTIME_HW_CONFIG.cpu_ft.avx)
            || (RUNTIME_HW_CONFIG.cpu_ft.sse && RUNTIME_HW_CONFIG.cpu_ft.sse2 && RUNTIME_HW_CONFIG.cpu_ft.ssse3)
    }
    // TODO: Use actual info from hardware
    pub fn get_cache_params() -> (usize, usize, usize) {
        (4800, 256, 128)
    }
}

#[cfg(target_arch = "x86")]
pub(crate) mod cpu_features {
    use super::HWModel;
    use super::RUNTIME_HW_CONFIG;

    pub fn hw_model() -> HWModel {
        RUNTIME_HW_CONFIG.hw_model
    }

    pub fn has_f32_compute() -> bool {
        // RUNTIME_HW_CONFIG.cpu_ft.avx512f || RUNTIME_HW_CONFIG.cpu_ft.avx
        // dont use above since some avx512f also rely on avx instructions
        // (even though avx512f should imply), we are being super conservative here
        RUNTIME_HW_CONFIG.cpu_ft.sse
    }

    pub fn has_c32_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.sse && RUNTIME_HW_CONFIG.cpu_ft.sse3
    }

    pub fn has_f16f32_compute() -> bool {
        false
    }
    pub fn has_f64_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.sse && RUNTIME_HW_CONFIG.cpu_ft.sse2
    }

    pub fn has_c64_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.sse && RUNTIME_HW_CONFIG.cpu_ft.sse2 && RUNTIME_HW_CONFIG.cpu_ft.sse3
    }

    pub fn has_f16_compute() -> bool {
        false
    }
    pub fn has_i16i32_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.sse && RUNTIME_HW_CONFIG.cpu_ft.sse2
    }
    pub fn has_i8i32_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.sse && RUNTIME_HW_CONFIG.cpu_ft.sse2 && RUNTIME_HW_CONFIG.cpu_ft.ssse3
    }
    // TODO: Use actual info from hardware
    pub fn get_cache_params() -> (usize, usize, usize) {
        (4800, 256, 128)
    }
}
#[cfg(target_arch = "aarch64")]
pub(crate) mod cpu_features {

    // neon is required for all the compute since
    // it is used for packing and unpacking and
    // available for all arch for which other extension are available
    // unless something weird happends with the vendor
    // For those (marginal) cases, we probably dont want to bother supporting
    use super::HWModel;
    use super::RUNTIME_HW_CONFIG;

    pub fn hw_model() -> HWModel {
        RUNTIME_HW_CONFIG.hw_model
    }

    pub fn has_f32_compute() -> bool {
        // RUNTIME_HW_CONFIG.cpu_ft.avx512f || RUNTIME_HW_CONFIG.cpu_ft.avx
        // dont use above since some avx512f also rely on avx instructions
        // (even though avx512f should imply), we are being super conservative here
        RUNTIME_HW_CONFIG.cpu_ft.neon
    }

    pub fn has_c32_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.neon
    }

    pub fn has_f16f32_compute() -> bool {
        false
    }
    pub fn has_f64_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.neon
    }

    pub fn has_c64_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.neon
    }

    pub fn has_f16_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.fp16 && RUNTIME_HW_CONFIG.cpu_ft.neon
    }
    pub fn has_i16i32_compute() -> bool {
        // currenty we do not support this
        // since the only insturction is smlal, whose throupout is not high enough
        false
    }
    pub fn has_i8i32_compute() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.i8mm && RUNTIME_HW_CONFIG.cpu_ft.neon
    }
    // TODO: Use actual info from hardware
    pub fn get_cache_params() -> (usize, usize, usize) {
        (4800, 256, 128)
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
pub(crate) mod cpu_features {
    use super::HWModel;
    use super::RUNTIME_HW_CONFIG;

    pub fn hw_model() -> HWModel {
        RUNTIME_HW_CONFIG.hw_model
    }

    pub fn has_f32_compute() -> bool {
        false
    }

    pub fn has_c32_compute() -> bool {
        false
    }

    pub fn has_f16f32_compute() -> bool {
        false
    }
    pub fn has_f64_compute() -> bool {
        false
    }

    pub fn has_c64_compute() -> bool {
        false
    }

    pub fn has_f16_compute() -> bool {
        false
    }
    pub fn has_i16i32_compute() -> bool {
        false
    }
    pub fn has_i8i32_compute() -> bool {
        false
    }
    pub fn get_cache_params() -> (usize, usize, usize) {
        (4800, 256, 128)
    }
}
pub use cpu_features::*;

pub struct PackPool {
    pub buffer: RwLock<Vec<Mutex<Vec<u8>>>>,
}

pub static PACK_POOL: PackPool = PackPool { buffer: RwLock::new(vec![]) };

pub fn acquire<'a>(
    pool_guard: &'a RwLockReadGuard<'a, Vec<Mutex<Vec<u8>>>>,
    pack_size: usize,
) -> Option<MutexGuard<'a, Vec<u8>>> {
    // find the first free buffer with enough size
    // let x = PACK_POOL.buffer.read().unwrap();
    for i in pool_guard.iter() {
        // TODO: this might be the most optimal algo in terms of fragmentation/meory reuse
        // It is very optimal for all cases (except for a few exceptional cases)
        // Exceptional cases: You have  mulththreading along mc loop that is changing in run time (so it requires varying number of packa pool for 1 gemm run)
        // This is exceptional since this can happen only if the threadConfig is created by user and threadconfig is changing its parallelsis along mc during run.
        // I cannot think of a rason why someone would do that (maybe unusual hardware, or just experimentation).
        // Also, the current algo is very simple and easy  to understand.
        let lock = i.try_lock();
        if let Ok(mutex) = lock {
            if mutex.len() >= pack_size {
                return Some(mutex);
            }
        }
    }

    None
}

pub fn extend<'a>(pool_vec: Vec<u8>) {
    let mut pool_guard = PACK_POOL.buffer.write().unwrap();
    pool_guard.push(Mutex::new(pool_vec));
}

pub struct PireThreadConfig<'a> {
    pub ic_id: usize,
    // pc_id: usize,
    pub jc_id: usize,
    pub ir_id: usize,
    pub jr_id: usize,
    pub i_load_p_idx: usize,
    pub j_load_p_idx: usize,
    pub mc_eff: usize,
    pub nc_eff: usize,
    pub kc_eff: usize,
    pub par: PirePar,
    pub packa_barrier: &'a [Barrier],
    pub packb_barrier: &'a [Barrier],
}

pub fn get_apbp_barrier(par: &PirePar) -> (Vec<Barrier>, Vec<Barrier>) {
    let mut packa_barrier = vec![];
    for _ in 0..par.ic_par {
        let barrier = Barrier::new(par.jc_par * par.pc_par * par.ir_par * par.jr_par);
        packa_barrier.push(barrier);
    }

    let mut packb_barrier = vec![];
    for _ in 0..par.jc_par {
        let barrier = Barrier::new(par.ic_par * par.pc_par * par.ir_par * par.jr_par);
        packb_barrier.push(barrier);
    }

    (packa_barrier, packb_barrier)
}

impl<'a> PireThreadConfig<'a> {
    pub fn new(
        par: PirePar,
        packa_barrier: &'a [Barrier],
        packb_barrier: &'a [Barrier],
        t_id: usize,
        mc_eff: usize,
        nc_eff: usize,
        kc_eff: usize,
    ) -> Self {
        let ic_id = par.get_ic_id(t_id);
        // let pc_id = par.get_pc_id(t_id);
        let jc_id = par.get_jc_id(t_id);
        let ir_id = par.get_ir_id(t_id);
        let jr_id = par.get_jr_id(t_id);
        let i_load_p_idx = jc_id * par.ir_par * par.jr_par + ir_id * par.jr_par + jr_id;
        let j_load_p_idx = ic_id * par.ir_par * par.jr_par + ir_id * par.jr_par + jr_id;

        Self {
            ic_id,
            // pc_id,
            jc_id,
            ir_id,
            jr_id,
            i_load_p_idx,
            j_load_p_idx,
            mc_eff,
            nc_eff,
            kc_eff,
            par,
            packa_barrier,
            packb_barrier,
        }
    }
    #[inline]
    pub fn wait_packa(&self) {
        if self.par.jc_par * self.par.pc_par * self.par.ir_par * self.par.jr_par > 1 {
            self.packa_barrier[self.ic_id].wait();
        }
    }

    #[inline]
    pub fn wait_packb(&self) {
        if self.par.ic_par * self.par.pc_par * self.par.ir_par * self.par.jr_par > 1 {
            self.packb_barrier[self.jc_id].wait();
        }
    }
}

// pub fn check_mem_size(mem_size: usize, rs: usize, cs: usize, m: usize, n: usize) {
//     assert!(mem_size >= rs * cs * m * n);
//     assert!(rs >= 1 && cs >= 1 && m >= 0 && n >= 0);
// }

// once this is read, this cannot be changed for the time being.
#[inline(always)]
pub fn pire_num_threads() -> usize {
    return *PIRE_NUM_THREADS;
}

#[derive(Copy, Clone)]
pub struct PirePar {
    pub num_threads: usize,
    pub ic_par: usize,
    pub pc_par: usize,
    pub jc_par: usize,
    pub ir_par: usize,
    pub jr_par: usize,
}

// greedy algo to distribute the number of threads evenly
// simple works for the time being
#[inline(always)]
fn inc_par(ic_par: usize, jc_par: usize, ic_max: usize, jc_max: usize, factor: usize) -> (usize, usize, usize, usize) {
    if (ic_par < jc_par && ic_par < ic_max) || (jc_par >= jc_max && ic_par < ic_max) {
        (ic_par * factor, jc_par, ic_max / factor, jc_max)
    } else if (ic_par >= jc_par && jc_par < jc_max) || (ic_par >= ic_max && jc_par < jc_max) {
        (ic_par, jc_par * factor, ic_max, jc_max / factor)
    } else {
        (ic_par, jc_par, ic_max, jc_max)
    }
}
impl PirePar {
    pub fn new(num_threads: usize, ic_par: usize, pc_par: usize, jc_par: usize, ir_par: usize, jr_par: usize) -> Self {
        assert_eq!(num_threads, jc_par * pc_par * ic_par * jr_par * ir_par);
        Self { num_threads, ic_par, pc_par, jc_par, ir_par, jr_par }
    }
    pub fn from_num_threads(num_threads: usize, m: usize, n: usize) -> Self {
        let mut num_threads = num_threads;
        let mut ic_par_max = if m < 96 {
            1
        } else if m < 400 {
            2
        } else {
            m / 200
        };
        let mut jc_par_max = if n < 48 {
            1
        } else if n < 200 {
            2
        } else {
            n / 100
        };

        if num_threads <= 12 {
            let jc_par_max = jc_par_max.min(num_threads);
            let n_thread = (num_threads / jc_par_max) * jc_par_max;
            return Self::new(n_thread, num_threads / jc_par_max, 1, jc_par_max, 1, 1);
        }
        // let mut jr_par_max = if k < 96 { 1 } else if jc_par_max => 4 { 4.min(k / 4) };
        num_threads = num_threads.min(ic_par_max * jc_par_max);
        let mut ic_par = 1;
        let pc_par = 1;
        let mut jc_par = 1;
        let mut ir_par = 1;
        let jr_par = 1;

        while num_threads > 1 {
            if num_threads % 2 == 0 {
                num_threads = num_threads / 2;
                (ic_par, jc_par, ic_par_max, jc_par_max) = inc_par(ic_par, jc_par, ic_par_max, jc_par_max, 2);
            } else if num_threads % 3 == 0 {
                num_threads = num_threads / 3;
                (ic_par, jc_par, ic_par_max, jc_par_max) = inc_par(ic_par, jc_par, ic_par_max, jc_par_max, 3);
            } else if num_threads % 5 == 0 {
                num_threads = num_threads / 5;
                (ic_par, jc_par, ic_par_max, jc_par_max) = inc_par(ic_par, jc_par, ic_par_max, jc_par_max, 5);
                continue;
            } else if num_threads % 7 == 0 {
                num_threads = num_threads / 7;
                (ic_par, jc_par, ic_par_max, jc_par_max) = inc_par(ic_par, jc_par, ic_par_max, jc_par_max, 7);
                continue;
            } else {
                // if it is non trivial prime factor (i.e. not divisible by 2,3,5,7)
                // round it so it is a "nice" number
                num_threads = num_threads / 2 * 2;
            }
            // if num_threads % 11 == 0 {
            //     num_threads = num_threads / 11;
            //     (ic_par, jc_par, ic_par_max, jc_par_max) = inc_par(ic_par, jc_par, ic_par_max, jc_par_max, 11);
            //     continue;
            // }
            // if num_threads % 13 == 0 {
            //     num_threads = num_threads / 13;
            //     (ic_par, jc_par, ic_par_max, jc_par_max) = inc_par(ic_par, jc_par, ic_par_max, jc_par_max, 13);
            //     continue;
            // }
            // if num_threads % 17 == 0 {
            //     num_threads = num_threads / 17;
            //     (ic_par, jc_par, ic_par_max, jc_par_max) = inc_par(ic_par, jc_par, ic_par_max, jc_par_max, 17);
            //     continue;
            // }
        }
        if ic_par >= 8 {
            ic_par = ic_par / 2;
            ir_par = 2;
        }
        let num_threads = ic_par * pc_par * jc_par * ir_par * jr_par;
        Self { num_threads, ic_par, pc_par, jc_par, ir_par, jr_par }
    }
    #[inline(always)]
    pub fn default(m: usize, n: usize) -> Self {
        let num_threads = pire_num_threads();
        Self::from_num_threads(num_threads, m, n)
    }
    #[inline]
    fn get_ic_id(&self, t_id: usize) -> usize {
        (t_id / (self.pc_par * self.jc_par * self.ir_par * self.jr_par)) % self.ic_par
    }

    //    #[inline]
    //    fn get_pc_id(&self, t_id: usize) -> usize {
    //        (t_id / (self.jr_par*self.ir_par*self.ic_par)) % self.pc_par
    //    }
    #[inline]
    fn get_jc_id(&self, t_id: usize) -> usize {
        (t_id / (self.jr_par * self.ir_par)) % self.jc_par
    }
    #[inline]
    fn get_jr_id(&self, t_id: usize) -> usize {
        (t_id / self.ir_par) % self.jr_par
    }
    #[inline]
    fn get_ir_id(&self, t_id: usize) -> usize {
        t_id % self.ir_par
    }

    pub fn get_load_par(
        &self,
        gemm_mode: &GemmPool,
        m: usize,
        n: usize,
        mc_eff: usize,
        nc_eff: usize,
    ) -> (usize, usize) {
        let m = (m / self.ic_par).min(mc_eff);
        let n = (n / self.jc_par).min(nc_eff);
        let i_load_par = ((m + 127) / 128).min(self.num_threads / self.ic_par);
        let j_load_par = ((n + 127) / 128).min(self.num_threads / self.jc_par);
        let i_load_par = match gemm_mode {
            GemmPool::Goto => i_load_par,
            GemmPool::SmallM => i_load_par,
            GemmPool::SmallN => 1,
            GemmPool::SmallMN => 1,
        };
        (i_load_par.max(1), j_load_par.max(1))
    }
}

#[inline]
pub fn split_c_range(m: usize, mc: usize, mr: usize, ic_id: usize, ic_par: usize) -> (usize, usize, bool) {
    let chunk_len = (m / (mr * ic_par)) * mr;
    let rem = m % (mr * ic_par);
    if ic_id == 0 {
        let x = chunk_len + rem % mr;
        let mc_left = ((((x + mc - 1) / mc) * mc) * ic_par) < m;
        return (m - chunk_len - (rem % mr), m, mc_left);
    }
    let ic_id = ic_id - 1;
    let m0 = (m / mr) * mr;
    let rem = m0 % (mr * ic_par);
    let start_delta = rem.min(ic_id * mr);
    let end_delta = rem.min((ic_id + 1) * mr);
    //    let is_m_boundary = (chunk_len + end_delta - start_delta ) % mc == 0;
    let mc_coeff = (chunk_len + end_delta - start_delta + mc - 1) / mc;
    let mc_left = ((mc_coeff * mc) * ic_par) < m;
    //    let mc_left = is_m_boundary && rem != 0 && end_delta == start_delta;
    (chunk_len * ic_id + start_delta, chunk_len * (ic_id + 1) + end_delta, mc_left)
}

#[inline]
pub fn split_range(range_len: usize, unit_len: usize, r_id: usize, r_par: usize) -> (usize, usize) {
    let chunk_start = (range_len / (unit_len * r_par)) * unit_len * r_id;
    let chunk_end = (range_len / (unit_len * r_par)) * unit_len * (r_id + 1);
    let rem = range_len % (unit_len * r_par);
    let rem = rem - rem % unit_len;
    let rem_start = rem.min(r_id * unit_len);
    let rem_end = rem.min((r_id + 1) * unit_len);
    if r_id == r_par - 1 {
        return (chunk_start + rem_start, range_len);
    }
    (chunk_start + rem_start, chunk_end + rem_end)
}

pub trait BaseNum: Copy + 'static + Send {}

impl<T> BaseNum for T where T: Copy + 'static + Send {}

#[derive(Copy, Clone)]
pub struct PoolSize {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub ap_pool_size: usize,
    pub ap_pool_multiplicity: usize,
    pub bp_pool_size: usize,
    pub bp_pool_multiplicity: usize,
}

impl PoolSize {
    // add alignment padding for ab only for total memory pool sizes
    pub fn mem_pool_size_b<TA, TB>(&self) -> usize {
        // be conservative and add 2 * AB_ALIGN padding always
        self.ap_pool_size * size_of::<TA>() * self.ap_pool_multiplicity
            + self.bp_pool_size * size_of::<TB>() * self.bp_pool_multiplicity
            + 2 * AB_ALIGN
    }

    pub fn ap_size_b<TA>(&self) -> usize {
        self.ap_pool_size * size_of::<TA>()
    }

    pub fn bp_size_b<TB>(&self) -> usize {
        self.bp_pool_size * size_of::<TB>()
    }

    pub fn ap_size_t_b<TA>(&self) -> usize {
        self.ap_pool_size * size_of::<TA>() * self.ap_pool_multiplicity
    }

    pub fn bp_size_t_b<TB>(&self) -> usize {
        self.bp_pool_size * size_of::<TB>() * self.bp_pool_multiplicity
    }

    pub fn slice_mut_from_pool<TA, TB>(
        &self,
        mem_pool: &mut [u8],
        i_load_par: usize,
        j_load_par: usize,
        pool_size: PoolSize,
        mr: usize,
        nr: usize,
        // mc: usize, nc: usize, kc: usize, mr: usize, nr: usize,
    ) -> (Vec<RangeLock<'_, TA>>, Vec<RangeLock<'_, TB>>) {
        let m_size = pool_size.m;
        let n_size = pool_size.n;
        let k_size = pool_size.k;
        let ap_pool_size = self.ap_pool_size;
        let ap_pool_size_b = ap_pool_size * size_of::<TA>();
        let a_alignment = core::mem::align_of::<TA>();
        assert_eq!(ap_pool_size_b % a_alignment, 0);
        let bp_pool_size = self.bp_pool_size;
        let bp_pool_size_b = bp_pool_size * size_of::<TB>();
        let b_alignment = core::mem::align_of::<TB>();
        assert_eq!(bp_pool_size_b % b_alignment, 0);
        let mut ap = vec![];
        let mut bp = vec![];
        // safety for pointer to slice casting: assert len of mem_pool is enough
        // ap_pool_size
        assert!(mem_pool.len() >= self.mem_pool_size_b::<TA, TB>());
        // align mem_pool
        let align_offset = mem_pool.as_ptr().align_offset(AB_ALIGN);
        let mut mem_pool = &mut mem_pool[align_offset..];
        // safety for pointer to slice casting: ap has right alignment
        assert_eq!(mem_pool.as_ptr().align_offset(a_alignment), 0);
        for _ in 0..self.ap_pool_multiplicity {
            let (a, rest) = mem_pool.split_at_mut(ap_pool_size_b);
            let ap_pool = unsafe { core::slice::from_raw_parts_mut::<TA>(a.as_mut_ptr() as *mut TA, ap_pool_size) };
            if ap_pool_size == 0 {
                ap.push(RangeLock::from(ap_pool, i_load_par, 0, k_size, mr));
            } else {
                ap.push(RangeLock::from(ap_pool, i_load_par, m_size, k_size, mr));
            }
            mem_pool = rest;
        }
        let align_offset = mem_pool.as_ptr().align_offset(AB_ALIGN);
        let mut mem_pool = &mut mem_pool[align_offset..];
        // safety for pointer to slice casting: bp has right alignment
        assert_eq!(mem_pool.as_ptr().align_offset(b_alignment), 0);
        for _ in 0..self.bp_pool_multiplicity {
            let (b, rest) = mem_pool.split_at_mut(bp_pool_size_b);
            let bp_pool = unsafe { core::slice::from_raw_parts_mut::<TB>(b.as_mut_ptr() as *mut TB, bp_pool_size) };
            if bp_pool_size == 0 {
                bp.push(RangeLock::from(bp_pool, j_load_par, 0, k_size, nr));
            } else {
                bp.push(RangeLock::from(bp_pool, j_load_par, n_size, k_size, nr));
            }
            mem_pool = rest;
        }
        (ap, bp)
    }
}

pub fn get_mem_pool_size_goto<AP: BaseNum, BP: BaseNum, HWConfig: GemmCache>(
    hw_config: &HWConfig,
    par: &PirePar,
    a_need_pool: bool,
    b_need_pool: bool,
) -> PoolSize {
    let m = hw_config.get_mc_eff(par.ic_par);
    let n = hw_config.get_nc_eff(par.jc_par);
    let k = hw_config.get_kc_eff();
    let (ap_pool_size, ap_pool_multiplicity) = if a_need_pool {
        let ap_pool_multiplicity = par.ic_par;
        let ap_pool_size = hw_config.get_ap_pool_size(par.ic_par) + CACHELINE_PAD / size_of::<AP>();
        (ap_pool_size, ap_pool_multiplicity)
    } else {
        (0, 1)
    };
    let (bp_pool_size, bp_pool_multiplicity) = if b_need_pool {
        let bp_pool_multiplicity = par.jc_par;
        let bp_pool_size = hw_config.get_bp_pool_size(par.jc_par) + CACHELINE_PAD / size_of::<BP>();
        (bp_pool_size, bp_pool_multiplicity)
    } else {
        (0, 1)
    };
    PoolSize { m, n, k, ap_pool_size, ap_pool_multiplicity, bp_pool_size, bp_pool_multiplicity }
}

pub fn get_mem_pool_size_small_m<AP: BaseNum, BP: BaseNum, HWConfig: GemmCache>(
    hw_config: &HWConfig,
    par: &PirePar,
    a_need_pool: bool,
) -> PoolSize {
    let m = hw_config.get_mc_eff(par.ic_par);
    let n = hw_config.get_nc_eff(par.jc_par);
    let k = hw_config.get_kc_eff();
    if a_need_pool {
        let ap_pool_multiplicity = par.ic_par;
        let ap_pool_size = hw_config.get_ap_pool_size(par.ic_par) + CACHELINE_PAD / size_of::<AP>();
        PoolSize { m, n, k, ap_pool_size, ap_pool_multiplicity, bp_pool_size: 0, bp_pool_multiplicity: 1 }
    } else {
        PoolSize { m, n, k, ap_pool_size: 0, ap_pool_multiplicity: 1, bp_pool_size: 0, bp_pool_multiplicity: 1 }
    }
}

pub fn get_mem_pool_size_small_n<AP: BaseNum, BP: BaseNum, HWConfig: GemmCache>(
    hw_config: &HWConfig,
    par: &PirePar,
    b_need_pool: bool,
) -> PoolSize {
    let ap_pool_size = hw_config.get_ap_pool_size2() + CACHELINE_PAD / size_of::<AP>();
    let ap_pool_multiplicity = par.num_threads;
    let m = hw_config.mr();
    let n = hw_config.get_nc_eff(par.jc_par);
    let k = hw_config.get_kc_eff();
    if b_need_pool {
        let bp_pool_multiplicity = par.jc_par;
        let bp_pool_size = hw_config.get_bp_pool_size(par.jc_par) + CACHELINE_PAD / size_of::<BP>();
        PoolSize { m, n, k, ap_pool_size, ap_pool_multiplicity, bp_pool_size, bp_pool_multiplicity }
    } else {
        PoolSize { m, n, k, ap_pool_size, ap_pool_multiplicity, bp_pool_size: 0, bp_pool_multiplicity: 1 }
    }
}

// Choose ap_size, bp_size as arguments since they are specific to Gemm implementation,
// It is determined by hardware, gemm implementation (e.g. f64, f32, f16),
// Otherwise, this base crate would include code coupled with other gemm crates,
// this would require either cyclic dep (Not allowed of course) or separate code for each specii hardware and gemm
// imple inside this crate, which is not desirable. We want this crate to be as decoupled as possbile from
// specific gemm implementation and hardware.

pub fn run_small_m(m: usize) -> bool {
    m < 144
}

pub fn run_small_n(n: usize) -> bool {
    n < 144
}

pub enum GemmPool {
    Goto,
    SmallM,
    SmallN,
    SmallMN,
}

#[derive(Clone, Copy)]
pub struct StridedMatrix<T> {
    pub(crate) src: *const T,
    pub(crate) rs: usize,
    pub(crate) cs: usize,
}

impl<T> StridedMatrix<T> {
    pub fn new(src: *const T, rs: usize, cs: usize) -> Self {
        Self { src, rs, cs }
    }
}

unsafe impl<T> Send for StridedMatrix<T> {}

#[derive(Clone, Copy)]
pub struct StridedMatrixMut<T> {
    pub(crate) src: *mut T,
    pub(crate) rs: usize,
    pub(crate) cs: usize,
}

unsafe impl<T> Send for StridedMatrixMut<T> {}

impl<T> StridedMatrixMut<T> {
    pub fn new(src: *mut T, rs: usize, cs: usize) -> Self {
        Self { src, rs, cs }
    }
}

#[derive(Clone)]
pub struct StridedMatrixP<'a, T, U> {
    pub(crate) src: *const T,
    pub(crate) rs: usize,
    pub(crate) cs: usize,
    pub(crate) dst: &'a RangeLock<'a, U>,
}

unsafe impl<'a, T, U> Send for StridedMatrixP<'a, T, U> {}

impl<'a, T, U> StridedMatrixP<'a, T, U> {
    pub fn src(&self) -> *const T {
        self.src
    }
    pub fn dst_write(&self, idx: usize, kc: usize) -> RangeLockWriteGuard<'a, 'a, U> {
        self.dst.write(idx, kc).unwrap()
    }
    pub fn dst_read(&self) -> RangeLockReadGuard<'a, 'a, U> {
        self.dst.read().unwrap()
    }
    pub fn get_mc(&self) -> usize {
        self.dst.get_mc()
    }
    pub fn rs(&self) -> usize {
        self.rs
    }
    pub fn cs(&self) -> usize {
        self.cs
    }
}

#[derive(Clone, Copy)]
pub struct PackedMatrix<T> {
    pub(crate) src: *const T,
    pub(crate) k: usize,
    pub(crate) m: usize,
    // pub(crate) m0: usize,
    // pub(crate) k0: usize,
}

unsafe impl<T> Send for PackedMatrix<T> {}

impl<T> PackedMatrix<T> {
    pub fn src(&self) -> *const T {
        self.src
    }
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn m(&self) -> usize {
        self.m
    }
    // pub fn at(&self, m: usize, k: usize) -> *const T {
    //     let m_rounded = (m+m0-1) / m0 * m0;
    //     self.src.add(m_rounded)
}

#[derive(Clone)]
pub struct PackedMatrixMixed<'a, X, Y> {
    pub(crate) src: *const X,
    pub(crate) dst: &'a RangeLock<'a, Y>,
    pub(crate) k: usize,
    pub(crate) m: usize,
}

impl<'a, X, Y> PackedMatrixMixed<'a, X, Y> {
    pub fn src(&self) -> *const X {
        self.src
    }
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn m(&self) -> usize {
        self.m
    }

    pub fn dst_write(&self, idx: usize, kc: usize) -> RangeLockWriteGuard<'a, 'a, Y> {
        self.dst.write(idx, kc).unwrap()
    }

    pub fn get_mc(&self) -> usize {
        self.dst.get_mc()
    }

    pub fn dst_read(&self) -> RangeLockReadGuard<'a, 'a, Y> {
        self.dst.read().unwrap()
    }
}

unsafe impl<X, Y> Send for PackedMatrixMixed<'_, X, Y> {}

// must be multiple largest vector size that we support
// Now, it avx512 -> 64 bytes
pub const AB_ALIGN: usize = 1024;

pub trait GemmCache {
    fn mr(&self) -> usize;
    fn get_mc_eff(&self, par: usize) -> usize;
    fn get_kc_eff(&self) -> usize;
    fn get_nc_eff(&self, par: usize) -> usize;
    fn get_ap_pool_size(&self, ic_par: usize) -> usize {
        let mc_eff = self.get_mc_eff(ic_par);
        let kc_eff = self.get_kc_eff();
        mc_eff * kc_eff
    }
    fn get_ap_pool_size2(&self) -> usize {
        let kc_eff = self.get_kc_eff();
        self.mr() * kc_eff
    }
    fn get_bp_pool_size(&self, jc_par: usize) -> usize {
        let nc_eff = self.get_nc_eff(jc_par);
        let kc_eff = self.get_kc_eff();
        nc_eff * kc_eff
    }
}

#[derive(Copy, Clone)]
pub enum Array<X> {
    StridedMatrix(StridedMatrix<X>),
    PackedMatrix(PackedMatrix<X>),
}

impl<X> Array<X> {
    pub fn strided_matrix(src: *const X, rs: usize, cs: usize) -> Self {
        Array::StridedMatrix(StridedMatrix::new(src, rs, cs))
    }
    pub fn packed_matrix(src: *const X, m: usize, k: usize) -> Self {
        Array::PackedMatrix(PackedMatrix { src, k, m })
    }
    pub fn into_pack_array<'a>(&self, a: &'a [RangeLock<'a, X>], p_id: usize) -> PArray<'a, X> {
        match self {
            Array::StridedMatrix(x) => {
                let x = StridedMatrixP { src: x.src, rs: x.rs, cs: x.cs, dst: &a[p_id] };
                PArray::<X>::StridedMatrix(x)
            }
            Array::PackedMatrix(x) => {
                let x = PackedMatrix { src: x.src, k: x.k, m: x.m };
                PArray::PackedMatrix(x)
            }
        }
    }
    pub fn into_pack_array2<'a, Y>(&self, a: &'a [RangeLock<'a, Y>], p_id: usize) -> PArrayMixed<'a, X, Y> {
        match self {
            Array::StridedMatrix(x) => {
                let x = StridedMatrixP { src: x.src, rs: x.rs, cs: x.cs, dst: &a[p_id] };
                PArrayMixed::<X, Y>::StridedMatrix(x)
            }
            Array::PackedMatrix(x) => {
                let x = PackedMatrixMixed { src: x.src, dst: &a[p_id], k: x.k, m: x.m };
                PArrayMixed::PackedMatrix(x)
            }
        }
    }

    pub fn src(&self) -> *const X {
        match self {
            Array::StridedMatrix(x) => x.src,
            Array::PackedMatrix(x) => x.src,
        }
    }

    pub fn transpose(&mut self) {
        match self {
            Array::StridedMatrix(x) => {
                let temp = x.rs;
                x.rs = x.cs;
                x.cs = temp;
            }
            _ => {
                panic!("Only StridedMatrix has transpose");
            }
        }
    }

    pub fn rs(&self) -> usize {
        match self {
            Array::StridedMatrix(x) => x.rs,
            _ => {
                panic!("Only StridedMatrix has rs");
            }
        }
    }

    pub fn cs(&self) -> usize {
        match self {
            Array::StridedMatrix(x) => x.cs,
            _ => {
                panic!("Only StridedMatrix has cs");
            }
        }
    }

    pub fn is_strided(&self) -> bool {
        match self {
            Array::StridedMatrix(_) => true,
            _ => false,
        }
    }
}

#[derive(Copy, Clone)]
pub enum ArrayMut<X> {
    StridedMatrix(StridedMatrixMut<X>),
}

impl<X> ArrayMut<X> {
    pub fn strided_matrix(src: *mut X, rs: usize, cs: usize) -> Self {
        ArrayMut::StridedMatrix(StridedMatrixMut::new(src, rs, cs))
    }

    pub fn src(&self) -> *mut X {
        match self {
            ArrayMut::StridedMatrix(x) => x.src,
        }
    }

    pub fn transpose(&mut self) {
        match self {
            ArrayMut::StridedMatrix(x) => {
                let temp = x.rs;
                x.rs = x.cs;
                x.cs = temp;
            }
        }
    }

    pub fn rs(&self) -> usize {
        match self {
            ArrayMut::StridedMatrix(x) => x.rs,
        }
    }

    pub fn cs(&self) -> usize {
        match self {
            ArrayMut::StridedMatrix(x) => x.cs,
        }
    }
}

#[derive(Clone)]
pub enum PArray<'a, X> {
    StridedMatrix(StridedMatrixP<'a, X, X>),
    PackedMatrix(PackedMatrix<X>),
}

impl<'a, X> PArray<'a, X> {
    pub fn src(&self) -> *const X {
        match self {
            Self::StridedMatrix(x) => x.src,
            Self::PackedMatrix(x) => x.src,
        }
    }

    pub fn rs(&self) -> usize {
        match self {
            Self::StridedMatrix(x) => x.rs,
            _ => {
                panic!("Only StridedMatrix has rs");
            }
        }
    }

    pub fn cs(&self) -> usize {
        match self {
            Self::StridedMatrix(x) => x.cs,
            _ => {
                panic!("Only StridedMatrix has cs");
            }
        }
    }

    pub fn dst_write(&self, idx: usize, kc: usize) -> RangeLockWriteGuard<'a, 'a, X> {
        match self {
            Self::StridedMatrix(x) => x.dst.write(idx, kc).unwrap(),
            _ => {
                panic!("Only StridedMatrix has write guard");
            }
        }
    }

    pub fn dst_read(&self) -> RangeLockReadGuard<'a, 'a, X> {
        match self {
            Self::StridedMatrix(x) => x.dst.read().unwrap(),
            _ => {
                panic!("Only StridedMatrix has read guard");
            }
        }
    }

    pub fn is_strided(&self) -> bool {
        match self {
            Self::StridedMatrix(_) => true,
            _ => false,
        }
    }
}

#[derive(Clone)]
pub enum PArrayMixed<'a, X, Y> {
    StridedMatrix(StridedMatrixP<'a, X, Y>),
    PackedMatrix(PackedMatrixMixed<'a, X, Y>),
}

impl<'a, X, Y> PArrayMixed<'a, X, Y> {
    pub fn src(&self) -> *const X {
        match self {
            Self::StridedMatrix(x) => x.src,
            Self::PackedMatrix(x) => x.src,
        }
    }

    pub fn rs(&self) -> usize {
        match self {
            Self::StridedMatrix(x) => x.rs,
            _ => {
                panic!("Only StridedMatrix has rs");
            }
        }
    }

    pub fn cs(&self) -> usize {
        match self {
            Self::StridedMatrix(x) => x.cs,
            _ => {
                panic!("Only StridedMatrix has cs");
            }
        }
    }

    pub fn dst_write(&self, idx: usize, kc: usize) -> RangeLockWriteGuard<'a, 'a, Y> {
        match self {
            Self::StridedMatrix(x) => x.dst.write(idx, kc).unwrap(),
            Self::PackedMatrix(x) => x.dst.write(idx, kc).unwrap(),
        }
    }
    pub fn dst_read(&self) -> RangeLockReadGuard<'a, 'a, Y> {
        match self {
            Self::StridedMatrix(x) => x.dst.read().unwrap(),
            Self::PackedMatrix(x) => x.dst.read().unwrap(),
        }
    }
    pub fn is_strided(&self) -> bool {
        match self {
            Self::StridedMatrix(_) => true,
            _ => false,
        }
    }
}

pub enum PtrData<'a, X> {
    RefData(RangeLockReadGuard<'a, 'a, X>),
    PtrData(*const X),
}

impl<'a, X> PtrData<'a, X> {
    pub fn src(&self) -> *const X {
        match self {
            PtrData::RefData(x) => x.get().as_ptr(),
            PtrData::PtrData(x) => x.clone(),
        }
    }
}

pub fn matrix_size_strided(m: usize, n: usize, rs: usize, cs: usize) -> usize {
    (m - 1) * rs + (n - 1) * cs
}

#[macro_export]
macro_rules! packing_api {
    ($ta:ty, $tb:ty) => {
        fn a_size_packed(m: usize, k: usize) -> usize {
            let round_m_fn = dispatch_round_m();
            let round_k_fn = dispatch_round_k();
            let m_round = round_m_fn(m);
            let k_round = round_k_fn(k);
            return m_round * k_round;
        }

        fn b_size_packed(n: usize, k: usize) -> usize {
            let round_k_fn = dispatch_round_k();
            let k_round = round_k_fn(k);
            return n * k_round;
        }
        // block idx for packa and packb is s.t.
        // m dim for block idx is contiguous and n dim is contiguous
        // this is to ensure that indexing for parallelization over these dims are easy  (otherwise ranges would have to be in the same mc, nc range)
        // this is not an issue since we do not parallelize over k dim (think about this when we parallelize over k dim in the future, which is only beneficial only
        // in the special case of very large k and small m, n

        /// # Safety
        ///
        /// a and ap must have big enough size to store the packed matrix
        pub unsafe fn pack_a_unchecked(
            m: usize,
            k: usize,
            a: *const $ta,
            a_rs: usize,
            a_cs: usize,
            ap: *mut $ta,
        ) -> Array<TA> {
            assert_eq!(ap.align_offset(AB_ALIGN), 0);
            if m == 1 {
                for j in 0..k {
                    *ap.add(j) = *a.add(j * a_cs);
                }
                return Array::strided_matrix(ap, 1, m);
            }
            let pack_fn = dispatch_pack_a();
            let round_m_fn = dispatch_round_m();
            let round_k_fn = dispatch_round_k();

            let (mc, _, kc) = dispatch_get_mcnckc();
            let mut ap_cur = ap;
            for p in (0..k).step_by(kc) {
                let kc_len = kc.min(k - p);
                let kc_len_eff = round_k_fn(kc_len);
                for i in (0..m).step_by(mc) {
                    let mc_len = mc.min(m - i);
                    let mc_len_eff = round_m_fn(mc_len);
                    let a_cur = a.add(i * a_rs + p * a_cs);
                    pack_fn(a_cur, ap_cur, mc_len, kc_len, a_rs, a_cs);
                    ap_cur = ap_cur.add(mc_len_eff * kc_len_eff);
                }
            }
            return Array::packed_matrix(ap, m, k);
        }

        /// # Safety
        ///
        /// b and bp must have big enough size to store the packed matrix
        pub unsafe fn pack_b_unchecked(
            n: usize,
            k: usize,
            b: *const $tb,
            b_rs: usize,
            b_cs: usize,
            bp: *mut $tb,
        ) -> Array<TB> {
            assert_eq!(bp.align_offset(AB_ALIGN), 0);
            if n == 1 {
                for j in 0..k {
                    *bp.add(j) = *b.add(j * b_rs);
                }
                return Array::strided_matrix(bp, 1, k);
            }
            let pack_fn = dispatch_pack_b();
            let round_k_fn = dispatch_round_k();

            let (_, nc, kc) = dispatch_get_mcnckc();
            let mut bp_cur = bp;
            for p in (0..k).step_by(kc) {
                let kc_len = kc.min(k - p);
                let kc_len_eff = round_k_fn(kc_len);
                for i in (0..n).step_by(nc) {
                    let nc_len = nc.min(n - i);
                    let b_cur = b.add(i * b_cs + p * b_rs);
                    pack_fn(b_cur, bp_cur, nc_len, kc_len, b_rs, b_cs);
                    bp_cur = bp_cur.add(nc_len * kc_len_eff);
                }
            }
            return Array::packed_matrix(bp, n, k);
        }

        pub fn pack_a(m: usize, k: usize, a: &[$ta], a_rs: usize, a_cs: usize, ap: &mut [$ta]) -> Array<TA> {
            // panics if ap does not have enough size
            // safety check for size
            assert!(ap.len() >= a_size_packed(m, k));
            assert!(a.len() >= pire_base::matrix_size_strided(m, k, a_rs, a_cs));
            // safety: ap has enough size due to the assert above
            unsafe { pack_a_unchecked(m, k, a.as_ptr(), a_rs, a_cs, ap.as_mut_ptr()) }
        }

        pub fn pack_b(n: usize, k: usize, b: &[$tb], b_rs: usize, b_cs: usize, bp: &mut [$tb]) -> Array<TB> {
            // panics if bp does not have enough size
            // safety check for size
            assert!(bp.len() >= b_size_packed(n, k));
            assert!(b.len() >= pire_base::matrix_size_strided(k, n, b_rs, b_cs));
            // safety: bp has enough size due to the assert above
            unsafe { pack_b_unchecked(n, k, b.as_ptr(), b_rs, b_cs, bp.as_mut_ptr()) }
        }
    };
}

#[macro_export]
macro_rules! is_mixed {
    (T, $st1:expr, $st2:expr) => {
        $st1
    };
    (F, $src:expr, $st2:expr) => {
        $st2
    };
}

#[macro_export]
macro_rules! def_pa {
    ($packa_ty:tt,F,$ta:tt,$tap:tt) => {
        type $packa_ty<'a> = PArray<'a, $tap>;
    };
    ($packa_ty:tt,T,$ta:tt,$tap:tt) => {
        type $packa_ty<'a> = PArrayMixed<'a, $ta, $tap>;
    };
}

#[macro_export]
macro_rules! def_pire_gemm {
    (
        $t_dispatcher:tt,
        $ta:tt,$tap:ty,$tb:ty,$tbp:ty,$tc:ty,$t_as:ty,$t_bs:ty,
        $packa_ty:tt,$packb_ty:tt,
        $one:expr,
        $name:ident, $name_mt:ident,
        $goto_name:ident, $goto_kernel:ident,
        $small_m_name:ident, $small_m_kernel:ident,
        $small_n_name:ident, $small_n_kernel:ident,
        $small_mn_name:ident, $small_mn_kernel:ident,
        $gemv_name:ident, $gemv_name2:ident,
        $packa_name:ident, $packb_name:ident,
        $packa_name0:ident, $packb_name0:ident,
        $run_small_m:expr, $run_small_n:expr, $run_small_mn:expr,
        $pack_fn:tt, $include_flag:tt,
    ) => {
        def_pa!($packa_ty,$include_flag,$ta,$tap);
        def_pa!($packb_ty,$include_flag,$tb,$tbp);
        pub(crate) unsafe fn $name <F:UnaryFnC>(
            hw_config: &$t_dispatcher <F>,
            m: usize, n: usize, k: usize,
            alpha: $t_as,
            a: Array<$ta>,
            b: Array<$tb>,
            beta: $t_bs,
            c: ArrayMut<$tc>,
            par: &PirePar,
        )
        {
            let a_need_pool = a.is_strided() || !hw_config.is_compute_native();
            let b_need_pool = b.is_strided() || !hw_config.is_compute_native();
            if n == 1 && a.is_strided() {
                let alpha = &alpha as *const $t_as;
                let beta = &beta as *const $t_bs;
                $gemv_name(hw_config, m, k, alpha, a, b, beta, c);
                return;
            }
            if m == 1 && b.is_strided() {
                let alpha = &alpha as *const $t_as;
                let beta = &beta as *const $t_bs;
                let mut a = a;
                a.transpose();
                let mut b = b;
                b.transpose();
                let mut c = c;
                c.transpose();
                $gemv_name2(hw_config, n, k, alpha.into(), b, a, beta, c);
                return;
            }
            let is_dot_kernel = if a.is_strided() && b.is_strided() {
                a.cs() == 1 && b.rs() == 1 && (m < 300 || n < 300)
            } else {
                false
            };
            let (gemm_mode, gemm_fun, pool_info)
            : (
                GemmPool, unsafe fn(
                    &$t_dispatcher <F>, usize, usize, usize, *const $t_as, $packa_ty, $packb_ty, *const $t_bs, ArrayMut<$tc>, &PireThreadConfig,
                ),
                PoolSize
            ) = if $run_small_mn && is_dot_kernel {
                (GemmPool::SmallMN, $small_mn_name, get_mem_pool_size_goto::<$tap,$tbp,$t_dispatcher::<F>>(hw_config, par, a_need_pool, b_need_pool))
             }
            else if run_small_m(m) && $run_small_m && b.is_strided() {
                (GemmPool::SmallM, $small_m_name, get_mem_pool_size_small_m::<$tap,$tbp,$t_dispatcher::<F>>(hw_config, par, a_need_pool))
            } else if run_small_n(n) && $run_small_n && a.is_strided() {
                (GemmPool::SmallN, $small_n_name, get_mem_pool_size_small_n::<$tap,$tbp,$t_dispatcher::<F>>(hw_config, par, b_need_pool))
            } else {
                (GemmPool::Goto, $goto_name, get_mem_pool_size_goto::<$tap,$tbp,$t_dispatcher::<F>>(hw_config, par, a_need_pool, b_need_pool))
            };
            let mem_pool_size = pool_info.mem_pool_size_b::<$tap,$tbp>();
            // TODO: zero pool size case is very special (aonly packed and b) to optimize, optimization will not be worth it
            // if mem_pool_size == 0 {
            //     let mut pool_vec = [0_u8; 1];
            //     let pool_buf = &mut pool_vec;
            //     $name_mt(
            //         hw_config, m, n, k, alpha, a, b, beta, c, par, pool_buf, gemm_mode, pool_info, gemm_fun
            //     );
            //     return;
            // }
            // run goto algo
            {
                let pool_guard = PACK_POOL.buffer.read().unwrap();
                let y = acquire(&pool_guard, mem_pool_size);
                if let Some(mut pool_vec) = y {
                    let pool_buf = &mut pool_vec;
                    $name_mt(
                        hw_config, m, n, k, alpha, a, b, beta, c, par, pool_buf, gemm_mode, pool_info, gemm_fun
                    );
                    return;
                }
            }
            let mut pool_vec = vec![0_u8; mem_pool_size];
            let pool_buf = &mut pool_vec;
            $name_mt(
                hw_config, m, n, k, alpha, a, b, beta, c, par, pool_buf, gemm_mode, pool_info, gemm_fun
            );
            extend(pool_vec);
        }

        pub(crate) unsafe fn $name_mt<F:UnaryFnC>(
            hw_config: &$t_dispatcher <F>,
            m: usize, n: usize, k: usize,
            alpha: $t_as,
            a: Array<$ta>,
            b: Array<$tb>,
            beta: $t_bs,
            c: ArrayMut<$tc>,
            par: &PirePar,
            pool_buf: &mut [u8],
            gemm_mode: GemmPool,
            pool_info: PoolSize,
            gemm_fn: unsafe fn(
                &$t_dispatcher <F>, usize, usize, usize, *const $t_as, $packa_ty, $packb_ty, *const $t_bs, ArrayMut<$tc>, &PireThreadConfig
            )
        )
        where $t_dispatcher <F>: GemmCache
        {

            let mc_eff = <$t_dispatcher::<F> as GemmCache>::get_mc_eff(hw_config, par.ic_par);
            let nc_eff = <$t_dispatcher::<F> as GemmCache>::get_nc_eff(hw_config, par.jc_par);
            let kc_eff = <$t_dispatcher::<F> as GemmCache>::get_kc_eff(hw_config);
            let (pa_br_vec_ref, pb_br_vec_ref) = get_apbp_barrier(par);

            let (i_load_par, j_load_par) = par.get_load_par(&gemm_mode, m, n, mc_eff, nc_eff);
            let (ap_pool_vec, bp_pool_vec) = pool_info.slice_mut_from_pool::<$tap,$tbp>(
                pool_buf, i_load_par, j_load_par, pool_info, hw_config.mr, hw_config.nr
            );
            let (ap_pool, bp_pool) = (&ap_pool_vec, &bp_pool_vec);

            // remove par.clone
            std::thread::scope(|s| {
                for t_id in 1..par.num_threads {
                    let t_cfg = PireThreadConfig::new(
                        par.clone(), &pa_br_vec_ref, &pb_br_vec_ref, t_id, mc_eff, nc_eff, kc_eff
                    );
                    let ic_id = t_cfg.ic_id;
                    let jc_id = t_cfg.jc_id;
                    let ap_id = match gemm_mode {
                        GemmPool::Goto => ic_id,
                        GemmPool::SmallM => ic_id,
                        GemmPool::SmallN => t_id,
                        GemmPool::SmallMN => ic_id,
                    };
                    let bp_id = match gemm_mode {
                        GemmPool::Goto => jc_id,
                        GemmPool::SmallM => 0,
                        GemmPool::SmallN => jc_id,
                        GemmPool::SmallMN => 0,
                    };
                    let ap_cur = a.$pack_fn(ap_pool, ap_id);
                    let bp_cur = b.$pack_fn(bp_pool, bp_id);
                    let g = hw_config;
                    s.spawn(move || {
                            let alpha = &alpha as *const $t_as;
                            let beta = &beta as *const $t_bs;
                            gemm_fn(g, m, n, k, alpha, ap_cur, bp_cur, beta, c, &t_cfg);
                        }
                    );
                }
                {
                    let ap = a.$pack_fn(ap_pool, 0);
                    let bp = b.$pack_fn(bp_pool, 0);
                    let t_id: usize = 0;
                    let t_cfg = PireThreadConfig::new(par.clone(), &pa_br_vec_ref, &pb_br_vec_ref, t_id, mc_eff, nc_eff, kc_eff);
                    let alpha = &alpha as *const $t_as;
                    let beta = &beta as *const $t_bs;
                    gemm_fn(hw_config, m, n, k, alpha, ap, bp, beta, c, &t_cfg);
                }
            });
        }

        unsafe fn $goto_name<F:UnaryFnC>(
            hw_cfg: &$t_dispatcher <F>,
            m: usize, n: usize, k: usize,
            alpha: *const $t_as,
            a: $packa_ty,
            b: $packb_ty,
            beta: *const $t_bs,
            c: ArrayMut<$tc>,
            t_cfg: &PireThreadConfig
        ) {
            let ic_id = t_cfg.ic_id;
            let jc_id = t_cfg.jc_id;
            let ir_id = t_cfg.ir_id;
            let jr_id = t_cfg.jr_id;
            let ir_par = t_cfg.par.ir_par;
            let jr_par = t_cfg.par.jr_par;
            let ic_par = t_cfg.par.ic_par;
            let jc_par = t_cfg.par.jc_par;
            let mc = t_cfg.mc_eff;
            let nc = t_cfg.nc_eff;
            let kc = t_cfg.kc_eff;
            let mr = hw_cfg.mr;
            let nr = hw_cfg.nr;
            let (mc_start, mc_end, mc_left) = split_c_range(m, mc, mr, ic_id, ic_par);
            let (nc_start, nc_end, nc_left) = split_c_range(n, nc, nr, jc_id, jc_par);
            let (kc_start, d1_end) = (0, k);
            let one = $one;
            let c_rs = c.rs();
            let c_cs = c.cs();
            let c_ptr = c.src();
            let mut mc_i = mc_start;
            while mc_i < mc_end {
                let mc_len = mc.min(mc_end - mc_i);
                let mut kc_i = kc_start;
                let (mr_start, mr_end) = split_range(mc_len, mr, ir_id, ir_par);
                let mr_len = mr_end - mr_start;
                let c_i = c_ptr.add((mc_i+mr_start) * c_rs);
                while kc_i < d1_end {
                    let kc_len = kc.min(d1_end - kc_i);
                    let kc_len_eff = hw_cfg.round_k(kc_len);
                    let mut nc_i = nc_start;
                    let kc_last = kc_i + kc_len == d1_end;
                    let beta_t = if kc_i == kc_start { beta } else { &one as *const $t_bs};
                    let ap_data = $packa_name(hw_cfg, &a, mc_i, kc_i, mc_len, kc_len, t_cfg);
                    let ap = ap_data.src();
                    let ap = ap.add(mr_start*kc_len_eff);
                    while nc_i < nc_end {
                        let nc_len = nc.min(nc_end - nc_i);
                        let (nr_start, nr_end) = split_range(nc_len, nr, jr_id, jr_par);
                        let nr_len = nr_end - nr_start;
                        let c_ij = c_i.add((nc_i+nr_start) * c_cs);
                        let bp_data = $packb_name(hw_cfg, &b, nc_i, kc_i, nc_len, kc_len, t_cfg);
                        let bp = bp_data.src();
                        let bp = bp.add(nr_start*kc_len_eff);
                        $goto_kernel(
                            hw_cfg, mr_len, nr_len, kc_len, alpha, beta_t, c_ij, c_rs, c_cs,
                            ap, bp,
                            kc_last,
                        );

                        nc_i += nc;
                    }
                    if nc_left {
                        t_cfg.wait_packb();
                        t_cfg.wait_packb();
                    }
                    kc_i += kc;
                }
                mc_i += mc;
            }
            if mc_left {
                let mut kc_i = kc_start;
                while kc_i < d1_end {
                    let kc_len = kc.min(d1_end -kc_i);
                    t_cfg.wait_packa();
                    t_cfg.wait_packa();
                    let mut nc_i = nc_start;
                    while nc_i < nc_end {
                        let nc_len = nc.min(nc_end - nc_i);
                        let _ = $packb_name(hw_cfg, &b, nc_i, kc_i, nc_len, kc_len, t_cfg);
                        nc_i += nc;
                    }
                    if nc_left{
                        t_cfg.wait_packb();
                        t_cfg.wait_packb();
                    }
                    kc_i += kc;
                }
            }
        }
        unsafe fn $small_m_name<F:UnaryFnC>(
            hw_cfg: &$t_dispatcher <F>,
            m: usize, n: usize, k: usize,
            alpha: *const $t_as,
            a: $packa_ty,
            b: $packb_ty,
            beta: *const $t_bs,
            c: ArrayMut<$tc>,
            t_cfg: &PireThreadConfig
        ) {
            let par = &t_cfg.par;
            let ic_id = t_cfg.ic_id;
            let jc_id = t_cfg.jc_id;
            let ir_id = t_cfg.ir_id;
            let ir_par = par.ir_par;
            let jr_id = t_cfg.jr_id;
            let jr_par = par.jr_par;
            let mc = t_cfg.mc_eff;
            let nc = t_cfg.nc_eff;
            let kc = t_cfg.kc_eff;
            let mr = hw_cfg.mr;
            let nr = hw_cfg.nr;
            let (mc_start, mc_end, mc_left) = split_c_range(m, mc, mr, ic_id, par.ic_par);
            let (nc_start, nc_end, _) = split_c_range(n, nc, nr, jc_id, par.jc_par);
            let (kc_start, kc_end) = (0, k);
            let one = $one;

            let b_ptr = b.src();
            let b_rs = b.rs();
            let b_cs = b.cs();
            let c_rs = c.rs();
            let c_cs = c.cs();
            let c_ptr = c.src();
            let mut mc_i = mc_start;
            while mc_i < mc_end {
                let mc_len = mc.min(mc_end - mc_i);
                let (mr_start, mr_end) = split_range(mc_len, mr, ir_id, ir_par);
                let mr_len = mr_end - mr_start;
                let c_i = c_ptr.add((mc_i+mr_start) * c_rs);
                let mut kc_i = kc_start;
                while kc_i < kc_end {
                    let kc_len = kc.min(kc_end - kc_i);
                    let kc_len_eff = hw_cfg.round_k(kc_len);
                    let beta_t = if kc_i == kc_start { beta } else { &one as *const $t_bs};
                    let kc_last = kc_i + kc_len == kc_end;
                    let mut nc_i = nc_start;
                    let ap_data = $packa_name(hw_cfg, &a, mc_i, kc_i, mc_len, kc_len, t_cfg);
                    let ap = ap_data.src();
                    let ap = ap.add(mr_start*kc_len_eff);
                    let b_j = b_ptr.add(kc_i * b_rs);
                    while nc_i < nc_end {
                        let nc_len = nc.min(nc_end - nc_i);
                        let (nr_start, nr_end) = split_range(nc_len, nr, jr_id, jr_par);
                        let nr_len = nr_end - nr_start;
                        let c_ij = c_i.add((nc_i + nr_start) * c_cs);
                        let b_cur = b_j.add((nc_i + nr_start) * b_cs);
                        $small_m_kernel(
                            hw_cfg, mr_len, nr_len, kc_len, alpha, beta_t,
                            b_cur, b_rs, b_cs,
                            c_ij, c_rs, c_cs,
                            ap,
                            kc_last,
                        );
                        nc_i += nc;
                    }
                    kc_i += kc;
                }
                mc_i += mc;
            }

            if mc_left {
                let mut kc_i = kc_start;
                while kc_i < kc_end {
                    t_cfg.wait_packa();
                    t_cfg.wait_packa();
                    kc_i += kc;
                }
            }
        }
        unsafe fn $small_n_name<F:UnaryFnC>(
            hw_cfg: &$t_dispatcher <F>,
            m: usize, n: usize, k: usize,
            alpha: *const $t_as,
            a: $packa_ty,
            b: $packb_ty,
            beta: *const $t_bs,
            c: ArrayMut<$tc>,
            t_cfg: &PireThreadConfig
        ) {
            let par = &t_cfg.par;
            let ic_id = t_cfg.ic_id;
            let jc_id = t_cfg.jc_id;
            let ir_id = t_cfg.ir_id;
            let ir_par = par.ir_par;
            let jr_id = t_cfg.jr_id;
            let jr_par = par.jr_par;
            let mc = t_cfg.mc_eff;
            let nc = t_cfg.nc_eff;
            let kc = t_cfg.kc_eff;
            let mr = hw_cfg.mr;
            let nr = hw_cfg.nr;
            let (mc_start, mc_end, mc_left) = split_c_range(m, mc, mr, ic_id, par.ic_par);
            let (nc_start, nc_end, nc_left) = split_c_range(n, nc, nr, jc_id, par.jc_par);
            let (kc_start, kc_end) = (0, k);
            let one = $one;

            let c_rs = c.rs();
            let c_cs = c.cs();
            let c_ptr = c.src();
            let a_ptr = a.src();
            let a_rs = a.rs();
            let a_cs = a.cs();
            // make sure this ap is hwole slice
            let a_dst = a.dst_write(0, kc);
            let a_dst_ref = a_dst.get();
            let a_dst_ptr = a_dst_ref.as_mut_ptr();
            let mut mc_i = mc_start;
            while mc_i < mc_end {
                let mc_len = mc.min(mc_end - mc_i);
                let (mr_start, mr_end) = split_range(mc_len, mr, ir_id, ir_par);
                let mr_len = mr_end - mr_start;
                let c_i = c_ptr.add((mc_i+mr_start) * c_rs);
                let a_i = a_ptr.add((mc_i+mr_start) * a_rs);
                let mut kc_i = kc_start;
                while kc_i < kc_end {
                    let kc_len = kc.min(kc_end - kc_i);
                    let kc_last = kc_i + kc_len == kc_end;
                    let beta_t = if kc_i == kc_start { beta } else { &one as *const $t_bs};
                    let a_cur = a_i.add(kc_i*a_cs);
                    let mut nc_i = nc_start;
                    while nc_i < nc_end {
                        let nc_len = nc.min(nc_end - nc_i);
                        let (nr_start, nr_end) = split_range(nc_len, nr, jr_id, jr_par);
                        let nr_len = nr_end - nr_start;
                        let bp_data = $packb_name(hw_cfg, &b, nc_i, kc_i, nc_len, kc_len, t_cfg);
                        let bp = bp_data.src();
                        let c_ij = c_i.add((nc_i + nr_start) * c_cs);
                        $small_n_kernel(
                            hw_cfg, mr_len, nr_len, kc_len, alpha, beta_t,
                            a_cur, a_rs, a_cs,
                            a_dst_ptr, bp,
                            c_ij, c_rs, c_cs,
                            kc_last,
                        );
                        nc_i += nc;
                    }
                    if nc_left {
                        t_cfg.wait_packb();
                        t_cfg.wait_packb();
                    }
                    kc_i += kc;
                }
                mc_i += mc;
            }
            if mc_left {
                let mut kc_i = kc_start;
                while kc_i < kc_end {
                    let kc_len = kc.min(kc_end - kc_i);
                    let mut nc_i = nc_start;
                    while nc_i < nc_end {
                        let nc_len = nc.min(nc_end - nc_i);
                        let _ = $packb_name(hw_cfg, &b, nc_i, kc_i, nc_len, kc_len, t_cfg);
                        nc_i += nc;
                    }
                    if nc_left{
                        t_cfg.wait_packb();
                        t_cfg.wait_packb();
                    }
                    kc_i += kc;
                }
            }
        }
        unsafe fn $small_mn_name<F:UnaryFnC>(
            hw_cfg: &$t_dispatcher <F>,
            m: usize, n: usize, k: usize,
            alpha: *const $t_as,
            a: $packa_ty,
            b: $packb_ty,
            beta: *const $t_bs,
            c: ArrayMut<$tc>,
            t_cfg: &PireThreadConfig
        ) {
            let par = &t_cfg.par;
            let ic_id = t_cfg.ic_id;
            let jc_id = t_cfg.jc_id;
            let ir_id = t_cfg.ir_id;
            let ir_par = par.ir_par;
            let jr_id = t_cfg.jr_id;
            let jr_par = par.jr_par;
            let mc = t_cfg.mc_eff;
            let nc = t_cfg.nc_eff;
            let kc = t_cfg.kc_eff;
            let mr = hw_cfg.mr;
            let nr = hw_cfg.nr;
            let (mc_start, mc_end, mc_left) = split_c_range(m, mc, mr, ic_id, par.ic_par);
            let (nc_start, nc_end, _) = split_c_range(n, nc, nr, jc_id, par.jc_par);
            let (kc_start, kc_end) = (0, k);
            let one = $one;

            let a_ptr = a.src();
            let b_ptr = b.src();
            let a_rs = a.rs();
            let a_cs = a.cs();
            let b_rs = b.rs();
            let b_cs = b.cs();
            let c_rs = c.rs();
            let c_cs = c.cs();
            let c_ptr = c.src();
            let mut mc_i = mc_start;
            while mc_i < mc_end {
                let mc_len = mc.min(mc_end - mc_i);
                let (mr_start, mr_end) = split_range(mc_len, mr, ir_id, ir_par);
                let mr_len = mr_end - mr_start;
                let c_i = c_ptr.add((mc_i+mr_start) * c_rs);
                let a_i = a_ptr.add((mc_i+mr_start) * a.rs());
                let mut kc_i = kc_start;
                while kc_i < kc_end {
                    let kc_len = kc.min(kc_end - kc_i);
                    let kc_len_eff = hw_cfg.round_k(kc_len);
                    let beta_t = if kc_i == kc_start { beta } else { &one as *const $t_bs};
                    let kc_last = kc_i + kc_len == kc_end;
                    let mut nc_i = nc_start;
                    let a_cur = a_i.add(kc_i*a.cs());
                    let b_j = b_ptr.add(kc_i * b_rs);
                    while nc_i < nc_end {
                        let nc_len = nc.min(nc_end - nc_i);
                        let (nr_start, nr_end) = split_range(nc_len, nr, jr_id, jr_par);
                        let nr_len = nr_end - nr_start;
                        let c_ij = c_i.add((nc_i + nr_start) * c_cs);
                        let b_cur = b_j.add((nc_i + nr_start) * b_cs);
                        $small_mn_kernel(
                            hw_cfg, mr_len, nr_len, kc_len, alpha, beta_t,
                            a_cur, a_rs, a_cs,
                            b_cur, b_rs, b_cs,
                            c_ij, c_rs, c_cs,
                            kc_last,
                        );
                        nc_i += nc;
                    }
                    kc_i += kc;
                }
                mc_i += mc;
            }
        }
        // for packed api mc_i(nc_i) should be multiple of mr (nr, which we ensure by the split_c_range
        // for packed api kc_i should be multiple of kc_eff, which is always true since we dont parallelize over kc
        // this is subject to change if we parallelize over kc, but this is not in the plan
        // sync right before write and right before read
        // NOTE: dont return before the second packa as it ensures sync between threads
        pub(crate) unsafe fn $packa_name<'a,'b,F:UnaryFnC>(hw_cfg: &$t_dispatcher <F>, x: &'b $packa_ty<'a>, mc_i: usize, kc_i: usize, mc_len: usize, kc_len: usize, t_cfg: &PireThreadConfig) -> PtrData<'a,$tap> {
            t_cfg.wait_packa();
            let xp_ptr = match x {
                $packa_ty::StridedMatrix(x_i) => {
                    let mc_par = x_i.get_mc();
                    let mc_offset = mc_par * t_cfg.i_load_p_idx;
                    if mc_len > mc_offset {
                        let kc_len_ro = hw_cfg.round_k(kc_len);
                        let mc_len_x = (mc_len - mc_offset).min(mc_par);
                        let mc_i = mc_i + mc_offset;
                        let (rs, cs) = (x_i.rs(), x_i.cs());
                        let src_ptr = x_i.src().add(mc_i*rs + kc_i*cs);
                        let dst = x_i.dst_write(t_cfg.i_load_p_idx, kc_len_ro);
                        let dst_ref = dst.get();
                        let dst_ptr = dst_ref.as_mut_ptr();
                        $packa_name0(src_ptr, dst_ptr, mc_len_x, kc_len, rs, cs);
                    }
                    t_cfg.wait_packa();
                    PtrData::RefData(x_i.dst_read())
                }
                $packa_ty::PackedMatrix(x_i) => {
                    let m_ro = hw_cfg.round_m(x_i.m());
                    let kc_len_ro = hw_cfg.round_k(kc_len);
                    let res = is_mixed!(
                        $include_flag,
                        {
                            let mc_par = x_i.get_mc();
                            let mc_offset = mc_par * t_cfg.i_load_p_idx;
                            if mc_len > mc_offset {
                                let mc_len_x = (mc_len - mc_offset).min(mc_par);
                                let mc_i = mc_i + mc_offset;
                                let src_ptr = x_i.src().add(mc_i*kc_len_ro + kc_i*m_ro);
                                let dst = x_i.dst_write(t_cfg.i_load_p_idx, kc_len_ro);
                                let dst_ref = dst.get();
                                let dst_ptr = dst_ref.as_mut_ptr();
                                let mc_len_x_ro = hw_cfg.round_m(mc_len_x);
                                hw_cfg.cvt_mixed(src_ptr, dst_ptr, mc_len_x_ro*kc_len_ro);
                            }
                            t_cfg.wait_packa();
                            PtrData::RefData(x_i.dst_read())
                        },
                        {
                            let src_ptr = x_i.src().add(mc_i*kc_len_ro + kc_i*m_ro);
                            t_cfg.wait_packa();
                            PtrData::PtrData(src_ptr)
                        }

                    );
                    res

                }
            };
            xp_ptr
        }
        // NOTE: dont return before the second packa as it ensures sync between threads
        pub(crate) unsafe fn $packb_name<'a,'b,F:UnaryFnC>(hw_cfg: & $t_dispatcher <F>, x: &'b$packb_ty<'a>, nc_i: usize, kc_i: usize, nc_len: usize, kc_len: usize, t_cfg: &PireThreadConfig) -> PtrData<'a,$tbp> {
            t_cfg.wait_packb();
            let xp_ptr = match x {
                $packb_ty::StridedMatrix(x_i) => {
                    let nc_par = x_i.get_mc();
                    let nc_offset = nc_par * t_cfg.j_load_p_idx;
                    if nc_len > nc_offset {
                        let kc_len_ro = hw_cfg.round_k(kc_len);
                        let nc_len_x = (nc_len - nc_offset).min(nc_par);
                        let nc_i = nc_i + nc_offset;
                        let rs = x_i.rs();
                        let cs = x_i.cs();
                        let src_ptr = x_i.src().add(kc_i*rs + nc_i*cs);
                        let dst = x_i.dst_write(t_cfg.j_load_p_idx, kc_len_ro);
                        let dst_ref = dst.get();
                        let dst_ptr = dst_ref.as_mut_ptr();
                        $packb_name0(src_ptr, dst_ptr, nc_len_x, kc_len, rs, cs);
                    }
                    t_cfg.wait_packb();
                    PtrData::RefData(x_i.dst_read())
                }
                $packb_ty::PackedMatrix(x_i) => {
                    let kc_len_ro = hw_cfg.round_k(kc_len);
                    let n_ro = x_i.m();
                    let res = is_mixed!(
                        $include_flag,
                        {
                            let nc_par = x_i.get_mc();
                            let nc_offset = nc_par * t_cfg.j_load_p_idx;
                            if nc_len > nc_offset {
                                let nc_len_x = (nc_len - nc_offset).min(nc_par);
                                let nc_i = nc_i + nc_offset;
                                let src_ptr = x_i.src().add(nc_i*kc_len_ro + kc_i*n_ro);
                                let dst = x_i.dst_write(t_cfg.j_load_p_idx, kc_len_ro);
                                let dst_ref = dst.get();
                                let dst_ptr = dst_ref.as_mut_ptr();
                                hw_cfg.cvt_mixed(src_ptr, dst_ptr, nc_len_x*kc_len_ro);
                            }
                            t_cfg.wait_packb();
                            PtrData::RefData(x_i.dst_read())
                        },
                        {
                            let src_ptr = x_i.src().add(nc_i*kc_len_ro + kc_i*n_ro);
                            t_cfg.wait_packb();
                            PtrData::PtrData(src_ptr)
                        }

                    );
                    res
                }
            };
            xp_ptr
        }
    }
}

#[macro_export]
macro_rules! put_statement {
    (T,$st:stmt) => {
        $st
    };
    (F,$st:stmt) => {};
}

#[macro_export]
macro_rules! def_kernel_bb_v0 {
    (
        $t_ap:ty, $t_bp:ty, $t_c:ty, $t_s:ty,
        $no_partial:tt, $l2_prefetch:tt,
        $RS:tt,
        $MR:tt, $NR:tt, $pf1_0:tt, $pf_step:tt
    ) => {

        pub unsafe fn kernel_bb<F: UnaryFnC, const STRIDED: bool>(
            m: usize, n: usize, k: usize,
            alpha: *const $t_s,
            beta: *const $t_s,
            c: *mut $t_c, c_rs: usize, c_cs: usize,
            ap: *const $t_ap, bp: *const $t_bp,
            f: F,
        ) {
            let vs = simd_vector_length();
            let mr = $MR * vs;
            const NR: usize = $NR;
            let m_rounded = m / mr * mr;
            let m_left = m % mr;

            let c_cs_f = if STRIDED { mr } else { c_cs };
            let mut c_buf = [ZERO; $MR*VS_MAX*NR];
            let mut d_arr = [0, 0];

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let ap_cur = ap.add(m_i * k);
                pire_base::put_statement!($l2_prefetch, d_arr[0] = $pf1_0 * k);
                let mut n_i = 0;
                while n_i < n {
                    let bp_cur = bp.add(n_i * k);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    let nr = NR.min(n - n_i);
                    let c_cur1_f = if STRIDED {
                        pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, mr, nr, mr);
                        c_buf.as_mut_ptr()
                    } else {
                        c_cur1
                    };
                    ukernel_bbc(ap_cur, bp_cur, c_cur1_f, alpha, beta, k, d_arr, 1, c_cs_f, mr, nr);
                    for j in 0..nr {
                        f.call(c_cur1_f.add(j*c_cs_f), mr);
                    }
                    if STRIDED {
                        pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, mr, nr, mr);
                    }
                    n_i += NR;
                    pire_base::put_statement!($l2_prefetch, d_arr[0] += $pf_step * k);
                }
                m_i += mr;
            }


            seq_macro::seq!(mr_vs in 1..=$MR {
                if (m_left+vs-1) / vs == mr_vs {
                    let mr_left = mr_vs * vs;
                    let c_cs_f = if STRIDED || $no_partial { mr_left } else { c_cs };
                    let c_cur0 = c.add(m_i * c_rs);
                    let ap_cur = ap.add(m_i * k);
                    let mut n_i = 0;
                    while n_i < n {
                        let bp_cur = bp.add(n_i * k);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        let nr = NR.min(n - n_i);
                        let c_cur1_f = if STRIDED || $no_partial {
                            pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, m_left, nr, mr_left);
                            c_buf.as_mut_ptr()
                        } else {
                            c_cur1
                        };
                        paste::paste! {
                            [<ukernel_ mr_vs _bbp>](ap_cur, bp_cur, c_cur1_f, alpha, beta, k, d_arr, 1, c_cs_f, m_left, nr);
                        }
                        for j in 0..nr {
                            f.call(c_cur1_f.add(j*c_cs_f), m_left);
                        }
                        if STRIDED || $no_partial {
                            pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, m_left, nr, mr_left);
                        }
                        n_i += NR;
                    }
                }
            });
        }

        pub(crate) unsafe fn kernel<F: UnaryFnC>(
            m: usize,
            n: usize,
            k: usize,
            alpha: *const $t_s,
            beta: *const $t_s,
            c: *mut $t_c,
            c_rs: usize,
            c_cs: usize,
            ap: *const $t_ap,
            bp: *const $t_bp,
            f: F,
        ) {
            let k = (k + $RS - 1) / $RS * $RS;
            if c_rs == 1 {
                kernel_bb::<_, false>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
            } else {
                kernel_bb::<_, true>(m, n, k, alpha, beta, c, c_rs, c_cs, ap, bp, f)
            }
        }

    };
}

#[macro_export]
macro_rules! def_kernel_sb_v0 {
    (
        $t_a:ty, $t_ap:ty, $t_bp:ty, $t_c:ty, $t_s:ty,
        $no_partial:tt, $l2_prefetch:tt,
        $pack_fn:tt,
        $RS:tt,
        $MR:tt, $NR:tt, $pf1_0:tt, $pf_step:tt
    ) => {
        pub unsafe fn kernel_sb_v0<F: UnaryFnC, const STRIDED: bool>(
            m: usize, n: usize, k: usize,
            alpha: *const $t_s, beta: *const $t_s,
            a: *const $t_a, a_rs: usize, a_cs: usize,
            bp: *const $t_bp,
            c: *mut $t_c, c_rs: usize, c_cs: usize,
            ap: *mut $t_ap,
            f: F,
        ) {
            let k_eff = (k+$RS-1) / $RS * $RS;
            let vs = simd_vector_length();
            let mr = $MR * vs;
            const NR: usize = $NR;
            let m_rounded = m / mr * mr;
            let m_left = m % mr;
            let mut c_buf = [ZERO; $MR*VS_MAX*NR];
            let mut d_arr = [0, 0];
            let c_cs_f = if STRIDED { mr } else { c_cs };

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let a_cur = a.add(m_i * a_rs);
                pire_base::put_statement!($l2_prefetch, d_arr[0] = $pf1_0 * k);
                $pack_fn(mr, k, a_cur, a_rs, a_cs, ap, vs);
                let mut n_i = 0;
                while n_i < n {
                    let bp_cur = bp.add(n_i * k_eff);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    let nr = NR.min(n - n_i);
                    let c_cur1_f = if STRIDED {
                        pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, mr, nr, mr);
                        c_buf.as_mut_ptr()
                    } else {
                        c_cur1
                    };
                    ukernel_bbc(ap, bp_cur, c_cur1_f, alpha, beta, k_eff, d_arr, 1, c_cs_f, mr, nr);
                    for j in 0..nr {
                        f.call(c_cur1_f.add(j*c_cs_f), mr);
                    }
                    if STRIDED {
                        pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, mr, nr, mr);
                    }
                    n_i += NR;
                    pire_base::put_statement!($l2_prefetch, d_arr[0] += $pf_step * k);
                }
                m_i += mr;
            }

            seq_macro::seq!(mr_vs in 1..=$MR {
                if (m_left+vs-1) / vs == mr_vs {
                    let mr_left = mr_vs * vs;
                    let c_cs_f = if STRIDED || $no_partial { mr_left } else { c_cs };
                    let c_cur0 = c.add(m_i * c_rs);
                    let a_cur = a.add(m_i * a_rs);
                    $pack_fn(m_left, k, a_cur, a_rs, a_cs, ap, vs);
                    let mut n_i = 0;
                    while n_i < n {
                        let bp_cur = bp.add(n_i * k_eff);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        let nr = NR.min(n - n_i);
                        let c_cur1_f = if STRIDED {
                            pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, m_left, nr, mr_left);
                            c_buf.as_mut_ptr()
                        } else {
                            c_cur1
                        };
                        paste::paste! {
                            [<ukernel_ mr_vs _bbp>](ap, bp_cur, c_cur1_f, alpha, beta, k_eff, d_arr, 1, c_cs_f, m_left, nr);
                        }
                        for j in 0..nr {
                            f.call(c_cur1_f.add(j*c_cs_f), m_left);
                        }
                        if STRIDED || $no_partial {
                            pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, m_left, nr, mr_left);
                        }
                        n_i += NR;
                    }
                    return;
                }
            });
        }

        pub(crate) unsafe fn kernel_sb<F: UnaryFnC>(
            m: usize,
            n: usize,
            k: usize,
            alpha: *const $t_s,
            beta: *const $t_s,
            a: *const $t_a,
            a_rs: usize,
            a_cs: usize,
            b: *const $t_bp,
            c: *mut $t_c,
            c_rs: usize,
            c_cs: usize,
            ap_buf: *mut $t_ap,
            f: F,
        ) {
            if c_rs == 1 {
                kernel_sb_v0::<_, false>(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, f);
            } else {
                kernel_sb_v0::<_, true>(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, f);
            }
        }
    };
}

#[macro_export]
macro_rules! def_kernel_sb_v1 {
    (
        $t_a:ty, $t_ap:ty, $t_bp:ty, $t_c:ty, $t_s:ty,
        $no_partial:tt, $l2_prefetch:tt,
        $pack_fn:tt,
        $RS:tt,
        $MR:tt, $NR:tt, $pf1_0:tt, $pf_step:tt
    ) => {
        pub unsafe fn kernel_sb_v0<F: UnaryFnC, const STRIDED: bool>(
            m: usize, n: usize, k: usize,
            alpha: *const $t_s, beta: *const $t_s,
            a: *const $t_a, a_rs: usize, a_cs: usize,
            bp: *const $t_bp,
            c: *mut $t_c, c_rs: usize, c_cs: usize,
            ap: *mut $t_ap,
            f: F,
        ) {
            let k_eff = (k+$RS-1) / $RS * $RS;
            let vs = simd_vector_length();
            let mr = $MR * vs;
            const NR: usize = $NR;
            let m_rounded = m / mr * mr;
            let m_left = m % mr;
            let mut c_buf = [ZERO; $MR*VS_MAX*NR];
            let mut d_arr = [0, 0];
            let c_cs_f = if STRIDED { mr } else { c_cs };

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let a_cur = a.add(m_i * a_rs);
                pire_base::put_statement!($l2_prefetch, d_arr[0] = $pf1_0 * k);
                $pack_fn(mr, k, a_cur, a_rs, a_cs, ap, vs);
                let mut n_i = 0;
                while n_i < n {
                    let bp_cur = bp.add(n_i * k_eff);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    let nr = NR.min(n - n_i);
                    let c_cur1_f = if STRIDED {
                        pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, mr, nr, mr);
                        c_buf.as_mut_ptr()
                    } else {
                        c_cur1
                    };
                    ukernel_bbc(ap, bp_cur, c_cur1_f, alpha, beta, k_eff, d_arr, 1, c_cs_f, mr, nr);
                    for j in 0..nr {
                        f.call(c_cur1_f.add(j*c_cs_f), mr);
                    }
                    if STRIDED {
                        pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, mr, nr, mr);
                    }
                    n_i += NR;
                    pire_base::put_statement!($l2_prefetch, d_arr[0] += $pf_step * k);
                }
                m_i += mr;
            }

            seq_macro::seq!(mr_vs in 1..=$MR {
                if (m_left+vs-1) / vs == mr_vs {
                    let mr_left = mr_vs * vs;
                    let c_cs_f = if STRIDED || $no_partial { mr_left } else { c_cs };
                    let c_cur0 = c.add(m_i * c_rs);
                    let a_cur = a.add(m_i * a_rs);
                    $pack_fn(m_left, k, a_cur, a_rs, a_cs, ap, vs);
                    let mut n_i = 0;
                    while n_i < n {
                        let bp_cur = bp.add(n_i * k_eff);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        let nr = NR.min(n - n_i);
                        let c_cur1_f = if STRIDED {
                            pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, m_left, nr, mr_left);
                            c_buf.as_mut_ptr()
                        } else {
                            c_cur1
                        };
                        paste::paste! {
                            [<ukernel_ mr_vs _bbp>](ap, bp_cur, c_cur1_f, alpha, beta, k_eff, d_arr, 1, c_cs_f, m_left, nr);
                        }
                        for j in 0..nr {
                            f.call(c_cur1_f.add(j*c_cs_f), m_left);
                        }
                        if STRIDED || $no_partial {
                            pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, m_left, nr, mr_left);
                        }
                        n_i += NR;
                    }
                    return;
                }
            });
        }


        pub unsafe fn kernel_rb_v0<F: UnaryFnC, const STRIDED: bool>(
            m: usize, n: usize, k: usize,
            alpha: *const $t_s, beta: *const $t_s,
            a: *const $t_a, a_rs: usize, a_cs: usize,
            bp: *const $t_bp,
            c: *mut $t_c, c_rs: usize, c_cs: usize,
            ap: *mut $t_ap,
            f: F,
        ) {
            let k_eff = (k+$RS-1) / $RS * $RS;
            let vs = simd_vector_length();
            let mr = $MR * vs;
            const NR: usize = $NR;
            let m_rounded = m / mr * mr;
            let m_left = m % mr;
            let mut c_buf = [ZERO; $MR*VS_MAX*NR];
            let mut d_arr = [0, 0];
            let c_cs_f = if STRIDED { mr } else { c_cs };

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let a_cur = a.add(m_i * a_rs);
                pire_base::put_statement!($l2_prefetch, d_arr[0] = $pf1_0 * k);
                $pack_fn(mr, k, a_cur, a_rs, a_cs, ap, vs);
                let mut n_i = 0;
                while n_i < n {
                    let bp_cur = bp.add(n_i * k_eff);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    let nr = NR.min(n - n_i);
                    let c_cur1_f = if STRIDED {
                        pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, mr, nr, mr);
                        c_buf.as_mut_ptr()
                    } else {
                        c_cur1
                    };
                    ukernel_bbc(ap, bp_cur, c_cur1_f, alpha, beta, k_eff, d_arr, 1, c_cs_f, mr, nr);
                    for j in 0..nr {
                        f.call(c_cur1_f.add(j*c_cs_f), mr);
                    }
                    if STRIDED {
                        pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, mr, nr, mr);
                    }
                    n_i += NR;
                    pire_base::put_statement!($l2_prefetch, d_arr[0] += $pf_step * k);
                }
                m_i += mr;
            }

            seq_macro::seq!(mr_vs in 1..=$MR {
                if (m_left+vs-1) / vs == mr_vs {
                    let mr_left = mr_vs * vs;
                    let c_cs_f = if STRIDED || $no_partial { mr_left } else { c_cs };
                    let c_cur0 = c.add(m_i * c_rs);
                    let a_cur = a.add(m_i * a_rs);
                    $pack_fn(m_left, k, a_cur, a_rs, a_cs, ap, vs);
                    let mut n_i = 0;
                    while n_i < n {
                        let bp_cur = bp.add(n_i * k_eff);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        let nr = NR.min(n - n_i);
                        let c_cur1_f = if STRIDED {
                            pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, m_left, nr, mr_left);
                            c_buf.as_mut_ptr()
                        } else {
                            c_cur1
                        };
                        paste::paste! {
                            [<ukernel_ mr_vs _bbp>](ap, bp_cur, c_cur1_f, alpha, beta, k_eff, d_arr, 1, c_cs_f, m_left, nr);
                        }
                        for j in 0..nr {
                            f.call(c_cur1_f.add(j*c_cs_f), m_left);
                        }
                        if STRIDED || $no_partial {
                            pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, m_left, nr, mr_left);
                        }
                        n_i += NR;
                    }
                    return;
                }
            });
        }

        pub unsafe fn kernel_cb_v0<F: UnaryFnC, const STRIDED: bool>(
            m: usize, n: usize, k: usize,
            alpha: *const $t_s, beta: *const $t_s,
            a: *const $t_a, a_rs: usize, a_cs: usize,
            bp: *const $t_bp,
            c: *mut $t_c, c_rs: usize, c_cs: usize,
            ap: *mut $t_ap,
            f: F,
        ) {
            let k_eff = (k+$RS-1) / $RS * $RS;
            let vs = simd_vector_length();
            let mr = $MR * vs;
            const NR: usize = $NR;
            let m_rounded = m / mr * mr;
            let m_left = m % mr;
            let mut c_buf = [ZERO; $MR*VS_MAX*NR];
            let mut d_arr = [0, 0];
            let c_cs_f = if STRIDED { mr } else { c_cs };

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let a_cur = a.add(m_i * a_rs);
                pire_base::put_statement!($l2_prefetch, d_arr[0] = $pf1_0 * k);
                $pack_fn(mr, k, a_cur, a_rs, a_cs, ap, vs);
                let mut n_i = 0;
                while n_i < n {
                    let bp_cur = bp.add(n_i * k_eff);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    let nr = NR.min(n - n_i);
                    let c_cur1_f = if STRIDED {
                        pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, mr, nr, mr);
                        c_buf.as_mut_ptr()
                    } else {
                        c_cur1
                    };
                    ukernel_bbc(ap, bp_cur, c_cur1_f, alpha, beta, k_eff, d_arr, 1, c_cs_f, mr, nr);
                    for j in 0..nr {
                        f.call(c_cur1_f.add(j*c_cs_f), mr);
                    }
                    if STRIDED {
                        pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, mr, nr, mr);
                    }
                    n_i += NR;
                    pire_base::put_statement!($l2_prefetch, d_arr[0] += $pf_step * k);
                }
                m_i += mr;
            }

            seq_macro::seq!(mr_vs in 1..=$MR {
                if (m_left+vs-1) / vs == mr_vs {
                    let mr_left = mr_vs * vs;
                    let c_cs_f = if STRIDED || $no_partial { mr_left } else { c_cs };
                    let c_cur0 = c.add(m_i * c_rs);
                    let a_cur = a.add(m_i * a_rs);
                    $pack_fn(m_left, k, a_cur, a_rs, a_cs, ap, vs);
                    let mut n_i = 0;
                    while n_i < n {
                        let bp_cur = bp.add(n_i * k_eff);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        let nr = NR.min(n - n_i);
                        let c_cur1_f = if STRIDED {
                            pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, m_left, nr, mr_left);
                            c_buf.as_mut_ptr()
                        } else {
                            c_cur1
                        };
                        paste::paste! {
                            [<ukernel_ mr_vs _bbp>](ap, bp_cur, c_cur1_f, alpha, beta, k_eff, d_arr, 1, c_cs_f, m_left, nr);
                        }
                        for j in 0..nr {
                            f.call(c_cur1_f.add(j*c_cs_f), m_left);
                        }
                        if STRIDED || $no_partial {
                            pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, m_left, nr, mr_left);
                        }
                        n_i += NR;
                    }
                    return;
                }
            });
        }

        pub(crate) unsafe fn kernel_sb<F: UnaryFnC>(
            m: usize,
            n: usize,
            k: usize,
            alpha: *const $t_s,
            beta: *const $t_s,
            a: *const $t_a,
            a_rs: usize,
            a_cs: usize,
            b: *const $t_bp,
            c: *mut $t_c,
            c_rs: usize,
            c_cs: usize,
            ap_buf: *mut $t_ap,
            f: F,
        ) {
            if c_rs == 1 && a_cs == 1 {
                kernel_rb_v0::<_, false>(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, f);
                return;
            }
            if c_rs == 1 {
                kernel_sb_v0::<_, false>(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, f);
            } else {
                kernel_sb_v0::<_, true>(m, n, k, alpha, beta, a, a_rs, a_cs, b, c, c_rs, c_cs, ap_buf, f);
            }
        }
    };
}

#[macro_export]
macro_rules! def_kernel_bs {
    (
        $t_ap:ty, $t_b:ty, $t_c:ty, $t_s:ty,
        $MR:tt, $NR:tt
    ) => {
        pub unsafe fn kernel_bs_v0<F: UnaryFnC, const STRIDED: bool>(
            m: usize, n: usize, k: usize,
            alpha: *const $t_s, beta: *const $t_s,
            b: *const $t_b, b_rs: usize, b_cs: usize,
            c: *mut $t_c, c_rs: usize, c_cs: usize,
            ap: *const $t_ap,
            f: F,
        ) {
            const MR: usize = $MR * VS;
            const NR: usize = $NR;
            let m_rounded = m / MR * MR;
            let m_left = m % MR;
            let mut c_buf = [ZERO;MR*NR];
            let mut d_arr = [b_rs, b_cs];
            let c_cs_f = if STRIDED { MR } else { c_cs };

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let ap_cur = ap.add(m_i * k);
                let mut n_i = 0;
                while n_i < n {
                    let b_cur = b.add(n_i * b_cs);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    let nr = NR.min(n - n_i);
                    let c_cur1_f = if STRIDED {
                        pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, MR, nr, MR);
                        c_buf.as_mut_ptr()
                    } else {
                        c_cur1
                    };
                    ukernel_bsc(ap_cur, b_cur, c_cur1_f, alpha, beta, k, d_arr, 1, c_cs_f, MR, nr);
                    for j in 0..nr {
                        f.call(c_cur1_f.add(j*c_cs_f), MR);
                    }
                    if STRIDED {
                        pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, MR, nr, MR);
                    }
                    n_i += NR;
                }
                m_i += MR;
            }
            seq_macro::seq!(mr_left in 1..=$MR {
                if (m_left+VS-1) / VS == mr_left {
                    const MR_LEFT: usize = mr_left * VS;
                    let c_cs_f = if STRIDED { MR_LEFT } else { c_cs };
                    let c_cur0 = c.add(m_i * c_rs);
                    let ap_cur = ap.add(m_i * k);
                    let mut n_i = 0;
                    while n_i < n {
                        let b_cur = b.add(n_i * b_cs);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        let nr = NR.min(n - n_i);
                        let c_cur1_f = if STRIDED {
                            pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, m_left, nr, MR_LEFT);
                            c_buf.as_mut_ptr()
                        } else {
                            c_cur1
                        };
                        paste::paste! {
                            [<ukernel_ mr_left _bsp>](ap_cur, b_cur, c_cur1_f, alpha, beta, k, d_arr, 1, c_cs_f, m_left, nr);
                        }
                        for j in 0..nr {
                            f.call(c_cur1_f.add(j*c_cs_f), m_left);
                        }
                        if STRIDED {
                            pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, m_left, nr, MR_LEFT);
                        }
                        n_i += NR;
                    }
                    return;
                }
            });
        }

        pub(crate) unsafe fn kernel_bs<F: UnaryFnC>(
            m: usize,
            n: usize,
            k: usize,
            alpha: *const $t_s,
            beta: *const $t_s,
            b: *const $t_b,
            b_rs: usize,
            b_cs: usize,
            c: *mut $t_c,
            c_rs: usize,
            c_cs: usize,
            ap: *const $t_ap,
            f: F,
        ) {
            if c_rs == 1 {
                kernel_bs_v0::<_, false>(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, f);
            } else {
                kernel_bs_v0::<_, true>(m, n, k, alpha, beta, b, b_rs, b_cs, c, c_rs, c_cs, ap, f);
            }
        }
    };
}

#[macro_export]
macro_rules! def_kernel_ss {
    (
        $t_a:ty, $t_b:ty, $t_c:ty, $t_s:ty,
        $MR:tt, $NR:tt
    ) => {
        pub unsafe fn kernel_ss_v0<F: UnaryFnC, const STRIDED: bool>(
            m: usize, n: usize, k: usize,
            alpha: *const $t_s, beta: *const $t_s,
            a: *const $t_a, a_rs: usize, a_cs: usize,
            b: *const $t_b, b_rs: usize, b_cs: usize,
            c: *mut $t_c, c_rs: usize, c_cs: usize,
            f: F,
        ) {
            const MR: usize = $MR * VS;
            const NR: usize = $NR;
            let m_rounded = m / MR * MR;
            let m_left = m % MR;
            let mut c_buf = [ZERO;MR*NR];
            let mut d_arr = [b_rs, b_cs];
            let c_cs_f = if STRIDED { MR } else { c_cs };

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let a_cur = a.add(m_i * a_rs);
                let mut n_i = 0;
                while n_i < n {
                    let b_cur = b.add(n_i * b_cs);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    let nr = NR.min(n - n_i);
                    let c_cur1_f = if STRIDED {
                        pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, MR, nr, MR);
                        c_buf.as_mut_ptr()
                    } else {
                        c_cur1
                    };
                    ukernel_ssc(a_cur, b_cur, c_cur1_f, alpha, beta, k, d_arr, a_cs, c_cs_f, MR, nr);
                    for j in 0..nr {
                        f.call(c_cur1_f.add(j*c_cs_f), MR);
                    }
                    if STRIDED {
                        pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, MR, nr, MR);
                    }
                    n_i += NR;
                }
                m_i += MR;
            }
            seq_macro::seq!(mr_left in 1..=$MR {
                if (m_left+VS-1) / VS == mr_left {
                    const MR_LEFT: usize = mr_left * VS;
                    let c_cs_f = if STRIDED { MR_LEFT } else { c_cs };
                    let c_cur0 = c.add(m_i * c_rs);
                    let a_cur = a.add(m_i * a_rs);
                    let mut n_i = 0;
                    while n_i < n {
                        let b_cur = b.add(n_i * b_cs);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        let nr = NR.min(n - n_i);
                        let c_cur1_f = if STRIDED {
                            pire_base::load_buf(c_cur1, c_rs, c_cs, &mut c_buf, m_left, nr, MR_LEFT);
                            c_buf.as_mut_ptr()
                        } else {
                            c_cur1
                        };
                        paste::paste! {
                            [<ukernel_ mr_left _ssp>](a_cur, b_cur, c_cur1_f, alpha, beta, k, d_arr, a_cs, c_cs_f, m_left, nr);
                        }
                        for j in 0..nr {
                            f.call(c_cur1_f.add(j*c_cs_f), m_left);
                        }
                        if STRIDED {
                            pire_base::store_buf(c_cur1, c_rs, c_cs, &c_buf, m_left, nr, MR_LEFT);
                        }
                        n_i += NR;
                    }
                    return;
                }
            });
        }

        pub(crate) unsafe fn kernel_ss<F: UnaryFnC>(
            m: usize,
            n: usize,
            k: usize,
            alpha: *const $t_s,
            beta: *const $t_s,
            a: *const $t_a,
            a_rs: usize,
            a_cs: usize,
            b: *const $t_b,
            b_rs: usize,
            b_cs: usize,
            c: *mut $t_c,
            c_rs: usize,
            c_cs: usize,
            f: F,
        ) {
            if c_rs == 1 {
                kernel_ss_v0::<_, false>(m, n, k, alpha, beta, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, c_cs, f);
            } else {
                kernel_ss_v0::<_, true>(m, n, k, alpha, beta, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, c_cs, f);
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_export]
macro_rules! mem {
    ($m0:tt, $b0:expr) => {
        concat!($b0, "+", $m0)
    };
}

#[cfg(target_arch = "aarch64")]
#[macro_export]
macro_rules! mem {
    ($m0:tt, $b0:tt, $b1:tt) => {
        concat!("[", $m0, ", #", $b0, ", ", $b1, "]")
    };
    ($m0:tt, $b0:tt) => {
        concat!("[", $m0, ", #", $b0, "]")
    };
    ($m0:tt) => {
        concat!("[", $m0, "]")
    };
}
#[macro_export]
macro_rules! inc_a {
    ($mr:tt) => {
        concat!("add $", vs!(), "*", $mr, ", {ax}", "\n",)
    };
    (P, $mr:tt) => {
        concat!("add {x5}, {ax}\n")
    };
    (C, $mr:tt) => {
        concat!("add {x5}, {ax}\n")
    };
}
#[macro_export]
macro_rules! inc_a_k_unroll {
    ($X:tt, $K:tt) => {
        concat!("add $", vs!(), "*", $K, "*", $X, ",{ax}", "\n",)
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_export]
macro_rules! load_a {
    ($mr:tt) => {
        concat!(pire_base::loadp!(B, $mr, "0({ax})"), pire_base::inc_a!($mr))
    };
    (B, $mr:tt) => {
        concat!(pire_base::loadp!(B, $mr, "0({ax})"), pire_base::inc_a!($mr))
    };
    (C, $mr:tt) => {
        concat!(pire_base::loadp!(C, $mr, "0({ax})"), pire_base::inc_a!(C, $mr))
    };
    (P, $mr:tt) => {
        concat!(pire_base::loadp!(C, $mr, "0({ax})"), pire_base::inc_a!(P, $mr))
    };
    ($mr:tt, $K:tt) => {
        concat!(pire_base::loadp!(B, $mr, concat!($mr, "*", vs!(), "*", $K, "({ax})")),)
    };
}

#[cfg(target_arch = "aarch64")]
#[macro_export]
macro_rules! load_a {
    ($mr:tt) => {
        concat!(pire_base::loadp!($mr, "{ax}"), inc_a!($mr))
    };
}
/*

x1 -> cs_a
x2 -> cs_b
x3 -> ax + 3*cs_a
x4 -> bx + 3*cs_b

*/

#[macro_export]
macro_rules! init_ab {
    (B) => {
        concat!(
            "/* {x5} */\n",
            "/* {x4} */\n",
            "/* {x3} */\n",
            "/* {x2} */\n",
            "/* {x1} */\n",
            "mov 24({dim_arrx}),{x0}\n",
        )
    };
    (S) => {
        concat!(
            // mov cs_b to reg
            "mov ({dim_arrx}), {x1}\n",
            "mov 8({dim_arrx}), {x2}\n",
            "lea ({x2}, {x2}, 2), {x5}\n",
            "lea ({bx}, {x5}, 1), {x3}\n",
            "lea ({x3}, {x5}, 1), {x4}\n",
            "lea ({x4}, {x5}, 1), {x5}\n",
            "mov 24({dim_arrx}),{x0}\n",
        )
    };
}

#[macro_export]
macro_rules! init_ab_2 {
    (B) => {
        concat!("mov 8({dim_arrx}),{x0}\n",)
    };
}

#[macro_export]
macro_rules! c_load {
    () => {
        concat!(
            "mov 16({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x3}\n",
            "lea ({cx}, {x3},), {x1}\n",
            "lea ({x1}, {x3},), {x2}\n",
            "lea ({x2}, {x3},), {x3}\n",
        )
    };
}

#[macro_export]
macro_rules! init_ab_avx {
    (B) => {
        concat!("/* {x3} */\n", "/* {x2} */\n", "/* {x1} */\n", "mov 24({dim_arrx}),{x0}\n",)
    };
    (S) => {
        concat!(
            // mov cs_b to reg
            "mov ({dim_arrx}), {x1}\n",
            "mov 8({dim_arrx}), {x2}\n",
            "lea ({x2}, {x2}, 2), {x3}\n",
            "lea ({bx}, {x3}, 1), {x3}\n",
            "mov 24({dim_arrx}),{x0}\n",
        )
    };
}

#[macro_export]
macro_rules! b_mem {
    (S,$nr:tt, 0, $K:tt) => {
        "({bx})"
    };
    (S,$nr:tt, 1, $K:tt) => {
        "({bx},{x2},1)"
    };
    (S,$nr:tt, 2, $K:tt) => {
        "({bx},{x2},2)"
    };
    (S,$nr:tt, 3, $K:tt) => {
        "({x3})"
    };
    (S,$nr:tt, 4, $K:tt) => {
        "({x3},{x2},1)"
    };
    (S,$nr:tt, 5, $K:tt) => {
        "({x3},{x2},2)"
    };
    (S,$nr:tt, 6, $K:tt) => {
        "({x4})"
    };
    (S,$nr:tt, 7, $K:tt) => {
        "({x4},{x2},1)"
    };
    (S,$nr:tt, 8, $K:tt) => {
        "({x4},{x2},2)"
    };
    (S,$nr:tt, 9, $K:tt) => {
        "({x5})"
    };
    (S,$nr:tt, 10, $K:tt) => {
        "({x5},{x2},1)"
    };
    (S,$nr:tt, 11, $K:tt) => {
        "({x5},{x2},2)"
    };

    (B, $nr:tt, $ni:tt, $K:tt) => {
        concat!($K, "*", $nr, "*", bs!(), "+", $ni, "*", bs!(), "({bx})")
    };
}

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! c_mem {
    (0) => {
        "0({cx})"
    };
    (1) => {
        "0({cx}, {x0})"
    };
    (2) => {
        "0({cx}, {x0}, 2)"
    };
    (3) => {
        "0({x1})"
    };
    (4) => {
        "0({x1}, {x0})"
    };
    (5) => {
        "0({x1}, {x0}, 2)"
    };
    (6) => {
        "0({x2})"
    };
    (7) => {
        "0({x2}, {x0})"
    };
    (8) => {
        "0({x2}, {x0}, 2)"
    };
    (9) => {
        "0({x3})"
    };
    (10) => {
        "0({x3}, {x0})"
    };
    (11) => {
        "0({x3}, {x0}, 2)"
    };
    (12) => {
        "0({x4})"
    };
    (13) => {
        "0({x4}, {x0})"
    };
    (14) => {
        "0({x4}, {x0}, 2)"
    };
}

#[cfg(target_arch = "x86")]
#[macro_export]
macro_rules! c_mem {
    (0) => {
        "0({cx})"
    };
    (1) => {
        "0({cx}, {x0})"
    };
    (2) => {
        "0({cx}, {x0}, 2)"
    };
    (3) => {
        "0({bx})"
    };
    (4) => {
        "0({x1}, {x0})"
    };
}

#[cfg(target_arch = "aarch64")]
#[macro_export]
macro_rules! c_mem {
    (0) => {
        "{cx}"
    };
    (1) => {
        "{x1}"
    };
    (2) => {
        "{x2}"
    };
    (3) => {
        "{x3}"
    };
    (4) => {
        "{x4}"
    };
    (5) => {
        "{x5}"
    };
    (6) => {
        "{x6}"
    };
    (7) => {
        "{x7}"
    };
    (8) => {
        "{x8}"
    };
    (9) => {
        "{x9}"
    };
    (10) => {
        "{x10}"
    };
    (11) => {
        "{x11}"
    };
    (12) => {
        "{x12}"
    };
    (13) => {
        "{x13}"
    };
    (14) => {
        "{x14}"
    };
}

#[macro_export]
macro_rules! acc_3 {
    ($layout:tt, $ni:tt, $q:tt) => {
        pire_base::acc_p!($layout, pire_base::c_mem!($ni), $q, cr!(0, $ni), cr!(1, $ni), cr!(2, $ni))
    };
}
#[macro_export]
macro_rules! store_3 {
    ($layout:tt, $ni:tt) => {
        pire_base::storep!($layout, pire_base::c_mem!($ni), cr!(0, $ni), cr!(1, $ni), cr!(2, $ni))
    };
}

#[macro_export]
macro_rules! acc_2 {
    ($layout:tt, $ni:tt, $q:tt) => {
        pire_base::acc_p!($layout, pire_base::c_mem!($ni), $q, cr!(0, $ni), cr!(1, $ni))
    };
}
#[macro_export]
macro_rules! store_2 {
    ($layout:tt, $ni:tt) => {
        pire_base::storep!($layout, pire_base::c_mem!($ni), cr!(0, $ni), cr!(1, $ni))
    };
}
#[macro_export]
macro_rules! acc_1 {
    ($layout:tt, $ni:tt, $q:tt) => {
        pire_base::acc_p!($layout, pire_base::c_mem!($ni), $q, cr!(0, $ni))
    };
}
#[macro_export]
macro_rules! store_1 {
    ($layout:tt, $ni:tt) => {
        pire_base::storep!($layout, pire_base::c_mem!($ni), cr!(0, $ni))
    };
}

#[macro_export]
macro_rules! step_3 {
    ($b_layout:tt, $nr:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, $nr, n, $K, br_3),
                    vfmadd!(0, n, br_3),
                    vfmadd!(1, n, br_3),
                    vfmadd!(2, n, br_3),
                )*
            )
        })
    };
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, n, br_3),
                    vfmadd!(0, n, br_3),
                    vfmadd!(1, n, br_3),
                    vfmadd!(2, n, br_3),
                )*
            )
        })
    };
}

#[macro_export]
macro_rules! step_2 {
    ($b_layout:tt, $nr:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, $nr, n, $K, br_2),
                    vfmadd!(0, n, br_2),
                    vfmadd!(1, n, br_2),
                )*
            )
        })
    };
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, n, br_2),
                    vfmadd!(0, n, br_2),
                    vfmadd!(1, n, br_2),
                )*
            )
        })
    };
}

#[macro_export]
macro_rules! step_1 {
    ($b_layout:tt, $nr:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, $nr, n, $K, br_1),
                    vfmadd!(0, n, br_1),
                )*
            )
        })
    };
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, n, br_1),
                    vfmadd!(0, n, br_1),
                )*
            )
        })
    };
}

#[macro_export]
macro_rules! step_3_c {
    ($b_layout:tt, $nr:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, $nr, n, $K, br_3, 0),
                    vfmadd!(0, n, br_3, 0),
                    vfmadd!(1, n, br_3, 0),
                    vfmadd!(2, n, br_3, 0),

                    load_b!($b_layout, $nr, n, $K, br_3, 1),
                    vfmadd!(0, n, br_3, 1),
                    vfmadd!(1, n, br_3, 1),
                    vfmadd!(2, n, br_3, 1),
                )*
            )
        })
    };
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, n, br_3, 0),
                    vfmadd!(0, n, br_3, 0),
                    vfmadd!(1, n, br_3, 0),
                    vfmadd!(2, n, br_3, 0),

                    load_b!($b_layout, n, br_3, 1),
                    vfmadd!(0, n, br_3, 1),
                    vfmadd!(1, n, br_3, 1),
                    vfmadd!(2, n, br_3, 1),
                )*
            )
        })
    };
}

#[macro_export]
macro_rules! step_2_c {
    ($b_layout:tt, $nr:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, $nr, n, $K, br_2, 0),
                    vfmadd!(0, n, br_2, 0),
                    vfmadd!(1, n, br_2, 0),

                    load_b!($b_layout, $nr, n, $K, br_2, 1),
                    vfmadd!(0, n, br_2, 1),
                    vfmadd!(1, n, br_2, 1),
                )*
            )
        })
    };
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, n, br_2, 0),
                    vfmadd!(0, n, br_2, 0),
                    vfmadd!(1, n, br_2, 0),

                    load_b!($b_layout, n, br_2, 1),
                    vfmadd!(0, n, br_2, 1),
                    vfmadd!(1, n, br_2, 1),
                )*
            )
        })
    };
}

#[macro_export]
macro_rules! step_1_c {
    ($b_layout:tt, $nr:tt, $K:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, $nr, n, $K, br_1, 0),
                    vfmadd!(0, n, br_1, 0),

                    load_b!($b_layout, $nr, n, $K, br_1, 1),
                    vfmadd!(0, n, br_1, 1),
                )*
            )
        })
    };
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, n, br_1, 0),
                    vfmadd!(0, n, br_1, 0),

                    load_b!($b_layout, n, br_1, 1),
                    vfmadd!(0, n, br_1, 1),
                )*
            )
        })
    };
}

#[macro_export]
macro_rules! loadp {
    (P, 3, $m0:expr) => {
        concat!(loadp_unit!($m0, 0, C), loadp_unit!($m0, 1, C), loadp_unit!($m0, 2, P),)
    };
    (P, 2, $m0:expr) => {
        concat!(loadp_unit!($m0, 0, C), loadp_unit!($m0, 1, P),)
    };
    (P, 1, $m0:expr) => {
        concat!(loadp_unit!($m0, 0, P),)
    };
    ($layout:tt, 3, $m0:expr) => {
        concat!(loadp_unit!($m0, 0, $layout), loadp_unit!($m0, 1, $layout), loadp_unit!($m0, 2, $layout),)
    };
    ($layout:tt, 2, $m0:expr) => {
        concat!(loadp_unit!($m0, 0, $layout), loadp_unit!($m0, 1, $layout),)
    };
    ($layout:tt, 1, $m0:expr) => {
        concat!(loadp_unit!($m0, 0, $layout),)
    };
}

#[macro_export]
macro_rules! storep {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            storep_unit!(C, $r1, v_i!($m0, 0)),
            storep_unit!(C, $r2, v_i!($m0, 1)),
            storep_unit!($layout, $r3, v_i!($m0, 2)),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(storep_unit!(C, $r1, v_i!($m0, 0)), storep_unit!($layout, $r2, v_i!($m0, 1)),)
    };
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(storep_unit!($layout, $r1, v_i!($m0, 0)),)
    };
}

#[macro_export]
macro_rules! acc_p {
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, v_i!($m0, 0), $r1, $q),
            beta_fmadd!(C, v_i!($m0, 1), $r2, $q),
            beta_fmadd!($layout, v_i!($m0, 2), $r3, $q),
        )
    };
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr) => {
        concat!(beta_fmadd!(C, v_i!($m0, 0), $r1, $q), beta_fmadd!($layout, v_i!($m0, 1), $r2, $q),)
    };
    ($layout:tt, $m0:expr, $q:tt, $r1:expr) => {
        concat!(beta_fmadd!($layout, v_i!($m0, 0), $r1, $q),)
    };
}

#[macro_export]
macro_rules! cum_seq {
    ($step_macro:tt, $layout:tt, $nr:tt, $b:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!($layout, n, $b),)*)
        })
    };
    ($step_macro:tt, $layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!($layout, n),)*)
        })
    };
}

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! prefetch_c_avx512 {
    (3, $nr:tt, $c:tt, $ldc:tt) => {
        use core::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
            _mm_prefetch(c_u8.add(64), 3);
            _mm_prefetch(c_u8.add(128), 3);
        });
    };
    (2, $nr:tt, $c:tt, $ldc:tt) => {
        use core::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
            _mm_prefetch(c_u8.add(64), 3);
        });
    };
    (1, $nr:tt, $c:tt, $ldc:tt) => {
        use core::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
        });
    };
}

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! prefetch_c_avx {
    (3, $nr:tt, $c:tt, $ldc:tt) => {
        use core::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
            _mm_prefetch(c_u8.add(64), 3);
            _mm_prefetch(c_u8.add(92), 3);
        });
    };
    (2, $nr:tt, $c:tt, $ldc:tt) => {
        use core::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
            _mm_prefetch(c_u8.add(60), 3);
        });
    };
    (1, $nr:tt, $c:tt, $ldc:tt) => {
        use core::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
        });
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_export]
macro_rules! prefetch_c_sse {
    (3, $nr:tt, $c:tt, $ldc:tt) => {
        #[cfg(target_arch="x86")]
        use core::arch::x86::_mm_prefetch;
        #[cfg(target_arch="x86_64")]
        use core::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
            _mm_prefetch(c_u8.add(64), 3);
        });
    };
    (2, $nr:tt, $c:tt, $ldc:tt) => {
        #[cfg(target_arch="x86")]
        use core::arch::x86::_mm_prefetch;
        #[cfg(target_arch="x86_64")]
        use core::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
        });
    };
    (1, $nr:tt, $c:tt, $ldc:tt) => {
        #[cfg(target_arch="x86")]
        use core::arch::x86::_mm_prefetch;
        #[cfg(target_arch="x86_64")]
        use core::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
        });
    };
}

#[macro_export]
macro_rules! prefetch_b {
    (S) => {
        ""
    };
    (B) => {
        concat!("prefetcht0 192({bx}) \n",)
    };
}

#[cfg(target_arch = "x86")]
#[macro_export]
macro_rules! asm_body_sse {
    (
        $step_macro:tt, $acc_macro:tt, $store_macro:tt,
        $mr:tt, $nr:tt, $b_layout:tt, $is_partial:tt,
        $a:tt, $b:tt, $c:tt, $ptr_arr:tt,
        $dim_arr:tt,
        [$($vreg:tt,)*]
    ) => {
        core::arch::asm!(
            vzero_kernel!(),

            init_ab!($b_layout),
            "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

            "2:", // KITER
            pire_base::prefetch_b!($b_layout),
            $step_macro!($nr, $b_layout, 0),
            $step_macro!($nr, $b_layout, 1),
            $step_macro!($nr, $b_layout, 2),
            $step_macro!($nr, $b_layout, 3),

            inc_a_k_unroll!($mr, 4),
            inc_b_k_unroll!($b_layout, $nr, 4),

            "dec {x0}", "jne 2b", // KITER

            "3:", // CONSIDKLEFT
            "mov 16({dim_arrx}), {x0}",
            "test {x0},{x0}", "je 5f", // POSTACCUM

            "4:", // KLEFT
            $step_macro!($nr, $b_layout, 0),
            inc_a_k_unroll!($mr, 1),
            inc_b_k_unroll!($b_layout, $nr, 1),

            "dec {x0}", "jne 4b", // KLEFT

            "5:", // POSTACCUM
            c_load!(),

            "cmpw $0, 24({dim_arrx})",
            "je 9f",
            alpha_scale!(),
            "9:",

            "cmpw $0, 20({dim_arrx})",
            "je 6f",

            "cmpw $1, 20({dim_arrx})",
            "je 15f",

            load_beta!(),
            pire_base::cum_seq!($acc_macro,$nr,$is_partial,2),
            "jmp 6f",

            "15:",
            pire_base::cum_seq!($acc_macro,$nr,$is_partial,1),

            "6:",
            pire_base::cum_seq!($store_macro,$nr,$is_partial),

            ax = inout(reg) $a => _,
            bx = inout(reg) $b => _,
            cx = inout(reg) $c => _,
            ptr_arrx = inout(reg) $ptr_arr.as_ptr() => _,
            dim_arrx = inout(reg) $dim_arr.as_ptr() => _,
            x0 = out(reg) _,
            $(out($vreg) _,)*
            options(att_syntax)
        );
    }
}

#[macro_export]
macro_rules! asm_body_avx {
    (
        $step_macro:tt, $acc_macro:tt, $store_macro:tt,
        $mr:tt, $nr:tt, $b_layout:tt, $is_partial:tt,
        $a:tt, $b:tt, $c:tt, $alpha:tt, $beta:tt, $alpha_st:tt, $beta_st:tt,
        $dim_arr:tt, | $($mask_ptr:ident,)? |
        [$($vreg:tt,)*]
    ) => {
        core::arch::asm!(
            vzero_kernel!(),

            init_ab_avx!($b_layout),

            "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

            "2:", // KITER
            pire_base::prefetch_b!($b_layout),
            pire_base::load_a!($mr, 0),
            $step_macro!($b_layout, $nr, 0),
            inc_b!($b_layout,$nr),

            pire_base::load_a!($mr, 1),
            $step_macro!($b_layout, $nr, 1),
            inc_b!($b_layout,$nr),

            pire_base::load_a!($mr, 2),
            $step_macro!($b_layout, $nr, 2),
            inc_b!($b_layout,$nr),

            pire_base::load_a!($mr, 3),
            $step_macro!($b_layout, $nr, 3),
            inc_b!($b_layout,$nr),

            pire_base::inc_a_k_unroll!($mr, 4),
            inc_b_k_unroll!($b_layout, $nr, 4),

            "dec {x0}", "jne 2b", // KITER

            "3:", // CONSIDKLEFT
            "mov 32({dim_arrx}), {x0}",
            "test {x0},{x0}", "je 5f", // POSTACCUM

            "4:", // KLEFT
            pire_base::load_a!($mr, 0),
            $step_macro!($b_layout, $nr, 0),
            inc_b!($b_layout,$nr),

            pire_base::inc_a_k_unroll!($mr, 1),
            inc_b_k_unroll!($b_layout, $nr, 1),

            "dec {x0}", "jne 4b", // KLEFT

            "5:", // POSTACCUM
            c_load!(),

            "cmpw $0, ({alpha_st})",
            "je 9f",
            alpha_scale!(),
            "9:",

            load_mask!($is_partial),

            "cmpw $0, ({beta_st})",
            "je 6f",

            "cmpw $1, ({beta_st})",
            "je 15f",

            load_beta!(),
            pire_base::cum_seq!($acc_macro,$is_partial,$nr,2),
            "jmp 6f",

            "15:",
            pire_base::cum_seq!($acc_macro,$is_partial,$nr,1),

            "6:",
            pire_base::cum_seq!($store_macro,$is_partial,$nr),
            "vzeroupper",

            ax = inout(reg) $a => _,
            bx = inout(reg) $b => _,
            cx = inout(reg) $c => _,
            dim_arrx = inout(reg) $dim_arr.as_ptr() => _,
            alphax = inout(reg) $alpha => _,
            betax = inout(reg) $beta => _,
            beta_st = in(reg) &$beta_st,
            alpha_st = in(reg) &$alpha_st,
            $(maskx = inout(reg) $mask_ptr => _,)?
            x0 = out(reg) _,
            x1 = out(reg) _,
            x2 = out(reg) _,
            x3 = out(reg) _,
            $(out($vreg) _,)*
            options(att_syntax)
        );
    }
}

#[macro_export]
macro_rules! asm_body_avx_2 {
    (
        $step_macro:tt, $acc_macro:tt, $store_macro:tt,
        $mr:tt, $nr:tt,
        $a:tt, $b:tt, $c:tt, $alpha:tt, $beta:tt, $alpha_st:tt, $beta_st:tt,
        $dim_arr:tt,
        [$($vreg:tt,)*]
    ) => {
        core::arch::asm!(
            vzero_kernel!(),
            init_ab_2!(B),
            "test {x0},{x0}",
            "je 3f",
            "mov {cx}, {x2}",
            "mov {ax}, {x5}",
            "mov 24({dim_arrx}),{x1}",
            "add {x1}, {x5}",
            "mov ({dim_arrx}),{x1}",
            "2:",
            pire_base::load_a!($mr, 0),
            $step_macro!(B, $nr, 0),
            inc_b!($nr),
            "movq $64*4, {x4}",
            // divisiblity by 4
            "testq $3, {x0}",
            "cmovz {x1},{x4}",
            pire_base::load_a!($mr, 1),
            $step_macro!(B, $nr, 1),
            inc_b!($nr),
            "prefetcht1 ({x2})",

            "subq $64*3, {x2}",
            "addq {x4}, {x2}",
            pire_base::load_a!($mr, 2),
            $step_macro!(B, $nr, 2),
            inc_b!($nr),
            "prefetcht1 ({x5})",
            "addq $16, {x5}",

            "testq $63, {x0}",
            "cmovz {cx},{x2}",
            pire_base::load_a!($mr, 3),
            $step_macro!(B, $nr, 3),
            inc_b!($nr),
            pire_base::inc_a_k_unroll!($mr, 4),
            inc_b_k_unroll!(B, $nr, 4),

            "dec {x0}",
            "jne 2b",
            "3:",
            "mov 16({dim_arrx}),{x0}",
            "test {x0},{x0}", "je 5f", // POSTACCUM

            "mov {cx}, {x2}",
            "mov ({dim_arrx}),{x1}",
            "4:",
            "prefetcht0 ({x2})",
            "prefetcht0 64({x2})",
            "prefetcht0 92({x2})",
            pire_base::load_a!($mr, 0),
            $step_macro!(B, $nr, 0),
            inc_b!($nr),
            pire_base::inc_a_k_unroll!($mr, 1),
            inc_b_k_unroll!(B, $nr, 1),
            "add {x1}, {x2}", "dec {x0}", "jne 4b",

            "5:",
            c_load_2!(),

            "cmpw $0, ({alpha_st})",
            "je 9f",
            alpha_scale!(),
            "9:",
            "cmpw $0, ({beta_st})",
            "je 6f",

            "cmpw $1, ({beta_st})",
            "je 15f",

            load_beta!(),
            pire_base::cum_seq!($acc_macro,C,$nr,2),
            "jmp 6f",

            "15:",
            pire_base::cum_seq!($acc_macro,C,$nr,1),

            "6:",
            pire_base::cum_seq!($store_macro,C, $nr),
            "vzeroupper",

            ax = inout(reg) $a => _,
            bx = inout(reg) $b => _,
            cx = inout(reg) $c => _,
            dim_arrx = inout(reg) $dim_arr.as_ptr() => _,
            alphax = inout(reg) $alpha => _,
            betax = inout(reg) $beta => _,
            beta_st = in(reg) &$beta_st,
            alpha_st = in(reg) &$alpha_st,
            x0 = out(reg) _,
            x1 = out(reg)_,
            x2 = out(reg) _,
            x3 = out(reg) _,
            x4 = out(reg) _,
            x5 = out(reg) _,
            $(out($vreg) _,)*
            options(att_syntax)
        );
    }
}

#[macro_export]
macro_rules! asm_body_avx512 {
    (
        $step_macro:tt, $acc_macro:tt, $store_macro:tt,
        $mr:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $is_partial:tt,
        $a:tt, $b:tt, $c:tt, $alpha:tt, $beta:tt, $alpha_st:tt, $beta_st:tt,
        $dim_arr:tt, | $($mask_ptr:ident,)? |
        [$($vreg:tt,)*]
    ) => {
        core::arch::asm!(
            vzero_kernel!(),
            load_mask!($is_partial),


            init_ab!($b_layout),
            "mov 40({dim_arrx}), {x5}",
            "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

            "2:", // KITER
            pire_base::load_a!($a_layout,$mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout,$nr),
            pire_base::load_a!($a_layout,$mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout,$nr),
            pire_base::load_a!($a_layout,$mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout,$nr),
            pire_base::load_a!($a_layout,$mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout,$nr),
            "dec {x0}", "jne 2b", // KITER

            "3:", // CONSIDKLEFT
            "mov 32({dim_arrx}), {x0}",
            "test {x0},{x0}", "je 5f", // POSTACCUM

            "4:", // KLEFT
            pire_base::load_a!($a_layout,$mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout,$nr),
            "dec {x0}", "jne 4b", // KLEFT

            "5:", // POSTACCUM
            c_load!(),

            "cmpw $0, ({alpha_st})",
            "je 9f",
            alpha_scale!(),
            "9:",

            // load_mask!($is_partial),

            "cmpw $0, ({beta_st})",
            "je 6f",

            "cmpw $1, ({beta_st})",
            "je 15f",

            load_beta!(),
            pire_base::cum_seq!($acc_macro,$is_partial,$nr,2),
            "jmp 6f",

            "15:",
            pire_base::cum_seq!($acc_macro,$is_partial,$nr,1),

            "6:",
            pire_base::cum_seq!($store_macro,$is_partial,$nr),
            "vzeroupper",

            ax = inout(reg) $a => _,
            bx = inout(reg) $b => _,
            cx = inout(reg) $c => _,
            dim_arrx = inout(reg) $dim_arr.as_ptr() => _,
            alphax = inout(reg) $alpha => _,
            betax = inout(reg) $beta => _,
            beta_st = in(reg) &$beta_st,
            alpha_st = in(reg) &$alpha_st,
            $(maskx = inout(reg) $mask_ptr => _,)?
            x0 = out(reg) _,
            x1 = out(reg) _,
            x2 = out(reg) _,
            x3 = out(reg) _,
            x4 = out(reg) _,
            x5 = out(reg) _,
            $(out($vreg) _,)*
            options(att_syntax)
        );
    }
}

#[macro_export]
macro_rules! asm_body_avx512_dot {
    (
        $step_macro:tt, $acc_macro:tt, $store_macro:tt,
        $b_macro:tt, $c_macro:tt,
        $mr:tt, $nr:tt,
        $a:tt, $b:tt, $c:tt, $alpha:tt, $beta:tt, $alpha_st:tt, $beta_st:tt,
        $dim_arr:tt, | $($mask_ptr:ident,)? |
        [$($vreg:tt,)*]
    ) => {
        core::arch::asm!(
            vzero_kernel!(),

            init_ab_dot!(),
            "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

            "2:", // KITER
            load_a_dot!($mr),
            $step_macro!($mr,$nr, $b_macro, $c_macro),
            "add $64, {ax}",
            "add $64, {x2}",
            "add $64, {x3}",
            "add $64, {bx}",
            "add $64, {x4}",
            load_a_dot!($mr),
            $step_macro!($mr,$nr, $b_macro, $c_macro),
            "add $64, {ax}",
            "add $64, {x2}",
            "add $64, {x3}",
            "add $64, {bx}",
            "add $64, {x4}",
            load_a_dot!($mr),
            $step_macro!($mr,$nr, $b_macro, $c_macro),
            "add $64, {ax}",
            "add $64, {x2}",
            "add $64, {x3}",
            "add $64, {bx}",
            "add $64, {x4}",
            load_a_dot!($mr),
            $step_macro!($mr,$nr, $b_macro, $c_macro),
            "add $64, {ax}",
            "add $64, {x2}",
            "add $64, {x3}",
            "add $64, {bx}",
            "add $64, {x4}",
            "dec {x0}", "jne 2b", // KITER

            "3:", // CONSIDKLEFT
            "mov 40({dim_arrx}), {x0}",
            "test {x0},{x0}", "je 23f", // POSTACCUM

            "4:", // KLEFT
            load_a_dot!($mr),
            $step_macro!($mr,$nr, $b_macro, $c_macro),
            "add $64, {ax}",
            "add $64, {x2}",
            "add $64, {x3}",
            "add $64, {bx}",
            "add $64, {x4}",
            "dec {x0}", "jne 4b", // KLEFT

            "23:", // CONSIDKLEFT
            "mov 48({dim_arrx}), {x0}",
            "test {x0},{x0}", "je 5f", // POSTACCUM

            "24:", // KLEFT
            load_mask!(P),
            load_a_dot_partial!($mr),
            $step_macro!($mr,$nr, $b_macro, $c_macro),

            "5:", // POSTACCUM
            c_load_dot!(),

            "cmpw $0, ({alpha_st})",
            "je 9f",
            alpha_scale!(),
            "9:",

            "cmpw $0, ({beta_st})",
            "je 6f",

            "cmpw $1, ({beta_st})",
            "je 15f",

            load_beta!(),
            acc_dot!($mr,$nr,2, $c_macro),
            "jmp 6f",

            "15:",
            acc_dot!($mr,$nr,1, $c_macro),

            "6:",
            store_dot!($mr,$nr, $c_macro),
            "vzeroupper",

            ax = inout(reg) $a => _,
            bx = inout(reg) $b => _,
            cx = inout(reg) $c => _,
            dim_arrx = inout(reg) $dim_arr.as_ptr() => _,
            alphax = inout(reg) $alpha => _,
            betax = inout(reg) $beta => _,
            beta_st = in(reg) &$beta_st,
            alpha_st = in(reg) &$alpha_st,
            x0 = out(reg) _,
            x1 = out(reg) _,
            x2 = out(reg) _,
            x3 = out(reg) _,
            x4 = out(reg) _,
            x5 = out(reg) _,
            $(maskx = inout(reg) $mask_ptr => _,)?
            $(out($vreg) _,)*
            options(att_syntax)
        );
    }
}

#[macro_export]
macro_rules! asm_body_avx512_2 {
    (
        $step_macro:tt, $acc_macro:tt, $store_macro:tt,
        $mr:tt, $nr:tt,
        $a:tt, $b:tt, $c:tt, $alpha:tt, $beta:tt, $alpha_st:tt, $beta_st:tt,
        $dim_arr:tt, $pf1_step:tt,
        [$($vreg:tt,)*]
    ) => {
        core::arch::asm!(
            vzero_kernel!(),
            init_ab_2!(B),
            "test {x0},{x0}", "je 3f",

            "mov {cx}, {x2}",
            "mov {ax}, {x5}",
            "mov 24({dim_arrx}),{x1}",
            "add {x1}, {x5}",
            "mov ({dim_arrx}),{x1}",

            "2:", // KITER
            pire_base::load_a!($mr),
            $step_macro!(B, $nr),
            inc_b!($nr),
            "movq $64*4, {x4}",
            // divisiblity by 4
            "testq $3, {x0}",
            "cmovz {x1},{x4}",

            pire_base::load_a!($mr),
            $step_macro!(B, $nr),
            inc_b!($nr),
            "prefetcht1 ({x2})",

            "subq $64*3, {x2}",
            "addq {x4}, {x2}",

            pire_base::load_a!($mr),
            $step_macro!(B, $nr),
            inc_b!($nr),

            "prefetcht1 ({x5})",
            concat!("addq $", $pf1_step, ", {x5}"),

            "testq $63, {x0}",
            "cmovz {cx},{x2}",

            pire_base::load_a!($mr),
            $step_macro!(B, $nr),
            inc_b!($nr),

            "dec {x0}", "jne 2b", // KITER

            "3:",
            "mov 16({dim_arrx}),{x0}",
            "test {x0},{x0}", "je 5f", // POSTACCUM


            "mov {cx}, {x2}",
            "mov ({dim_arrx}),{x1}",

            "4:", // KLEFT
            "prefetcht0 ({x2})",
            "prefetcht0 64({x2})",
            "prefetcht0 128({x2})",
            pire_base::load_a!($mr),
            $step_macro!(B, $nr),
            inc_b!($nr),
            "add {x1}, {x2}", "dec {x0}", "jne 4b", // KLEFT

            "5:", // POSTACCUM
            c_load_2!(),

            "cmpw $0, ({alpha_st})",
            "je 9f",
            alpha_scale!(),
            "9:",

            "cmpw $0, ({beta_st})",
            "je 6f",

            "cmpw $1, ({beta_st})",
            "je 15f",

            load_beta!(),
            pire_base::cum_seq!($acc_macro,C,$nr,2),
            "jmp 6f",

            "15:",
            pire_base::cum_seq!($acc_macro,C,$nr,1),

            "6:",
            pire_base::cum_seq!($store_macro,C,$nr),
            "vzeroupper",

            ax = inout(reg) $a => _,
            bx = inout(reg) $b => _,
            cx = inout(reg) $c => _,
            dim_arrx = inout(reg) $dim_arr.as_ptr() => _,
            alphax = inout(reg) $alpha => _,
            betax = inout(reg) $beta => _,
            beta_st = in(reg) &$beta_st,
            alpha_st = in(reg) &$alpha_st,
            x0 = out(reg) _,
            x1 = out(reg)_,
            x2 = out(reg) _,
            x3 = out(reg) _,
            x4 = out(reg) _,
            x5 = out(reg) _,
            $(out($vreg) _,)*
            options(att_syntax)
        );
    }
}

// *********************************************** def ukernel ************************************************

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! def_ukernel_sse {
    (
        $k_unit:tt,
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        pub(crate) unsafe fn $func_name(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 2], a_cs: usize, c_cs: usize,
            m: usize, n: usize,
        ) {
            use core::mem::size_of;
            let dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / ($k_unit*4), (k % ($k_unit*4)) / $k_unit];
            let alpha_st = if *alpha == ONE_SCALAR {
                0i32
            } else {
                1i32
            };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if n == $nr {
                pire_base::prefetch_c_sse!($mr,$nr,c,c_cs);
                pire_base::asm_body_avx!(
                    $step_macro, $acc_macro, $store_macro,
                    $mr, $nr, $b_layout, $is_partial,
                    a, b, c, alpha, beta, alpha_st, beta_st,
                    dim_arr, | |
                    [
                        "xmm0", "xmm1", "xmm2", "xmm3",
                        "xmm4", "xmm5", "xmm6", "xmm7",
                        "xmm8", "xmm9", "xmm10", "xmm11",
                        "xmm12", "xmm13", "xmm14", "xmm15",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if ni == n {
                            pire_base::prefetch_c_sse!($mr,ni,c,c_cs);
                            pire_base::asm_body_avx!(
                                $step_macro, $acc_macro, $store_macro,
                                $mr, ni, $b_layout, $is_partial,
                                a, b, c, alpha, beta, alpha_st, beta_st,
                                dim_arr, | |
                                [
                                    "xmm0", "xmm1", "xmm2", "xmm3",
                                    "xmm4", "xmm5", "xmm6", "xmm7",
                                    "xmm8", "xmm9", "xmm10", "xmm11",
                                    "xmm12", "xmm13", "xmm14", "xmm15",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! def_ukernel_avx {
    (
        $k_unit:tt,
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        pub(crate) unsafe fn $func_name(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 2], a_cs: usize, c_cs: usize,
            m: usize, n: usize,
        ) {
            use core::mem::size_of;
            mask_ptr!($is_partial, m, x, mask_ptr);
            let dim_arr = [d_arr[0]*size_of::<TA>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / ($k_unit*4), (k % ($k_unit*4)) / $k_unit];
            let alpha_st = if *alpha == ONE_SCALAR {
                0i32
            } else {
                1i32
            };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if n == $nr {
                pire_base::prefetch_c_avx!($mr,$nr,c,c_cs);
                pire_base::asm_body_avx!(
                    $step_macro, $acc_macro, $store_macro,
                    $mr, $nr, $b_layout, $is_partial,
                    a, b, c, alpha, beta, alpha_st, beta_st,
                    dim_arr, | mask_ptr, |
                    [
                        "ymm0", "ymm1", "ymm2", "ymm3",
                        "ymm4", "ymm5", "ymm6", "ymm7",
                        "ymm8", "ymm9", "ymm10", "ymm11",
                        "ymm12", "ymm13", "ymm14", "ymm15",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni {
                            pire_base::prefetch_c_avx!($mr,ni,c,c_cs);
                            pire_base::asm_body_avx!(
                                $step_macro, $acc_macro, $store_macro,
                                $mr, ni, $b_layout, $is_partial,
                                a, b, c, alpha, beta, alpha_st, beta_st,
                                dim_arr, | mask_ptr, |
                                [
                                    "ymm0", "ymm1", "ymm2", "ymm3",
                                    "ymm4", "ymm5", "ymm6", "ymm7",
                                    "ymm8", "ymm9", "ymm10", "ymm11",
                                    "ymm12", "ymm13", "ymm14", "ymm15",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            }
        }
    };
}

#[macro_export]
macro_rules! def_ukernel_avx512_dot {
    (
        $k_unit:tt,
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $b_macro:tt, $c_macro:tt,
        $mr:tt, $nr:tt,
        $func_name:ident
    ) => {
        pub(crate) unsafe fn $func_name(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            lda: usize, ldb: usize,
            c_rs: usize, c_cs: usize,
            m: usize, n: usize, k: usize,
        ) {
            use core::mem::size_of;
            mask_ptr!(P, k, x, mask_ptr);
            let dim_arr = [lda*TC_SIZE, ldb*TC_SIZE, c_rs*TC_SIZE, c_cs*TC_SIZE, k / ($k_unit*4), (k % ($k_unit*4)) / $k_unit, k % $k_unit];
            let alpha_st = if *alpha == ONE_SCALAR {
                0i32
            } else {
                1i32
            };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if n == $nr {
                pire_base::asm_body_avx512_dot!(
                    $step_macro, $acc_macro, $store_macro,
                    $b_macro, $c_macro,
                    $mr, $nr,
                    a, b, c, alpha, beta, alpha_st, beta_st,
                    dim_arr,| mask_ptr, |
                    [
                        "zmm0", "zmm1", "zmm2", "zmm3",
                        "zmm4", "zmm5", "zmm6", "zmm7",
                        "zmm8", "zmm9", "zmm10", "zmm11",
                        "zmm12", "zmm13", "zmm14", "zmm15",
                        "zmm16", "zmm17", "zmm18", "zmm19",
                        "zmm20", "zmm21", "zmm22", "zmm23",
                        "zmm24", "zmm25", "zmm26", "zmm27",
                        "zmm28", "zmm29", "zmm30", "zmm31",
                        "k1",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni {
                            pire_base::asm_body_avx512_dot!(
                                $step_macro, $acc_macro, $store_macro,
                                $b_macro, $c_macro,
                                $mr, ni,
                                a, b, c, alpha, beta, alpha_st, beta_st,
                                dim_arr,| mask_ptr, |
                                [
                                    "zmm0", "zmm1", "zmm2", "zmm3",
                                    "zmm4", "zmm5", "zmm6", "zmm7",
                                    "zmm8", "zmm9", "zmm10", "zmm11",
                                    "zmm12", "zmm13", "zmm14", "zmm15",
                                    "zmm16", "zmm17", "zmm18", "zmm19",
                                    "zmm20", "zmm21", "zmm22", "zmm23",
                                    "zmm24", "zmm25", "zmm26", "zmm27",
                                    "zmm28", "zmm29", "zmm30", "zmm31",
                                    "k1",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            }
        }
    };
}

#[macro_export]
macro_rules! def_ukernel_avx512 {
    (
        $k_unit:tt,
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $a_layout:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        pub(crate) unsafe fn $func_name(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 2], a_cs: usize, c_cs: usize,
            m: usize, n: usize,
        ) {
            use core::mem::size_of;
            mask_ptr!($is_partial, m, x, mask_ptr);
            let dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / ($k_unit*4), (k % ($k_unit*4)) / $k_unit, a_cs*TC_SIZE];
            let alpha_st = if *alpha == ONE_SCALAR {
                0i32
            } else {
                1i32
            };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if n == $nr {
                pire_base::prefetch_c_avx512!($mr,$nr,c,c_cs);
                pire_base::asm_body_avx512!(
                    $step_macro, $acc_macro, $store_macro,
                    $mr, $nr, $a_layout, $b_layout, $is_partial,
                    a, b, c, alpha, beta, alpha_st, beta_st,
                    dim_arr, | mask_ptr, |
                    [
                        "zmm0", "zmm1", "zmm2", "zmm3",
                        "zmm4", "zmm5", "zmm6", "zmm7",
                        "zmm8", "zmm9", "zmm10", "zmm11",
                        "zmm12", "zmm13", "zmm14", "zmm15",
                        "zmm16", "zmm17", "zmm18", "zmm19",
                        "zmm20", "zmm21", "zmm22", "zmm23",
                        "zmm24", "zmm25", "zmm26", "zmm27",
                        "zmm28", "zmm29", "zmm30", "zmm31",
                        "k1",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni {
                            pire_base::prefetch_c_avx512!($mr,ni,c,c_cs);
                            pire_base::asm_body_avx512!(
                                $step_macro, $acc_macro, $store_macro,
                                $mr, ni, $a_layout, $b_layout, $is_partial,
                                a, b, c, alpha, beta, alpha_st, beta_st,
                                dim_arr, | mask_ptr, |
                                [
                                    "zmm0", "zmm1", "zmm2", "zmm3",
                                    "zmm4", "zmm5", "zmm6", "zmm7",
                                    "zmm8", "zmm9", "zmm10", "zmm11",
                                    "zmm12", "zmm13", "zmm14", "zmm15",
                                    "zmm16", "zmm17", "zmm18", "zmm19",
                                    "zmm20", "zmm21", "zmm22", "zmm23",
                                    "zmm24", "zmm25", "zmm26", "zmm27",
                                    "zmm28", "zmm29", "zmm30", "zmm31",
                                    "k1",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! def_ukernel_avx_2 {
    ($k_unit:tt, $step_macro:tt, $acc_macro:tt, $store_macro:tt, $mr:tt, $nr:tt, $kl_pf:tt, $pf1_step:tt) => {
        pub(crate) unsafe fn ukernel_bbc(
            a: *const TA,
            b: *const TB,
            c: *mut TC,
            alpha: *const TS,
            beta: *const TS,
            k: usize,
            d_arr: [usize; 2],
            a_cs: usize,
            c_cs: usize,
            _m: usize,
            n: usize,
        ) {
            let k_l0 = k % $kl_pf;
            let k_l = if k_l0 == 0 { $kl_pf / $k_unit } else { k_l0 / $k_unit };
            let k_i = (k - k_l * $k_unit) / (4 * $k_unit);

            let dim_arr = [c_cs * TC_SIZE, k_i, k_l, d_arr[0]];
            let alpha_st = if *alpha == ONE_SCALAR { 0i32 } else { 1i32 };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if n == $nr {
                pire_base::asm_body_avx_2!(
                    $step_macro,
                    $acc_macro,
                    $store_macro,
                    $mr,
                    $nr,
                    a,
                    b,
                    c,
                    alpha,
                    beta,
                    alpha_st,
                    beta_st,
                    dim_arr,
                    [
                        "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
                        "ymm12", "ymm13", "ymm14", "ymm15",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni {
                            pire_base::asm_body_avx_2!(
                                $step_macro,
                                $acc_macro,
                                $store_macro,
                                $mr,
                                ni,
                                a,
                                b,
                                c,
                                alpha,
                                beta,
                                alpha_st,
                                beta_st,
                                dim_arr,
                                [
                                    "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
                                    "ymm12", "ymm13", "ymm14", "ymm15",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! def_ukernel_avx512_2 {
    ($k_unit:tt, $step_macro:ident, $acc_macro:ident, $store_macro:ident, $mr:tt, $nr:tt, $kl_pf:tt, $pf1_step:tt) => {
        pub(crate) unsafe fn ukernel_bbc(
            a: *const TA,
            b: *const TB,
            c: *mut TC,
            alpha: *const TS,
            beta: *const TS,
            k: usize,
            d_arr: [usize; 2],
            a_cs: usize,
            c_cs: usize,
            _m: usize,
            n: usize,
        ) {
            let k_l0 = k % $kl_pf;
            let k_l = if k_l0 == 0 { $kl_pf / $k_unit } else { k_l0 / $k_unit };
            let k_i = (k - k_l * $k_unit) / (4 * $k_unit);

            let dim_arr = [c_cs * TC_SIZE, k_i, k_l, d_arr[0]];
            let alpha_st = if *alpha == ONE_SCALAR { 0i32 } else { 1i32 };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if n == $nr {
                pire_base::asm_body_avx512_2!(
                    $step_macro,
                    $acc_macro,
                    $store_macro,
                    $mr,
                    $nr,
                    a,
                    b,
                    c,
                    alpha,
                    beta,
                    alpha_st,
                    beta_st,
                    dim_arr,
                    $pf1_step,
                    [
                        "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11",
                        "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22",
                        "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni {
                            pire_base::asm_body_avx512_2!(
                                $step_macro,
                                $acc_macro,
                                $store_macro,
                                $mr,
                                ni,
                                a,
                                b,
                                c,
                                alpha,
                                beta,
                                alpha_st,
                                beta_st,
                                dim_arr,
                                $pf1_step,
                                [
                                    "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11",
                                    "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22",
                                    "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            }
        }
    };
}

#[cfg(target_arch = "x86")]
#[macro_export]
macro_rules! def_ukernel_sse {
    (
        $k_unit:tt,
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        pub(crate) unsafe fn $func_name(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 2], c_cs: usize,
            m: usize, n: usize,
        ) {
            let alpha_st = if *alpha == ONE_SCALAR {
                0i32
            } else {
                1i32
            };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            let dim_arr = [c_cs*TC_SIZE, k / ($k_unit*4), (k % ($k_unit*4)) / $k_unit, beta_st as usize, alpha_st as usize];
            let mut ptr_arr = [alpha, beta];
            if n == $nr {
                pire_base::prefetch_c_sse!($mr,$nr,c,c_cs);
                core::arch::asm!(
                    vzero_kernel!(),

                    init_ab!($b_layout),
                    "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

                    "2:", // KITER
                    pire_base::prefetch_b!($b_layout),

                    pire_base::load_a!($mr, 0),
                    $step_macro!($b_layout, $nr, 0),
                    inc_b!($b_layout,$nr),

                    pire_base::load_a!($mr, 1),
                    $step_macro!($b_layout, $nr, 1),
                    inc_b!($b_layout,$nr),

                    pire_base::load_a!($mr, 2),
                    $step_macro!($b_layout, $nr, 2),
                    inc_b!($b_layout,$nr),

                    pire_base::load_a!($mr, 3),
                    $step_macro!($b_layout, $nr, 3),
                    inc_b!($b_layout,$nr),

                    pire_base::inc_a_k_unroll!($mr, 4),
                    inc_b_k_unroll!($b_layout, $nr, 4),

                    "dec {x0}", "jne 2b", // KITER

                    "3:", // CONSIDKLEFT
                    "mov 8({dim_arrx}), {x0}",
                    "test {x0},{x0}", "je 5f", // POSTACCUM

                    "4:", // KLEFT
                    pire_base::load_a!($mr, 0),
                    $step_macro!($b_layout, $nr, 0),
                    inc_b!($b_layout,$nr),

                    pire_base::inc_a_k_unroll!($mr, 1),
                    inc_b_k_unroll!($b_layout, $nr, 1),

                    "dec {x0}", "jne 4b", // KLEFT

                    "5:", // POSTACCUM
                    c_load!(),

                    "cmpw $0, 16({dim_arrx})",
                    "je 9f",
                    alpha_scale!(),
                    "9:",

                    "cmpw $0, 12({dim_arrx})",
                    "je 6f",

                    "cmpw $1, 12({dim_arrx})",
                    "je 15f",

                    load_beta!(),
                    pire_base::cum_seq!($acc_macro,$is_partial,$nr,2),
                    "jmp 6f",

                    "15:",
                    pire_base::cum_seq!($acc_macro,$is_partial,$nr,1),

                    "6:",
                    pire_base::cum_seq!($store_macro,$is_partial,$nr),

                    ax = inout(reg) a => _,
                    bx = inout(reg) b => _,
                    cx = inout(reg) c => _,
                    ptr_arrx = inout(reg) ptr_arr.as_ptr() => _,
                    dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                    x0 = out(reg) _,
                    out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                    out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                    options(att_syntax)
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni {
                            pire_base::prefetch_c_sse!($mr,ni,c,c_cs);
                            core::arch::asm!(
                                vzero_kernel!(),

                                init_ab!($b_layout),
                                "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

                                "2:", // KITER
                                pire_base::prefetch_b!($b_layout),
                                pire_base::load_a!($mr, 0),
                                $step_macro!($b_layout, ni, 0),
                                inc_b!($b_layout,ni),
                                pire_base::load_a!($mr, 1),
                                $step_macro!($b_layout, ni, 1),
                                inc_b!($b_layout,ni),
                                pire_base::load_a!($mr, 2),
                                $step_macro!($b_layout, ni, 2),
                                inc_b!($b_layout,ni),
                                pire_base::load_a!($mr, 3),
                                $step_macro!($b_layout, ni, 3),
                                inc_b!($b_layout,ni),

                                pire_base::inc_a_k_unroll!($mr, 4),
                                inc_b_k_unroll!($b_layout, ni, 4),

                                "dec {x0}", "jne 2b", // KITER

                                "3:", // CONSIDKLEFT
                                "mov 8({dim_arrx}), {x0}",
                                "test {x0},{x0}", "je 5f", // POSTACCUM

                                "4:", // KLEFT
                                pire_base::load_a!($mr, 0),
                                $step_macro!($b_layout, ni, 0),
                                inc_b!($b_layout,ni),
                                pire_base::inc_a_k_unroll!($mr, 1),
                                inc_b_k_unroll!($b_layout, ni, 1),

                                "dec {x0}", "jne 4b", // KLEFT

                                "5:", // POSTACCUM
                                c_load!(),

                                "cmpw $0, 16({dim_arrx})",
                                "je 9f",
                                alpha_scale!(),
                                "9:",

                                "cmpw $0, 12({dim_arrx})",
                                "je 6f",

                                "cmpw $1, 12({dim_arrx})",
                                "je 15f",

                                load_beta!(),
                                pire_base::cum_seq!($acc_macro,$is_partial,ni,2),
                                "jmp 6f",

                                "15:",
                                pire_base::cum_seq!($acc_macro,$is_partial,ni,1),

                                "6:",
                                pire_base::cum_seq!($store_macro,$is_partial,ni),

                                ax = inout(reg) a => _,
                                bx = inout(reg) b => _,
                                cx = inout(reg) c => _,
                                ptr_arrx = inout(reg) ptr_arr.as_ptr() => _,
                                dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                                x0 = out(reg) _,
                                out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                                out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                                options(att_syntax)
                            );
                            break 'blk;
                        }
                    });
                };
            };
        }
    };
}

#[macro_export]
macro_rules! asm_body_neon {
    (
        $step_macro:tt, $acc_macro:tt, $store_macro:tt,
        $mr:tt, $nr:tt, $b_layout:tt, $is_partial:tt,
        $a:tt, $b:tt, $c:tt, $alpha:tt, $beta:tt, $alpha_st:tt, $beta_st:tt,
        $dim_arr:tt, | $($alt:ident,)? |,
        | $($xreg:ident,)* |,
        [$($vreg:tt,)*]
    ) => {
        core::arch::asm!(
            prefetch_c!(),
            vzero_kernel!(),

            init_ab!($b_layout),

            // 3 -> CONSIDKLEFT
            "cmp {x0}, #0", "BEQ 3f",

            // 2 -> KITER
            "2:",
            pire_base::load_a!($mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout, $nr),
            pire_base::load_a!($mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout, $nr),
            pire_base::load_a!($mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout, $nr),
            pire_base::load_a!($mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout, $nr),

            "sub {x0}, {x0}, #1",
            // 2 -> KITER
            "cmp {x0}, 0",
            "BNE 2b",

            // 3 -> CONSIDKLEFT
            "3:",
            "ldr {x0}, [{dim_arrx}, #32]",
            "cmp {x0}, #0",

            // 5 -> POSTACCUM
            "BEQ 5f",
            // 4 -> KLEFT
            "4:",
            pire_base::load_a!($mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout, $nr),

            "sub {x0}, {x0}, #1",

            // 4 -> KLEFT
            "cmp {x0}, 0",
            "BNE 4b",

            // 5 -> POSTACCUM
            "5:",
            c_load!(),
            "cmp {alpha_st:w}, #0",
            "BEQ 13f",
            alpha_scale!(),
            "13:",

            "cmp {beta_st:w}, #0",
            "BEQ 6f",

            load_beta!(),

            pire_base::cum_seq!($acc_macro,$is_partial,$nr,2),

            // 6 -> BETAZERO
            "6:",
            pire_base::cum_seq!($store_macro,$is_partial,$nr),

            ax = inout(reg) $a => _,
            bx = inout(reg) $b => _,
            cx = inout(reg) $c => _,
            dim_arrx = inout(reg) $dim_arr.as_ptr() => _,
            alphax = inout(reg) $alpha => _,
            betax = inout(reg) $beta => _,
            beta_st = in(reg) &$beta_st,
            alpha_st = in(reg) &$alpha_st,
            $(altx = inout(reg) $alt.as_ptr() => _,)?
            $($xreg = out(reg) _,)*
            $(out($vreg) _,)*
        );
    }
}

#[macro_export]
macro_rules! asm_body_sve {
    (
        $step_macro:tt, $acc_macro:tt, $store_macro:tt,
        $mr:tt, $nr:tt, $b_layout:tt, $is_partial:tt,
        $a:tt, $b:tt, $c:tt, $alpha:tt, $beta:tt, $alpha_st:tt, $beta_st:tt,
        $m_left:tt, $inc_a:tt,
        $dim_arr:tt, | $($alt:ident,)? |,
        | $($xreg:ident,)* |,
        [$($vreg:tt,)*]
    ) => {
        core::arch::asm!(
            "ptrue p0.h",
            prefetch_c!(),
            vzero_kernel!(),

            init_ab!($b_layout),
            set_predicate!($is_partial),

            // 3 -> CONSIDKLEFT
            "cmp {x0}, #0", "BEQ 3f",

            // 2 -> KITER
            "2:",
            pire_base::load_a!($mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout, $nr),
            pire_base::load_a!($mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout, $nr),
            pire_base::load_a!($mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout, $nr),
            pire_base::load_a!($mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout, $nr),

            "sub {x0}, {x0}, #1",
            // 2 -> KITER
            "cmp {x0}, 0",
            "BNE 2b",

            // 3 -> CONSIDKLEFT
            "3:",
            "ldr {x0}, [{dim_arrx}, #32]",
            "cmp {x0}, #0",

            // 5 -> POSTACCUM
            "BEQ 5f",
            // 4 -> KLEFT
            "4:",
            pire_base::load_a!($mr),
            $step_macro!($b_layout, $nr),
            inc_b!($b_layout, $nr),

            "sub {x0}, {x0}, #1",

            // 4 -> KLEFT
            "cmp {x0}, 0",
            "BNE 4b",

            // 5 -> POSTACCUM
            "5:",
            c_load!(),
            "cmp {alpha_st:w}, #0",
            "BEQ 13f",
            alpha_scale!(),
            "13:",

            "cmp {beta_st:w}, #0",
            "BEQ 6f",

            "cmp {beta_st:w}, #1",
            "BEQ 9f",

            load_beta!(),
            pire_base::cum_seq!($acc_macro,$is_partial,$nr,2),
            "B 6f",

            "9:",
            // 9 -> BETAONE
            pire_base::cum_seq!($acc_macro,$is_partial,$nr,1),

            // 6 -> BETAZERO
            "6:",
            pire_base::cum_seq!($store_macro,$is_partial,$nr),
            ax = inout(reg) $a => _,
            bx = inout(reg) $b => _,
            cx = inout(reg) $c => _,
            dim_arrx = inout(reg) $dim_arr.as_ptr() => _,
            alphax = inout(reg) $alpha => _,
            betax = inout(reg) $beta => _,
            beta_st = in(reg) &$beta_st,
            alpha_st = in(reg) &$alpha_st,
            incax = in(reg) $inc_a as u64,
            m_s = out(reg) _,
            m_e = inout(reg) $m_left as u64 => _,
            $(altx = inout(reg) $alt.as_ptr() => _,)?
            $($xreg = out(reg) _,)*
            $(out($vreg) _,)*
        );
    }
}

#[macro_export]
macro_rules! def_ukernel_neon {
    (
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon")]
        pub(crate) unsafe fn $func_name(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 2], c_cs: usize,
            m: usize, n: usize,
        ) {
            use core::mem::size_of;
            let dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 4, k % 4];
            let alpha_st = if *alpha == ONE_SCALAR {
                0i32
            } else {
                1i32
            };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else {
                1i32
            };
            if n == $nr {
                pire_base::asm_body_neon!(
                    $step_macro, $acc_macro, $store_macro,
                    $mr, $nr, $b_layout, $is_partial,
                    a, b, c, alpha, beta, alpha_st, beta_st,
                    dim_arr, | |,
                    | x0, x1, x2, x3, x4, x5, x6, x7, |,
                    [
                        "v0", "v1", "v2", "v3",
                        "v4", "v5", "v6", "v7",
                        "v8", "v9", "v10", "v11",
                        "v12", "v13", "v14", "v15",
                        "v16", "v17", "v18", "v19",
                        "v20", "v21", "v22", "v23",
                        "v24", "v25", "v26", "v27",
                        "v28", "v29", "v30", "v31",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni {
                            pire_base::asm_body_neon!(
                                $step_macro, $acc_macro, $store_macro,
                                $mr, ni, $b_layout, $is_partial,
                                a, b, c, alpha, beta, alpha_st, beta_st,
                                dim_arr, | |,
                                | x0, x1, x2, x3, x4, x5, x6, x7, |,
                                [
                                    "v0", "v1", "v2", "v3",
                                    "v4", "v5", "v6", "v7",
                                    "v8", "v9", "v10", "v11",
                                    "v12", "v13", "v14", "v15",
                                    "v16", "v17", "v18", "v19",
                                    "v20", "v21", "v22", "v23",
                                    "v24", "v25", "v26", "v27",
                                    "v28", "v29", "v30", "v31",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            };
        }
    };
}

#[macro_export]
macro_rules! def_ukernel_neon_alt {
    (
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon")]
        pub(crate) unsafe fn $func_name(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 2], c_cs: usize,
            m: usize, n: usize,
        ) {
            alt_arr!(alt);
            use core::mem::size_of;
            let dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 4, k % 4];
            let alpha_st = if *alpha == ONE_SCALAR {
                0i32
            } else {
                1i32
            };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else {
                1i32
            };
            if n == $nr {
                pire_base::asm_body_neon!(
                    $step_macro, $acc_macro, $store_macro,
                    $mr, $nr, $b_layout, $is_partial,
                    a, b, c, alpha, beta, alpha_st, beta_st,
                    dim_arr, | alt, |,
                    | x0, x1, x2, x3, x4, x5, |,
                    [
                        "v0", "v1", "v2", "v3",
                        "v4", "v5", "v6", "v7",
                        "v8", "v9", "v10", "v11",
                        "v12", "v13", "v14", "v15",
                        "v16", "v17", "v18", "v19",
                        "v20", "v21", "v22", "v23",
                        "v24", "v25", "v26", "v27",
                        "v28", "v29", "v30", "v31",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni {
                            pire_base::asm_body_neon!(
                                $step_macro, $acc_macro, $store_macro,
                                $mr, ni, $b_layout, $is_partial,
                                a, b, c, alpha, beta, alpha_st, beta_st,
                                dim_arr, | alt, |,
                                | x0, x1, x2, x3, x4, x5, |,
                                [
                                    "v0", "v1", "v2", "v3",
                                    "v4", "v5", "v6", "v7",
                                    "v8", "v9", "v10", "v11",
                                    "v12", "v13", "v14", "v15",
                                    "v16", "v17", "v18", "v19",
                                    "v20", "v21", "v22", "v23",
                                    "v24", "v25", "v26", "v27",
                                    "v28", "v29", "v30", "v31",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            };
        }
    };
}

#[macro_export]
macro_rules! def_ukernel_neon_fp16 {
    (
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon,fp16")]
        pub(crate) unsafe fn $func_name(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 2], c_cs: usize,
            m: usize, n: usize,
        ) {
            use core::mem::size_of;
            let dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 4, k % 4];
            let alpha_st = if *alpha == ONE_SCALAR {
                0i32
            } else {
                1i32
            };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else {
                1i32
            };
            if n == $nr {
                pire_base::asm_body_neon!(
                    $step_macro, $acc_macro, $store_macro,
                    $mr, $nr, $b_layout, $is_partial,
                    a, b, c, alpha, beta, alpha_st, beta_st,
                    dim_arr, | |,
                    | x0, x1, x2, x3, x4, x5, x6, x7, |,
                    [
                        "v0", "v1", "v2", "v3",
                        "v4", "v5", "v6", "v7",
                        "v8", "v9", "v10", "v11",
                        "v12", "v13", "v14", "v15",
                        "v16", "v17", "v18", "v19",
                        "v20", "v21", "v22", "v23",
                        "v24", "v25", "v26", "v27",
                        "v28", "v29", "v30", "v31",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni {
                            pire_base::asm_body_neon!(
                                $step_macro, $acc_macro, $store_macro,
                                $mr, ni, $b_layout, $is_partial,
                                a, b, c, alpha, beta, alpha_st, beta_st,
                                dim_arr, | |,
                                | x0, x1, x2, x3, x4, x5, x6, x7, |,
                                [
                                    "v0", "v1", "v2", "v3",
                                    "v4", "v5", "v6", "v7",
                                    "v8", "v9", "v10", "v11",
                                    "v12", "v13", "v14", "v15",
                                    "v16", "v17", "v18", "v19",
                                    "v20", "v21", "v22", "v23",
                                    "v24", "v25", "v26", "v27",
                                    "v28", "v29", "v30", "v31",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            };
        }
    };
}

#[macro_export]
macro_rules! def_ukernel_neon_i8mm {
    (
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon,i8mm")]
        pub(crate) unsafe fn $func_name(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 2], c_cs: usize,
            m: usize, n: usize,
        ) {
            use core::mem::size_of;
            let dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 32, (k % 32) / 8];
            let alpha_st = if *alpha == ONE_SCALAR {
                0i32
            } else {
                1i32
            };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if n == $nr {
                pire_base::asm_body_neon!(
                    $step_macro, $acc_macro, $store_macro,
                    $mr, $nr, $b_layout, $is_partial,
                    a, b, c, alpha, beta, alpha_st, beta_st,
                    dim_arr, | |,
                    | x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, |,
                    [
                        "v0", "v1", "v2", "v3",
                        "v4", "v5", "v6", "v7",
                        "v8", "v9", "v10", "v11",
                        "v12", "v13", "v14", "v15",
                        "v16", "v17", "v18", "v19",
                        "v20", "v21", "v22", "v23",
                        "v24", "v25", "v26", "v27",
                        "v28", "v29", "v30", "v31",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni {
                            pire_base::asm_body_neon!(
                                $step_macro, $acc_macro, $store_macro,
                                $mr, ni, $b_layout, $is_partial,
                                a, b, c, alpha, beta, alpha_st, beta_st,
                                dim_arr, | |,
                                | x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, |,
                                [
                                    "v0", "v1", "v2", "v3",
                                    "v4", "v5", "v6", "v7",
                                    "v8", "v9", "v10", "v11",
                                    "v12", "v13", "v14", "v15",
                                    "v16", "v17", "v18", "v19",
                                    "v20", "v21", "v22", "v23",
                                    "v24", "v25", "v26", "v27",
                                    "v28", "v29", "v30", "v31",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            };
        }
    };
}

#[macro_export]
macro_rules! def_ukernel_sve {
    (
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon,sve")]
        pub(crate) unsafe fn $func_name(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 2], c_cs: usize,
            m: usize, n: usize,
        ) {
            use core::mem::size_of;
            let vs = sve_vs();
            let m_left = if m % vs == 0 {vs} else {m%vs};
            let inc_a = vs * $mr * size_of::<TA>();
            let mr = $mr * vs;
            let dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 4, k % 4];
            let alpha_st = if *alpha == ONE_SCALAR {
                0i32
            } else {
                1i32
            };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if n == $nr {
                pire_base::asm_body_sve!(
                    $step_macro, $acc_macro, $store_macro,
                    $mr, $nr, $b_layout, $is_partial,
                    a, b, c, alpha, beta, alpha_st, beta_st,
                    m_left, inc_a,
                    dim_arr, | |,
                    | x0, x1, x2, x3, x4, x5, x6, x7, |,
                    [
                        "v0", "v1", "v2", "v3",
                        "v4", "v5", "v6", "v7",
                        "v8", "v9", "v10", "v11",
                        "v12", "v13", "v14", "v15",
                        "v16", "v17", "v18", "v19",
                        "v20", "v21", "v22", "v23",
                        "v24", "v25", "v26", "v27",
                        "v28", "v29", "v30", "v31",
                        "p0", "p1", "p2", "p3",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni{
                            pire_base::asm_body_sve!(
                                $step_macro, $acc_macro, $store_macro,
                                $mr, ni, $b_layout, $is_partial,
                                a, b, c, alpha, beta, alpha_st, beta_st,
                                m_left, inc_a,
                                dim_arr, | |,
                                | x0, x1, x2, x3, x4, x5, x6, x7, |,
                                [
                                    "v0", "v1", "v2", "v3",
                                    "v4", "v5", "v6", "v7",
                                    "v8", "v9", "v10", "v11",
                                    "v12", "v13", "v14", "v15",
                                    "v16", "v17", "v18", "v19",
                                    "v20", "v21", "v22", "v23",
                                    "v24", "v25", "v26", "v27",
                                    "v28", "v29", "v30", "v31",
                                    "p0", "p1", "p2", "p3",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            };
        }
    };
}

#[macro_export]
macro_rules! def_ukernel_sve_i8mm {
    (
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $b_layout:tt,
        $is_partial:tt,
        // $feature_enable:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon,sve,i8mm")]
        pub(crate) unsafe fn $func_name(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 2], c_cs: usize,
            m: usize, n: usize,
        ) {
            use core::mem::size_of;
            let vs = sve_vs();
            let m_left = if m % vs == 0 {vs} else {m%vs};
            let inc_a = $mr * vs * size_of::<TA>() * 8;
            let mr = $mr * vs;
            let dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 32, (k % 32) / 8];
            let alpha_st = if *alpha == ONE_SCALAR {
                0i32
            } else {
                1i32
            };
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if n == $nr {
                pire_base::asm_body_sve!(
                    $step_macro, $acc_macro, $store_macro,
                    $mr, $nr, $b_layout, $is_partial,
                    a, b, c, alpha, beta, alpha_st, beta_st,
                    m_left, inc_a,
                    dim_arr, | |,
                    | x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, |,
                    [
                        "v0", "v1", "v2", "v3",
                        "v4", "v5", "v6", "v7",
                        "v8", "v9", "v10", "v11",
                        "v12", "v13", "v14", "v15",
                        "v16", "v17", "v18", "v19",
                        "v20", "v21", "v22", "v23",
                        "v24", "v25", "v26", "v27",
                        "v28", "v29", "v30", "v31",
                        "p0", "p1", "p2", "p3",
                    ]
                );
            } else {
                let _ = 'blk: {
                    seq!(ni in 1..$nr {
                        if n == ni {
                            pire_base::asm_body_sve!(
                                $step_macro, $acc_macro, $store_macro,
                                $mr, ni, $b_layout, $is_partial,
                                a, b, c, alpha, beta, alpha_st, beta_st,
                                m_left, inc_a,
                                dim_arr, | |,
                                | x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, |,
                                [
                                    "v0", "v1", "v2", "v3",
                                    "v4", "v5", "v6", "v7",
                                    "v8", "v9", "v10", "v11",
                                    "v12", "v13", "v14", "v15",
                                    "v16", "v17", "v18", "v19",
                                    "v20", "v21", "v22", "v23",
                                    "v24", "v25", "v26", "v27",
                                    "v28", "v29", "v30", "v31",
                                    "p0", "p1", "p2", "p3",
                                ]
                            );
                            break 'blk;
                        }
                    });
                };
            };
        }
    };
}

// mod test {
//     // test split_c_range
//     #[test]
//     fn test_split_c_range() {
//         let m = 143;
//         let mc = 4800;
//         let mr = 24;
//         let ic_par = 4;
//         for ic_id in 0..ic_par {
//             let (mc_start, mc_end, mc_left) = super::split_c_range(m, mc, mr, ic_id, ic_par);
//             println!("mc_start: {}, mc_end: {}, mc_left: {}", mc_start, mc_end, mc_left);
//         }
//         assert!(false);
//     }
// }
