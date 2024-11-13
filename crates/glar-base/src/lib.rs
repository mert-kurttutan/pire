use std::sync::{Barrier, Mutex, MutexGuard, RwLock, RwLockReadGuard};
// Consider Once Cell
use once_cell::sync::Lazy;

use core::mem::size_of;

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

#[macro_export]
macro_rules! env_or {
    ($name:expr, $default:expr) => {
        if let Some(value) = core::option_env!($name) {
            const_str::parse!(value, usize)
        } else {
            $default
        }
    };
}

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

// padding in bytes
const CACHELINE_PAD: usize = 1024;

pub struct HWConfig {
    pub cpu_ft: CpuFeatures,
    pub hw_model: HWModel,
    is_l1_shared: bool,
    is_l2_shared: bool,
    is_l3_shared: bool,
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
    pub fn from_hw(family_id: u8, model_id: u8) -> Self {
        if family_id == 6 {
            if SKYLAKE.contains(&model_id) {
                return HWModel::Skylake;
            }
            if HASWELL.contains(&model_id) {
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
        let extended_prcoessor_info = cpuid.get_extended_processor_and_feature_identifiers().unwrap();
        let fma4 = extended_prcoessor_info.has_fma4();
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
        let hw_model = HWModel::from_hw(family_id, model_id);
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_model.get_cache_info();
        return HWConfig { cpu_ft, hw_model, is_l1_shared, is_l2_shared, is_l3_shared };
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
        let hw_model = HWModel::from_hw(family_id, model_id);
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
        // println!("neon: {}, sve: {}, fp16: {}, f32mm: {}, fcma: {}", neon, sve, fp16, f32mm, fcma);
        // let fcma = is_aarch64_feature_detected!("fcma");

        return HWConfig {
            cpu_ft: CpuFeatures { neon, sve, fp16, f32mm, fcma, i8mm },
            hw_model: HWModel::Reference,
            is_l1_shared: false,
            is_l2_shared: false,
            is_l3_shared: true,
        };
    }
}

pub static RUNTIME_HW_CONFIG: Lazy<HWConfig> = Lazy::new(|| detect_hw_config());

pub static GLAR_NUM_THREADS: Lazy<usize> = Lazy::new(|| {
    let n_core = std::thread::available_parallelism().unwrap().get();
    // GLAR_NUM_THREADS or the number of logical cores
    let x = std::env::var("GLAR_NUM_THREADS").unwrap_or(n_core.to_string());
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

pub struct GlarThreadConfig<'a> {
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
    pub par: GlarPar,
    pub packa_barrier: &'a [Barrier],
    pub packb_barrier: &'a [Barrier],
}

pub fn get_apbp_barrier(par: &GlarPar) -> (Vec<Barrier>, Vec<Barrier>) {
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

impl<'a> GlarThreadConfig<'a> {
    pub fn new(
        par: GlarPar,
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
pub fn glar_num_threads() -> usize {
    return *GLAR_NUM_THREADS;
}

#[derive(Copy, Clone)]
pub struct GlarPar {
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
impl GlarPar {
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
        let num_threads = glar_num_threads();
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
        let a_alignment = std::mem::align_of::<TA>();
        assert_eq!(ap_pool_size_b % a_alignment, 0);
        let bp_pool_size = self.bp_pool_size;
        let bp_pool_size_b = bp_pool_size * size_of::<TB>();
        let b_alignment = std::mem::align_of::<TB>();
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
            let ap_pool = unsafe { std::slice::from_raw_parts_mut::<TA>(a.as_mut_ptr() as *mut TA, ap_pool_size) };
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
            let bp_pool = unsafe { std::slice::from_raw_parts_mut::<TB>(b.as_mut_ptr() as *mut TB, bp_pool_size) };
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
    par: &GlarPar,
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
    par: &GlarPar,
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
    par: &GlarPar,
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
macro_rules! def_glar_gemm {
    (
        $t_dispatcher:tt,
        $ta:tt,$tap:ty,$tb:ty,$tbp:ty,$tc:ty,$t_as:ty,$t_bs:ty,
        $packa_ty:tt,$packb_ty:tt,
        $one:expr,
        $name:ident, $name_mt:ident,
        $goto_name:ident, $goto_kernel:ident,
        $small_m_name:ident, $small_m_kernel:ident,
        $small_n_name:ident, $small_n_kernel:ident,
        $gemv_name:ident, $gemv_name2:ident,
        $packa_name:ident, $packb_name:ident,
        $packa_name0:ident, $packb_name0:ident,
        $run_small_m:expr, $run_small_n:expr,
        $pack_fn:tt, $include_flag:tt,
    ) => {
        def_pa!($packa_ty,$include_flag,$ta,$tap);
        def_pa!($packb_ty,$include_flag,$tb,$tbp);
        pub unsafe fn $name <F:UnaryFnC>(
            hw_config: &$t_dispatcher <F>,
            m: usize, n: usize, k: usize,
            alpha: $t_as,
            a: Array<$ta>,
            b: Array<$tb>,
            beta: $t_bs,
            c: ArrayMut<$tc>,
            par: &GlarPar,
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
            let (gemm_mode, gemm_fun, pool_info)
            : (
                GemmPool, unsafe fn(
                    &$t_dispatcher <F>, usize, usize, usize, *const $t_as, $packa_ty, $packb_ty, *const $t_bs, ArrayMut<$tc>, &GlarThreadConfig,
                ),
                PoolSize
            )
             = if run_small_m(m) && $run_small_m && b.is_strided() {
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

        pub unsafe fn $name_mt<F:UnaryFnC>(
            hw_config: &$t_dispatcher <F>,
            m: usize, n: usize, k: usize,
            alpha: $t_as,
            a: Array<$ta>,
            b: Array<$tb>,
            beta: $t_bs,
            c: ArrayMut<$tc>,
            par: &GlarPar,
            pool_buf: &mut [u8],
            gemm_mode: GemmPool,
            pool_info: PoolSize,
            gemm_fn: unsafe fn(
                &$t_dispatcher <F>, usize, usize, usize, *const $t_as, $packa_ty, $packb_ty, *const $t_bs, ArrayMut<$tc>, &GlarThreadConfig
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
                    let t_cfg = GlarThreadConfig::new(
                        par.clone(), &pa_br_vec_ref, &pb_br_vec_ref, t_id, mc_eff, nc_eff, kc_eff
                    );
                    let ic_id = t_cfg.ic_id;
                    let jc_id = t_cfg.jc_id;
                    let ap_id = match gemm_mode {
                        GemmPool::Goto => ic_id,
                        GemmPool::SmallM => ic_id,
                        GemmPool::SmallN => t_id,
                    };
                    let bp_id = match gemm_mode {
                        GemmPool::Goto => jc_id,
                        GemmPool::SmallM => 0,
                        GemmPool::SmallN => jc_id,
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
                    let t_cfg = GlarThreadConfig::new(par.clone(), &pa_br_vec_ref, &pb_br_vec_ref, t_id, mc_eff, nc_eff, kc_eff);
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
            t_cfg: &GlarThreadConfig
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
            t_cfg: &GlarThreadConfig
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
            t_cfg: &GlarThreadConfig
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
        // for packed api mc_i(nc_i) should be multiple of mr (nr, which we ensure by the split_c_range
        // for packed api kc_i should be multiple of kc_eff, which is always true since we dont parallelize over kc
        // this is subject to change if we parallelize over kc, but this is not in the plan
        // sync right before write and right before read
        // NOTE: dont return before the second packa as it ensures sync between threads
        pub(crate) unsafe fn $packa_name<'a,'b,F:UnaryFnC>(hw_cfg: &$t_dispatcher <F>, x: &'b $packa_ty<'a>, mc_i: usize, kc_i: usize, mc_len: usize, kc_len: usize, t_cfg: &GlarThreadConfig) -> PtrData<'a,$tap> {
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
        pub(crate) unsafe fn $packb_name<'a,'b,F:UnaryFnC>(hw_cfg: & $t_dispatcher <F>, x: &'b$packb_ty<'a>, nc_i: usize, kc_i: usize, nc_len: usize, kc_len: usize, t_cfg: &GlarThreadConfig) -> PtrData<'a,$tbp> {
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
macro_rules! partial_strided {
    ($strided:tt, $strided2:tt, F) => {
        $strided
    };
    ($strided:tt, $strided2:tt, T) => {
        $strided2
    };
}

pub fn avx_vzeroupper() {
    #[cfg(target_arch = "x86_64")]
    if (*RUNTIME_HW_CONFIG).cpu_ft.avx {
        unsafe {
            core::arch::asm!("vzeroupper");
        }
    }
}

#[macro_export]
macro_rules! def_kernel_bb_pf1 {
    (
        $t_ap:ty, $t_bp:ty, $t_c:ty, $t_s:ty,
        $no_partial:tt,
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
            const STRIDED_PARTIAL: bool = true;
            const MR: usize = $MR * VS;
            const NR: usize = $NR;
            let m_rounded = m / MR * MR;
            let n_rounded = n / NR * NR;
            let m_left = m % MR;
            let n_left = n % NR;

            let d_arr = [0, 0, c_rs];

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let ap_cur = ap.add(m_i * k);
                let mut a_pft1_offset = $pf1_0 * k;
                let mut n_i = 0;
                while n_i < n_rounded {
                    let bp_cur = bp.add(n_i * k);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    ukernel_bbc::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, a_pft1_offset, NR, f);
                    n_i += NR;
                    a_pft1_offset += $pf_step * k;
                }
                // let a_pft1_offset = ($MR+(n_iter0-n_iter)*2)*4*k;
                if n_left != 0 {
                    let bp_cur = bp.add(n_i * k);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    ukernel_n_bbc::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, MR, n_left, f);
                }
                m_i += MR;
            }


            seq_macro::seq!(mr_left in 1..=$MR {
                if (m_left+VS-1) / VS == mr_left {
                    let c_cur0 = c.add(m_i * c_rs);
                    let ap_cur = ap.add(m_i * k);
                    let mut n_i = 0;
                    while n_i < n_rounded {
                        let bp_cur = bp.add(n_i * k);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        paste::paste! {
                            [<ukernel_ mr_left _bbp>]::<_, glar_base::partial_strided!(STRIDED,STRIDED_PARTIAL,$no_partial)>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, m_left, NR, f);
                        }
                        n_i += NR;
                    }
                    if n_left !=0 {
                        let bp_cur = bp.add(n_i * k);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        paste::paste! {
                            [<ukernel_ mr_left xn_bbp>]::<_, glar_base::partial_strided!(STRIDED,STRIDED_PARTIAL,$no_partial)>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, m_left, n_left, f);
                        }
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
            glar_base::avx_vzeroupper();
        }

    };
}

#[macro_export]
macro_rules! def_kernel_bb_v0 {
    (
        $t_ap:ty, $t_bp:ty, $t_c:ty, $t_s:ty,
        $no_partial:tt,
        $RS:tt,
        $MR:tt, $NR:tt
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
            const STRIDED_PARTIAL: bool = true;
            let mr = $MR * vs;
            const NR: usize = $NR;
            let m_rounded = m / mr * mr;
            let n_rounded = n / NR * NR;
            let m_left = m % mr;
            let n_left = n % NR;

            let d_arr = [0, 0, c_rs];

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let ap_cur = ap.add(m_i * k);
                let mut n_i = 0;
                while n_i < n_rounded {
                    let bp_cur = bp.add(n_i * k);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    ukernel_bbc::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, mr, NR, f);
                    n_i += NR;
                }
                if n_left != 0 {
                    let bp_cur = bp.add(n_i * k);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    ukernel_n_bbc::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, mr, n_left, f);
                }
                m_i += mr;
            }

            seq_macro::seq!(mr_left in 1..=$MR {
                if (m_left+vs-1) / vs == mr_left {
                    let c_cur0 = c.add(m_i * c_rs);
                    let ap_cur = ap.add(m_i * k);
                    let mut n_i = 0;
                    while n_i < n_rounded {
                        let bp_cur = bp.add(n_i * k);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        paste::paste! {
                            [<ukernel_ mr_left _bbp>]::<_, glar_base::partial_strided!(STRIDED,STRIDED_PARTIAL,$no_partial)>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, m_left, NR, f);
                        }
                        n_i += NR;
                    }
                    if n_left !=0 {
                        let bp_cur = bp.add(n_i * k);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        paste::paste! {
                            [<ukernel_ mr_left xn_bbp>]::<_, glar_base::partial_strided!(STRIDED,STRIDED_PARTIAL,$no_partial)>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, c_cs, m_left, n_left, f);
                        }
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
            glar_base::avx_vzeroupper();
        }
    };
}

#[macro_export]
macro_rules! def_kernel_sb_pf1 {
    (
        $t_a:ty, $t_ap:ty, $t_bp:ty, $t_c:ty, $t_s:ty,
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
            const MR: usize = $MR * VS;
            const NR: usize = $NR;
            let m_rounded = m / MR * MR;
            let n_rounded = n / NR * NR;
            let m_left = m % MR;
            let n_left = n % NR;

            let d_arr = [0, 0, c_rs];

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let a_cur = a.add(m_i * a_rs);
                let a_pft1_offset = $pf1_0 * k;
                $pack_fn(MR, k, a_cur, a_rs, a_cs, ap, VS);
                let mut n_i = 0;
                while n_i < n_rounded {
                    let bp_cur = bp.add(n_i * k_eff);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    ukernel_bbc::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, c_cs, a_pft1_offset, NR, f);
                    n_i += NR;
                }
                if n_left != 0 {
                    let bp_cur = bp.add(n_i * k_eff);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    ukernel_n_bbc::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, c_cs, MR, n_left, f);
                }
                m_i += MR;
            }

            seq_macro::seq!(mr_left in 1..=$MR {
                if (m_left+VS-1) / VS == mr_left {
                    let c_cur0 = c.add(m_i * c_rs);
                    let a_cur = a.add(m_i * a_rs);
                    $pack_fn(m_left, k, a_cur, a_rs, a_cs, ap, VS);
                    let mut n_i = 0;
                    while n_i < n_rounded {
                        let bp_cur = bp.add(n_i * k_eff);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        paste::paste! {
                            [<ukernel_ mr_left _bbp>]::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, c_cs, m_left, NR, f);
                        }
                        n_i += NR;
                    }
                    if n_left != 0 {
                        let bp_cur = bp.add(n_i * k_eff);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        paste::paste! {
                            [<ukernel_ mr_left xn_bbp>]::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, c_cs, m_left, n_left, f);
                        }
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
            glar_base::avx_vzeroupper();
        }
    };
}

#[macro_export]
macro_rules! def_kernel_sb_v0 {
    (
        $t_a:ty, $t_ap:ty, $t_bp:ty, $t_c:ty, $t_s:ty,
        $no_partial:tt,
        $pack_fn:tt,
        $RS:tt,
        $MR:tt, $NR:tt
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
            const STRIDED_PARTIAL: bool = true;
            let vs = simd_vector_length();
            let mr = $MR * vs;
            const NR: usize = $NR;
            let m_rounded = m / mr * mr;
            let n_rounded = n / NR * NR;
            let m_left = m % mr;
            let n_left = n % NR;

            let d_arr = [0, 0, c_rs];

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let a_cur = a.add(m_i * a_rs);
                $pack_fn(mr, k, a_cur, a_rs, a_cs, ap, vs);
                let mut n_i = 0;
                while n_i < n_rounded {
                    let bp_cur = bp.add(n_i * k_eff);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    ukernel_bbc::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, c_cs, mr, NR, f);
                    n_i += NR;
                }
                if n_left != 0 {
                    let bp_cur = bp.add(n_i * k_eff);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    ukernel_n_bbc::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, c_cs, mr, n_left, f);
                }
                m_i += mr;
            }

            seq_macro::seq!(mr_left in 1..=$MR {
                if (m_left+vs-1) / vs == mr_left {
                    let c_cur0 = c.add(m_i * c_rs);
                    let a_cur = a.add(m_i * a_rs);
                    $pack_fn(m_left, k, a_cur, a_rs, a_cs, ap, vs);
                    let mut n_i = 0;
                    while n_i < n_rounded {
                        let bp_cur = bp.add(n_i * k_eff);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        paste::paste! {
                            [<ukernel_ mr_left _bbp>]::<_, glar_base::partial_strided!(STRIDED,STRIDED_PARTIAL,$no_partial)>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, c_cs, m_left, NR, f);
                        }
                        n_i += NR;
                    }
                    if n_left != 0 {
                        let bp_cur = bp.add(n_i * k_eff);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        paste::paste! {
                            [<ukernel_ mr_left xn_bbp>]::<_, glar_base::partial_strided!(STRIDED,STRIDED_PARTIAL,$no_partial)>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, c_cs, m_left, n_left, f);
                        }
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
            glar_base::avx_vzeroupper();
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
            let n_rounded = n / NR * NR;
            let m_left = m % MR;
            let n_left = n % NR;

            let d_arr = [b_rs, b_cs, c_rs];

            let mut m_i = 0;
            while m_i < m_rounded {
                let c_cur0 = c.add(m_i * c_rs);
                let ap_cur = ap.add(m_i * k);
                let mut n_i = 0;
                while n_i < n_rounded {
                    let b_cur = b.add(n_i * b_cs);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    ukernel_bsc::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, d_arr, c_cs, MR, NR, f);
                    n_i += NR;
                }
                if n_left != 0 {
                    let b_cur = b.add(n_i * b_cs);
                    let c_cur1 = c_cur0.add(n_i * c_cs);
                    ukernel_n_bsc::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, d_arr, c_cs, MR, n_left, f);
                }
                m_i += MR;
            }
            seq_macro::seq!(mr_left in 1..=$MR {
                if (m_left+VS-1) / VS == mr_left {
                    let c_cur0 = c.add(m_i * c_rs);
                    let ap_cur = ap.add(m_i * k);
                    let mut n_i = 0;
                    while n_i < n_rounded {
                        let b_cur = b.add(n_i * b_cs);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        paste::paste! {
                            [<ukernel_ mr_left _bsp>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, d_arr, c_cs, m_left, NR, f);
                        }
                        n_i += NR;
                    }
                    if n_left != 0 {
                        let b_cur = b.add(n_i * b_cs);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        paste::paste! {
                            [<ukernel_ mr_left xn_bsp>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, d_arr, c_cs, m_left, n_left, f);
                        }
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
            glar_base::avx_vzeroupper();
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_export]
macro_rules! mem {
    ($m0:tt, $b0:tt) => {
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
macro_rules! n_cond {
    (1, $ni:tt, $nr:tt) => {
        $ni == $nr
    };
    ($n0:tt, $ni:tt, $nr:tt) => {
        true
    };
}

#[macro_export]
macro_rules! load_a_avx {
    ($mr:tt, $K:tt) => {
        glar_base::loadp_avx!($mr, concat!($mr, "*32*", $K, "({ax})"))
    };
}
#[macro_export]
macro_rules! load_a_avx512 {
    ($mr:tt) => {
        glar_base::loadp_avx512!($mr, "0({ax})")
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
    (11) => {
        concat!(
            "mov 16({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x3}\n",
            "lea ({cx}, {x3},), {x1}\n",
            "lea ({x1}, {x3},), {x2}\n",
            "lea ({x2}, {x3},), {x3}\n",
        )
    };
    (10) => {
        concat!(
            "mov 16({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x3}\n",
            "lea ({cx}, {x3},), {x1}\n",
            "lea ({x1}, {x3},), {x2}\n",
            "lea ({x2}, {x3},), {x3}\n",
        )
    };
    (9) => {
        concat!(
            "mov 16({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x3}\n",
            "lea ({cx}, {x3},), {x1}\n",
            "lea ({x1}, {x3},), {x2}\n",
        )
    };
    (8) => {
        concat!(
            "mov 16({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x3}\n",
            "lea ({cx}, {x3},), {x1}\n",
            "lea ({x1}, {x3},), {x2}\n",
        )
    };
    (7) => {
        concat!(
            "mov 16({dim_arrx}),{x0}\n",
            "lea ({x0}, {x0}, 2), {x3}\n",
            "lea ({cx}, {x3},), {x1}\n",
            "lea ({x1}, {x3},), {x2}\n",
        )
    };
    (6) => {
        concat!("mov 16({dim_arrx}),{x0}", "\n", "lea ({x0}, {x0}, 2), {x1}", "\n", "lea ({cx}, {x1},), {x1}", "\n",)
    };
    (5) => {
        concat!("mov 16({dim_arrx}),{x0}", "\n", "lea ({x0}, {x0}, 2), {x1}", "\n", "lea ({cx}, {x1},), {x1}", "\n",)
    };
    (4) => {
        concat!("mov 16({dim_arrx}),{x0}", "\n", "lea ({x0}, {x0}, 2), {x1}", "\n", "lea ({cx}, {x1},), {x1}", "\n",)
    };
    (3) => {
        concat!("mov 16({dim_arrx}),{x0}", "\n",)
    };
    (2) => {
        concat!("mov 16({dim_arrx}),{x0}", "\n",)
    };
    (1) => {
        concat!("mov 16({dim_arrx}),{x0}", "\n",)
    };
    ($nr:tt) => {
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
macro_rules! b_reg {
    (0) => {
        "({bx})"
    };
    (1) => {
        "({bx},{x2},1)"
    };
    (2) => {
        "({bx},{x2},2)"
    };
    (3) => {
        "({x3})"
    };
    (4) => {
        "({x3},{x2},1)"
    };
    (5) => {
        "({x3},{x2},2)"
    };
    (6) => {
        "({x4})"
    };
    (7) => {
        "({x4},{x2},1)"
    };
    (8) => {
        "({x4},{x2},2)"
    };
    (9) => {
        "({x5})"
    };
    (10) => {
        "({x5},{x2},1)"
    };
    (11) => {
        "({x5},{x2},2)"
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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
macro_rules! c_reg_2x4 {
    (0,0) => {
        4
    };
    (1,0) => {
        5
    };
    (0,1) => {
        6
    };
    (1,1) => {
        7
    };
    (0,2) => {
        8
    };
    (1,2) => {
        9
    };
    (0,3) => {
        10
    };
    (1,3) => {
        11
    };
}
#[macro_export]
macro_rules! c_reg_1x4 {
    (0,0) => {
        7
    };
    (0,1) => {
        8
    };
    (0,2) => {
        9
    };
    (0,3) => {
        10
    };
}
#[macro_export]
macro_rules! c_reg_3x4 {
    (0,0) => {
        4
    };
    (1,0) => {
        5
    };
    (2,0) => {
        6
    };
    (0,1) => {
        7
    };
    (1,1) => {
        8
    };
    (2,1) => {
        9
    };
    (0,2) => {
        10
    };
    (1,2) => {
        11
    };
    (2,2) => {
        12
    };
    (0,3) => {
        13
    };
    (1,3) => {
        14
    };
    (2,3) => {
        15
    };
}
#[macro_export]
macro_rules! c_reg_2x6 {
    (0,0) => {
        4
    };
    (1,0) => {
        5
    };
    (0,1) => {
        6
    };
    (1,1) => {
        7
    };
    (0,2) => {
        8
    };
    (1,2) => {
        9
    };
    (0,3) => {
        10
    };
    (1,3) => {
        11
    };
    (0,4) => {
        12
    };
    (1,4) => {
        13
    };
    (0,5) => {
        14
    };
    (1,5) => {
        15
    };
}
#[macro_export]
macro_rules! c_reg_1x6 {
    (0,0) => {
        7
    };
    (0,1) => {
        8
    };
    (0,2) => {
        9
    };
    (0,3) => {
        10
    };
    (0,4) => {
        11
    };
    (0,5) => {
        12
    };
}
#[macro_export]
macro_rules! c_reg_3x8 {
    (0,0) => {
        8
    };
    (1,0) => {
        9
    };
    (2,0) => {
        10
    };
    (0,1) => {
        11
    };
    (1,1) => {
        12
    };
    (2,1) => {
        13
    };
    (0,2) => {
        14
    };
    (1,2) => {
        15
    };
    (2,2) => {
        16
    };
    (0,3) => {
        17
    };
    (1,3) => {
        18
    };
    (2,3) => {
        19
    };
    (0,4) => {
        20
    };
    (1,4) => {
        21
    };
    (2,4) => {
        22
    };
    (0,5) => {
        23
    };
    (1,5) => {
        24
    };
    (2,5) => {
        25
    };
    (0,6) => {
        26
    };
    (1,6) => {
        27
    };
    (2,6) => {
        28
    };
    (0,7) => {
        29
    };
    (1,7) => {
        30
    };
    (2,7) => {
        31
    };
}
#[macro_export]
macro_rules! c_reg_2x12 {
    (0,0) => {
        8
    };
    (1,0) => {
        9
    };
    (0,1) => {
        10
    };
    (1,1) => {
        11
    };
    (0,2) => {
        12
    };
    (1,2) => {
        13
    };
    (0,3) => {
        14
    };
    (1,3) => {
        15
    };
    (0,4) => {
        16
    };
    (1,4) => {
        17
    };
    (0,5) => {
        18
    };
    (1,5) => {
        19
    };
    (0,6) => {
        20
    };
    (1,6) => {
        21
    };
    (0,7) => {
        22
    };
    (1,7) => {
        23
    };
    (0,8) => {
        24
    };
    (1,8) => {
        25
    };
    (0,9) => {
        26
    };
    (1,9) => {
        27
    };
    (0,10) => {
        28
    };
    (1,10) => {
        29
    };
    (0,11) => {
        30
    };
    (1,11) => {
        31
    };
}
#[macro_export]
macro_rules! c_reg_1x12 {
    (0,0) => {
        20
    };
    (0,1) => {
        21
    };
    (0,2) => {
        22
    };
    (0,3) => {
        23
    };
    (0,4) => {
        24
    };
    (0,5) => {
        25
    };
    (0,6) => {
        26
    };
    (0,7) => {
        27
    };
    (0,8) => {
        28
    };
    (0,9) => {
        29
    };
    (0,10) => {
        30
    };
    (0,11) => {
        31
    };
}

#[macro_export]
macro_rules! acc_3x4 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx!($layout, c_mem!($ni), $q, c_reg_3x4!(0, $ni), c_reg_3x4!(1, $ni), c_reg_3x4!(2, $ni))
    };
}
#[macro_export]
macro_rules! store_3x4 {
    ($ni:tt, $layout:tt) => {
        storep_avx!($layout, c_mem!($ni), c_reg_3x4!(0, $ni), c_reg_3x4!(1, $ni), c_reg_3x4!(2, $ni))
    };
}
#[macro_export]
macro_rules! acc_2x6 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx!($layout, c_mem!($ni), $q, c_reg_2x6!(0, $ni), c_reg_2x6!(1, $ni))
    };
}
#[macro_export]
macro_rules! store_2x6 {
    ($ni:tt, $layout:tt) => {
        storep_avx!($layout, c_mem!($ni), c_reg_2x6!(0, $ni), c_reg_2x6!(1, $ni))
    };
}
#[macro_export]
macro_rules! acc_1x6 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx!($layout, c_mem!($ni), $q, c_reg_1x6!(0, $ni))
    };
}
#[macro_export]
macro_rules! store_1x6 {
    ($ni:tt, $layout:tt) => {
        storep_avx!($layout, c_mem!($ni), c_reg_1x6!(0, $ni))
    };
}

#[macro_export]
macro_rules! acc_3x8 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx512!($layout, c_mem!($ni), $q, c_reg_3x8!(0, $ni), c_reg_3x8!(1, $ni), c_reg_3x8!(2, $ni))
    };
}
#[macro_export]
macro_rules! store_3x8 {
    ($ni:tt, $layout:tt) => {
        storep_avx512!($layout, c_mem!($ni), c_reg_3x8!(0, $ni), c_reg_3x8!(1, $ni), c_reg_3x8!(2, $ni))
    };
}
#[macro_export]
macro_rules! acc_2x12 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx512!($layout, c_mem!($ni), $q, c_reg_2x12!(0, $ni), c_reg_2x12!(1, $ni))
    };
}
#[macro_export]
macro_rules! store_2x12 {
    ($ni:tt, $layout:tt) => {
        storep_avx512!($layout, c_mem!($ni), c_reg_2x12!(0, $ni), c_reg_2x12!(1, $ni))
    };
}
#[macro_export]
macro_rules! acc_1x12 {
    ($ni:tt, $layout:tt, $q:tt) => {
        acc_p_avx512!($layout, c_mem!($ni), $q, c_reg_1x12!(0, $ni))
    };
}
#[macro_export]
macro_rules! store_1x12 {
    ($ni:tt, $layout:tt) => {
        storep_avx512!($layout, c_mem!($ni), c_reg_1x12!(0, $ni))
    };
}
#[macro_export]
macro_rules! acc_p_avx {
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1, $q),
            beta_fmadd!(C, glar_base::mem!($m0, "0x20"), $r2, $q),
            beta_fmadd!($layout, glar_base::mem!($m0, "0x40"), $r3, $q),
        )
    };
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr) => {
        concat!(beta_fmadd!(C, $m0, $r1, $q), beta_fmadd!($layout, glar_base::mem!($m0, "0x20"), $r2, $q),)
    };
    ($layout:tt, $m0:expr, $q:tt, $r1:expr) => {
        concat!(beta_fmadd!($layout, $m0, $r1, $q),)
    };
}

#[macro_export]
macro_rules! loadp_avx {
    (3, $m0:expr) => {
        concat!(
            loadp_unit!($m0, 0),
            loadp_unit!(glar_base::mem!($m0, "0x20"), 1),
            loadp_unit!(glar_base::mem!($m0, "0x40"), 2),
        )
    };
    (2, $m0:expr) => {
        concat!(loadp_unit!($m0, 0), loadp_unit!(glar_base::mem!($m0, "0x20"), 1),)
    };
    (1, $m0:expr) => {
        concat!(loadp_unit!($m0, 0),)
    };
}

#[macro_export]
macro_rules! storep_avx {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            storep_unit!(C, $r1, $m0),
            storep_unit!(C, $r2, glar_base::mem!($m0, "0x20")),
            storep_unit!($layout, $r3, glar_base::mem!($m0, "0x40")),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(storep_unit!(C, $r1, $m0), storep_unit!($layout, $r2, glar_base::mem!($m0, "0x20")),)
    };
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(storep_unit!($layout, $r1, $m0),)
    };
}

#[macro_export]
macro_rules! acc_p_avx512 {
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1, $q),
            beta_fmadd!(C, glar_base::mem!($m0, "0x40"), $r2, $q),
            beta_fmadd!($layout, glar_base::mem!($m0, "0x80"), $r3, $q),
        )
    };
    ($layout:tt, $m0:expr, $q:tt, $r1:expr, $r2:expr) => {
        concat!(beta_fmadd!(C, $m0, $r1, $q), beta_fmadd!($layout, glar_base::mem!($m0, "0x40"), $r2, $q),)
    };
    ($layout:tt, $m0:expr, $q:tt, $r1:expr) => {
        concat!(beta_fmadd!($layout, $m0, $r1, $q),)
    };
}

#[macro_export]
macro_rules! loadp_avx512 {
    (3, $m0:expr) => {
        concat!(
            loadp_unit!($m0, 0),
            loadp_unit!(glar_base::mem!($m0, "0x40"), 1),
            loadp_unit!(glar_base::mem!($m0, "0x80"), 2),
        )
    };
    (2, $m0:expr) => {
        concat!(loadp_unit!($m0, 0), loadp_unit!(glar_base::mem!($m0, "0x40"), 1),)
    };
    (1, $m0:expr) => {
        concat!(loadp_unit!($m0, 0),)
    };
}

#[macro_export]
macro_rules! storep_avx512 {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            storep_unit!(C, $r1, $m0),
            storep_unit!(C, $r2, glar_base::mem!($m0, "0x40")),
            storep_unit!($layout, $r3, glar_base::mem!($m0, "0x80")),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(storep_unit!(C, $r1, $m0), storep_unit!($layout, $r2, glar_base::mem!($m0, "0x40")),)
    };
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(storep_unit!($layout, $r1, $m0),)
    };
}

#[macro_export]
macro_rules! dim_to_reg {
    ($f:tt, 3, 8) => {
        $f!(8, 31)
    };
    ($f:tt, 3, 7) => {
        $f!(8, 28)
    };
    ($f:tt, 3, 6) => {
        $f!(8, 25)
    };
    ($f:tt, 3, 5) => {
        $f!(8, 22)
    };
    ($f:tt, 3, 4) => {
        $f!(8, 19)
    };
    ($f:tt, 3, 3) => {
        $f!(8, 16)
    };
    ($f:tt, 3, 2) => {
        $f!(8, 13)
    };
    ($f:tt, 3, 1) => {
        $f!(8, 10)
    };

    ($f:tt, 2, 12) => {
        $f!(8, 31)
    };
    ($f:tt, 2, 11) => {
        $f!(8, 29)
    };
    ($f:tt, 2, 10) => {
        $f!(8, 27)
    };
    ($f:tt, 2, 9) => {
        $f!(8, 25)
    };
    ($f:tt, 2, 8) => {
        $f!(8, 23)
    };
    ($f:tt, 2, 7) => {
        $f!(8, 21)
    };
    ($f:tt, 2, 6) => {
        $f!(8, 19)
    };
    ($f:tt, 2, 5) => {
        $f!(8, 17)
    };
    ($f:tt, 2, 4) => {
        $f!(8, 15)
    };
    ($f:tt, 2, 3) => {
        $f!(8, 13)
    };
    ($f:tt, 2, 2) => {
        $f!(8, 11)
    };
    ($f:tt, 2, 1) => {
        $f!(8, 9)
    };

    ($f:tt, 1, 12) => {
        $f!(20, 31)
    };
    ($f:tt, 1, 11) => {
        $f!(20, 30)
    };
    ($f:tt, 1, 10) => {
        $f!(20, 29)
    };
    ($f:tt, 1, 9) => {
        $f!(20, 28)
    };
    ($f:tt, 1, 8) => {
        $f!(20, 27)
    };
    ($f:tt, 1, 7) => {
        $f!(20, 26)
    };
    ($f:tt, 1, 6) => {
        $f!(20, 25)
    };
    ($f:tt, 1, 5) => {
        $f!(20, 24)
    };
    ($f:tt, 1, 4) => {
        $f!(20, 23)
    };
    ($f:tt, 1, 3) => {
        $f!(20, 22)
    };
    ($f:tt, 1, 2) => {
        $f!(20, 21)
    };
    ($f:tt, 1, 1) => {
        $f!(20, 20)
    };
}

#[macro_export]
macro_rules! dim_to_reg_avx {
    ($f:tt, 3, 4) => {
        $f!(4, 15)
    };
    ($f:tt, 3, 3) => {
        $f!(4, 12)
    };
    ($f:tt, 3, 2) => {
        $f!(4, 9)
    };
    ($f:tt, 3, 1) => {
        $f!(4, 6)
    };

    ($f:tt, 2, 6) => {
        $f!(4, 15)
    };
    ($f:tt, 2, 5) => {
        $f!(4, 13)
    };
    ($f:tt, 2, 4) => {
        $f!(4, 11)
    };
    ($f:tt, 2, 3) => {
        $f!(4, 9)
    };
    ($f:tt, 2, 2) => {
        $f!(4, 7)
    };
    ($f:tt, 2, 1) => {
        $f!(4, 5)
    };

    ($f:tt, 1, 6) => {
        $f!(7, 12)
    };
    ($f:tt, 1, 5) => {
        $f!(7, 11)
    };
    ($f:tt, 1, 4) => {
        $f!(7, 10)
    };
    ($f:tt, 1, 3) => {
        $f!(7, 9)
    };
    ($f:tt, 1, 2) => {
        $f!(7, 8)
    };
    ($f:tt, 1, 1) => {
        $f!(7, 7)
    };
}

#[macro_export]

macro_rules! cum_seq {
    ($step_macro:tt, $nr:tt, $layout:tt, $b:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!(n, $layout, $b),)*)
        })
    };
    ($step_macro:tt, $nr:tt, $layout:tt) => {
        seq!(n in 0..$nr {
            concat!(#($step_macro!(n, $layout),)*)
        })
    };
}

#[macro_export]
macro_rules! b_num_3x8 {
    (0) => {
        3
    };
    (1) => {
        4
    };
    (2) => {
        5
    };
    (3) => {
        6
    };
    (4) => {
        7
    };
    (5) => {
        3
    };
    (6) => {
        4
    };
    (7) => {
        5
    };
}
#[macro_export]
macro_rules! b_num_2x12 {
    (0) => {
        2
    };
    (1) => {
        3
    };
    (2) => {
        4
    };
    (3) => {
        5
    };
    (4) => {
        6
    };
    (5) => {
        7
    };
    (6) => {
        2
    };
    (7) => {
        3
    };
    (8) => {
        4
    };
    (9) => {
        5
    };
    (10) => {
        6
    };
    (11) => {
        7
    };
}
#[macro_export]
macro_rules! b_num_1x12 {
    (0) => {
        1
    };
    (1) => {
        2
    };
    (2) => {
        3
    };
    (3) => {
        4
    };
    (4) => {
        5
    };
    (5) => {
        6
    };
    (6) => {
        7
    };
    (7) => {
        8
    };
    (8) => {
        9
    };
    (9) => {
        10
    };
    (10) => {
        11
    };
    (11) => {
        12
    };
}

#[macro_export]
macro_rules! b_num_2x4 {
    (0) => {
        2
    };
    (1) => {
        3
    };
    (2) => {
        2
    };
    (3) => {
        3
    };
}
#[macro_export]
macro_rules! b_num_1x4 {
    (0) => {
        1
    };
    (1) => {
        2
    };
    (2) => {
        3
    };
    (3) => {
        4
    };
}
#[macro_export]
macro_rules! b_num_2x6 {
    (0) => {
        2
    };
    (1) => {
        3
    };
    (2) => {
        2
    };
    (3) => {
        3
    };
    (4) => {
        2
    };
    (5) => {
        3
    };
}

#[macro_export]
macro_rules! b_num_1x6 {
    (0) => {
        1
    };
    (1) => {
        2
    };
    (2) => {
        3
    };
    (3) => {
        4
    };
    (4) => {
        5
    };
    (5) => {
        6
    };
}

#[macro_export]
macro_rules! fmadd_3x8 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, b_num_3x8!($ni), c_reg_3x8!(0, $ni)),
            vfmadd!(1, b_num_3x8!($ni), c_reg_3x8!(1, $ni)),
            vfmadd!(2, b_num_3x8!($ni), c_reg_3x8!(2, $ni)),
        )
    };
}
#[macro_export]
macro_rules! fmadd_2x12 {
    ($ni:tt) => {
        concat!(vfmadd!(0, b_num_2x12!($ni), c_reg_2x12!(0, $ni)), vfmadd!(1, b_num_2x12!($ni), c_reg_2x12!(1, $ni)),)
    };
}
#[macro_export]
macro_rules! fmadd_1x12 {
    ($ni:tt) => {
        concat!(vfmadd!(0, b_num_1x12!($ni), c_reg_1x12!(0, $ni)),)
    };
}

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! prefetch_c_avx512 {
    (3, $nr:tt, $c:tt, $ldc:tt) => {
        use std::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
            _mm_prefetch(c_u8.add(64), 3);
            _mm_prefetch(c_u8.add(128), 3);
        });
    };
    (2, $nr:tt, $c:tt, $ldc:tt) => {
        use std::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
            _mm_prefetch(c_u8.add(64), 3);
        });
    };
    (1, $nr:tt, $c:tt, $ldc:tt) => {
        use std::arch::x86_64::_mm_prefetch;
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
        use std::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
            _mm_prefetch(c_u8.add(64), 3);
            _mm_prefetch(c_u8.add(92), 3);
        });
    };
    (2, $nr:tt, $c:tt, $ldc:tt) => {
        use std::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
            _mm_prefetch(c_u8.add(60), 3);
        });
    };
    (1, $nr:tt, $c:tt, $ldc:tt) => {
        use std::arch::x86_64::_mm_prefetch;
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
        use std::arch::x86::_mm_prefetch;
        #[cfg(target_arch="x86_64")]
        use std::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
            _mm_prefetch(c_u8.add(64), 3);
        });
    };
    (2, $nr:tt, $c:tt, $ldc:tt) => {
        #[cfg(target_arch="x86")]
        use std::arch::x86::_mm_prefetch;
        #[cfg(target_arch="x86_64")]
        use std::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
        });
    };
    (1, $nr:tt, $c:tt, $ldc:tt) => {
        #[cfg(target_arch="x86")]
        use std::arch::x86::_mm_prefetch;
        #[cfg(target_arch="x86_64")]
        use std::arch::x86_64::_mm_prefetch;
        seq!(j in 0..$nr {
            let c_u8 = $c.add(j*$ldc) as *const i8;
            _mm_prefetch(c_u8, 3);
        });
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_export]
macro_rules! prefetch_0 {
    ($dist:tt, $reg:tt) => {
        concat!("prefetcht0 ", $dist, "(", $reg, ")\n",)
    };
}

#[cfg(target_arch = "aarch64")]
#[macro_export]
macro_rules! prefetch_0 {
    ($dist:tt, $reg:tt) => {
        concat!("prfm pldl1keep, [", $reg, ", #", $dist, "] \n",)
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

// *********************************************** def ukernel ************************************************

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! def_ukernel_avx {
    (
        $k_unit:tt,
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $n0:tt, $n1:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            m: usize, n: usize,
            f: F,
        ) {
            use core::mem::size_of;
            const MR: usize = $mr * VS;
            mask_ptr!($is_partial, m, x, mask_ptr);
            let mut dim_arr = [d_arr[0]*size_of::<TA>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / ($k_unit*4), (k % ($k_unit*4)) / $k_unit];
            let mut c_k = c;
            let mut c_buf = [ZERO;MR*$nr];
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if BUF {
                glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, MR);
                dim_arr[2] = MR*TC_SIZE;
                c_k = c_buf.as_mut_ptr();
            }
            let _ = 'blk: {
                seq!(ni in $n0..$n1 {
                    if glar_base::n_cond!($n0, ni, n) {
                        glar_base::prefetch_c_avx!($mr,ni,c,c_cs);
                        asm!(
                            vzero_kernel!(),

                            init_ab_avx!($b_layout),

                            "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

                            "2:", // KITER
                            glar_base::prefetch_b!($b_layout),
                            $step_macro!(ni, $b_layout, 0),
                            $step_macro!(ni, $b_layout, 1),
                            $step_macro!(ni, $b_layout, 2),
                            $step_macro!(ni, $b_layout, 3),

                            inc_a_k_unroll!($mr, 4),
                            inc_b_k_unroll!($b_layout, ni, 4),

                            "dec {x0}", "jne 2b", // KITER

                            "3:", // CONSIDKLEFT
                            "mov 32({dim_arrx}), {x0}",
                            "test {x0},{x0}", "je 5f", // POSTACCUM

                            "4:", // KLEFT
                            $step_macro!(ni, $b_layout, 0),
                            inc_a_k_unroll!($mr, 1),
                            inc_b_k_unroll!($b_layout, ni, 1),

                            "dec {x0}", "jne 4b", // KLEFT

                            "5:", // POSTACCUM
                            glar_base::c_load!(ni),

                            alpha_scale!($mr, ni),

                            load_mask!($is_partial),

                            "cmpw $0, ({beta_st})",
                            "je 6f",

                            "cmpw $1, ({beta_st})",
                            "je 15f",

                            load_beta!(),
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,2),
                            "jmp 6f",

                            "15:",
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,1),

                            "6:",
                            glar_base::cum_seq!($store_macro,ni,$is_partial),

                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) c_k => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            betax = inout(reg) beta => _,
                            beta_st = in(reg) &beta_st,
                            maskx = inout(reg) mask_ptr => _,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
                            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
                            out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
                            out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
                            options(att_syntax)
                        );
                        break 'blk;
                    }
                });
            };
            if BUF {
                for j in 0..n {
                    f.call(c_k.add(j*MR), MR);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, m, n, MR);
            } else {
                for j in 0..n {
                    f.call(c_k.add(j*c_cs), m);
                }
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
        $n0:tt, $n1:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            m: usize, n: usize,
            f: F,
        ) {
            use core::mem::size_of;
            const MR: usize = $mr * VS;
            mask_ptr!($is_partial, m, x, mask_ptr);
            let mut dim_arr = [d_arr[0]*size_of::<TA>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / ($k_unit*4), (k % ($k_unit*4)) / $k_unit];
            let mut c_k = c;
            let mut c_buf = [ZERO;MR*$nr];
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if BUF {
                glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, MR);
                dim_arr[2] = MR*TC_SIZE;
                c_k = c_buf.as_mut_ptr();
            }
            let _ = 'blk: {
                seq!(ni in $n0..$n1 {
                    if glar_base::n_cond!($n0, ni, n) {
                        glar_base::prefetch_c_avx512!($mr,ni,c,c_cs);
                        asm!(
                            vzero_kernel!(),

                            init_ab!($b_layout),
                            "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

                            "2:", // KITER
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            "dec {x0}", "jne 2b", // KITER

                            "3:", // CONSIDKLEFT
                            "mov 32({dim_arrx}), {x0}",
                            "test {x0},{x0}", "je 5f", // POSTACCUM

                            "4:", // KLEFT
                            $step_macro!(ni, $b_layout),

                            "dec {x0}", "jne 4b", // KLEFT

                            "5:", // POSTACCUM
                            glar_base::c_load!(ni),

                            alpha_scale!($mr, ni),

                            load_mask!($is_partial),

                            "cmpw $0, ({beta_st})",
                            "je 6f",

                            "cmpw $1, ({beta_st})",
                            "je 15f",

                            load_beta!(),
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,2),
                            "jmp 6f",

                            "15:",
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,1),

                            "6:",
                            glar_base::cum_seq!($store_macro,ni,$is_partial),

                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) c_k => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            betax = inout(reg) beta => _,
                            beta_st = in(reg) &beta_st,
                            maskx = inout(reg) mask_ptr => _,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            x4 = out(reg) _,
                            x5 = out(reg) _,
                            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
                            out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
                            out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
                            out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
                            out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
                            out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
                            out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
                            out("zmm28") _, out("zmm29") _, out("zmm30") _, out("zmm31") _,
                            out("k1") _,
                            options(att_syntax)
                        );
                        break 'blk;
                    }
                });
            };
            if BUF {
                for j in 0..n {
                    f.call(c_k.add(j*MR), MR);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, m, n, MR);
            } else {
                for j in 0..n {
                    f.call(c_k.add(j*c_cs), m);
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! def_ukernel_sse {
    (
        $k_unit:tt,
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $n0:tt, $n1:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            m: usize, n: usize,
            f: F,
        ) {
            use core::mem::size_of;
            const MR: usize = $mr * VS;
            let mut dim_arr = [d_arr[0]*size_of::<TA>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / ($k_unit*4), (k % ($k_unit*4)) / $k_unit];
            let mut c_k = c;
            let mut c_buf = [ZERO;MR*$nr];
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if BUF {
                glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, MR);
                dim_arr[2] = MR*TC_SIZE;
                c_k = c_buf.as_mut_ptr();
            }
            let _ = 'blk: {
                seq!(ni in $n0..$n1 {
                    if glar_base::n_cond!($n0, ni, n) {
                        glar_base::prefetch_c_sse!($mr,ni,c,c_cs);
                        asm!(
                            vzero_kernel!(),

                            init_ab_avx!($b_layout),
                            "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

                            "2:", // KITER
                            glar_base::prefetch_b!($b_layout),
                            $step_macro!(ni, $b_layout, 0),
                            $step_macro!(ni, $b_layout, 1),
                            $step_macro!(ni, $b_layout, 2),
                            $step_macro!(ni, $b_layout, 3),

                            inc_a_k_unroll!($mr, 4),
                            inc_b_k_unroll!($b_layout, ni, 4),
                            "dec {x0}", "jne 2b", // KITER

                            "3:", // CONSIDKLEFT
                            "mov 32({dim_arrx}), {x0}",
                            "test {x0},{x0}", "je 5f", // POSTACCUM

                            "4:", // KLEFT
                            $step_macro!(ni, $b_layout, 0),
                            inc_a_k_unroll!($mr, 1),
                            inc_b_k_unroll!($b_layout, ni, 1),

                            "dec {x0}", "jne 4b", // KLEFT

                            "5:", // POSTACCUM
                            glar_base::c_load!(ni),

                            alpha_scale!($mr, ni),

                            "cmpw $0, ({beta_st})",
                            "je 6f",

                            "cmpw $1, ({beta_st})",
                            "je 15f",

                            load_beta!(),
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,2),
                            "jmp 6f",

                            "15:",
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,1),

                            "6:",
                            glar_base::cum_seq!($store_macro,ni,$is_partial),

                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) c_k => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            betax = inout(reg) beta => _,
                            beta_st = in(reg) &beta_st,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                            out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                            out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
                            out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
                            options(att_syntax)
                        );
                        break 'blk;
                    }
                });
            };
            if BUF {
                for j in 0..n {
                    f.call(c_k.add(j*MR), MR);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, m, n, MR);
            } else {
                for j in 0..n {
                    f.call(c_k.add(j*c_cs), m);
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! def_ukernel_avx_2 {
    ($k_unit:tt, $step:ident, $acc:ident, $store:ident, $mr:tt, $nr:tt, $kl_pf:tt, $pf1_step:tt) => {
        pub(crate) unsafe fn ukernel_bbc<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            a_pft1_offset: usize, _n: usize,
            f: F,
        ) {
            const MR: usize = $mr * VS;
            let k_l0 = k % $kl_pf;
            let k_l = if k_l0 == 0 {$kl_pf/$k_unit} else {k_l0/$k_unit};
            let k_i = (k - k_l*$k_unit) / (4*$k_unit);
            let mut c_k = c;

            let mut dim_arr = [c_cs*TC_SIZE, k_i, k_l, a_pft1_offset];
            let mut c_buf = [ZERO; MR*$nr];
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if BUF {
                glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, MR, $nr, MR);
                dim_arr[0] = MR*TC_SIZE;
                c_k = c_buf.as_mut_ptr();
            }
            asm!(
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
                prefetch_0!(256, "{bx}"),
                $step!($nr, B, 0),

                "movq $64*4, {x4}",
                // divisiblity by 4
                "testq $3, {x0}",
                "cmovz {x1},{x4}",

                $step!($nr, B, 1),

                "prefetcht1 ({x2})",

                "subq $64*3, {x2}",
                "addq {x4}, {x2}",

                $step!($nr, B, 2),

                "prefetcht1 ({x5})",
                "addq $16, {x5}",

                "testq $63, {x0}",
                "cmovz {cx},{x2}",

                $step!($nr, B, 3),

                inc_a_k_unroll!($mr, 4),
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
                $step!($nr, B, 0),
                inc_a_k_unroll!($mr, 1),
                inc_b_k_unroll!(B, $nr, 1),
                "add {x1}, {x2}", "dec {x0}", "jne 4b",

                "5:",
                "mov ({dim_arrx}),{x0}",
                "lea ({x0}, {x0}, 2), {x3}",
                "lea ({cx}, {x3},), {x1}",
                "lea ({x1}, {x3},), {x2}",

                alpha_scale!($mr, $nr),
                "cmpw $0, ({beta_st})",
                "je 6f",

                "cmpw $1, ({beta_st})",
                "je 15f",

                load_beta!(),
                glar_base::cum_seq!($acc,$nr,C,2),
                "jmp 6f",

                "15:",
                glar_base::cum_seq!($acc,$nr,C,1),

                "6:",
                glar_base::cum_seq!($store,$nr,C),

                ax = inout(reg) a => _,
                bx = inout(reg) b => _,
                cx = inout(reg) c_k => _,
                dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                alphax = inout(reg) alpha => _,
                betax = inout(reg) beta => _,
                beta_st = in(reg) &beta_st,
                x0 = out(reg) _,
                x1 = out(reg)_,
                x2 = out(reg) _,
                x3 = out(reg) _,
                x4 = out(reg) _,
                x5 = out(reg) _,
                out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
                out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
                out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
                out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
                options(att_syntax)
            );
            if BUF {
                for j in 0..$nr {
                    f.call(c_k.add(j*MR), MR);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, MR, $nr, MR);
            } else {
                for j in 0..$nr {
                    f.call(c_k.add(j*c_cs), MR);
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! def_ukernel_avx512_2 {
    ($k_unit:tt, $step:ident, $acc:ident, $store:ident, $mr:tt, $nr:tt, $kl_pf:tt, $pf1_step:tt) => {
        pub(crate) unsafe fn ukernel_bbc<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            a_pft1_offset: usize, _n: usize,
            f: F,
        ) {
            const MR: usize = $mr * VS;
            let k_l0 = k % $kl_pf;
            let k_l = if k_l0 == 0 {$kl_pf/$k_unit} else {k_l0/$k_unit};
            let k_i = (k - k_l*$k_unit) / (4*$k_unit);
            let mut c_k = c;

            let mut dim_arr = [c_cs*TC_SIZE, k_i, k_l, a_pft1_offset];
            let mut c_buf = [ZERO; MR * $nr];
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            if BUF {
                glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, MR, $nr, MR);
                dim_arr[0] = MR*TC_SIZE;
                c_k = c_buf.as_mut_ptr();
            }
            asm!(
                vzero_kernel!(),
                init_ab_2!(B),
                "test {x0},{x0}", "je 3f",

                "mov {cx}, {x2}",
                "mov {ax}, {x5}",
                "mov 24({dim_arrx}),{x1}",
                "add {x1}, {x5}",
                "mov ({dim_arrx}),{x1}",

                "2:", // KITER
                $step!($nr, B),

                "movq $64*4, {x4}",
                // divisiblity by 4
                "testq $3, {x0}",
                "cmovz {x1},{x4}",

                $step!($nr, B),

                "prefetcht1 ({x2})",

                "subq $64*3, {x2}",
                "addq {x4}, {x2}",

                $step!($nr, B),

                "prefetcht1 ({x5})",
                concat!("addq $", $pf1_step, ", {x5}"),

                "testq $63, {x0}",
                "cmovz {cx},{x2}",

                $step!($nr, B),

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
                $step!($nr, B),
                "add {x1}, {x2}", "dec {x0}", "jne 4b", // KLEFT

                "5:", // POSTACCUM
                "mov ({dim_arrx}),{x0}",
                "lea ({x0}, {x0}, 2), {x3}",
                "lea ({cx}, {x3},), {x1}",
                "lea ({x1}, {x3},), {x2}",

                alpha_scale!($mr, $nr),

                "cmpw $0, ({beta_st})",
                "je 6f",

                "cmpw $1, ({beta_st})",
                "je 15f",

                load_beta!(),
                glar_base::cum_seq!($acc,$nr,C,2),
                "jmp 6f",

                "15:",
                glar_base::cum_seq!($acc,$nr,C,1),

                "6:",
                glar_base::cum_seq!($store,$nr,C),

                ax = inout(reg) a => _,
                bx = inout(reg) b => _,
                cx = inout(reg) c_k => _,
                dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                alphax = inout(reg) alpha => _,
                betax = inout(reg) beta => _,
                beta_st = in(reg) &beta_st,
                x0 = out(reg) _,
                x1 = out(reg)_,
                x2 = out(reg) _,
                x3 = out(reg) _,
                x4 = out(reg) _,
                x5 = out(reg) _,
                out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
                out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
                out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
                out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
                out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
                out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
                out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
                out("zmm28") _, out("zmm29") _, out("zmm30") _, out("zmm31") _,
                out("k1") _,
                options(att_syntax)
            );
            if BUF {
                for j in 0..$nr {
                    f.call(c_k.add(j*MR), MR);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, MR, $nr, MR);
            } else {
                for j in 0..$nr {
                    f.call(c_k.add(j*c_cs), MR);
                }
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
        $n0:tt, $n1:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            m: usize, n: usize,
            f: F,
        ) {
            let beta_st = if *beta == ZERO_SCALAR {
                0i32
            } else if *beta == ONE_SCALAR {
                1i32
            } else {
                2i32
            };
            const MR: usize = $mr * VS;
            let mut dim_arr = [d_arr[0]*8, d_arr[1]*8, c_cs*TC_SIZE, k / ($k_unit*4), (k % ($k_unit*4)) / $k_unit, beta_st as usize];
            let mut ptr_arr = [alpha, beta];
            let mut cf = c;
            let mut c_buf = [ZERO;MR*$nr];
            if BUF {
                glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, m, n, MR);
                dim_arr[2] = MR*TC_SIZE;
                cf = c_buf.as_mut_ptr();
            }
            let _ = 'blk: {
                seq!(ni in $n0..$n1 {
                    if glar_base::n_cond!($n0, ni, n) {
                        glar_base::prefetch_c_sse!($mr,ni,c,c_cs);
                        asm!(
                            vzero_kernel!(),

                            init_ab!($b_layout),
                            "test {x0}, {x0}", "je 3f", // CONSIDKLEFT

                            "2:", // KITER
                            glar_base::prefetch_b!($b_layout),
                            $step_macro!(ni, $b_layout, 0),
                            $step_macro!(ni, $b_layout, 1),
                            $step_macro!(ni, $b_layout, 2),
                            $step_macro!(ni, $b_layout, 3),

                            inc_a_k_unroll!($mr, 4),
                            inc_b_k_unroll!($b_layout, ni, 4),

                            "dec {x0}", "jne 2b", // KITER

                            "3:", // CONSIDKLEFT
                            "mov 16({dim_arrx}), {x0}",
                            "test {x0},{x0}", "je 5f", // POSTACCUM

                            "4:", // KLEFT
                            $step_macro!(ni, $b_layout, 0),
                            inc_a_k_unroll!($mr, 1),
                            inc_b_k_unroll!($b_layout, ni, 1),

                            "dec {x0}", "jne 4b", // KLEFT

                            "5:", // POSTACCUM
                            c_load!(ni),

                            alpha_scale!($mr, ni),

                            "cmpw $0, 20({dim_arrx})",
                            "je 6f",

                            "cmpw $1, 20({dim_arrx})",
                            "je 15f",

                            load_beta!(),
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,2),
                            "jmp 6f",

                            "15:",
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,1),

                            "6:",
                            glar_base::cum_seq!($store_macro,ni,$is_partial),

                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) cf => _,
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
            if BUF {
                for j in 0..n {
                    f.call(cf.add(j*MR), MR);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, m, n, MR);
            } else {
                for j in 0..n {
                    f.call(cf.add(j*c_cs), m);
                }
            }
        }
    };
}

#[macro_export]
macro_rules! def_ukernel_neon {
    (
        $step_macro:tt,
        $acc_macro:tt,
        $store_macro:tt,
        $mr:tt, $nr:tt,
        $n0:tt, $n1:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon")]
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            m: usize, n: usize,
            f: F,
        ) {
            const MR: usize = $mr * VS;
            use core::mem::size_of;
            let mut dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 4, k % 4];
            let mut cf = c;
            let mut c_buf = [ZERO;MR*$nr];
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
            let _ = 'blk: {
                seq!(ni in $n0..$n1 {
                    if glar_base::n_cond!($n0, ni, n) {
                        if BUF {
                            glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, m, ni, MR);
                            dim_arr[2] = MR*TC_SIZE;
                            cf = c_buf.as_mut_ptr();
                        }
                        asm!(
                            prefetch_c!(),
                            vzero_kernel!(),

                            asm_init_ab!($b_layout),

                            // 3 -> CONSIDKLEFT
                            "BEQ 3f",

                            // 2 -> KITER
                            "2:",
                            prefetch_0!(128, "{bx}"),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),

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
                            $step_macro!(ni, $b_layout),

                            "sub {x0}, {x0}, #1",

                            // 4 -> KLEFT
                            "cmp {x0}, 0",
                            "BNE 4b",

                            // 5 -> POSTACCUM
                            "5:",
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),

                            "cmp {beta_st:w}, #0", "\n",
                            "BEQ 6f",

                            load_beta!(),

                            glar_base::cum_seq!($acc_macro,ni,$is_partial,2),

                            // 6 -> BETAZERO
                            "6:",
                            glar_base::cum_seq!($store_macro,ni,$is_partial),

                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) cf => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            betax = inout(reg) beta => _,
                            alpha_st = in(reg) alpha_st,
                            beta_st = in(reg) beta_st,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            x4 = out(reg) _,
                            x5 = out(reg) _,
                            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                            out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                            out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                        );
                        break 'blk;
                    }
                });
            };
            if BUF {
                for j in 0..n {
                    f.call(cf.add(j*MR), MR);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, m, n, MR);
            } else {
                for j in 0..n {
                    f.call(cf.add(j*c_cs), m);
                }
            }
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
        $n0:tt, $n1:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon")]
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            m: usize, n: usize,
            f: F,
        ) {
            alt_arr!(alt);
            const MR: usize = $mr * VS;
            use core::mem::size_of;
            let mut dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 4, k % 4];
            let mut cf = c;
            let mut c_buf = [ZERO;MR*$nr];
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
            let _ = 'blk: {
                seq!(ni in $n0..$n1 {
                    if glar_base::n_cond!($n0, ni, n) {
                        if BUF {
                            glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, m, ni, MR);
                            dim_arr[2] = MR*TC_SIZE;
                            cf = c_buf.as_mut_ptr();
                        }
                        asm!(
                            prefetch_c!(),
                            vzero_kernel!(),

                            asm_init_ab!($b_layout),

                            // 3 -> CONSIDKLEFT
                            "BEQ 3f",

                            // 2 -> KITER
                            "2:",
                            prefetch_0!(128, "{bx}"),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),

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
                            $step_macro!(ni, $b_layout),

                            "sub {x0}, {x0}, #1",

                            // 4 -> KLEFT
                            "cmp {x0}, 0",
                            "BNE 4b",

                            // 5 -> POSTACCUM
                            "5:",
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),

                            "cmp {beta_st:w}, #0", "\n",
                            "BEQ 6f",

                            load_beta!(),

                            glar_base::cum_seq!($acc_macro,ni,$is_partial,2),

                            // 6 -> BETAZERO
                            "6:",
                            glar_base::cum_seq!($store_macro,ni,$is_partial),

                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) cf => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            betax = inout(reg) beta => _,
                            alpha_st = in(reg) alpha_st,
                            beta_st = in(reg) beta_st,
                            altx = inout(reg) alt.as_ptr() => _,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            x4 = out(reg) _,
                            x5 = out(reg) _,
                            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                            out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                            out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                        );
                        break 'blk;
                    }
                });
            };
            if BUF {
                for j in 0..n {
                    f.call(cf.add(j*MR), MR);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, m, n, MR);
            } else {
                for j in 0..n {
                    f.call(cf.add(j*c_cs), m);
                }
            }
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
        $n0:tt, $n1:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon,fp16")]
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            m: usize, n: usize,
            f: F,
        ) {
            const MR: usize = $mr * VS;
            use core::mem::size_of;
            let mut dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 4, k % 4];
            let mut cf = c;
            let mut c_buf = [ZERO;MR*$nr];
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
            let _ = 'blk: {
                seq!(ni in $n0..$n1 {
                    if BUF {
                        glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, m, ni, MR);
                        dim_arr[2] = MR*TC_SIZE;
                        cf = c_buf.as_mut_ptr();
                    }
                    if glar_base::n_cond!($n0, ni, n) {
                        asm!(
                            prefetch_c!(),
                            vzero_kernel!(),

                            asm_init_ab!($b_layout),

                            // 3 -> CONSIDKLEFT
                            "BEQ 3f",

                            // 2 -> KITER
                            "2:",
                            prefetch_0!(128, "{bx}"),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),

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
                            $step_macro!(ni, $b_layout),

                            "sub {x0}, {x0}, #1",

                            // 4 -> KLEFT
                            "cmp {x0}, 0",
                            "BNE 4b",

                            // 5 -> POSTACCUM
                            "5:",
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),

                            "cmp {beta_st:w}, #0", "\n",
                            "BEQ 6f",

                            load_beta!(),

                            glar_base::cum_seq!($acc_macro,ni,$is_partial,2),

                            // 6 -> BETAZERO
                            "6:",
                            glar_base::cum_seq!($store_macro,ni,$is_partial),

                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) cf => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            betax = inout(reg) beta => _,
                            alpha_st = in(reg) alpha_st,
                            beta_st = in(reg) beta_st,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            x4 = out(reg) _,
                            x5 = out(reg) _,
                            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                            out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                            out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                        );
                        break 'blk;
                    }
                });
            };
            if BUF {
                for j in 0..n {
                    f.call(cf.add(j*MR), MR);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, m, n, MR);
            } else {
                for j in 0..n {
                    f.call(cf.add(j*c_cs), m);
                }
            }
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
        $n0:tt, $n1:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon,i8mm")]
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            m: usize, n: usize,
            f: F,
        ) {
            const MR: usize = $mr * VS;
            use core::mem::size_of;
            let mut dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 32, (k % 32) / 8];
            let mut cf = c;
            let mut c_buf = [ZERO;MR*$nr];
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
            let _ = 'blk: {
                seq!(ni in $n0..$n1 {
                    if glar_base::n_cond!($n0, ni, n) {
                        if BUF {
                            glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, m, ni, MR);
                            dim_arr[2] = MR*TC_SIZE;
                            cf = c_buf.as_mut_ptr();
                        }
                        asm!(
                            prefetch_c!(),
                            vzero_kernel!(),

                            asm_init_ab!($b_layout),

                            // 3 -> CONSIDKLEFT
                            "BEQ 3f",

                            // 2 -> KITER
                            "2:",
                            prefetch_0!(128, "{bx}"),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),

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
                            $step_macro!(ni, $b_layout),

                            "sub {x0}, {x0}, #1",

                            // 4 -> KLEFT
                            "cmp {x0}, 0",
                            "BNE 4b",

                            // 5 -> POSTACCUM
                            "5:",
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),

                            "cmp {beta_st:w}, #0", "\n",
                            "BEQ 6f",

                            "cmp {beta_st:w}, #1", "\n",
                            "BEQ 9f",

                            load_beta!(),
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,2),
                            "B 6f",

                            "9:",
                            // 9 -> BETAONE
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,1),

                            // 6 -> BETAZERO
                            "6:",
                            glar_base::cum_seq!($store_macro,ni,$is_partial),

                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) cf => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            betax = inout(reg) beta => _,
                            alpha_st = in(reg) alpha_st,
                            beta_st = in(reg) beta_st,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            x4 = out(reg) _,
                            x5 = out(reg) _,
                            x6 = out(reg) _,
                            x7 = out(reg) _,
                            x8 = out(reg) _,
                            x9 = out(reg) _,
                            x10 = out(reg) _,
                            x11 = out(reg) _,
                            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                            out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                            out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                        );
                        break 'blk;
                    }
                });
            };
            if BUF {
                for j in 0..n {
                    f.call(cf.add(j*MR), MR);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, m, n, MR);
            } else {
                for j in 0..n {
                    f.call(cf.add(j*c_cs), m);
                }
            }
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
        $n0:tt, $n1:tt,
        $b_layout:tt,
        $is_partial:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon,sve")]
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TA, beta: *const TB,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            m: usize, n: usize,
            f: F,
        ) {
            use core::mem::size_of;
            let vs = sve_vs();
            let m_left = if m % vs == 0 {vs} else {m%vs};
            let inc_a = vs * $mr * size_of::<TA>();
            let mr = $mr * vs;
            let mut dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 4, k % 4];
            let mut cf = c;
            let mut c_buf = [ZERO;(256/size_of::<TC>())*$mr*$nr];
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
            let _ = 'blk: {
                seq!(ni in $n0..$n1 {
                    // usingy dynamic n leads to bug due sve on windows
                    // see: https://github.com/llvm/llvm-project/issues/80009
                    if BUF {
                        glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, m, ni, mr);
                        dim_arr[2] = mr*TC_SIZE;
                        cf = c_buf.as_mut_ptr();
                    }
                    if glar_base::n_cond!($n0, ni, n) {
                        asm!(
                            "ptrue p0.h",
                            "mov {m_s}, #0",
                            "/* {m_e} */", "\n",
                            prefetch_c!(),
                            vzero_kernel!(),

                            asm_init_ab!($b_layout),

                            // 3 -> CONSIDKLEFT
                            "BEQ 3f",

                            // 2 -> KITER
                            "2:",
                            prefetch_0!(128, "{bx}"),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),

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
                            $step_macro!(ni, $b_layout),

                            "sub {x0}, {x0}, #1",

                            // 4 -> KLEFT
                            "cmp {x0}, 0",
                            "BNE 4b",

                            // 5 -> POSTACCUM
                            "5:",
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),

                            "cmp {beta_st:w}, #0", "\n",
                            "BEQ 6f",

                            load_beta!(),

                            glar_base::cum_seq!($acc_macro,ni,$is_partial),

                            // 6 -> BETAZERO
                            "6:",
                            glar_base::cum_seq!($store_macro,ni,$is_partial),

                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) cf => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            betax = inout(reg) beta => _,
                            incax = in(reg) inc_a as u64,
                            alpha_st = in(reg) alpha_st,
                            beta_st = in(reg) beta_st,
                            m_s = out(reg) _,
                            m_e = inout(reg) m_left as u64 => _,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            x4 = out(reg) _,
                            x5 = out(reg) _,
                            x6 = out(reg) _,
                            x7 = out(reg) _,
                            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                            out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                            out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                            out("p0") _, out("p1") _, out("p2") _, out("p3") _,
                        );
                        break 'blk;
                    }
                });
            };
            if BUF {
                for j in 0..n {
                    f.call(cf.add(j*mr), mr);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, m, n, mr);
            } else {
                for j in 0..n {
                    f.call(cf.add(j*c_cs), m);
                }
            }
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
        $n0:tt, $n1:tt,
        $b_layout:tt,
        $is_partial:tt,
        // $feature_enable:tt,
        $func_name:ident
    ) => {
        #[target_feature(enable="neon,sve,i8mm")]
        pub(crate) unsafe fn $func_name<F: UnaryFnC, const BUF: bool>(
            a: *const TA, b: *const TB, c: *mut TC,
            alpha: *const TS, beta: *const TS,
            k: usize,
            d_arr: [usize; 3], c_cs: usize,
            m: usize, n: usize,
            f: F,
        ) {
            use core::mem::size_of;
            let vs = sve_vs();
            let m_left = if m % vs == 0 {vs} else {m%vs};
            let inc_a = $mr * vs * size_of::<TA>() * 8;
            let mr = $mr * vs;
            let mut dim_arr = [d_arr[0]*size_of::<TB>(), d_arr[1]*size_of::<TB>(), c_cs*TC_SIZE, k / 32, (k % 32) / 8];
            let mut cf = c;
            let mut c_buf = [ZERO;(256/size_of::<TC>())*$mr*$nr];
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
            let _ = 'blk: {
                seq!(ni in $n0..$n1 {
                    // usingy dynamic n leads to bug due to llvm bug sve on windows
                    // see: https://github.com/llvm/llvm-project/issues/80009
                    if BUF {
                        glar_base::load_buf(c, d_arr[2], c_cs, &mut c_buf, m, ni, mr);
                        dim_arr[2] = mr*TC_SIZE;
                        cf = c_buf.as_mut_ptr();
                    }
                    if glar_base::n_cond!($n0, ni, n) {
                        asm!(
                            "ptrue p0.h",
                            "/* {m_s} */", "\n",
                            "/* {m_e} */", "\n",
                            prefetch_c!(),
                            vzero_kernel!(),

                            asm_init_ab!($b_layout),

                            // 3 -> CONSIDKLEFT
                            "BEQ 3f",

                            // 2 -> KITER
                            "2:",
                            prefetch_0!(128, "{bx}"),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),
                            $step_macro!(ni, $b_layout),

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
                            $step_macro!(ni, $b_layout),

                            "sub {x0}, {x0}, #1",

                            // 4 -> KLEFT
                            "cmp {x0}, 0",
                            "BNE 4b",

                            // 5 -> POSTACCUM
                            "5:",
                            asm_c_load!(ni),
                            // scale by alpha
                            asm_alpha_scale!($mr, ni),

                            "cmp {beta_st:w}, #0", "\n",
                            "BEQ 6f",

                            "cmp {beta_st:w}, #1", "\n",
                            "BEQ 9f",

                            load_beta!(),
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,2),
                            "B 6f",

                            "9:",
                            // 9 -> BETAONE
                            glar_base::cum_seq!($acc_macro,ni,$is_partial,1),

                            // 6 -> BETAZERO
                            "6:",
                            glar_base::cum_seq!($store_macro,ni,$is_partial),
                            ax = inout(reg) a => _,
                            bx = inout(reg) b => _,
                            cx = inout(reg) cf => _,
                            dim_arrx = inout(reg) dim_arr.as_ptr() => _,
                            alphax = inout(reg) alpha => _,
                            betax = inout(reg) beta => _,
                            incax = in(reg) inc_a as u64,
                            alpha_st = in(reg) alpha_st,
                            beta_st = in(reg) beta_st,
                            m_s = in(reg) 0 as u64,
                            m_e = in(reg) m_left as u64,
                            x0 = out(reg) _,
                            x1 = out(reg) _,
                            x2 = out(reg) _,
                            x3 = out(reg) _,
                            x4 = out(reg) _,
                            x5 = out(reg) _,
                            x6 = out(reg) _,
                            x7 = out(reg) _,
                            x8 = out(reg) _,
                            x9 = out(reg) _,
                            x10 = out(reg) _,
                            x11 = out(reg) _,
                            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                            out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                            out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                            out("p0") _, out("p1") _, out("p2") _, out("p3") _,
                        );
                        break 'blk;
                    }
                });
            };
            if BUF {
                for j in 0..n {
                    f.call(cf.add(j*mr), mr);
                }
                glar_base::store_buf(c, d_arr[2], c_cs, &c_buf, m, n, mr);
            } else {
                for j in 0..n {
                    f.call(cf.add(j*c_cs), m);
                }
            }
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
