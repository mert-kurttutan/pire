use std::sync::{Barrier, Mutex, MutexGuard, RwLock, RwLockReadGuard};
// Consider Once Cell
use once_cell::sync::Lazy;

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
        if let Some(value) = std::option_env!($name) {
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
        self.ap_pool_size * std::mem::size_of::<TA>() * self.ap_pool_multiplicity
            + self.bp_pool_size * std::mem::size_of::<TB>() * self.bp_pool_multiplicity
            + 2 * AB_ALIGN
    }

    pub fn ap_size_b<TA>(&self) -> usize {
        self.ap_pool_size * std::mem::size_of::<TA>()
    }

    pub fn bp_size_b<TB>(&self) -> usize {
        self.bp_pool_size * std::mem::size_of::<TB>()
    }

    pub fn ap_size_t_b<TA>(&self) -> usize {
        self.ap_pool_size * std::mem::size_of::<TA>() * self.ap_pool_multiplicity
    }

    pub fn bp_size_t_b<TB>(&self) -> usize {
        self.bp_pool_size * std::mem::size_of::<TB>() * self.bp_pool_multiplicity
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
        let ap_pool_size_b = ap_pool_size * std::mem::size_of::<TA>();
        let a_alignment = std::mem::align_of::<TA>();
        assert_eq!(ap_pool_size_b % a_alignment, 0);
        let bp_pool_size = self.bp_pool_size;
        let bp_pool_size_b = bp_pool_size * std::mem::size_of::<TB>();
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
        let ap_pool_size = hw_config.get_ap_pool_size(par.ic_par) + CACHELINE_PAD / std::mem::size_of::<AP>();
        (ap_pool_size, ap_pool_multiplicity)
    } else {
        (0, 1)
    };
    let (bp_pool_size, bp_pool_multiplicity) = if b_need_pool {
        let bp_pool_multiplicity = par.jc_par;
        let bp_pool_size = hw_config.get_bp_pool_size(par.jc_par) + CACHELINE_PAD / std::mem::size_of::<BP>();
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
        let ap_pool_size = hw_config.get_ap_pool_size(par.ic_par) + CACHELINE_PAD / std::mem::size_of::<AP>();
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
    let ap_pool_size = hw_config.get_ap_pool_size2() + CACHELINE_PAD / std::mem::size_of::<AP>();
    let ap_pool_multiplicity = par.num_threads;
    let m = hw_config.mr();
    let n = hw_config.get_nc_eff(par.jc_par);
    let k = hw_config.get_kc_eff();
    if b_need_pool {
        let bp_pool_multiplicity = par.jc_par;
        let bp_pool_size = hw_config.get_bp_pool_size(par.jc_par) + CACHELINE_PAD / std::mem::size_of::<BP>();
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

pub fn ap_size<T>(m: usize, k: usize) -> usize {
    let vs = 256 / std::mem::size_of::<T>();
    let m_max = (m + vs - 1) / vs * vs;
    m_max * k + AB_ALIGN / std::mem::size_of::<T>()
}

pub fn bp_size<T>(n: usize, k: usize) -> usize {
    n * k + AB_ALIGN / std::mem::size_of::<T>()
}

pub fn ap_size_int<T, P>(m: usize, k: usize) -> usize {
    let vs = 64 / std::mem::size_of::<T>();
    let c_r = (std::mem::size_of::<P>() / std::mem::size_of::<T>()) * 2;
    let k_r = (k + c_r - 1) / c_r * c_r;
    let m_max = (m + vs - 1) / vs * vs;
    m_max * k_r + AB_ALIGN / std::mem::size_of::<T>()
}

pub fn bp_size_int<T, P>(n: usize, k: usize) -> usize {
    let c_r = (std::mem::size_of::<P>() / std::mem::size_of::<T>()) * 2;
    let k_r = (k + c_r - 1) / c_r * c_r;
    n * k_r + AB_ALIGN / std::mem::size_of::<T>()
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
    pub fn dst_w(&self, idx: usize, kc: usize) -> RangeLockWriteGuard<'a, 'a, U> {
        self.dst.write(idx, kc).unwrap()
    }
    pub fn dst_r(&self) -> RangeLockReadGuard<'a, 'a, U> {
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

    pub fn dst_w(&self, idx: usize, kc: usize) -> RangeLockWriteGuard<'a, 'a, Y> {
        self.dst.write(idx, kc).unwrap()
    }

    pub fn get_mc(&self) -> usize {
        self.dst.get_mc()
    }

    pub fn dst_r(&self) -> RangeLockReadGuard<'a, 'a, Y> {
        self.dst.read().unwrap()
    }
}

unsafe impl<X, Y> Send for PackedMatrixMixed<'_, X, Y> {}

// must be multiple largest vector size that we support
// Now, it avx512 -> 64 bytes
pub const AB_ALIGN: usize = 1024;

pub trait GemmCache {
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;
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

    pub fn dst_w(&self, idx: usize, kc: usize) -> RangeLockWriteGuard<'a, 'a, X> {
        match self {
            Self::StridedMatrix(x) => x.dst.write(idx, kc).unwrap(),
            _ => {
                panic!("Only StridedMatrix has write guard");
            }
        }
    }

    pub fn dst_r(&self) -> RangeLockReadGuard<'a, 'a, X> {
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

    pub fn dst_w(&self, idx: usize, kc: usize) -> RangeLockWriteGuard<'a, 'a, Y> {
        match self {
            Self::StridedMatrix(x) => x.dst.write(idx, kc).unwrap(),
            Self::PackedMatrix(x) => x.dst.write(idx, kc).unwrap(),
        }
    }
    pub fn dst_r(&self) -> RangeLockReadGuard<'a, 'a, Y> {
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
                    let kc_len_eff = hw_cfg.round_up(kc_len);
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
                    let kc_len_eff = hw_cfg.round_up(kc_len);
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
            let mut a_dst = a.dst_w(0, kc);
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
                        let kc_len_ro = hw_cfg.round_up(kc_len);
                        let mc_len_x = (mc_len - mc_offset).min(mc_par);
                        let mc_i = mc_i + mc_offset;
                        let rs = x_i.rs();
                        let cs = x_i.cs();
                        let src_ptr = x_i.src().add(mc_i*rs + kc_i*cs);
                        let mut dst = x_i.dst_w(t_cfg.i_load_p_idx, kc_len_ro);
                        let dst_ref = dst.get();
                        let dst_ptr = dst_ref.as_mut_ptr();
                        hw_cfg.packa_fn(src_ptr, dst_ptr, mc_len_x, kc_len, rs, cs);
                    }
                    t_cfg.wait_packa();
                    PtrData::RefData(x_i.dst_r())
                }
                $packa_ty::PackedMatrix(x_i) => {
                    let vs = hw_cfg.vs;
                    let m_ro = (x_i.m() + vs - 1) / vs * vs;
                    let kc_len_ro = hw_cfg.round_up(kc_len);
                    let res = is_mixed!(
                        $include_flag,
                        {
                            let mc_par = x_i.get_mc();
                            let mc_offset = mc_par * t_cfg.i_load_p_idx;
                            let mc_len_ro = (mc_len + vs - 1) / vs * vs;
                            if mc_len_ro > mc_offset {
                                let mc_len_ro_x = (mc_len_ro - mc_offset).min(mc_par);
                                let mc_i = mc_i + mc_offset;
                                let src_ptr = x_i.src().add(mc_i*kc_len_ro + kc_i*m_ro);
                                let mut dst = x_i.dst_w(t_cfg.i_load_p_idx, kc_len_ro);
                                let dst_ref = dst.get();
                                let dst_ptr = dst_ref.as_mut_ptr();
                                hw_cfg.cvt_mixed(src_ptr, dst_ptr, mc_len_ro_x*kc_len_ro);
                            }
                            t_cfg.wait_packa();
                            PtrData::RefData(x_i.dst_r())
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
                        let kc_len_ro = hw_cfg.round_up(kc_len);
                        let nc_len_x = (nc_len - nc_offset).min(nc_par);
                        let nc_i = nc_i + nc_offset;
                        let rs = x_i.rs();
                        let cs = x_i.cs();
                        let src_ptr = x_i.src().add(kc_i*rs + nc_i*cs);
                        let mut dst = x_i.dst_w(t_cfg.j_load_p_idx, kc_len_ro);
                        let dst_ref = dst.get();
                        let dst_ptr = dst_ref.as_mut_ptr();
                        hw_cfg.packb_fn(src_ptr, dst_ptr, nc_len_x, kc_len, rs, cs);
                    }
                    t_cfg.wait_packb();
                    PtrData::RefData(x_i.dst_r())
                }
                $packb_ty::PackedMatrix(x_i) => {
                    let kc_len_ro = hw_cfg.round_up(kc_len);
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
                                let mut dst = x_i.dst_w(t_cfg.j_load_p_idx, kc_len_ro);
                                let dst_ref = dst.get();
                                let dst_ptr = dst_ref.as_mut_ptr();
                                hw_cfg.cvt_mixed(src_ptr, dst_ptr, nc_len_x*kc_len_ro);
                            }
                            t_cfg.wait_packb();
                            PtrData::RefData(x_i.dst_r())
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
macro_rules! def_kernel_bb_pf1 {
    (
        $ta:ty, $tb:ty, $tc:ty, $t_as:ty, $t_bs:ty,
        $MR:tt, $NR:tt, $pf1_0:tt, $pf_step:tt, $($mr_left:tt),*
    ) => {
        paste! {
            pub unsafe fn kernel_bb<F: UnaryFnC, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const $t_as,
                beta: *const $t_bs,
                c: *mut $tc, c_rs: usize, c_cs: usize,
                ap: *const $ta, bp: *const $tb,
                f: F,
            ) {
                const MR: usize = $MR * VS;
                const NR: usize = $NR;
                let m_rounded = m / MR * MR;
                let n_rounded = n / NR * NR;
                let m_left = m % MR;
                let n_left = n % NR;

                let d_arr = [0, 0, c_rs, c_cs];

                let mut m_i = 0;
                while m_i < m_rounded {
                    let c_cur0 = c.add(m_i * c_rs);
                    let ap_cur = ap.add(m_i * k);
                    let mut a_pft1_offset = $pf1_0 * k;
                    let mut n_i = 0;
                    while n_i < n_rounded {
                        let bp_cur = bp.add(n_i * k);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        ukernel_bb::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, a_pft1_offset, f);
                        n_i += NR;
                        a_pft1_offset += $pf_step * k;
                    }
                    // let a_pft1_offset = ($MR+(n_iter0-n_iter)*2)*4*k;
                    if n_left != 0 {
                        let bp_cur = bp.add(n_i * k);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        ukernel_n_bb::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, MR, n_left, f);
                    }
                    m_i += MR;
                }


                $(
                    if (m_left+VS-1) / VS == $mr_left {
                        let c_cur0 = c.add(m_i * c_rs);
                        let ap_cur = ap.add(m_i * k);
                        let mut n_i = 0;
                        while n_i < n_rounded {
                            let bp_cur = bp.add(n_i * k);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left _bb_partial>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, m_left, f);
                            n_i += NR;
                        }
                        if n_left !=0 {
                            let bp_cur = bp.add(n_i * k);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left x n_bb_partial>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, m_left, n_left, f);
                        }
                    }
                )*
            }
        }
    };
}

#[macro_export]
macro_rules! def_kernel_bb_v0 {
    (
        $ta:ty, $tb:ty, $tc:ty, $t_as:ty, $t_bs:ty,
        $MR:tt, $NR:tt, $($mr_left:tt),*
    ) => {
        paste! {
            pub unsafe fn kernel_bb<F: UnaryFnC, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const $t_as,
                beta: *const $t_bs,
                c: *mut $tc, c_rs: usize, c_cs: usize,
                ap: *const $ta, bp: *const $tb,
                f: F,
            ) {
                const MR: usize = $MR * VS;
                const NR: usize = $NR;
                let m_rounded = m / MR * MR;
                let n_rounded = n / NR * NR;
                let m_left = m % MR;
                let n_left = n % NR;

                let d_arr = [0, 0, c_rs, c_cs];

                let mut m_i = 0;
                while m_i < m_rounded {
                    let c_cur0 = c.add(m_i * c_rs);
                    let ap_cur = ap.add(m_i * k);
                    let mut n_i = 0;
                    while n_i < n_rounded {
                        let bp_cur = bp.add(n_i * k);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        ukernel_bb::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, MR, f);
                        n_i += NR;
                    }
                    if n_left != 0 {
                        let bp_cur = bp.add(n_i * k);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        ukernel_n_bb::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, MR, n_left, f);
                    }
                    m_i += MR;
                }

                $(
                    if (m_left+VS-1) / VS == $mr_left {
                        let c_cur0 = c.add(m_i * c_rs);
                        let ap_cur = ap.add(m_i * k);
                        let mut n_i = 0;
                        while n_i < n_rounded {
                            let bp_cur = bp.add(n_i * k);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left _bb_partial>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, m_left, f);
                            n_i += NR;
                        }
                        if n_left !=0 {
                            let bp_cur = bp.add(n_i * k);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left x n_bb_partial>]::<_, STRIDED>(ap_cur, bp_cur, c_cur1, alpha, beta, k, d_arr, m_left, n_left, f);
                        }
                    }
                )*
            }
        }
    };
}

#[macro_export]
macro_rules! def_kernel_sb_pf1 {
    (
        $ta:ty, $tb:ty, $tc:ty, $t_as:ty, $t_bs:ty,
        $pack_fn:tt,
        $RS:tt,
        $MR:tt, $NR:tt, $pf1_0:tt, $pf_step:tt, $($mr_left:tt),*
    ) => {
        paste! {
            pub unsafe fn kernel_sb_v0<F: UnaryFnC, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const $t_as, beta: *const $t_bs,
                a: *const $ta, a_rs: usize, a_cs: usize,
                bp: *const $tb,
                c: *mut $tc, c_rs: usize, c_cs: usize,
                ap: *mut $ta,
                f: F,
            ) {
                let k_eff = (k+$RS-1) / $RS * $RS;
                const MR: usize = $MR * VS;
                const NR: usize = $NR;
                let m_rounded = m / MR * MR;
                let n_rounded = n / NR * NR;
                let m_left = m % MR;
                let n_left = n % NR;

                let d_arr = [0, 0, c_rs, c_cs];

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
                        ukernel_bb::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, a_pft1_offset, f);
                        n_i += NR;
                    }
                    if n_left != 0 {
                        let bp_cur = bp.add(n_i * k_eff);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        ukernel_n_bb::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, MR, n_left, f);
                    }
                    m_i += MR;
                }

                $(
                    if (m_left+VS-1) / VS == $mr_left {
                        let c_cur0 = c.add(m_i * c_rs);
                        let a_cur = a.add(m_i * a_rs);
                        $pack_fn(m_left, k, a_cur, a_rs, a_cs, ap, VS);
                        let mut n_i = 0;
                        while n_i < n_rounded {
                            let bp_cur = bp.add(n_i * k_eff);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left _bb_partial>]::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, m_left, f);
                            n_i += NR;
                        }
                        if n_left != 0 {
                            let bp_cur = bp.add(n_i * k_eff);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left xn_bb_partial>]::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, m_left, n_left, f);
                        }
                        return;
                    }
                )*
            }
        }
    };
}

#[macro_export]
macro_rules! def_kernel_sb_v0 {
    (
        $ta:ty, $tb:ty, $tc:ty, $t_as:ty, $t_bs:ty,
        $pack_fn:tt,
        $RS:tt,
        $MR:tt, $NR:tt, $($mr_left:tt),*
    ) => {
        paste! {
            pub unsafe fn kernel_sb_v0<F: UnaryFnC, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const $t_as, beta: *const $t_bs,
                a: *const $ta, a_rs: usize, a_cs: usize,
                bp: *const $tb,
                c: *mut $tc, c_rs: usize, c_cs: usize,
                ap: *mut $ta,
                f: F,
            ) {
                let k_eff = (k+$RS-1) / $RS * $RS;
                const MR: usize = $MR * VS;
                const NR: usize = $NR;
                let m_rounded = m / MR * MR;
                let n_rounded = n / NR * NR;
                let m_left = m % MR;
                let n_left = n % NR;

                let d_arr = [0, 0, c_rs, c_cs];

                let mut m_i = 0;
                while m_i < m_rounded {
                    let c_cur0 = c.add(m_i * c_rs);
                    let a_cur = a.add(m_i * a_rs);
                    $pack_fn(MR, k, a_cur, a_rs, a_cs, ap, VS);
                    let mut n_i = 0;
                    while n_i < n_rounded {
                        let bp_cur = bp.add(n_i * k_eff);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        ukernel_bb::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, MR, f);
                        n_i += NR;
                    }
                    if n_left != 0 {
                        let bp_cur = bp.add(n_i * k_eff);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        ukernel_n_bb::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, MR, n_left, f);
                    }
                    m_i += MR;
                }

                $(
                    if (m_left+VS-1) / VS == $mr_left {
                        let c_cur0 = c.add(m_i * c_rs);
                        let a_cur = a.add(m_i * a_rs);
                        $pack_fn(m_left, k, a_cur, a_rs, a_cs, ap, VS);
                        let mut n_i = 0;
                        while n_i < n_rounded {
                            let bp_cur = bp.add(n_i * k_eff);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left _bb_partial>]::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, m_left, f);
                            n_i += NR;
                        }
                        if n_left != 0 {
                            let bp_cur = bp.add(n_i * k_eff);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left xn_bb_partial>]::<_, STRIDED>(ap, bp_cur, c_cur1, alpha, beta, k_eff, d_arr, m_left, n_left, f);
                        }
                        return;
                    }
                )*
            }
        }
    };
}

#[macro_export]
macro_rules! def_kernel_bs {
    (
        $ta:ty, $tb:ty, $tc:ty, $t_as:ty, $t_bs:ty,
        $MR:tt, $NR:tt, $($mr_left:tt),*
    ) => {
        paste! {
            pub unsafe fn kernel_bs_v0<F: UnaryFnC, const STRIDED: bool>(
                m: usize, n: usize, k: usize,
                alpha: *const $t_as, beta: *const $t_bs,
                b: *const $tb, b_rs: usize, b_cs: usize,
                c: *mut $tc, c_rs: usize, c_cs: usize,
                ap: *const $ta,
                f: F,
            ) {
                const MR: usize = $MR * VS;
                const NR: usize = $NR;
                let m_rounded = m / MR * MR;
                let n_rounded = n / NR * NR;
                let m_left = m % MR;
                let n_left = n % NR;

                let d_arr = [b_rs, b_cs, c_rs, c_cs];

                let mut m_i = 0;
                while m_i < m_rounded {
                    let c_cur0 = c.add(m_i * c_rs);
                    let ap_cur = ap.add(m_i * k);
                    let mut n_i = 0;
                    while n_i < n_rounded {
                        let b_cur = b.add(n_i * b_cs);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        ukernel_bs::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, d_arr, MR, f);
                        n_i += NR;
                    }
                    if n_left != 0 {
                        let b_cur = b.add(n_i * b_cs);
                        let c_cur1 = c_cur0.add(n_i * c_cs);
                        ukernel_n_bs::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, d_arr, MR, n_left, f);
                    }
                    m_i += MR;
                }

                $(
                    if (m_left+VS-1) / VS == $mr_left {
                        let c_cur0 = c.add(m_i * c_rs);
                        let ap_cur = ap.add(m_i * k);
                        let mut n_i = 0;
                        while n_i < n_rounded {
                            let b_cur = b.add(n_i * b_cs);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left _bs_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, d_arr, m_left, f);
                            n_i += NR;
                        }
                        if n_left != 0 {
                            let b_cur = b.add(n_i * b_cs);
                            let c_cur1 = c_cur0.add(n_i * c_cs);
                            [<ukernel_$mr_left xn_bs_partial>]::<_, STRIDED>(ap_cur, b_cur, c_cur1, alpha, beta, k, d_arr, m_left, n_left, f);
                        }
                        return;
                    }
                )*
            }
        }
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
