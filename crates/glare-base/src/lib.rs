use std::sync::{
    Arc,Barrier,Mutex,
    RwLock,RwLockReadGuard, MutexGuard
};
use std::convert::Into;
// Consider Once Cell
use once_cell::sync::Lazy;

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

#[derive(Copy,Clone)]
pub struct CpuFeatures {
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512f16: bool,
    pub avx512bf16: bool,
    pub fma: bool,
    pub fma4: bool,
    pub f16c: bool,
}


#[cfg(target_arch = "x86_64")]
pub struct HWConfig {
    pub cpu_ft: CpuFeatures,
    hw_model: HWModel,
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
}

#[cfg(target_arch = "aarch64")]
pub struct HWConfig {
    neon: bool,
}

#[derive(Copy,Clone)]
pub enum HWModel {
    Reference,
    Haswell,
    Zen1,
    Zen2,
}

impl HWModel {
    pub fn get_cache_info(&self) -> (bool, bool, bool) {
        match self {
            HWModel::Reference => (false, false, false),
            HWModel::Haswell => (false, false, true),
            HWModel::Zen1 => (false, false, true),
            HWModel::Zen2 => (false, false, true),
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
        let avx = feature_info.has_avx();
        let fma = feature_info.has_fma();
        let avx2 = extended_feature_info.has_avx2();
        let avx512f16 = extended_feature_info.has_avx512_fp16();
        let avx512bf16 = extended_feature_info.has_avx512_bf16();
        let avx512f = extended_feature_info.has_avx512f();
        let f16c = feature_info.has_f16c();
        let extended_prcoessor_info = cpuid.get_extended_processor_and_feature_identifiers().unwrap();
        let fma4 = extended_prcoessor_info.has_fma4();
        let cpu_ft = CpuFeatures {
            avx,
            avx2,
            avx512f,
            avx512f16,
            avx512bf16,
            fma,
            fma4,
            f16c,
        };
        let hw_model = HWModel::Haswell;
        let (is_l1_shared, is_l2_shared, is_l3_shared) = hw_model.get_cache_info();
        return HWConfig {
            cpu_ft,
            hw_model: HWModel::Reference,
            is_l1_shared,
            is_l2_shared,
            is_l3_shared,
        };
    }
    #[cfg(target_arch = "aarch64")]
    {
        return HWConfig {
            neon: true,
        };
    }
}


pub static RUNTIME_HW_CONFIG: Lazy<HWConfig> = Lazy::new(|| {
    detect_hw_config()
});

pub static CORENUM_NUM_THREADS: Lazy<usize> = Lazy::new(|| {
    let n_core = std::thread::available_parallelism().unwrap().get();
    // CORENUM_NUM_THREADS or the number of logical cores
    let x = std::env::var("CORENUM_NUM_THREADS").unwrap_or(n_core.to_string());
    x.parse::<usize>().unwrap()
});
#[cfg(target_arch = "x86_64")]
pub(crate) mod cpu_features{
    use super::RUNTIME_HW_CONFIG;
    use super::HWModel;

    pub fn hw_model() -> HWModel {
        RUNTIME_HW_CONFIG.hw_model
    }
    
    pub fn is_null_f16() -> bool {
        !RUNTIME_HW_CONFIG.cpu_ft.avx512f16 && !RUNTIME_HW_CONFIG.cpu_ft.avx512f && !RUNTIME_HW_CONFIG.cpu_ft.avx
    }

    pub fn hw_avx512f16() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.avx512f16
    }

    pub fn hw_avx512bf16() -> bool {
        RUNTIME_HW_CONFIG.cpu_ft.avx512f
    }
}
#[cfg(target_arch = "aarch64")]
pub(crate) mod cpu_features{ 
    use super::RUNTIME_HW_CONFIG;
    pub fn hw_neon() -> bool {
        RUNTIME_HW_CONFIG.neon
    }
}
pub use cpu_features::*;

struct PackPool {
    buffer: RwLock<Vec<Mutex<Vec<u8>>>>
}

static PACK_POOL: PackPool = PackPool {
    buffer: RwLock::new(vec![])
};

pub fn acquire<'a>(pool_guard: &'a RwLockReadGuard<'a, Vec<Mutex<Vec<u8>>>>, pack_size: usize) -> Option<MutexGuard<'a, Vec<u8>>> {
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


pub struct CorenumThreadConfig {
    ic_id: usize,
    // pc_id: usize,
    jc_id: usize,
    ir_id: usize,
    jr_id: usize,
    mc_eff: usize,
    nc_eff: usize,
    kc_eff: usize,
    run_packa: bool,
    run_packb: bool,
   par: CorenumPar,
   packa_barrier: Arc<Vec<Arc<Barrier>>>,
   packb_barrier: Arc<Vec<Arc<Barrier>>>,
}

fn get_apbp_barrier(par: &CorenumPar) -> (Arc<Vec<Arc<Barrier>>>, Arc<Vec<Arc<Barrier>>>) {
    let mut packa_barrier = vec![];
    for _ in 0..par.ic_par {
        let barrier = Arc::new(Barrier::new(par.jc_par * par.pc_par * par.ir_par * par.jr_par));
        packa_barrier.push(barrier);
    }

    let mut packb_barrier = vec![];
    for _ in 0..par.jc_par {
        let barrier = Arc::new(Barrier::new(par.ic_par * par.pc_par * par.ir_par * par.jr_par));
        packb_barrier.push(barrier);
    }

    (Arc::new(packa_barrier), Arc::new(packb_barrier))
}


impl<'a> CorenumThreadConfig {
   fn new(par: CorenumPar, packa_barrier: Arc<Vec<Arc<Barrier>>>, packb_barrier: Arc<Vec<Arc<Barrier>>>, t_id: usize, mc_eff: usize, nc_eff: usize, kc_eff: usize) -> Self {
         let ic_id = par.get_ic_id(t_id);
            // let pc_id = par.get_pc_id(t_id);
            let jc_id = par.get_jc_id(t_id);
            let ir_id = par.get_ir_id(t_id);
            let jr_id = par.get_jr_id(t_id);
            let run_packa = jc_id == 0 && ir_id == 0 && jr_id == 0;
            let run_packb = ic_id == 0 && ir_id == 0 && jr_id == 0;
       Self {
            ic_id,
            // pc_id,
            jc_id,
            ir_id,
            jr_id,
            mc_eff,
            nc_eff,
            kc_eff,
            run_packa,
            run_packb,
           par,
           packa_barrier,
           packb_barrier,
       }
   }
   #[inline]
   fn wait_packa(&self) {
    if self.par.jc_par * self.par.pc_par * self.par.ir_par * self.par.jr_par  > 1 {
        self.packa_barrier[self.ic_id].wait();
    }
   }

    #[inline]
    fn wait_packb(&self) {
        if self.par.ic_par * self.par.pc_par * self.par.ir_par * self.par.jr_par > 1 {
            self.packb_barrier[self.jc_id].wait();
        }
    }
}

// once this is read, this cannot be changed for the time being.
#[inline(always)]
pub fn glare_num_threads() -> usize {
    return *CORENUM_NUM_THREADS;
}

#[derive(Copy,Clone)]
pub struct CorenumPar {
   num_threads: usize,
   ic_par: usize,
   pc_par: usize,
   jc_par: usize,
   ir_par: usize,
   jr_par: usize,
}


impl CorenumPar {
   pub fn new(
       num_threads: usize,
       ic_par: usize,
       pc_par: usize,
       jc_par: usize,
       ir_par: usize,
       jr_par: usize,
   ) -> Self {
       assert_eq!(num_threads, jc_par*pc_par*ic_par*jr_par*ir_par);
       Self {
           num_threads,
           ic_par,
           pc_par,
           jc_par,
           ir_par,
           jr_par,
       }
   }
   pub fn from_num_threads(num_threads: usize) -> Self {
       let ic_par = num_threads;
       let pc_par = 1;
       let jc_par = 1;
       let ir_par = 1;
       let jr_par = 1;
       Self {
           num_threads,
           ic_par,
           pc_par,
           jc_par,
           ir_par,
           jr_par,
       }
   }
   #[inline(always)]
   pub fn default() -> Self {
       let num_threads = glare_num_threads();
       Self::from_num_threads(num_threads)
   }
   #[inline]
   fn get_ic_id(&self, t_id: usize) -> usize {
       (t_id / (self.pc_par*self.jc_par*self.ir_par*self.jr_par)) % self.ic_par
   }

//    #[inline]
//    fn get_pc_id(&self, t_id: usize) -> usize {
//        (t_id / (self.jr_par*self.ir_par*self.ic_par)) % self.pc_par
//    }
   #[inline]
   fn get_jc_id(&self, t_id: usize) -> usize {
       (t_id / (self.jr_par*self.ir_par)) % self.jc_par
   }
   #[inline]
   fn get_jr_id(&self, t_id: usize) -> usize {
       (t_id / self.ir_par) % self.jr_par
   }
   #[inline]
   fn get_ir_id(&self, t_id: usize) -> usize {
       t_id % self.ir_par
   }
}

#[inline]
fn split_c_range(m: usize, mc: usize, mr: usize, ic_id: usize, ic_par: usize) -> (usize, usize, bool) {
   let chunk_len = (m / (mr*ic_par)) * mr;
   let rem = m % (mr*ic_par);
   if ic_id == 0 {
        let x = chunk_len + rem%mr;
        let mc_left = ((((x+mc-1) / mc ) * mc) * ic_par) < m;
        return (m - chunk_len - (rem%mr), m, mc_left);
    }
    let ic_id = ic_id - 1;
    let m0 = (m / mr) * mr;
   let rem = m0 % (mr*ic_par);
   let start_delta =  rem.min(ic_id*mr);
   let end_delta = rem.min((ic_id + 1)*mr);
//    let is_m_boundary = (chunk_len + end_delta - start_delta ) % mc == 0;
   let mc_coeff = (chunk_len + end_delta - start_delta + mc -1) / mc;
   let mc_left = ((mc_coeff * mc) * ic_par ) < m;
//    let mc_left = is_m_boundary && rem != 0 && end_delta == start_delta;
   (chunk_len * ic_id + start_delta, chunk_len * (ic_id + 1) + end_delta, mc_left)
}

#[inline]
fn split_range(range_len: usize, unit_len: usize, r_id: usize, r_par: usize) -> (usize, usize) {
   let chunk_start = (range_len / (unit_len*r_par)) * unit_len * r_id;
   let chunk_end = (range_len / (unit_len*r_par)) * unit_len * (r_id + 1);
   let rem = range_len % (unit_len*r_par);
   let rem = rem - rem % unit_len;
   let rem_start =  rem.min(r_id*unit_len);
   let rem_end = rem.min((r_id + 1)*unit_len);
   if r_id == r_par - 1 {
       return (chunk_start + rem_start, range_len);
   }
   (chunk_start + rem_start, chunk_end + rem_end)
}

pub trait Tensor2D {
    fn rs(&self) -> usize;
    fn cs(&self) -> usize;
    unsafe fn at(&self, i: usize, j: usize) -> Self;
}

pub trait GemmArray<Y>: Copy + Send + 'static + Tensor2D{
    type X;
    // type Y: BaseNum;
    type PackArray: GemmArrayP<Self::X,Y>+Copy+Send;
    fn is_packing_needed() -> bool;
    fn is_compute_native() -> bool;
    fn into_pack_array(self, a: *mut Y) -> Self::PackArray;
    fn get_data_ptr(&self) -> *const Self::X;
    fn transpose(&mut self);
}

pub trait GemmArrayP<T,U>: Tensor2D
{
    // type StridedArray;
    unsafe fn packa_dispatch_hw<H:GemmPackA<T,U>>(&self, x: &H, mc: usize, kc: usize, mc_len: usize, kc_len: usize, mc_i: usize, run: bool) -> *const U;
    unsafe fn packb_dispatch_hw<H:GemmPackB<T,U>>(&self, x: &H, nc: usize, kc: usize, nc_len: usize, kc_len: usize, run: bool) -> *const U;
    unsafe fn add_p(&self, offset: usize) -> Self;
    unsafe fn get_data_p_ptr(&self) -> *mut U;
    fn get_data_ptr(&self) -> *const T;
}

pub trait GemmOut: Copy+Send+'static {
    type X: BaseNum;
    type Y: BaseNum;

    fn data_ptr(&self) -> *mut Self::X;
    fn rs(&self) -> usize;
    fn cs(&self) -> usize;

    unsafe fn add(self, i: usize) -> Self;

    fn transpose(&mut self);
}

pub trait GemmPackA<AX,AY> 
where Self: Sized
{
    unsafe fn packa_fn(self: &Self, a: *const AX, b: *mut AY, m: usize, k: usize, a_rs: usize, a_cs: usize);
    unsafe fn packa<A: GemmArray<AY,X=AX>>(
        x: &Self,
        a: A::PackArray, 
    mc_i: usize, kc_i: usize,
    mc_len: usize, kc_len: usize,
    t_cfg: &CorenumThreadConfig
    ) -> *const AY {
        t_cfg.wait_packa();
        let x = a.packa_dispatch_hw::<Self>(x, mc_i, kc_i, mc_len, kc_len, 0, t_cfg.run_packa);
        t_cfg.wait_packa();
        x
    }
}


pub trait GemmPackB<BX,BY> 
where Self: Sized
{
    unsafe fn packb_fn(self: &Self, a: *const BX, b: *mut BY, m: usize, k: usize, b_rs: usize, b_cs: usize);
    unsafe fn packb<B: GemmArray<BY,X=BX>>(
        self: &Self,
        b: B::PackArray, 
        nc: usize, kc: usize,
        nc_len: usize, kc_len: usize,
        t_cfg: &CorenumThreadConfig
    ) -> *const BY {
        t_cfg.wait_packb();
        let x = b.packb_dispatch_hw::<Self>(self, nc, kc, nc_len, kc_len, t_cfg.run_packb);
        t_cfg.wait_packb();
        x
    }
}


pub trait BaseNum: Copy + 'static + Send {}

impl<T> BaseNum for T where T: Copy + 'static + Send {}


fn get_mem_pool_size<
AP: BaseNum,
BP: BaseNum,
A: GemmArray<AP>,
B: GemmArray<BP>,
C: GemmOut,
HWConfig: GemmGotoPackaPackb<AP,BP,A,B,C> + GemmSmallM<AP,BP,A,B,C> + GemmSmallN<AP,BP,A,B,C>
>(hw_config: &HWConfig, par: &CorenumPar, m: usize, n: usize) -> usize
{
    if run_small_m::<BP,B>(m) && HWConfig::IS_EFFICIENT{
        return get_mem_pool_size_small_m(hw_config, par);
    }
    if run_small_n::<AP,A>(n) {
        return get_mem_pool_size_small_n(hw_config, par);
    }
    get_mem_pool_size_goto(hw_config, par)
}

fn get_mem_pool_size_goto<
AP: BaseNum,
BP: BaseNum,
A: GemmArray<AP>, 
B: GemmArray<BP>,
C: GemmOut,
HWConfig: GemmGotoPackaPackb<AP,BP,A,B,C>
>(hw_config: &HWConfig, par: &CorenumPar) -> usize
{
    let mut mem_pool_size = 0;
    if A::is_packing_needed() {
        let ap_pool_multiplicity = par.num_threads;
        let ap_pool_size = hw_config.get_ap_pool_size(par.ic_par);
        mem_pool_size += ap_pool_size * std::mem::size_of::<AP>() * ap_pool_multiplicity;
    }
    if B::is_packing_needed() {
        let bp_pool_multiplicity = par.jc_par;
        let bp_pool_size = hw_config.get_bp_pool_size(par.jc_par);
        mem_pool_size += bp_pool_size * std::mem::size_of::<BP>() * bp_pool_multiplicity;
    }
    if mem_pool_size == 0 {
        return 0;
    }
    mem_pool_size += 1024;
    mem_pool_size
}

fn get_mem_pool_size_small_m<
AP: BaseNum,
BP: BaseNum,
A: GemmArray<AP>, 
B: GemmArray<BP>,
C: GemmOut,
HWConfig: GemmSmallM<AP,BP,A,B,C>
>(hw_config: &HWConfig,par: &CorenumPar) -> usize
{
    let mut mem_pool_size = 0;
    if A::is_packing_needed() {
        let ap_pool_multiplicity = par.ic_par;
        let ap_pool_size = hw_config.get_ap_pool_size(par.ic_par);
        mem_pool_size += ap_pool_size * std::mem::size_of::<AP>() * ap_pool_multiplicity;
    }
    if mem_pool_size == 0 {
        return 0;
    }
    mem_pool_size += 1024;
    mem_pool_size
}

fn get_mem_pool_size_small_n<
AP: BaseNum,
BP: BaseNum,
A: GemmArray<AP>, 
B: GemmArray<BP>,
C: GemmOut,
HWConfig: GemmGotoPackaPackb<AP,BP,A,B,C>
>(hw_config: &HWConfig, par: &CorenumPar) -> usize
{
    let mut mem_pool_size = 0;
    if A::is_packing_needed() {
        let ap_pool_multiplicity = par.num_threads;
        let ap_pool_size = hw_config.get_ap_pool_size2();
        mem_pool_size += ap_pool_size * std::mem::size_of::<AP>() * ap_pool_multiplicity;
    }
    if B::is_packing_needed() {
        let bp_pool_multiplicity = par.jc_par;
        let bp_pool_size = hw_config.get_bp_pool_size(par.jc_par);
        mem_pool_size += bp_pool_size * std::mem::size_of::<BP>() * bp_pool_multiplicity;
    }
    if mem_pool_size == 0 {
        return 0;
    }
    mem_pool_size += 1024;
    mem_pool_size
}

// Choose ap_size, bp_size as arguments since they are specific to Gemm implementation,
// It is determined by hardware, gemm implementation (e.g. f64, f32, f16),
// Otherwise, this base crate would include code coupled with other gemm crates,
// this would require either cyclic dep (Not allowed of course) or separate code for each specii hardware and gemm
// imple inside this crate, which is not desirable. We want this crate to be as decoupled as possbile from
// specific gemm implementation and hardware.

fn run_small_m<BP, B: GemmArray<BP>>(m: usize) -> bool {
    B::is_packing_needed() && m < 144 && B::is_compute_native()
}

fn run_small_n<AP, A: GemmArray<AP>>(n: usize) -> bool {
    A::is_packing_needed() && n < 144 && A::is_compute_native()
}

pub trait AccCoef {
    type AS: BaseNum;
    type BS: BaseNum;
}

pub unsafe fn glare_gemm<
AP: BaseNum,
BP: BaseNum,
A: GemmArray<AP>, 
B: GemmArray<BP>,
C: GemmOut,
HWConfig: GemmGotoPackaPackb<AP,BP,A,B,C> + GemmSmallM<AP,BP,A,B,C> + GemmSmallN<AP,BP,A,B,C> + Gemv<AP,BP,A,B,C> + Gemv<BP,AP,B,A,C> + AccCoef,
>(
    hw_config: &HWConfig,
	m: usize, n: usize, k: usize,
	alpha: HWConfig::AS,
	a: A,
	b: B,
	beta: HWConfig::BS,
	c: C,
	par: &CorenumPar,
)
{
    if n == 1 && A::is_packing_needed() {
        glare_gemv::<AP,BP,A,B,C,HWConfig>(hw_config, m, k, alpha, a, b, beta, c, par);
        return;
    }
    if m == 1 && B::is_packing_needed() {
        let mut a = a;
        a.transpose();
        let mut b = b;
        b.transpose();
        let mut c = c;
        c.transpose();
        glare_gemv::<BP,AP,B,A,C,HWConfig>(hw_config, n, k, alpha.into(), b, a, beta, c, par);
        return;
    }
    let gemm_fun = if run_small_m::<BP,B>(m) && HWConfig::IS_EFFICIENT {
        HWConfig::gemm_small_m
    } else if run_small_n::<AP,A>(n) {
        HWConfig::gemm_small_n
    } else {
        HWConfig::gemm_packa_packb
    };
    
    let mem_pool_size = get_mem_pool_size::<AP,BP,A,B,C,HWConfig>(hw_config, par, m, n);
    if mem_pool_size == 0 {
        let mut pool_vec = vec![0_u8; 1];
        let pool_buf = pool_vec.as_mut_ptr();
        gemm_fun(
            hw_config, m, n, k, alpha, a, b, beta, c, par, pool_buf
        );
        return;
    }
	// run goto algo
	{
		let pool_guard = PACK_POOL.buffer.read().unwrap();
		let y = acquire(&pool_guard, mem_pool_size);
		if let Some(mut pool_vec) = y {
            let pool_buf = pool_vec.as_mut_ptr();
            gemm_fun( 
                hw_config, m, n, k, alpha, a, b, beta, c, par, pool_buf
            );
			return;
		}
	}
    let mut pool_vec = vec![0_u8; mem_pool_size];
    let pool_buf = pool_vec.as_mut_ptr();
    gemm_fun( 
        hw_config, m, n, k, alpha, a, b, beta, c, par, pool_buf
    );
	extend(pool_vec);

}

pub unsafe fn glare_gemv<
AP,
BP,
A: GemmArray<AP>, 
B: GemmArray<BP>,
C: GemmOut,
HWConfig: Gemv<AP,BP,A,B,C> + AccCoef,
>(	
    hw_config: &HWConfig,
    m: usize, n: usize,
	alpha: HWConfig::AS,
	a: A,
	x: B,
	beta: HWConfig::BS,
	y: C,
	par: &CorenumPar
) {
    HWConfig::gemv(
        hw_config, m, n, alpha, a, x, beta, y, par
    );
}

#[derive(Clone, Copy)]
pub struct StridedMatrix<T> {
    pub data_ptr : *const T,
    pub rs: usize,
    pub cs: usize,
}

impl<T> StridedMatrix<T> {
    pub fn new(data_ptr: *const T, rs: usize, cs: usize) -> Self {
        Self {
            data_ptr,
            rs,
            cs,
        }
    }
}

unsafe impl<T> Send for StridedMatrix<T> {}
// unsafe impl<T> Sync for StridedMatrix<T> {}

impl<T> Tensor2D for StridedMatrix<T> {
    fn rs(&self) -> usize {
        self.rs
    }
    fn cs(&self) -> usize {
        self.cs
    }
    unsafe fn at(&self, i: usize, j: usize) -> Self {
        Self {
            data_ptr: self.data_ptr.add(i*self.rs + j*self.cs),
            rs: self.rs,
            cs: self.cs,
        }
    }
}

#[derive(Clone, Copy)]
pub struct StridedMatrixMut<T> {
    pub data_ptr : *mut T,
    pub rs: usize,
    pub cs: usize,
}

unsafe impl<T> Send for StridedMatrixMut<T> {}
// unsafe impl<T> Sync for StridedMatrixMut<T> {}

impl<T> StridedMatrixMut<T> {
    pub fn new(data_ptr: *mut T, rs: usize, cs: usize) -> Self {
        Self {
            data_ptr,
            rs,
            cs,
        }
    }
}

impl<T: BaseNum> GemmOut for StridedMatrixMut<T> {
	type X = T;
	type Y = T;

    fn data_ptr(&self) -> *mut T {
        self.data_ptr
    }
    fn rs(&self) -> usize {
        self.rs
    }
    fn cs(&self) -> usize {
        self.cs
    }

    unsafe fn add(self, i: usize) -> Self {
        Self {
            data_ptr: self.data_ptr.add(i),
            rs: self.rs,
            cs: self.cs,
        }
    }
    fn transpose(&mut self) {
        let temp = self.rs;
        self.rs = self.cs;
        self.cs = temp;
    }
}

#[derive(Clone, Copy)]
pub struct StridedMatrixP<T,U> {
    pub data_ptr : *const T,
    pub rs: usize,
    pub cs: usize,
    pub data_p_ptr: *mut U,
}

unsafe impl<T,U> Send for StridedMatrixP<T,U> {}
// unsafe impl<T,U> Sync for StridedMatrixP<T,U> {}

impl<T:BaseNum> GemmArray<T> for StridedMatrix<T> {
    type X = T;
    // type Y = T;
    type PackArray = StridedMatrixP<T,T>;
    fn is_packing_needed() -> bool {
        true
    }
    fn is_compute_native() -> bool {
        true
    }
    fn into_pack_array(self, a: *mut T) -> Self::PackArray {
        StridedMatrixP { data_ptr: self.data_ptr, rs: self.rs, cs: self.cs, data_p_ptr: a }
    }

    fn get_data_ptr(&self) -> *const T {
        self.data_ptr
    }

    fn transpose(&mut self) {
        let temp = self.rs;
        self.rs = self.cs;
        self.cs = temp;
    }
}

#[cfg(feature = "f16")]
use half::f16;

#[cfg(feature = "f16")]
impl GemmArray<f32> for StridedMatrix<f16> {
    type X = f16;
    type PackArray = StridedMatrixP<f16,f32>;
    fn is_packing_needed() -> bool {
        true
    }
    fn is_compute_native() -> bool {
        false
    }
    fn into_pack_array(self, a: *mut f32) -> Self::PackArray {
        StridedMatrixP { data_ptr: self.data_ptr, rs: self.rs, cs: self.cs, data_p_ptr: a }
    }

    fn get_data_ptr(&self) -> *const f16 {
        self.data_ptr
    }

    fn transpose(&mut self) {
        let temp = self.rs;
        self.rs = self.cs;
        self.cs = temp;
    }
}

impl<T,U> Tensor2D for StridedMatrixP<T,U> {
    fn rs(&self) -> usize {
        self.rs
    }
    fn cs(&self) -> usize {
        self.cs
    }
    unsafe fn at(&self, i: usize, j: usize) -> Self {
        Self {
            data_ptr: self.data_ptr.add(i*self.rs + j*self.cs),
            rs: self.rs,
            cs: self.cs,
            data_p_ptr: self.data_p_ptr,
        }
    }
}



impl<T, U> GemmArrayP<T,U> for StridedMatrixP<T,U> {
    unsafe fn packa_dispatch_hw<H:GemmPackA<T,U>>(&self, x: &H, mc: usize, kc: usize, mc_len: usize, kc_len: usize, mc_i: usize, run: bool) -> *const U
    {
        if run {
            let a = self.data_ptr.add(mc*self.rs + kc*self.cs);
            x.packa_fn(a, self.data_p_ptr.add(mc_i*kc_len), mc_len, kc_len , self.rs, self.cs);
        }
        self.data_p_ptr
    }
    unsafe fn packb_dispatch_hw<H:GemmPackB<T,U>>(&self, x: &H, nc: usize, kc: usize, nc_len: usize, kc_len: usize, run: bool) -> *const U
    {
        if run {
            let a = self.data_ptr.add(kc*self.rs + nc*self.cs);
            x.packb_fn(a, self.data_p_ptr, nc_len, kc_len , self.rs, self.cs);
        }

        self.data_p_ptr
    }

    unsafe fn add_p(&self, offset: usize) -> Self {
        Self {
            data_ptr: self.data_ptr,
            rs: self.rs,
            cs: self.cs,
            data_p_ptr: self.data_p_ptr.add(offset),
        }
    }

    unsafe fn get_data_p_ptr(&self) -> *mut U {
        self.data_p_ptr
    }

    fn get_data_ptr(&self) -> *const T {
        self.data_ptr
    }
}


#[derive(Clone, Copy)]
pub struct PackedMatrix<T> {
    pub data_ptr : *const T,
    pub mc: usize,
    pub kc: usize,
    pub mr: usize,
    pub k: usize,
    pub m: usize,
    pub rs: usize,
    pub cs: usize,
}

unsafe impl<T> Send for PackedMatrix<T> {}
// unsafe impl<T> Sync for PackedMatrix<T> {}

impl<T:BaseNum> GemmArray<T> for PackedMatrix<T> {
    type X = T;
    // type Y = T;
    type PackArray = PackedMatrix<T>;
    fn is_packing_needed() -> bool {
        false
    }
    fn is_compute_native() -> bool {
        true
    }
    fn into_pack_array(self, _a: *mut T) -> Self::PackArray {
        self
    }

    fn get_data_ptr(&self) -> *const T {
        self.data_ptr
    }

    fn transpose(&mut self) {
        let temp = self.rs;
        self.rs = self.cs;
        self.cs = temp;
    }
}

impl<T> Tensor2D for PackedMatrix<T> {
    fn rs(&self) -> usize {
        self.rs
    }
    fn cs(&self) -> usize {
        self.cs
    }
    unsafe fn at(&self, _i: usize, _j: usize) -> Self {
        Self {
            data_ptr: self.data_ptr,
            mc: self.mc,
            kc: self.kc,
            mr: self.mr,
            k: self.k,
            m: self.m,
            rs: self.rs,
            cs: self.cs,
        }
    }
}

impl<T> GemmArrayP<T,T> for PackedMatrix<T> {
    unsafe fn packa_dispatch_hw<H>(&self, _x: &H, mc: usize, kc: usize, _mc_len: usize, kc_len: usize, _mc_i: usize, _run: bool) -> *const T
    {
        let ib = mc / self.mc;
        let jb = kc / self.kc;
        let mr_block = (mc % self.mc) / self.mr;
        let m_eff = ((self.m.min(self.mc)+self.mr-1) / self.mr) * self.mr;

        self.data_ptr.add(ib*self.k*self.mc + jb*m_eff*self.kc + mr_block * kc_len*self.mr)
    }
    unsafe fn packb_dispatch_hw<H>(&self, _x: &H, nc: usize, kc: usize, _nc_len: usize, kc_len: usize, _run: bool) -> *const T
    {
        let ib = nc / self.mc;
        let jb = kc / self.kc;
        let nr_block = (nc % self.mc) / self.mr;

        self.data_ptr.add(ib*self.k*self.mc + jb*self.m*self.kc + nr_block * kc_len*self.mr)

    }

    unsafe fn add_p(&self, offset: usize) -> Self {
        Self {
            data_ptr: self.data_ptr.add(offset*0),
            mc: self.mc,
            kc: self.kc,
            mr: self.mr,
            k: self.k,
            m: self.m,
            rs: self.rs,
            cs: self.cs,
        }
    }

    unsafe fn get_data_p_ptr(&self) -> *mut T {
        self.data_ptr as *mut T
    }

    fn get_data_ptr(&self) -> *const T {
        self.data_ptr
    }
}



const AP_ALIGN: usize = 1024;
const BP_ALIGN: usize = 1024;


#[inline]
unsafe fn get_ap_bp<TA,TB>(mem_pool: *mut u8, ap_pool_size: usize, _bp_pool_size: usize, ic_mul: usize, _jc_mul: usize) -> (*mut TA, *mut TB) {
   let align_offset = mem_pool.align_offset(AP_ALIGN);
   let mem_pool_aligned = mem_pool.add(align_offset);
   let ap = mem_pool_aligned as *mut TA;

   let mem_pool_2 = mem_pool_aligned.add(ap_pool_size*std::mem::size_of::<TA>()*ic_mul);
    let align_offset = mem_pool_2.align_offset(BP_ALIGN);
    let mem_pool_2_aligned = mem_pool_2.add(align_offset);
    let bp = mem_pool_2_aligned as *mut TB;
   (ap, bp)
}

pub trait GemmCache<
AP,
BP,
A: GemmArray<AP>,
B: GemmArray<BP>,
> {
    const CACHELINE_PAD: usize = 4096;
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;
    fn get_mc_eff(&self,par: usize) -> usize;
    fn get_kc_eff(&self) -> usize;
    fn get_nc_eff(&self,par: usize) -> usize;
    fn get_ap_pool_size(&self,ic_par: usize) -> usize {
        if A::is_packing_needed() {
            let mc_eff = self.get_mc_eff(ic_par);
            let kc_eff = self.get_kc_eff();
            mc_eff * kc_eff + Self::CACHELINE_PAD / std::mem::size_of::<AP>()
        } else {
            0
        }
    }
    fn get_ap_pool_size2(&self) -> usize {
        if A::is_packing_needed() {
            let kc_eff = self.get_kc_eff();
            self.mr() * kc_eff + Self::CACHELINE_PAD / std::mem::size_of::<AP>()
        } else {
            0
        }
    }
    fn get_bp_pool_size(&self,jc_par: usize) -> usize {
        if B::is_packing_needed() {
            let nc_eff = self.get_nc_eff(jc_par);
            let kc_eff = self.get_kc_eff();
            nc_eff * kc_eff + Self::CACHELINE_PAD / std::mem::size_of::<BP>()
        } else {
            0
        }
    }
}

pub trait GemmGotoPackaPackb<
AP: BaseNum,
BP: BaseNum,
A: GemmArray<AP>, 
B: GemmArray<BP>,
C: GemmOut,
> 
where Self: Sized + GemmCache<AP,BP,A,B>,
Self: GemmPackB<B::X, BP> + Sync + AccCoef,
Self: GemmPackA<A::X, AP>,
{
    const ONE: Self::BS;
   unsafe fn kernel(
        self: &Self,
       m: usize, n: usize, k: usize,
       alpha: *const Self::AS,
       beta: *const Self::BS,
       c: *mut C::X,
       c_rs: usize, c_cs: usize,
       ap: *const AP, bp: *const BP,
       kc_last: bool
   );
   unsafe fn gemm_packa_packb(
    self: &Self,
    m: usize, n: usize, k: usize,
    alpha: Self::AS,
    a: A,
    b: B,
    beta: Self::BS,
    c: C,
    par: &CorenumPar,
    pool_buf: *mut u8,
)
{
    let mc_eff = self.get_mc_eff(par.ic_par);
    let nc_eff = self.get_nc_eff(par.jc_par);
    let kc_eff = self.get_kc_eff();
    let ap_pool_size = self.get_ap_pool_size(par.ic_par);
    let bp_pool_size = self.get_bp_pool_size(par.jc_par);
    let (ap_ptr, bp_ptr) = get_ap_bp::<AP,BP>(pool_buf, ap_pool_size, bp_pool_size, par.ic_par, par.jc_par);
    let ap = a.into_pack_array(ap_ptr);
    let bp = b.into_pack_array(bp_ptr);
    let (pa_br_vec_ref, pb_br_vec_ref) = get_apbp_barrier(par);

    std::thread::scope(|s| {
        for t_id in 1..par.num_threads {
            let t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref.clone(), pb_br_vec_ref.clone(), t_id, mc_eff, nc_eff, kc_eff);
            let ic_id = t_cfg.ic_id;
            let jc_id = t_cfg.jc_id;
            let ap_cur = ap.add_p(ic_id*ap_pool_size);
            let bp_cur = bp.add_p(jc_id*bp_pool_size);
            let g = self;
            s.spawn(move || {
                    let alpha = &alpha as *const Self::AS;
                    let beta = &beta as *const Self::BS;
                    g.gemm_packa_packb_serial(m, n, k, alpha, ap_cur, bp_cur, beta, c, &t_cfg);
                }
            );
        }
        {
            let t_id: usize = 0;
            let t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref, pb_br_vec_ref, t_id, mc_eff, nc_eff, kc_eff);
            let alpha = &alpha as *const Self::AS;
            let beta = &beta as *const Self::BS;
            self.gemm_packa_packb_serial(m, n, k, alpha, ap, bp, beta, c, &t_cfg);
        }
    });
}
   #[inline]
   unsafe fn gemm_packa_packb_serial(
        self: &Self,
       m: usize, n: usize, k: usize,
       alpha: *const Self::AS,
       a: A::PackArray,
       b: B::PackArray,
       beta: *const Self::BS,
       c: C,
       t_cfg: &CorenumThreadConfig
   ) {
       let ic_id = t_cfg.ic_id;
       let jc_id = t_cfg.jc_id;
       let ir_id = t_cfg.ir_id;
       let jr_id = t_cfg.jr_id;
       let ir_par = t_cfg.par.ir_par;
       let jr_par = t_cfg.par.jr_par;
       let mc = t_cfg.mc_eff;
       let nc = t_cfg.nc_eff;
       let kc = t_cfg.kc_eff;
       let mr = self.mr();
       let nr = self.nr();
       let (mc_start, mc_end, mc_left) = split_c_range(m, mc, mr, ic_id, t_cfg.par.ic_par);
       let (nc_start, nc_end, nc_left) = split_c_range(n, nc, nr, jc_id, t_cfg.par.jc_par);
       let (kc_start, kc_end) = (0, k);
       let one = Self::ONE;
       let mut mc_i = mc_start;
       let c_rs = c.rs();
       let c_cs = c.cs();
       let c_ptr = c.data_ptr();
       while mc_i < mc_end {
           let mc_len = mc.min(mc_end - mc_i);
           let (mr_start, mr_end) = split_range(mc_len, mr, ir_id, ir_par);
           let mr_len = mr_end - mr_start;
           let c_i = c_ptr.add((mc_i+mr_start) * c_rs);
           let mut kc_i = kc_start;
           while kc_i < kc_end {
               let kc_len = kc.min(kc_end - kc_i);
               let kc_last = kc_i + kc_len == kc_end;
               let beta_t = if kc_i == kc_start { beta } else { &one as *const Self::BS};
               let mut nc_i = nc_start;
               let ap = Self::packa::<A>(self,a, mc_i, kc_i, mc_len, kc_len, t_cfg);
               let ap = ap.add(mr_start*kc_len);
               while nc_i < nc_end {
                   let nc_len = nc.min(nc_end - nc_i);
                   let (nr_start, nr_end) = split_range(nc_len, nr, jr_id, jr_par);
                   let nr_len = nr_end - nr_start;
                   let c_ij = c_i.add((nc_i+nr_start) * c_cs);
                   let bp = Self::packb::<B>(self, b, nc_i, kc_i, nc_len, kc_len, t_cfg);
                   let bp = bp.add(nr_start*kc_len);
                    self.kernel(
                        mr_len, nr_len, kc_len, alpha, beta_t, c_ij, c_rs, c_cs,
                        ap, bp,
                        kc_last
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
            t_cfg.wait_packa();
            t_cfg.wait_packa();
            let mut nc_i = nc_start;
            while nc_i < nc_end {
                let nc_len = nc.min(nc_end - nc_i);
                let _ = Self::packb::<B>(self,b, nc_i, kc_i, nc_len, kc_len, t_cfg);
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
}

pub trait GemmSmallM<
AP: BaseNum,
BP: BaseNum,
A: GemmArray<AP>, 
B: GemmArray<BP>,
C: GemmOut,
> 
where Self: Sized + GemmCache<AP,BP,A,B> + Sync,
Self: GemmPackA<A::X, AP> + AccCoef
{
    const ONE: Self::BS;
    // for some gemm impl, it is more efficient to use goto than small m
    const IS_EFFICIENT: bool = true;
    unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const Self::AS,
        beta: *const Self::BS,
        b: *const B::X, b_rs: usize, b_cs: usize,
        c: *mut C::X, c_rs: usize, c_cs: usize,
        ap: *const AP,
    );
    unsafe fn gemm_small_m(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: Self::AS,
        a: A,
        b: B,
        beta: Self::BS,
        c: C,
        par: &CorenumPar,
        pack_pool: *mut u8,
    )
    {
        let mc_eff = self.get_mc_eff(par.ic_par);
        let nc_eff = self.get_nc_eff(par.jc_par);
        let kc_eff = self.get_kc_eff();
        let ap_pool_size = self.get_ap_pool_size(par.ic_par);
        let align_offset = pack_pool.align_offset(AP_ALIGN);
        let ap_ptr = (pack_pool.add(align_offset)) as *mut AP;
        let ap = a.into_pack_array(ap_ptr);
        let (pa_br_vec_ref, pb_br_vec_ref) = get_apbp_barrier(par);
    
        std::thread::scope(|s| {
            // let c 
            for t_id in 1..par.num_threads {
                let t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref.clone(), pb_br_vec_ref.clone(), t_id, mc_eff, nc_eff, kc_eff);
                let ic_id = t_cfg.ic_id;
                let ap_cur = ap.add_p(ic_id*ap_pool_size);
        
                s.spawn(move || {
                        let alpha = &alpha as *const Self::AS;
                        let beta = &beta as *const Self::BS;
                        self.gemm_small_m_serial(m, n, k, alpha, ap_cur, b, beta, c, &t_cfg);
                    }
                );
            }
        
            {
                let t_id: usize = 0;
                let t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref, pb_br_vec_ref, t_id, mc_eff, nc_eff, kc_eff);
                let alpha = &alpha as *const Self::AS;
                let beta = &beta as *const Self::BS;
                self.gemm_small_m_serial(m, n, k, alpha, ap, b, beta, c, &t_cfg);
            }
        });
    }
    #[inline]
    unsafe fn gemm_small_m_serial(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const Self::AS,
        a: A::PackArray,
        b: B,
        beta: *const Self::BS,
        c: C,
        t_cfg: &CorenumThreadConfig
    ) {
        let par = &t_cfg.par;
        let ic_id = t_cfg.ic_id;
        let jc_id = t_cfg.jc_id;
        let ir_id = t_cfg.ir_id;
        let ir_par = par.ir_par;
        let jr_id = t_cfg.jr_id;
        let jr_par = par.jr_par;
        let mc_eff = t_cfg.mc_eff;
        let nc_eff = t_cfg.nc_eff;
        let kc_eff = t_cfg.kc_eff;
        let mr = self.mr();
        let nr = self.nr();
        let (mc_start, mc_end, mc_left) = split_c_range(m, mc_eff, mr, ic_id, par.ic_par);
        let (nc_start, nc_end, _) = split_c_range(n, nc_eff, nr, jc_id, par.jc_par);
        let (kc_start, kc_end) = (0, k);
        let one = Self::ONE;

        let mut mc = mc_start;
        let mc_end = mc_end;
        let b_ptr = b.get_data_ptr();
        let b_rs = b.rs();
        let b_cs = b.cs();
        let c_rs = c.rs();
        let c_cs = c.cs();
        let c_ptr = c.data_ptr();
        while mc < mc_end {
            let mc_len = mc_eff.min(mc_end - mc);
             let (mr_start, mr_end) = split_range(mc_len, mr, ir_id, ir_par);
 
            let mr_len = mr_end - mr_start;
            let c_i = c_ptr.add((mc+mr_start) * c_rs);
            let mut kc = kc_start;
            while kc < kc_end {
                let kc_len = kc_eff.min(kc_end - kc);
                let beta_t = if kc == kc_start { beta } else { &one as *const Self::BS};
                let mut nc = nc_start;
                let ap = Self::packa::<A>(self, a, mc, kc, mc_len, kc_len, t_cfg);
                let ap = ap.add(mr_start*kc_len);
                let b_j = b_ptr.add(kc * b_rs);
                while nc < nc_end {
                    let nc_len = nc_eff.min(nc_end - nc);
                    let (nr_start, nr_end) = split_range(nc_len, nr, jr_id, jr_par);
                    let nr_len = nr_end - nr_start;
                    let c_ij = c_i.add((nc + nr_start) * c_cs);
                    let b_cur = b_j.add((nc+nr_start) * b_cs);
                    self.kernel(
                        mr_len, nr_len, kc_len, alpha, beta_t, 
                        b_cur, b_rs, b_cs,
                        c_ij, c_rs, c_cs,
                        ap
                    );
                    nc += nc_eff;
                }
                kc += kc_eff;
            }
            mc += mc_eff;
        }

        if mc_left {
            let mut kc_i = kc_start;
            while kc_i < kc_end {
                t_cfg.wait_packa();
                t_cfg.wait_packa();
                kc_i += kc_eff;
            }
           }
    }
 }

 pub trait GemmSmallN<
 AP: BaseNum,
 BP: BaseNum,
 A: GemmArray<AP>, 
 B: GemmArray<BP>,
C: GemmOut,
 >
where Self: Sized + GemmCache<AP,BP,A,B> + Sync + AccCoef,
Self: GemmPackB<B::X, BP>,
 {
    const ONE: Self::BS;
    unsafe fn kernel(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const Self::AS,
        beta: *const Self::BS,
        a: *const A::X, a_rs: usize, a_cs: usize,
        ap: *mut AP,
        b: *const BP,
        c: *mut C::X, c_rs: usize, c_cs: usize,
    );
    unsafe fn gemm_small_n(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: Self::AS,
        a: A,
        b: B,
        beta: Self::BS,
        c: C,
        par: &CorenumPar,
        pack_pool: *mut u8,
    )
    {
        let mc_eff = self.get_mc_eff(par.ic_par);
        let nc_eff = self.get_nc_eff(par.jc_par);
        let kc_eff = self.get_kc_eff();
        let ap_pool_size = self.get_ap_pool_size2();
        let bp_pool_size = self.get_bp_pool_size(par.jc_par);
        let (ap_ptr, bp_ptr) = get_ap_bp::<AP,BP>(pack_pool, ap_pool_size, bp_pool_size, par.num_threads, par.jc_par);
        let b = b;
        let bp = b.into_pack_array(bp_ptr);
        let ap = a.into_pack_array(ap_ptr);
        let (pa_br_vec_ref, pb_br_vec_ref) = get_apbp_barrier(par);
    
        std::thread::scope(|s| {
            for t_id in 1..par.num_threads {
                let t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref.clone(), pb_br_vec_ref.clone(), t_id, mc_eff, nc_eff, kc_eff);
                let jc_id = t_cfg.jc_id;
                let ap_cur = ap.add_p(t_id*ap_pool_size);
                let bp_cur = bp.add_p(jc_id*bp_pool_size);
                s.spawn(move || {
                        let alpha = &alpha as *const Self::AS;
                        let beta = &beta as *const Self::BS;
                        self.gemm_small_n_serial(m, n, k, alpha, ap_cur, bp_cur, beta, c, &t_cfg);
                    }
                );
            }
    
            let t_id: usize = 0;
            let t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref, pb_br_vec_ref, t_id, mc_eff, nc_eff, kc_eff);
            let alpha = &alpha as *const Self::AS;
            let beta = &beta as *const Self::BS;
            self.gemm_small_n_serial(m, n, k, alpha, ap, bp, beta, c, &t_cfg);
        });
    }
    #[inline]
    unsafe fn gemm_small_n_serial(
        self: &Self,
        m: usize, n: usize, k: usize,
        alpha: *const Self::AS,
        a: A::PackArray,
        b: B::PackArray,
        beta: *const Self::BS,
        c: C,
        t_cfg: &CorenumThreadConfig
    ) {
        let par = &t_cfg.par;
        let ic_id = t_cfg.ic_id;
        let jc_id = t_cfg.jc_id;
        let ir_id = t_cfg.ir_id;
        let ir_par = par.ir_par;
        let jr_id = t_cfg.jr_id;
        let jr_par = par.jr_par;
        let mc_eff = t_cfg.mc_eff;
        let nc_eff = t_cfg.nc_eff;
        let kc_eff = t_cfg.kc_eff;
        let mr = self.mr();
        let nr = self.nr();
        let (mc_start, mc_end, mc_left) = split_c_range(m, mc_eff, mr, ic_id, par.ic_par);
        let (nc_start, nc_end, nc_left) = split_c_range(n, nc_eff, nr, jc_id, par.jc_par);
        let (kc_start, kc_end) = (0, k);
        let one = Self::ONE;

        let mut mc = mc_start;
        let mc_end = mc_end;
        let c_rs = c.rs();
        let c_cs = c.cs();
        let c_ptr = c.data_ptr();
        let a_ptr = a.get_data_ptr();
        let a_rs = a.rs();
        let a_cs = a.cs();
        let ap = a.get_data_p_ptr();
        while mc < mc_end {
            let mc_len = mc_eff.min(mc_end - mc);
             let (mr_start, mr_end) = split_range(mc_len, mr, ir_id, ir_par);
 
            let mr_len = mr_end - mr_start;
            let c_i = c_ptr.add((mc+mr_start) * c_rs);
            let a_i = a_ptr.add((mc+mr_start) * a_rs);
            let mut kc = kc_start;
            while kc < kc_end {
                let kc_len = kc_eff.min(kc_end - kc);
                let beta_t = if kc == kc_start { beta } else { &one as *const Self::BS};
                let a_cur = a_i.add(kc*a_cs);
                let mut nc = nc_start;
 
                while nc < nc_end {
                    let nc_len = nc_eff.min(nc_end - nc);
                    let (nr_start, nr_end) = split_range(nc_len, nr, jr_id, jr_par);
                    let nr_len = nr_end - nr_start;
                    let b_cur = self.packb::<B>(b, nc, kc, nc_len, kc_len, t_cfg);
                    let c_ij = c_i.add((nc + nr_start) * c_cs);
                    self.kernel(
                        mr_len, nr_len, kc_len, alpha, beta_t, 
                        a_cur, a_rs, a_cs,
                        ap,
                        b_cur,
                        c_ij, c_rs, c_cs,
                    );
                    nc += nc_eff;
                }
                if nc_left {
                    t_cfg.wait_packb();
                    t_cfg.wait_packb();
                }

                kc += kc_eff;
            }
            mc += mc_eff;
        }
        if mc_left {
            let mut kc_i = kc_start;
            while kc_i < kc_end {
                let kc_len = kc_eff.min(kc_end - kc_i);
                let mut nc_i = nc_start;
                while nc_i < nc_end {
                    let nc_len = nc_eff.min(nc_end - nc_i);
                    let _ = self.packb::<B>(b, nc_i, kc_i, nc_len, kc_len, t_cfg);
                    nc_i += nc_eff;
                }
                if nc_left{
                    t_cfg.wait_packb();
                    t_cfg.wait_packb();
                }
                kc_i += kc_eff;
            }
           }
    }
 }
 

pub trait Gemv<
AP,
BP,
A: GemmArray<AP>, 
B: GemmArray<BP>,
C: GemmOut,
>
where Self: Sized + AccCoef,
 {
   unsafe fn gemv(
        self: &Self,
       m: usize, n: usize,
       alpha: Self::AS,
       a: A,
       x: B,
       beta: Self::BS,
       y: C,
       _par: &CorenumPar
   ) {
       let alpha = &alpha as *const Self::AS;
       let beta = &beta as *const Self::BS;
       self.gemv_serial(m, n, alpha, a, x, beta, y);
   }
   unsafe fn gemv_serial(
        self: &Self,
       m: usize, n: usize,
       alpha: *const Self::AS,
       a: A,
       x: B,
       beta: *const Self::BS,
       y: C
   );
}

mod test {
    // test split_c_range
    #[test]
    fn test_split_c_range() {
        let m = 143;
        let mc = 4800;
        let mr = 24;
        let ic_par = 4;
        for ic_id in 0..ic_par {
            let (mc_start, mc_end, mc_left) = super::split_c_range(m, mc, mr, ic_id, ic_par);
            println!("mc_start: {}, mc_end: {}, mc_left: {}", mc_start, mc_end, mc_left);
        }
        assert!(false);
    }
}