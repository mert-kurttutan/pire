

use std::sync::Mutex;
use std::sync::Barrier;
use std::sync::Arc;

pub mod asm_macro;


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


pub enum HWConfig {
    Reference,
    Haswell,
}


// Consider Once Cell
use once_cell::sync::Lazy;

// Use family and model id instead of cache size parameters
// since the relation between optimal parameters (based on performance) and cache size parameters  can be non-trivial
// e.g. it might be cpu model dependent

#[inline]
fn detect_hw_config() -> HWConfig {
    #[cfg(target_arch = "x86_64")]
    {
        let cpuid = raw_cpuid::CpuId::new();
        let feature_info = cpuid.get_feature_info().unwrap();
        let extended_prcoessor_info = cpuid.get_extended_processor_and_feature_identifiers().unwrap();
        if feature_info.has_avx() && feature_info.has_fma() && !extended_prcoessor_info.has_fma4() {
            // Default for hw that has avx and fma is haswell
            return HWConfig::Haswell;
        }
    }
    // Fall to Generic Unless Specified
    HWConfig::Reference
}


pub static RUNTIME_HW_CONFIG: Lazy<HWConfig> = Lazy::new(|| {
    detect_hw_config()
});



// Consider Replacing with UnsafeCell
// Mutex so that external multithreading is safe and working
use std::sync::RwLock;

pub struct PackPool {
    pub buffer: RwLock<Vec<Mutex<Vec<u8>>>>
}

pub static PACK_POOL: PackPool = PackPool {
    buffer: RwLock::new(vec![])
};
use std::sync::RwLockReadGuard;
use std::sync::MutexGuard;


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
   pub par: CorenumPar,
   pub packa_barrier: Arc<Vec<Arc<Barrier>>>,
   pub packb_barrier: Arc<Vec<Arc<Barrier>>>,
}


impl<'a> CorenumThreadConfig {
   pub fn new(par: CorenumPar, packa_barrier: Arc<Vec<Arc<Barrier>>>, packb_barrier: Arc<Vec<Arc<Barrier>>>, t_id: usize, mc_eff: usize, nc_eff: usize, kc_eff: usize) -> Self {
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
   pub fn wait_packa(&self) {
    if self.par.jc_par * self.par.pc_par * self.par.ir_par * self.par.jr_par  > 1 {
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


#[derive(Copy,Clone)]
pub struct CorenumPar {
   pub num_threads: usize,
   pub ic_par: usize,
   pub pc_par: usize,
   pub jc_par: usize,
   pub ir_par: usize,
   pub jr_par: usize,
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
       let jc_par = 1;
       let pc_par = 1;
       let ic_par = num_threads;
       let jr_par = 1;
       let ir_par = 1;
       Self {
           num_threads,
           jc_par,
           pc_par,
           ic_par,
           jr_par,
           ir_par,
       }
   }
   #[inline]
   pub fn get_ic_id(&self, t_id: usize) -> usize {
       (t_id / (self.pc_par*self.jc_par*self.ir_par*self.jr_par)) % self.ic_par
   }

   #[inline]
   pub fn get_pc_id(&self, t_id: usize) -> usize {
       (t_id / (self.jr_par*self.ir_par*self.ic_par)) % self.pc_par
   }
   #[inline]
   pub fn get_jc_id(&self, t_id: usize) -> usize {
       (t_id / (self.jr_par*self.ir_par)) % self.jc_par
   }
   #[inline]
   pub fn get_jr_id(&self, t_id: usize) -> usize {
       (t_id / self.ir_par) % self.jr_par
   }
   #[inline]
   pub fn get_ir_id(&self, t_id: usize) -> usize {
       t_id % self.ir_par
   }
}



#[inline]
pub fn split_c_range(m: usize, mc: usize, mr: usize, ic_id: usize, ic_par: usize) -> (usize, usize, bool) {
   let chunk_len = (m / (mr*ic_par)) * mr;
   let rem = m % (mr*ic_par);
   let start_delta =  rem.min(ic_id*mr);
   let end_delta = rem.min((ic_id + 1)*mr);
   let is_m_boundary = (chunk_len + end_delta - start_delta ) % mc == 0;
   let mc_left = is_m_boundary && rem != 0 && end_delta == start_delta;
   (chunk_len * ic_id + start_delta, chunk_len * (ic_id + 1) + end_delta, mc_left)
}

#[inline]
pub fn split_range(range_len: usize, unit_len: usize, r_id: usize, r_par: usize) -> (usize, usize) {
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


pub trait GemmArray {
    type T;
    type U: BaseNum;
    type PackArray: GemmArrayP<Self::T,Self::U>+Copy+Send +Sync+'static;
    fn is_packing_needed() -> bool;
    fn into_pack_array(self, a: *mut Self::U) -> Self::PackArray;
    fn get_rs(&self) -> usize;
    fn get_cs(&self) -> usize;
    fn get_data_ptr(&self) -> *const Self::T;
    fn transpose(&mut self);
}

pub trait GemmArrayP<T,U>
{
    // type StridedArray;
    unsafe fn packa_dispatch_hw<H:GemmPack<T,U>>(&self, mc: usize, kc: usize, mc_len: usize, kc_len: usize, mc_i: usize, run: bool) -> *const U;
    unsafe fn packb_dispatch_hw<H:GemmPack<T,U>>(&self, nc: usize, kc: usize, nc_len: usize, kc_len: usize, run: bool) -> *const U;
    unsafe fn add_p(&self, offset: usize) -> Self;
}

pub trait GemmOut: Copy+Send +Sync+'static {
    type X: BaseNum;
    type Y: BaseNum;

    fn data_ptr(&self) -> *mut Self::X;
    fn rs(&self) -> usize;
    fn cs(&self) -> usize;

    unsafe fn add(self, i: usize) -> Self;

    fn transpose(&mut self);
}

pub trait GemmPack<T,U> {
    unsafe fn packa_fn(a: *const T, b: *mut U, m: usize, k: usize, a_rs: usize, a_cs: usize);
    unsafe fn packb_fn(a: *const T, b: *mut U, m: usize, k: usize, b_rs: usize, b_cs: usize);
}

pub trait BaseNum: Copy + Send + 'static + Send + Sync{}

impl<T> BaseNum for T where T: Copy + Send + 'static + Send + Sync{}

fn get_mem_pool_size<
A: Copy+GemmArray, 
B: Copy+GemmArray,
C: Copy+GemmOut,
Activation: UnaryOp<C::X,C::Y>,
HWConfig: GemmGotoPackaPackb<A,B,C,Activation>
>(par: &CorenumPar) -> usize
{
    let mut mem_pool_size = 0;
    if A::is_packing_needed() {
        let ap_pool_multiplicity = par.ic_par;
        let ap_pool_size = HWConfig::get_ap_pool_size(par.ic_par);
        mem_pool_size += ap_pool_size * std::mem::size_of::<A::U>() * ap_pool_multiplicity;
    }
    if B::is_packing_needed() {
        let bp_pool_multiplicity = par.jc_par;
        let bp_pool_size = HWConfig::get_bp_pool_size(par.jc_par);
        mem_pool_size += bp_pool_size * std::mem::size_of::<B::U>() * bp_pool_multiplicity;
    }
    if mem_pool_size == 0 {
        return 0;
    }
    mem_pool_size += 1024;
    mem_pool_size
}

fn get_mem_pool_size_small_m<
A: Copy+GemmArray, 
B: Copy+GemmArray,
C: Copy+GemmOut,
HWConfig: GemmSmallM<A,B,C>
>(par: &CorenumPar) -> usize
{
    let mut mem_pool_size = 0;
    if A::is_packing_needed() {
        let ap_pool_multiplicity = par.ic_par;
        let ap_pool_size = HWConfig::get_ap_pool_size(par.ic_par);
        mem_pool_size += ap_pool_size * std::mem::size_of::<A::U>() * ap_pool_multiplicity;
    }
    if mem_pool_size == 0 {
        return 0;
    }
    mem_pool_size += 1024;
    mem_pool_size
}

fn get_mem_pool_size_small_n<
A: Copy+GemmArray, 
B: Copy+GemmArray,
C: Copy+GemmOut,
Activation: UnaryOp<C::X,C::Y>,
HWConfig: GemmGotoPackaPackb<A,B,C,Activation>
>(par: &CorenumPar) -> usize
{
    let mut mem_pool_size = 0;
    if B::is_packing_needed() {
        let bp_pool_multiplicity = par.jc_par;
        let bp_pool_size = HWConfig::get_bp_pool_size(par.jc_par);
        mem_pool_size += bp_pool_size * std::mem::size_of::<B::U>() * bp_pool_multiplicity;
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

pub fn run_small_m<A: GemmArray, B: GemmArray>(m: usize) -> bool {
    B::is_packing_needed() && m < 144
}

pub fn run_small_n<A: GemmArray, B: GemmArray>(n: usize) -> bool {
    A::is_packing_needed() && B::is_packing_needed() && n < 144
}
use std::convert::Into;
pub unsafe fn corenum_gemm<
InputA: Copy+GemmArray + Send +Sync+'static, 
InputB: Copy+GemmArray + Send +Sync+'static,
C: Copy+GemmOut,
Activation: UnaryOp<C::X,C::Y>,
HWConfig: GemmGotoPackaPackb<InputA,InputB,C,Activation> + GemmSmallM<InputA,InputB,C> + GemmSmallN<InputA,InputB,C> + Gemv<InputA,InputB,C> + Gemv<InputB,InputA,C>,
>(
	m: usize, n: usize, k: usize,
	alpha: InputA::U,
	a: InputA,
	b: InputB,
	beta: C::X,
	c: C,
	par: &CorenumPar,
)
where InputA::U: Into<InputB::U>
{
    if n == 1 {
        corenum_gemv::<InputA,InputB,C,HWConfig>(m, k, alpha, a, b, beta, c, par);
        return;
    }
    if m == 1 {
        let mut a = a;
        a.transpose();
        let mut b = b;
        b.transpose();
        let mut c = c;
        c.transpose();
        corenum_gemv::<InputB,InputA,C,HWConfig>(n, k, alpha.into(), b, a, beta, c, par);
        return;
    }
    if run_small_m::<InputA,InputB>(m) {
        let mem_pool_size = get_mem_pool_size_small_m::<InputA,InputB,C,HWConfig>(par);
        if mem_pool_size == 0 {
            let mut pool_vec = vec![0_u8; 1];
            let pool_buf = pool_vec.as_mut_ptr();
            HWConfig::gemm_small_m(
                m, n, k, alpha, a, b, beta, c, par, pool_buf
            );
            return;
        }
        // run goto algo
        {
            let pool_guard = PACK_POOL.buffer.read().unwrap();
            let y = acquire(&pool_guard, mem_pool_size);
            if let Some(mut pool_vec) = y {
                let pool_buf = pool_vec.as_mut_ptr();
                HWConfig::gemm_small_m(
                    m, n, k, alpha, a, b, beta, c, par, pool_buf
                );
                return;
            }
        }
        let mut pool_vec = vec![0_u8; mem_pool_size];
        let pool_buf = pool_vec.as_mut_ptr();
        HWConfig::gemm_small_m(
            m, n, k, alpha, a, b, beta, c, par, pool_buf
        );
        extend(pool_vec);
        return;
    }

    if run_small_n::<InputA,InputB>(m) {
        let mem_pool_size = get_mem_pool_size_small_n::<InputA,InputB,C,Activation,HWConfig>(par);
        if mem_pool_size == 0 {
            let mut pool_vec = vec![0_u8; 1];
            let pool_buf = pool_vec.as_mut_ptr();
            HWConfig::gemm_small_n(
                m, n, k, alpha, a, b, beta, c, par, pool_buf
            );
            return;
        }
        // run goto algo
        {
            let pool_guard = PACK_POOL.buffer.read().unwrap();
            let y = acquire(&pool_guard, mem_pool_size);
            if let Some(mut pool_vec) = y {
                let pool_buf = pool_vec.as_mut_ptr();
                HWConfig::gemm_small_n(
                    m, n, k, alpha, a, b, beta, c, par, pool_buf
                );
                return;
            }
        }
        let mut pool_vec = vec![0_u8; mem_pool_size];
        let pool_buf = pool_vec.as_mut_ptr();
        HWConfig::gemm_small_n(
            m, n, k, alpha, a, b, beta, c, par, pool_buf
        );
        extend(pool_vec);
        return;
    }
    let mem_pool_size = get_mem_pool_size::<InputA,InputB,C,Activation,HWConfig>(par);
    if mem_pool_size == 0 {
        let mut pool_vec = vec![0_u8; 1];
        let pool_buf = pool_vec.as_mut_ptr();
        HWConfig::gemm_packa_packb(
            m, n, k, alpha, a, b, beta, c, par, pool_buf
        );
        return;
    }
	// run goto algo
	{
		let pool_guard = PACK_POOL.buffer.read().unwrap();
		let y = acquire(&pool_guard, mem_pool_size);
		if let Some(mut pool_vec) = y {
            let pool_buf = pool_vec.as_mut_ptr();
            HWConfig::gemm_packa_packb(
                m, n, k, alpha, a, b, beta, c, par, pool_buf
            );
			return;
		}
	}
    let mut pool_vec = vec![0_u8; mem_pool_size];
    let pool_buf = pool_vec.as_mut_ptr();
    HWConfig::gemm_packa_packb(
        m, n, k, alpha, a, b, beta, c, par, pool_buf
    );
	extend(pool_vec);

}

pub unsafe fn corenum_gemv<
InputA: Copy+GemmArray, 
InputB: Copy+GemmArray,
C: Copy+GemmOut,
HWConfig: Gemv<InputA,InputB,C>,
>(	m: usize, n: usize,
	alpha: InputA::U,
	a: InputA,
	x: InputB,
	beta: C::X,
	y: C,
	par: &CorenumPar
) {
    HWConfig::gemv(
        m, n, alpha, a, x, beta, y, par
    );
}



#[derive(Clone, Copy)]
pub struct StridedMatrix<T> {
    pub data_ptr : *const T,
    pub rs: usize,
    pub cs: usize,
}

unsafe impl<T> Send for StridedMatrix<T> {}
unsafe impl<T> Sync for StridedMatrix<T> {}

#[derive(Clone, Copy)]
pub struct StridedMatrixMut<T> {
    pub data_ptr : *mut T,
    pub rs: usize,
    pub cs: usize,
}

unsafe impl<T> Send for StridedMatrixMut<T> {}
unsafe impl<T> Sync for StridedMatrixMut<T> {}


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
unsafe impl<T,U> Sync for StridedMatrixP<T,U> {}

impl<T:BaseNum> GemmArray for StridedMatrix<T> {
    type T = T;
    type U = T;
    type PackArray = StridedMatrixP<T,T>;
    fn is_packing_needed() -> bool {
        true
    }
    fn into_pack_array(self, a: *mut T) -> Self::PackArray {
        StridedMatrixP { data_ptr: self.data_ptr, rs: self.rs, cs: self.cs, data_p_ptr: a }
    }

    fn get_rs(&self) -> usize {
        self.rs
    }

    fn get_cs(&self) -> usize {
        self.cs
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

impl<T, U> GemmArrayP<T,U> for StridedMatrixP<T,U> {
    unsafe fn packa_dispatch_hw<H:GemmPack<T,U>>(&self, mc: usize, kc: usize, mc_len: usize, kc_len: usize, mc_i: usize, run: bool) -> *const U
    {
        if run {
            let a = self.data_ptr.add(mc*self.rs + kc*self.cs);
            H::packa_fn(a, self.data_p_ptr.add(mc_i*kc_len), mc_len, kc_len , self.rs, self.cs);
        }
        self.data_p_ptr
    }
    unsafe fn packb_dispatch_hw<H:GemmPack<T,U>>(&self, nc: usize, kc: usize, nc_len: usize, kc_len: usize, run: bool) -> *const U
    {
        if run {
            let a = self.data_ptr.add(kc*self.rs + nc*self.cs);
            H::packb_fn(a, self.data_p_ptr, nc_len, kc_len , self.rs, self.cs);
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
}


#[derive(Clone, Copy)]
pub struct PackedMatrix<T> {
    pub data_ptr : *const T,
    pub mc: usize,
    pub kc: usize,
    pub k: usize,
    pub m: usize,
    pub rs: usize,
    pub cs: usize,
}

unsafe impl<T> Send for PackedMatrix<T> {}
unsafe impl<T> Sync for PackedMatrix<T> {}

impl<T:BaseNum> GemmArray for PackedMatrix<T> {
    type T = T;
    type U = T;
    type PackArray = PackedMatrix<T>;
    fn is_packing_needed() -> bool {
        false
    }
    fn into_pack_array(self, _a: *mut T) -> Self::PackArray {
        self
    }

    fn get_rs(&self) -> usize {
        self.rs
    }
    fn get_cs(&self) -> usize {
        self.cs
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

impl<T> GemmArrayP<T,T> for PackedMatrix<T> {
    unsafe fn packa_dispatch_hw<H>(&self, mc: usize, kc: usize, mc_len: usize, kc_len: usize, mc_i: usize, run: bool) -> *const T
    {
        let mc_block = mc / self.mc;
        let kc_block = kc / self.kc;
        let m_eff = ((mc_len.min(self.mc)+23) / 24) * 24;
        self.data_ptr.add(mc_block*self.k*self.mc + kc_block*m_eff*self.kc)
    }
    unsafe fn packb_dispatch_hw<H>(&self, nc: usize, kc: usize, nc_len: usize, kc_len: usize, run: bool) -> *const T
    {
        let i_block = kc / self.kc;
        let j_block = kc / self.kc;
        self.data_ptr.add((i_block*self.k + j_block*self.kc)*self.mc)
    }

    unsafe fn add_p(&self, offset: usize) -> Self {
        Self {
            data_ptr: self.data_ptr.add(offset),
            mc: self.mc,
            kc: self.kc,
            k: self.k,
            m: self.m,
            rs: self.rs,
            cs: self.cs,
        }
    }
}



const AP_ALIGN: usize = 256;
const BP_ALIGN: usize = 256;


#[inline]
pub unsafe fn get_ap_bp<TA,TB>(mem_pool: *mut u8, ap_pool_size: usize, _bp_pool_size: usize, ic_mul: usize, _jc_mul: usize) -> (*mut TA, *mut TB) {
   let align_offset = mem_pool.align_offset(AP_ALIGN);
   let mem_pool_aligned = mem_pool.add(align_offset);
   let ap = mem_pool_aligned as *mut TA;

   let mem_pool_2 = mem_pool_aligned.add(ap_pool_size*std::mem::size_of::<TA>()*ic_mul);
    let align_offset = mem_pool_2.align_offset(BP_ALIGN);
    let mem_pool_2_aligned = mem_pool_2.add(align_offset);
    let bp = mem_pool_2_aligned as *mut TB;
   (ap, bp)
}

pub trait UnaryOp<X,Y> {
    const IS_IDENTITY: bool;
    unsafe fn apply_inplace(x: *mut X);
    unsafe fn map(x: *const X, y: *mut Y);
}

pub trait GemmGotoPackaPackb<
A: Copy+GemmArray, 
B: Copy+GemmArray,
C: Copy+GemmOut,
Activation: UnaryOp<C::X,C::Y>
> 
where Self: Sized,
Self: GemmPack<B::T, B::U>,
Self: GemmPack<A::T, A::U>,
{
    const CACHELINE_PAD: usize = 256;
   const MC: usize; const NC: usize; const KC: usize;
   const MR: usize; const NR: usize;
   const ONE: C::X;
   const IS_L3_SHARED: bool;
   const IS_L2_SHARED: bool;
    const IS_L1_SHARED: bool;

    fn get_mc_eff(par: usize) -> usize {
        if Self::IS_L3_SHARED {
            Self::MC / par
        } else {
            Self::MC
        }
    }
    fn get_nc_eff(par: usize) -> usize {
        if Self::IS_L2_SHARED {
            Self::NC / par
        } else {
            Self::NC
        }
    }
    fn get_ap_pool_size(ic_par: usize) -> usize {
        let mc_eff = Self::get_mc_eff(ic_par);
        mc_eff * Self::KC + Self::CACHELINE_PAD / std::mem::size_of::<A::U>()
    }
    fn get_bp_pool_size(jc_par: usize) -> usize {
        let nc_eff = Self::get_nc_eff(jc_par);
        nc_eff * Self::KC + Self::CACHELINE_PAD / std::mem::size_of::<B::U>()
    }
   unsafe fn packa(
        a: A::PackArray, 
    mc_i: usize, kc_i: usize,
    mc_len: usize, kc_len: usize,
    t_cfg: &CorenumThreadConfig
    ) -> *const A::U {
        t_cfg.wait_packa();
        let x = a.packa_dispatch_hw::<Self>(mc_i, kc_i, mc_len, kc_len, 0, t_cfg.run_packa);
        t_cfg.wait_packa();
        x
    }
    unsafe fn packb(
        b: B::PackArray, 
        nc: usize, kc: usize,
        nc_len: usize, kc_len: usize,
        t_cfg: &CorenumThreadConfig
    ) -> *const B::U {
        t_cfg.wait_packb();
        let x = b.packb_dispatch_hw::<Self>(nc, kc, nc_len, kc_len, t_cfg.run_packb);
        t_cfg.wait_packb();
        x
    }
   unsafe fn kernel(
       m: usize, n: usize, k: usize,
       alpha: *const A::U,
       beta: *const C::X,
       c: *mut C::X,
       c_rs: usize, c_cs: usize,
       ap: *const A::U, bp: *const B::U,
   );

   unsafe fn kernel_n(
       m: usize, n: usize, k: usize,
       alpha: *const A::U,
       beta: *const C::X,
       c: C,
       ap: *const A::U, bp: *const B::U,
   );
   unsafe fn gemm_packa_packb(
    m: usize, n: usize, k: usize,
    alpha: A::U,
    a: A,
    b: B,
    beta: C::X,
    c: C,
    par: &CorenumPar,
    pool_buf: *mut u8,
) 
where 
A: Send +Sync+'static,
B: Send+Sync+'static,
{
    let mc_eff = Self::get_mc_eff(par.ic_par);
    let nc_eff = Self::get_nc_eff(par.jc_par);
    let kc_eff = Self::KC;
    let ap_pool_size = Self::get_ap_pool_size(par.ic_par);
    let bp_pool_size = Self::get_bp_pool_size(par.jc_par);
    let (ap_ptr, bp_ptr) = get_ap_bp::<A::U,B::U>(pool_buf, ap_pool_size, bp_pool_size, par.ic_par, par.jc_par);
    let ap = a.into_pack_array(ap_ptr);
    let bp = b.into_pack_array(bp_ptr);
    let mut pa_br_vec = Vec::with_capacity(par.ic_par);
    for _ in 0..par.ic_par {
        pa_br_vec.push(Arc::new(Barrier::new(par.jc_par*par.jr_par*par.ir_par)));
    }

    let mut pb_br_vec = Vec::with_capacity(par.jc_par);
    for _ in 0..par.jc_par {
        pb_br_vec.push(Arc::new(Barrier::new(par.ic_par*par.ir_par*par.jr_par)));
    }
    let pa_br_vec_ref = Arc::new(pa_br_vec);
    let pb_br_vec_ref = Arc::new(pb_br_vec);

    let mut handle_vec = Vec::with_capacity(par.num_threads-1);

    // let c 
    for t_id in 1..par.num_threads {
        let mut t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref.clone(), pb_br_vec_ref.clone(), t_id, mc_eff, nc_eff, kc_eff);
        let ic_id = t_cfg.ic_id;
        let jc_id = t_cfg.jc_id;
        let ap_cur = ap.add_p(ic_id*ap_pool_size);
        let bp_cur = bp.add_p(jc_id*bp_pool_size);

        let x = std::thread::spawn(move || {
                let alpha = &alpha as *const A::U;
                let beta = &beta as *const C::X;
                Self::gemm_packa_packb_serial(m, n, k, alpha, ap_cur, bp_cur, beta, c, &mut t_cfg);
            }
        );

        handle_vec.push(x);
    }

    {
        let t_id: usize = 0;
        let t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref, pb_br_vec_ref, t_id, mc_eff, nc_eff, kc_eff);
        let alpha = &alpha as *const A::U;
        let beta = &beta as *const C::X;
        Self::gemm_packa_packb_serial(m, n, k, alpha, ap, bp, beta, c, &t_cfg);
    }
    for handle in handle_vec {
        handle.join().unwrap();
    }
}
   #[inline]
   unsafe fn gemm_packa_packb_serial(
       m: usize, n: usize, k: usize,
       alpha: *const A::U,
       a: A::PackArray,
       b: B::PackArray,
       beta: *const C::X,
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
       let (mc_start, mc_end, mc_left) = split_c_range(m, mc, Self::MR, ic_id, t_cfg.par.ic_par);
       let (nc_start, nc_end, nc_left) = split_c_range(n, nc, Self::NR, jc_id, t_cfg.par.jc_par);
       let (kc_start, kc_end) = (0, k);
       let one = Self::ONE;
       let mut mc_i = mc_start;
       let c_rs = c.rs();
       let c_cs = c.cs();
       let c_ptr = c.data_ptr();
       while mc_i < mc_end {
           let mc_len = mc.min(mc_end - mc_i);
           let (mr_start, mr_end) = split_range(mc_len, Self::MR, ir_id, ir_par);
           let mr_len = mr_end - mr_start;
           let c_i = c_ptr.add((mc_i+mr_start) * c_rs);
           let mut kc_i = kc_start;
           while kc_i < kc_end {
               let kc_len = kc.min(kc_end - kc_i);
               let kc_last = kc_i + kc_len == kc_end;
               let run_nonlinear = kc_last && Activation::IS_IDENTITY;
               let beta_t = if kc_i == kc_start { beta } else { &one as *const C::X};
               let mut nc_i = nc_start;
               let ap = Self::packa(a, mc_i, kc_i, mc_len, kc_len, t_cfg);
               let ap = ap.add(mr_start*kc_len);
               while nc_i < nc_end {
                   let nc_len = nc.min(nc_end - nc_i);
                   let (nr_start, nr_end) = split_range(nc_len, Self::NR, jr_id, jr_par);
                   let nr_len = nr_end - nr_start;
                   let c_ij = c_i.add((nc_i+nr_start) * c_cs);
                   let bp = Self::packb(b, nc_i, kc_i, nc_len, kc_len, t_cfg);
                   let bp = bp.add(nr_start*kc_len);
                   if run_nonlinear {
                        Self::kernel(
                            mr_len, nr_len, kc_len, alpha, beta_t, c_ij, c_rs, c_cs,
                            ap, bp,
                        );
                   } else {
                        let c_t = c.add((mc_i+mr_start) * c_rs + (nc_i+nr_start) * c_cs);
                        Self::kernel_n(
                            mr_len, nr_len, kc_len, alpha, beta_t, c_t,
                            ap, bp,
                        );
                   }

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
            t_cfg.wait_packa();
            t_cfg.wait_packa();
            let mut nc_i = nc_start;
            while nc_i < nc_end {
             t_cfg.wait_packb();
             t_cfg.wait_packb();
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
A: Copy+GemmArray, 
B: Copy+GemmArray,
C: Copy+GemmOut,
> 
where Self: Sized,
Self: GemmPack<A::T, A::U>
{
    const MC: usize; const NC: usize; const KC: usize;
    const MR: usize; const NR: usize;
    const ONE: C::X;
    const IS_L3_SHARED: bool;
    const IS_L2_SHARED: bool;
    const IS_L1_SHARED: bool;
    const CACHELINE_PAD: usize = 256;
    fn get_mc_eff(par: usize) -> usize {
        Self::MC
    }
    fn get_nc_eff(par: usize) -> usize {
        if Self::IS_L2_SHARED {
            Self::NC / par
        } else {
            Self::NC
        }
    }
    fn get_ap_pool_size(ic_par: usize) -> usize {
        let mc_eff = Self::get_mc_eff(ic_par);
        mc_eff * Self::KC + Self::CACHELINE_PAD / std::mem::size_of::<A::U>()
    }
    unsafe fn packa(
        a: A::PackArray,
        mc: usize, kc: usize,
        mc_len: usize, kc_len: usize,
        t_cfg: &CorenumThreadConfig
    ) -> *const A::U {
        t_cfg.wait_packa();
         let x = a.packa_dispatch_hw::<Self>(mc, kc, mc_len, kc_len, 0, t_cfg.run_packa);
        t_cfg.wait_packa();
        x
    }
    unsafe fn kernel(
        m: usize, n: usize, k: usize,
        alpha: *const A::U,
        beta: *const C::X,
        b: B, b_rs: usize, b_cs: usize,
        c: *mut C::X, c_rs: usize, c_cs: usize,
        ap: *const A::U,
    );
    unsafe fn gemm_small_m(
        m: usize, n: usize, k: usize,
        alpha: A::U,
        a: A,
        b: B,
        beta: C::X,
        c: C,
        par: &CorenumPar,
        pack_pool: *mut u8,
    ) 
    where 
    A: Send +Sync+'static,
    B: Send+Sync+'static,
    {
        let mc_eff = Self::get_mc_eff(par.ic_par);
        let nc_eff = Self::get_nc_eff(par.jc_par);
        let kc_eff = Self::KC;
        let ap_pool_size = Self::get_ap_pool_size(par.ic_par);
        let align_offset = pack_pool.align_offset(AP_ALIGN);
        let ap_ptr = (pack_pool.add(align_offset)) as *mut A::U;
        let ap = a.into_pack_array(ap_ptr);
        let mut pa_br_vec = Vec::with_capacity(par.ic_par);
        for _ in 0..par.ic_par {
            pa_br_vec.push(Arc::new(Barrier::new(par.jc_par*par.jr_par*par.ir_par)));
        }
    
        let mut pb_br_vec = Vec::with_capacity(par.jc_par);
        for _ in 0..par.jc_par {
            pb_br_vec.push(Arc::new(Barrier::new(par.ic_par*par.ir_par*par.jr_par)));
        }
        let pa_br_vec_ref = Arc::new(pa_br_vec);
        let pb_br_vec_ref = Arc::new(pb_br_vec);
    
        let mut handle_vec = Vec::with_capacity(par.num_threads-1);
    
        // let c 
        for t_id in 1..par.num_threads {
            let t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref.clone(), pb_br_vec_ref.clone(), t_id, mc_eff, nc_eff, kc_eff);
            let ic_id = t_cfg.ic_id;
            let ap_cur = ap.add_p(ic_id*ap_pool_size);
    
            let x = std::thread::spawn(move || {
                    let alpha = &alpha as *const A::U;
                    let beta = &beta as *const C::X;
                    Self::gemm_small_m_serial(m, n, k, alpha, ap_cur, b, beta, c, &t_cfg);
                }
            );
    
            handle_vec.push(x);
        }
    
        {
            let t_id: usize = 0;
            let t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref, pb_br_vec_ref, t_id, mc_eff, nc_eff, kc_eff);
            let alpha = &alpha as *const A::U;
            let beta = &beta as *const C::X;
            Self::gemm_small_m_serial(m, n, k, alpha, ap, b, beta, c, &t_cfg);
        }
        for handle in handle_vec {
            handle.join().unwrap();
        }
    }
    #[inline]
    unsafe fn gemm_small_m_serial(
        m: usize, n: usize, k: usize,
        alpha: *const A::U,
        a: A::PackArray,
        b: B,
        beta: *const C::X,
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
        let (mc_start, mc_end, _) = split_c_range(m, mc_eff, Self::MR, ic_id, par.ic_par);
        let (nc_start, nc_end, _) = split_c_range(n, nc_eff, Self::NR, jc_id, par.jc_par);
        let (kc_start, kc_end) = (0, k);
        let one = Self::ONE;

        let mut mc = mc_start;
        let mc_end = mc_end;
        let c_rs = c.rs();
        let c_cs = c.cs();
        let c_ptr = c.data_ptr();
        while mc < mc_end {
            let mc_len = mc_eff.min(mc_end - mc);
             let (mr_start, mr_end) = split_range(mc_len, Self::MR, ir_id, ir_par);
 
            let mr_len = mr_end - mr_start;
            let c_i = c_ptr.add((mc+mr_start) * c_rs);
            let mut kc = kc_start;
            while kc < kc_end {
                let kc_len = Self::KC.min(kc_end - kc);
                let beta_t = if kc == kc_start { beta } else { &one as *const C::X};
                let mut nc = nc_start;
                let ap = Self::packa(a, mc, kc, mc_len, kc_len, t_cfg);
                let ap = ap.add(mr_start*kc_len);
                while nc < nc_end {
                    let nc_len = nc_eff.min(nc_end - nc);
                    let (nr_start, nr_end) = split_range(nc_len, Self::NR, jr_id, jr_par);
                    let nr_len = nr_end - nr_start;
                    let c_ij = c_i.add((nc + nr_start) * c_cs);
                    Self::kernel(
                        mr_len, nr_len, kc_len, alpha, beta_t, 
                        b, kc, nc+nr_start,
                        c_ij, c_rs, c_cs,
                        ap
                    );
                    nc += nc_eff;
                }
                kc += Self::KC;
            }
            mc += mc_eff;
        }
    }
 }

 pub trait GemmSmallN<
A: Copy+GemmArray, 
B: Copy+GemmArray,
C: Copy+GemmOut,
 >
where Self: Sized,
Self: GemmPack<B::T, B::U>,
 {
    const MC: usize; const NC: usize; const KC: usize;
    const MR: usize; const NR: usize;
    const IS_L3_SHARED: bool;
    const IS_L2_SHARED: bool;
    const IS_L1_SHARED: bool;
    const CACHELINE_PAD: usize = 256;
    fn get_mc_eff(par: usize) -> usize {
        Self::MC
    }
    fn get_nc_eff(par: usize) -> usize {
        if Self::IS_L2_SHARED {
            (Self::NC / (par*24))*24
        } else {
            Self::NC
        }
    }
    fn get_bp_pool_size(jc_par: usize) -> usize {
        let nc_eff = Self::get_nc_eff(jc_par);
        nc_eff * Self::KC + Self::CACHELINE_PAD / std::mem::size_of::<B::U>()
    }
    const ONE: C::X;
    unsafe fn packb(
        b: B::PackArray, 
        nc: usize, kc: usize,
        nc_len: usize, kc_len: usize,
        t_cfg: &CorenumThreadConfig
    ) -> *const B::U {
        t_cfg.wait_packb();
        let x = b.packa_dispatch_hw::<Self>(nc, kc, nc_len, kc_len, 0, t_cfg.run_packb);
        t_cfg.wait_packb();
        x
    }
    unsafe fn kernel(
        m: usize, n: usize, k: usize,
        alpha: *const A::U,
        beta: *const C::X,
        a: A, a_rs: usize, a_cs: usize,
        c: *mut C::X, c_rs: usize, c_cs: usize,
        bp: *const B::U,
    );
    unsafe fn gemm_small_n(
        m: usize, n: usize, k: usize,
        alpha: A::U,
        a: A,
        b: B,
        beta: C::X,
        c: C,
        par: &CorenumPar,
        pack_pool: *mut u8,
    ) 
    where
    A: Send +Sync+'static,
    B: Send+Sync+'static,
    {
        let mc_eff = Self::get_mc_eff(par.ic_par);
        let nc_eff = Self::get_nc_eff(par.jc_par);
        let kc_eff = Self::KC;
        let ap_pool_size = Self::get_bp_pool_size(par.jc_par);
        let align_offset = pack_pool.align_offset(BP_ALIGN);
        let bp_ptr = (pack_pool.add(align_offset)) as *mut B::U;
        let mut b = b;
        b.transpose();
        let bp = b.into_pack_array(bp_ptr);
        let mut pa_br_vec = Vec::with_capacity(par.ic_par);
        for _ in 0..par.ic_par {
            pa_br_vec.push(Arc::new(Barrier::new(par.jc_par*par.jr_par*par.ir_par)));
        }
    
        let mut pb_br_vec = Vec::with_capacity(par.jc_par);
        for _ in 0..par.jc_par {
            pb_br_vec.push(Arc::new(Barrier::new(par.ic_par*par.ir_par*par.jr_par)));
        }
        let pa_br_vec_ref = Arc::new(pa_br_vec);
        let pb_br_vec_ref = Arc::new(pb_br_vec);
    
        let mut handle_vec = Vec::with_capacity(par.num_threads-1);
    
        // let c 
        for t_id in 1..par.num_threads {
            let t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref.clone(), pb_br_vec_ref.clone(), t_id, mc_eff, nc_eff, kc_eff);
            let jc_id = t_cfg.jc_id;
            let bp_cur = bp.add_p(jc_id*ap_pool_size);
    
            let x = std::thread::spawn(move || {
                    let alpha = &alpha as *const A::U;
                    let beta = &beta as *const C::X;
                    Self::gemm_small_n_serial(m, n, k, alpha, a, bp_cur, beta, c, &t_cfg);
                }
            );
    
            handle_vec.push(x);
        }
    
        {
            let t_id: usize = 0;
            let t_cfg = CorenumThreadConfig::new(par.clone(), pa_br_vec_ref, pb_br_vec_ref, t_id, mc_eff, nc_eff, kc_eff);
            let alpha = &alpha as *const A::U;
            let beta = &beta as *const C::X;
            Self::gemm_small_n_serial(m, n, k, alpha, a, bp, beta, c, &t_cfg);
        }
        for handle in handle_vec {
            handle.join().unwrap();
        }
    }
    #[inline]
    unsafe fn gemm_small_n_serial(
        m: usize, n: usize, k: usize,
        alpha: *const A::U,
        a: A,
        b: B::PackArray,
        beta: *const C::X,
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
        let (mc_start, mc_end, mc_left) = split_c_range(m, mc_eff, Self::MR, ic_id, par.ic_par);
        let (nc_start, nc_end, nc_left) = split_c_range(n, nc_eff, Self::NR, jc_id, par.jc_par);
        let (kc_start, kc_end) = (0, k);
        let one = Self::ONE;

        let mut mc = mc_start;
        let mc_end = mc_end;
        let c_rs = c.rs();
        let c_cs = c.cs();
        let c_ptr = c.data_ptr();
        while mc < mc_end {
            let mc_len = mc_eff.min(mc_end - mc);
             let (mr_start, mr_end) = split_range(mc_len, Self::MR, ir_id, ir_par);
 
            let mr_len = mr_end - mr_start;
            let c_i = c_ptr.add((mc+mr_start) * c_rs);
            let mut kc = kc_start;
            while kc < kc_end {
                let kc_len = Self::KC.min(kc_end - kc);
                let beta_t = if kc == kc_start { beta } else { &one as *const C::X};
 
                let mut nc = nc_start;
 
                while nc < nc_end {
                    let nc_len = nc_eff.min(nc_end - nc);
                    let (nr_start, nr_end) = split_range(nc_len, Self::NR, jr_id, jr_par);
                    let nr_len = nr_end - nr_start;
                    let bp = Self::packb(b, nc, kc, nc_len, kc_len, t_cfg).add(nr_start*kc_len);
                    let c_ij = c_i.add((nc + nr_start) * c_cs);
                    Self::kernel(
                        mr_len, nr_len, kc_len, alpha, beta_t, 
                        a, mc+mr_start, kc,
                        c_ij, c_rs, c_cs,
                        bp
                    );
                    nc += nc_eff;
                }
                if nc_left {
                    t_cfg.wait_packb();
                    t_cfg.wait_packb();
                }

                kc += Self::KC;
            }
            mc += mc_eff;
        }
        if mc_left {
            let mut kc = kc_start;
            while kc < kc_end {
                let mut nc = nc_start;
                while nc < nc_end {
                    t_cfg.wait_packb();
                    t_cfg.wait_packb();
                    nc += nc_eff;
                }
                if nc_left {
                    t_cfg.wait_packb();
                    t_cfg.wait_packb();
                }
                kc += Self::KC;
            }
        }
    }
 }
 

pub trait Gemv<
InputA: Copy+GemmArray, 
InputB: Copy+GemmArray,
C: Copy+GemmOut,
> {
   unsafe fn gemv(
       m: usize, n: usize,
       alpha: InputA::U,
       a: InputA,
       x: InputB,
       beta: C::X,
       y: C,
       par: &CorenumPar
   ) {
       let alpha = &alpha as *const InputA::U;
       let beta = &beta as *const C::X;
       let incy = y.rs();
       Self::gemv_serial(m, n, alpha, a, x, beta, y);
   }
   unsafe fn gemv_serial(
       m: usize, n: usize,
       alpha: *const InputA::U,
       a: InputA,
       x: InputB,
       beta: *const C::X,
       y: C
   );
}
