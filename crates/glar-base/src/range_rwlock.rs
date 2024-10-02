use std::cell::UnsafeCell;
use std::sync::Mutex;

/// # RangeLockResult
/// The result of a range locking attempt
pub enum RangeLockResult<Guard> {
    Ok(Guard),
    RangeConflict,
    BadRange,
    OtherError,
}

impl<Guard> RangeLockResult<Guard> {
    pub fn unwrap(self) -> Guard {
        match self {
            RangeLockResult::Ok(guard) => guard,
            RangeLockResult::RangeConflict => panic!("RangeConflict Error"),
            RangeLockResult::BadRange => panic!("BadRange"),
            RangeLockResult::OtherError => panic!("OtherError"),
        }
    }
}

/// make sure the RangeLock can be shared between threads
unsafe impl<'a, T> Sync for RangeLock<'a, T> {}

pub struct RangeLockWriteGuard<'lock, 'a, T: 'lock> {
    rlock: &'lock RangeLock<'a, T>,
    idx: usize,
    start: usize,
    end: usize,
}

impl<'lock, 'a, T> RangeLockWriteGuard<'lock, 'a, T>
where
    T: 'lock,
{
    pub fn get(&self) -> &mut [T] {
        &mut self.rlock.data_mut()[self.start..self.end]
    }

    pub fn change_kc(&self, kc: usize) {
        self.rlock.change_kc(kc);
    }
}

impl<'a, 'lock, T> Drop for RangeLockWriteGuard<'a, 'lock, T>
where
    T: 'lock,
{
    fn drop(&mut self) {
        self.rlock.remove_write(self.idx);
    }
}

pub struct RangeLockReadGuard<'lock, 'a, T: 'lock> {
    rlock: &'lock RangeLock<'a, T>,
}

impl<'lock, 'a, T> RangeLockReadGuard<'lock, 'a, T>
where
    T: 'lock,
{
    pub fn get(&self) -> &[T] {
        self.rlock.data()
    }
}

impl<'a, 'lock, T> Drop for RangeLockReadGuard<'a, 'lock, T>
where
    T: 'lock,
{
    fn drop(&mut self) {
        self.rlock.remove_read();
    }
}

// variation of rwlock where
// write access has also idx features to subslices with len n see struct field
// read access is to the entire slice
// this is has the least complexity and enough # of features to fit my purpose

/// # RangeLock
/// Allows multiple immutable and mutable borrows based on access ranges.
pub struct RangeLock<'a, T> {
    n: usize,
    mc_chunk_len: usize,
    ranges: Mutex<(Vec<bool>, usize, usize)>,
    data: UnsafeCell<&'a mut [T]>,
}

impl<'a, T> RangeLock<'a, T> {
    pub fn from(data: &'a mut [T], n: usize, mc: usize, kc: usize, mr: usize) -> Self {
        let pool_size = mc * kc;
        assert!(pool_size <= data.len(), "pool_size: {}, data.len(): {}", pool_size, data.len());
        let mc_chunk_len = ((mc + n * mr - 1) / (n * mr)) * mr;
        let ranges = Mutex::new((vec![false; n], 0, kc));
        RangeLock { n, mc_chunk_len, ranges, data: UnsafeCell::new(data) }
    }

    pub fn get_mc(&self) -> usize {
        self.mc_chunk_len
    }

    pub fn change_kc(&self, kc: usize) {
        let mut x = self.ranges.lock().unwrap();
        assert!(x.1 == 0, "read mode is on, cannot change kc: {}", x.1);
        x.2 = kc;
    }

    pub fn len(&self) -> usize {
        unsafe { (*self.data.get()).len() }
    }

    /// get a reference to the data
    fn data(&self) -> &[T] {
        unsafe { *self.data.get() }
    }

    /// get a mutable reference to the data
    fn data_mut(&self) -> &mut [T] {
        unsafe { *self.data.get() }
    }

    pub fn read(&self) -> RangeLockResult<RangeLockReadGuard<'a, '_, T>> {
        let mut x = self.ranges.lock().unwrap();
        let ranges = &mut x.0;
        // if ranges is not empty, then there is a conflict
        if ranges.iter().any(|&x| x) {
            return RangeLockResult::RangeConflict;
        }
        let read_mode = &mut x.1;
        // println!("reading read_mode: {}", *read_mode);
        *read_mode += 1;
        RangeLockResult::Ok(RangeLockReadGuard { rlock: &self })
    }

    pub fn write(&self, idx: usize, kc: usize) -> RangeLockResult<RangeLockWriteGuard<'a, '_, T>> {
        if idx > self.n {
            return RangeLockResult::BadRange;
        }
        let mut x = self.ranges.lock().unwrap();
        let is_occupied = &x.0;
        let read_mode = &x.1;
        let chunk_len = self.mc_chunk_len * kc;

        // TODO: add check for kc_len stays the same from write->write
        // it is fine to use contains since len of ranges is small ( ~ num threads / ic_par or jc_par)
        // on average 2-4
        // conflict if the idx is already in ranges or read_mode is on
        if is_occupied[idx] || *read_mode > 0 {
            return RangeLockResult::RangeConflict;
        }
        let is_occupied = &mut x.0;
        is_occupied[idx] = true;

        let (start, end) = (idx * chunk_len, ((idx + 1) * chunk_len).min(self.len()));
        RangeLockResult::Ok(RangeLockWriteGuard { rlock: &self, idx, start, end })
    }

    fn remove_write(&self, idx: usize) {
        let mut x = self.ranges.lock().unwrap();
        let ranges = &mut x.0;
        ranges[idx] = false;
    }

    fn remove_read(&self) {
        let mut x = self.ranges.lock().unwrap();
        let read_mode = &mut x.1;
        *read_mode -= 1;
    }
}

// mod test {
//     use super::*;
//     // #[test]
//     fn range_lock_read_test() {
//         let mut data: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
//         let lock = RangeLock::from(&mut data, 2);
//         assert!(lock.data()[0..3] == [0, 1, 2])
//     }

//     // #[test]
//     fn range_lock_write_test() {
//         let mut data: Vec<usize> = vec![0usize; 12];
//         let lock = RangeLock::from(&mut data, 4);
//         {
//             let guard = lock.write(0).unwrap();
//             let guard_ref = guard.get();

//             guard_ref[0] = 2;
//             guard_ref[1] = 1;
//             guard_ref[2] = 0;
//         }
//         assert!(data[0..3] == [2, 1, 0])
//     }

//     // #[test]
//     fn range_lock_write_test_mt() {
//         use glar_dev::random_matrix_uniform;
//         // create vec of random integers
//         let n_thread = 4;
//         let chunk_num = 4;
//         let vec_len = 12;
//         let chunk_len = vec_len / chunk_num;
//         let mut data_0 = vec![0i32; vec_len];
//         random_matrix_uniform(vec_len, 1, &mut data_0, vec_len);

//         // lock for the data_0
//         let lock0 = RangeLock::from(&mut data_0, n_thread);
//         let lock0_r = &lock0;

//         let mut data = vec![0i32; vec_len];
//         let lock = RangeLock::from(&mut data, n_thread);
//         let lock_r = &lock;
//         use std::thread;

//         thread::scope(|s| {
//             for i in 0..n_thread {
//                 s.spawn(move || {
//                     let guard = lock_r.write(i).unwrap();
//                     let data_slice = guard.get();
//                     let guard0 = lock0_r.read().unwrap();
//                     let data_slice0 = guard0.get();
//                     let offset = i * chunk_len;
//                     for j in 0..chunk_len {
//                         data_slice[j] = data_slice0[j + offset];
//                         std::thread::sleep(std::time::Duration::from_secs(1));
//                     }
//                 });
//             }
//         });

//         let n_thread_r = 13;
//         thread::scope(|s| {
//             for _ in 0..n_thread_r {
//                 s.spawn(move || {
//                     let guard = lock_r.read().unwrap();
//                     let data_slice = guard.get();
//                     let guard0 = lock0_r.read().unwrap();
//                     let data_slice0 = guard0.get();

//                     assert!(data_slice[0..12] == data_slice0[0..12]);
//                 });
//             }
//         });

//     }
// }
