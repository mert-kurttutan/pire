Use GemmArray with associated type
SInce GemmArray is uniquely implemented for each struct
e.g. 
StirdedMatrix<f32> & StridedMatrix<f64> and MixedStrided<f32,f64> are all difrferent instances of GemmArray


avx wihthoiut fma => use 16x4 ukernel since it requires additional register accumulate axb coming from vmulps a, b