Blasfeo paper and blis paper
Use GemmArray with associated type
SInce GemmArray is uniquely implemented for each struct
e.g. 
StirdedMatrix<f32> & StridedMatrix<f64> and MixedStrided<f32,f64> are all difrferent instances of GemmArray


avx wihthoiut fma => use 16x4 ukernel since it requires additional register accumulate axb coming from vmulps a, b

https://stackoverflow.com/questions/53443249/do-all-cpus-which-support-avx2-also-support-sse4-2-and-avx



Small dim opt + packed:

- Cases:
small m: pack a + no pack b => sup_m
small m + prepacked a => no change still use sup_m
small_m + prepacked b => can go with packed api of goto (no sup_m)


split_c_range

m = cr * mr + mr_left

cr = ic_par * cr_ic + cr_left

m = cr_ic * (mr*ic_par) + cr_left*mr + mr_left


d_mi = cr_ic * mr + mr


sup_n: use libshalom, 
- better benchmark than blasfeo's approach (even for x86 hw, even though paper studies only arm hardware)
- simpler inline assembly code and more compatible with packing api
https://jianbinfang.github.io/files/2021-06-22-sc.pdf


neon:

8x12:
2x12+2+3

12x8:
3x8+3+2
12x9:
3x9+3+3

16x6:
4x6+4+2

20x5:
5x5+5+2

24x4:
6x4+6+1









Embedded broadcasting is import for perf of f16 (broadcastw is not on par in performance)

getting compiler to return right assembly for embedded broadcasting is harder in intrinsics