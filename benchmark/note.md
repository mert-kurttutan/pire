- intel mkl cblas_hgemm wrong result for 
```
m = 33, n = 1, k = 2 
col_major, notrans, notrans
lda = 33, ldc = 33, ldb = 2

m: 38, n: 1, k: 8,
col_major, notrans, notrans

 m: 1, n: 64, k: 8, alpha: 1, beta: 1
 col_major, notrans, trans


 m: 1, n: 64, k: 8, alpha: 1, beta: 
  col_major, trans, trans
```

(also for other k values 8 and m e.g. 37)


used pip to isntall
intel-openmp 2024.1.2
mkl          2024.1.0
cpu info
CPU: 11th Gen Intel(R) Core(TM) i7-11850H @ 2.50GHz
os: windows 10


maybe: We need to set omp_num_threads=4, but I am sure it gives incorrect result for omp_num_threads
but not sure aobut omp_num_threads=1


the following is slower than mkl and it is not slower when tn is used
 .\target\release\bench.exe --m 4800 --n 91 --k 4800 --t-layout nn --bench-type sgemm --backend glar

This is probably because of packing used in a regardless of whether it is transposed or not.
We can easily fix this but more code

cblas_hgemm on intel sde on the same machine as above gives (exit code: 0xc0000005 STATUS_ACCESS_VIOLATION)
MKL_NUM_THREADS=1

but not on windows ec2 saphhier rapid machine from aws



rust gemm bug, on aarch64 platform

./target/release/bench --m 3600 --n 1000 --k 3600 --bench-type sgemm --backend rustgemm --alpha 3.1 --check