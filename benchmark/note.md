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
