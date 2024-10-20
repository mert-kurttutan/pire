tried s16s16s32 on arm with smlal(2). It is not fast enough to implement. E.g., it is slower than sgemm for same dims
Go back to this, if there is a demand

| Hardware      | CPU Features               | Functions Supported                                      |
|---------------|----------------------------|----------------------------------------------------------|
| Haswell | avx,f16c,fma| s,d,c,z,h, s8u8,s16s32 gemm         |
| Skylake | avx512f,f16c| s,d,c,z,h, s8u8,s16s32 gemm         |
| sandy bridge | avx| s,d,c,z, s8u8,s16s32 gemm    (hgemm naive)      |
