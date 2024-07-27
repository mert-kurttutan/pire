- Move definition of mask_ptr inside ukernel, use m as input instead of mask_ptr, test if this has any perf cost
- This api is more convenient since api of ukernel will be the same across isa (since different isa(-extensiosn) have different ways to define/use mask_ptr)

- mkl batched api does not perform any better than gemm+external batching (even for small dim), investigaet this further (only for small dim since big dim cannot exceed this)