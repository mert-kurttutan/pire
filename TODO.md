- Move definition of mask_ptr inside ukernel, use m as input instead of mask_ptr, test if this has any perf cost
- This api is more convenient since api of ukernel will be the same across isa (since different isa(-extensiosn) have different ways to define/use mask_ptr)

- mkl batched api does not perform any better than gemm+external batching (even for small dim), investigaet this further (only for small dim since big dim cannot exceed this)
- check dispatching of sgemv through sgemm is the same as sgemv in terms of performance. It should be, but a rigourous check is better to be sure. (Only additional overhead should be due to dispathcing of cache params, mc, nc, kc)

-Add test for packing

-Add not about how alpha and beta f32 are used in integer gemm (e.g. at which steps they are converted ot/from i32/f32). This will have impact on end result

-Add note about scaling parameter for quantized gemm and their precision