
# Pire: Package for high performance cpu kernels
State of the art gemm-like kernels for cpus (using inline assembly)

Include quantized gemm, sgemm, hgemm, dgemm, integer gemm.

Working on putting more kernels used in many LLMs.

Features:
- packed api for matrices
- gemm+unary function fusion

State of the Art:
- The same performance as MKL within 1% performance, you can check benchmark directory
- Needed some optimization for layout with tn, nt (dot kernel in blis terminology)


Why this, not blis, or openblas?

- I wanted to write somehting on my own so I can explore the path towards state of the art faster.
I also wanted to write something in Rust since Rust had two features I liked very much, run time dispatch features target
with  `#[target_feature(enable = "avx,avx2,fma")]` and its inline assembly combined with its macro system seemed very convenient to write hand optimized assembly code for 
gemm kernels.
- Those projects also dont offer several features I wanted: packed interface, unary function fusion, integer gemm, inline assembly for llm quantized gemm kernels (to be worked)

Another way to look at this project is to have collection most optimized cpu kernels for their respective gemm ops so that it can be used as a reference for high performance numerical jit kernels.
