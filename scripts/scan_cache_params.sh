#!/bin/bash

# Define arrays of literal values
nc_array=(128 192 256 320 320 448 512 578 640 768 960)
kc_array=(128 192 256 320 320 448 512 578 640 768 960)

mr=24
nr=4

export CORENUM_SGEMM_MR=$mr
export CORENUM_SGEMM_NR=$nr

# Outer loop
for n_i in "${nc_array[@]}"; do
    # Inner loop
    for k_i in "${kc_array[@]}"; do
        # Set environment variables x and y
        export NC=$n_i
        export KC=$k_i
        
        # echo nc and kc to out.txt
        echo "nc: $NC, kc: $KC" >> out.txt
        # run ./target/release/corenum-gemm-f32 and output to a out.txt using pipe
        ./target/release/corenum-gemm-f32 --m 4800 --n 2400 --k 2400 --t-layout nt >> out2.txt
        sleep 1
    done
done