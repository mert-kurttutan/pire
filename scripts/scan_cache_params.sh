#!/bin/bash

# Define arrays of literal values
nc_array=(128 192 256 320 384 448)
kc_array=(128 192 256 320 384 448)

mr=16
nr=4

# export GLARE_SGEMM_MR=$mr
# export GLARE_SGEMM_NR=$nr

# Outer loop
for n_i in "${nc_array[@]}"; do
    # Inner loop
    for k_i in "${kc_array[@]}"; do
        # Set environment variables x and y
        export NC=$n_i
        export KC=$k_i
        
        # echo nc and kc to out.txt
        echo "nc: $NC, kc: $KC" >> out.txt
        taskset -c 0 ./target/release/bench --m 4800 --n 4800 --k 4800 --n-repeats 2 --t-layout nt  --bench-type gemm_s16s16s32 --backend corenum >> out.txt
        sleep 1
    done
done