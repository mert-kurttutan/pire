#!/bin/bash

# Define arrays of literal values
nc_array=(128 192 256 320 320 448 512 578 640 768 960)
kc_array=(128 192 256 320 320 448 512 578 640 768 960)

mr=12
nr=4

# export CORENUM_SGEMM_MR=$mr
# export CORENUM_SGEMM_NR=$nr

# Outer loop
for n_i in "${nc_array[@]}"; do
    # Inner loop
    for k_i in "${kc_array[@]}"; do
        # Set environment variables x and y
        export NC=$n_i
        export KC=$k_i
        
        # echo nc and kc to out.txt
        echo "nc: $NC, kc: $KC" >> out.txt
        taskset -c 0 ./target/release/bench --m 4000 --n 4000 --k 4000 --n-repeats 2 --t-layout nn  --bench-type dgemm --backend corenum >> out.txt
        sleep 1
    done
done