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
        ./target/release/bench --m 6000 --n 6000 --k 6000 --t-layout nt >> out.txt
        sleep 1
    done
done