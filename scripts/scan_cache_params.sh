#!/bin/bash

# Define arrays of literal values
nc_array=(192 256 320 320 448)
kc_array=(128 192 256 320)

mr=24
nr=4

export CORENUM_SGEMM_MR=$mr
export CORENUM_SGEMM_NR=$nr

# Outer loop
for n_i in "${nc_array[@]}"; do
    # Inner loop
    for k_i in "${kc_array[@]}"; do
        # Set environment variables x and y
        export CORENUM_SGEMM_NC=$n_i
        export CORENUM_SGEMM_KC=$k_i
        
        # Echo the concatenated values of x and y
        cargo build -p corenum-gemm-f32 --release --features "corenum"
        cp ./target/release/corenum-gemm-f32 ./corenum-bin/corenum-gemm-f32-native-${mr}x${nr}-${n_i}x${k_i}
    done
done