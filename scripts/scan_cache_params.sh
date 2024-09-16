#!/bin/bash

# Define arrays of literal values
nc_array=(192 256 320 384 448 512 640 704 768)
kc_array=(384 512 576 640 704 768 832 960 1024)

# Outer loop
for n_i in "${nc_array[@]}"; do
    # Inner loop
    for k_i in "${kc_array[@]}"; do
        # Set environment variables x and y
        export GLARE_NC=$n_i
        export GLARE_KC=$k_i
        
        # echo nc and kc to out.txt
        echo "nc: $GLARE_NC, kc: $GLARE_KC" >> out.txt
        ./target/release/bench --m 4800 --n 4800 --k 4800 --t-layout nt --bench-type dgemm --backend glare >> out.txt
        ./target/release/bench --m 4800 --n 4800 --k 4800 --t-layout nt --bench-type dgemm --backend mkl >> out.txt
        echo "-----------------------------------------" >> out.txt
        sleep 1
    done
done