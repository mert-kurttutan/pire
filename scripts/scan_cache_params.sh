#!/bin/bash

# Check if enough arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 number1 number2"
    exit 1
fi

# Define arrays of literal values
nc_array=(192 256 320 384 448 512 640 704 768)
kc_array=(384 512 576 640 704 768 832 960 1024)

# Outer loop
for n_i in "${nc_array[@]}"; do
    # Inner loop
    for k_i in "${kc_array[@]}"; do
        # Set environment variables x and y
        export PIRE_NC=$n_i
        export PIRE_KC=$k_i
        
        # echo nc and kc to out.txt
        echo "nc: $PIRE_NC, kc: $PIRE_KC" >> out.txt
        ./target/release/bench --m $1 --n $1 --k $1 --t-layout nt --bench-type $3 --backend pire >> out.txt
        ./target/release/bench --m $1 --n $1 --k $1 --t-layout nt --bench-type $3 --backend mkl >> out.txt
        echo "-----------------------------------------" >> out.txt
        sleep 1
    done
done