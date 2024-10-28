#!/bin/bash

env_dir=$1
export GLAR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export GLAR_MKL_PATH=$env_dir/lib
export GLAR_BLIS_PATH=/home/ubuntu/blis/lib/armsve
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GLAR_MKL_PATH:$GLAR_BLIS_PATH
# export GLAR_SGEMM_NC=
# export GLAR_SGEMM_KC=
# export GLAR_SGEMM_MR=
# export GLAR_SGEMM_NR=