#!/bin/bash

env_dir=$1
export PIRE_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export PIRE_MKL_PATH=$env_dir/.env/lib
export PIRE_BLIS_PATH=/home/ubuntu/blis/lib/armsve
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PIRE_MKL_PATH:$PIRE_BLIS_PATH
# export PIRE_SGEMM_NC=
# export PIRE_SGEMM_KC=
# export PIRE_SGEMM_MR=
# export PIRE_SGEMM_NR=