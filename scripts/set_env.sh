#!/bin/bash

export GLARE_MKL_PATH=/home/ubuntu/.env/lib
export GLARE_BLIS_PATH=/home/ubuntu/blis/lib/haswell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GLARE_MKL_PATH:$GLARE_BLIS_PATH
# export GLARE_SGEMM_NC=
# export GLARE_SGEMM_KC=
# export GLARE_SGEMM_MR=
# export GLARE_SGEMM_NR=