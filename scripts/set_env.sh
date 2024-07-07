#!/bin/bash

export CORENUM_MKL_PATH=/tmp/corenum/.env/lib
export CORENUM_BLIS_PATH=/opt/blis/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CORENUM_MKL_PATH:$CORENUM_BLIS_PATH
# export CORENUM_SGEMM_NC=
# export CORENUM_SGEMM_KC=
# export CORENUM_SGEMM_MR=
# export CORENUM_SGEMM_NR=