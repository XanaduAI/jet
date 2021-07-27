#!/bin/bash

export OMP_PROC_BIND=false 
export OMP_PLACES=cores 
export OMP_DYNAMIC=false 
export OMP_SCHEDULE=static 
export CUDA_VISIBLE_DEVICES=0

#Cotengra 
for i in $(seq 1 1); do
    for th in 24; do
        echo "### RUN=${i} ###" >> cot_omp${th}.out;
        OMP_NUM_THREADS=${th} python sliced.py --job_rank 0 --file_name $CTG_ROOT/examples/circuit_n53_m10_s0_e0_pABCDCDAB.qsim --simplify_string="R" --search_time=3600 --save_suffix="$2_" --swap_trick=False >> cot_omp${th}.out
    done
done

