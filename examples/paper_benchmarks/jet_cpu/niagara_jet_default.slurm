#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=jet_m10_full
#SBATCH --tasks-per-node=80

module load intel/2019u4 cmake

export num_runs=10

export OMP_PROC_BIND=false 
export OMP_PLACES=cores 
export OMP_DYNAMIC=false 
export OMP_SCHEDULE=static 

# JET: full network
for p in 1 2 4 8 16; do
    for th in 1 2 4 8 16 32; do
        for i in $(seq 1 $num_runs); do
            echo "### RUN=${i} ###" >> jet_omp${th}_pthread${p}.out;
            OMP_NUM_THREADS=${th} ./tbcc_bm ./m10_slowdown.json ${p} >> jet_omp${th}_pthread${p}.out
        done
    done
done