#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=thewlarus_gbs
#SBATCH --ntasks-per-node=80

cd $SLURM_SUBMIT_DIR

source $1/bin/activate # sbatch <filename> <py_env_path>

module load intel/2019u4 cmake

export num_runs=1

export OMP_PROC_BIND=false 
export OMP_PLACES=cores 
export OMP_DYNAMIC=false 
export OMP_SCHEDULE=static 

# The Walrus: 8x8 (64 qumodes/qudits) 000.. GBS
for i in $(seq 1 10); do
    for th in 16 32 64; do
        echo "### RUN=${i} ###" >> walrus_omp${th}_r0.out;
        OMP_NUM_THREADS=${th} python ../../data_files/gbs_8x8.xbb 0 >> walrus_omp${th}_r0.out;
    done
done

# The Walrus: 8x8 (64 qumodes/qudits) random GBS
for i in $(seq 1 1); do
    for th in 16 32 64; do
        echo "### RUN=${i} ###" >> walrus_omp${th}_r1.out;
        OMP_NUM_THREADS=${th} python ../../data_files/gbs_8x8.xbb 1 >> walrus_omp${th}_r1.out;
    done
done