#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --job-name=cot_m10
#SBATCH --ntasks=80

cd $SLURM_SUBMIT_DIR

module load cmake python intel
source py_benchenv/bin/activate

export num_runs=10

export OMP_PROC_BIND=false 
export OMP_PLACES=cores 
export OMP_DYNAMIC=false 
export OMP_SCHEDULE=static 

for th in 1 2 4 8 16 32 64; do
for i in $(seq 1 $num_runs); do
    echo "### RUN=${i} ###" >> cotengra_omp${th}.out;
    OMP_NUM_THREADS=${th} python ./cotengra_m10.py ${p} ${sl} >> cotengra_omp${th}.out;
done
done
