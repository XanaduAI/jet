#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --time=24:00:00
#SBATCH --job-name Jet_m12_sliced
#SBATCH --output=omp_m12_single_output_%j.txt
#SBATCH --mail-type=FAIL
 
cd $SLURM_SUBMIT_DIR
module load intel cmake

num_runs=10
max_slices=6

export OMP_PROC_BIND=false 
export OMP_PLACES=cores 
export OMP_DYNAMIC=false 
export OMP_SCHEDULE=static
export OMP_NUM_THREADS=8
export TASKFLOW_THREADS=4

# JET m=12 sliced with 2^sl different slices
for sl in 9; do
    for i in $(seq 1 $num_runs); do
        echo "### RUN=${i} ###" >> jet_sliced_m12_omp${th}_pthread${p}_slice${sl}.out;
        OMP_NUM_THREADS=${th} ./jet_m12_sliced ../data_files/m12.json ${p} ${sl} >> jet_sliced_m12_omp${th}_pthread${p}_slice${sl}.out;
    done
done

# Format data as CSV
touch jet_m12_full_slice_data.csv

t_samples=""
for i in $(seq 0 $(($num_runs-1))); do t_samples+="t${i},"; done

echo "OMP,PTHREAD,${t_samples}" | sed 's/.$//' >> jet_m12_full_slice_data.csv
for i in $(ls -trah | grep jet_sliced_m12_omp[1-9]); 
do 
    dat_line=$(echo $i | sed "s/omp//" | sed "s/pthread//" |  sed "s/.out//" | awk '{split($0,a,"_"); print a[4]","a[5]};' | tr -s '\n' ',');
    dat_line+=$(cat $i | grep "t=[0-9]" | tr '\n' ',' | sed "s/t\=//g" | sed "s/s//g" | sed 's/.$//');
    dat_line+=$(echo "");
    echo $dat_line >> jet_m12_full_slice_data.csv
done
