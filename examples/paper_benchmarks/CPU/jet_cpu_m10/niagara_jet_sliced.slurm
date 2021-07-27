#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=jet_m10_sliced
#SBATCH --ntasks-per-node=80

cd $SLURM_SUBMIT_DIR

module load intel/2019u4 cmake

export num_runs=10

export OMP_PROC_BIND=false 
export OMP_PLACES=cores 
export OMP_DYNAMIC=false 
export OMP_SCHEDULE=static 

# JET: sliced network simulations (2, 4, 6 slices)
for sl in 2 4 6; do
    for p in 1 2 4 8 16; do
        for th in 1 2 4 8 16 32 ; do
            for i in $(seq 1 $num_runs); do
                echo "### RUN=${i} ###" >> jet_shared_omp${th}_pthread${p}_slice${sl}.out;
                OMP_NUM_THREADS=${th} ./jet_sliced ../data_files/m10.json ${p} ${sl} >> jet_shared_omp${th}_pthread${p}_slice${sl}.out
            done
        done
    done
done

# Format data as CSV
touch jet_sliced_data.csv

t_samples=""
for i in $(seq 0 $(($num_runs-1))); do t_samples+="t${i},"; done

echo "OMP,PTHREAD,SLICES,${t_samples}" | sed 's/.$//' >> jet_sliced_data.csv
for i in $(ls -trah | grep jet_shared_omp[1-9]); 
do 
    dat_line=$(echo $i | sed "s/omp//" | sed "s/pthread//" | sed "s/slice//" | sed "s/.out//" | awk '{split($0,a,"_"); print a[3]","a[4]","a[5]};' | tr -s '\n' ',');
    dat_line+=$(cat $i | grep "t=[0-9]" | tr '\n' ',' | sed "s/t\=//g" | sed "s/s//g" | sed 's/.$//');
    dat_line+=$(echo "");
    echo $dat_line >> jet_sliced_data.csv
done
