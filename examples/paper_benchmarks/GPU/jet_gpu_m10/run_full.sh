#!/bin/bash

## It is essential to have CuTensor >= v1.3.0 available
## on the system you wish to run this job.

export num_runs=10

export OMP_PROC_BIND=false
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_SCHEDULE=static
export CUDA_VISIBLE_DEVICES=2

# Warm-up to cache library loads; seems needed
OMP_NUM_THREADS=1 ./jet_cuda_full ../../data_files/m10.json 1 > /dev/null

# We restrict to slicing 5 edges at a minimum to avoid hitting the memory wall of the device.

    for p in 1 2 4; do
        for th in 1 2 4; do
            for i in $(seq 1 $num_runs); do
                echo "### RUN=${i} ###" >> jet_cuda_full_omp${th}_pthread${p}_slice${sl}.out;
                OMP_NUM_THREADS=${th} ./jet_cuda_full ../../data_files/m10.json ${p} ${sl} >> jet_cuda_full_omp${th}_pthread${p}.out
            done
        done
    done

# Format data as CSV
touch jet_full_data.csv

t_samples=""
for i in $(seq 0 $(($num_runs-1))); do t_samples+="t${i},"; done

echo "OMP,PTHREAD,SLICE,${t_samples}" | sed 's/.$//' >> jet_full_data.csv
for i in $(ls -trah | grep jet_cuda_full_omp[1-9]);
do
    dat_line=$(echo $i | sed "s/omp//" | sed "s/pthread//" | sed "s/.out//" | awk '{split($0,a,"_"); print a[4]","a[5]};' | tr -s '\n' ',');
    dat_line+=$(cat $i | grep "t=[0-9]" | tr '\n' ',' | sed "s/t\=//g" | sed "s/s//g" | sed 's/.$//');
    dat_line+=$(echo "");
    echo $dat_line >> jet_full_data.csv
done
