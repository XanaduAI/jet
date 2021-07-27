#!/bin/bash
#----------------------------------------------------
#SBATCH -J gbs_eig           # Job name
#SBATCH -o gbs_%j.o       # Name of stdout output file
#SBATCH -e gbs_%j.e       # Name of stderr error file
#SBATCH -N 1               # Total # of nodes
#SBATCH -n 40             # Total # of mpi tasks
#SBATCH -t 23:00:00        # Run time (hh:mm:ss)
#SBATCH -p compute
#SBATCH --mail-user=trevor@xanadu.ai
#SBATCH --mail-type=all    # Send email at begin and end of job

cd $SLURM_SUBMIT_DIR
module purge && module load NiaEnv/2019b gcc/8.3.0 intelmpi/2019u5 python/3.6.8

source $SCRATCH/.virtualenvs/cotengra/bin/activate

python3 run.py gbs_8x8.xbb 0 > results_0.txt
python3 run.py gbs_8x8.xbb 10 > results_10.txt
python3 run.py gbs_8x8.xbb 20 > results_20.txt
python3 run.py gbs_8x8.xbb 30 > results_30.txt
python3 run.py gbs_8x8.xbb 40 > results_40.txt
python3 run.py gbs_8x8.xbb 50 > results_50.txt
python3 run.py gbs_8x8.xbb 60 > results_60.txt

# ---------------------------------------------------
