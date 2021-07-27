#!/bin/bash
#----------------------------------------------------
#SBATCH -J jet_gbs_runs           # Job name
#SBATCH -N 1               # Total # of nodes
#SBATCH -n 40             # Total # of mpi tasks
#SBATCH -t 5:00:00        # Run time (hh:mm:ss)
#SBATCH -p compute
#SBATCH --account soscip-3-044
# Other commands must follow all #SBATCH directives...

module load intel
cd $SLURM_SUBMIT_DIR 
export OMP_NUM_THREADS=8
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock8_total0_0.kraken.jet 2 >> 8_0_tf2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock8_total60_0.kraken.jet 2 >> 8_60_tf2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock4_total20_0.kraken.jet 2 >> 4_20_tf2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock8_total40_0.kraken.jet 2 >> 8_40_tf2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock8_total20_0.kraken.jet 2 >> 8_20_tf2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock4_total60_0.kraken.jet 2 >> 4_60_tf2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock4_total50_0.kraken.jet 2 >> 4_50_tf2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock8_total10_0.kraken.jet 2 >> 8_10_tf2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock8_total30_0.kraken.jet 2 >> 8_30_tf2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock4_total30_0.kraken.jet 2 >> 4_30_tf2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock8_total50_0.kraken.jet 2 >> 8_50_tf2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock4_total0_0.kraken.jet 2 >> 4_0_2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock4_total40_0.kraken.jet 2 >> 4_40_2.out
./jet_full ./gbs_dim2_nc1_lw8_rp5_fock4_total10_0.kraken.jet 2 >> 4_10_2.out
