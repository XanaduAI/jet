The scripts in this directory allow a user to replicate the benchmarking results included in the Jet paper [link]().

To build all Jet examples, we use the following CMake commands to prepare the compilation:

```bash
# CPU-only build
cmake . -DENABLE_OPENMP=1 -DENABLE_NATIVE=1 -DENABLE_WARNINGS=0 -DCMAKE_BUILD_TYPE=Release

#GPU-build
cmake . -DENABLE_OPENMP=1 -DENABLE_NATIVE=1 -DENABLE_WARNINGS=0 -DCMAKE_BUILD_TYPE=Release -DENABLE_CUTENSOR=1

# Compile
make -j4
```

# Jet m=10 full network and sliced network
The examples in `jet_cpu_m10` and the associated SLURM submission scripts for Niagara were used to collect the data of the full network (default) simulations, and the sliced (shared work) simulations for the m=10 Sycamore circuit. We sweep over the OpenMP and the Taskflow (Pthread) thread-spaces to find the optimal parameters, and ensure we are utilizing the node's resources at maximum capacity.

# Cotengra m=10 full network
The associated runs with Cotengra for the full-network m=10 comparison can be found in `cotengra_cpu_m10`. All required dependencies are specified in the given `requirements.txt`file, and `env_setup.sh` will create a runnable Python environment with these installed.

To run the same comparison for the full network with Cotengra, we fix the path (provided in the Python file) to the same as used for Jet. Additionally, we sweep the OpenMP parameter space as several backend packages of Cotengra can gain performance from these.

# Jet m=12 single network slice
The examples in `jet_cpu_m12` and the associated SLURM submission scripts for Niagara were used to collect data for the much larger m=12 circuit. Given that the m=12 circuit requires significantly more RAM than a consumer workstation, we instead opt to slice 7,8, and 9 nodes of the network respectively, partitioning it into 64, 128, and 512 slices for each instance. For this example, we showcase the contraction of a single slice (`jet_sliced_single.cpp`).

The comparative GPU example can be found in `jet_gpu_m12` (`jet_sliced_single.cpp`).

For brevity, we include an example to contract all slices of the network for both instances (`jet_sliced_all.cpp`).


# Plot figures
The submission scripts will aggregate the runtimes into CSV files for each set of run-type. These CSV files can be plotted using the given Python environment created with `env_setup.sh`.
Two options exist for plotting: non-sliced data-sets (CSV columns are `OMP,PTHREADS,t0,..,t9`) and sliced data-sets (CSV columns are `OMP,PTHREADS,SLICES,t0,..,t9`).
They can be run respectively using:

```bash
# sliced data
python ./plot_data.py --name=output_filename --csv=path/to/sliced.csv --sliced=y

# non-sliced data
python ./plot_data.py --name=output_filename --csv=path/to/nonsliced.csv --sliced=n
```

The script will create `output_filename.pdf` for non-sliced data, and `output_filename_slice{slice_number}.pdf` for sliced.
