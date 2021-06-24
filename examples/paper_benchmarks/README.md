The scripts in this directory allow a user to replicate the benchmarking results included in the JET paper [link]().

Since we compare with Cotengra and 



# JET $$m=10$$ full network and sliced network
The examples in `jet_cpu_m10` and the associated SLURM submission scripts for Niagara were used to collect the data of the full network (default) simulations, and the sliced (shared work) simulations for the $$m=10$$ Sycamore circuit. We sweep over the OpenMP and the Taskflow (Pthread) thread-spaces to find the optimal parameters, and ensure we are utilising the node's resources at maximum capacity.

The associated runs with Cotengra for the above can be found in `cotengra_cpu_m10`.

# Cotengra $$m=10$$ full network and sliced network
To run the same comparison for the full network with Cotengra, we fix the path (provided in the Python file) to the same as used for JET. Additionally, we sweep the OpenMP parameter space as several backend packages of Cotengra can gain performance from these.

For the slicing example, there is no native concurrent execution built into Cotengra for multiple slices, and so we avoid setting up instances to exploit this, instead providing simulations with native behaviour only.

# JET $$m=12$$ single network slice
Given the $$m=12$$ Sycamore circuit requires significantly more RAM than a consumer workstation, we instead opt to slice 9 nodes of the network, partitioning it into 512 slices. For this example, we showcase the contract of a single slice, and extrapolate the time required for the full network taking account of all slices.

Next, by running a sample of 10 such slices, we can get an estimation of how much benefit our shared-work model provides over the default contractions.