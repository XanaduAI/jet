#!/bin/bash

# Setup Python env for JET paper benchmarks

# 0. Setup python env
python -m venv py_benchenv
source ./py_benchenv/bin/activate
PIP_PKGS=(numpy scipy networkx numexpr opt_einsum kahypar cma jax jaxlib tqdm quimb baytune diskcache scikit-learn igraph jgraph autoray pandas seaborn strawberryfields thewalrus)
for pkg in ${PIP_PKGS[@]};
do
    python -m pip install ${pkg}
done

mkdir -p ext_packages && cd ext_packages
git clone https://github.com/AIworx-Labs/chocolate
python -m pip install -e ./chocolate/

# 1. Cotengra
git clone https://github.com/jcmgray/cotengra
export CTG_ROOT=$PWD/cotengra
python -m pip install -e ${CTG_ROOT}
export NUMBA_NUM_THREADS=6

# 2. ACQDP
git clone https://github.com/alibaba/acqdp
sed -i 's/open_indices = \[0, 1, 2, 3, 4, 5\]/open_indices = \[\]/g' ./acqdp/examples/circuit_simulation.py
sed -i 's/num_samps = 5/num_samps = 1/g' ./acqdp/examples/circuit_simulation.py
python -m pip install -e ./acqdp
