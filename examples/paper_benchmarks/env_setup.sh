#!/bin/bash

# Setup Python env for JET paper benchmarks

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

git clone https://github.com/jcmgray/cotengra
export CTG_ROOT=$PWD/cotengra
python -m pip install -e ${CTG_ROOT}
export NUMBA_NUM_THREADS=6
