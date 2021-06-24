#!/bin/bash

# Setup Python env for JET paper benchmarks

# 0. Setup python env
python -m venv py_benchenv
source ./py_benchenv/bin/activate
python -m pip install \
    numpy \
    scipy \
    networkx \
    numexpr \
    opt_einsum \
    kahypar \
    cma \
    jax \
    jaxlib \
    tqdm \
    quimb \
    chocolate \
    baytune \
    diskcache \
    scikit-learn \
    igraph \
    jgraph \
    autoray \
    distribution \
    pandas

# 1. Cotengra
python -m pip install git+https://github.com/jcmgray/cotengra#egg=cotengra

# 2. ACQDP
git clone https://github.com/alibaba/acqdp
sed -i 's/open_indices = \[0, 1, 2, 3, 4, 5\]/open_indices = \[\]/g' ./acqdp/examples/circuit_simulation.py
sed -i 's/num_samps = 5/num_samps = 1/g' ./acqdp/examples/circuit_simulation.py
python -m pip install -e ./acqdp
