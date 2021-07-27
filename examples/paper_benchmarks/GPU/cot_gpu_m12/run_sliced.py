import time
import sys

import quimb.tensor as qtn
import cotengra as ctg
import tqdm

from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear

def load_circuit(
    n=53,
    depth=10,
    seed=0 ,
    elided=0,
    sequence='ABCDCDAB',
    swap_trick=False
):
    file = f'circuit_n{n}_m{depth}_s{seed}_e{elided}_p{sequence}.qsim'

    if swap_trick:
        gate_opts={'contract': 'swap-split-gate', 'max_bond': 2}  
    else:
        gate_opts={}
    
    # instantiate the `Circuit` object that 
    # constructs the initial tensor network:
    return qtn.Circuit.from_qasm_file(file, gate_opts=gate_opts)

circ = load_circuit(depth=12, swap_trick=True)
sampler = qtn.MPS_computational_state('0' * (circ.N))
tn = circ.psi & sampler
tn.full_simplify_(output_inds=[])
tn.astype_('complex64')

ctg.hyper._HYPER_SEARCH_SPACE['kahypar']['imbalance']['max'] = 0.1

opt = ctg.HyperOptimizer(
    methods=['kahypar'],
    max_time=120,              # just search for 2 minutes
    max_repeats=1000,
    progbar=True,
    minimize='flops',
    slicing_opts={'target_slices': int(sys.argv[1])}
)

info = tn.contract(all, optimize=opt, get='path-info', output_inds=[])

sf = ctg.SliceFinder(info, target_slices=int(sys.argv[1]))

ix_sl, cost_sl = sf.search(temperature=1.0)
ix_sl, cost_sl = sf.search(temperature=0.1)
ix_sl, cost_sl = sf.search(temperature=0.01)


arrays = [t.data for t in tn] 
sc = sf.SlicedContractor(arrays)

start = time.time()
c = sc.contract_slice(0, backend="jax")
end = time.time()
print(f"t_0(contract_slice[0])={end-start}")
print(f"res_0(contract_slice[0])={c}")

print("#########################################################")

for i in tqdm.tqdm(range(1, sc.nslices)):
    start = time.time()
    c = c + sc.contract_slice(i, backend="jax")
    end = time.time()
    print(f"t_0(contract_slice[{i}])={end-start}")
    print(f"res_0(sum to contract_slice[{i}])={c}")

print("#########################################################")
print("#########################################################")
print("#########################################################")

# second run

tn = circ.psi & qtn.MPS_rand_computational_state(circ.N, seed=42)
tn.full_simplify_(output_inds=[]).astype_('complex64')

# update the SlicedContractor's arrays
sc.arrays = tuple(t.data for t in tn)

# perform the contraction

start = time.time()
c = sc.contract_slice(0, backend="jax")
end = time.time()
print(f"t_0(contract_slice[0])={end-start}")
print(f"res_0(contract_slice[0])={c}")

print("#########################################################")
res=0
for i in tqdm.tqdm(range(sc.nslices)):
    start = time.time()
    res += sc.contract_slice(i, backend="jax")
    end = time.time()
    print(f"t_1(contract_slice[{i}])={end-start}")
    print(f"res_1(contract_slice[{i}])={res}")

# update the SlicedContractor's arrays
sc.arrays = tuple(t.data for t in tn)
print("#########################################################")
# perform the contraction
res=0
for i in tqdm.tqdm(range(sc.nslices)):
    start = time.time()
    res += sc.contract_slice(i, backend="jax")
    end = time.time()
    print(f"t_2(contract_slice[{i}])={end-start}")
    print(f"res_2(contract_slice[{i}])={res}")
