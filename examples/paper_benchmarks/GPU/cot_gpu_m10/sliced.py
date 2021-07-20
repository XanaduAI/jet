import argparse, io, math, os, pickle, sys, time, concurrent.futures

import cotengra as ctg
import numpy as np
import opt_einsum
import pandas as pd
import quimb as qu
import quimb.tensor as qtn
import tqdm

ALPHABET_SIZE_ = 52
ALPHABET_ = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
             "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
             "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
             "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def GenerateStringIndex(ind):
    div_id = int(ind/ALPHABET_SIZE_)
    prefix = ALPHABET_[ind % ALPHABET_SIZE_]
    suffix = "" if div_id == 0 else str((div_id) - 1)
    index_char = prefix+suffix
    return index_char

def write_kraken_file(tn,path,sliced_inds,inv_map,filew):
    print(str(opt_einsum.paths.linear_to_ssa(path)).replace("[","{").replace("]","}").replace(")","}").replace("(","{"),file=filew)
    print(sliced_inds,file=filew)
    for i in tn:
        print(str(i._tags).replace("{","[").replace("}","]"),end=' ',file=filew)
        print(str([inv_map[j] for j in i._inds]).replace("(","[").replace(",)",")").replace(")","]"),end=' ',file=filew)
        print(str(i._data.shape).replace("(","[").replace(")","]").replace(",]","]"),end=' ',file=filew)
        tensor_data = "["
        for j in i._data.flatten():
            tensor_data += "(" + str(j.real) + "," + str(j.imag) + ");"
        tensor_data += "]"
        tensor_data = tensor_data.replace(";]","]")
        print(tensor_data,file=filew)

def read_qasm_file(file, swap_trick=True):
    if swap_trick:
        gate_opts={'contract': 'swap-split-gate', 'max_bond': 2}  
    else:
        gate_opts={}
    return qtn.Circuit.from_qasm_file(file, gate_opts=gate_opts)
        
def read_cotengra_file(file_name):
    df = pd.read_csv(file_name, sep=' ', header = None)
    tensors = []
    for i in range(len(df[0])):
        tens_data = df[3][i].replace("[","").replace("]","").replace("'","")
        tens_data = [complex(s) for s in tens_data.split(',')]                
        tens_shape = df[2][i].replace("[","").replace("]","").replace("'","")
        tens_shape = [int(s) for s in tens_shape.split(',')]
        tens_tags = df[0][i].replace("[","").replace("]","").replace("'","")
        tens_tags = [str(s) for s in tens_tags.split(',')]
        tens_inds = df[1][i].replace("[","").replace("]","").replace("'","")
        tens_inds = [str(s) for s in tens_inds.split(',')]
        data = np.array(tens_data).reshape(tens_shape)
        inds = tens_inds
        tags = tens_tags
        tensors.append(qtn.Tensor(data, inds, tags))
    return qtn.TensorNetwork(tensors)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='Search for benchmark contraction paths')
    my_parser.add_argument('--file_name',
                           type=str,
                           help='.cotengra file')

    my_parser.add_argument('--save_suffix',
                           type=str,
                           default="",
                           help='optional suffix for any saved files')

    my_parser.add_argument('--simplify_string',
                           type=str,
                           default="RC",
                           help='Cotengra simplify string')

    my_parser.add_argument('--search_time',
                           type=int,
                           default=30,
                           help='Cotengra search time')

    my_parser.add_argument('--job_rank',
                           type=int,
                           default=0,
                           help='Cotengra job rank')

    my_parser.add_argument('--swap_trick',
                           type=bool,
                           default=True,
                           help='swap_trick')
        
    args = my_parser.parse_args()
    file_name = args.file_name
    save_suffix = args.save_suffix
    job_rank = args.job_rank
    search_time = args.search_time
    simplify_string = args.simplify_string
    swap_trick = args.swap_trick

    print("file_name = " + str(file_name))
    print("save_suffix = " + str(save_suffix))
    print("search_time = " + str(search_time))
    print("simplify_string = " + str(simplify_string))
    print("swap_trick = " + str(swap_trick))


    circ = read_qasm_file(file_name,swap_trick)

    import random as rd
    rd.seed(42)

    bitstring = "".join(rd.choice('01') for _ in range(53))
    print(bitstring)

    # the squeeze removes all size 1 bonds
    psi_sample = qtn.MPS_computational_state(bitstring, tags='PSI_f').squeeze()
    tn = circ.psi & psi_sample

    print("num tensors = " + str(tn.num_tensors))
    print("num indices = " + str(tn.num_indices))
    tn.full_simplify_(simplify_string,output_inds=[])

    print("num tensors after simplify = " + str(tn.num_tensors))
    print("num indices after simplify = " + str(tn.num_indices))
    tn.astype_('complex64')

    opt = ctg.ReusableHyperOptimizer(
          methods=['kahypar','greedy'],
          max_repeats=1_000_000,
          max_time=search_time,
          directory="ctg_path_cache_" + str(job_rank),
          slicing_reconf_opts={
             'target_size': 2**20,
             'forested': True,
             'num_trees': 2,
             'reconf_opts': {
                 'subtree_size': 12,
                 'forested': True,
                 'num_trees': 2,
                 "parallel" : False, #all options here cause failures
             }
         }
    )

    info = tn.contract(all, optimize=opt, get='path-info',output_inds=[])
    symmap = tn.contract(all, optimize=opt, get='symbol-map',output_inds=[])

    print(info)
    print("opts = ")
    print(opt)
    print(vars(opt))

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    base=os.path.basename(file_name)
    kraken_file_name = os.path.splitext(base)[0] + "_" + save_suffix + ".kraken"

    filew=open(kraken_file_name,'w')
    inv_map = {v: k for k, v in symmap.items()}

    counter = 0
    for k, v in inv_map.items():
        inv_map[k] = GenerateStringIndex(counter)
        counter += 1

    write_kraken_file(tn,[],"",inv_map,filew)

    start = time.time()
    res = tn.contract(all,optimize=opt,output_inds=[])
    end = time.time()
    print("res =",res)
    print("no jax time = " + str(end - start))

    start = time.time()
    res = tn.contract(all,optimize=opt,output_inds=[],backend='jax')
    end = time.time()
    print("res =",res)
    print("jax 1 time = " + str(end - start))

    start = time.time()
    res = tn.contract(all,optimize=opt,output_inds=[],backend='jax')
    end = time.time()
    print("res =",res)
    print("jax 2 time = " + str(end - start))


    #slice to 2**20
    sf = ctg.SliceFinder(info, target_size=2**20)
    ix_sl, cost_sl = sf.search(temperature=1.0)
    ix_sl, cost_sl = sf.search(temperature=0.1)
    ix_sl, cost_sl = sf.search(temperature=0.01)
    print(ix_sl,cost_sl)
    arrays = [t.data for t in tn] 
    sc = sf.SlicedContractor(arrays)
    c = 0
    start = time.time()
    for i in tqdm.tqdm(range(0, sc.nslices)):
          c = c + sc.contract_slice(i)
    end = time.time()
    print("c =",c)
    print("20 slice time = " + str(end - start))
    sliced_inds = [symmap[j] for j in ix_sl]
    inv_sliced_inds = [inv_map[j] for j in sliced_inds]
    print("sliced_inds =",sliced_inds)
    print("inv_sliced_inds =",inv_sliced_inds)

    #slice to 2**23
    sf = ctg.SliceFinder(info, target_size=2**23)
    ix_sl, cost_sl = sf.search(temperature=1.0)
    ix_sl, cost_sl = sf.search(temperature=0.1)
    ix_sl, cost_sl = sf.search(temperature=0.01)
    print(ix_sl,cost_sl)
    arrays = [t.data for t in tn] 
    sc = sf.SlicedContractor(arrays)
    c = 0
    start = time.time()
    for i in tqdm.tqdm(range(0, sc.nslices)):
          c = c + sc.contract_slice(i)
    end = time.time()
    print("c =",c)
    print("23 slice time = " + str(end - start))
    sliced_inds = [symmap[j] for j in ix_sl]
    inv_sliced_inds = [inv_map[j] for j in sliced_inds]
    print("sliced_inds =",sliced_inds)
    print("inv_sliced_inds =",inv_sliced_inds)

