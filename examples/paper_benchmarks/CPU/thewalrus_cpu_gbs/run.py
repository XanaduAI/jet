import strawberryfields as sf
from strawberryfields import ops
from random import random
from thewalrus.quantum import pure_state_amplitude
import numpy as np
import time
import sys, os

if len(sys.argv) < 3:
    print("Please provide a file to load and option for random value.")
    print("Run script as python walrus.py <random=0/1>")
    exit(1)

filename = sys.argv[1]

if not os.path.isfile(filename):
    print(f"File {filename} is not valid.")
    exit(1)

prog = sf.load(filename)
eng = sf.Engine("gaussian")
result = eng.run(prog)
cov = result.state.cov()
lw = 8

amp = np.zeros(lw * lw).astype(int)

random = int(sys.argv[2])
if random == 1:
    amp=[2,0,2,2,2,0,0,2,0,2,0,2,0,2,2,2,2,2,0,2,0,0,0,2,0,0,0,0,2,0,2,2,2,0,2,2,0,2,0,2,2,2,0,2,0,0,2,2,0,2,2,0,0,2,2,0,0,0,0,0,0,2,2,2]

mu = np.zeros(len(cov))

start = time.time()
ans = pure_state_amplitude(mu,cov,amp,check_purity=False)
end = time.time()

print("t=", end-start)
print("result=", ans)