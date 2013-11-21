import numpy as np
import multiprocessing as mp
from solve_flows import solve_flow
import FlowCalculation as fc

modes = ['lin', 'sqr']
alphas = ['aHO0', 'aHO1']

fc_list = []

for m in modes:
    for a in alphas:
        fc_list.append(fc.FlowCalculation('w', a, 'copper', m))
        print fc_list[-1]

pool = mp.Pool(mp.cpu_count())
pool.map(solve_flow, fc_list)


