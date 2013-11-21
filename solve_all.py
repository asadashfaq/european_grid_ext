import multiprocessing as mp
import FlowCalculation as fc

from solve_flows import solve_flow

layouts = ['w', 'EU_RU', 'EU_NA', 'EU_ME', 'EU_RU_NA_ME', 'CN_JK',
           'CN_SE', 'CN_SE_JK']
alphas = ['aHE']
capacities = ['q99', 'hq99']
modes = ['lin', 'sqr']

flow_calcs = []
for L in layouts:
    for a in alphas:
        for c in capacities:
            for m in modes:
                flow_calcs.append(fc.FlowCalculation(L, a, c, m))

pool = mp.Pool(mp.cpu_count())
pool.map(solve_flow, flow_calcs)

