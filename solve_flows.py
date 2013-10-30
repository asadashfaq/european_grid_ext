import sys
import numpy as np
import multiprocessing as mp

import aurespf.solvers as au
from worldgrid import world_Nodes
from FlowCalculation import FlowCalculation # my own class for passing info about calculations

def solve_flow(flow_calc):
    admat = ''.join(['./settings/', flow_calc.layout, 'admat.txt'])
    filename = str(flow_calc)
    copperflow_filename = ''.join(['./results/', flow_calc.layout, '_',
        flow_calc.alphas, '_copper_', flow_calc.solvermode, '_flows.npy'])

    if flow_calc.alphas=='aHE':
        nodes = world_Nodes(admat=admat)
    else:
        sys.stderr.write('The object has an distribution of mixes that\
                          is not accounted for.')

    if flow_calc.solvermode=='lin':
        mode = 'linear'
    elif flow_calc.solvermode=='sqr':
        mode = 'square'
    else:
        sys.stderr.write('The solver mode must be "lin" or "sqr"')

    flowfilename = ''.join(['./results/', str(flow_calc), '_flows.npy'])
    print flowfilename
    print flow_calc.capacities
    if flow_calc.capacities=='copper':
        solved_nodes, flows = au.solve(nodes, copper=1, mode=mode)
        print "Checkpt. copper"
    elif flow_calc.capacities=='q99':
        h0 = au.get_quant_caps(filename=copperflow_filename)
        print h0
        print h0.mean()
        solved_nodes, flows = au.solve(nodes, copper=0, h0=h0, mode=mode)
        print "Checkpt. q99"
    elif flow_calc.capacities=='hq99': # corresponds to half the capacities
                                         # of the 99% quantile layout
        h0 = 0.5*au.get_quant_caps(filename=copperflow_filename)
        print h0
        print h0.mean()
        solved_nodes, flows = au.solve(nodes, copper=0, h0=h0, mode=mode)
        print "Checkpt. hq99"
    else:
        sys.stderr.write('The capacities must be either "copper", "q99" or\
                            "hq99"')

    solved_nodes.save_nodes(filename)
    print filename
    try:
        flows
    except NameError:
        print "Flows not defined."
    np.save('./results/' + filename + '_flows', flows)


####### This is where the calculation is actally performedi ######

layouts = ['w', 'EU_RU', 'EU_NA', 'EU_ME', 'EU_RU_NA_ME', 'CN_JK',
           'CN_SE', 'CN_SE_JK']
alphas = ['aHE']
capacities = ['q99', 'hq99']
modes = ['sqr'] # ['lin', 'sqr']

flow_calcs = []
for L in layouts:
    for a in alphas:
        for c in capacities:
            for m in modes:
                flow_calcs.append(FlowCalculation(L, a, c, m))

pool = mp.Pool(mp.cpu_count())
pool.map(solve_flow, flow_calcs)

