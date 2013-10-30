import numpy as np
import multiprocessing as mp

import aurespf.solvers as au
from worldgrid import world_Nodes

layouts = ['w', 'EU_RU', 'EU_NA', 'EU_ME', 'EU_RU_NA_ME', 'CN_JK',
            'CN_SE', 'CN_SE_JK'] # w means the whole world


def solve_copper(layout):
    admat = ''.join(['./settings/', layout, 'admat.txt'])
    nodes = world_Nodes(admat=admat)
    solved_nodes, flows = au.solve(nodes, copper=1, mode='square')
    filename = ''.join([layout, '_aHE_copper_sqr'])
    solved_nodes.save_nodes(filename)
    np.save('./results/' + filename + '_flows', flows)

pool = mp.Pool(mp.cpu_count())

pool.map(solve_copper, layouts)



