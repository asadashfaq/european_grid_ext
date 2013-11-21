import numpy as np
import fig_utils as fig
import FlowCalculation as fc


EU_layouts = ['w', 'EU_RU', 'EU_NA', 'EU_ME', 'EU_RU_NA_ME']

CN_layouts = ['w', 'CN_JK', 'CN_SE', 'CN_SE_JK']
alphas = 'aHE'
capacities = ['zerotrans', 'copper', 'q99', 'hq99']
modes = ['lin', 'sqr']
rois = ['EU', 'CN']


for r in rois:
    if r == 'EU':
        layouts = EU_layouts
    if r == 'CN':
        layouts = CN_layouts

######### Layouts ######
    for m in modes:
        for c in capacities:
            fig.mismatch_hist_layout(r, alphas, c, m)
######## Capacities #####
    for m in modes:
        for L in layouts:
            fig.mismatch_hist_capacities(r, L, alphas, m)

####### Modes ########
    for L in layouts:
        for c in capacities:
            fig.mismatch_hist_solvermode(r, L, alphas, c)
