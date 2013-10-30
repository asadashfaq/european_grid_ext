from FlowCalculation import FlowCalculation
def build_flow_calcs():
    layouts = ['w','EU_RU']# 'EU_NA']#, 'EU_ME', 'EU_RU_NA_ME', 'CN_JK',
          # 'CN_SE', 'CN_SE_JK']
    alphas = ['aHE']
    capacities = ['zerotrans', 'q99', 'hq99']#, 'hq99']
    modes = ['lin', 'sqr']

    flow_calcs = []
    for L in layouts:
        for a in alphas:
            for c in capacities:
                for m in modes:
                    flow_calcs.append(FlowCalculation(L, a, c, m))

    return flow_calcs
