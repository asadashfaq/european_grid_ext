import fig_utils as fig
import FlowCalculation as fc

modes = ['lin', 'sqr']
alphas = ['aHE', 'aHO1', 'aHO0']

lin_list = []
sqr_list = []

for a in alphas:
    lin_list.append(fc.FlowCalculation('w', a, 'copper', 'lin'))
    sqr_list.append(fc.FlowCalculation('w', a, 'copper', 'sqr'))
    print lin_list[-1], sqr_list[-1]

fig.load_mismatch_hist(['EU'], lin_list, variedparam='alphas', xmax=5)
fig.load_mismatch_hist(['EU'], sqr_list, variedparam='alphas')

