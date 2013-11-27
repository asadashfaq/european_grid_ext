import FlowCalculation as fc
import fig_utils as fig

## World
config_linw = [\
            fc.FlowCalculation('EU', 'aHE', 'zerotrans', 'raw'),
            fc.FlowCalculation('w', 'aHE', 'copper', 'lin'),
            fc.FlowCalculation('EU_RU_NA_ME', 'aHE', 'copper', 'lin')
            ]

fig.load_mismatch_hist(['EU'], config_linw, variedparam='layout',
                        no_title=True,
                        path='./results/FigsOfInterest/Figs_v2/',
                        fig_filename="w_mismatch_hist_lin")

config_sqrw = [\
            fc.FlowCalculation('EU', 'aHE', 'zerotrans', 'raw'),
            fc.FlowCalculation('w', 'aHE', 'copper', 'sqr'),
            fc.FlowCalculation('EU_RU_NA_ME', 'aHE', 'copper', 'sqr')
            ]


fig.load_mismatch_hist(['EU'], config_sqrw, variedparam='layout',
                        no_title=True,
                        path='./results/FigsOfInterest/Figs_v2/',
                        fig_filename="w_mismatch_hist_sqr")
