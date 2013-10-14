import multiprocessing as mp
import numpy as np

import magnus_figutils as fig

pool = mp.Pool(8)

scalefactorsA = [0.5, 1, 2, 4, 6, 8, 10, 12, 14]
#scalefactorsA = np.linspace(0,1,11) # for use with the alternative A rule
                                    # that is downscaling the unsconst.
                                    # flow.
pool.map(fig.solve_lin_interpol, scalefactorsA)

scalefactorsB = np.linspace(0, 2.5, 10)
#pool.map(fig.solve_linquant_interpol, scalefactorsB)

quantiles = [0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1]
#pool.map(fig.solve_quant_interpol, quantiles)


