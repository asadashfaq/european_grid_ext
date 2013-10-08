#! /usr/bin/env python
import numpy as np
import aurespf.solvers as au

alphas = [0.674268, 0.795557, 0.716455, 0.68291, 0.755273, 0.849023, 0.701074, 0.787354, 0.739453, 0.659473, 0.641748, 0.660791, 0.690674, 0.639551, 0.68862, 0.713086, 0.662549, 0.695068, 0.716016, 0.754102, 0.817676, 0.731543, 0.646582, 0.650977, 0.696533, 0.70708, 0.706787, 0.812915, 0.810791, 0.789551]

def EU_Nodes(load_filename=None, full_load=False):
    return au.Nodes(admat='./settings/eadmat.txt', path='./data/', prefix = "ISET_country_", files=['AT.npz', 'FI.npz', 'NL.npz', 'BA.npz', 'FR.npz', 'NO.npz', 'BE.npz', 'GB.npz', 'PL.npz', 'BG.npz', 'GR.npz', 'PT.npz', 'CH.npz', 'HR.npz', 'RO.npz', 'CZ.npz', 'HU.npz', 'RS.npz', 'DE.npz', 'IE.npz', 'SE.npz', 'DK.npz', 'IT.npz', 'SI.npz', 'ES.npz', 'LU.npz', 'SK.npz', 'EE.npz', 'LV.npz', 'LT.npz'], load_filename=load_filename, full_load=full_load, alphas=alphas, gammas=np.ones(30))
