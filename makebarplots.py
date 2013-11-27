import numpy as np
import fig_utils as fig
from EUgrid import EU_Nodes

##  Europe
EU_BClin = np.loadtxt("./data/BC99q_EU_countries.txt")
EU_BCsqr = np.loadtxt("./data/BC99q_EU_sqr.txt")

EUnodes = EU_Nodes()
EUlabels = []
for n in EUnodes:
    EUlabels.append(str(n.label))

fig.BC_barplot(EU_BClin, EUlabels, EU_BCsqr,
        savepath="./results/FigsOfInterest/Figs_v2/",
        figfilename = "EU_BC_barplot.pdf")

## World

w_data = np.load("./results/w_BC_regions.npz")
fig.BC_barplot(w_data['BClin'], w_data['labels'], w_data['BCsqr'],
        savepath="./results/FigsOfInterest/Figs_v2/",
        figfilename = "w_BC_barplot.pdf")


