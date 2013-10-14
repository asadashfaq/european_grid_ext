import numpy as np

import aurespf.solvers as au
from EUgrid import EU_Nodes
from magnus_figutils import simple_plot, smooth_hist, flow_hist, \
balancing_energy, layouts_hist

europe = EU_Nodes(load_filename="copper.npz") # load solved copper
                                              # flow Europe, Nodes-
                                              # object

####### Make figure of mismatch in Denmark over the 8-year period ######
print "Plotting Danish mismatch"
denmark = europe[21] # this is a node-object
Ntime = denmark.mismatch.size
time = np.linspace(0, Ntime-1, Ntime)

simple_plot(time, denmark.mismatch, "Time (h)", "Mismatch (MW)",
            title="Mismatch time series Denmark", path="Plotting practice/")


###### Make a figure showing the Danish unnormalized load timeseries ###
print "Plotting unnormalize load signal"
load = denmark.load # this is the unnormalize load signal

simple_plot(time, load, "Time (h)", "Load (MW)",
        title="Load time series Denmark", path="Plotting practice/")

##### Make figure of timeseries for solar energy in all of Europe
##### the last month ##
print "Plotting solar production in Europe"
solar_eu = np.zeros_like(denmark.get_solar())

for n in europe:
    solar_eu += n.get_solar()

simple_plot(time[69000:Ntime],solar_eu[69000:Ntime],'Time (h)',
            'Solar produduction (MW)',
            title="Solar production in Europe", path="Plotting practice/")

### Reproduce Rolando et. al. 2013 Fig 1 (Mismatch distributions) ####
print "Plotting Mismatch distributions for Denmark, Spain and Germany"
smooth_hist(EU_Nodes()) # note how we us an unsolved Nodes object, as we
                        # are interested in the no-transmission picture

### Reproduce Rolando et. al. 2013 Fig 4 (Distribution of
### powerflow (FR-ES) ###
print "Plotting distribution of power flow between France and Spain"
for n in europe:
    if n.label == 'FR':
        france = n

for j in au.AtoKh(europe)[-1]:
    for i in j:
        if (i=='FR to ES' or i=='ES to FR'):
             link = j[1]

flow_hist(link,"Flow from France to Spain", mean=france.mean, quantiles=True,
          savepath="./Plotting practice/", xlim=[-1.5,1], ylim=[0,1.5])

### Reproduce Rolando et. al 2013 fig. 5 (Balancing vs transmission cap)
### Saves in balancing.pdf
balancing_energy()

### Reproduce Rolando et. al 2013 fig. 7 (Normalized non-zero dists.
### for residual load (below zero) and excess generation (above zero)
### Saves in Multilayouts.pdf

layouts_hist()
