import numpy as np
import pylab
import matplotlib.pyplot as plt
from worldgrid import world_Nodes

N = world_Nodes()

totalmismatch = np.zeros_like(N[0].mismatch)
nonzeromismatch = []
numberofnodes=8
for i in xrange(len(totalmismatch)):
    normedmismatch = 0
    for n in N:
        if n.id<numberofnodes:
            totalmismatch[i] = totalmismatch[i] + n.mismatch[i]
            normedmismatch = normedmismatch + n.mismatch[i]#/n.mean
    if totalmismatch[i] >= numberofnodes or totalmismatch[i] <=-numberofnodes:
        nonzeromismatch.append(normedmismatch)

meanload = np.sum(n.mean for n in N)
mean_nonzeromismatch = [m/meanload for m in nonzeromismatch]


print len(totalmismatch), len(mean_nonzeromismatch)
#plt.plot(mean_nonzeromismatch)
bins = np.linspace(-2,3.001,200)
histy=pylab.hist(mean_nonzeromismatch, bins=bins, visible=0, normed=1)[0]\
        #/(abs(bins[0]-bins[1])*len(mean_nonzeromismatch))
print histy.max()

plt.plot(bins[0:-1], histy)
plt.axis([-2,3.001,0,2.75])
plt.show()
#nonzero_normed_mismatch = []

#for w in totalmismatch:
#    if w>=8 or w<=-8:
#        nonzero_normed_mismatch.append(w)

