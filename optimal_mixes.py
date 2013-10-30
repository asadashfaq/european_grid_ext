import numpy as np
import matplotlib.pyplot as plt
import aurespf.solvers as au

from worldgrid import world_Nodes

nodes = world_Nodes() # this loads an unsolved nodes object,
                      # with the 8 superregions

alphas = np.linspace(0,1,100)
plt.ion()

for n in nodes:
    mean_balancing=[]
    min_balancing = 1e7
    for i in xrange(len(alphas)):
        n.set_alpha(alphas[i])
        mean_balancing.append(-np.sum(n.mismatch[n.mismatch<0])
                /n.mismatch.size)
        if mean_balancing[i]<min_balancing:
            min_balancing = mean_balancing[i]
            min_index = i;

    print n.label, alphas[min_index]

    plt.plot(alphas, mean_balancing, '-', label=str(n.label))

plt.legend()


