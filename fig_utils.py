import sys
import numpy as np
import matplotlib.pyplot as plt
import pylab

import aurespf.solvers as au
import FlowCalculation as fc
from worldgrid import world_Nodes

def load_hist_noiseinv(node):
    """ Creates a normalized histogram of the load
        of a node with different amplitudes of the noise.
        Note that this only works properly if the node doesn't
        already have had noise added. Build again without noise
        with make_node_files.py script in /date/realnodes.
        The figure is saved to:
        ./results/Loadnoise/noise_lvl_<node.label>.pdf.

        """

    plt.clf()
    noiseamplitudes = np.linspace(0.02,0.035,7)

    Load = -node.load/node.mean
    bins = np.linspace(-1.5,-0.5,200)

    hist_ydata = plt.hist(Load, bins=bins, visible=0, normed=1)[0]
    plt.plot(bins[0:-1], hist_ydata, label='No noise')

    for a in noiseamplitudes:
        Load = Load + np.random.normal(0, a, Load.size)
        hist_ydata = plt.hist(Load, bins=bins, visible=0, normed=1)[0]
        plt.plot(bins[0:-1], hist_ydata, label=str(a))

    plt.legend()

    plt.ylabel("P(Load)")
    plt.xlabel("Load [normalized]")
    plt.title(node.label)
    filename = 'noise_lvl_' + str(node.label) + '.pdf'
    plt.savefig('./results/Loadnoise/'+filename)
    plt.close()

def load_hist(node):
    """ Makes a normalized histogram of the load of a node.
        saves to ./results/Loadhists/<node.label>_load_hist.pdf

        """


    Load = -node.load/node.mean
    bins = np.linspace(-1.5,-0.5,200)
    hist_ydata = plt.hist(Load, bins=bins, visible=0, normed=1)[0]
    plt.plot(bins[0:-1], hist_ydata)
    plt.ylabel("P(Load)")
    plt.xlabel("Load [normalized]")
    plt.title(node.label)

    filename = str(node.label) + '_load_hist' + '.pdf'
    plt.savefig('./results/Loadhists/'+filename)
    plt.close()

def load_mismatch_hist(regions, configs, variedparam='all'):
    """ This function creates a histogram of the load and
        the mismatch distributions, in a number of regions
        and configurations. variedparam is the parameter
        being varied, e. g. capacities, and is used in
        making smart labeling.
        Example: load_mismatch_hist([EU],
                [flow_calc1, flow_cal2], 'solvermode')
        The second argument is a list of FlowCalculation
        objects, from my own class.

        """

    plt.close()
    xmin = -2
    xmax = 3.001
    bins = np.linspace(xmin, xmax, 200)
    hist_y = [] # list for collecting y-data for the histograms
    made_loadhist = False

######## this takes out configs that don't make sense, such as zerotrans lin
    for flow_calc in configs:
        if flow_calc.capacities=='zerotrans':
            if flow_calc.solvermode=='lin' or flow_calc.solvermode=='sqr':
                configs.remove(flow_calc)
###########################################################################

    for flow_calc in configs:
        if flow_calc.alphas=='aHE':
            if flow_calc.capacities == 'zerotrans':
                nodes = world_Nodes()
            else:
                nodes = world_Nodes(load_filename=str(flow_calc)+'.npz')
        else:
            sys.stderr.write("This configuration of alphas, has not\
                             been accounted for.")
        for region in regions:
            for n in nodes:
                if str(n.label) == region:
                    node = n

            Load = -node.load/node.mean
            if not made_loadhist:
                hist_y.append(plt.hist(Load, bins=bins, visible=0,
                              normed=1)[0])
                plt.plot(bins[0:-1], hist_y[-1], label='Load')
                max_loadhist = abs(np.max(hist_y[-1]))

                made_loadhist = True

            if flow_calc.capacities!='zerotrans':
                mismatch = node.curtailment - node.balancing
                nonzero_mismatch = []
                for w in mismatch:
                    if w>=1 or w<=-1:
                        nonzero_mismatch.append(w/node.mean)

                hist_y.append(pylab.hist(nonzero_mismatch, bins=bins, visible=0,
                                    normed=0)[0]\
                                    /(abs(bins[0]-bins[1])*len(mismatch)))
                plt.plot(bins[0:-1], hist_y[-1],
                        label=''.join(['Mismatch, ', flow_calc.label(variedparam)]))
            else:
                mismatch = node.mismatch/node.mean
                hist_y.append(plt.hist(mismatch, bins=bins, visible=0,
                              normed=1)[0])
                plt.plot(bins[0:-1], hist_y[-1],
                        label=''.join(['Mismatch, ', flow_calc.label(variedparam)]))

    plt.xlabel('Mismatch power [normalized]')
    plt.ylabel('P($\Delta - KF$)')
    plt.axis([xmin, xmax, 0, 1.1*max_loadhist])
    plt.legend()
    savepath = ''.join(['./results/', variedparam, '/'])
    fig_filename =''.join([region, '_',
                configs[0].str_without(variedparam)])
    plt.title(fig_filename.replace('_', ' '))

    plt.savefig(''.join([savepath, fig_filename, '.pdf']))
    print "Plot saved to file", ''.join([savepath, fig_filename, '.pdf'])

def mismatch_hist_layout(region, alphas, capacities, solvermode):
    """ This function makes a normalized histogram of the load and
        mismatch in the region, with alphas, capacities and solvermode
        specified. All the different possible layouts are plotted
        with this configuration.
        Example: mismatch_hist_layout('EU', 'aHe', 'q99', 'lin')
        The figure is saved to:
        ./results/layout/<region>_<alphas>_<capacities>_<solvermode>.pdf

        """

    EU_layouts = ['w', 'EU_RU', 'EU_NA', 'EU_ME', 'EU_RU_NA_ME']
    CN_layouts = ['w', 'CN_JK', 'CN_SE', 'CN_SE_JK']

    fc_list = []

    if region=='EU':
        layout_list = EU_layouts
    elif region=='CN':
        layout_list = CN_layouts
    else:
        layout_list = EU_layouts + CN_layouts

    for L in layout_list:
        fc_list.append(fc.FlowCalculation(L, alphas, capacities,
                        solvermode))

    load_mismatch_hist([region], fc_list, 'layout')


def mismatch_hist_capacities(region, layout, alphas, solvermode):
    """ This function makes a normalized histogram of the load and
        mismatch in the region, with layout, alphas and solvermode
        specified. All the different possible capacities are plotted
        with this configuration.
        Example: mismatch_hist_capacities('EU', 'EU_RU', 'aHe', 'lin')
        The figure is saved to:
         ./results/capacities/<region>_<layout>_<alphas>_<solvermode>.pdf

        """
    capacities = ['copper', 'hq99', 'q99', 'zerotrans']

    fc_list = []
    for c in capacities:
        fc_list.append(fc.FlowCalculation(layout, alphas, c, solvermode))

    load_mismatch_hist([region], fc_list, 'capacities')


def mismatch_hist_solvermode(region, layout, alphas, capacities):
    """ This function makes a normalized histogram of the load and
        mismatch in the region, with layout, alphas and capacities
        specified. All the different possible solvermodes are plotted
        with this configuration.
        Example: mismatch_hist_solvermode('EU', 'EU_RU', 'aHe', 'hq99')
        The figure is saved to:
         ./results/capacities/<region>_<layout>_<alphas>_<capacities>.pdf

        """
    modes = ['lin', 'sqr']

    fc_list = []
    for m in modes:
        fc_list.append(fc.FlowCalculation(layout, alphas, capacities, m))

    load_mismatch_hist([region], fc_list, 'solvermode')












