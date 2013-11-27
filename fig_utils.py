import sys
import numpy as np
import matplotlib.pyplot as plt
import pylab

import aurespf.solvers as au
import FlowCalculation as fc
from worldgrid import world_Nodes
from EUgrid import EU_Nodes

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

def DE_load_mismatch_hist(savepath='./results/'):
    """ This function generates a load/mismatch histogram
        of germany with the data Rolando sent me.

        """

    plt.close()
    plt.rc('lines', lw=2)
    plt.rcParams['axes.color_cycle'] = au.c_cycle

    DE0mean = 54236.0
    DE1mean = 170463.0
    DE2mean = 308749.0
    absolute_meanloads = {'DE_0':DE0mean, 'DE_1':DE1mean, 'DE_2':DE2mean}
    data = np.load('./results/DE_mismatches.npz')

    xmin = -2
    xmax = 3.001
    bins = np.linspace(xmin, xmax, 200)
    lines = []
    EU = EU_Nodes()
    Load = -EU[18].load/EU[18].mean # This is the normalized load of DE

    lines.append(plt.hist(Load, bins=bins, visible=0, normed=1)[0])

    for k in data.keys():
        if k!='DE_3':
            print k
            nonzeromismatch = []
            for w in data[k]:
                if w > 1/absolute_meanloads[k] or w < -1/absolute_meanloads[k]:
                    nonzeromismatch.append(w)
            lines.append(pylab.hist(nonzeromismatch, bins=bins, visible=0,
                                    normed=0)[0]\
                                    /(abs(bins[0]-bins[1])*len(data[k])))

    plt.plot(bins[0:-1], lines[0], label='Load', color='k')
    plt.plot(bins[0:-1], lines[1], label='DE')
    plt.plot(bins[0:-1], lines[2], label='DE and 1st neighbors')
    plt.plot(bins[0:-1], lines[3], label='DE and 1st and 2nd neighbors')

    plt.gcf().set_size_inches([au.dcolwidth, 0.6*au.dcolwidth])
    plt.xlabel('$\Delta$ [normalized]')
    plt.ylabel('P($\Delta - KF$)')
    plt.legend()
    plt.axis([xmin, xmax, 0, 1.1*max(lines[0])])
    plt.tight_layout()
    plt.savefig(savepath+'DE_mismatch_hist_w_neighbors.pdf')


def load_mismatch_hist(regions, configs, variedparam='all', xmax=None,
                        title=None, no_title=False, path=None,
                        fig_filename=None):
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
    plt.rc('lines', lw=2)
    plt.rcParams['axes.color_cycle'] = au.c_cycle
    xmin = -2
    if not xmax:
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
        if flow_calc.capacities == 'zerotrans':
            nodes = world_Nodes()
        else:
            nodes = world_Nodes(load_filename=str(flow_calc)+'.npz')

        for region in regions:
            for n in nodes:
                if str(n.label) == region:
                    node = n

            Load = -node.load/node.mean
            if not made_loadhist:
                hist_y.append(plt.hist(Load, bins=bins, visible=0,
                              normed=1)[0])
                plt.plot(bins[0:-1], hist_y[-1], label='Load', color='k')
                max_loadhist = abs(np.max(hist_y[-1]))

                made_loadhist = True

            if flow_calc.capacities!='zerotrans':
                mismatch = node.curtailment - node.balancing
                nonzero_mismatch = []
                for w in mismatch:
                    if w>=1 or w<=-1: ################## NOW NONZERO MISMATCHES ARE INCLUDED
                        nonzero_mismatch.append(w/node.mean)

                hist_y.append(pylab.hist(nonzero_mismatch, bins=bins, visible=0,
                                    normed=0)[0]\
                                    /(abs(bins[0]-bins[1])*len(mismatch)))
                plt.plot(bins[0:-1], hist_y[-1],
                        label=''.join([flow_calc.label(variedparam)]))
            else:
                mismatch = node.mismatch/node.mean
                hist_y.append(plt.hist(mismatch, bins=bins, visible=0,
                              normed=1)[0])
                plt.plot(bins[0:-1], hist_y[-1],
                        label=''.join([flow_calc.label(variedparam)]))

    plt.gcf().set_size_inches([au.dcolwidth, 0.6*au.dcolwidth])
    plt.axis([xmin, xmax, 0, 1.1*max_loadhist])
    plt.xlabel('$\Delta$ [normalized]')
    plt.ylabel('P($\Delta - KF$)')
    plt.legend()
    plt.tight_layout()
    if not path:
        savepath = ''.join(['./results/', variedparam, '/'])
    else:
        savepath = path

    if not fig_filename:
        fig_filename =''.join([region, '_',
                configs[0].str_without(variedparam)])

    if not no_title:
        if not title:
            plt.title(fig_filename.replace('_', ' '))
        else:
            plt.title(title)

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

def bal_vs_cap_data():

    number_of_nodes = 8
    length_of_timeseries = 280512
    hours_per_year = length_of_timeseries/32

    modes = ['lin', 'sqr']
    scalefactors = np.linspace(0, 2, 41)
    #scalefactors = [0.1, 1.45]

    for mode in modes:
        balancing_energy = [] # this is for saving the balancing energy
        total_capacity = []
        max_balancing = [] # this is for saving the balancing capacity
        balancing_caps99 = [] # this is for saving the balancing capacity,
                              # found as the 99% quantile of the balancing
                              # energy

        if mode == 'lin':
            h0 = au.get_quant_caps(filename=\
                                   './results/w_aHE_copper_lin_flows.npy')
        if mode == 'sqr':
            h0 = au.get_quant_caps(filename=\
                                    './results/w_aHE_copper_sqr_flows.npy')

        total_caps_q99 = np.sum(au.biggestpair(h0))
        print total_caps_q99

        for a in scalefactors:
            flow_calc = fc.FlowCalculation('w', 'aHE', ''.join([str(a), 'q99']), mode)
            filename = ''.join([str(flow_calc), '.npz'])
            nodes = world_Nodes(load_filename = filename)
            total_balancing_power = sum([sum(n.balancing) for n in nodes])

            total_mean_load = np.sum([n.mean for n in nodes])
            normalized_BE = total_balancing_power/\
                    (length_of_timeseries*total_mean_load)

            balancing_energy.append(normalized_BE)
            max_balancing.append(np.sum(np.max(n.balancing) for n in nodes)\
                                    /total_mean_load)

            balancing_caps99.append(np.sum(au.get_q(n.balancing,0.99) for n in nodes)/total_mean_load)
            total_capacity.append(a*total_caps_q99)

        if mode == 'lin':
            TC_lin = total_capacity
            BE_lin = balancing_energy
            BC_lin = max_balancing
            BC99_lin = balancing_caps99
        if mode == 'sqr':
            TC_sqr = total_capacity
            BE_sqr = balancing_energy
            BC_sqr = max_balancing
            BC99_sqr = balancing_caps99

    np.savez('./results/w_bal_vs_cap_data99.npz', TC_lin=TC_lin,\
            BE_lin=BE_lin, BC_lin=BC_lin, BC99_lin=BC99_lin,\
            TC_sqr=TC_sqr, BE_sqr=BE_sqr, BC_sqr=BC_sqr, BC99_sqr=BC99_sqr)

def BC_vs_region_data():
    linNodes = world_Nodes(load_filename="w_aHE_copper_lin.npz")
    BC99_lin = []
    labels = []
    for n in linNodes:
        BC99_lin.append(au.get_q(n.balancing, 0.99)/n.mean)
        labels.append(str(n.label))

    sqrNodes = world_Nodes(load_filename="w_aHE_copper_sqr.npz")
    BC99_sqr = au.get_q(sqrNodes[4].balancing, 0.99)/sqrNodes[4].mean

    np.savez('./results/w_BC_regions.npz', BClin = BC99_lin, labels = labels,
            BCsqr = BC99_sqr)




def BE_vs_TC(datafilename='./results/w_bal_vs_cap_data.npz', region='w',
        savepath=None, fig_filename=None):
    """ This function creates a plot of the total balancing energy of the
        world normalized to the total annual consumption of the  world,
        as a function of installed transmission capacity (0-4 TW)
        The data is taken from ./results/bal_vs_cap_data.npz, generated
        with the bal_vs_cap_data() function.

        """
    plt.close()
    plt.rc('lines', lw=2)

    data = np.load(datafilename)
    balancing_energy = data['BE_sqr']
    transmission_capacity = data['TC_sqr']

    plt.plot(transmission_capacity/1e6, balancing_energy, color=au.blue)

    if region=='w':
        minBE = 0.08294672
    if region=='EU':
        minBE = min(balancing_energy)

    plt.plot([0, 4], [minBE, minBE], linestyle='dashed', color='k')
    plt.text(0.01*transmission_capacity[-1]/1e6,
            minBE+0.003, 'Minimum balancing energy', size=12)
    plt.gcf().set_size_inches([au.dcolwidth, 0.6*au.dcolwidth])
    plt.xlabel('Total installed transmission capacity [TW]')
    plt.ylabel('Balancing energy [normalized]')
    plt.xlim([0, 0.6*transmission_capacity[-1]/1e6])
    plt.tight_layout()

    if region=='w':
        title = 'World'
    else:
        title = region
    plt.title(title)

    if not savepath:
        savepath = './results/'
    if not fig_filename:
        fig_filename = ''.join([region, '_BE_vs_TC.pdf'])
    plt.savefig(savepath+fig_filename)
    plt.close()

def BC_vs_TC(datafilename='./results/w_bal_vs_cap_data.npz', region='w',
             savepath=None, fig_filename=None, q99=False):

    plt.close()
    plt.rc('lines', lw=2)
    plt.rcParams['axes.color_cycle'] = au.c_cycle
    data = np.load(datafilename)

    if q99:
        BC_lin = data['BC99_lin']
        BC_sqr = data['BC99_sqr']
    else:
        BC_lin = data['BC_lin']
        BC_sqr = data['BC_sqr']

    plt.plot(data['TC_lin']/1e6, BC_lin, label='Linear')
    plt.plot(data['TC_sqr']/1e6, BC_sqr, label='Square')


    plt.gcf().set_size_inches([au.dcolwidth, 0.6*au.dcolwidth])
    plt.xlabel('Total installed transmission capacity [TW]')
    plt.ylabel('Necessary balancing capacity [normalized]')
    plt.legend(loc=3)
    midindex = len(BC_lin)/2
    plt.plot(data['TC_lin'][midindex]/1e6, BC_lin[midindex], 'x')
    plt.plot(data['TC_sqr'][midindex]/1e6, BC_sqr[midindex], 'x')
    #plt.vlines(1e-6*0.5*data['TC_lin'][-1], 0, 2, linestyle='dashed')
    #plt.text(1.04*1e-6*0.5*data['TC_lin'][-1],\
    #        1.2*np.mean([BC_lin[-1],BC_sqr[-1]]),
    #        r'99$\%$ quantile - linear', rotation='vertical')
    #plt.vlines(1e-6*0.5*data['TC_sqr'][-1], 0, 2, linestyle='dashed')
    #plt.text(1.03*1e-6*0.5*data['TC_sqr'][-1],\
    #        1.2*np.mean([BC_lin[-1],BC_sqr[-1]]),
    #        r'99$\%$ quantile - square', rotation='vertical')

    if region=='w':
        title = 'World'
    else:
        title = region
    plt.title(title)

    plt.xlim([0, 0.55*data['TC_sqr'][-1]/1e6])
    plt.ylim([0.95*BC_sqr[-1], 1.05*BC_lin[0]])
    plt.tight_layout()

    if not savepath:
        savepath = './results/'
    if not fig_filename:
        fig_filename = ''.join([region, '_BC_vs_TC.pdf'])
    plt.savefig(savepath+fig_filename)
    plt.close()

def BC_vs_mixing(layouts):
    plt.close()
    plt.rc('lines',lw=2)

    alphas = np.linspace(0,1, 20)

    for layout in layouts:
        admat = ''.join(['./settings/', layout, 'admat.txt'])
        nodes = world_Nodes(admat=admat)

        total_mean_load = sum([n.mean for n in nodes if str(n.label) in layout or layout == 'w'])
        print total_mean_load
        total_BC_norm = []
        for a in alphas:
            nodes.set_alphas(a)

            BC = []
            for n in nodes:
                if str(n.label) in layout or layout =='w':
                    mismatch = n.mismatch
                    balancing = -mismatch[mismatch<0]
                    BC.append(max(balancing))

            total_BC_norm.append(sum(BC)/total_mean_load)

        plt.plot(alphas, total_BC_norm, label=layout)


    plt.xlabel(r'$\alpha_W$')
    plt.ylabel('Necessary balancing capacity [normalized]')
    plt.legend()
    plt.savefig('./results/BC_vs_alpha.pdf')

def BC_barplot(heights, labels, BC99sqr,\
                 savepath="./results/", figfilename="BC_barplot.pdf"):
    assert(len(heights)==len(labels)), "The number of Balancing capacities\
                                        doesn't match the number of labels"

    plt.close()
    plt.rc('lines',lw=2)

    N = len(heights)
    width = 0.5
    xloc = np.arange(N) + width/2
    xmin = 0
    xmax = xloc[-1]+3*width/2
    print xloc

    plt.bar(xloc, heights, width, color = au.blue)
    plt.plot([xmin,xmax],BC99sqr*np.ones(2), linestyle='dashed', color='k')

    plt.gcf().set_size_inches([1.3*au.dcolwidth, 0.6*au.dcolwidth])
    if N>10:
        plt.xticks(xloc+0.2*width, labels, rotation=60, va='top', ha='center')
    if N<10:
        plt.xticks(xloc+0.5*width, labels, rotation=60, va='top', ha='right')
    plt.tick_params(\
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='on')
    plt.ylabel("Balancing capacity [normalized]")
    plt.axis([xmin, xmax, 0, max(heights)*1.2])

    plt.savefig(savepath+figfilename)
    plt.close()
