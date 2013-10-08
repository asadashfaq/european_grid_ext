import numpy as np
import matplotlib.pyplot as plt

import aurespf.solvers as au
from EUgrid import EU_Nodes

def simple_plot(xData, yData, xLabel, yLabel, title, path=None, label=None):
    """ Creates a simple plot and saves it. """

    plt.close()
    plt.plot(xData, yData, label = label)
    plt.xlim(xmax=max(xData))
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)

    if label:
        plt.legend()

    if (path!=None):
        if not path.endswith('/'):
            path = path + '/'
        filename = ''.join([path,title.replace(' ','_'),'.pdf'])
    else:
        filename = ''.join([title.replace(' ','_'),'.pdf'])

    plt.savefig(filename)
    plt.close()

def smooth_hist(N):
    """ Takes a Nodes object, fx. Europe
        this plot mimics Rolando's from
        figutils.py

        """

    alphas = [0, 0.7, 1.0]
    xmin = -2
    xmax = 3.001
    bins = np.linspace(xmin, xmax, 200)

    for n in N:
        if n.label in ["ES", "DE", "DK"]:
            Load = -n.load/n.mean # This gives us the normalized load

            hist_ydata = [] # list for saving the yvalues of the
            # histograms.  Note that the histogram returns a list, the
            # first of which is the height of the bins (the others are
            # the bins and a list of rectangle objects from
            # matplotlib.pyplot
            hist_ydata.append(plt.hist(Load, bins=bins, visible=0,
                normed=1)[0])

            # Calculate the normalized mismatch histograms for
            # different alphas.
            for alpha in alphas:
                n.set_alpha(alpha)
                hist_ydata.append(plt.hist(n.mismatch/n.mean, bins=bins,
                    visible=0, normed=1)[0])

            plt.close()
            plt.plot(bins[0:-1], hist_ydata[0], label="Load", linewidth=2.0,
                    color='k')
            plt.plot(bins[0:-1], hist_ydata[1], label=r"$\alpha^W$ = 0.0",
                    linewidth=1.5)
            plt.plot(bins[0:-1], hist_ydata[2], label=r"$\alpha^W$ = 0.7",
                    linewidth=1.5)
            plt.plot(bins[0:-1], hist_ydata[3], label=r"$\alpha^W$ = 1.0",
                    linewidth=1.5)

            plt.gcf().set_size_inches([10, 4])
            plt.ylim(0,max(hist_ydata[0])*1.075)
            plt.xlim(xmin,xmax)
            plt.legend()
            plt.xlabel("Normalized mismatch power")
            plt.ylabel("P($\Delta$)")
            plt.tight_layout()

            plt.savefig('./Plotting practice/MismatchDist'
                    + str(n.label) + '.pdf', dpi=400)

            plt.close()

def flow_hist(link, title, mean=None, quantiles = False,
              flow_filename = 'results/copper_flows.npy',
              number_of_bins = 250, xlim=None, ylim=None, savepath = None):

    """ Draws a histogram (normalized to mean if provided, takes a link
    number that can be found by running AtoKh() on the Nodes object
    that was solve to obtain the Flow vector.

    """

    if not mean: # this is equivalent to if mean == None
        mean = 1.0

    flows = np.load(flow_filename)
    flow = flows[link]/mean # this is the normalized flow timeseries of the
                            # link in question

    bins = np.linspace(min(flow), max(flow), number_of_bins)
    hist_ydata = plt.hist(flow, bins=bins, normed=1, histtype='stepfilled',
                           visible=0)[0] # the first argument returned from
                                         # plt.hist is the ydata we need

    plt.fill_between(bins[1:], 0, hist_ydata)

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    plt.title(title)
    plt.xlabel('Directed power flow [normalized]')
    plt.ylabel('$P(F_l)$')

    if quantiles:
        Q1 = -au.get_quant_caps(0.99)[2*link+1]/mean
        Q99 = au.get_quant_caps(0.99)[2*link]/mean
        Q01 = -au.get_quant_caps(0.999)[2*link+1]/mean
        Q999 = au.get_quant_caps(0.999)[2*link]/mean
        Q001 = -au.get_quant_caps(0.9999)[2*link+1]/mean
        Q9999 = au.get_quant_caps(0.9999)[2*link]/mean

        Qs = [Q1, Q99, Q01, Q999, Q001, Q9999]
        Qlabels = ['1%', '99%', '0.1%', '99.9%', '0.01%', '99.99%']

        for i in range(len(Qs)):
            plt.vlines(Qs[i],0,1)
            plt.text(Qs[i], 1.0+0.02*i, Qlabels[i])

    fig_filename = ''.join([title.replace(' ','_'), '.pdf'])
    if not savepath:
        savepath = ''

    plt.savefig(savepath + fig_filename)

    plt.close()

def balancing_energy():
    """ This function replicates Figure 5 in Rolando et. al 2013. """

    europe_raw = EU_Nodes()
    europe_copper = EU_Nodes(load_filename = "copper.npz")

    ########### Calculate the minimum balancing energy #############
    # The balancing energy is the smallest in the case of unconstrained
    # flow. This is the total balancing energy for all of Europe,
    # averaged over the time and normalized to the total mean load

    total_balancing_copper = np.sum(n.balancing for n in europe_copper)
    mean_balancing_copper = np.mean(total_balancing_copper)
    total_mean_load = np.sum(n.mean for n in europe_copper)

    min_balancing = mean_balancing_copper/total_mean_load
    print "The minimum balancing energy is:", min_balancing

    ######### Calculate the maximum balancing energy #############
    # The maximum balancing energy is the negative mismatch from
    # the raw date, that is the unsolved system, before any flow
    # has been taken into account.

    # Note, that contratry to total_balancing_copper (a timeseries)
    # total_balancing raw is just a number as it has been summed over
    # time.
    total_balancing_raw = -np.sum(np.sum(n.mismatch[n.mismatch<0])
                                  for n in europe_raw)
    mean_balancing_raw = total_balancing_raw/\
                                            total_balancing_copper.size

    max_balancing = mean_balancing_raw/total_mean_load

    print "The minimum balancing energy is:",max_balancing

def solve_lin_interpol(scalefactors):
    """ This function solves the European power grid several times,
        with different scalings of the current capacity layout after
        rule A in Rolando et. al. 2013. The results are saved as
        lin_int_<factor>.npz, and lin_int_<factor_flows.npy.
        The total capacities for the layouts are saved to
        total_caps_A.npy, arranged in the same way as the scalefactors
        list.

        """
    europe = EU_Nodes()
    h0s = get_h0s_A(scalefactors, europe)

    for i in xrange(len(scalefactors)):
        europe_solved, flows = au.solve(europe, h0=h0s[i])
        filename = "".join(["lin_int_",str(scalefactors[i])])
        flowpath = "".join(["./results/lin_int_",str(scalefactors[i]),
                            "_flows"])
        europe_solved.save_nodes(filename)
        np.save(flowpath, flows)

    total_capacities = get_total_capacities(h0s)
    np.save("./results/total_caps_A", total_capacities)


def solve_quant_interpol(quantiles):
    """ This function solves the European power grid several times,
        with different capacity layouts after, scaled af
        rule C in Rolando et. al. 2013. The results are saved as
        quant_int_<factor>.npz, and quant_int_<factor_flows.npy.
        The total capacities for the layouts are saved to
        total_caps_C.npy, arranged in the same order as the
        quantiles list.

        """

    europe = EU_Nodes()
    h0s = get_h0s_C(quantiles)

    for i in xrange(len(quantiles)):
        europe_solved, flows = au.solve(europe, h0=h0s[i])
        filename = "".join(["quant_int_", str(quantiles[i]).replace(".","_")])
        flowpath = "".join(["./results/quant_int_",
                            str(quantiles[i]).replace(".","_"), "_flows"])
        europe_solved.save_nodes(filename)
        np.save(flowpath, flows)

    total_capacities = get_total_capacities(h0s)
    np.save("./results/total_caps_C", total_capacities)


def get_h0s_A(scalefactors, nodes):
    """ This function returns a vector of link capacities to be used
        with the solver. This function generates after the rule for
        interpolation A in Rolando et. al. 2013, that is
        f_l = min(a*f_l, f_l99Q), (see pp. 10).

        """

    h99 = au.get_quant_caps(0.99)
    h_present = au.AtoKh(nodes)[-2];
    h0s = []

    for a in scalefactors:
        h0 = a*h_present
        for i in xrange(h99.size):
            if (h99[i] < a*h_present[i]):
                h0[i] = h99[i]
        h0s.append(h0)

    return h0s


def get_h0s_C(quantiles):
    """ This function returns a vector of link capacities to be used
        with the solver. The capacities are genereated after rule C
        for interpolation in Rolando et. al. 2013 so f_l = f_l^cQ

        """

    h0s = []

    for c in quantiles:
        h0s.append(au.get_quant_caps(c))

    return h0s


def get_total_capacities(h0s):
    """ This function takes a list of capacity-vectors and returns
        a list of total capacities installed.

        """

    total_capacities = []
    for h0 in h0s:
        total_capacities.append(sum(au.biggestpair(h0)))

    return total_capacities

