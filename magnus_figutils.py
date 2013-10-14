import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.interpolate import interp1d

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

def layouts_hist():
    """ Draws the figure 7. from Rolando et. al. 2013, with
        mismatch and load histograms for different capacity layouts
        in Spain, Germany and Denmark.

        """
    N_zero = EU_Nodes()
    N_present = EU_Nodes(load_filename="present.npz")
    N_interm = EU_Nodes(load_filename="intermediate.npz")
    N_99Q = EU_Nodes(load_filename="quant_int_0_99.npz")

    layouts = [N_present, N_interm, N_99Q]

    ## create a dictionary so countries are easier to get out
    ## of the nodes-objects. e.g as N_interm[countrydict['ES']]
    countries = ['ES', 'DE', 'DK']
    countrydict = {}
    for n in N_zero:
        if n.label in countries:
            countrydict[str(n.label)] = n.id

    xmin = -2
    xmax = 3.001
    bins = np.linspace(xmin, xmax, 250)

    plt.ion()
    f, axarr = plt.subplots(3, sharex=True)
    labels = ['Present layout', 'Intermediate layout',
              '99% Quantile layout']
    plotcount = 0;
    for country in countries:
        n0 = N_zero[countrydict[country]]
        Load = -n0.load/n0.mean # This gives us the normalized load
        load_hist = plt.hist(Load, bins=bins, visible=0, normed=1)[0]
        axarr[plotcount].plot(bins[0:-1], load_hist, label="Load")
        axarr[plotcount].set_title(country)
        zero_hist = plt.hist(n0.mismatch/n0.mean,bins=bins,visible=0,
                                normed=1)[0]
        axarr[plotcount].plot(bins[0:-1], zero_hist,
                               label="Zero transmission")

        linecount = 0;
        for layout in layouts:
            node = layout[countrydict[country]]

            mismatch = node.curtailment - node.balancing
            nonzero_mismatch = []
            for w in mismatch:
                if w>=1 or w<=-1:
                    nonzero_mismatch.append(w/node.mean)
            print len(nonzero_mismatch), node.label, str(layout), linecount
            hist_ydata = pylab.hist(nonzero_mismatch, bins=bins, normed=0,
                                    visible=0)[0]/(abs(bins[0]-bins[1])*70128)
            axarr[plotcount].plot(bins[0:-1], hist_ydata,
                                  label = labels[linecount])
            linecount += 1;

        axarr[plotcount].legend()
        axarr[plotcount].set_ylabel("$P(\Delta - KF)$")
        axarr[plotcount].axis([-2,3.001,0,1.1*max(load_hist)])
        plotcount += 1


    plt.gcf().set_size_inches([8.5, 3*8.5*0.4])
    axarr[2].set_xlabel("Mismatch power [normalized]")
    f.savefig("./Plotting practice/Multilayout.pdf")


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



    #### Calculate the current total capacity ################
    # the hardcoded 10000 stems from a link, we don't know the
    # actual capacity of, so it has been set to 10000 in the
    # eadmat.txt file.
    current_total_cap = sum(au.biggestpair(au.AtoKh(europe_raw)[-2])) - 10000

    print min_balancing, max_balancing
    scalefactorsA = [0.5, 1, 2, 4, 6, 8, 10, 12, 14]

    #scalefactorsA = np.linspace(0,1,11) # for use with the alternative A rule
    smoothA_cap, smoothA = get_bal_vs_cap(scalefactorsA, 'lin_int_', get_h0_A)
    plt.plot(smoothA_cap, smoothA(smoothA_cap), 'r-', label='Interpolation A')

    scalefactorsB = np.linspace(0, 2.5, 10)
    smoothB_cap, smoothB = get_bal_vs_cap(scalefactorsB, 'linquant_int_', get_h0_B)
    plt.plot(smoothB_cap, smoothB(smoothB_cap), 'g-', label='Interpolation B')

    quantiles = [0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1]
    smoothC_cap,smoothC = get_bal_vs_cap(quantiles, 'quant_int_', get_h0_C)
    plt.plot(smoothC_cap,smoothC(smoothC_cap),'b-',
             label="Interpolation C")

    plt.hlines(min_balancing, 0, 900, linestyle='dashed')
    plt.vlines(current_total_cap/1000, 0, 0.27, linestyle='dashed')
    plt.xlabel("Total installed transmission capacity, [GW]")
    plt.ylabel("Balancing energy [normalized]")
    plt.axis([0, 900.1, .125, .27])
    plt.yticks([.15,.17,.19,.21,.23,.25,.27])
    plt.xticks([0,100,200,300,400,500,600,700,800,900])
    plt.legend()
    plt.tight_layout()

    plt.savefig("./Plotting practice/balancing.pdf")

def get_bal_vs_cap(iterlist, prefix, get_h0_fun):
    """ Auxilary function for balancing_energy(). Example of use:
        capacity, balancing = get_bal_vs_cap(scalefactors, 'lin_int_',
        get_h0_A). Note the balancing that is returned is a function
        of capacity, interpolated with cubic spline. The capacity
        is in units of GW!

        """
    balancing = []
    total_capacity = []

    for q in iterlist:
        filename = "".join([prefix, str(q).replace(".","_"), ".npz"])
        nodes = EU_Nodes(load_filename = filename)
        mean_balancing = np.mean(np.sum(n.balancing for n in nodes))
        total_mean_load = np.sum(n.mean for n in nodes)
        balancing.append(mean_balancing/total_mean_load)

        total_capacity.append(get_total_capacity(get_h0_fun(q))/1000)

    # uncomment this to force the interpolation to go through the
    # endpoint expected from min_balancing and max_balancing
    #if prefix == 'lin_int_':
    #    total_capacity.insert(0,0)
    #    total_capacity.append(900)
    #   balancing.insert(0,0.242791209623)
    #    balancing.append(0.151116828435)

    smooth_balancing = interp1d(total_capacity, balancing, kind='cubic')
    smooth_cap = np.linspace(total_capacity[0], total_capacity[-1],200)

    return smooth_cap, smooth_balancing


def solve_lin_interpol(scalefactor):
    """ This function solves the European power grid
        with the current capacity layout scaled after
        rule A in Rolando et. al. 2013. The result is saved as
        lin_int_<factor>.npz, and lin_int_<factor>_flows.npy.

        """
    europe = EU_Nodes()
    h0 = get_h0_A(scalefactor)

    europe_solved, flows = au.solve(europe, h0=h0)
    filename = "".join(["lin_int_",str(scalefactor).replace('.','_')])
    flowpath = "".join(["./results/lin_int_",
                        str(scalefactor).replace('.','_'), "_flows"])
    europe_solved.save_nodes(filename)
    np.save(flowpath, flows)

def solve_linquant_interpol(scalefactor):
    """ This function solves the European power grid
        with capacity layout determined by rule B in
        Rolando et. al. 2013. The results are saved as
        linquant_int_<scalefactor>.npz,
        linquant_int_<scalefactor>_flows.npy.

        """

    europe = EU_Nodes()
    h0 = get_h0_B(scalefactor)

    europe_solved, flows = au.solve(europe, h0=h0)
    filename = "".join(["linquant_int_",str(scalefactor).replace('.','_')])
    flowpath = "".join(["./results/linquant_int_",
                        str(scalefactor).replace('.','_'), "_flows"])
    europe_solved.save_nodes(filename)
    np.save(flowpath, flows)


def solve_quant_interpol(quantile):
    """ This function solves the European power grid
        with capacity layout determined by rule C in
        Rolando et. al 2013. The result is saved as
        quant_int_<quantile>.npz and quant_int_<quantile>_flows.npy

        """

    europe = EU_Nodes()
    h0 = get_h0_C(quantile)

    europe_solved, flows = au.solve(europe, h0=h0)
    filename = "".join(["quant_int_", str(quantile).replace(".","_")])
    flowpath = "".join(["./results/quant_int_",
                            str(quantile).replace(".","_"), "_flows"])
    europe_solved.save_nodes(filename)
    np.save(flowpath, flows)


def get_h0_A(scalefactor):
    """ This function returns a h0 vector of link capacities to be
        used with the solver. This function generates after the rule
        for interpolation A in Rolando et. al. 2013, that is
        f_l = min(a*f_l, f_l99Q), (see pp. 10).

        """

    nodes = EU_Nodes()
    h99 = au.get_quant_caps(0.99)
    h_present = au.AtoKh(nodes)[-2];

    h0 = scalefactor*h_present
    for i in xrange(h99.size):
        if (h99[i] < h0[i]):
            h0[i] = h99[i]

    return h0

def get_h0_A2(scalefactor):
    """ This function works returns a h0 vector,
        created by downscaling the capacities in
        the unconstrained flow by scalefactor.

        """

    return scalefactor*au.get_quant_caps(1)


def get_h0_B(scalefactor):
    return scalefactor*au.get_quant_caps(0.99)


def get_h0_C(quantile):
    """ This function returns a vector of link capacities to be used
        with the solver. The capacities are genereated after rule C
        for interpolation in Rolando et. al. 2013 so f_l = f_l^cQ

        """

    h0 = au.get_quant_caps(quantile)
    return h0


def get_total_capacity(h0):
    """ This function takes a list of capacity-vectors and returns
        the total capacity installed.

        """

    total_capacity = sum(au.biggestpair(h0))

    return total_capacity


