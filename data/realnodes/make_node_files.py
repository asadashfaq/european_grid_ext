import numpy as np

regions = {"China_":"CN", "Europe_":"EU", "India_":"IN", "JSKorea_":"JK",
            "MEast_":"ME", "NAfrica_":"NA", "SEAsia_":"SE", "WRussia_":"RU"}
meanloads = {"China_":526631.781593, "Europe_":1, "India_":374294.092208,
            "JSKorea_":35883.1879475, "MEast_":205546.876271,
            "NAfrica_":99222.6007344, "SEAsia_":167618.824449,
            "WRussia_":71954.5728944} # note that europe is not normalized

for r in regions.keys():
    print r

    Gw = np.load("".join([r,"WND_TS.npy"]))
    Gs = np.load("".join([r,"SOL_TS.npy"]))

    if r in ["Europe_", "JSKorea_", "SEAsia_"]:
        shortload = np.load("".join([r,"LOD_TS.npy"]))
        loadlist = [shortload for i in xrange(4)]
        L = np.concatenate(loadlist)

    elif r == "MEeast_":
        L = np.load("".join([r,"LOD_TS.npy"]))

    else:
        shortload = np.load("".join([r,"LOD_TS.npy"]))
        loadlist = [shortload for i in xrange(32)]
        loadlist.append(shortload[0:192])
        L = np.concatenate(loadlist)

    print L.size, Gs.size, Gs.size
    assert L.size == Gw.size, "Load timeseries is the wrong length"
    assert Gw.size == Gs.size, "Wind and solar timeseries are of \
                                #different length."
    t = np.linspace(0, Gw.size-1, Gw.size)

    path = "../"
    filename = "".join(["VE_", regions[r], ".npz"])
    L = meanloads[r]*L/1000 # now the load is unnormalized and in MW
    noiseamplitude = (0.036/1000)*meanloads[r]
    ### add a little noise to the loads for the regions other that europe
    ### to get rid of highly periodic components in the load, that make
    ### the histograms ugly
    if r!="Europe_":
        L = L  + np.random.normal(0, noiseamplitude, L.size)

    np.savez(path+filename, L=L, Gw=Gw, Gs=Gs, t=t, datalabel=regions[r])
