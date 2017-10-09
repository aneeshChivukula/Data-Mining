import collections
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, title, xlabel, ylabel, savefig, legend
import sys
import pandas as pd
import numpy as np
import pickle
from scipy import stats


QmemodelrmaePath = "/home/achivuku/PycharmProjects/financedataanalysis/qmemodelrmae.pkl"
QmemodelrmsePath = "/home/achivuku/PycharmProjects/financedataanalysis/qmemodelrmse.pkl"
QmemodelurmaePath = "/home/achivuku/PycharmProjects/financedataanalysis/qmemodelurmae.pkl"
QmemodelurmsePath = "/home/achivuku/PycharmProjects/financedataanalysis/qmemodelurmse.pkl"

MsemodelrmaePath = "/home/achivuku/PycharmProjects/financedataanalysis/msemodelrmae.pkl"
MsemodelrmsePath = "/home/achivuku/PycharmProjects/financedataanalysis/msemodelrmse.pkl"
MsemodelurmaePath = "/home/achivuku/PycharmProjects/financedataanalysis/msemodelurmae.pkl"
MsemodelurmsePath = "/home/achivuku/PycharmProjects/financedataanalysis/msemodelurmse.pkl"


columns = ['AAPL_restricted', 'ABT_restricted', 'AEM_restricted', 'AFG_restricted', 'APA_restricted', 'B_restricted', 'CAT_restricted', 'IXIC_restricted', 'LAKE_restricted', 'MCD_restricted', 'MSFT_restricted', 'ORCL_restricted', 'SUN_restricted', 'T_restricted', 'UTX_restricted', 'WWD_restricted','AAPL_unrestricted', 'ABT_unrestricted', 'AEM_unrestricted', 'AFG_unrestricted', 'APA_unrestricted', 'B_unrestricted', 'CAT_unrestricted', 'IXIC_unrestricted', 'LAKE_unrestricted', 'MCD_unrestricted', 'MSFT_unrestricted', 'ORCL_unrestricted', 'SUN_unrestricted', 'T_unrestricted', 'UTX_unrestricted', 'WWD_unrestricted']
indexes = ['AAPL_prices', 'ABT_prices', 'AEM_prices', 'AFG_prices', 'APA_prices', 'B_prices', 'CAT_prices', 'IXIC_prices', 'LAKE_prices', 'MCD_prices', 'MSFT_prices', 'ORCL_prices', 'SUN_prices', 'T_prices', 'UTX_prices', 'WWD_prices']
nodestoremove = ['B', 'IXIC']

qmemodelrmae = collections.defaultdict(dict)
qmemodelrmse = collections.defaultdict(dict)
qmemodelurmae = collections.defaultdict(dict)
qmemodelurmse = collections.defaultdict(dict)

msemodelrmae = collections.defaultdict(dict)
msemodelrmse = collections.defaultdict(dict)
msemodelurmae = collections.defaultdict(dict)
msemodelurmse = collections.defaultdict(dict)


with open(QmemodelrmaePath, 'rb') as handle:
    qmemodelrmae = pickle.load(handle)

# with open(QmemodelrmsePath, 'rb') as handle:
#     qmemodelrmse = pickle.load(handle)

with open(QmemodelurmaePath, 'rb') as handle:
    qmemodelurmae = pickle.load(handle)

# with open(QmemodelurmsePath, 'rb') as handle:
#     qmemodelurmse = pickle.load(handle)


with open(MsemodelrmaePath, 'rb') as handle:
    msemodelrmae = pickle.load(handle)

# with open(MsemodelrmsePath, 'rb') as handle:
#     msemodelrmse = pickle.load(handle)

with open(MsemodelurmaePath, 'rb') as handle:
    msemodelurmae = pickle.load(handle)

# with open(MsemodelurmsePath, 'rb') as handle:
#     msemodelurmse = pickle.load(handle)

df = pd.read_csv("/home/achivuku/Documents/financedataanalysis/pricesvolumes.csv")
cols = [1,2,3,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,30,32,33,34,36,38,40,42]
df.drop(df.columns[cols],axis=1,inplace=True)

MSEGraphPath = "/home/achivuku/PycharmProjects/financedataanalysis/MSE-Graph.png"
QMEGraphPath = "/home/achivuku/PycharmProjects/financedataanalysis/QME-Graph.png"

MSEErrorsDataframePath = "/home/achivuku/PycharmProjects/financedataanalysis/MSEErrors-Graph.pkl"
QMEErrorsDataframePath = "/home/achivuku/PycharmProjects/financedataanalysis/QMEErrors-Graph.pkl"

MSEModelFstatMAEPath = "/home/achivuku/PycharmProjects/financedataanalysis/MSECauses-Graph.pkl"
QMEModelFstatMAEPath = "/home/achivuku/PycharmProjects/financedataanalysis/QMECauses-Graph.pkl"

numdecimalplaces = 3

def calculatefstatistic(nestedrd, nestedurd, colour):

    nestedfstat = collections.defaultdict(dict)
    G = nx.DiGraph()

    dfout = pd.DataFrame(columns=columns, index=indexes)

    for outerkey, outervalue in nestedrd.iteritems():
        for innerkey, innervalue in nestedrd[outerkey].iteritems():
            restrictederror = nestedrd[outerkey][innerkey]
            unrestrictederror = nestedurd[outerkey][innerkey]

            dfout.loc[outerkey][innerkey.replace("prices", "restricted")] = restrictederror
            dfout.loc[outerkey][innerkey.replace("prices", "unrestricted")] = unrestrictederror

            fstat = (restrictederror - unrestrictederror) / unrestrictederror

            if ( fstat > 0.05 ):
                nestedfstat[outerkey][innerkey] = fstat
                G.add_edge(innerkey.rstrip('_prices'), outerkey.rstrip('_prices'), color=colour, weight=round(fstat,numdecimalplaces))

    # print(G.edges(data=True))
    # sys.exit()
    # innerkey is granger influencing outerkey stock
    return nestedfstat, G, dfout

def plotskewedhistogram(stocskdf,stockname,colour,htype,filename):
    for c in stocskdf.columns.tolist()[1:]:
        ts = stocskdf[[c]]
        print(ts.skew())

    ts = stocskdf[[stockname]]
    tss = ts[stockname]
    # tss = ts.diff()[stockname]
    # tss[0] = 0

    # fig, ax = plt.subplots(figsize=(10, 8))

    tssh = tss.tolist()

    plt.figure()
    n, bins, patches = plt.hist(tssh,bins=100,alpha=0.75,histtype=htype, color=colour)

    xlabel(stockname.rstrip("_prices") + ' Price')
    ylabel(stockname.rstrip("_prices") + ' Frequency')
    # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\$')
    plt.grid(True)

    savefig(filename, dpi=300)

    # pyplot.hist(tssh2, bins1, alpha=0.5, color="blue", label='T')
    # width = 0.7 * (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2
    # ax.bar(center2, hist2, align='center', width=width1,color="blue",label='T')

def savenetworkxgraph(modelgraph, nodecolour, SavePath):
    modelgraph.remove_nodes_from(nodestoremove)
    edge_labels = dict([((u, v,), d['weight'])
                        for u, v, d in modelgraph.edges(data=True)])


    plt.figure(figsize=(20,10), dpi=300, facecolor='w', edgecolor='k')
    graph_pos = nx.circular_layout(modelgraph)
    # graph_pos = nx.circular_layout(qmemodelgraph)
    nx.draw_networkx_edge_labels(modelgraph, graph_pos, edge_labels=edge_labels)
    nx.draw(modelgraph, graph_pos, node_size=2000, font_size=16, font_weight='bold', with_labels=True,
                     arrows=True, node_color=nodecolour)

    plt.tight_layout()
    plt.savefig(SavePath, format="PNG")

def findtstats(l1, l2):
    ttest = stats.ttest_ind(l1, l2, equal_var=True)
    print 't-statistic independent = %6.3f testing pvalue = ' % ttest[0], "{:.2e}".format(ttest[1])



l1 = []
l2 = []

# MSE

# AAPL

# l1.append(0.452)
# l1.append(0.191)
# l1.append(0.387)
# l1.append(0.091)
# l1.append(0.185)
# l1.append(0.105)
# l1.append(0.391)
# l1.append(0.204)
# l1.append(0.103)
# l1.append(0.185)
# l1.append(0.054)
# l1.append(0.315)
# l1.append(0.209)
#
# l2.append(0.259)
# l2.append(1.053)
# l2.append(0.298)
# l2.append(0.918)
# l2.append(0.685)
# l2.append(0.984)
# l2.append(0.378)
# l2.append(0.266)
# l2.append(0.523)
# l2.append(1.180)
# l2.append(0.348)
# l2.append(0.568)
# l2.append(0.402)

# ABT

# l1.append(0.636)
# l1.append(0.204)
# l1.append(0.361)
# l1.append(0.136)
# l1.append(0.137)
# l1.append(0.094)
# l1.append(0.405)
# l1.append(0.251)
# l1.append(0.221)
# l1.append(0.203)
# l1.append(0.046)
# l1.append(0.287)
# l1.append(0.401)
#
# l2.append(1.073)
# l2.append(0.706)
# l2.append(0.447)
# l2.append(0.672)
# l2.append(0.387)
# l2.append(0.431)
# l2.append(0.391)
# l2.append(0.211)
# l2.append(0.194)
# l2.append(0.585)
# l2.append(0.183)
# l2.append(0.292)
# l2.append(0.590)


# AEM

# l1.append(0.501)
# l1.append(0.458)
# l1.append(0.372)
# l1.append(0.145)
# l1.append(0.073)
# l1.append(0.106)
# l1.append(0.363)
# l1.append(0.226)
# l1.append(0.151)
# l1.append(0.190)
# l1.append(0.114)
# l1.append(0.328)
# l1.append(0.202)

# l2.append(1.172)
# l2.append(0.439)
# l2.append(0.675)
# l2.append(0.311)
# l2.append(0.463)
# l2.append(0.576)
# l2.append(0.594)
# l2.append(0.451)
# l2.append(0.559)
# l2.append(0.793)
# l2.append(0.208)
# l2.append(0.494)
# l2.append(0.651)


# AFG

# l1.append(0.539)
# l1.append(0.487)
# l1.append(0.218)
# l1.append(0.093)
# l1.append(0.141)
# l1.append(0.132)
# l1.append(0.397)
# l1.append(0.231)
# l1.append(0.133)
# l1.append(0.171)
# l1.append(0.073)
# l1.append(0.218)
# l1.append(0.238)
#
# l2.append(0.842)
# l2.append(0.161)
# l2.append(0.784)
# l2.append(0.756)
# l2.append(0.436)
# l2.append(0.548)
# l2.append(0.275)
# l2.append(0.146)
# l2.append(0.222)
# l2.append(0.700)
# l2.append(0.172)
# l2.append(0.296)
# l2.append(0.437)


# APA

# l1.append(0.475)
# l1.append(0.437)
# l1.append(0.169)
# l1.append(0.409)
# l1.append(0.056)
# l1.append(0.094)
# l1.append(0.341)
# l1.append(0.251)
# l1.append(0.164)
# l1.append(0.211)
# l1.append(0.065)
# l1.append(0.250)
# l1.append(0.148)
#
# l2.append(1.154)
# l2.append(0.439)
# l2.append(0.354)
# l2.append(0.614)
# l2.append(0.348)
# l2.append(0.478)
# l2.append(0.521)
# l2.append(0.431)
# l2.append(0.498)
# l2.append(0.684)
# l2.append(0.162)
# l2.append(0.420)
# l2.append(0.587)


# CAT

# l1.append(0.303)
# l1.append(0.412)
# l1.append(0.105)
# l1.append(0.351)
# l1.append(0.111)
# l1.append(0.121)
# l1.append(0.329)
# l1.append(0.218)
# l1.append(0.152)
# l1.append(0.153)
# l1.append(0.051)
# l1.append(0.258)
# l1.append(0.364)
#
# l2.append(0.941)
# l2.append(0.257)
# l2.append(0.526)
# l2.append(0.413)
# l2.append(0.421)
# l2.append(0.413)
# l2.append(0.406)
# l2.append(0.275)
# l2.append(0.275)
# l2.append(0.621)
# l2.append(0.194)
# l2.append(0.230)
# l2.append(0.421)


# LAKE

# l1.append(0.612)
# l1.append(0.383)
# l1.append(0.201)
# l1.append(0.451)
# l1.append(0.101)
# l1.append(0.138)
# l1.append(0.423)
# l1.append(0.235)
# l1.append(0.205)
# l1.append(0.185)
# l1.append(0.036)
# l1.append(0.254)
# l1.append(0.241)
#
# l2.append(1.311)
# l2.append(0.384)
# l2.append(0.717)
# l2.append(0.607)
# l2.append(0.631)
# l2.append(0.511)
# l2.append(0.669)
# l2.append(0.328)
# l2.append(0.561)
# l2.append(0.499)
# l2.append(0.166)
# l2.append(0.510)
# l2.append(0.813)

# MCD

# l1.append(0.446)
# l1.append(0.452)
# l1.append(0.166)
# l1.append(0.483)
# l1.append(0.079)
# l1.append(0.112)
# l1.append(0.088)
# l1.append(0.233)
# l1.append(0.209)
# l1.append(0.152)
# l1.append(0.054)
# l1.append(0.307)
# l1.append(0.280)
#
# l2.append(0.885)
# l2.append(0.151)
# l2.append(0.652)
# l2.append(0.355)
# l2.append(0.588)
# l2.append(0.319)
# l2.append(0.505)
# l2.append(0.196)
# l2.append(0.139)
# l2.append(0.714)
# l2.append(0.137)
# l2.append(0.221)
# l2.append(0.427)

# MSFT

# l1.append(0.548)
# l1.append(0.464)
# l1.append(0.191)
# l1.append(0.315)
# l1.append(0.076)
# l1.append(0.135)
# l1.append(0.137)
# l1.append(0.402)
# l1.append(0.175)
# l1.append(0.196)
# l1.append(0.067)
# l1.append(0.360)
# l1.append(0.218)
#
# l2.append(1.268)
# l2.append(0.313)
# l2.append(0.791)
# l2.append(0.427)
# l2.append(0.772)
# l2.append(0.647)
# l2.append(0.399)
# l2.append(0.585)
# l2.append(0.433)
# l2.append(0.697)
# l2.append(0.148)
# l2.append(0.564)
# l2.append(0.834)


# ORCL

# l1.append(0.455)
# l1.append(0.434)
# l1.append(0.292)
# l1.append(0.441)
# l1.append(0.083)
# l1.append(0.111)
# l1.append(0.109)
# l1.append(0.386)
# l1.append(0.282)
# l1.append(0.222)
# l1.append(0.066)
# l1.append(0.291)
# l1.append(0.252)
#
# l2.append(0.723)
# l2.append(0.151)
# l2.append(0.618)
# l2.append(0.273)
# l2.append(0.572)
# l2.append(0.361)
# l2.append(0.439)
# l2.append(0.377)
# l2.append(0.201)
# l2.append(0.660)
# l2.append(0.117)
# l2.append(0.326)
# l2.append(0.534)


# SUN

# l1.append(0.571)
# l1.append(0.458)
# l1.append(0.185)
# l1.append(0.432)
# l1.append(0.081)
# l1.append(0.058)
# l1.append(0.105)
# l1.append(0.525)
# l1.append(0.200)
# l1.append(0.149)
# l1.append(0.063)
# l1.append(0.304)
# l1.append(0.340)
#
# l2.append(1.396)
# l2.append(0.431)
# l2.append(0.835)
# l2.append(0.649)
# l2.append(0.663)
# l2.append(0.604)
# l2.append(0.403)
# l2.append(0.758)
# l2.append(0.438)
# l2.append(0.664)
# l2.append(0.175)
# l2.append(0.593)
# l2.append(0.878)

# T

# l1.append(0.603)
# l1.append(0.441)
# l1.append(0.217)
# l1.append(0.344)
# l1.append(0.095)
# l1.append(0.066)
# l1.append(0.088)
# l1.append(0.395)
# l1.append(0.241)
# l1.append(0.133)
# l1.append(0.152)
# l1.append(0.301)
# l1.append(0.336)
#
# l2.append(1.451)
# l2.append(0.428)
# l2.append(0.887)
# l2.append(0.679)
# l2.append(0.781)
# l2.append(0.679)
# l2.append(0.462)
# l2.append(0.697)
# l2.append(0.258)
# l2.append(0.508)
# l2.append(0.694)
# l2.append(0.601)
# l2.append(0.981)


# UTX

# l1.append(0.533)
# l1.append(0.444)
# l1.append(0.231)
# l1.append(0.443)
# l1.append(0.071)
# l1.append(0.057)
# l1.append(0.107)
# l1.append(0.378)
# l1.append(0.225)
# l1.append(0.119)
# l1.append(0.177)
# l1.append(0.094)
# l1.append(0.243)
#
# l2.append(0.953)
# l2.append(0.141)
# l2.append(0.604)
# l2.append(0.331)
# l2.append(0.521)
# l2.append(0.282)
# l2.append(0.438)
# l2.append(0.276)
# l2.append(0.205)
# l2.append(0.188)
# l2.append(0.600)
# l2.append(0.147)
# l2.append(0.426)


# WWD

# l1.append(0.518)
# l1.append(0.436)
# l1.append(0.138)
# l1.append(0.373)
# l1.append(0.091)
# l1.append(0.058)
# l1.append(0.114)
# l1.append(0.448)
# l1.append(0.211)
# l1.append(0.103)
# l1.append(0.180)
# l1.append(0.053)
# l1.append(0.252)
#
# l2.append(0.712)
# l2.append(0.163)
# l2.append(0.664)
# l2.append(0.295)
# l2.append(0.569)
# l2.append(0.331)
# l2.append(0.582)
# l2.append(0.172)
# l2.append(0.159)
# l2.append(0.184)
# l2.append(0.681)
# l2.append(0.121)
# l2.append(0.206)



# QME

# AAPL

# l1.append(0.476)
# l1.append(0.116)
# l1.append(0.607)
# l1.append(0.101)
# l1.append(0.319)
# l1.append(0.122)
# l1.append(0.305)
# l1.append(0.177)
# l1.append(0.201)
# l1.append(0.179)
# l1.append(0.071)
# l1.append(0.296)
# l1.append(0.283)
#
# l2.append(0.196)
# l2.append(0.827)
# l2.append(0.265)
# l2.append(0.768)
# l2.append(0.475)
# l2.append(0.601)
# l2.append(0.289)
# l2.append(0.183)
# l2.append(0.217)
# l2.append(1.058)
# l2.append(0.301)
# l2.append(0.351)
# l2.append(0.304)

# l2.append(0.476)
# l2.append(0.116)
# l2.append(0.607)
# l2.append(0.101)
# l2.append(0.319)
# l2.append(0.122)
# l2.append(0.305)
# l2.append(0.177)
# l2.append(0.201)
# l2.append(0.179)
# l2.append(0.071)
# l2.append(0.296)
# l2.append(0.283)


# ABT

# l1.append(0.517)
# l1.append(0.199)
# l1.append(0.468)
# l1.append(0.092)
# l1.append(0.261)
# l1.append(0.131)
# l1.append(0.321)
# l1.append(0.206)
# l1.append(0.553)
# l1.append(0.193)
# l1.append(0.075)
# l1.append(0.294)
# l1.append(0.345)
#
# l2.append(0.957)
# l2.append(0.814)
# l2.append(0.431)
# l2.append(0.681)
# l2.append(0.489)
# l2.append(0.435)
# l2.append(0.521)
# l2.append(0.203)
# l2.append(0.274)
# l2.append(0.572)
# l2.append(0.201)
# l2.append(0.556)
# l2.append(0.661)

# l2.append(0.517)
# l2.append(0.199)
# l2.append(0.468)
# l2.append(0.092)
# l2.append(0.261)
# l2.append(0.131)
# l2.append(0.321)
# l2.append(0.206)
# l2.append(0.553)
# l2.append(0.193)
# l2.append(0.075)
# l2.append(0.294)
# l2.append(0.345)



# AEM

# l1.append(0.324)
# l1.append(0.515)
# l1.append(0.497)
# l1.append(0.128)
# l1.append(0.474)
# l1.append(0.222)
# l1.append(0.486)
# l1.append(0.141)
# l1.append(0.533)
# l1.append(0.174)
# l1.append(0.087)
# l1.append(0.396)
# l1.append(0.418)

# l2.append(1.145)
# l2.append(0.401)
# l2.append(0.626)
# l2.append(0.383)
# l2.append(0.441)
# l2.append(0.438)
# l2.append(0.516)
# l2.append(0.399)
# l2.append(0.476)
# l2.append(0.826)
# l2.append(0.137)
# l2.append(0.441)
# l2.append(0.618)

# l2.append(0.324)
# l2.append(0.515)
# l2.append(0.497)
# l2.append(0.128)
# l2.append(0.474)
# l2.append(0.222)
# l2.append(0.486)
# l2.append(0.141)
# l2.append(0.533)
# l2.append(0.174)
# l2.append(0.087)
# l2.append(0.396)
# l2.append(0.418)



# AFG

# l1.append(0.522)
# l1.append(0.489)
# l1.append(0.167)
# l1.append(0.133)
# l1.append(0.251)
# l1.append(0.114)
# l1.append(0.379)
# l1.append(0.146)
# l1.append(0.349)
# l1.append(0.308)
# l1.append(0.101)
# l1.append(0.335)
# l1.append(0.502)
#
# l2.append(0.900)
# l2.append(0.222)
# l2.append(0.763)
# l2.append(0.726)
# l2.append(0.434)
# l2.append(0.464)
# l2.append(0.461)
# l2.append(0.181)
# l2.append(0.269)
# l2.append(0.726)
# l2.append(0.191)
# l2.append(0.609)
# l2.append(0.556)

# l2.append(0.522)
# l2.append(0.489)
# l2.append(0.167)
# l2.append(0.133)
# l2.append(0.251)
# l2.append(0.114)
# l2.append(0.379)
# l2.append(0.146)
# l2.append(0.349)
# l2.append(0.308)
# l2.append(0.101)
# l2.append(0.335)
# l2.append(0.502)



# APA

# l1.append(0.607)
# l1.append(0.511)
# l1.append(0.165)
# l1.append(0.521)
# l1.append(0.194)
# l1.append(0.111)
# l1.append(0.418)
# l1.append(0.134)
# l1.append(0.349)
# l1.append(0.166)
# l1.append(0.055)
# l1.append(0.361)
# l1.append(0.312)
#
# l2.append(1.149)
# l2.append(0.366)
# l2.append(0.605)
# l2.append(0.571)
# l2.append(0.443)
# l2.append(0.409)
# l2.append(0.497)
# l2.append(0.346)
# l2.append(0.446)
# l2.append(0.657)
# l2.append(0.198)
# l2.append(0.381)
# l2.append(0.586)

# l2.append(0.607)
# l2.append(0.511)
# l2.append(0.165)
# l2.append(0.521)
# l2.append(0.194)
# l2.append(0.111)
# l2.append(0.418)
# l2.append(0.134)
# l2.append(0.349)
# l2.append(0.166)
# l2.append(0.055)
# l2.append(0.361)
# l2.append(0.312)


# CAT

# l1.append(0.463)
# l1.append(0.476)
# l1.append(0.118)
# l1.append(0.495)
# l1.append(0.135)
# l1.append(0.133)
# l1.append(0.425)
# l1.append(0.123)
# l1.append(0.454)
# l1.append(0.171)
# l1.append(0.083)
# l1.append(0.449)
# l1.append(0.264)
#
# l2.append(0.978)
# l2.append(0.255)
# l2.append(0.564)
# l2.append(0.397)
# l2.append(0.481)
# l2.append(0.401)
# l2.append(0.475)
# l2.append(0.276)
# l2.append(0.315)
# l2.append(0.609)
# l2.append(0.215)
# l2.append(0.391)
# l2.append(0.551)

# l2.append(0.463)
# l2.append(0.476)
# l2.append(0.118)
# l2.append(0.495)
# l2.append(0.135)
# l2.append(0.133)
# l2.append(0.425)
# l2.append(0.123)
# l2.append(0.454)
# l2.append(0.171)
# l2.append(0.083)
# l2.append(0.449)
# l2.append(0.264)



# LAKE

# l1.append(0.588)
# l1.append(0.548)
# l1.append(0.114)
# l1.append(0.472)
# l1.append(0.111)
# l1.append(0.226)
# l1.append(0.412)
# l1.append(0.151)
# l1.append(0.364)
# l1.append(0.185)
# l1.append(0.039)
# l1.append(0.446)
# l1.append(0.458)
#
# l2.append(1.283)
# l2.append(0.406)
# l2.append(0.685)
# l2.append(0.597)
# l2.append(0.639)
# l2.append(0.528)
# l2.append(0.654)
# l2.append(0.317)
# l2.append(0.557)
# l2.append(0.532)
# l2.append(0.151)
# l2.append(0.577)
# l2.append(0.838)

# l2.append(0.588)
# l2.append(0.548)
# l2.append(0.114)
# l2.append(0.472)
# l2.append(0.111)
# l2.append(0.226)
# l2.append(0.412)
# l2.append(0.151)
# l2.append(0.364)
# l2.append(0.185)
# l2.append(0.039)
# l2.append(0.446)
# l2.append(0.458)


# MCD

# l1.append(0.730)
# l1.append(0.494)
# l1.append(0.204)
# l1.append(0.591)
# l1.append(0.143)
# l1.append(0.287)
# l1.append(0.113)
# l1.append(0.195)
# l1.append(0.476)
# l1.append(0.181)
# l1.append(0.062)
# l1.append(0.195)
# l1.append(0.233)
#
# l2.append(0.784)
# l2.append(0.341)
# l2.append(0.621)
# l2.append(0.319)
# l2.append(0.574)
# l2.append(0.374)
# l2.append(0.426)
# l2.append(0.219)
# l2.append(0.191)
# l2.append(0.812)
# l2.append(0.223)
# l2.append(0.376)
# l2.append(0.604)

# l2.append(0.730)
# l2.append(0.494)
# l2.append(0.204)
# l2.append(0.591)
# l2.append(0.143)
# l2.append(0.287)
# l2.append(0.113)
# l2.append(0.195)
# l2.append(0.476)
# l2.append(0.181)
# l2.append(0.062)
# l2.append(0.195)
# l2.append(0.233)



# MSFT

# l1.append(0.719)
# l1.append(0.503)
# l1.append(0.178)
# l1.append(0.538)
# l1.append(0.119)
# l1.append(0.306)
# l1.append(0.115)
# l1.append(0.365)
# l1.append(0.404)
# l1.append(0.168)
# l1.append(0.038)
# l1.append(0.319)
# l1.append(0.408)
#
# l2.append(1.323)
# l2.append(0.309)
# l2.append(0.819)
# l2.append(0.582)
# l2.append(0.867)
# l2.append(0.716)
# l2.append(0.478)
# l2.append(0.606)
# l2.append(0.403)
# l2.append(0.722)
# l2.append(0.111)
# l2.append(0.634)
# l2.append(1.022)

# l2.append(0.719)
# l2.append(0.503)
# l2.append(0.178)
# l2.append(0.538)
# l2.append(0.119)
# l2.append(0.306)
# l2.append(0.115)
# l2.append(0.365)
# l2.append(0.404)
# l2.append(0.168)
# l2.append(0.038)
# l2.append(0.319)
# l2.append(0.408)


# ORCL

# l1.append(0.662)
# l1.append(0.497)
# l1.append(0.112)
# l1.append(0.486)
# l1.append(0.111)
# l1.append(0.201)
# l1.append(0.119)
# l1.append(0.453)
# l1.append(0.117)
# l1.append(0.178)
# l1.append(0.063)
# l1.append(0.311)
# l1.append(0.411)
#
# l2.append(0.857)
# l2.append(0.249)
# l2.append(0.631)
# l2.append(0.335)
# l2.append(0.591)
# l2.append(0.429)
# l2.append(0.409)
# l2.append(0.205)
# l2.append(0.222)
# l2.append(0.631)
# l2.append(0.151)
# l2.append(0.317)
# l2.append(0.501)

# l2.append(0.662)
# l2.append(0.497)
# l2.append(0.112)
# l2.append(0.486)
# l2.append(0.111)
# l2.append(0.201)
# l2.append(0.119)
# l2.append(0.453)
# l2.append(0.117)
# l2.append(0.178)
# l2.append(0.063)
# l2.append(0.311)
# l2.append(0.411)


# SUN

# l1.append(0.615)
# l1.append(0.473)
# l1.append(0.128)
# l1.append(0.535)
# l1.append(0.128)
# l1.append(0.363)
# l1.append(0.117)
# l1.append(0.519)
# l1.append(0.129)
# l1.append(0.311)
# l1.append(0.089)
# l1.append(0.266)
# l1.append(0.301)
#
# l2.append(1.368)
# l2.append(0.433)
# l2.append(0.782)
# l2.append(0.657)
# l2.append(0.632)
# l2.append(0.555)
# l2.append(0.324)
# l2.append(0.731)
# l2.append(0.351)
# l2.append(0.607)
# l2.append(0.178)
# l2.append(0.566)
# l2.append(0.866)

# l2.append(0.615)
# l2.append(0.473)
# l2.append(0.128)
# l2.append(0.535)
# l2.append(0.128)
# l2.append(0.363)
# l2.append(0.117)
# l2.append(0.519)
# l2.append(0.129)
# l2.append(0.311)
# l2.append(0.089)
# l2.append(0.266)
# l2.append(0.301)


# T

# l1.append(0.624)
# l1.append(0.535)
# l1.append(0.129)
# l1.append(0.562)
# l1.append(0.141)
# l1.append(0.291)
# l1.append(0.115)
# l1.append(0.515)
# l1.append(0.126)
# l1.append(0.371)
# l1.append(0.189)
# l1.append(0.399)
# l1.append(0.268)
#
# l2.append(1.409)
# l2.append(0.351)
# l2.append(1.031)
# l2.append(0.579)
# l2.append(0.821)
# l2.append(0.756)
# l2.append(0.459)
# l2.append(0.621)
# l2.append(0.269)
# l2.append(0.593)
# l2.append(0.671)
# l2.append(0.619)
# l2.append(0.969)

# l2.append(0.624)
# l2.append(0.535)
# l2.append(0.129)
# l2.append(0.562)
# l2.append(0.141)
# l2.append(0.291)
# l2.append(0.115)
# l2.append(0.515)
# l2.append(0.126)
# l2.append(0.371)
# l2.append(0.189)
# l2.append(0.399)
# l2.append(0.268)


# UTX

# l1.append(0.473)
# l1.append(0.515)
# l1.append(0.138)
# l1.append(0.546)
# l1.append(0.112)
# l1.append(0.231)
# l1.append(0.116)
# l1.append(0.554)
# l1.append(0.111)
# l1.append(0.273)
# l1.append(0.188)
# l1.append(0.041)
# l1.append(0.451)
#
# l2.append(1.013)
# l2.append(0.211)
# l2.append(0.622)
# l2.append(0.345)
# l2.append(0.559)
# l2.append(0.391)
# l2.append(0.389)
# l2.append(0.381)
# l2.append(0.237)
# l2.append(0.272)
# l2.append(0.628)
# l2.append(0.226)
# l2.append(0.521)

# l2.append(0.473)
# l2.append(0.515)
# l2.append(0.138)
# l2.append(0.546)
# l2.append(0.112)
# l2.append(0.231)
# l2.append(0.116)
# l2.append(0.554)
# l2.append(0.111)
# l2.append(0.273)
# l2.append(0.188)
# l2.append(0.041)
# l2.append(0.451)



# WWD

# l1.append(0.584)
# l1.append(0.468)
# l1.append(0.116)
# l1.append(0.676)
# l1.append(0.105)
# l1.append(0.345)
# l1.append(0.143)
# l1.append(0.263)
# l1.append(0.132)
# l1.append(0.566)
# l1.append(0.206)
# l1.append(0.161)
# l1.append(0.234)
#
# l2.append(0.773)
# l2.append(0.224)
# l2.append(0.614)
# l2.append(0.286)
# l2.append(0.584)
# l2.append(0.321)
# l2.append(0.405)
# l2.append(0.255)
# l2.append(0.157)
# l2.append(0.237)
# l2.append(0.711)
# l2.append(0.131)
# l2.append(0.224)

# l2.append(0.584)
# l2.append(0.468)
# l2.append(0.116)
# l2.append(0.676)
# l2.append(0.105)
# l2.append(0.345)
# l2.append(0.143)
# l2.append(0.263)
# l2.append(0.132)
# l2.append(0.566)
# l2.append(0.206)
# l2.append(0.161)
# l2.append(0.234)






# l1.append()
# l1.append()
# l1.append()
# l1.append()
# l1.append()
# l1.append()
# l1.append()
# l1.append()
# l1.append()
# l1.append()
# l1.append()
# l1.append()
# l1.append()
#
# l2.append()
# l2.append()
# l2.append()
# l2.append()
# l2.append()
# l2.append()
# l2.append()
# l2.append()
# l2.append()
# l2.append()
# l2.append()
# l2.append()
# l2.append()



# findtstats(l1, l2)
#
# sys.exit()


# plotskewedhistogram(df,'AAPL_prices','green','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "AAPL" +".png")
# plotskewedhistogram(df,'ABT_prices','red','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "ABTs" +".png")
# plotskewedhistogram(df,'AEM_prices','blue','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "AEM" +".png")
# plotskewedhistogram(df,'AFG_prices','violet','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "AFG" +".png")
#
# plotskewedhistogram(df,'APA_prices','green','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "APA" +".png")
# plotskewedhistogram(df,'CAT_prices','red','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "CAT" +".png")
# plotskewedhistogram(df,'LAKE_prices','blue','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "LAKE" +".png")
# plotskewedhistogram(df,'MCD_prices','violet','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "MCD" +".png")
#
# plotskewedhistogram(df,'MSFT_prices','green','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "MSFT" +".png")
# plotskewedhistogram(df,'ORCL_prices','red','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "ORCL" +".png")
# plotskewedhistogram(df,'SUN_prices','blue','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "SUN" +".png")
# plotskewedhistogram(df,'T_prices','violet','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "T" +".png")
#
# plotskewedhistogram(df,'UTX_prices','green','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "UTX" +".png")
# plotskewedhistogram(df,'WWD_prices','red','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "WWD" +".png")

# plotskewedhistogram(df,'AFG_prices','green','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "AFG" +".png")
# plotskewedhistogram(df,'T_prices','red','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "T" +".png")
# plotskewedhistogram(df,'AAPL_prices','blue','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "AAPL" +".png")
# plotskewedhistogram(df,'ORCL_prices','black','step',"/home/achivuku/PycharmProjects/financedataanalysis/"+ "ORCL" +".png")
# sys.exit()

msemodelfstatmae, msemodelgraph, mseerrorsframe = calculatefstatistic(msemodelrmae,msemodelurmae, 'green')
qmemodelfstatmae, qmemodelgraph, qmeerrorsframe = calculatefstatistic(qmemodelrmae,qmemodelurmae, 'red')

print('msemodelfstatmae',msemodelfstatmae)
print('qmemodelfstatmae',qmemodelfstatmae)


print('qmemodelgraph',qmemodelgraph.edges())
print('qmemodelgraph',qmemodelgraph.nodes())


# graph_pos=nx.circular_layout(msemodelgraph)
# nx.draw_networkx(msemodelgraph, graph_pos, node_size=1500, font_size=8, font_weight='bold', with_labels=True, arrows=True, node_color = 'green')
# plt.tight_layout()
# plt.savefig(MSEGraphPath, format="PNG")
savenetworkxgraph(msemodelgraph, 'green', MSEGraphPath)

# graph_pos = nx.circular_layout(qmemodelgraph)
# # graph_pos = nx.circular_layout(qmemodelgraph)
# nx.draw_networkx(qmemodelgraph, graph_pos, node_size=1500, font_size=8, font_weight='bold', with_labels=True, arrows=True, node_color = 'red')
# plt.tight_layout()
# plt.savefig(QMEGraphPath, format="PNG")
savenetworkxgraph(qmemodelgraph, 'red', QMEGraphPath)

# sys.exit()

mseerrorsframe.to_pickle(MSEErrorsDataframePath)
qmeerrorsframe.to_pickle(QMEErrorsDataframePath)
# QMEErrors = pd.read_pickle("QMEErrors-Graph.pkl")
# QMEErrors.loc['ABT_prices']['SUN_restricted']


with open(MSEModelFstatMAEPath, 'wb') as handle:
    pickle.dump(msemodelfstatmae,handle,protocol=pickle.HIGHEST_PROTOCOL)

with open(QMEModelFstatMAEPath, 'wb') as handle:
    pickle.dump(qmemodelfstatmae,handle,protocol=pickle.HIGHEST_PROTOCOL)
# handle1 = open("QMECauses-Graph.pkl", 'rb')
# QMECauses = pickle.load(handle1)
# QMECauses['B_prices']['LAKE_prices']

# Plot Options :
# https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#networkx.drawing.nx_pylab.draw_networkx1
# https://www.udacity.com/wiki/creating-network-graphs-with-python
# TO DO : Remove duplicate vertices and add Ftest values as edge weights

# https://matplotlib.org/1.2.1/examples/pylab_examples/histogram_demo.html