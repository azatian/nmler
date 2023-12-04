import navis
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import plotly.express as px
import plotly.graph_objects as go
from nmler import core

'''
custom function to map monkey-v1 axon tracings from bin2 to bin1 space
this means multiplying all node coordinates by 2
'''
def rewriter(input, output):
    _tree = ET.parse(input)
    root = _tree.getroot()

    for element in root:
        for x in element:
            if x.tag == "nodes":
                for y in x:
                    y.set("x", str(int(float(y.get("x"))*2)))
                    y.set("y", str(int(float(y.get("y"))*2)))
                    y.set("z", str(int(float(y.get("z"))*2)))

    _tree.write(output)
    return

'''
returns a Navis NeuronList, extracting all skeleton metadata from the nml file
'''
def get_neurons(input):
    _tree = ET.parse(input)
    root = _tree.getroot()
    neurons = navis.core.NeuronList([])
    # Copy the attributes dict
    for element in root:
        if element.tag == 'thing':
            nodes = pd.DataFrame.from_records([n.attrib for n in element[0]])
            edges = pd.DataFrame.from_records([n.attrib for n in element[1]])
            #edges = edges.astype(self._dtypes['node_id'])
            if (len(nodes) != 0) & (len(edges) != 0):
                nodes.rename({'id': 'node_id'}, axis=1, inplace=True)
                
                G = nx.Graph()
                G.add_edges_from(edges.values)

                tree = nx.bfs_tree(G, list(G.nodes)[0])

                edges = pd.DataFrame(list(tree.edges), columns=['source', 'target'])

                nodes['parent_id'] = edges.set_index('target').reindex(nodes.node_id.values).source.values
                nodes['parent_id'] = nodes.parent_id.fillna(-1)
                nodes["node_id"] = nodes["node_id"].astype(int)

                nodes.sort_values('node_id', inplace=True)

                nodes["node_id"] = nodes["node_id"].astype(int)
                nodes["parent_id"] = nodes["parent_id"].astype(int)
                nodes["x"] = nodes["x"].astype(int)
                nodes["y"] = nodes["y"].astype(int)
                nodes["z"] = nodes["z"].astype(int)
                nodes["radius"] = nodes["radius"].astype(float)
                nodes["rotX"] = nodes["rotX"].astype(float)
                nodes["rotY"] = nodes["rotY"].astype(float)
                nodes["rotZ"] = nodes["rotZ"].astype(float)

                _tree = navis.core.TreeNeuron(nodes)

                neurons.append(_tree)

    return neurons


'''
gets first principal component and means of all neurons in a neuron list
returns a dataframe
'''
def calculate_pc(nl):
    index_to_pc = {}
    index_to_mean = {}
    counter = 0
    for x in nl:
        x_p = x.nodes["x"].tolist()
        y_p = x.nodes["y"].tolist()
        z_p = x.nodes["z"].tolist()
        
        x_p = np.array(x_p)
        y_p = np.array(y_p)
        z_p = np.array(z_p)
        
        data = np.concatenate((x_p[:, np.newaxis], 
                        y_p[:, np.newaxis], 
                        z_p[:, np.newaxis]), 
                        axis=1)
        
        data = data.astype(np.float64)
        
        
        datamean = data.mean(axis=0)
        
        uu, dd, vv = np.linalg.svd(data - datamean)

        index_to_pc[counter] = vv[0]
        index_to_mean[counter] = datamean

        counter += 1 

    first_pcs = pd.DataFrame.from_dict(index_to_pc, orient="index")
    means = pd.DataFrame.from_dict(index_to_mean, orient="index")

    return first_pcs, means

'''
gets best fit line from first prinicipal component(fpc)
'''
def best_fit(pcs, datameans, name, NeuronNumber):    
    linepts = pcs* np.mgrid[-700:700:6j][:, np.newaxis]
    linepts += datameans
    
    area = {'Area': pd.Series(str(name), index=range(len(linepts)))}
    lineNumber = {str(NeuronNumber): pd.Series(str(NeuronNumber), index=range(len(linepts)))}
    
    linepts = np.hstack((pd.DataFrame(lineNumber), pd.DataFrame(linepts), pd.DataFrame(area)))
    linepts = pd.DataFrame(linepts, columns=('Neuron', 'x', 'y', 'z', 'Area'))
    
    return linepts

'''
gets all the best fit lines from every fpc
'''
def best_fit_all(pcs, datameans, name):

    assert len(pcs) == len(datameans)
    best_fit_0 = core.best_fit(np.array(pcs.loc[0,:]), np.array(datameans.loc[0,:]), str(name), "0")

    def best_fit_(x):
        best_fit = core.best_fit(np.array(pcs.loc[x,:]), np.array(datameans.loc[x,:]), str(name), str(x))
        return best_fit

    arrays = [best_fit_0] + [best_fit_(i) for i in range(1, len(pcs))]
    best_fit_all = np.vstack(tuple(arrays))
                                 
    best_fit_all = pd.DataFrame(best_fit_all, columns=('Neuron', 'x', 'y', 'z', 'Area'))
    return best_fit_all


'''
Print fig of pcs
'''
def fig_pcs(pcs):

    fig_3d = go.Figure(data=[go.Scatter3d(x=pcs[0], y=pcs[1], z=pcs[2],
                        mode='markers')])
    
    fig_xy = px.scatter(x=pcs[0], y=pcs[1],
                        labels={"x":"x", "y":"y"})
    fig_yz = px.scatter(x=pcs[1], y=pcs[2],
                        labels={"x":"y", "y":"z"})
    fig_zx = px.scatter(x=pcs[2], y=pcs[0],
                        labels={"x":"z", "y":"x"})

    return fig_3d, fig_xy, fig_yz, fig_zx

'''
print fig of first principle components for all annotators
'''
def fig_pcs_all (n_1, n_2, n_3, n_4):
    pcs_1, datameans_1 = core.calculate_pc(n_1)
    pcs_2, datameans_2 = core.calculate_pc(n_2)
    pcs_3, datameans_3 = core.calculate_pc(n_3)
    pcs_4, datameans_4 = core.calculate_pc(n_4)

    n1 = {'x': pcs_1[0], 'y':pcs_1[1], 'z':pcs_1[2], 'Area': pd.Series("V1_Lower", index=range(len(pcs_1)))}
    n2 = {'x': pcs_2[0], 'y':pcs_2[1], 'z':pcs_2[2], 'Area': pd.Series("V1_White_Matter", index=range(len(pcs_2)))}
    n3 = {'x': pcs_3[0], 'y':pcs_3[1], 'z':pcs_3[2], 'Area': pd.Series("V2-White_Matter", index=range(len(pcs_3)))}
    n4 = {'x': pcs_4[0], 'y':pcs_4[1], 'z':pcs_4[2], 'Area': pd.Series("V1_Upper", index=range(len(pcs_4)))}
    
    n_al = np.vstack((pd.DataFrame(n1), pd.DataFrame(n2), pd.DataFrame(n3), pd.DataFrame(n4)))
    numbers = pd.DataFrame(list(range(len(n_al))))
    n_all = pd.DataFrame(np.hstack((n_al, numbers)), columns=('x', 'y', 'z', 'Area','Neuron'))
    
    dm1 = {'x': datameans_1[0], 'y':datameans_1[1], 'z':datameans_1[2], 'Area': pd.Series("V1_Lower", index=range(len(datameans_1)))}
    dm2 = {'x': datameans_2[0], 'y':datameans_2[1], 'z':datameans_2[2], 'Area': pd.Series("V1_Lower", index=range(len(datameans_2)))}
    dm3 = {'x': datameans_3[0], 'y':datameans_3[1], 'z':datameans_3[2], 'Area': pd.Series("V1_Lower", index=range(len(datameans_3)))}
    dm4 = {'x': datameans_4[0], 'y':datameans_4[1], 'z':datameans_4[2], 'Area': pd.Series("V1_Lower", index=range(len(datameans_4)))}

    dm_al = np.vstack((pd.DataFrame(dm1), pd.DataFrame(dm2), pd.DataFrame(dm3), pd.DataFrame(dm4)))
    dm_all = pd.DataFrame(np.hstack((dm_al, numbers)), columns=('x', 'y', 'z', 'Area','Neuron'))

    nall = pd.DataFrame(np.delete(n_al, 3, 1))
    dmall = pd.DataFrame(np.delete(dm_al, 3, 1))
    
    fig_3d = px.scatter_3d(n_all, x='x',y='y',z='z',color='Neuron')

    fig_xy = px.scatter(n_all, x="x", y="y", color="Area")
    fig_yz = px.scatter(n_all, x="y", y="z", color="Area")
    fig_zx = px.scatter(n_all, x="z", y="x", color="Area")

    return n_all, dm_all, fig_3d, fig_xy, fig_yz, fig_zx, dmall, nall

'''
Print best fit line for singular neurons
'''
def fig_best_fit(best_fit):
    
    fig_3d = px.line_3d(best_fit, x="x", y="y", z="z", color='Neuron')

    fig_xy = px.line(best_fit, x="x", y="y", color='Neuron')
    fig_yz = px.line(best_fit, x="y", y="z", color='Neuron')
    fig_zx = px.line(best_fit, x="z", y="x", color='Neuron')

    return fig_3d, fig_xy, fig_yz, fig_zx
    

'''
Histogram of cable lengths
'''
def histo_cable_len(nl):
    plt = sns.histplot(x=nl.cable_length, kde=True)
    return plt

'''
Violin plots of cable lengths
'''
def vio_cable_len(nl):
    print(nl)
    print(nl.cable_length)
    plt = sns.violinplot(x=nl.cable_length)
    return plt

def vio_cable_len_all(n_br, n_km, n_ko, n_ss):
    n1 = {'cable_lengths': pd.Series(n_br.cable_length),
            'Area': pd.Series("V1_Lower", index=range(len(n_br.cable_length)))}
    n2 = {'cable_lengths': pd.Series(n_km.cable_length),
            'Area': pd.Series("V1_White_Matter", index=range(len(n_km.cable_length)))}
    n3 = {'cable_lengths': pd.Series(n_ko.cable_length),
            'Area': pd.Series("V2_White_Matter", index=range(len(n_ko.cable_length)))}
    n4 = {'cable_lengths': pd.Series(n_ss.cable_length),
            'Area': pd.Series("V1-Upper", index=range(len(n_ss.cable_length)))}

    n_12 = np.vstack((pd.DataFrame(n1), pd.DataFrame(n2)))
    n_34 = np.vstack((pd.DataFrame(n3), pd.DataFrame(n4)))
    n_all = pd.DataFrame(np.vstack((pd.DataFrame(n_12), pd.DataFrame(n_34))),
                         columns=('Cable_Lengths', 'Area'))
    
    fig = px.violin(x=n_all.loc[:,'Area'], y=n_all.loc[:,'Cable_Lengths'],
                     labels={"x":"Area", "y":"Cable Lengths"},
                     box=True, points="all")
    fig.show()

    return n_all, fig
