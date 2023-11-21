import navis
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

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
                    y.set("x", str(int(y.get("x"))*2))
                    y.set("y", str(int(y.get("y"))*2))
                    y.set("z", str(int(y.get("z"))*2))

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
gets best fit line from first prinicipal component
'''
def best_fit(pc, datamean):
    linepts = pc * np.mgrid[-700:700:8j][:, np.newaxis]
    linepts += datamean
    return linepts






