import igraph as ig
import pandas as pd
import os

def prep_graph(file, file_format='gml', numbers=False, directed=False, header=False, verbose=False):
    file_formats = ["edgelist", "pajek", "ncol", "lgl", "graphml", "dimacs", "graphdb", "gml", "dl", "igraph"]
    if file_format not in file_formats:
        raise ValueError("Invalid file_format. Expected one of: %s" % file_formats)
    
    if verbose:
        print("Detected file format:", file_format)
    
    if file_format == "gml":
        net = ig.Graph.Read_GML(file)
        ind = [v.index for v in net.vs if net.degree(v) == 0]  # isolate node
        net.delete_vertices(ind)
        graph = net.simplify()
    elif file_format == "edgelist" and numbers:
        edge = pd.read_table(file, header=None, dtype=str, quoting=3)
        edge = edge.to_numpy()
        net = ig.Graph.TupleList(edge, directed=directed)
        ind = [v.index for v in net.vs if net.degree(v) == 0]  # isolate node
        net.delete_vertices(ind)
        graph = net.simplify()
    else:
        net = ig.Graph.Read(file, format=file_format, directed=directed)
        ind = [v.index for v in net.vs if net.degree(v) == 0]  # isolate node
        net.delete_vertices(ind)
        graph = net.simplify()

    return graph

def random_graph(graph, verbose=False):
    if verbose:
        print("Randomizing the graph edges.")
    
    z = graph.ecount()  # number of edges
    graph_random = graph.rewire(n=z)

    return graph_random

def method_community(graph,
                     method="louvain",
                     FUN=None,
                     directed=False,
                     weights=None,
                     steps=4,
                     spins=25,
                     e_weights=None,
                     v_weights=None,
                     nb_trials=10,
                     resolution=1,
                     verbose=False):
    method = method.lower()

    if verbose:
        print(f"Applying community method {method}")

    if weights is None and method in ["walktrap", "edgebetweenness", "fastgreedy"]:
        weights = graph.es["weight"]

    if steps == 4 and method == "leadingeigen":
        steps = -1

    if method == "optimal":
        communities = graph.community_optimal(weights=weights)
    elif method == "louvain":
        communities = graph.community_multilevel(weights=weights, return_levels=False)
    elif method == "walktrap":
        communities = graph.community_walktrap(weights=weights, steps=steps).as_clustering()
    elif method == "spinglass":
        communities = graph.community_spinglass(weights=weights, spins=spins)
    elif method == "leadingeigen":
        communities = graph.community_leading_eigenvector(weights=weights, niter=steps)
    elif method == "edgebetweenness":
        communities = graph.community_edge_betweenness(weights=weights, directed=directed).as_clustering()
    elif method == "fastgreedy":
        communities = graph.community_fastgreedy(weights=weights).as_clustering()
    elif method == "labelprop":
        communities = graph.community_label_propagation(weights=weights)
    elif method == "infomap":
        communities = graph.community_infomap(edge_weights=e_weights, vertex_weights=v_weights, trials=nb_trials)
    elif method == "leiden":
        communities = graph.community_leiden(weights=weights, resolution_parameter=resolution)
    elif method == "other":
        communities = FUN(graph, weights)
    else:
        raise ValueError("Invalid community detection method.")

    return communities

def membership_communities(graph,
                           method="louvain",
                           FUN=None,
                           directed=False,
                           weights=None,
                           steps=4,
                           spins=25,
                           e_weights=None,
                           v_weights=None,
                           nb_trials=10,
                           resolution=1):
    method = method.lower()
    communities = method_community(graph=graph, method=method,
                                   FUN=FUN,
                                   directed=directed,
                                   weights=weights,
                                   steps=steps,
                                   spins=spins,
                                   e_weights=e_weights,
                                   v_weights=v_weights,
                                   nb_trials=nb_trials,
                                   resolution=resolution)
    
    return communities.membership

import networkx as nx
from pyvis.network import Network

def plot_graph(graph):
    network = Network(notebook=True)
    network.from_nx(nx.Graph(graph))
    network.show_buttons(filter_=['physics'])
    network.set_options("""
    var options = {
      "nodes": {
        "color": {
          "border": "rgba(46,102,172,1)",
          "background": "rgba(46,102,172,1)",
          "highlight": {
            "border": "rgba(46,102,172,1)",
            "background": "rgba(46,102,172,1)"
          },
          "hover": {
            "border": "rgba(46,102,172,1)",
            "background": "rgba(46,102,172,1)"
          }
        },
        "font": {
          "color": "rgba(0,0,0,1)",
          "size": 12
        }
      },
      "edges": {
        "color": {
          "inherit": true,
          "opacity": 0.8
        }
      },
      "interaction": {
        "zoomView": true
      },
      "physics": {
        "enabled": true
      }
    }
    """)
    return network.show("graph.html")

my_file = os.path.join("datasets", "karate.gml")
graph = prep_graph(file=my_file, file_format="gml")
graph_random = random_graph(graph=graph)
communities = method_community(graph=graph, method="louvain")
membership = membership_communities(graph=graph, method="louvain")
graph_nx = nx.Graph(graph.get_edgelist())
plot_graph(graph_nx)
ig.plot(graph)