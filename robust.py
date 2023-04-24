import igraph as ig
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt


def prep_graph(file, file_format='gml'):
    net = ig.Graph.Read(f=file, format=file_format)
    ind = [v.index for v in net.vs if net.degree(v) == 0]  # isolate node
    net.delete_vertices(ind)
    graph = net.simplify()
    return graph

graph = prep_graph(file="datasets/football.gml", file_format="gml")
#ig.plot(graph)


def random(graph):
    z = graph.ecount()
    graph_random = graph.to_networkx()
    nx.double_edge_swap(graph_random, nswap=z, max_tries=1e75)
    graph_random = ig.Graph.from_networkx(graph_random)
    return graph_random

graph_random = random(graph.copy())
#ig.plot(graph_random)


def method_community(graph,
                     method="louvain",
                     directed=False,
                     weights=None,
                     steps=4,
                     spins=25,
                     e_weights=None,
                     v_weights=None,
                     nb_trials=10,
                     resolution=1):
    method = method.lower()

    if steps == 4 and method == "leadingeigen":
        steps = -1

    if method == "optimal":
        communities = graph.community_optimal_modularity(weights=weights)
    elif method == "louvain":
        communities = graph.community_multilevel(weights=weights, return_levels=False)
    elif method == "walktrap":
        communities = graph.community_walktrap(weights=weights, steps=steps).as_clustering()
    elif method == "spinglass":
        communities = graph.community_spinglass(weights=weights, spins=spins)
    elif method == "leadingeigen":
        communities = graph.community_leading_eigenvector(weights=weights)
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
    else:
        raise ValueError("Invalid community detection method.")

    return communities

#communities = method_community(graph=graph, method="louvain")
#ig.plot(communities, mark_groups=True)


def membership_communities(graph,
                           method="louvain",
                           directed=False,
                           weights=None,
                           steps=4,
                           spins=25,
                           e_weights=None,
                           v_weights=None,
                           nb_trials=10,
                           resolution=1):
    
    communities = method_community(graph=graph, method=method,
                                   directed=directed,
                                   weights=weights,
                                   steps=steps,
                                   spins=spins,
                                   e_weights=e_weights,
                                   v_weights=v_weights,
                                   nb_trials=nb_trials,
                                   resolution=resolution)
    
    return communities.membership

#membership = membership_communities(graph=graph, method="louvain")
#print(membership)


def rewire_compl(data, number, community,
                 method="louvain",
                 measure="vi",
                 directed=False,
                 weights=None,
                 steps=4,
                 spins=25,
                 e_weights=None,
                 v_weights=None,
                 nb_trials=10,
                 resolution=1):
    method = method.lower()
    measure = measure.lower()
    
    graph_data = data.to_networkx()
    nx.double_edge_swap(graph_data, nswap=number, max_tries=1e75)
    graph_data = ig.Graph.from_networkx(graph_data)
    data = graph_data

    com_r = membership_communities(graph=data, method=method,
                                   directed=directed, weights=weights, steps=steps,
                                   spins=spins, e_weights=e_weights,
                                   v_weights=v_weights, nb_trials=nb_trials,
                                   resolution=resolution)
    
    measure_result = ig.compare_communities(community, com_r, method=measure)
    output = {'Measure': measure_result, 'graph_rewire': data}
    
    return output

#rewire_compl_test = rewire_compl(data=graph, number=100, community=membership, method="louvain", measure="vi")
#print(rewire_compl_test['Measure'])


def rewire_onl(graph, trials):
    graph_data = graph.to_networkx()
    nx.double_edge_swap(graph_data, nswap=trials, max_tries=1e75)
    graph_data = ig.Graph.from_networkx(graph_data)
    graph = graph_data
    return graph

#rewire_onl_test = rewire_onl(graph=graph, trials=1)
#ig.plot(rewire_onl_test)


def robin_robust(graph, graph_random,
                 method=None,
                 measure="vi",
                 type="independent",
                 directed=False,
                 weights=None,
                 steps=4,
                 spins=25,
                 e_weights=None,
                 v_weights=None,
                 nb_trials=10,
                 resolution=1):

    if method is None:
        method = "louvain"

    available_methods = ["walktrap", "edgebetweenness", "fastgreedy", "louvain", "spinglass",
                             "leadingeigen", "labelprop", "infomap", "optimal", "leiden"]
    
    if method not in available_methods:
        raise ValueError("Invalid method. Available methods: %s" % available_methods)

    measure = measure.lower()
    type = type.lower()
    method = method.lower()
    nrep = 10
    com_real = membership_communities(graph=graph, method=method,
                                      directed=directed,
                                      weights=weights,
                                      steps=steps,
                                      spins=spins,
                                      e_weights=e_weights,
                                      v_weights=v_weights,
                                      nb_trials=nb_trials,
                                      resolution=resolution)  # real network

    com_random = membership_communities(graph=graph_random, method=method,
                                        directed=directed,
                                        weights=weights,
                                        steps=steps,
                                        spins=spins,
                                        e_weights=e_weights,
                                        v_weights=v_weights,
                                        nb_trials=nb_trials,
                                        resolution=resolution)  # random network

    de = graph.ecount()
    N = graph.vcount()
    Measure = None
    vector = [None] * nrep
    vect_random = [None] * nrep
    graph_rewire = None
    count = 1
    n_rewire = list(range(0, 61, 5))

    #INDEPENDENT
    if type == "independent":
        
        #OUTPUT MATRIX
        measure_real = np.zeros((nrep ** 2, len(n_rewire)))
        measure_random = np.zeros((nrep ** 2, len(n_rewire)))
        mean_random = np.zeros((nrep, len(n_rewire)))
        mean_ = np.zeros((nrep, len(n_rewire)))
        vet1 = list(range(5, 65, 5))
        vet = [round(x * de / 100) for x in vet1]

        for z in vet:
            count2 = 0
            count += 1
            for s in range(nrep):
                count2 += 1
                k = 0
                #REAL
                real = rewire_compl(data=graph,
                                    number=z,
                                    community=com_real,
                                    method=method,
                                    measure=measure,
                                    directed=directed,
                                    weights=weights,
                                    steps=steps,
                                    spins=spins,
                                    e_weights=e_weights,
                                    v_weights=v_weights,
                                    nb_trials=nb_trials,
                                    resolution=resolution)
                if measure == "vi":
                    vector[k] = (real["Measure"]) / np.log2(N)
                elif measure == "split.join":
                    vector[k] = (real["Measure"]) / (2 * N)
                else:
                    vector[k] = 1 - (real["Measure"])

                measure_real[count2 - 1, count - 1] = vector[k]
                graph_rewire = real["graph_rewire"]

                #RANDOM
                random = rewire_compl(data=graph_random,
                                    number=z,
                                    community=com_random,
                                    method=method,
                                    measure=measure,
                                    directed=directed,
                                    weights=weights,
                                    steps=steps,
                                    spins=spins,
                                    e_weights=e_weights,
                                    v_weights=v_weights,
                                    nb_trials=nb_trials,
                                    resolution=resolution)
                if measure == "vi":
                    vect_random[k] = (random["Measure"]) / np.log2(N)
                elif measure == "split.join":
                    vect_random[k] = (random["Measure"]) / (2 * N)
                else:
                    vect_random[k] = 1 - (random["Measure"])

                measure_random[count2 - 1, count - 1] = vect_random[k]
                graph_rewire_random = random["graph_rewire"]

                for k in range(1, nrep):
                    count2 += 1
                    real = rewire_compl(data=graph_rewire,
                                        number=round(0.01 * de),
                                        community=com_real,
                                        method=method,
                                        measure=measure,
                                        directed=directed,
                                        weights=weights,
                                        steps=steps,
                                        spins=spins,
                                        e_weights=e_weights,
                                        v_weights=v_weights,
                                        nb_trials=nb_trials,
                                        resolution=resolution)
                    if measure == "vi":
                        vector[k] = (real["Measure"]) / np.log2(N)
                    elif measure == "split.join":
                        vector[k] = (real["Measure"]) / (2 * N)
                    else:
                        vector[k] = 1 - (real["Measure"])
                    measure_real[count2 - 1, count - 1] = vector[k]
                    random = rewire_compl(data=graph_rewire_random,
                                            number=round(0.01 * de),
                                            community=com_random,
                                            method=method,
                                            measure=measure,
                                            directed=directed,
                                            weights=weights,
                                            steps=steps,
                                            spins=spins,
                                            e_weights=e_weights,
                                            v_weights=v_weights,
                                            nb_trials=nb_trials,
                                            resolution=resolution)
                    if measure == "vi":
                        vect_random[k] = (random["Measure"]) / np.log2(N)
                    elif measure == "split.join":
                        vect_random[k] = (random["Measure"]) / (2 * N)
                    else:
                        vect_random[k] = 1 - (random["Measure"])
                    measure_random[count2 - 1, count - 1] = vect_random[k]
                mean_random[s, count - 1] = np.mean(vect_random)
                mean_[s, count - 1] = np.mean(vector)

    # DEPENDENT
    else:
        z = round((5 * de) / 100, 0)  # the 5% of the edges
        measure_real = np.zeros((nrep, nrep))
        measure_real_1 = None
        measure_random = np.zeros((nrep, nrep))
        measure_random_1 = None
        mean_random = np.zeros(nrep)
        mean_ = np.zeros(nrep)
        mean_random_1 = None
        mean_1 = None
        diff = None
        diff_r = None
        vet = [z] * (len(n_rewire) - 1)

        for z in vet:
            count2 = 0
            count += 1

            for s in range(nrep):
                count2 += 1
                k = 0

                # REAL
                graph_rewire = rewire_onl(data=graph, number=z)
                graph_rewire = ig.union(graph_rewire, diff)
                comr = membership_communities(graph=graph_rewire,
                                              method=method,
                                              directed=directed,
                                              weights=weights,
                                              steps=steps,
                                              spins=spins,
                                              e_weights=e_weights,
                                              v_weights=v_weights,
                                              nb_trials=nb_trials,
                                              resolution=resolution)
                Measure = ig.compare_communities(com_real, comr, method=measure)
                if measure == "vi":
                    vector[k] = Measure / np.log2(N)
                elif measure == "split.join":
                    vector[k] = Measure / (2 * N)
                else:
                    vector[k] = 1 - Measure
                measure_real_1[count2 - 1] = vector[k]

                graph_nx = graph.to_networkx()
                graph_rewire_nx = graph_rewire.to_networkx()
                diff = nx.difference(graph_nx, graph_rewire_nx)
                diff = ig.Graph.from_networkx(diff)

                # RANDOM
                graph_rewire_random = rewire_onl(data=graph_random, number=z)
                graph_rewire_random = ig.union(graph_rewire_random, diff_r)
                comr = membership_communities(graph=graph_rewire_random,
                                              method=method,
                                              directed=directed,
                                              weights=weights,
                                              steps=steps,
                                              spins=spins,
                                              e_weights=e_weights,
                                              v_weights=v_weights,
                                              nb_trials=nb_trials,
                                              resolution=resolution)
                Measure = ig.compare_communities(com_random, comr, method=measure)
                if measure == "vi":
                    vect_random[k] = Measure / np.log2(N)
                elif measure == "split.join":
                    vect_random[k] = Measure / (2 * N)
                else:
                    vect_random[k] = 1 - Measure
                measure_random_1[count2 - 1] = vect_random[k]

                graph_nx = graph_random.to_networkx()
                graph_rewire_nx = graph_rewire_random.to_networkx()
                diff_r = nx.difference(graph_nx, graph_rewire_nx)
                diff_r = ig.Graph.from_networkx(diff_r)
                
                for k in range(1, nrep):
                    count2 += 1

                    # REAL
                    real = rewire_compl(data=graph_rewire, number=round(0.01 * de),
                                        method=method,
                                        measure=measure,
                                        community=com_real,
                                        directed=directed,
                                        weights=weights,
                                        steps=steps,
                                        spins=spins,
                                        e_weights=e_weights,
                                        v_weights=v_weights,
                                        nb_trials=nb_trials,
                                        resolution=resolution)
                    if measure == "vi":
                        vector[k] = (real["Measure"]) / np.log2(N)
                    elif measure == "split.join":
                        vector[k] = (real["Measure"]) / (2 * N)
                    else:
                        vector[k] = 1 - (real["Measure"])
                    measure_real_1[count2 - 1] = vector[k]

                    # RANDOM
                    random = rewire_compl(data=graph_rewire_random, number=round(0.01 * de),
                                          method=method,
                                          measure=measure,
                                          community=com_random,
                                          directed=directed,
                                          weights=weights,
                                          steps=steps,
                                          spins=spins,
                                          e_weights=e_weights,
                                          v_weights=v_weights,
                                          nb_trials=nb_trials,
                                          resolution=resolution)
                    if measure == "vi":
                        vect_random[k] = (random["Measure"]) / np.log2(N)
                    elif measure == "split.join":
                        vect_random[k] = (random["Measure"]) / (2 * N)
                    else:
                        vect_random[k] = 1 - (random["Measure"])
                    measure_random_1[count2 - 1] = vect_random[k]

                mean_1[s] = np.mean(measure_real_1)
                mean_random_1[s] = np.mean(measure_random_1)
            
            graph = ig.intersection(graph, graph_rewire)
            graph_random = ig.intersection(graph_random, graph_rewire_random)
            measure_random = np.column_stack((measure_random, measure_random_1))
            measure_real = np.column_stack((measure_real, measure_real_1))
            mean_ = np.column_stack((mean_, mean_1))
            mean_random = np.column_stack((mean_random, mean_random_1))

    measure_random = pd.DataFrame(measure_random, columns=n_rewire)
    measure_real = pd.DataFrame(measure_real, columns=n_rewire)
    mean_random = pd.DataFrame(mean_random, columns=n_rewire)
    mean_ = pd.DataFrame(mean_, columns=n_rewire)
    output = {
        "Mean": mean_,
        "MeanRandom": mean_random
    }

    return output

#output = robin_robust(graph=graph, graph_random=graph_random, measure="vi", method="louvain", type="independent")
#for item in output: print(item, "\n", output[item], "\n")


def plot_robin(graph, model1, model2, legend=("model1", "model2"), title="Robin plot"):
    mvimodel1 = model1.mean(axis=0)
    mvimodel2 = model2.mean(axis=0)

    perc_pert = [i / 100 for i in range(0, 61, 5)] * 2
    mvi = list(mvimodel1) + list(mvimodel2)
    model = [legend[0]] * 13 + [legend[1]] * 13

    data_frame = {'percPert': perc_pert, 'mvi': mvi, 'model': model}
    
    for m, mvi_values in zip(legend, [mvimodel1, mvimodel2]):
        plt.plot(perc_pert[:13], mvi_values, marker='o', label=m)

    plt.xlabel("Percentage of perturbation")
    plt.ylabel("Measure")
    plt.ylim(0, 0.5)
    plt.title(title)
    plt.legend()
    plt.show()

#output = robin_robust(graph=graph, graph_random=graph_random, measure="vi", method="louvain", type="independent")
#plot_robin(graph, output["Mean"], output["MeanRandom"], legend=("real data", "null model"))