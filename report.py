from robust import *


#graph = prep_graph(file="datasets/power.gml", file_format="gml")
#ig.plot(graph)


#graph_random = random(graph.copy())
#ig.plot(graph_random)


#communities = method_community(graph=graph, method="louvain")
#ig.plot(communities, mark_groups=True)


#membership = membership_communities(graph=graph, method="louvain")
#print(membership)


#rewire_compl_test = rewire_compl(data=graph, number=100, community=membership, method="louvain", measure="vi")
#print(rewire_compl_test['Measure'])


#rewire_onl_test = rewire_onl(graph=graph, trials=1)
#ig.plot(rewire_onl_test)


#robust_output = robin_robust(graph=graph, graph_random=graph_random)
#for item in robust_output: print(item, "\n", robust_output[item], "\n")


#robust_output = robin_robust(graph=graph, graph_random=graph_random, measure="vi", method="louvain", type="independent")
#plot_robin(graph, robust_output["Mean"], robust_output["MeanRandom"], legend=("real data", "null model"))


#compare_output = robin_compare(graph=graph)
#plot_robin(graph, compare_output["Mean1"], compare_output["Mean2"], legend=("Louvain", "Fast Greedy"))