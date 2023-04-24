# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 22:18:19 2022

@author: gambi
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community as com
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

def readvillage(path):
    
    import math
    import numpy as np
    import networkx as nx
    
    a = np.genfromtxt (path, delimiter=",")  # The network data is imported as a vector. We need to reshape it into a matrix
    
    a = np.reshape(a,[int(math.sqrt(len(a))),int(math.sqrt(len(a)))])
    
    g = nx.from_numpy_matrix(a)
    
    return g

def plot(G, title="", layout = nx.random_layout, node_color='black', edge_color='#dddddd', node_size=50, width=1):
    
    ''' Plot the nx graph 
       Args:
           G: The graph to plot
           title: Title the plot
           layout: The type of layout. This will be a nx.layout function. Default is nx.random_layout
           node_color: Node color for the plot. Default is black. This can also be a dictionary of numbers corresponding to distinct node colors
           edge_color: Edge color for the plot. Default is light grey
           node_size: Node size for plot. Default is 50
           width: Edge width for plot. Default is 1
           
   
       Returns:
           None
       '''
    pos = layout(G)
    plt.figure()
    ax = plt.gca()
    ax.set_title(title)
    nx.draw(G,pos,node_size=node_size,node_color=node_color,edge_color=edge_color,width=width, cmap=plt.cm.tab20b)
    
class Multiplex:
    '''This class represents multiplex networks.'''
    
    def __init__(self, graphs):
    
        '''
        Args:
            graphs: List of nx graph objects. Each graph in the list will be a layer in the multiplex network
            
    
        Returns:
            None
        '''
        nodes = graphs[0].nodes()
        
        for graph in graphs:
            
            if not graph.nodes() == nodes:
                raise ValueError("The node sets of each layer needs to be the same in a multiplex network")
        
        
        self.layers = graphs
        
        self.nodes = graphs[0].nodes()
        
        self.edges = [graph.edges() for graph in graphs]
        
        for i in range(len(graphs)):
            if not nx.is_connected(graphs[i]):
                print("Warning: In layer " + str(i) + ", the graph is not connected.")
                
        return
    
    def flatten(self, weight_count = False):
        '''Combine all layers into a single network
        
        Args:
            weight_count: This will make the flattened network weighted with the weight of an edge as the number 
                            of times an arc appears over all layers
            
    
        Returns:
            G: The flattened network
            
        The flattened network will create a network on the node set such that two nodes are adjacent in the flattened
        network if they are adjacent on one layer of the multiplex network
        '''
        
        
        edges = []
        for edgeset in self.edges:
            for edge in edgeset:
                edges.append(edge)
                
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(edges)
        
        if weight_count:
            #This will set the edge weight as the 
            counts = nx.adjacency_matrix(self.layers[0]).todense()
            for i in range(1,len(self.layers)):
                counts +=  nx.adjacency_matrix(self.layers[i]).todense()
            
            for edge in G.edges:
                G[edge[0]][edge[1]]['weight'] = counts[edge[0],[edge[1]]]
                
        return G
    
    
    def plots(self, layout = nx.random_layout, node_color='black', edge_color='#dddddd', node_size=50, width=1):
        
        
        '''Plot all layers of the multiplex network
        
        Args:
            layout: The type of layout. This will be a nx.layout function. Default is nx.random_layout
            node_color: Node color for the plot. Default is black. This can also be a dictionary of numbers corresponding to distinct node colors
            edge_color: Edge color for the plot. Default is light grey
            node_size: Node size for plot. Default is 50
            width: Edge width for plot. Default is 1
            
    
        Returns:
            None
        '''
        
        for i in range(len(self.layers)):
            plot(self.layers[i], "Layer " + str(i), layout = layout, node_size=node_size,node_color=node_color,edge_color=edge_color,width=width)
            
        return
    
    def plot(self, i, layout = nx.random_layout, node_color='black', edge_color='#dddddd', node_size=50, width=1):
        
        '''Plot layer i in the multiplex network
        
        Args:
            i: The layer number.
            layout: The type of layout. This will be a nx.layout function. Default is nx.random_layout
            node_color: Node color for the plot. Default is black. This can also be a dictionary of numbers corresponding to distinct node colors
            edge_color: Edge color for the plot. Default is light grey
            node_size: Node size for plot. Default is 50
            width: Edge width for plot. Default is 1
            
    
        Returns:
            None
        '''
        
        if not i in range(len(self.layers)):
            raise ValueError("Index out of range.")
            
        plot(self.layers[i], "Layer " + str(i), layout = layout, node_size=node_size,node_color=node_color,edge_color=edge_color,width=width)
        
        return
    
    def plot_flat(self, layout = nx.random_layout, node_color='black', edge_color='#dddddd', node_size=50, width=1):
        '''Plot the flattened network
        
        Args:
            layout: The type of layout. This will be a nx.layout function. Default is nx.random_layout
            node_color: Node color for the plot. Default is black. This can also be a dictionary of numbers corresponding to distinct node colors
            edge_color: Edge color for the plot. Default is light grey
            node_size: Node size for plot. Default is 50
            width: Edge width for plot. Default is 1
            
    
        Returns:
            None
        '''
        
        G = self.flatten()
        
        plot(G, '''Flattened Network''', layout = layout, node_size=node_size,node_color=node_color,edge_color=edge_color,width=width)
        
        return
    
    def spectral_embedding(self, i, k):
        '''
        Computes the first several diffusion eigenmodes of the network at layer i of the multiplex network.
    
        Args:
            i: The layer number.
            k: The number of non- trivial eigenmodes.
    
        Returns:
            X: An n-by-k numpy array holding the eigenmodes.
    
        Let G be the network on layer i.
        Each column of X holds an eigenmode of G.  Columns will be the eigenmodes associated 
        with increasing positive eigenvalues.
        The entries in each column are ordered according to the natural ordering of the
        nodes in the NetworkX data structure.
        
        From More Clustering Assignment from KSU Math 726 Spring 2022, Taught by Albin and Poggi-Corradini.
        '''
    
        import networkx as nx
        import scipy.sparse.linalg as sla
        
        try:
            G = self.layers[i]
        except:
            print(str(i) + " is not one of the layers in the multiplex network.")
        L = nx.normalized_laplacian_matrix(G).astype(float)
    
        #Get the k smallest eigenvalues and their eigenvectors
        e,X = sla.eigsh(L,k+1,which='SM')
        
        #Split the nodes based on the cluster
        return X[:,1:]
    
    def spectral_cluster(self, i, k, m = 2, random_state=None):
        
        '''
        Computes the spectral cluster of the given layer
    
        Args:
            i: The layer number.
            k: The number of non-trivial eigenmodes.
            m: The number of clusters. Default is 2
            random_state: The seed used for kmeans. Default is None
    
        Returns:
            kmeans.labels_: A list with the cluster number of each node
    
        Let G be the network on layer i.
        Each column of X holds an eigenmode of G.  Columns will be the eigenmodes associated 
        with increasing positive eigenvalues.
        The entries in each column are ordered according to the natural ordering of the
        nodes in the NetworkX data structure.
        
        From More Clustering Assignment from KSU Math 726 Spring 2022, Taught by Albin and Poggi-Corradini.
        '''
    
        from sklearn.cluster import KMeans
        
        #Get the spectral embedding
        X = self.spectral_embedding(i, k)
        
        kmeans = KMeans(n_clusters=m, random_state=random_state)
        kmeans.fit(X)
        
        return kmeans.labels_
    
    def multi_cluster1(self, k, m=2, eps=False, min_samples = 2):
        
        '''
        This will attempt to cluster the nodes considering the contributions from each layer 
        of the multiplex network.
        
        k = number of eigenvalues to compute when clustering by layer
        m = number of clusters in a layer
        '''
        
        
        from sklearn.cluster import DBSCAN
        
        #First let's compute a metric between nodes. The first attempt will build on Hamming distance
        
        if not eps:
            eps = len(self.layers)/5 + 1
            
        n = len(self.nodes)
        
        #We will start all pairs of distinct nodes with a distance of 1 apart.
        X = np.ones((n,n))
        for i in range(n):
            X[i][i] = 0
        
        #Now, for each layer, we will add 1 to the distance if the nodes are in a different cluster
        numlayers = len(self.layers)
        
        for i in range(numlayers):
            clust = self.spectral_cluster(i, k, m)
            
            for x in range(n):
                for y in range(n):
                    if not clust[x] == clust[y]:
                        X[x][y] += 1
                        
        #Now that we have the Hamming distance built, let's use this to cluster the nodes
        db = DBSCAN(eps, metric="precomputed", min_samples = min_samples)
        db.fit(X)
        
        return db.labels_
    
    def multi_cluster2(self, eps=False, min_samples = 2):
        
        '''
        This will attempt to cluster the nodes considering the average distance between nodes between layers
        
        '''
        
        from sklearn.cluster import DBSCAN
        
        #First let's compute a metric between nodes. The first attempt will build on average path length between nodes
        
        if not eps:
            eps = 2
            
        n = len(self.nodes)
        
        #Initialize the distance matrix
        X = np.zeros((n,n))
        
        #Now, for each layer, we will add 1 to the distance if the nodes are in a different cluster
        numlayers = len(self.layers)
        
        for i in range(numlayers):
            
            paths = dict(nx.shortest_path_length(self.layers[i]))
            
            for x in range(n):
                for y in range(n):
                    X[x][y] += paths[x][y] 
        
        X /= len(self.layers)
        #Now that we have the Hamming distance built, let's use this to cluster the nodes
        db = DBSCAN(eps, metric="precomputed")
        db.fit(X)
        
        return db.labels_
    
    def multi_cluster3(self, k, m=2, eps=False, min_samples = 2, linkage='average'):
        
        '''
        This will attempt to cluster the nodes considering the contributions from each layer 
        of the multiplex network.
        
        k = number of eigenvalues to compute when clustering by layer
        m = number of clusters in a layer
        '''
        
        from sklearn.cluster import AgglomerativeClustering
        
        #First let's compute a metric between nodes. The first attempt will build on Hamming distance
        
        if not eps:
            eps = len(self.layers)/5 + 1
            
        n = len(self.nodes)
        
        #We will start all pairs of distinct nodes with a distance of 1 apart.
        X = np.ones((n,n))
        for i in range(n):
            X[i][i] = 0
        
        #Now, for each layer, we will add 1 to the distance if the nodes are in a different cluster
        numlayers = len(self.layers)
        
        for i in range(numlayers):
            clust = self.spectral_cluster(i, k, m)
            
            for x in range(n):
                for y in range(n):
                    if not clust[x] == clust[y]:
                        X[x][y] += 1
                        
        #Now that we have the Hamming distance built, let's use this to cluster the nodes
        ac = AgglomerativeClustering(eps, affinity="precomputed", linkage=linkage)
        ac.fit(X)
        
        return ac.labels_
    
    def multi_cluster4(self, eps=False, min_samples = 2, linkage='average'):
        
        '''
        This will attempt to cluster the nodes considering the average distance between nodes between layers. 
        This method is broken
        
        '''
        
        from sklearn.cluster import AgglomerativeClustering
        
        #First let's compute a metric between nodes. The first attempt will build on average path length between nodes
        
        if not eps:
            eps = 2
            
        n = len(self.nodes)
        
        #Initialize the distance matrix
        X = np.zeros((n,n))
        
        #Now, for each layer, we will add 1 to the distance if the nodes are in a different cluster
        numlayers = len(self.layers)
        
        for i in range(numlayers):
            
            paths = dict(nx.shortest_path_length(self.layers[i]))
            
            for x in range(n):
                for y in range(n):
                    X[x][y] += paths[x][y] 
        
        X /= len(self.layers)
        #Now that we have the Hamming distance built, let's use this to cluster the nodes
        ac = AgglomerativeClustering(eps, affinity="precomputed", linkage=linkage)
        ac.fit(X)
        
        return ac.labels_
    
    def multi_cluster5(self, k, m=2, mds_dim = 2, kmeans_clusters=2):
        
        
        '''Cluster the nodes by building a hamming distance based on the spectral cluster on each layer
    
        Args:
            k: The number of eigenvalues used to compute the clustering for the individual layers
            m: The number of clusters in a layer
            mds_dim: Number of dimensions to immerse the dissimilarities. Default is 2
            kmeans_clusters: The number of clusters to form in the final result
            
    
        Returns:
            kmeans.labels_: A list with the cluster number of each node
    
        '''
        
        
        #First let's compute a metric between nodes. The first attempt will build on Hamming distance
        
            
        n = len(self.nodes)
        
        #We will start all pairs of distinct nodes with a distance of 1 apart.
        X = np.ones((n,n))
        for i in range(n):
            X[i][i] = 0
        
        #Now, for each layer, we will add 1 to the distance if the nodes are in a different cluster
        numlayers = len(self.layers)
        
        for i in range(numlayers):
            clust = self.spectral_cluster(i, k, m)
            
            for x in range(n):
                for y in range(n):
                    if not clust[x] == clust[y]:
                        X[x][y] += 1
                        
                        
        Xhat = MDS(n_components = mds_dim, dissimilarity = 'precomputed').fit_transform(X)
                        
        #Now that we have the Hamming distance built, let's use this to cluster the nodes
        
        kmeans = KMeans(n_clusters= kmeans_clusters, random_state=0).fit(Xhat)

        
        return kmeans.labels_

    def multi_cluster6(self, mds_dim = 2, kmeans_clusters=2):
        

        '''Cluster the nodes by building a hamming distance based on the average distance between nodes on each layer.
        This method assumes the multiplex network is connceted on each layer
    
        Args:
            mds_dim: Number of dimensions to immerse the dissimilarities. Default is 2
            kmeans_clusters: The number of clusters to form in the final result
            
    
        Returns:
            kmeans.labels_: A list with the cluster number of each node
    
        '''
            
        n = len(self.nodes)
        
        #Initialize the distance matrix
        X = np.zeros((n,n))
        
        #Now, for each layer, we will add 1 to the distance if the nodes are in a different cluster
        numlayers = len(self.layers)
        
        for i in range(numlayers):
            
            try:
                paths = dict(nx.shortest_path_length(self.layers[i]))
                
                for x in range(n):
                    for y in range(n):
                        X[x][y] += paths[x][y] 
            except:
                print("This method will not work unless the netowrk is connected")
                return
            
        X /= len(self.layers)
        
        Xhat = MDS(n_components = mds_dim, dissimilarity = 'precomputed').fit_transform(X)
        
        #Now that we have the Hamming distance built, let's use this to cluster the nodes
        
        kmeans = KMeans(n_clusters= kmeans_clusters, random_state=0).fit(Xhat)
        
        return kmeans.labels_
    
    def multi_cluster7(self, mds_dim = 2, kmeans_clusters=2, t1=False, t2=False):
        
        
        '''Cluster the nodes by building a hamming distance based on the number of neighbors in a layer 
        and the number of paths of length 2 between layers
    
        Args:
            mds_dim: Number of dimensions to immerse the dissimilarities. Default is 2
            kmeans_clusters: The number of clusters to form in the final result
            t1: The weighting of the Type 1 connections. If left False (as default), this will be picked automatically
            t2: The weighting of the Type 2 connections. If left False (as default), this will be picked automatically
            
    
        Returns:
            kmeans.labels_: A list with the cluster number of each node
    
        '''
        
        #First let's compute a metric between nodes. The first attempt will build on average path length between nodes
        
            
        n = len(self.nodes)
        numlayers = len(self.layers)
        
        
        #Now, we will compute the similarities. First, we will count the number of connections of each type
        
        #Type 1 connections, adjacent on the same layer
        T1 = np.zeros((n,n))
        
        for x in self.nodes:
            for i in range(numlayers):
                for y in self.layers[i].neighbors(x):
                    T1[x][y] += 1
        
        #Type 2 connections, two nodes share a neighbor this neighbor could be on the same or a different layer
        T2 = np.matmul(T1,T1)
        
        for i in range(n):
            T2[i][i] = 0
        
        #Set t1, t2 if they are not already set. This will keep t2 from dominating
        if (not t1) and (not t1==0):
            t1 = 1/max(T1)
            
        if (not t2) and (not t2==0):
            t2 = 1/max(T2)
            
        #Combine the two connection types with a weighted sum
        T = t1*T1 + t2*T2
        #Now, scale T so no entry can be >= 1.
        T = T/(np.max(T)+1)
        
        #Create the similarity matrix
        #Initialize the similarity matrix
        X = np.identity(n)
        X += T
        
        #Convert the similarity matrix into a distance matrix
        X = 1-X
        
        Xhat = MDS(n_components = mds_dim, dissimilarity = 'precomputed').fit_transform(X)
        #Now that we have the Hamming distance built, let's use this to cluster the nodes
        
        kmeans = KMeans(n_clusters= kmeans_clusters, random_state=0).fit(Xhat)
        
        return kmeans.labels_

def getclust(M,kmeans_clusters = 3, mds_dim = 2,cl = 3, eigs = False, t1 = False, t2 = False, plots = False, layout = nx.circular_layout, node_color='black', edge_color='#dddddd', node_size=50, width=1):
    
    '''This function will get the clusters for M using the multi_cluster5, multi_cluster6, multi_cluster7. 
    This function can also return plots if desired.
    
    args:
    
        Clustering inputs
        
        M: A Multiplex network object
        kmeans_clusters: The number of clusters to form in the final result
        mds_dim: mds_dim: Number of dimensions to immerse the dissimilarities. Default is 2
        cl: The number of clusters in a layer for multi_cluster5
        eigs: The number of eigenvalues to use in multi_cluster5
        t1: The weighting of the Type 1 connections for multi_cluster7. If left False (as default), this will be picked automatically
        t2: The weighting of the Type 2 connections for multi_cluster7. If left False (as default), this will be picked automatically
    
        Plot inputs
        
        plots: If True, this function will plot the clusters. The default is False
        layout: The type of layout. This will be a nx.layout function. Default is nx.random_layout
        node_color: Node color for the plot. Default is black. This can also be a dictionary of numbers corresponding to distinct node colors
        edge_color: Edge color for the plot. Default is light grey
        node_size: Node size for plot. Default is 50
        width: Edge width for plot. Default is 1
    
    returns:
        
        c1: The cluster from multi_cluster5
        c2: The cluster from multi_cluster6
        c3: The cluster from multi_cluster7
        
    
    '''
    
    if not eigs:
        eigs = len(M.layers[0]) - 2
    
    #Compute clusters
    c1 = M.multi_cluster5(eigs, cl,mds_dim = mds_dim, kmeans_clusters = kmeans_clusters)
    c2 = M.multi_cluster6(mds_dim = mds_dim, kmeans_clusters = kmeans_clusters)
    c3 = M.multi_cluster7(mds_dim = mds_dim, kmeans_clusters = kmeans_clusters, t1=t1, t2=t2)

    #Plot Results
    
    if plots:
        #Flattened networks
        M.plot_flat(node_color = c1, layout = layout, edge_color = edge_color, node_size = node_size, width = width)
        M.plot_flat(node_color = c2, layout = layout, edge_color = edge_color, node_size = node_size, width = width)
        M.plot_flat(node_color = c3, layout = layout, edge_color = edge_color, node_size = node_size, width = width)
    
        #C1 Layers
        M.plots(node_color = c1, layout = layout, edge_color = edge_color, node_size = node_size, width = width)
        #C2 Layers
        M.plots(node_color = c2, layout = layout, edge_color = edge_color, node_size = node_size, width = width)
        #C3 Layers
        M.plots(node_color = c3, layout = layout, edge_color = edge_color, node_size = node_size, width = width)
    
    return c1,c2,c3

def convert_cluster(cluster):
    
    '''This is a helper function that will convert a cluster into a partition of nodes
    
    args:
        
        cluster: A list with the cluster number for each node
    
    returns:
        
        communities A partition of the nodes, with each list in the partition as a part
    '''
    
    nums = set(cluster)
    communities = []
    community = []
    
    for clust in nums:
        for i in range(len(cluster)):
            if cluster[i] == clust:
                community.append(i)
        communities.append(community)
        community = []
    
    return communities

def layer_modularity(M, cluster):
    
    '''This function will compute the modularity for each layer of M, using the cluster given.
    
    args:
        
        M: A multiplex network of the Multiplex type
        cluster: A clustering of the nodes in the network, given as a list
        
    returns:
        
        mods: A list of modularities from the layers
        
    '''
    
    c = convert_cluster(cluster)
    mods = []
    
    for i in range(len(M.layers)):
        mods.append(com.modularity(M.layers[i], c))
    
    return mods

def flatten_modularity(M, cluster, weight_count = True):
    
    '''This function will compute the modularity for the flattened M.
    
    args:
        
        M: A multiplex network of the Multiplex type
        cluster: A clustering of the nodes in the network, given as a list
        weight_count: This will find the modularity assuming that the weight of the flattened edge is 
                        the count of an edge over all layers
    
    returns:
        
        mod: The modularity of the flattened network
    
    '''
    
    c = convert_cluster(cluster)
    G = M.flatten(weight_count = weight_count)
    mod = com.modularity(G, c, weight='weight')
    
    return float(mod)

def load(name):
    '''This will load the village HH1 data as a multiplex network
    
    inputs: 
        
    name: A string containing the name of the village, ie "HH_1"
    
    outputs
    
    M: The multitool.Multiplex object representing the village
    '''
    
    name = "Village/" + name
    paths = [name + "_borrowmoney.csv", name + "_giveadvice.csv", name + "_helpdecision.csv", name + "_keroricecome.csv",
             name + "_keroricego.csv", name + "_lendmoney.csv", name + "_medic.csv", name + "_nonrel.csv",
             name + "_rel.csv", name + "_templecompany.csv", name + "_visitcome.csv", name + "_visitgo.csv"]
    
    graphs = [readvillage(path) for path in paths]
    M = Multiplex(graphs)
    
    return M