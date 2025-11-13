# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np




####### GRAPH 1 #######

#G = nx.Graph() # simple undirected graph
# G = nx.Digraph() # directed graph
# G = nx.MultiGraph() # multiple edges between two nodes
# G = nx.MultiDiGraph() # multiple edges between two nodes in a directed manner


#1-2-3 (ONE GRAPH)
# G.add_edge(1, 2) #there is an edge between node 1 and node 2 - if nodes don't exist, they will be created!
# G.add_edge(2, 3, weight=0.9) #higher the weight, the stronger the edge (more difficult/longer to get from 2 to 3)

# # A-B-B (ONE GRAPH)
# G.add_edge("A", "B")
# G.add_edge("B", "B")

# # C-D (ANOTHER GRAPH)
# G.add_edge("C", "D")

# #print is another graph
# G.add_node(print) 

# nx.draw_spring(G, with_labels=True)
# plt.savefig('graph_output.png', dpi=300, bbox_inches='tight')
# print("Graph saved to graph_output.png")
# # plt.show()  # Uncomment if you have a display available





####### GRAPH 2 #######
#G = nx.from_numpy_array(np.array([[0,1,0],
#          [1,0,1],
#          [0,1,0]]))




#### GRAPH 3 #####
# G = nx.Graph()
# edge_list = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
# G.add_edges_from(edge_list)




# nx.draw_spring(G, with_labels=True)
# plt.savefig('graph_output3.png', dpi=300, bbox_inches='tight')
# print("Graph saved to graph_output3.png")
# # plt.show()  # Uncomment if you have a display available
