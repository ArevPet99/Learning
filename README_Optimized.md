Python Network Structures and Shortest Path Algorithms
This project provides a pure Python implementation of a versatile network data structure along with tools for solving graph-based optimization problems. It demonstrates how to represent networks and solve problems like the shortest path using both a direct network object implementation and a traditional tabular (matrix-based) implementation.

Features ✨
Network Class: A robust Network class to build weighted graphs, supporting both directed and undirected modes.

Weighted Edges: Easily add edges with associated weights, which are essential for many optimization problems.

Adjacency Matrix: Generate a weighted vertex-to-vertex adjacency matrix from any network. Non-existent edges are represented by infinity.

Incidence Matrix: Generate a standard vertex-to-edge incidence matrix.

Shortest Path Solvers: Includes implementations of Dijkstra's algorithm to find the single-source shortest path:

shortest_path_network(): Operates directly on the Network object.

shortest_path_tabular(): Operates on the weighted adjacency matrix.

No Dependencies: The entire implementation uses standard Python libraries.

How to Use
The following examples demonstrate how to use the Network class and the shortest path functions.

1. Creating a Weighted Network
Instantiate the Network class and add edges with weights.

from network_structures import Network

# Create an undirected network
# The 'is_directed' flag can be set to True for directed graphs
net = Network(is_directed=False)

# Add edges with corresponding weights
net.add_edge('A', 'B', weight=7)
net.add_edge('A', 'C', weight=9)
net.add_edge('C', 'F', weight=2)
net.add_edge('F', 'E', weight=9)

2. Generating Matrices
You can convert the network object into its matrix representations.

# Generate the weighted adjacency matrix
# 'inf' represents no direct path between nodes.
adj_matrix, nodes = net.to_adjacency_matrix()

# Generate the incidence matrix
# Note: The incidence matrix does not store weight information.
inc_matrix, nodes, edges = net.to_incidence_matrix()

3. Solving the Shortest Path Problem
Find the shortest path between two nodes using either the network or the tabular approach.

from network_structures import shortest_path_network, shortest_path_tabular

start_node = 'A'
end_node = 'E'

# Method 1: Using the network object directly
path, distance = shortest_path_network(net, start_node, end_node)
print(f"Network-based Path: {path}")
print(f"Distance: {distance}")
# Expected Path: ['A', 'C', 'F', 'E']
# Expected Distance: 20


# Method 2: Using the adjacency matrix
adj_matrix, nodes = net.to_adjacency_matrix()
path_tab, dist_tab = shortest_path_tabular(adj_matrix, nodes, start_node, end_node)
print(f"Tabular-based Path: {path_tab}")
print(f"Distance: {dist_tab}")
# Expected Path: ['A', 'C', 'F', 'E']
# Expected Distance: 20

Running the Demonstration
To see a full demonstration and verification that both shortest path methods produce the same result, you can run the Python script directly from your terminal.

python network_structures.py

This will execute the example code within the if __name__ == '__main__': block, print the results, and confirm the correctness of the algorithms.
