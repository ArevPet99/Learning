
# Python Network and Matrix Structures

This project provides a Python implementation of a network data structure and includes functions for converting between this structure and its corresponding tabular forms: the **adjacency matrix** and the **incidence matrix**. The implementation supports both **directed** and **undirected** networks.

## Features ✨

-   **Network Class**: A versatile `Network` class to build graphs by adding nodes and edges.
-   **Directed & Undirected**: Easily specify whether a network is directed or undirected.
-   **Adjacency Matrix**: Generate a vertex-to-vertex adjacency matrix from any network object.
-   **Incidence Matrix**: Generate a vertex-to-edge incidence matrix from any network object.
-   **Reconstruction**: Create a network object from an existing adjacency or incidence matrix.
-   **Pure Python**: Written in standard Python with no external dependencies required.

---

## How to Use

Below are examples of how to use the `Network` class and the conversion functions.

### Creating a Network

You can create either an undirected or a directed network.

```python
from network_structures import Network

# Create an undirected network
undirected_net = Network(is_directed=False)
undirected_net.add_edge('A', 'B')
undirected_net.add_edge('B', 'C')

# Create a directed network
directed_net = Network(is_directed=True)
directed_net.add_edge('1', '2')
directed_net.add_edge('2', '1') # A separate edge from 2 to 1
```

### Generating Matrices from a Network

Convert your `Network` object into its matrix representations.

```python
# Generate the adjacency matrix for the undirected network
adj_matrix, nodes = undirected_net.to_adjacency_matrix()
# adj_matrix is a 2D list, nodes is the list of corresponding labels

# Generate the incidence matrix
inc_matrix, nodes, edges = undirected_net.to_incidence_matrix()
# inc_matrix is a 2D list, nodes are rows, edges are columns
```

### Recreating a Network from a Matrix

You can also build a network from existing matrix data.

```python
from network_structures import from_adjacency_matrix, from_incidence_matrix

# Example adjacency data
nodes = ['A', 'B', 'C']
matrix = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

# Recreate the network
net_from_adj = from_adjacency_matrix(matrix, nodes, is_directed=False)
print(net_from_adj)
# Expected output: Network(nodes=['A', 'B', 'C'], edges=[('A', 'B'), ('B', 'C')], is_directed=False)
```

---

## Running the Script

The script `network_structures.py` can be executed directly to see a demonstration of all features for both a sample undirected and a sample directed network.

```bash
python network_structures.py<img width="993" alt="Screenshot 2025-06-15 at 2 00 02 AM" src="https://github.com/user-attachments/assets/0e96144b-685b-4d4b-9963-b3647ccb735f" />
<img width="981" alt="Screenshot 2025-06-15 at 1 59 54 AM" src="https://github.com/user-attachments/assets/c5a0ebbb-790e-4ac8-ab67-daca0227dcf4" />

```
