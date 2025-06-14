import pprint

class Network:
    """
    A class to represent a network graph, which can be either directed or
    undirected. It stores nodes and a list of edges.
    """
    def __init__(self, is_directed=False):
        """
        Initializes the Network object.

        Args:
            is_directed (bool): If True, the network is directed. 
                                Otherwise, it's undirected.
        """
        self._nodes = set()
        self._edges = []
        self.is_directed = is_directed
        self._node_map = {} # Maps node names to integer indices

    def add_node(self, node):
        """Adds a single node to the network."""
        if node not in self._nodes:
            self._nodes.add(node)
            self._update_node_map()
            
    def add_edge(self, u, v):
        """
        Adds an edge between node u and node v.
        Nodes are automatically added if they don't exist.

        Args:
            u: The starting node of the edge.
            v: The ending node of the edge.
        """
        # Add nodes if they are not already in the network
        if u not in self._nodes:
            self.add_node(u)
        if v not in self._nodes:
            self.add_node(v)
        
        edge = (u, v)
        # For undirected graphs, store edges in a canonical order (e.g., sorted)
        # to avoid duplicates like (v, u) if (u, v) is already there.
        if not self.is_directed:
            canonical_edge = tuple(sorted(edge))
            if canonical_edge not in self._edges:
                self._edges.append(canonical_edge)
        else:
            # For directed graphs, the order matters.
            if edge not in self._edges:
                self._edges.append(edge)
    
    def _update_node_map(self):
        """Updates the internal mapping of node names to matrix indices."""
        # Sort nodes to ensure consistent matrix ordering
        sorted_nodes = sorted(list(self._nodes))
        self._node_map = {node: i for i, node in enumerate(sorted_nodes)}

    def get_nodes(self):
        """Returns a sorted list of nodes."""
        return sorted(list(self._nodes))

    def get_edges(self):
        """Returns the list of edges."""
        return self._edges

    def to_adjacency_matrix(self):
        """
        Generates the vertex-to-vertex adjacency matrix for the network.

        Returns:
            A tuple containing:
            - A 2D list representing the adjacency matrix.
            - A list of nodes corresponding to the matrix rows/columns.
        """
        nodes = self.get_nodes()
        num_nodes = len(nodes)
        # Create a matrix of zeros
        adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]

        for u, v in self._edges:
            u_idx = self._node_map[u]
            v_idx = self._node_map[v]
            adj_matrix[u_idx][v_idx] = 1
            # If the graph is undirected, the matrix is symmetric
            if not self.is_directed:
                adj_matrix[v_idx][u_idx] = 1
        
        return adj_matrix, nodes

    def to_incidence_matrix(self):
        """
        Generates the vertex-to-edge incidence matrix for the network.

        Returns:
            A tuple containing:
            - A 2D list representing the incidence matrix.
            - A list of nodes (rows).
            - A list of edges (columns).
        """
        nodes = self.get_nodes()
        edges = self.get_edges()
        num_nodes = len(nodes)
        num_edges = len(edges)

        # Create a matrix of zeros with dimensions |V| x |E|
        inc_matrix = [[0] * num_edges for _ in range(num_nodes)]

        for j, edge in enumerate(edges):
            u, v = edge
            u_idx = self._node_map[u]
            v_idx = self._node_map[v]

            if self.is_directed:
                # For directed graphs: +1 for the source, -1 for the sink
                inc_matrix[u_idx][j] = 1
                inc_matrix[v_idx][j] = -1
            else:
                # For undirected graphs: 1 for both vertices
                inc_matrix[u_idx][j] = 1
                inc_matrix[v_idx][j] = 1
        
        return inc_matrix, nodes, edges

    def __repr__(self):
        """String representation of the Network object."""
        return (f"Network(nodes={self.get_nodes()}, "
                f"edges={self._edges}, is_directed={self.is_directed})")

# --- Functions to create networks from matrices ---

def from_adjacency_matrix(matrix, nodes, is_directed=False):
    """
    Creates a Network object from an adjacency matrix.

    Args:
        matrix (list of lists): The adjacency matrix.
        nodes (list): The list of node labels corresponding to matrix indices.
        is_directed (bool): Specifies if the resulting network should be directed.

    Returns:
        Network: The reconstructed network object.
    """
    net = Network(is_directed=is_directed)
    num_nodes = len(nodes)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if matrix[i][j] == 1:
                net.add_edge(nodes[i], nodes[j])
    return net

def from_incidence_matrix(matrix, nodes, is_directed=False):
    """
    Creates a Network object from an incidence matrix.

    Args:
        matrix (list of lists): The incidence matrix.
        nodes (list): The list of node labels corresponding to matrix rows.
        is_directed (bool): Specifies if the resulting network should be directed.

    Returns:
        Network: The reconstructed network object.
    """
    net = Network(is_directed=is_directed)
    if not matrix or not matrix[0]:
        # Handle empty matrix
        for node in nodes:
            net.add_node(node)
        return net
        
    num_nodes = len(matrix)
    num_edges = len(matrix[0])

    for j in range(num_edges):  # Iterate through columns (edges)
        if is_directed:
            try:
                # Find the source (+1) and sink (-1) for the directed edge
                source_idx = [row[j] for row in matrix].index(1)
                sink_idx = [row[j] for row in matrix].index(-1)
                net.add_edge(nodes[source_idx], nodes[sink_idx])
            except ValueError:
                # This could happen with an invalid matrix format
                print(f"Warning: Column {j} in directed incidence matrix is invalid.")
        else:
            # Find the two vertices (1s) for the undirected edge
            incident_nodes_indices = [i for i in range(num_nodes) if matrix[i][j] == 1]
            if len(incident_nodes_indices) == 2:
                u_idx, v_idx = incident_nodes_indices
                net.add_edge(nodes[u_idx], nodes[v_idx])
            else:
                 # This could happen for self-loops or invalid matrices
                print(f"Warning: Column {j} in undirected incidence matrix is invalid.")

    return net


if __name__ == '__main__':
    # Use pprint for cleaner dictionary/list printing
    pp = pprint.PrettyPrinter(indent=4)

    # --- 1. Undirected Network Example ---
    print("=" * 50)
    print("UNDIRECTED NETWORK EXAMPLE")
    print("=" * 50)
    
    # Create an undirected network
    undirected_net = Network(is_directed=False)
    undirected_net.add_edge('A', 'B')
    undirected_net.add_edge('A', 'C')
    undirected_net.add_edge('B', 'C')
    undirected_net.add_edge('C', 'D')

    print("Original Undirected Network:")
    print(undirected_net, "\n")

    # Get and print the adjacency matrix
    adj_matrix, adj_nodes = undirected_net.to_adjacency_matrix()
    print("Vertex-to-Vertex Adjacency Matrix:")
    print("Nodes:", adj_nodes)
    pp.pprint(adj_matrix)
    print("-" * 20)

    # Get and print the incidence matrix
    inc_matrix, inc_nodes, inc_edges = undirected_net.to_incidence_matrix()
    print("Vertex-to-Edge Incidence Matrix:")
    print("Nodes (Rows):", inc_nodes)
    print("Edges (Cols):", inc_edges)
    pp.pprint(inc_matrix)
    print("-" * 20)

    # Reconstruct from matrices
    reconstructed_from_adj = from_adjacency_matrix(adj_matrix, adj_nodes, is_directed=False)
    print("Reconstructed from Adjacency Matrix:")
    print(reconstructed_from_adj, "\n")

    reconstructed_from_inc = from_incidence_matrix(inc_matrix, inc_nodes, is_directed=False)
    print("Reconstructed from Incidence Matrix:")
    print(reconstructed_from_inc, "\n")


    # --- 2. Directed Network Example ---
    print("=" * 50)
    print("DIRECTED NETWORK EXAMPLE")
    print("=" * 50)

    # Create a directed network
    directed_net = Network(is_directed=True)
    directed_net.add_edge('1', '2')
    directed_net.add_edge('1', '3')
    directed_net.add_edge('2', '3')
    directed_net.add_edge('3', '1') # Cycle

    print("Original Directed Network:")
    print(directed_net, "\n")

    # Get and print the adjacency matrix
    adj_matrix_dir, adj_nodes_dir = directed_net.to_adjacency_matrix()
    print("Vertex-to-Vertex Adjacency Matrix (Directed):")
    print("Nodes:", adj_nodes_dir)
    pp.pprint(adj_matrix_dir)
    print("-" * 20)

    # Get and print the incidence matrix
    inc_matrix_dir, inc_nodes_dir, inc_edges_dir = directed_net.to_incidence_matrix()
    print("Vertex-to-Edge Incidence Matrix (Directed):")
    print("Nodes (Rows):", inc_nodes_dir)
    print("Edges (Cols):", inc_edges_dir)
    pp.pprint(inc_matrix_dir)
    print("-" * 20)
    
    # Reconstruct from matrices
    reconstructed_from_adj_dir = from_adjacency_matrix(adj_matrix_dir, adj_nodes_dir, is_directed=True)
    print("Reconstructed from Adjacency Matrix:")
    print(reconstructed_from_adj_dir, "\n")

    reconstructed_from_inc_dir = from_incidence_matrix(inc_matrix_dir, inc_nodes_dir, is_directed=True)
    print("Reconstructed from Incidence Matrix:")
    print(reconstructed_from_inc_dir, "\n")

