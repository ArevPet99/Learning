import pprint
import heapq

class Network:
    """
    A class to represent a network graph, which can be either directed or
    undirected. It stores nodes and a list of weighted edges.
    """
    def __init__(self, is_directed=False):
        """
        Initializes the Network object.

        Args:
            is_directed (bool): If True, the network is directed. 
                                Otherwise, it's undirected.
        """
        self._nodes = set()
        # Edges are now stored as a list of tuples: (u, v, weight)
        self._edges = []
        self.is_directed = is_directed
        self._node_map = {} # Maps node names to integer indices
        # Adjacency list representation for efficient lookups: {node: {neighbor: weight}}
        self.adj = {}

    def add_node(self, node):
        """Adds a single node to the network."""
        if node not in self._nodes:
            self._nodes.add(node)
            self.adj[node] = {}
            self._update_node_map()
            
    def add_edge(self, u, v, weight=1):
        """
        Adds a weighted edge between node u and node v.
        Nodes are automatically added if they don't exist.

        Args:
            u: The starting node of the edge.
            v: The ending node of the edge.
            weight (int or float): The weight of the edge (default is 1).
        """
        # Add nodes if they are not already in the network
        if u not in self._nodes:
            self.add_node(u)
        if v not in self._nodes:
            self.add_node(v)
        
        # Add edge to adjacency list for network-based algorithms
        self.adj[u][v] = weight
        if not self.is_directed:
            self.adj[v][u] = weight

        # Add to the canonical edge list for matrix representations
        edge = (u, v, weight)
        if not self.is_directed:
            # Sort by node name to create a canonical representation for undirected edges
            canonical_edge = (min(u,v), max(u,v), weight)
            # Avoid adding duplicate edges for undirected graphs
            if not any(e[0] == canonical_edge[0] and e[1] == canonical_edge[1] for e in self._edges):
                 self._edges.append(canonical_edge)
        else:
            if edge not in self._edges:
                self._edges.append(edge)
    
    def _update_node_map(self):
        """Updates the internal mapping of node names to matrix indices."""
        sorted_nodes = sorted(list(self._nodes))
        self._node_map = {node: i for i, node in enumerate(sorted_nodes)}

    def get_nodes(self):
        """Returns a sorted list of nodes."""
        return sorted(list(self._nodes))

    def get_edges(self):
        """Returns the list of edges."""
        return sorted(self._edges)

    def to_adjacency_matrix(self):
        """
        Generates the weighted vertex-to-vertex adjacency matrix.
        Non-existent edges are represented by infinity.

        Returns:
            A tuple containing:
            - A 2D list representing the weighted adjacency matrix.
            - A list of nodes corresponding to the matrix rows/columns.
        """
        nodes = self.get_nodes()
        num_nodes = len(nodes)
        # Initialize matrix with infinity, 0 on the diagonal
        adj_matrix = [[float('inf')] * num_nodes for _ in range(num_nodes)]
        for i in range(num_nodes):
            adj_matrix[i][i] = 0

        for u, v, weight in self._edges:
            u_idx = self._node_map[u]
            v_idx = self._node_map[v]
            adj_matrix[u_idx][v_idx] = weight
            if not self.is_directed:
                adj_matrix[v_idx][u_idx] = weight
        
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

        inc_matrix = [[0] * num_edges for _ in range(num_nodes)]

        for j, edge in enumerate(edges):
            u, v, _ = edge # Weight is ignored for incidence matrix
            u_idx = self._node_map[u]
            v_idx = self._node_map[v]

            if self.is_directed:
                inc_matrix[u_idx][j] = 1
                inc_matrix[v_idx][j] = -1
            else:
                inc_matrix[u_idx][j] = 1
                inc_matrix[v_idx][j] = 1
        
        return inc_matrix, nodes, edges

    def __repr__(self):
        """String representation of the Network object."""
        return (f"Network(nodes={self.get_nodes()}, "
                f"edges={self.get_edges()}, is_directed={self.is_directed})")

# --- Shortest Path Problem Implementations ---

def shortest_path_network(network, start_node, end_node):
    """
    Finds the shortest path using Dijkstra's algorithm on the Network object.
    
    Args:
        network (Network): The network graph.
        start_node: The starting node.
        end_node: The target node.
        
    Returns:
        A tuple of (path, distance) or (None, float('inf')) if no path exists.
    """
    distances = {node: float('inf') for node in network.get_nodes()}
    distances[start_node] = 0
    previous_nodes = {node: None for node in network.get_nodes()}
    
    # Priority queue stores tuples of (distance, node)
    pq = [(0, start_node)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        # If we found a shorter path already, skip
        if current_distance > distances[current_node]:
            continue
            
        # If we reached the end, reconstruct the path
        if current_node == end_node:
            path = []
            while current_node is not None:
                path.insert(0, current_node)
                current_node = previous_nodes[current_node]
            return path, distances[end_node]
            
        # Check neighbors
        for neighbor, weight in network.adj[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
                
    return None, float('inf')


def shortest_path_tabular(adj_matrix, nodes, start_node, end_node):
    """
    Finds the shortest path using Dijkstra's algorithm on an adjacency matrix.
    
    Args:
        adj_matrix (list of lists): The weighted adjacency matrix.
        nodes (list): The list of node labels.
        start_node: The starting node.
        end_node: The target node.
        
    Returns:
        A tuple of (path, distance) or (None, float('inf')) if no path exists.
    """
    num_nodes = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}
    start_idx = node_map[start_node]
    end_idx = node_map[end_node]
    
    distances = [float('inf')] * num_nodes
    distances[start_idx] = 0
    previous_nodes_idx = [None] * num_nodes
    
    pq = [(0, start_idx)]
    
    while pq:
        current_distance, current_idx = heapq.heappop(pq)
        
        if current_distance > distances[current_idx]:
            continue
            
        if current_idx == end_idx:
            path = []
            node_idx = current_idx
            while node_idx is not None:
                path.insert(0, nodes[node_idx])
                node_idx = previous_nodes_idx[node_idx]
            return path, distances[end_idx]
            
        # Check all other nodes as potential neighbors
        for neighbor_idx in range(num_nodes):
            weight = adj_matrix[current_idx][neighbor_idx]
            if weight != float('inf'): # If there is an edge
                distance = current_distance + weight
                if distance < distances[neighbor_idx]:
                    distances[neighbor_idx] = distance
                    previous_nodes_idx[neighbor_idx] = current_idx
                    heapq.heappush(pq, (distance, neighbor_idx))

    return None, float('inf')


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)

    # --- Shortest Path Example (Undirected Weighted Graph) ---
    print("=" * 60)
    print("SHORTEST PATH PROBLEM EXAMPLE")
    print("=" * 60)
    
    # 1. Define the network with weighted edges
    weighted_net = Network(is_directed=False)
    weighted_net.add_edge('A', 'B', 7)
    weighted_net.add_edge('A', 'C', 9)
    weighted_net.add_edge('A', 'F', 14)
    weighted_net.add_edge('B', 'C', 10)
    weighted_net.add_edge('B', 'D', 15)
    weighted_net.add_edge('C', 'D', 11)
    weighted_net.add_edge('C', 'F', 2)
    weighted_net.add_edge('D', 'E', 6)
    weighted_net.add_edge('E', 'F', 9)

    print("Original Weighted Network:")
    print(weighted_net, "\n")

    start = 'A'
    end = 'E'
    print(f"Finding shortest path from '{start}' to '{end}'...\n")

    # 2. Solve using the network-based implementation
    print("--- 2.1 Solving with Network Object ---")
    path_net, dist_net = shortest_path_network(weighted_net, start, end)
    print(f"Path: {path_net}")
    print(f"Total Distance: {dist_net}\n")
    
    # 3. Get the tabular (matrix) representation
    print("--- 2.2 Solving with Tabular (Adjacency Matrix) ---")
    adj_matrix, nodes = weighted_net.to_adjacency_matrix()
    print("Generated Weighted Adjacency Matrix:")
    print("Nodes:", nodes)
    pp.pprint(adj_matrix)
    print("")

    # 4. Solve using the tabular implementation
    path_tab, dist_tab = shortest_path_tabular(adj_matrix, nodes, start, end)
    print(f"Path: {path_tab}")
    print(f"Total Distance: {dist_tab}\n")

    # 5. Verification
    print("--- 2.3 Verification ---")
    if path_net == path_tab and dist_net == dist_tab:
        print("✅ Success: Both methods produced the same result.")
    else:
        print("❌ Error: Methods produced different results.")

