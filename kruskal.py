import numpy as np

incidence_matrix = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 1]
])

edge_costs = np.array([2, 8, 2, 4, 1, 4, 8, 2, 2])


def check_incidence_matrix(incidence_matrix: np.array):
    """
    This function checks if the incidence matrix is wired correctly.
    """
    return all(np.sum(incidence_matrix, axis=0) == 2)


def kruskal_mst(incidence_matrix: np.array, edge_costs: np.array):
    """
    This function runs Kruskal's algorithm to find the minimum spanning tree at optimal cost.
    """
    total_cost = 0
    connected_vertices = []
    used_edges = []
    circle_check = np.array(range(incidence_matrix.shape[0]))

    if not check_incidence_matrix(incidence_matrix=incidence_matrix):
        print("Given incidence matrix is not correctly specified (Column sum not 2 for all edges.")
        return None

    for edge in np.argsort(edge_costs):
        # Get the connected vertices from an edge
        added_vertices = np.where(incidence_matrix[:, edge] == 1)
        added_vertices = [np.asscalar(vertice) for vertice in np.nditer(added_vertices)]

        if max(np.bincount(circle_check)) == incidence_matrix.shape[0]:
            print(f"All vertices connected, no more edge connections needed. Total cost of tree: {total_cost}")
            return used_edges, total_cost

        if not np.equal(circle_check[added_vertices[0]], circle_check[added_vertices[1]]):
            circle_check[added_vertices[1]] = circle_check[added_vertices[0]]
            circle_check[np.where(circle_check == added_vertices[1])] = circle_check[added_vertices[0]]

            used_edges.append(edge)
            total_cost += edge_costs[edge]

        else:
            print(f"Edge e{edge + 1} would lead to a circle in the minimum spanning tree.")

    return used_edges, total_cost


x, y = kruskal_mst(incidence_matrix=incidence_matrix, edge_costs=edge_costs)