
import gurobipy as gp
from tsp_data.utils import get_data_from_tsp_file,TSP_DATA
import itertools
import numpy as np
import gurobipy as gp

def gurobi_solver(filename=None,edge_weights:np.ndarray=None,node_coords=None):
    if edge_weights is None:
        if filename is not None:
            data = get_data_from_tsp_file(filename)
            edge_weights = data.edge_weights
        elif node_coords is not None:
            data = TSP_DATA(node_coords)
            edge_weights = data.edge_weights
        else:
            raise ValueError("No Input")
    if edge_weights.ndim == 2:
        tour = _gurobi_solver(edge_weights)
        tour = np.array(tour)
        return 
    else:
        assert edge_weights.ndim == 3
        tours = list()
        for i in range(edge_weights.shape[0]):
            tours.append(_gurobi_solver(edge_weights[i]))
        tours = np.array(tours)
        return tours
    
def _gurobi_solver(edge_weights:np.ndarray):
    n = edge_weights.shape[0]
    model = gp.Model()
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = model.addVar(vtype=gp.GRB.BINARY)
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if i != j) == 1)
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n) if i != j) == 1)
    obj = gp.quicksum(edge_weights[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)
    model.setObjective(obj)
    model.optimize()
    if model.status == gp.GRB.OPTIMAL:
        path = [[0] * n for _ in range(n)]
        for i, j in itertools.product(range(n), range(n)):
            if i != j:
                path[i][j] = int(x[i, j].x + 0.5)
        curr = 0
        visited = [False] * n
        visited[curr] = True
        tsp_order = [curr]
        for _ in range(n - 1):
            candidates = [(j, edge_weights[curr][j]) for j in range(n) if not visited[j]]
            candidates.sort(key=lambda x: x[1])
            curr = candidates[0][0]
            visited[curr] = True
            tsp_order.append(curr)
        return tsp_order
    else:
        return None