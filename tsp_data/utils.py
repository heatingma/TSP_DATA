import numpy as np
import argparse
from texttable import Texttable
from scipy.spatial.distance import cdist
import os
import tsplib95

############################################
##               TSP_DATA                 ##
############################################

class TSP_DATA:
    """
    Represents a batch of TSP problems, each with a fixed number of nodes and random node node_coords.

    Attributes:
        node_coords (ndarray): A 3D numpy array of shape (batch, nodes_num, 2), 
            representing the node node_coords of eachTSP problem in the batch.
        edge_weights (ndarray): A 3D numpy array of shape (batch, nodes_num, nodes_num), 
            representing the distance matrix of each TSP problem in the batch.
        tour (ndarray): A 2D numpy array of shape (batch, nodes_num), representing the tour of
            each TSP problem in the batch, if available.
            
    """
    def __init__(self,node_coords:np.ndarray=None,edge_weights:np.ndarray=None):
        if node_coords is None:
            self.node_coords = None
            if edge_weights is None:
                raise ValueError("node_coords and edge_weights cannot be both None")
            self.edge_weights = edge_weights
        else:
            if node_coords.ndim == 2:
                node_coords = np.expand_dims(node_coords,axis=0)
            assert node_coords.ndim == 3
            self.node_coords = node_coords
            self.edge_weights = np.array([cdist(coords,coords) for coords in self.node_coords])
        self.tour = None
    
    def __repr__(self):
        if self.node_coords is None:
            self.message = "edge_weights = {}".format(self.edge_weights.shape)
        else:
            self.message = "node_coords = {}, edge_weights = {}".format(self.node_coords.shape,self.edge_weights.shape)
        return f"{self.__class__.__name__}({self.message})"


############################################
##          generate node_coords          ##
############################################

def generate_tsp_data(batch=10,nodes_num=10,low=0,high=1,random_mode="uniform"):
    if random_mode == "uniform":
        node_coords = np.random.uniform(low,high,size=(batch,nodes_num,2))
    elif random_mode == "gaussian":
        node_coords = np.random.normal(loc=0,scale=1,size=(batch,nodes_num,2))
        max_value = np.max(node_coords) 
        min_value = np.min(node_coords)
        node_coords = np.interp(node_coords,(min_value,max_value),(low,high))
    else:
        raise ValueError(f"Unknown random_mode: {random_mode}")
    return node_coords        


#############################################
##        generate .tsp/opt.tour files     ##
#############################################

def generate_tsp_file(node_coords:np.ndarray, filename):
    """
    Generate a TSP problem data file based on the given point node_coords.

    Args:
        node_coords: A two-dimensional list containing the point node_coords, e.g. [[x1, y1], [x2, y2], ...]
        filename: The filename of the generated TSP problem data file

    """
    if node_coords.ndim == 3:
        shape = node_coords.shape
        if shape[0] == 1:
            node_coords = node_coords.squeeze(axis=0)
            _generate_tsp_file(node_coords,filename)
        else:
            for i in range(shape[0]):
                _filename = filename + '-' + str(i) 
                _generate_tsp_file(node_coords[i],_filename)
    else:
        assert node_coords.ndim == 2
        _generate_tsp_file(node_coords,filename)
             
def _generate_tsp_file(node_coords:np.ndarray, filename):
    num_points = node_coords.shape[0]
    file_basename = os.path.basename(filename)
    with open(filename, 'w') as f:
        f.write(f"NAME: {file_basename}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {num_points}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(num_points):
            x, y = node_coords[i]
            f.write(f"{i+1} {x} {y}\n")
        f.write("EOF\n")


def generate_opt_tour_file(tour:np.ndarray, filename):
    """
    Generate an opt.tour file based on the given tour.

    Args:
        tour: A one-dimensional numpy array containing the tour, e.g. [1, 5, 3, 2, 4]
        filename: The filename of the generated opt.tour file

    """
    if tour.ndim == 2:
        shape = tour.shape
        if shape[0] == 1:
            tour = tour.squeeze(axis=0)
            _generate_opt_tour_file(tour,filename)
        else:
            for i in range(shape[0]):
                _filename = filename + '-' + str(i) 
                _generate_opt_tour_file(tour[i],_filename)
    else:
        assert tour.ndim == 1
        _generate_opt_tour_file(tour,filename)
   
def _generate_opt_tour_file(tour:np.ndarray, filename):
    """
    Generate an opt.tour file based on the given tour.

    Args:
        tour: A one-dimensional numpy array containing the tour, e.g. [1, 5, 3, 2, 4]
        filename: The filename of the generated opt.tour file

    """
    num_points = len(tour)
    file_basename = os.path.basename(filename)
    with open(filename, 'w') as f:
        f.write(f"NAME: {file_basename}\n")
        f.write(f"TYPE: TOUR\n")
        f.write(f"DIMENSION: {num_points}\n")
        f.write(f"TOUR_SECTION\n")
        for i in range(num_points):
            f.write(f"{tour[i]}\n")
        f.write(f"-1\n")
        f.write(f"EOF\n")


#############################################
##         get data/tour from files        ##
#############################################

def get_data_from_tsp_file(filename):
    tsp_data = tsplib95.load(filename)
    if tsp_data.node_coords == {}:
        num_nodes = tsp_data.dimension
        edge_weights = tsp_data.edge_weights
        edge_weights = [elem for sublst in edge_weights for elem in sublst]
        new_edge_weights = np.zeros(shape=(num_nodes,num_nodes))
        if (num_nodes * (num_nodes-1) / 2) == len(edge_weights):
            """
            [[0,1,1,1,1],
             [0,0,1,1,1],
             [0,0,0,1,1],
             [0,0,0,0,1],
             [0,0,0,0,0]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i >= j:
                        continue
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1
            new_edge_weights = new_edge_weights.T + new_edge_weights
        elif (num_nodes * (num_nodes+1) / 2) == len(edge_weights):
            """
            [[x,1,1,1,1],
             [0,x,1,1,1],
             [0,0,x,1,1],
             [0,0,0,x,1],
             [0,0,0,0,x]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i > j:
                        continue
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1
            new_edge_weights = new_edge_weights.T + new_edge_weights
        elif ((num_nodes-1) * (num_nodes-1)) == len(edge_weights):
            """
            [[0,1,1,1,1],
             [1,0,1,1,1],
             [1,0,0,1,1],
             [1,1,1,0,1],
             [1,1,1,1,0]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i == j:
                        continue
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1
        elif (num_nodes * num_nodes) == len(edge_weights):
            """
            [[x,1,1,1,1],
             [1,x,1,1,1],
             [1,0,x,1,1],
             [1,1,1,x,1],
             [1,1,1,1,x]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1  
        else:
            raise ValueError("edge_weights cannot form a Symmetric matrix") 
        data = TSP_DATA(edge_weights=new_edge_weights)
    else:
        node_coords = np.array(list(tsp_data.node_coords.values()))
        data = TSP_DATA(node_coords)
    return data

def get_tour_from_tour_file(filename):
    tsp_tour = tsplib95.load(filename)
    tsp_tour = np.array(tsp_tour.tours).squeeze(axis=0)
    return tsp_tour


#############################################
##                parameter                ##
#############################################

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
    print(t.draw())
      
def parameter_parser():
    """
    Parses command line arguments and returns a Namespace object.

    Returns:
        argparse.Namespace: 
            A Namespace object containing the values of the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nodes_num", type=int, default=10)
    parser.add_argument("--low", type=int, default=0)
    parser.add_argument("--high", type=int, default=100)
    parser.add_argument("--random_mode", type=str, default="uniform")
    parser.add_argument("--filename", type=str, default=None)
    
    args = parser.parse_args()
    if args.filename is None:
        args.filename = f"tsp{args.nodes_num}_batch{args.batch_size}_{args.random_mode}"
    args.tsp_path = "tsp_data/generation/" + args.filename + '.tsp'       
    return args

