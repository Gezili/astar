from math import sqrt
from matplotlib import collections
import numpy as np
import matplotlib.pyplot as plt

LEN_GRAPH = 35

class Node:
    
    def __init__ (self, ID, Coordinate):
        self.Coordinate = Coordinate
        self.ID = ID

#Edges are used for visualization only
class Edge:
    
    def __init__(self, StartNode, EndNode):
        self.StartNode = StartNode
        self.EndNode = EndNode

class Shape:
    
    def __init__ (self):
        self.Coordinates = []
        
    def add_point(self, Coordinate):
        self.Coordinates.append(Coordinate)
        
'''Determine if a node is within a polygon
A lot of simplifications can be made if the shape
is a rectangle or a right triangle with x, y axes aligned
with 2 of the sides'''


def inside_polygon(polygon, point):
    #Square

    if len(polygon) == 2:
        x_min, y_min = polygon[0][0], polygon[0][1]
        x_max, y_max = polygon[1][0], polygon[1][1]
        
        if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max:
            return True
        return False
    
    #Triangle   
    if len(polygon) == 3:
        
        epsilon = 0.00001
        
        x_1, y_1 = polygon[0][0], polygon[0][1]
        x_mid, y_mid = polygon[1][0], polygon[1][1]
        x_3, y_3 = polygon[2][0], polygon[2][1]
        
        m = (y_3 - y_1)/(x_3 - x_1)
        
        b = y_1 - m*x_1
        
        if min(y_1, y_3) - epsilon >= point[1] \
        or max(y_1, y_3) + epsilon <= point[1]\
        or min(x_1, x_3) - epsilon >= point[0]\
        or max(x_1, x_3) + epsilon <= point[0]:
            return False
        
        if min(y_1, y_3) == y_mid and y_mid < m*x_mid + b:
            if point[1] - epsilon <= m*point[0] + b:
                return True
            return False
        elif max(y_1, y_3) == y_mid and y_mid > m*x_mid + b:
            if point[1] + epsilon >= m*point[0] + b:
                return True
            return False

def inside_region(node):
    
    coordinate = node.Coordinate
    flag = False
    if 0 <= coordinate[0] <= LEN_GRAPH - 1 and 0 <= coordinate[1] <= LEN_GRAPH - 1:
        flag = True
    
    return flag

'''Generate a list of all nodes in the graph'''
def generate_all_nodes(sparse = False):
    
    nodes = []
    
    if sparse:
        for i in range(LEN_GRAPH*LEN_GRAPH):
            nodes.append(0)
            
    for i in range(LEN_GRAPH):
        for j in range(LEN_GRAPH):
            
            node = Node(i*LEN_GRAPH + j, (i, j))
            inside_polygon_flag = False
            
            for shape in shapes:
                if inside_polygon(shape.Coordinates, node.Coordinate):
                    inside_polygon_flag = True
                    
            if not inside_polygon_flag and inside_region(node) and not sparse:
                nodes.append(node)
            elif not inside_polygon_flag and inside_region(node):
                nodes[i*LEN_GRAPH + j] = node
                
    return nodes
            
'''Generate a list of all edges in the graph'''

def generate_all_edges(return_nodes = False):
    
    nodes = generate_all_nodes()
    node_ids = [node.ID for node in nodes]
    
    edges = []
    
    for node_orig in nodes:
        
        #Check 4 directions
        check_nodes = [
            node_orig.ID + 1, #north
            node_orig.ID + LEN_GRAPH, #east
            node_orig.ID + LEN_GRAPH + 1, #northeast
            node_orig.ID + LEN_GRAPH - 1, #southeast
        ]
        
        for i, node_dest in enumerate (check_nodes):
            if node_dest in node_ids:
                
                node_dest = nodes[node_ids.index(node_dest)]
                dist = calculate_euclidean_distance(
                    node_dest.Coordinate,
                    node_orig.Coordinate
                )
                
                #Hacky! Python has issues with rounding sometimes
                if dist < sqrt(2) + 0.01:
                    edge = Edge(node_orig, node_dest)
                    edges.append(edge)
            
    return edges
        
def visualize_nodes(nodes):
    
    node_x = [node.Coordinate[0] for node in nodes]
    node_y = [node.Coordinate[1] for node in nodes]
    plt.plot(node_x, node_y, 'ro')
    plt.show()
    
def visualize_edges(edges):
    
    line_segs = []

    for edge in edges:
        line_segs.append([edge.StartNode.Coordinate, edge.EndNode.Coordinate])

    line_segs = collections.LineCollection(line_segs)
    fig, ax = plt.subplots()
    ax.add_collection(line_segs)
    ax.autoscale()
    fig.show()
    
def calculate_euclidean_distance(c1, c2):
    return sqrt((c1[0] -  c2[0])**2 + (c1[1] - c2[1])**2)

#Please refactor in future if I ever decide to improve upon project - runs O(n^2)
'''Generate a dictionary of all node IDs followed by their neighboring nodes'''

def generate_graph_dict():
    
    graph_dict = {}
    nodes = generate_all_nodes()
    node_ids = [node.ID for node in nodes]
    
    for node_orig in nodes:
        
        #Check all 8 directions
        check_nodes = [
            node_orig.ID + 1, #north
            node_orig.ID + LEN_GRAPH, #east
            node_orig.ID + LEN_GRAPH + 1, #northeast
            node_orig.ID + LEN_GRAPH - 1, #southeast 
            node_orig.ID - 1, #south
            node_orig.ID - LEN_GRAPH, #west
            node_orig.ID - LEN_GRAPH + 1, #northwest
            node_orig.ID - LEN_GRAPH - 1 #southwest
        ]
        
        for i, node_dest in enumerate (check_nodes):
            if node_dest in node_ids:
                
                node_dest = nodes[node_ids.index(node_dest)]
                dist = calculate_euclidean_distance(
                    node_dest.Coordinate,
                    node_orig.Coordinate
                )
                
                #Hacky! Python has issues with rounding sometimes
                if dist < sqrt(2) + 0.01:
                    if node_orig.ID not in graph_dict:
                        graph_dict[node_orig.ID] = [node_dest.ID]
                    else:
                        graph_dict[node_orig.ID].append(node_dest.ID)
                        
    return graph_dict
    
def a_star():
    
    graph = generate_graph_dict()
    closed_set = []
    open_set = []
    
    coord_begin = (2, 2)
    coord_end = (32, 32)
    
    nodes = generate_all_nodes(sparse = True)

    #Compute heuristic cost function, initialize cost function at inf
    
    for node in nodes:
        #We don't need to set a massive distance - the total distance is less than 100
        #And setting it at infinity gives numerical issues
        if node is not 0:
            node.Cost = 100
            node.HeuristicCost = calculate_euclidean_distance(
                node.Coordinate, 
                coord_end #End_Node Coordinate
            )

    #This is our beginning node
    nodes[LEN_GRAPH*coord_begin[0] + coord_begin[1]].Cost =\
    nodes[LEN_GRAPH*coord_begin[0] + coord_begin[1]].HeuristicCost
    node_current = nodes[LEN_GRAPH*coord_begin[0] + coord_begin[1]]
    open_set.append(node_current.ID)
         
    while open_set:
        
        #Find the index of the current node from the open set
        node_current_id = open_set[np.argmin(np.array(\
        [nodes[open_set[i]].Cost + nodes[open_set[i]].HeuristicCost\
        for i in range(len(open_set))]))]
        
        if nodes[node_current_id].Coordinate == coord_end:
            
            path = []
            
            while True:
                try:
                    path.append(node_current_id)
                    node_current_id = nodes[node_current_id].Origin
                except AttributeError:
                    return path
        
        closed_set.append(node_current_id)
        del open_set[open_set.index(node_current_id)]
        
        node_list = graph[node_current_id]
        
        #Node has been visited - let's move on
        for node_id in node_list:
            if node_id in closed_set:
                continue
            
            #Update cost
            temp_cost = nodes[node_current_id].Cost +\
            calculate_euclidean_distance(nodes[node_current_id].Coordinate,\
            nodes[node_id].Coordinate)
            
            if node_id not in open_set:
                open_set.append(node_id)
                
            #There is a better path - ignore
            elif temp_cost >= nodes[node_id].Cost:
                continue
            
            #Update costs, origin
            nodes[node_id].Origin = node_current_id
            nodes[node_id].Cost = temp_cost
            nodes[node_id].HeuristicCost = temp_cost + nodes[node_id].HeuristicCost
    
    #Check to make sure openset is not null - if it is, then there is NO path      
    assert open_set
        
def print_path():
    
    edges = generate_all_edges()
    fig, ax = plt.subplots()
    
    line_segs = []
    colors = []
    linewidth = []
    
    for shape in shapes:
        if len(shape.Coordinates) == 2:
            coordinates = [
                (shape.Coordinates[0][0], shape.Coordinates[0][1]),
                (shape.Coordinates[0][0], shape.Coordinates[1][1]),
                (shape.Coordinates[1][0], shape.Coordinates[1][1]),
                (shape.Coordinates[1][0], shape.Coordinates[0][1])
            ]
        else:
            coordinates = shape.Coordinates
            
        ax.fill([coordinates[i][0] for i in range(len(coordinates))], 
                [coordinates[i][1] for i in range(len(coordinates))], 
                'g')
            
    for edge in edges:
        line_segs.append([edge.StartNode.Coordinate, edge.EndNode.Coordinate])
        colors.append('b')
        linewidth.append(1)

    nodes = generate_all_nodes(sparse = True)
    path = a_star()

    for i in range(1, len(path)):
        line_segs.append((nodes[path[i]].Coordinate, nodes[path[i - 1]].Coordinate))
        colors.append('r')
        linewidth.append(5)

    line_segs = collections.LineCollection(
        line_segs, 
        colors = colors, 
        linewidth = linewidth)
    ax.add_collection(line_segs)
    ax.autoscale()
    fig.show()
    
if __name__ == '__main__':
    #Declare all the shapes
    
    s1 = Shape()
    s1.add_point((6, 8))
    s1.add_point((12, 10))
    
    s2 = Shape()
    s2.add_point((9, 16))
    s2.add_point((12, 20))
    
    s3 = Shape()
    s3.add_point((14, 11))
    s3.add_point((17, 15))
    
    s4 = Shape()
    s4.add_point((28, 19))
    s4.add_point((28, 6))
    s4.add_point((20, 6))
    
    s5 = Shape()
    s5.add_point((18, 16))
    s5.add_point((24, 19))
    
    s6 = Shape()
    s6.add_point((10, 4))
    s6.add_point((12, 8))
    
    s7 = Shape()
    s7.add_point((10, 4))
    s7.add_point((10, 8))
    s7.add_point((6, 8))
    
    s8 = Shape()
    s8.add_point((12, 25))
    s8.add_point((28, 28))
    
    s9 = Shape()
    s9.add_point((25, 22))
    s9.add_point((28, 25))
    
    shapes = [s1, s2, s3, s4, s5, s6, s7, s8, s9]
    
    #Generate the graph
    
    print_path()
