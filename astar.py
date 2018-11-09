from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import sqrt
from matplotlib import collections
import numpy as np

import matplotlib.pyplot as plt

class Node:
    
    def __init__ (self, ID, Coordinate):
        self.Coordinate = Coordinate
        self.ID = ID
        self.Neighbors = []
        
    def add_neighbor(self, Neighbor):
        self.Neighbors.append(Neighbor)
        
class Edge:
    
    def __init__(self, StartNode, EndNode, Cost):
        self.StartNode = StartNode
        self.EndNode = EndNode
        self.Cost = Cost

class Graph:
    
    def __init__(self, Edges):
        self.Edges = Edges
        
    def add_edge(self, Edge):
        self.Edges.append(Edge)

class Shape:
    
    def __init__ (self, Coordinates):
        self.Coordinates = Coordinates
        
    def add_point(self, Coordinate):
        self.Coordinates.append(Coordinate)
        
'''Determine if a node is within a polygon'''
        
def inside_polygon(polygon, point):
    
    polygon = Polygon(polygon)
    point = Point(point)
    return polygon.covers(point)
    
def inside_region(node):
    
    coordinate = node.Coordinate
    flag = False
    if 0 <= coordinate[0] <= 34 and 0 <= coordinate[1] <= 34:
        flag = True
    
    return flag

'''Generate a list of all nodes in the graph'''
def generate_all_nodes(sparse = False):
    
    shapes = [s1, s2, s3, s4, s5, s6]
    
    nodes = []
    
    if sparse:
        for i in range(35*35):
            nodes.append(0)
            
    for i in range(35):
        for j in range(35):
            
            node = Node(i*35 + j, (i, j))
            inside_polygon_flag = False
            
            for shape in shapes:
                if inside_polygon(shape.Coordinates, node.Coordinate):
                    inside_polygon_flag = True
                    
            if not inside_polygon_flag and inside_region(node) and not sparse:
                nodes.append(node)
            elif not inside_polygon_flag and inside_region(node):
                nodes[i*35 + j] = node
                
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
            node_orig.ID + 35, #east
            node_orig.ID + 36, #northeast
            node_orig.ID + 34, #southeast
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
                    edge = Edge(node_orig, node_dest, dist)
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
    
def construct_graph():
    
    edges = generate_all_edges()
    graph = Graph([])
    for edge in edges:
        graph.add_edge(edge)
        
    graph.ConnectivityGraph = generate_graph_dict()
        
    return graph

#Please refactor in future if I ever decide to improve upon project - runs O(n^2)
'''Generate a dictionary of all node IDs followed by their neighboring nodes'''

def generate_graph_dict():
    
    graph_dict = {}
    
    nodes = generate_all_nodes()
    edges = generate_all_edges()
    
    for node in nodes:
        for edge in edges:
            if edge.StartNode.ID == node.ID:
                if node.ID not in graph_dict:
                    graph_dict[node.ID] = [edge.EndNode.ID]
                else:
                    graph_dict[node.ID].append(edge.EndNode.ID)
                    
            if edge.EndNode.ID == node.ID:
                if node.ID not in graph_dict:
                    graph_dict[node.ID] = [edge.StartNode.ID]
                else:
                    graph_dict[node.ID].append(edge.StartNode.ID)
            
    return graph_dict
    
def a_star():
    
    graph = construct_graph()
    closed_set = []
    open_set = []
    
    nodes = generate_all_nodes(sparse = True)

    #Compute heuristic cost function, initialize cost function at inf
    
    for node in nodes:
        #We don't need to set a massive distance - the total distance is less than 100
        #And setting it at infinity gives numerical issues
        if node is not 0:
            node.Cost = 100
            node.HeuristicCost = calculate_euclidean_distance(
                node.Coordinate, 
                (32, 32) #End_Node Coordinate
            )

    #This is our beginning node
    nodes[35*2 + 2].Cost = nodes[35*2 + 2].HeuristicCost
    node_current = nodes[35*2 + 2]
    open_set.append(node_current.ID)
         
    while open_set != []:
        #Find the index of the current node from the open set
        node_current_id = open_set[np.argmin(np.array(\
        [nodes[open_set[i]].Cost + nodes[open_set[i]].HeuristicCost\
        for i in range(len(open_set))]))]
        
        if nodes[node_current_id].Coordinate == (32, 32):
            
            path = []
            
            while True:
                try:
                    path.append(node_current_id)
                    node_current_id = nodes[node_current_id].Origin
                except AttributeError:
                    return path
        
        closed_set.append(node_current_id)
        del open_set[open_set.index(node_current_id)]
        
        node_list = graph.ConnectivityGraph[node_current_id]
        
        for node_id in node_list:
            if node_id in closed_set:
                continue
            
            temp_cost = nodes[node_current_id].Cost +\
            calculate_euclidean_distance(nodes[node_current_id].Coordinate,\
            nodes[node_id].Coordinate)
            
            if node_id not in open_set:
                open_set.append(node_id)
            elif temp_cost >= nodes[node_id].Cost:
                continue
                
            nodes[node_id].Origin = node_current_id
            nodes[node_id].Cost = temp_cost
            nodes[node_id].HeuristicCost = temp_cost + nodes[node_id].HeuristicCost
    
    return nodes
        
def print_path():
    
    shapes = [s1, s2, s3, s4, s5, s6]
    edges = generate_all_edges()
    
    line_segs = []
    colors = []
    linewidth = []
    
    for shape in shapes:
        for i in range(len(shape.Coordinates)):
           
            line_segs.append([shape.Coordinates[i], shape.Coordinates[(i + 1) % len(shape.Coordinates)]])
            colors.append('g')
            linewidth.append(2)
            
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

    line_segs = collections.LineCollection(line_segs, colors = colors, linewidth = linewidth)
    fig, ax = plt.subplots()
    ax.add_collection(line_segs)
    ax.autoscale()
    fig.show()
    
    
if __name__ == '__main__':
    #Declare all the shapes
    
    s1 = Shape([])
    s1.add_point((6, 10))
    s1.add_point((12, 10))
    s1.add_point((12, 4))
    s1.add_point((10, 4))
    s1.add_point((6, 8))
    
    s2 = Shape([])
    s2.add_point((9, 16))
    s2.add_point((9, 20))
    s2.add_point((12, 20))
    s2.add_point((12, 16))
    
    s3 = Shape([])
    s3.add_point((17, 15))
    s3.add_point((17, 11))
    s3.add_point((14, 11))
    s3.add_point((14, 15))
    
    s4 = Shape([])
    s4.add_point((20, 6))
    s4.add_point((28, 19))
    s4.add_point((28, 6))
    
    s5 = Shape([])
    s5.add_point((18, 16))
    s5.add_point((18, 19))
    s5.add_point((24, 19))
    s5.add_point((24, 16))
    
    s6 = Shape([])
    s6.add_point((28, 22))
    s6.add_point((25, 22))
    s6.add_point((25, 25))
    s6.add_point((12, 25))
    s6.add_point((12, 28))
    s6.add_point((28, 28))
    
    #Generate the graph
    
    print_path()
