# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 23:35:48 2020

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:39:57 2020

@author: User
"""
import pandas as pd
import math
import random 
import copy
import numpy as np
from numpy import array_equal

# this class represents a node
class Node:
    
    def __init__(self, state, pre_node=None):
        self.state = state
        self.pre_node = pre_node
        self.g = 0 # distance to start node
        self.h = 0 # distance to goal node
        self.f = 0 # total cost
    
    # getters
    def get_state(self):
        return self.state
    
    def get_previous (self):
        return self.pre_node
        
    def get_f(self):
        return self.g + self.h
    
    def get_g(self):
        return self.g
    
    def get_h(self):
        return self.h
    
    # setters
    def set_state(self, state):
        self.state = state
        
    def set_previous (self, pre_node):
        self.pre_node = pre_node
        
    def set_h(self, h):
        self.h = h

    def set_g(self):
        self.g = self.get_previous().get_g() + 1
        
# -----------------------------------------------------------------------------------------------------------------------------------------------
# source_locations - the beginning of the search
# destination_locations - the locations the voters need to reach from the source_locations
# search_method - a number that has multiple possible values, represents a search method
# detail_output - if false: the output will be a full chain of locations.
                # if true: the output will contain also  a heuristic value for the first transformation.
def find_path (source_locations, destination_locations, search_method, detail_output):
   solutions = [] # list of solutions. each solution is a list of path
    
   for source_location, destination_location in zip(source_locations, destination_locations): 
       startNode = Node(source_location)
       endNode = Node(destination_location)
       solution = method_dictionary[search_method](adjacency, startNode, endNode, detail_output) # list of path
       if solution == None:
            print ("No path found")
            return None
       solutions.append(list(solution))

   return solutions
     
# A* - a function that restores the solution
def solution (startNode, endNode, method, detail_output = False): 
    current_node = endNode 
    heuristic_value = 0
    path = []
    
    while current_node.get_state() != startNode.get_state():
        path.append(current_node.get_state())
        heuristic_value = current_node.get_h()
        current_node = current_node.get_previous()
    
    if (detail_output): # when we reach to the second element in the path, we add it's heuristic value
       if method ==  1 :
           path[-1] += " (Heuristic value is " + str(heuristic_value) + ")"
       
        
    path.append(startNode.get_state()) # add the first element in the path
    return path[::-1] # Return a reversed path

def print_solutions(solutions):
    
    max_len = len(max(solutions)) # length of the longest path
    for i in range(max_len):
        line = "{"
        for solution in solutions:
            if i > len(solution)-1:
                line += solution[-1]
            else: line += solution[i]
            line += " ; "
        line = line[:-3]
        line += "}"
        print(line)     

def neighbor_list(file, node_state):
       
    neighbors = file.query('countyname == "' + node_state + '"')   # creating a tuple
    neighbors = list(neighbors['neighborname']) # creating a list of neighbors
    return neighbors
# -------------------------------------------a_star----------------------------------------------------------------
  
# A* - a function that calculates the heuristic for a node - BFS on countries
def heuristic (startNode, endNode): 
   # start and end countries
   startVertex = Node(country_string(startNode))
   endVertex = Node(country_string(endNode))
  
   solution = a_star(heuristic_file, startVertex, endVertex, h = True) # a call to A* algorithm, with another file of adjacency 
   if solution:
       return len(solution)-1
   return math.inf

# A* - a function that returns a substring of the country only, for example 'TX'       
def country_string (node):
    string = node.get_state()
    comma_index = string.find(',') 
    return(string[comma_index+2:])
    
# A* - a function that returns the f(n) value of a node
def sort_fun (node):
    return node.get_f()

def a_star (file, startNode, endNode, detail_output = False, h = False): 
    # initializing
    frontier_dic = {} # dictionary of nodes, key is state
    frontier = [] # a priority queue of nodes
    explored = set() # an empty set ,  nodes' states that were visited  
    
    frontier.append(startNode)
    frontier_dic[startNode.get_state()] = startNode # start node is the only element
    
    while frontier:
        
        frontier.sort(reverse=False, key=sort_fun) # ASC order by path cost, f(n)=g(n)+h(n)
        node = frontier.pop(0) # get the node with the lowest f(n) 
        
        # add the node to explored set
        explored.add(node.get_state())
        
        # check if goal was reached
        if node.get_state() == endNode.get_state():
            return solution(startNode, node, 1, detail_output)
        
        neighbors = neighbor_list(file, node.get_state())

        for neighbor_state in neighbors:
            neighbor = Node(neighbor_state, node) # creating a node from each neighbor
            neighbor.set_g() # set g(n)
            if not h: 
                neighbor.set_h(heuristic(neighbor, endNode)) # set h(n) only if the call wasn't from heuristic function
            
            # check if neighbor not in frontier, insert
            if neighbor_state not in frontier_dic and explored:
               frontier.append(neighbor)
               frontier_dic[neighbor_state] = neighbor
               
            # check if neighbor in frontier and it has a lower f(n) , replace
            elif neighbor_state in frontier_dic and neighbor.get_g() < frontier_dic[neighbor_state].get_g():
                frontier.remove(frontier_dic[neighbor_state])
                frontier.append(neighbor)
                frontier_dic[neighbor_state] = neighbor
                
    return None 

# -----------------------------------------hill_climbing---------------------------------------------------------

def hill_climbing (file, start_node, end_node, detail_output = False):
    for i in range (0, 5):
        solution = k_beam_search(file, start_node, end_node, detail_output, 1)
        if solution != None: return solution
    return solution
# ----------------------------------------genetic algorithm----------------------------------------------------------                
def create_starting_population(population_size, start, end):
    if start or end in islands: return None  
    population = []
    
    for i in range (0, population_size):
       population.append(create_new_route(start, end, population))
     
    return population

def create_new_route(start, end, population):
    # here we are going to create a new route
    # the new route can be in any number of steps
    # the structure of the route will be a vector of strings, each indicates a location
    # the route will be a valid route
    # each location will be chosen randomaly
    
    route = []     
    current = start
    route.append(current)
    
    while True:
        neighbors = neighbor_list(adjacency, current)
        neighbors.remove(current)
    
        if end in neighbors:
            route.append(end)
            if route not in population:
                return route
            else: 
                return create_new_route(start, end, population)
        
        random_index = random.randint(0, len(neighbors)-1)
        route.append(neighbors[random_index])
        current = neighbors[random_index]

def fitness(route): 
    # here we are going to rank each solution, i.e route. 
    # the code is pretty simple; each route is ranked by it's length. 
    return len(route)

def reproduce(route_a, route_b):
    # when the two routes crossover the resulting route produced is a valid route
    # the crossover point has to be at the same location.
    start = route_a[0]
    end = route_a[-1]
    
    common_elements = set(route_a) & set(route_b)
    
    # if the common elements are only start and end location, then no child can be produced
    if len(common_elements) == 2:
        return (route_a, route_b)
    
    else:
        common_elements.remove(start)
        common_elements.remove(end) # we will delete the start and end locations
    
        random_element = random.sample(common_elements, 1) # randomly select a cutting point
    
    cut_a = np.random.choice(np.where(np.isin(route_a, random_element))[0])
    cut_b = np.random.choice(np.where(np.isin(route_b, random_element))[0])

    new_a1 = copy.deepcopy(route_a[0:cut_a])
    new_a2 = copy.deepcopy(route_b[cut_b:])
    
    new_b1 = copy.deepcopy(route_b[0:cut_b])
    new_b2 = copy.deepcopy(route_a[cut_a:])
    
    new_a = np.append(new_a1, new_a2)
    new_b = np.append(new_b1, new_b2)
  
    return (new_a, new_b)
  
def mutate (route, probability, population):
    
    end = route[-1]
    if random.random() < probability:
           mutate_index = random.randint(0, len(route)-1)
           new_route = copy.deepcopy(route[0:mutate_index+1])
           current = new_route[-1]
           
           while True:
               neighbors = neighbor_list(adjacency, current)
               neighbors.remove(current)
               
               if end in neighbors:
                   new_route = np.append(new_route, end)
                   #if arreq_in_list(new_route, population):
                   return new_route
                   #else:
                   #    return mutate(route, 1, population)
                   
               random_index = random.randint(0, len(neighbors)-1)
               new_route = np.append(new_route, neighbors[random_index])#####
               current = neighbors[random_index]
               
    return route

def score_population(population):
    
    scores = []
    for i in range (0, len(population)):
        scores += [fitness(population[i])]
        
    return scores
  
def pick_mate(scores):
# here we are going to give probabilties to each solution' score by ranking-based method   
    array = np.array(scores)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    fitness = [len(ranks) - x for x in ranks]
        
    cum_scores = copy.deepcopy(fitness)
    
    for i in range(1,len(cum_scores)):
        cum_scores[i] = fitness[i] + cum_scores[i-1]
    
    probs = [x / cum_scores[-1] for x in cum_scores]
   # print(probs)    
    rand = random.random()
 #   print(rand)
    for i in range(0, len(probs)):
        if rand < probs[i]:
          #  print (i)
            return i

def print_population(population):
   print("[", end = " ")
   for i in range (0, len(population) - 1 ) :
       print (population[i], end = " ")
       print (",")
       
   print (population[-1], end = " ")
   print("]")  
   
def genetic_algorithm(adjacency, start_node, end_node, detail_output=False):
    
    # parameters
    start = start_node.get_state()
    end = end_node.get_state()
    population_size = 10
    number_of_iterations = 50
    number_of_couples = 4
    number_of_winners_to_keep = 2
    mutation_probability = 0.05
    
    # create the starting population
    population = create_starting_population(population_size, start, end)
    if population is None: return None
    
    if detail_output:
        print("starting population is: ")
        print_population(population)
    
    last_distance = 1000000000
    
    # for a large number of iterations do:
    for i in range(0, number_of_iterations):
        new_population = []

        # evaluate the fitness of the current population
        scores = score_population(population)
        
        best = population[np.argmin(scores)]
        distance = fitness(best)
 
        if detail_output and i > 0 :
            print("Iteration %i: population is: " % (i+1))
            print_population(population)
            
        if distance != last_distance:
            print('Iteration %i: Best so far is %i steps' % (i+1, distance))
            
        # allow members of the population to breed based on their relative score
        # i.e., if their score is lower they're more likely to breed
        for j in range(0, number_of_couples): 
            parent_1_index = pick_mate(scores)
            parent_2_index = pick_mate(scores)

            new_1, new_2 = reproduce(population[parent_1_index], population[parent_2_index])
            new_population = new_population + [new_1, new_2]
   
        # mutate
        for j in range(0, len(new_population)):
            new_population[j] = np.copy(mutate(new_population[j], mutation_probability, new_population))
    
        # keep members of previous generation
        new_population += [population[np.argmin(scores)]]
        for j in range(1, number_of_winners_to_keep):
            keeper = pick_mate(scores)            
            new_population += [population[keeper]]
    
        # add new random members
        while len(new_population) < population_size:
            new_population += [create_new_route(start, end, population)]

        #replace the old population with a real copy
        population = copy.deepcopy(new_population)
   
        last_distance = distance
    
    
    return(population[np.argmin(scores)])

# -----------------------------------simulated_annealing----------------------------------------
def simulated_annealing (file, start_node, end_node, detail_output = False):
    # initialize
    initial_temp = 50
    final_temp = 0.1
    alpha = 0.1
    explored = set()
    
    start_node.set_h(heuristic(start_node, end_node))
    current_temp = initial_temp
    current = start_node
    explored.add(current)
    
    while current_temp > final_temp:
        neighbors = neighbor_list(file, current.get_state())
        neighbors.remove(current.get_state())
        
        # check for goal node before randomlt peaking 
        if end_node.get_state() in neighbors:
            neighbor = Node(end_node.get_state(), current)
            return solution (start_node, neighbor, 3, detail_output)
        
        # a randomly selected successor of current
        possible_choice = list(set(neighbors).difference(explored)) # dont choose the same one twice
        neighbor_state = random.choice(possible_choice)
        neighbor = Node(neighbor_state, current)
        neighbor.set_h(heuristic(neighbor, end_node))
        
        # check if neighbor is best so far
        delta = current.get_h() - neighbor.get_h()
        
        # if the new neighbor is better, accept it
        if delta > 0 :
            current = neighbor
            
        # else accept it with probability of e^(-delta/current temp)
        else :
            if random.uniform(0, 1) < math.exp(delta/current_temp):
               current = neighbor
        
        explored.add(current)
        
        # decreament the temperature
        current_temp -= alpha
    
    if (current.get_state() == end_node.get_state()): 
        return solution(start_node, current, 3, detail_output)
    return None

# -----------------------------------------k_beam_search---------------------------------------------------------

def k_beam_search (file, start_node, end_node, detail_output = False, k=3):    

    # initializing 
    stage = 1
    k_successors = []  # k best successor
    k_successors.append(start_node)
    all_successors = []
  
    
    while stage < 500 :
        # generate all successors of k states
        for node in k_successors:
              neighbors = neighbor_list(file, node.get_state())
              neighbors.remove(node.get_state()) # each node is a neighbor of itself
              
              for neighbor_state in neighbors:
                  neighbor = Node(neighbor_state, node)
                  neighbor.set_h(heuristic(neighbor, end_node))
                  all_successors.append(neighbor)
                  if neighbor_state == end_node.get_state():
                      return solution(start_node, neighbor, 4, detail_output) 
        
        # print bag of actions considered every stage
        if detail_output and k != 1:
            print ("bag of actions for stage %i :" %(stage+1))
            print("[", end = " ")
            for node in all_successors: 
                print(node.state, end = " ")
            print("]")
            print()
            
        # select the k best successors
        all_successors.sort(reverse=False, key=lambda node: node.get_h()) # ASC order by heuristic
        k_successors.clear()
        
        for i in range (0, k):
            k_successors.append(all_successors.pop(0))
        
        stage += 1
        
    return None
#------------------------------------------------------------------------------------------------------------
# a dictionary with all search methods
method_dictionary = { 
     1 : a_star,  
     2 : hill_climbing,
     3 : simulated_annealing,
     4 : k_beam_search,
     5 : genetic_algorithm }

# reading from files
adjacency = pd.read_csv("adjacency.csv") 
heuristic_file = pd.read_csv("heuristic.csv")
islands = pd.DataFrame(np.unique(heuristic_file['countyname'], return_counts = True)).transpose()[pd.DataFrame(np.unique(heuristic_file['countyname'], return_counts = True)).transpose().iloc[:,1] == 1][0]

def main ():
    solutions = find_path(['Washington County, UT'], ['St. John Island, VI'],5, True)
    if solutions != None:
        print("Solutions: ")
        print_solutions(solutions)
     
# run the main function   
if __name__ == "__main__": main()