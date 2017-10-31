# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

#function that returns 1 if the list contains the item
def not_in (list,item):
    if(item in list ):
        return 1
    else:
        return 0

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """
        State: Coordinate of Node
        Parent: Node in Search Tree that generated this Node
        Action: Action applied to parent to generate this node
        Path-Cost: Cost from initial state to this node
    """
    def __init__(self, successor, parent=None):
        #successor(geitonas) =   ((34, 15), 'South', 1)
        #print("Node :: successor = "+str(successor))
        self.state = successor[0] #thesh 
        self.parent = parent #poios einai o pateras 
        self.action = successor[1]#pws phge apo ton patera ekei 
        #an o parent yparxei prosthetw to kostos
        if parent == None or parent.pathCost == None:
            self.pathCost = successor[2]
        else:
            self.pathCost = parent.pathCost + successor[2]
    def expand(self,problem):
        return [self.Node(successor, self)
                for successor in problem.getSuccessors(self.state)]
    def child_node(self, problem, action):
        "[Figure 3.10]"
        next = action[0]
        return Node(next, self, action[1], self.path_cost+action[2])


    def getPath(self):
        path = list()
        currentNode = self
        while currentNode.action != None:
            path.insert(0, currentNode.action)
            currentNode = currentNode.parent
        return path
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)




def construct_path(state, meta):
    action_list = []
    while True:
        if(state == None):
            break
        print(state)
        row = meta[state]
        if len(row) == 2:
            state = row[0]
            action = row[1]
            if(state!=None):
                action_list.append(action)
        else:
            break
    #print(action_list)
    #print(action_list.reverse())
    #L[::-1]
    #print(action_list[::-1])
    #return action_list.reverse()
    action_list.reverse() 
    return action_list

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    
    node = Node((problem.getStartState(), None, None))
    if problem.isGoalState(problem.getStartState()): return node.getPath()
    frontier = util.Stack()
    frontier.push(node)
    explored = set()
    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.state): return node.getPath()
        explored.add(node.state)
        succesor_list =problem.getSuccessors(node.state)

        for successor in succesor_list:
            child = Node(successor, node)
            print("child.state = ",child.state)
           
            if ( (child.state not in explored) and (child not in frontier.list) ): 
                #if problem.isGoalState(child.state): return child.getPath()
                frontier.push(child)
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    node = Node((problem.getStartState(), None, None))
    if problem.isGoalState(problem.getStartState()): return node.getPath()
    frontier = util.Queue()
    frontier.push(node)
    explored = set()
    while not frontier.isEmpty():
        node = frontier.pop()
        #if we check the node here the autograder runs perfectly
        #if problem.isGoalState(node.state): return node.getPath()
       
        explored.add(node.state)
        succesor_list =problem.getSuccessors(node.state)

        for successor in succesor_list:
            child = Node(successor, node)
            if ( (child.state not in explored) and child not in frontier.list): 
                if problem.isGoalState(child.state): return child.getPath() #checkaroume edw an einai goalstate
                frontier.push(child)
    return []

#na thn tsekarw pali gt den trexei o autograder
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    node = Node((problem.getStartState(), None, None))
    if problem.isGoalState(problem.getStartState()): return node.getPath()
    frontier = util.PriorityQueue()
    #frontier.update(node,node.pathCost)
    frontier.push(node,node.pathCost)
    explored = set()
    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.state): return node.getPath()
       
        explored.add(node.state)
        succesor_list =problem.getSuccessors(node.state)

        for successor in succesor_list:
            child = Node(successor, node)
            if ( (child.state not in explored) and child not in frontier.heap): 
                frontier.push(child,child.pathCost)
            elif (child.state in frontier.heap ):
                frontier.update(child,child.pathCost)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """


    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    node = Node((problem.getStartState(), None, None))
    if problem.isGoalState(problem.getStartState()): return node.getPath()
    frontier = util.PriorityQueue()
    #frontier.update(node,node.pathCost)
    if(node.pathCost ==None):    frontier.update(node, heuristic(node.state, problem))
    else:  frontier.update(node, node.pathCost+heuristic(node.state, problem))
    explored = set()
    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.state): return node.getPath()
       
        explored.add(node.state)
        succesor_list =problem.getSuccessors(node.state)

        for successor in succesor_list:
            child = Node(successor, node)

            if ( (child.state not in explored) and child not in frontier.heap): 
                 #if problem.isGoalState(child.state): return child.getPath() #checkaroume edw an einai goalstate
                 if(child.pathCost ==None):    frontier.update(child, heuristic(child.state, problem))
                 else:  frontier.update(child, child.pathCost+heuristic(child.state, problem))
   


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
