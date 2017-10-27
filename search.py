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
           
            if ( (child.state not in explored)):# and child not in frontier.list): 
                #if problem.isGoalState(child.state): return child.getPath()
                frontier.push(child)
    return []

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

class Nodeilias:

    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Nodei %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.getSuccessors(self.state)]

    def child_node(self, problem, action):
        "[Figure 3.10]"
        next = action[0]
        print("action0 = ",next)
        return Nodei(next, self, action[1], self.path_cost+action[2])

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    #gia thn isothta twn Node
    def __eq__(self, other):
        return isinstance(other, Nodei) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


def breadthFirstSearchilias(problem):
    print("liakou")
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    node = Nodei(problem.getStartState())
    if problem.isGoalState(problem.getStartState()): return node.solution()
    frontier = util.Queue()
    frontier.push(node)
    explored = set()
    while not frontier .isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.state): return node.solution()
        explored.add(node.state)
        for child in node.expand(problem):
            if (child.state not in explored) and (child not in frontier.list):
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
        if problem.isGoalState(node.state): return node.getPath()
        #if(node.state  in explored): continue
        explored.add(node.state)
        #print("frontier (queue )  == ",frontier.list)
        succesor_list =problem.getSuccessors(node.state)

        for successor in succesor_list:
            child = Node(successor, node)
            print("child.state = ",child.state)
            if ( (child.state not in explored) and child not in frontier.list): #and (flag ==0) ): #and (child not in frontier.list):
                #if problem.isGoalState(child.state): return child.getPath()
                frontier.push(child)
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
