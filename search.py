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
    open_set = util.Stack()
    #empty set to maintain visited nodes
    closed_set = set()    
    meta = dict()  # key -> (parent state, action to reach child)    print("AAAAAAAAAAAAA")

# initialize
    start = problem.getStartState()
    meta[start] = (None, None)
    open_set.push(start)

    while not open_set.isEmpty():
        #print("Mpeike sto while\n")
        parent_state = open_set.pop()
        #print("meta = "+str(meta))
        if problem.isGoalState(parent_state):
            mypath  = construct_path(parent_state, meta)
            #print("mypath = " + str(mypath)+"  parent state = "+str(parent_state))
            return mypath
        #adding the state of the node to the exploredset
        closed_set.add(parent_state)

        for (child_state, action,cost) in problem.getSuccessors(parent_state):
            #print(child_state,action)
            # if child_state in closed_set:
            #     continue
            #print("mpeike sthn for")
            if child_state not in open_set.list and child_state not in closed_set:
                meta[child_state] = (parent_state, action)
                open_set.push(child_state)

        closed_set.add(parent_state)

    print "No solution found"
    return []
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
def breadthFirstSearch2(problem):
 
    frontier = util.Queue()
    startNode = Node((problem.getStartState(), None, None))

    #Check if start node is goal
    if problem.isGoalState(startNode.state):
        return []

    for successors in problem.getSuccessors(problem.getStartState()):
        newNode = Node(successors, startNode)
        frontier.push(newNode)

    explored = list()
    explored.append(startNode.state)

    while not frontier.isEmpty():
        leafNode = frontier.pop()
        if problem.isGoalState(leafNode.state):
            path = leafNode.getPath()
            #print("path = "+str(path) )
            return path
        explored.append(leafNode.state)
        for successor in problem.getSuccessors(leafNode.state):
            newNode = Node(successor, leafNode)
            if newNode.state not in frontier.list and newNode.state not in explored:
                frontier.push(newNode)
    print "No solution found"
    return []

def breadthFirstSearch1(problem):
    """Search the shallowest nodes in the search tree first."""
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    print(dir(problem))


    open_set = util.Queue()
    #empty set to maintain visited nodes
    closed_set = set()    
    meta = dict()  # key -> (parent state, action to reach child)    print("AAAAAAAAAAAAA")

# initialize
    start = problem.getStartState()
    meta[start] = (None, None)
    open_set.push(start)

    while not open_set.isEmpty():
        #print("Mpeike sto while\n")
        parent_state = open_set.pop()
        #print("meta = "+str(meta))
        if problem.isGoalState(parent_state):
            mypath  = construct_path(parent_state, meta)
            #print("mypath = " + str(mypath)+"  parent state = "+str(parent_state))
            return mypath
        #adding the state of the node to the exploredset
        closed_set.add(parent_state)

        for (child_state, action,cost) in problem.getSuccessors(parent_state):
            #print(child_state,action)
            # if child_state in closed_set:
            #     continue
            #print("mpeike sthn for")
            if child_state not in open_set.list and child_state not in closed_set:
                meta[child_state] = (parent_state, action)
                open_set.push(child_state)

        closed_set.add(parent_state)

    print "No solution found"
    return []
    util.raiseNotDefined()
# def expand(self, problem):
# #"List the nodes reachable in one step from this node."
#     return [self.child_node(problem, action)
#                 for action in problem.getSuccessors(self.state)]

# def child_node(self, problem, action):
# #        "[Figure 3.10]"
#     next = action[0]
#     return Node(next, self, action[1], self.path_cost+action[2])


class Nodei:

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
            flag =0#mas deixnei an to child yparxei hdh sthn oura

            for x in frontier.list :
                print(x.state)
                if child.state == x.state :
                    flag =1
                    
            if ( (child.state not in explored) and (flag ==0) ): #and (child not in frontier.list):
                frontier.push(child)
    return []
def breadthFirstSearchdikia(problem):
   
  
    print("yolose")
    startNode = Node((problem.getStartState(), None, None))

    #Check if start node is goal
    if problem.isGoalState(startNode.state):
        return []
    frontier = util.Queue()

    frontier.push(startNode);

    

    explored =set()
    expanded_list = []
    ####loooooop
    while not frontier.isEmpty():
        poped_Node = frontier.pop()#popping Node
        if(problem.isGoalState(poped_Node.state)):  return poped_Node.getPath()

        explored.add(poped_Node.state)#adding Node to explored set

        # if(poped_Node.state in expanded_list):
        #     #print("o KOMBOS ",poped_Node.state  )
        #     if(poped_Node.state in explored): 
        #         print("to if me to set den doulepseeeeeeeeee")
        #     continue
        expanded_list.append(poped_Node.state)
        succesor_list =problem.getSuccessors(poped_Node.state)
        #print(succesor_list)
        for successor in succesor_list:
            newNode = Node(successor, poped_Node)
            
            if(newNode.state not in frontier.list ):
                if(newNode.state not in explored):
                    if(newNode.state not in expanded_list):
                        frontier.push(newNode)
    print "No solution found"
    return []

def breadthFirstSearch10(problem):
   
    #Node(successor, parent,action)

    print("yolose")
    startNode = Node((problem.getStartState(), None, None))

    #Check if start node is goal
    if problem.isGoalState(startNode.state):
        return []
    #frontier.push(newNode)
    #an o kombos den einai  o telikos tote vazw ta paidia sto queue
    #successors(geitones) =   [((34, 15), 'South', 1), ((33, 16), 'West', 1)]
    frontier = util.Queue()

    frontier.push(startNode);

    # for successors in problem.getSuccessors(problem.getStartState()):
    #     newNode = Node(successors, startNode)
    #     frontier.push(newNode)

    explored =list() #set()
    #explored.add(startNode.state)
    expanded_list = []
    while not frontier.isEmpty():
        poped_Node = frontier.pop()#popping Node
       # print("poped_NODE = ",poped_Node.state)
        #print("expanded list = ",expanded_list)
        if(problem.isGoalState(poped_Node.state)):
            return poped_Node.getPath()

        explored.append(poped_Node.state)#adding Node to explored set

        # if problem.isGoalState(Node.state): #if is goal return path
        #     path = leafNode.getPath()
        #     #print("path = "+str(path) )
        #     return path
        #gia kathe geitona kanw
        # if(poped_Node.state in expanded_list):
        #     #print("o KOMBOS ",poped_Node.state  )
        #     if(poped_Node.state in explored): 
        #         print("to if me to set den doulepseeeeeeeeee")
        #     continue
        expanded_list.append(poped_Node.state)
        succesor_list =problem.getSuccessors(poped_Node.state)
        #print(succesor_list)
        for successor in succesor_list:
            newNode = Node(successor, poped_Node)
            for i in range(len(frontier.list)):
                if(newNode.state == frontier.list[i].state):continue


            if ( (newNode not in frontier.list )and  (newNode.state not in explored)):
                #an o geitonas autos einai telikos goal tote epistrefoume to path toy 
                # if problem.isGoalState(newNode.state):
                #     return newNode.getPath()
                #if()
                frontier.push(newNode)
    print "No solution found"
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
