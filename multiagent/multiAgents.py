# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
	"""
	  A reflex agent chooses an action at each choice point by examining
	  its alternatives via a state evaluation function.

	  The code below is provided as a guide.  You are welcome to change
	  it in any way you see fit, so long as you don't touch our method
	  headers.
	"""


	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {North, South, West, East, Stop}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):#to succesorGameState einai to game me action to action dhladh h epomenh kinhsh 
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		#print("EVALUATIONNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
		# oloto board dhladh exei attributes gia diafora 
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		#newpos einai h epomenh thesh tou pacman
		newPos = successorGameState.getPacmanPosition()
		#newFood einai 
		newFood = successorGameState.getFood()
		#to newGhostStates einai lista me object pou exoun to state kathe fantasmatos(current position , action)
		newGhostStates = successorGameState.getGhostStates()#h states aytes einai meta to action
		#lista me pou exei toys xronous gia kathe fantasma poso fobatai dhladh [2,2,2] einai ta deyterolepta gia kathe fantasma
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

		"*** YOUR CODE HERE ***"
		

		if(action == 'Stop'):
			return float('-Inf')
		for ghost_state in newGhostStates:#an ton trwei sthn me epomenh kinhsh thn action tote epestrepse -INf
			if(ghost_state.getPosition() == tuple(newPos) and (ghost_state.scaredTimer == 0)):
				#print("EPISTREFEI -INF")
				return float('-Inf')



		
		#synarthsh pou epistrefei ton meso oro apostasewn apo fantasmata 
		def mesh_apostash_fantasmatwn(Pacman_position,Ghost_states):
			sum_distance = [manhattanDistance(Pacman_position,ghost_state.getPosition()) for ghost_state in Ghost_states] 
			sum_distance = sum(sum_distance)

			return sum_distance/ len(Ghost_states)



		#messos = mesh_apostash_fantasmatwn(newPos,newGhostStates)#ypologismos mesou orou
		
		min_food_distance =  float('Inf')#arxikopoiw me -apeiro to max
		min_ghost_distance = float('Inf')
		min_capsule_distance = float('Inf')
		min_scared_ghost_distance = float('Inf')

		pacmanPos = successorGameState.getPacmanPosition()
		ghostStates = successorGameState.getGhostStates()	

		Capsules = successorGameState.getCapsules()

		if(len(Capsules) == 0):
			#an den exoun meinei kapsoules 
			min_capsule_distance = 0 
		else:
			#ypologizoume thn apostash apo to kontinotero capsule
			for capsule in Capsules :
				temp  = 1.0/manhattanDistance(pacmanPos, capsule)#edw pairnoume to reciprocal
				if(temp< min_capsule_distance):
					min_capsule_distance = temp;

		#an ton trwei sthn me epomenh kinhsh thn action tote epestrepse -INf
		for ghost_state in ghostStates:
			if(ghost_state.getPosition() == tuple(pacmanPos) and (ghost_state.scaredTimer == 0)):
				#print("EPISTREFEI -INF")
				return float('-Inf')

		#vriskw thn pio kontinh apostash apo ta foods
		for food in newFood.asList() :
			temp = 1.0/manhattanDistance(pacmanPos, food)#edw pairnoume to reciprocal
			if(temp < min_food_distance):
				min_food_distance = temp

		#vriskw thn pio kontinh apostash apo ta ghosts
		if(len([x for x in newGhostStates if(x.scaredTimer >0 )]) >0 ):
			for ghost in newGhostStates:
				#leitourgoume mono gia ta not scared ghosts
				if(ghost.scaredTimer > 0):
					temp = 1.0/manhattanDistance(pacmanPos,ghost.getPosition())#pairnoume to reciprocal	
					if(temp < min_scared_ghost_distance):
						min_scared_ghost_distance = temp
		else:
			min_scared_ghost_distance = 0 


		#an exei faei capsule o pacman tote symfairei na faei kapoio kontino fantasma
		#efoson yparxoun fantasmata non scared
		if(len([x for x in newGhostStates if(x.scaredTimer == 0)]) >0 ):
			for ghost in newGhostStates:
				#leitourgoume mono gia ta not scared ghosts
				if(ghost.scaredTimer == 0):
					temp = 1.0/manhattanDistance(pacmanPos,ghost.getPosition())#pairnoume to reciprocal	
					if(temp < min_ghost_distance):
						min_ghost_distance = temp
		else: 
			min_ghost_distance =0

		return abs(scoreEvaluationFunction(successorGameState) - scoreEvaluationFunction(currentGameState)) + min_food_distance - min_ghost_distance + min_capsule_distance +min_scared_ghost_distance

		#sthn periptwsh pou o arithmos twn fobhsmenwn fantasmatwn einai 0		

		# if (ghostStates[0].scaredTimer != 0): 
		# 	#kanontas to min_ghost_distance arnhtiko einai sthn synexeia san na to prosthetoume sto value dhladh ay3anoume to value
		# 	min_ghost_distance = -1*min_ghost_distance


		#to value bgainei apo to score prosthetwntas to min_food_distance kai afairwntas to min_ghost_distance psosthetontas to min_capsule_distance kai prosthetwntas to min_scared_ghost_distance

		value = 0.2*scoreEvaluationFunction(successorGameState) + 0.4*min_food_distance - 0.1*min_ghost_distance + 0.2*min_capsule_distance +0.1*min_scared_ghost_distance
		
		#print(min_ghost_distance)

		# if(min_scared_ghost_distance > 0 ):
		# 	min_scared_ghost_distance = min_scared_ghost_distance *100

		messos = mesh_apostash_fantasmatwn(pacmanPos,newGhostStates)
		#messos = 1.0/messos
		#if(min_capsule_distance >= 0.1): return 10 #min_capsule_distance = 10
		
		# value = 0.8*min_food_distance - 0.2*min_ghost_distance  + min_capsule_distance +min_scared_ghost_distance
		# print(min_capsule_distance)

		if(messos < 4 ): value =  scoreEvaluationFunction(successorGameState) +0.9*min_food_distance - 0.1*min_ghost_distance + min_capsule_distance + min_scared_ghost_distance
		elif (messos<3): value =  scoreEvaluationFunction(successorGameState) +0.4*min_food_distance - 0.6*min_ghost_distance + min_capsule_distance + min_scared_ghost_distance
		else: value =  min_food_distance + min_capsule_distance + min_scared_ghost_distance- min_ghost_distance

		#value = scoreEvaluationFunction(successorGameState) +min_food_distance - min_ghost_distance + min_capsule_distance + min_scared_ghost_distance
		value = min_food_distance - (0.6)*min_ghost_distance
		print(value)
		return (-1)*value 
		#print(value)
		#return value
		#tried the reciprocal but didnt work
		# if(maxdistance != 0): maxdistance = 1/maxdistance
		# if(messos != 0): messos = 1/messos

		#print(maxdistance +messos)

		#an kata meso oro ta fantasmata einai konta kata <2 monades apostashs manhatan 
		# if(messos <4): value =   min_food_distance/0.7  + min_ghost_distance/0.3 #ayto pou epistrefoume kathorizetai 70% apo maxdistance kai 30% apo messh apostash fantasmatwn
		# if (messos<3): return  min_food_distance/0.4 +  min_ghost_distance/0.6  #ayto pou epistrefoume kathorizetai 40% apo maxdistance kai 60% apo messh apostash fantasmatwn
		# return   -1*min_food_distance/0.9 #+ min_ghost_distance/0.01 #90% maxdistance kai 10% messos


#def find_closer_food_distance(currentGameState):
  

def scoreEvaluationFunction(currentGameState):
	"""
	  This default evaluation function just returns the score of the state.
	  The score is the same one displayed in the Pacman GUI.

	  This evaluation function is meant for use with adversarial search agents
	  (not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	  This class provides some common elements to all of your
	  multi-agent searchers.  Any methods defined here will be available
	  to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	  You *do not* need to make any changes here, but you can if you want to
	  add functionality to all your adversarial search agents.  Please do not
	  remove anything, however.

	  Note: this is an abstract class: one that should not be instantiated.  It's
	  only partially specified, and designed to be extended.  Agent (game.py)
	  is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent (question 2)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action from the current gameState using self.depth
		  and self.evaluationFunction.

		  Here are some method calls that might be usef ul when implementing minimax.

		  gameState.getLegalActions(agentIndex):
			Returns a list of legal actions for an agent
			agentIndex=0 means Pacman, ghosts are >= 1

		  gameState.generateSuccessor(agentIndex, action):
			Returns the successor game state after an agent takes an action

		  gameState.getNumAgents():
			Returns the total number of agents in the game
		"""
		"*** YOUR CODE HERE ***"
		# print(self.depth)
		# print(dir(self.evaluationFunction) )
		# print(self.index)

		#print("\t\t\t\t\t\t\t\t\t\tNUMBEROF AGENTS ==== "+str(gameState.getNumAgents() ))
		#synarthsh pou epistrefei True h False an o kombos einai termatikos
		def is_terminalNode(minimaxdepth,gameState):
			if( (minimaxdepth == 0) or gameState.isWin() or gameState.isLose()):
				#return self.evaluationFunction(gameState)##an prokeitai gia termatiko kombo dhladh vathos == 0
				return True
			else :
				return False

		#h synarthsh epistrefei tuple me 2 stoixeia to prwto einai o arithmos max kai to deytero h best action me vash ton arithmo ayto
		def max_value(gameState, minimaxdepth,minimax_index):#sthn periptwsh mas to minimax_index = 0 pou einai o Pacman
			if(is_terminalNode(minimaxdepth,gameState)):#an o kombos einai termatikos
				return (self.evaluationFunction(gameState),"")
			v = float('-Inf')
			actionList = gameState.getLegalActions(minimax_index)
			
			bestaction  = actionList[0]#arxikopoiw to action 

			for action in actionList :
				next_state = gameState.generateSuccessor(minimax_index, action)
				minValue = min_value(next_state, minimaxdepth , minimax_index+1)#kaloyme thn min_value gia ton fantasma me minimax_index == 1 
				#kratame to max apo ta dyo
				if( minValue[0] > v):
					v = minValue[0]
					bestaction = action#action

			return (v,bestaction)


		#h synarthsh epistrefei tuple me 2 stoixeia to prwto einai o arithmos min kai to deytero h best action me vash ton arithmo ayto
		def min_value(gameState, minimaxdepth,minimax_index):#sthn periptwsh mas to minimax_index >= 1 pou einai ta fantasmata

			if(is_terminalNode(minimaxdepth,gameState)):#an o kombos einai termatikos
				return (self.evaluationFunction(gameState),"")
			v = float('Inf')
			actionList = gameState.getLegalActions(minimax_index) 
			
			bestaction  = actionList[0]#arxikopoiw to action 
			for action in actionList :
				next_state = gameState.generateSuccessor(minimax_index, action)
				if (minimax_index == gameState.getNumAgents()-1 ): #an eimaste sto teleytaio fantasma tote kaloume thn max_value dhladh ton pacman
					maxValue = max_value(next_state, minimaxdepth - 1,0)#to 0 einai gia ton index toy pacman 
				else :
					maxValue = min_value(next_state,minimaxdepth,minimax_index+1)	# edw kaloume thn min_value gia to fantasma me anagnwristiko minimax_index+1
				
				#kratame to max apo ta dyo
				if( maxValue[0] < v):
					v = maxValue[0]
					bestaction = action#action

			return (v,bestaction)


		def minimax(gameState, minimaxdepth,minimax_index):#minimaxdepth einai to bathos tou minimax kai minimax_index to 
			
			if(minimax_index == 0):
				return  max_value(gameState, minimaxdepth, minimax_index)
			elif(minimax_index >0):
				return min_value(gameState,minimaxdepth,minimax_index)

		bestaction = minimax(gameState, self.depth,0) #vazw 0 sto index gt 3ekina panta o pacman
		return bestaction[1]
		util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"

		def is_terminalNode(minimaxdepth,gameState):
			if( (minimaxdepth == 0) or gameState.isWin() or gameState.isLose()):
				#return self.evaluationFunction(gameState)##an prokeitai gia termatiko kombo dhladh vathos == 0
				return True
			else :
				return False

		#h synarthsh epistrefei tuple me 2 stoixeia to prwto einai o arithmos max kai to deytero h best action me vash ton arithmo ayto
		def max_value(gameState, minimaxdepth,minimax_index,alpha,beta):#sthn periptwsh mas to minimax_index = 0 pou einai o Pacman
			#print("\t\t\t\t\t\talpha = "+str(alpha) +"  " + " beta =  "+ str(beta))

			if(is_terminalNode(minimaxdepth,gameState)):#an o kombos einai termatikos
				return (self.evaluationFunction(gameState),"")
			v = float('-Inf')
			actionList = gameState.getLegalActions(minimax_index)
			
			bestaction  = actionList[0]#arxikopoiw to action 

			for action in actionList :
				next_state = gameState.generateSuccessor(minimax_index, action)
				minValue = min_value(next_state, minimaxdepth , minimax_index+1,alpha ,beta)#kaloyme thn min_value gia ton fantasma me minimax_index == 1 
				#kratame to max apo ta dyo
				if( minValue[0] > v):
					v = minValue[0]
					bestaction = action#action
				if(v > beta ):
					return (v,bestaction)
				alpha = max(alpha,v)
			return (v,bestaction)


		#h synarthsh epistrefei tuple me 2 stoixeia to prwto einai o arithmos min kai to deytero h best action me vash ton arithmo ayto
		def min_value(gameState, minimaxdepth,minimax_index,alpha,beta):#sthn periptwsh mas to minimax_index >= 1 pou einai ta fantasmata
			#print("alpha = "+str(alpha) +"  " + " beta =  "+ str(beta))
			if(is_terminalNode(minimaxdepth,gameState)):#an o kombos einai termatikos
				return (self.evaluationFunction(gameState),"")
			v = float('Inf')
			actionList = gameState.getLegalActions(minimax_index) 
			
			bestaction  = actionList[0]#arxikopoiw to action 
			for action in actionList :
				next_state = gameState.generateSuccessor(minimax_index, action)
				if (minimax_index == gameState.getNumAgents()-1 ): #an eimaste sto teleytaio fantasma tote kaloume thn max_value dhladh ton pacman
					maxValue = max_value(next_state, minimaxdepth - 1,0,alpha,beta)#to 0 einai gia ton index toy pacman 
				else :
					maxValue = min_value(next_state,minimaxdepth,minimax_index+1,alpha,beta)	# edw kaloume thn min_value gia to fantasma me anagnwristiko minimax_index+1
				
				#kratame to max apo ta dyo
				if( maxValue[0] < v):
					v = maxValue[0]
					bestaction = action#action
				if(v < alpha):
					return (v,bestaction)
				beta = min(beta,v)
			return (v,bestaction)


		def alpha_beta_search(gameState, alphabeta_depth,alphabeta_index):#minimaxdepth einai to bathos tou minimax kai minimax_index to 
			
			if(alphabeta_index == 0):
				return  max_value(gameState,alphabeta_depth, alphabeta_index,float('-Inf') ,float('Inf'))
			elif(alphabeta_index >0):
				return min_value(gameState,alphabeta_depth,alphabeta_index,float('-Inf') ,float('Inf'))

		bestaction = alpha_beta_search(gameState, self.depth,0) #vazw 0 sto index gt 3ekina panta o pacman
		return bestaction[1]
		util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		  Returns the expectimax action using self.depth and self.evaluationFunction

		  All ghosts should be modeled as choosing uniformly at random from their
		  legal moves.
		"""
		"*** YOUR CODE HERE ***"
		def is_terminalNode(minimaxdepth,gameState):
			if( (minimaxdepth == 0) or gameState.isWin() or gameState.isLose()):
				#return self.evaluationFunction(gameState)##an prokeitai gia termatiko kombo dhladh vathos == 0
				return True
			else :
				return False

		#h synarthsh epistrefei tuple me 2 stoixeia to prwto einai o arithmos max kai to deytero h best action me vash ton arithmo ayto
		def max_value(gameState, minimaxdepth,minimax_index):#sthn periptwsh mas to minimax_index = 0 pou einai o Pacman
			if(is_terminalNode(minimaxdepth,gameState)):#an o kombos einai termatikos
				return (self.evaluationFunction(gameState),"")
			v = float('-Inf')
			actionList = gameState.getLegalActions(minimax_index)
			
			bestaction  = actionList[0]#arxikopoiw to action 

			for action in actionList :
				next_state = gameState.generateSuccessor(minimax_index, action)
				minValue = expecti_max_value(next_state, minimaxdepth , minimax_index+1)#kaloyme thn min_value gia ton fantasma me minimax_index == 1 
				#kratame to max apo ta dyo
				if( minValue[0] > v):
					v = minValue[0]
					bestaction = action#action

			return (v,bestaction)


		#h synarthsh epistrefei tuple me 2 stoixeia to prwto einai o arithmos min kai to deytero h best action me vash ton arithmo ayto
		def expecti_max_value(gameState, minimaxdepth,minimax_index):#sthn periptwsh mas to minimax_index >= 1 pou einai ta fantasmata

			if(is_terminalNode(minimaxdepth,gameState)):#an o kombos einai termatikos
				return (self.evaluationFunction(gameState),"")
			v = float('Inf')

			actionList = gameState.getLegalActions(minimax_index) 
			probability = 1.0 /len(actionList)# h pithanothta ths kathe action tou fantasmatou
			v = 0
			bestaction  = actionList[0]#arxikopoiw to action 
			for action in actionList :
				next_state = gameState.generateSuccessor(minimax_index, action)
				if (minimax_index == gameState.getNumAgents()-1 ): #an eimaste sto teleytaio fantasma tote kaloume thn max_value dhladh ton pacman
					maxValue = max_value(next_state, minimaxdepth - 1,0)#to 0 einai gia ton index toy pacman 
				else :
					maxValue = expecti_max_value(next_state,minimaxdepth,minimax_index+1)	# edw kaloume thn min_value gia to fantasma me anagnwristiko minimax_index+1
				
				#kratame to max apo ta dyo
				#if( maxValue[0] < v):
				v += probability * maxValue[0]
				bestaction = action#action

			return (v,bestaction)


		def minimax(gameState, minimaxdepth,minimax_index):#minimaxdepth einai to bathos tou minimax kai minimax_index to 
			
			if(minimax_index == 0):
				return  max_value(gameState, minimaxdepth, minimax_index)
			elif(minimax_index >0):
				return expecti_max_value(gameState,minimaxdepth,minimax_index)

		bestaction = minimax(gameState, self.depth,0) #vazw 0 sto index gt 3ekina panta o pacman
		return bestaction[1]
		util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: <write something here so we know what you did>
	"""
	"*** YOUR CODE HERE ***"



	min_food_distance =  float('Inf')#arxikopoiw me -apeiro to max
	min_ghost_distance = float('Inf')
	min_capsule_distance = float('Inf')
	min_scared_ghost_distance = float('Inf')

	pacmanPos = currentGameState.getPacmanPosition()
	ghostStates = currentGameState.getGhostStates()	

	Capsules = currentGameState.getCapsules()


	#an ton trwei fantasma tote epestrepse -INf
	for ghost_state in ghostStates:
		if(ghost_state.getPosition() == tuple(pacmanPos) and (ghost_state.scaredTimer == 0)):
			#print("EPISTREFEI -INF")
			return float('-Inf')

	if(len(Capsules) == 0):
		#an den exoun meinei kapsoules 
		min_capsule_distance = 0 
	else:
		#ypologizoume thn apostash apo to kontinotero capsule
		for capsule in Capsules :
			temp  = 1.0/manhattanDistance(pacmanPos, capsule)#edw pairnoume to reciprocal
			if(temp< min_capsule_distance):
				min_capsule_distance = temp;

	

    #vriskw thn pio kontinh apostash apo ta foods
	for food in currentGameState.getFood().asList() :
		temp = 1.0/manhattanDistance(pacmanPos, food)#edw pairnoume to reciprocal
		if(temp < min_food_distance):
			min_food_distance = temp

	#vriskw thn pio kontinh apostash apo ta ghosts
	#efoson yparxoun fantasmata  scared
	if(len([x for x in ghostStates if(x.scaredTimer >0 )]) >0 ):
		for ghost in ghostStates:
			#leitourgoume mono gia ta not scared ghosts
			if(ghost.scaredTimer > 0):
				temp = 1.0/manhattanDistance(pacmanPos,ghost.getPosition())#pairnoume to reciprocal	
				if(temp < min_scared_ghost_distance):
					min_scared_ghost_distance = temp
	else:
		min_scared_ghost_distance = 0 


	#an exei faei capsule o pacman tote symfairei na faei kapoio kontino fantasma
	#efoson yparxoun fantasmata non scared
	if(len([x for x in ghostStates if(x.scaredTimer == 0)]) >0 ):
		for ghost in ghostStates:
			#leitourgoume mono gia ta not scared ghosts
			if(ghost.scaredTimer == 0):
				temp = 1.0/manhattanDistance(pacmanPos,ghost.getPosition())#pairnoume to reciprocal	
				if(temp < min_ghost_distance):
					min_ghost_distance = temp
	else: 
		min_ghost_distance =0

	#sthn periptwsh pou o arithmos twn fobhsmenwn fantasmatwn einai 0		
	
	# if (ghostStates[0].scaredTimer != 0): 
	# 	#kanontas to min_ghost_distance arnhtiko einai sthn synexeia san na to prosthetoume sto value dhladh ay3anoume to value
	# 	min_ghost_distance = -1*min_ghost_distance
	

	#to value bgainei apo to score prosthetwntas to min_food_distance kai afairwntas to min_ghost_distance psosthetontas to min_capsule_distance kai prosthetwntas to min_scared_ghost_distance
	#print(min_ghost_distance)

	value = scoreEvaluationFunction(currentGameState) + min_food_distance - min_ghost_distance + min_capsule_distance +min_scared_ghost_distance
	#value = scoreEvaluationFunction(currentGameState) + min_food_distance - min_ghost_distance 
	#print(value)
	return value

# Abbreviation
better = betterEvaluationFunction


