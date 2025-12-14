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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        '''
        CRITERIO EMPLEADO
        1. Siempre se calculara la distancia con la comida mas cercana, sumando a la puntuacion su inversa. 
        2. El peso de cada accion se determinara por la posicion del fantasma mas cercana. 
        3. Solo los fantasmas peligrosos se castigaran de forma muy fuerte
        4. Si la comida esta muy cerca, el valor del reciproco sera cercano a 1. Si esta muy lejos, sera lejano a 1
        '''

        # Variables que vamos a emplear en el programa
        successorGameState = currentGameState.generatePacmanSuccessor(action)       # Posicion del comecocos
        newPos = successorGameState.getPacmanPosition()                             # Nueva posicion del comecocos
        newFood = successorGameState.getFood()                                      # Posicion de la comida                                  
        newGhostStates = successorGameState.getGhostStates()                        # Posicion de todos los fantasmas
        score = successorGameState.getScore()                                       # Puntuacion del programa

        # Convertimos la comida como lista
        foodList = newFood.asList()

        # Obtenemos la distancia de Manhattan para cada comida
        foodDistancesList = [manhattanDistance(newPos, foodList[index]) for index in range(len(foodList))]

        # En caso de no haber distancias para la comida, hemos ganado
        if not foodDistancesList:
            return score
        
        # Obtenemos la distancia de Manhattan para cada comida
        ghostsDistancesList = []
        for ghostState in newGhostStates:
            # Siempre miraremos por los fantasmas peligrosos (No estan asustados)
            if ghostState.scaredTimer == 0:
                ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
                ghostsDistancesList.append(ghostDistance)

        # Premiamos primeramente por la comida
        foodPrize = 1 / (min(foodDistancesList) + 1)
        score += foodPrize

        # Castigamos segun la posicion del fantasma mas cercano
        if ghostsDistancesList:
            ghostPenalization = 1 / (min(ghostsDistancesList) + 1)
            score -= 10 * ghostPenalization

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        '''
        Tomar en cuenta:
        1) Pacman siempre sera el agentIndex = 0
        2) Los fantasmas seran agentIndex = 1, 2, ..., n
        3) Pacman siempre maximizara el score, los fantasmas lo minimizan
        4) En caso de profundidad, si depth = 1, es Pacman y todos los fantasmas
        '''
        numAgents = gameState.getNumAgents()
        def helper(gameState: GameState, agentIndex, depth):
            # Caso base para la recursion: Si la condicion es victoria o derrota, se devuelve la funcion evaluacion
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                return self.evaluationFunction(gameState)
            
            # Asignacion del mejor valor inicial (-infinito si es pacman, infinito si es un fantasma)
            bestValue = float("-inf") if agentIndex == 0 else float("inf")
            
            # Caso de recursion
            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex,action)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth

                # Llamada recursiva y obtencion del mejor valor
                value = helper(successor, nextAgent, nextDepth)
                bestValue = max(bestValue, value) if agentIndex == 0 else min(bestValue, value)            
            
            return bestValue
        
        # Elegir la accion optima del Pacman
        legalActionsPacman = gameState.getLegalActions(0)
        bestScore = float("-inf")
        bestAction = None

        for action in legalActionsPacman:
            # Recorrer cada accion en las acciones legales del comecocos
            successor = gameState.generateSuccessor(0, action)
            value = helper(successor, 1, 0)  

            if value > bestScore:
                bestScore = value
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        # Inicializamos alpha y beta
        alpha = float("-inf")
        beta = float("inf")

        # Variables empleadas
        legalActions = gameState.getLegalActions(0)
        bestScore = float("-inf")
        bestAction = None
        numAgents = gameState.getNumAgents()
        
        # Funciones auxiliares
        def value(gameState, agentIndex, depth, alpha, beta):
            # Caso base: estado terminal o profundidad máxima
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                return self.evaluationFunction(gameState)

            if (agentIndex == 0):  # Pacman -> MAX
                return maxValue(gameState, agentIndex, depth, alpha, beta)
            else:  # Fantasma -> MIN
                return minValue(gameState, agentIndex, depth, alpha, beta)
            
        # Aplicamos el pseudocodigo proporcionado para maxValue y minValue
        def maxValue(gameState, agentIndex, depth, alpha, beta):
            v = float('-inf')
            numAgents = gameState.getNumAgents()

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth

                v = max(v, value(successor, nextAgent, nextDepth, alpha, beta))
                if v > beta:  
                    return v
                alpha = max(alpha, v)

            return v

        def minValue(gameState, agentIndex, depth, alpha, beta):
            v = float('inf')
            numAgents = gameState.getNumAgents()

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth

                v = min(v, value(successor, nextAgent, nextDepth, alpha, beta))
                if v < alpha:  
                    return v
                beta = min(beta, v)

            return v
        
        # Recorremos todas las acciones legales
        for action in legalActions:
            # Obtenemos el sucesor y la puntuacion
            successor = gameState.generateSuccessor(0, action)
            score = value(successor, 1, 0, alpha, beta)

            # Si la puntuación supera a la maxima puntuacion, actualizamos los datos
            if (score > bestScore):
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
            
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Variables iniciales
        numAgents = gameState.getNumAgents()
        bestScore = float("-inf")
        bestAction = None

        # Funcion recursiva
        def expectimax(gameState: GameState, agentIndex, depth):
            # Caso base
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                return self.evaluationFunction(gameState)
            # Caso recursivo
            if (agentIndex == 0):
                # logica del pacman
                v = float("-inf")
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex,action)
                    nextAgent = (agentIndex + 1) % numAgents
                    nextDepth = depth + 1 if nextAgent == 0 else depth
                    v = max(v,expectimax(successor,nextAgent, nextDepth))
                return v
            else:
                # Logica fantasma
                v = 0
                actions = gameState.getLegalActions(agentIndex)
                probability = 1 / len(actions)
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    nextAgent = (agentIndex+1) % numAgents
                    nextDepth = depth + 1 if nextAgent == 0 else depth
                    v += probability * expectimax(successor,nextAgent,nextDepth)
                return v
            
        # Recorrer acciones legales
        for action in gameState.getLegalActions(0):  # Pacman
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(successor, 1, 0)  # primer fantasma, depth=0

            if score > bestScore:
                bestScore = score
                bestAction = action
        # Devolver la accion con mayor valor
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Empleando una serie de criterios, ajustamos el estado del comecocos.
    1) Estado muy bueno: Hay comida cerca del comecocos, y los fantasmas están muy alejados.
    2) Estado bueno: Hay comida más cercana se encuentra a una distancia razonable, y los fantasmas más cercanos están alejados.
    3) Estado medio: La distancia entre la comida más cercana y los fantasmas más cercanos es aproximadamente igual.
    4) Estado malo: La comida más cercana está medianamente cerca, pero los fantasmas están más cerca
    5) Crítico: Los fantasmas más cercanos están al lado del Pacman, independientemente de la distancia de la comida
    """
    "*** YOUR CODE HERE ***"

    # Variables iniciales
    baseScore = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostsStates = currentGameState.getGhostStates()

    # Filtramos los fantasmas
    scaryGhostsDistancesList = []
    dangerGhostsDistancesList = []
    for ghostState in ghostsStates:
        ghostDistance = manhattanDistance(pacmanPos, ghostState.getPosition())
        if (ghostState.scaredTimer == 0):
            dangerGhostsDistancesList.append(ghostDistance)
        else:
            scaryGhostsDistancesList.append(ghostDistance)
    
    # Filtramos la comida: OJO, LA DISTANCIA MINIMA PUEDE SER 0
    foodDistance = [manhattanDistance(pacmanPos, foodList[i]) for i in range(len(foodList))]

    # Establecemos los criterios: En caso de no haber distancia minima, establecemos un valor neutro (comida), en los fantasmas infinito.
    minimumFoodDistance = min(foodDistance) if len(foodDistance) != 0 else 0
    minimumDangerGhostDistance = min(dangerGhostsDistancesList) if dangerGhostsDistancesList else float('inf')

    # Aplicamos reciprocos: Comida = +, Fantasma asustado = +, Fantasma normal = -, Comida restante = -
    foodParameter = 1 / (minimumFoodDistance + 1)
    dangerGhostParameter = sum(1/(d+1) for d in dangerGhostsDistancesList) if dangerGhostsDistancesList else 0
    scaryGhostParameter = sum(1/(d+1) for d in scaryGhostsDistancesList) if scaryGhostsDistancesList else 0

    # Establecemos los pesos: Jerarquia (+ -): 1) Fantasmas peligrosos, 2) Comida cercana, 3) Fantasma asustado, 4) Comida restante
    dangerGhostWeight = 30 if minimumDangerGhostDistance <= 1 else 50 # Penalizacion dura si se encuentra en un estado critico (fantasma muy cerca)
    foodWeight = 15
    scaryGhostWeight = 10
    restFoodWeight = 2

    # Establecemos los ajustes
    score = baseScore
    score += (foodWeight * foodParameter)                       # Ajuste por distancia de la comida
    score -= (dangerGhostWeight * dangerGhostParameter)         # Ajuste por fantasma peligroso
    score += (scaryGhostWeight * scaryGhostParameter)           # Ajuste por fantasma asustado
    score += restFoodWeight * (1 / (len(foodList) + 1))         # Ajuste por comida restante
    return score

# Abbreviation
better = betterEvaluationFunction
