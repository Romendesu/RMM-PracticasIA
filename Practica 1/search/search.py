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
from game import Directions
from typing import List

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




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:

    from util import Stack
    # 1) Definimos las variables que vamos a emplear

    visitedNodes = set()
    stack = Stack()
    initialState = problem.getStartState()

    # 2) Definimos el nodo como una tupla bidimensional, el cual contendra los siguientes parametros:
    #   1. La posicion del comecocos en el tablero
    #   2. Un array dinamico con el camino recorrido
    node = (initialState, [])

    # 3) Encolamos el nodo
    stack.push(node)

    # 4) Iniciamos el algoritmo
    while not stack.isEmpty():
    
        # 5) Empezamos el analisis del algoritmo
        state, path = stack.pop()

        # 6) En caso de haber visitado el nodo, omitimos
        if state in visitedNodes:
            continue
        visitedNodes.add(state)
        # 7) Verificamos si se ha encontrado el nodo
        if (problem.isGoalState(state)):
            return path
        
        # 8) Al no encontrarse, analizaremos cada casilla adyacente
        for adyState, action, _ in problem.getSuccessors(state):

            # 9) En caso de no haber visitado el nodo, lo visitamos
            if (adyState not in visitedNodes):
                # 10) Hacemos un push del nodo visitado
                newPath = path + [action]
                node = (adyState, newPath)
                stack.push(node)
                
                
    # En caso de no encontrar el camino hacia la meta, devolvemos un array vacio
    return []

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    # 1) Importamos las estructuras de datos que vamos a emplear
    from util import Queue
    # 2) Definimos las estructuras de datos que vamos a emplear
    queue = Queue()
    visitedNodes = set()
    initialState = problem.getStartState()
    # 3) Definimos el nodo (NOTA: El nodo empleado es igual que el ejercicio anterior)
    # Parametros: La posicion inicial del comecocos en el tablero, Una lista de direcciones
    node = (initialState, [])

    # 4) Añadimos el nodo a la cola y lo marcamos como visitado
    visitedNodes.add(initialState)
    queue.push(node)
    # 5) Iteramos el algoritmo: Condicion de cierre: Se ha encontrado una meta o no se ha podido encontrarla
    while not queue.isEmpty():
        # 6) Analizamos los parametros del nodo
        state, path = queue.pop()
        # 7) Vemos si hemos alcanzado la meta
        if problem.isGoalState(state): return path
        # 8) En caso contrario, visitamos los nodos adyacentes
        for adyState, action, _ in problem.getSuccessors(state=state):
            # 9) En caso de no haber visitado la casilla, la marcamos como visitada
            if (adyState not in visitedNodes):
                newPath = path + [action]       # Añadimos la dirección del nodo a un nuevo array
                # 10) Encolamos nuestro nuevo nodo
                node = (adyState, newPath)
                queue.push(node)
                # 11) Marcamos el nodo como visitado
                visitedNodes.add(adyState)

    # No regresa nada si no existe la meta
    return []


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    from util import PriorityQueue

    priorityQueue = PriorityQueue()
    visitedNodes = set()

    # Estado inicial: (state, path, totalCost)
    initialState = problem.getStartState()
    priorityQueue.push((initialState, [], 0), 0)

    while not priorityQueue.isEmpty():
        state, path, totalCost = priorityQueue.pop()

        # Evitar reexpandir estados
        if state in visitedNodes:
            continue

        visitedNodes.add(state)

        # Comprobamos objetivo
        if problem.isGoalState(state):
            return path

        # Expandimos sucesores
        for successorState, action, stepCost in problem.getSuccessors(state):
            if successorState not in visitedNodes:
                newPath = path + [action]
                newCost = totalCost + stepCost
                priorityQueue.push((successorState, newPath, newCost), newCost)

    return []

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    # Importar estructuras de datos
    from util import PriorityQueue

    start = problem.getStartState()
    frontier = PriorityQueue()
    # Nodo
    frontier.push((start, [], 0), heuristic(start, problem)) 
    costSoFar = {start: 0}  # costo acumulado desde el inicio a cada nodo

    while not frontier.isEmpty():
        state, path, g = frontier.pop()
        # Si el estado es el de meta, finaliza el algoritmo
        if problem.isGoalState(state):
            return path
        # Recorremos cada vecino
        for successor, action, stepCost in problem.getSuccessors(state):
            new_g = g + stepCost       
            # En caso de encontrar un camino más barato para el sucesor
            if successor not in costSoFar or new_g < costSoFar[successor]:
                costSoFar[successor] = new_g
                f = new_g + heuristic(successor, problem)               # Calculamos la prioridad
                frontier.push((successor, path + [action], new_g), f)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
