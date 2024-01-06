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
import random
import util

from game import Agent
from pacman import GameState
import heapq

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="4", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    def print_grid(self, grid, delimeter = ''):
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                print(grid[i][j], end=delimeter)
            print()

    def create_grid(self, state: GameState):
        w = state.getWalls().width
        h = state.getWalls().height

        capsules = state.getCapsules()
        ghost_pos = state.getGhostPosition(agentIndex=1)
        pacman_pos = state.getPacmanPosition()

        grid = [[' ' for x in range(w)] for y in range(h)]
        for i in range(h):
            for j in range(w):
                if state.hasFood(j, h - i - 1):
                    grid[i][j] = '+'
                
                if state.hasWall(j, h - i - 1):
                    grid[i][j] = 'W'

        for caps in capsules:
            grid[h - int(caps[1]) - 1][int(caps[0])] = '*'

        grid[h - int(ghost_pos[1]) - 1][int(ghost_pos[0])] = 'G'

        grid[h - int(pacman_pos[1]) - 1][int(pacman_pos[0])] = 'P'

        return grid

    def is_terminal(self, state: GameState):
        return state.isLose() or state.isWin()

    def heuristic(self, state: GameState):
        grid = self.create_grid(state)
        w = state.getWalls().width
        h = state.getWalls().height
        
        capsules = state.getCapsules()
        ghost_pos = state.getGhostPosition(agentIndex=1)
        pacman_pos = state.getPacmanPosition()
        
        x = h - int(pacman_pos[1]) - 1
        y = int(pacman_pos[0])
        
        pq = []
        dis = [[float("inf") for i in range(w)] for j in range(h)]
        vis = [[False for i in range(w)] for j in range(h)]
        
        heapq.heappush(pq, (0, (x, y)))
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        
        while len(pq) != 0:
            u = heapq.heappop(pq)
            x = u[1][0]
            y = u[1][1]

            if vis[x][y]:
                continue
            
            vis[x][y] = True
            dis[x][y] = u[0]
                
            for i in range(4):
                if grid[x + dx[i]][y + dy[i]] == 'G':
                    dis[x + dx[i]][y + dy[i]] = dis[x][y] + 1
                    
                elif grid[x + dx[i]][y + dy[i]] != 'W':
                    if dis[x + dx[i]][y + dy[i]] > dis[x][y] + 1:
                        dis[x + dx[i]][y + dy[i]] = dis[x][y] + 1
                        heapq.heappush(pq, (dis[x + dx[i]][y + dy[i]], (x + dx[i], y + dy[i])))
                    
        value_state = 0
        
        for caps in capsules:
            value_state += 100 / (dis[h - int(caps[1]) - 1][int(caps[0])] + 0.1)
             
        value_state -= 100 / (dis[h - int(ghost_pos[1]) - 1][int(ghost_pos[0])] + 0.1)
        
        for i in range(h):
            for j in range(w):
                if grid[i][j] == '+': 
                    value_state += 20 / dis[i][j]
 
        return value_state
        
        
    def min_value(self, state: GameState, d, alpha, beta):
        if self.is_terminal(state) or d == self.depth:
            return self.heuristic(state) + state.getScore()

        value = float("inf")
        for action in state.getLegalActions(agentIndex=1):
            tmpState = state.deepCopy()
            value = min(value, self.max_value(tmpState.generateSuccessor(
                agentIndex=1, action=action), d + 1, alpha, beta))
            if value <= alpha:
                return value

            beta = min(beta, value)
        return value

    def max_value(self, state: GameState, d, alpha, beta):
        if self.is_terminal(state) or d == self.depth:
            return self.heuristic(state) + state.getScore()

        value = float("-inf")
        for action in state.getLegalActions():
            tmpState = state.deepCopy()
            value = max(value, self.min_value(
                tmpState.generatePacmanSuccessor(action), d + 1, alpha, beta))
            if value >= beta:
                return value

            alpha = max(alpha, value)
        return value

    def getAction(self, gameState: GameState):
        """
        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # util.raiseNotDefined()

        self.print_grid(self.create_grid(gameState))
        print("------------------------------------")
        self.heuristic(gameState)

        legal_actions = gameState.getLegalPacmanActions()
        print(legal_actions)
        best_action = legal_actions[0]
        max_value = float("-inf")
        for action in legal_actions:
            pState = gameState.deepCopy()
            value = self.max_value(pState.generatePacmanSuccessor(
                action), 0, float("-inf"), float("inf"))

            print(action, value)

            if action == "Stop":
                value -= 1

            if value >= max_value:
                max_value = value
                best_action = action

        print("DO:", best_action)

        return best_action
