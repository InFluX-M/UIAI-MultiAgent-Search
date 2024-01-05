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
    def is_terminal(self, state: GameState):
        return state.isLose() or state.isWin()

    def min_value(self, state: GameState, d, alpha, beta):
        if self.is_terminal(state) or d == self.depth:
            return state.getScore()
    
        value = float("inf")
        for action in state.getLegalActions(agentIndex=1):
            value = min(value, self.max_value(state.generateSuccessor(agentIndex=1, action=action), d + 1, alpha, beta))
            # if value <= alpha:
            #     return value
            
            beta = min(beta, value)
        return value
        

    def max_value(self, state: GameState, d, alpha, beta):
        if self.is_terminal(state) or d == self.depth:
            return state.getScore()
    
        value = float("-inf")
        for action in state.getLegalActions():
            value = max(value, self.min_value(state.generatePacmanSuccessor(action), d + 1, alpha, beta))
            # if value >= beta:
            #     return value
            
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

        legal_actions = gameState.getLegalPacmanActions()
        print(legal_actions)
        best_action = legal_actions[0]
        max_value = float("-inf")
        for action in legal_actions:
            value = self.max_value(gameState.generatePacmanSuccessor(action), 0, float("-inf"), float("inf"))

            print(action, value)

            if action == "Stop":
                value -= 1

            if value >= max_value:
                max_value = value
                best_action = action

        print("DO:", best_action)

        return best_action
