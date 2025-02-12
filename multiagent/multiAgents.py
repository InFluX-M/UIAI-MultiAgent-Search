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


import random
import util
import os
from game import Agent
from pacman import GameState
import heapq

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function returns the score of the state,
    which is the same score displayed in the Pacman GUI.
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This abstract class provides common elements for all multi-agent searchers.
    It is extended by specific agents (e.g., AIAgent) to implement custom search strategies.
    """
    def __init__(self, evalFn="scoreEvaluationFunction", depth="4", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)

class AIAgent(MultiAgentSearchAgent):
    """
    AIAgent implements a minimax search algorithm with alpha-beta pruning for the Pacman game.
    It uses a custom heuristic to evaluate game states and chooses the best action for Pacman.
    """

    def print_grid(self, grid, delimiter=''):
        """
        Clears the terminal screen and prints the current game grid.
        Useful for visualizing the game state during debugging or demonstration.
        """
        os.system("clear")  # Clear the terminal for a refreshed view.
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                print(grid[i][j], end=delimiter)
            print()

    def create_grid(self, state: GameState):
        """
        Converts the current game state into a 2D grid representation.
        Each cell in the grid represents a portion of the game board:
            - 'W' for walls.
            - '+' for food pellets.
            - '*' for capsules (power-ups).
            - 'G' for ghosts.
            - 'P' for Pacman.
        """
        w = state.getWalls().width  # Grid width.
        h = state.getWalls().height  # Grid height.

        capsules = state.getCapsules()  # List of capsule positions.
        ghosts = state.getGhostPositions()  # List of ghost positions.
        pacman_pos = state.getPacmanPosition()  # Pacman's position.

        # Initialize the grid with empty spaces.
        grid = [[' ' for x in range(w)] for y in range(h)]
        
        # Fill the grid with food and wall information.
        for i in range(h):
            for j in range(w):
                # If there is food at the cell, mark it with '+'.
                if state.hasFood(j, h - i - 1):
                    grid[i][j] = '+'
                # If there is a wall at the cell, mark it with 'W'.
                if state.hasWall(j, h - i - 1):
                    grid[i][j] = 'W'

        # Mark capsule positions on the grid with '*'.
        for caps in capsules:
            grid[h - int(caps[1]) - 1][int(caps[0])] = '*'

        # Mark ghost positions on the grid with 'G'.
        for ghost in ghosts:  
            grid[h - int(ghost[1]) - 1][int(ghost[0])] = 'G'
            
        # Mark Pacman's current position with 'P'.
        grid[h - int(pacman_pos[1]) - 1][int(pacman_pos[0])] = 'P'

        return grid

    def is_terminal(self, state: GameState):
        """
        Checks whether the current state is terminal.
        A state is terminal if Pacman has either won or lost.
        """
        return state.isLose() or state.isWin()

    def heuristic(self, state: GameState):
        """
        Evaluates the desirability of a given game state.
        Factors considered include:
            - Distance to food pellets.
            - Distance to capsules.
            - Proximity and state (scared or active) of ghosts.
            - Terminal conditions (win/loss).
        Uses a grid-based distance computation similar to Dijkstra's algorithm.
        """
        # Convert the state into a grid representation.
        grid = self.create_grid(state)
        w = state.getWalls().width
        h = state.getWalls().height
        
        capsules = state.getCapsules()  # Capsule positions.
        ghosts = state.getGhostPositions()  # Ghost positions.
        pacman_pos = state.getPacmanPosition()  # Pacman's position.
        
        # Convert Pacman's position into grid coordinates.
        x = h - int(pacman_pos[1]) - 1
        y = int(pacman_pos[0])
        
        # Initialize a priority queue for computing shortest distances.
        pq = []
        # Initialize a distance grid with "infinite" distances.
        dis = [[float("inf") for i in range(w)] for j in range(h)]
        # Initialize a visited grid to keep track of explored cells.
        vis = [[False for i in range(w)] for j in range(h)]
        
        # Start from Pacman's position with a distance of 0.
        heapq.heappush(pq, (0, (x, y)))
        # Directions for movement: up, right, down, left.
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        
        # Dijkstra-like algorithm to compute distances from Pacman's position.
        while len(pq) != 0:
            u = heapq.heappop(pq)
            x = u[1][0]
            y = u[1][1]

            # Skip the cell if it has been visited already.
            if vis[x][y]:
                continue
            
            vis[x][y] = True  # Mark the cell as visited.
            dis[x][y] = u[0]  # Record the computed distance for this cell.
                
            # Explore all four adjacent cells.
            for i in range(4):
                # Check if the neighbor cell is not a wall.
                if grid[x + dx[i]][y + dy[i]] != 'W':
                    # If a shorter path to the neighbor is found, update its distance.
                    if dis[x + dx[i]][y + dy[i]] > dis[x][y] + 1:
                        dis[x + dx[i]][y + dy[i]] = dis[x][y] + 1
                        # Only add the neighbor to the queue if it's not occupied by a ghost.
                        if grid[x + dx[i]][y + dy[i]] != 'G':
                            heapq.heappush(pq, (dis[x + dx[i]][y + dy[i]], (x + dx[i], y + dy[i])))
                    
        value_state = 0  # This will accumulate our heuristic adjustments.
        
        # Compute the "food value" by summing contributions from all food pellets.
        food_value = 0
        for i in range(h):
            for j in range(w):
                if grid[i][j] == '+':
                    # Contribution from food decreases rapidly with distance.
                    food_value += 1 / (pow(dis[i][j], 5))
                            
        # Evaluate ghost influences on the state.
        for ghostIdx in range(state.getNumAgents() - 1):
            # Determine the distance from Pacman to the ghost.
            ghost_distance = dis[h - int(ghosts[ghostIdx][1]) - 1][int(ghosts[ghostIdx][0])]
            # Check if the ghost is in a scared state.
            if state.getGhostState(agentIndex=ghostIdx+1).scaredTimer > ghost_distance:
                # Encourage pursuing scared ghosts.
                value_state += 450 / (pow(ghost_distance, 2) + 0.1)
            else:
                # Penalize when a non-scared ghost is dangerously close.
                if ghost_distance <= 3:
                    value_state -= 300 / (pow(ghost_distance, 2) + 0.1)
                # Otherwise, encourage collecting food.
                elif state.getGhostState(agentIndex=ghostIdx+1).scaredTimer <= 0:
                    value_state += food_value
        
        # Heavy penalty for a losing state.
        if state.isLose():
            value_state -= 10000
            
        # Adjust the evaluation in a winning state.
        if state.isWin():
            # If any ghost is still scared in a win state, apply a penalty.
            for ghostIdx in range(state.getNumAgents() - 1):
                if state.getGhostState(agentIndex=ghostIdx+1).scaredTimer > 0:   
                    value_state -= 10000
            # If capsules are present, factor in their distance negatively.
            if len(capsules) > 0:
                capsule_distance = dis[h - int(capsules[0][1]) - 1][int(capsules[0][0])]
                value_state -= 10000 / (pow(capsule_distance, 2) + 0.1)
                        
        # Return the sum of the game state's inherent score and our heuristic adjustments.
        return state.getScore() + value_state
               
    def min_value(self, state: GameState, depth, alpha, beta, ghostIdx=1):
        """
        Computes the minimum value for the ghost agent's move.
        Uses recursion to simulate adversarial play and incorporates alpha-beta pruning.
        
        Parameters:
            state   - current game state.
            depth   - current depth in the search tree.
            alpha   - best value found so far for the maximizer (Pacman).
            beta    - best value found so far for the minimizer (ghosts).
            ghostIdx- index of the current ghost agent.
        """
        # If the state is terminal or maximum depth is reached, return its heuristic value.
        if self.is_terminal(state) or depth == self.depth:
            return self.heuristic(state)

        value = float("inf")  # Initialize to positive infinity.
        # Evaluate each legal action for the ghost.
        for action in state.getLegalActions(agentIndex=ghostIdx):
            tmpState = state.deepCopy()  # Work on a copy of the state.
            
            if state.getNumAgents() == 2:
                # For a single ghost, switch to Pacman's turn after this move.
                value = min(value, self.max_value(tmpState.generateSuccessor(
                        agentIndex=1, action=action), depth + 1, alpha, beta))
            else:
                # For multiple ghosts, alternate between ghost agents.
                if ghostIdx == 1:
                    # Continue with the next ghost agent without increasing depth.
                    value = min(value, self.min_value(tmpState.generateSuccessor(
                            agentIndex=1, action=action), depth, alpha, beta, 2))
                else:
                    # After the last ghost's move, switch to Pacman's turn and increase depth.
                    value = min(value, self.max_value(tmpState.generateSuccessor(
                            agentIndex=2, action=action), depth + 1, alpha, beta))

            # Alpha-beta pruning: if the current value is less than or equal to alpha, stop searching.
            if value <= alpha:
                return value

            beta = min(beta, value)  # Update beta.
        return value

    def max_value(self, state: GameState, depth, alpha, beta):
        """
        Computes the maximum value for Pacman's move.
        Uses recursion to simulate adversarial play and incorporates alpha-beta pruning.
        
        Parameters:
            state - current game state.
            depth - current depth in the search tree.
            alpha - best value found so far for the maximizer (Pacman).
            beta  - best value found so far for the minimizer (ghosts).
        """
        # If the state is terminal or maximum depth is reached, return its heuristic value.
        if self.is_terminal(state) or depth == self.depth:
            return self.heuristic(state)

        value = float("-inf")  # Initialize to negative infinity.
        # Evaluate each legal action for Pacman.
        for action in state.getLegalActions():
            tmpState = state.deepCopy()  # Work on a copy of the state.
            # Update the value with the maximum value returned by min_value.
            value = max(value, self.min_value(tmpState.generateSuccessor(
                agentIndex=0, action=action), depth, alpha, beta))
            
            # Alpha-beta pruning: if the value is greater than or equal to beta, stop searching.
            if value >= beta:
                return value

            alpha = max(alpha, value)  # Update alpha.
        return value

    def getAction(self, gameState: GameState):
        """
        Determines and returns the best action for Pacman from the current game state.
        This is the main function that is called to decide Pacman's next move.
        
        It uses the minimax algorithm with alpha-beta pruning to explore possible moves,
        and selects the one that maximizes Pacman's expected outcome.
        """
        # Display the current game grid (useful for debugging and visualization).
        self.print_grid(self.create_grid(gameState))
        
        # Retrieve all legal actions for Pacman.
        legal_actions = gameState.getLegalPacmanActions()
        
        best_action = []  # Will hold the best action(s) found.
        max_value_found = float("-inf")  # Initialize the maximum value to negative infinity.
        
        # Evaluate each legal action using minimax.
        for action in legal_actions:
            pState = gameState.deepCopy()  # Copy the current game state.
            # Evaluate the state resulting from Pacman taking this action.
            value = self.max_value(pState.generatePacmanSuccessor(
                action), 0, float("-inf"), float("inf"))

            # Penalize the 'Stop' action to discourage inaction.
            if action == "Stop":
                value -= 1

            # Update the list of best actions based on the computed value.
            if value > max_value_found:
                best_action = [action]
                max_value_found = value
            elif value == max_value_found:
                best_action.append(action)
                
        # Randomly select one of the best actions (to add variety if there are ties).
        idx = random.randint(0, len(best_action) - 1)
        return best_action[idx]
