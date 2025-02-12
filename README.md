# Pacman AI Agent in a Multi Agent Environment

This repository contains an implementation of a Pacman AI agent that employs a minimax search algorithm enhanced with alpha-beta pruning. The project builds upon the classic Berkeley Pacman AI Projects and demonstrates how adversarial search techniques can be applied in a multi-agent environment. 


## Table of Contents
- [Overview](#overview)
- [Algorithm In-Depth](#algorithm-in-depth)
  - [Minimax Search](#minimax-search)
  - [Alpha-Beta Pruning](#alpha-beta-pruning)
- [Heuristic Function Details](#heuristic-function-details)
- [Implementation Details](#implementation-details)
- [Performance Demonstration](#performance-demonstration)
- [Group Project Information](#group-project-information)
- [References](#references)


## Overview

The agent is designed to navigate a Pacman maze while balancing the dual objectives of collecting food and avoiding ghosts. In this adversarial setup:
- **Pacman** acts as the maximizing agent, selecting moves that increase the game score.
- **Ghosts** serve as minimizing agents, choosing actions intended to lower Pacman’s score.

This approach provides a clear demonstration of adversarial search where multiple agents with opposing goals interact, a scenario often encountered in game-playing AI.


## Algorithm In-Depth

### Minimax Search

The minimax algorithm forms the backbone of this AI agent. It operates by recursively exploring the game tree to a fixed depth:
- **Maximizing Nodes (Pacman’s Turn):**  
  At these nodes, Pacman evaluates all possible actions and selects the one that maximizes the expected game score.
  
- **Minimizing Nodes (Ghosts’ Turn):**  
  At these nodes, ghost agents evaluate their moves with the objective of minimizing Pacman’s score. In a multi-agent context, the algorithm alternates between these roles as it simulates the interplay between Pacman and the ghosts.

This structured exploration allows the agent to plan several moves ahead, anticipating the adversary's responses and planning counter-strategies accordingly.

### Alpha-Beta Pruning

Alpha-beta pruning is incorporated to optimize the minimax search:
- **Alpha (α):**  
  Represents the best (highest) score that the maximizer (Pacman) can guarantee at that point in the tree.
  
- **Beta (β):**  
  Represents the best (lowest) score that the minimizer (ghosts) can ensure.

During the recursive search, if the algorithm finds that the current branch cannot yield a better outcome than the already discovered values (i.e., if the current value is worse than α for the maximizer or better than β for the minimizer), that branch is pruned. This significantly reduces the number of nodes that need to be evaluated, thereby enhancing efficiency without sacrificing optimality.


## Heuristic Function Details

A crucial component of the agent's performance is its heuristic evaluation function, which estimates the desirability of any given game state. Here’s how the heuristic is constructed:

1. **Grid Representation:**  
   The game state is converted into a grid that marks walls, food pellets (denoted by '+'), capsules (power-ups, denoted by '*'), ghost positions ('G'), and Pacman's current location ('P'). This spatial mapping facilitates distance calculations between entities.

   ![Grid Representation of a Game](./assets/pacman-cli-output.gif)

2. **Distance Computation:**  
   Using a priority queue (akin to Dijkstra’s algorithm), the heuristic computes the shortest-path distances from Pacman’s position to:
   - **Food Pellets:**  
     Each food pellet contributes to the evaluation score inversely proportional to the fifth power of its distance (i.e., `1 / (distance^5)`). This exponentiation prioritizes nearby food while diminishing the impact of distant pellets.
     
   - **Ghosts:**  
     The heuristic differentiates between ghosts based on their state:
     - **Scared Ghosts:**  
       If a ghost is scared and its distance is within a favorable range, a high positive value is added (using a formula like `450 / (distance^2 + 0.1)`) to incentivize chasing them.
     - **Active Ghosts:**  
       Conversely, if a ghost is not scared and is too close (e.g., within three steps), a significant negative penalty is applied (`-300 / (distance^2 + 0.1)`) to discourage moves that might lead to a collision.
     
   - **Capsules:**  
     Capsules are evaluated similarly, with their desirability factored in to either prolong the power-up effect or to avoid unnecessary risks if no capsules are present.

3. **Terminal State Adjustments:**  
   - **Win/Loss Conditions:**  
     The function applies very high penalties (or rewards) when reaching game-losing or game-winning states. For instance, a loss subtracts a large constant value, while winning (with certain conditions regarding capsules and ghost states) can add to the overall evaluation.
     
4. **Overall Evaluation:**  
   Finally, the heuristic combines these calculated values with the game’s inherent score to produce a final evaluation. This composite score drives the minimax decision-making, guiding the agent towards moves that are both immediately beneficial and strategically sound.


## Implementation Details

The project is implemented in Python and is structured around key classes and functions:

- **`MultiAgentSearchAgent`:**  
  A base class that sets up the evaluation function and search parameters such as depth and time limit.

- **`AIAgent`:**  
  Extends the base class to include:
  - **Grid Creation:** Transforms the game state into a navigable grid.
  - **Heuristic Function:** Implements the detailed evaluation method described above.
  - **Minimax Search Functions (`max_value` and `min_value`):**  
    These functions recursively explore game states. The `max_value` function is used when it’s Pacman’s turn, while `min_value` handles ghost moves. Both functions integrate alpha-beta pruning to eliminate suboptimal branches.

- **Action Selection:**  
  In the `getAction` method, Pacman iterates over all legal moves, computes the minimax value for each, and selects the action that leads to the best evaluated state. In cases where multiple actions yield the same score, one is chosen randomly to introduce variability.


## Performance Demonstration

The algorithm’s performance has been showcased through animated demonstrations under different ghost behaviors:

- **Directional Ghosts (2 ghosts, depth 3):**  
  In this scenario, ghosts follow predetermined directional patterns, providing a more strategic adversary.  
  ![Directional Ghosts](assets/pacman-k2-d3-dighost.gif)

- **Random Ghosts (2 ghosts, depth 3):**  
  Here, ghosts select moves at random, creating a dynamic and less predictable challenge for Pacman.  
  ![Random Ghosts](assets/pacman-k2-d3-random-ghost.gif)

These demonstrations highlight the robustness of the algorithm and the effectiveness of the heuristic in diverse game conditions.


## Contributors

This project was developed as part of a group assignment for Dr. Karshenas's Fundamentals of Artificial Intelligence course at the University of Isfahan.


**Group Members:**  
- Zahra Masoumi (Github: [@asAlwaysZahra](https://github.com/asAlwaysZahra))
- Matin Azami (Github: [@InFluX-M](https://github.com/InFluX-M))
- Amirali Lotfi (Github: [@liAmirali](https://github.com/liAmirali/))



## References

- **UC Berkeley Pacman Projects:**  
  The foundation of this project is based on the Berkeley AI course materials. More details can be found on the [Berkeley AI website](http://ai.berkeley.edu).

- **Artificial Intelligence: A Modern Approach (4th Edition):**  
  The minimax and alpha-beta pruning algorithms implemented here are thoroughly explained in this authoritative text by Peter Norvig and Stuart Russell.

