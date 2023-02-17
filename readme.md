# Author

Doug de Jesus

drd8913@nyu.edu


# Overview

This is a demonstration of the A* pathfinding algorithm. It simulates an agent
(e.g. a robot) navigating to a goal through an obstacle-filled environment.
Implementations for both single- and multi-agent systems are provided.


The environment is represented by a grid. Agents can move vertically or 
horizontally, but not diagonally. Obstacles that can be crossed with a movement 
penalty are shown as ® ('R' for "rock"), while impassable barriers are shown as 
a black square. An animation of the search process shows the set of squares that
have been reached in dark blue, while the frontier of nodes to expand is in 
light blue. The final shortest path found is drawn with arrows.


For the multi-agent version, only one agent moves at a time. Agents are not 
allowed to move into an already occupied space. Each agent has its own specific
goal location. The overall "path" is a sequence of states containing the 
positions of all the agents. This is an inefficient implementation, as the 
complexity grows factorially with the number of agents.


Note that the multi-agent version thus takes much longer to run than the 
single-agent version, and may not complete in a reasonable amount of time for
more than three agents or for large grid sizes.


# How to run

- Use python version 3.9 or higher
- To install the necessary libraries: pip install -r requirements.txt
- To run the single-agent simulation: python3 single_agent.py
- To run the multi-agent simulation: python3 multi_agent.py


# Configuration

Modify the provided config files, single_agent.ini or multi_agent.ini, to change
the parameters of the simulation. 
You can specify the size of the environment, the starting and goal positions for
the agent(s), and the positions of the obstacles and barriers.
Obstacles can be traversed by the agents, but have an increased movement cost
that can also be configured. The cost for a movement without obstacles is 1.
Barriers are spaces that cannot be entered at all.


For the single-agent version, you can also specify which heuristic to use for 
the estimated cost, either the L1-norm (Manhattan distance) or the L-infinity
norm (chessboard distance). FOr the multi-agent version, the L1 norm is used.
