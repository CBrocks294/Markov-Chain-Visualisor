# Markov Chain and Monte Carlo Visualisor
This is a small project to visualise montecarlo walks and random on a graph. You can input a graph and see how many matrix operations it takes to reach a stable solution. Aswell as trialing multiagent and time averaged Monte Carlo simulations.

## How to use
To operate this simulator you first set the number of nodes you have as an integer on `line 13`. Then in the following lines you set the edges. 
```py 
TransMatrix[A][B] = Fraction(C, D)
```
Then set your variables, B is the start node and A is the end node. Nodes are indexed from 0 to number of nodes-1. You then set the probability of a agent traversing that node. Note fraction can take a single number which will default as the numerator. 

For a valid markov chain the probability of leaving a node must sum to one. If you wish for the probability of traveling from node 4 to node 0 to be 1/2 you would write.
```py 
TransMatrix[0][4] = Fraction(1, 2)
```

## Dependancies
- numpy
- pygame
- networkx
- UIpygame