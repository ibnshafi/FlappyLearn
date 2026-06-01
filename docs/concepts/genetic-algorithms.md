# Genetic Algorithms

A genetic algorithm is an optimization method inspired by biological evolution.

It keeps a population of candidate solutions. Each candidate is evaluated, better candidates are selected, and new candidates are created through variation.

## The Loop

1. Create an initial population.
2. Evaluate every candidate.
3. Select better candidates as parents.
4. Produce children through mutation and crossover.
5. Repeat for many generations.

## In FlappyLearn

- A candidate is an `AdaptiveCircuitGenome`.
- Evaluation means playing seeded Flappy Bird episodes.
- Selection uses tournament competition.
- Mutation changes weights, activations, thresholds, topology, and evolution metadata.
- Crossover combines two parent genomes.

## Why Genetic Algorithms Are Useful For Learning

They can optimize systems that are hard to train with gradients. They also make learning visible: you can inspect populations, generations, mutations, and fitness changes directly.
