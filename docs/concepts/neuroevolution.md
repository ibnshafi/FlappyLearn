# What Is Neuroevolution?

Neuroevolution means training neural networks with evolutionary search instead of gradient descent.

In a typical neural network training setup, the system measures error and nudges weights in the direction that reduces that error. In neuroevolution, the system keeps a population of candidate networks, tests each one, keeps the better ones, and creates a new generation through mutation and crossover.

## Flappy Bird Example

Imagine 96 tiny brains trying to play Flappy Bird. Most fail quickly. A few survive longer because their flap timing is less bad. Those better brains become parents. Their children inherit similar behavior with small changes. After many generations, the population discovers timing patterns that pass more pipes.

## Why It Works Well Here

Flappy Bird is a good match for neuroevolution because:

- The action is simple: flap or do nothing.
- The observation space is small.
- Success is easy to measure.
- Bad policies fail quickly, so evaluation is cheap.
- Timing and memory matter more than huge models.

## What Evolves In FlappyLearn

FlappyLearn evolves:

- Weights and biases.
- Hidden-unit activations.
- Output threshold.
- Recurrent memory behavior.
- Network topology.
- Mutation behavior.
- Exploration noise.

The result is not a hand-written strategy. It is a policy discovered through repeated trial, selection, and variation.
