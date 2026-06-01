# Mutation

Mutation creates small random changes in a genome.

In FlappyLearn, mutation can affect:

- Input, recurrent, direct, and output weights.
- Hidden biases and output bias.
- Activation functions.
- Output threshold.
- Exploration noise.
- Mutation rates themselves.
- Hidden topology by adding or removing units.

## Why Mutation Matters

Mutation is the engine of exploration. Without it, the population would only recombine what already exists. With too much mutation, useful behavior is destroyed faster than selection can preserve it.

FlappyLearn uses self-adaptive mutation metadata so different genomes can carry different search tendencies.
