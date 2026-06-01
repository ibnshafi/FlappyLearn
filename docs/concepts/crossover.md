# Crossover

Crossover combines information from two parent genomes into one child genome.

In FlappyLearn, crossover can mix compatible parts of parent circuits, such as weights and unit-level choices. The child then mutates, giving it a chance to preserve useful parent behavior while exploring nearby alternatives.

## Why Crossover Helps

One parent may have useful flap timing. Another may survive better near pipe openings. Crossover gives evolution a way to combine partial solutions.

## Why It Is Not Enough Alone

Crossover rearranges existing information. Mutation is still needed to discover new weights, thresholds, activations, and structures.
