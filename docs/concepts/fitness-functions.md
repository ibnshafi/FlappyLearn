# Fitness Functions

A fitness function tells evolution what "better" means.

In FlappyLearn, a genome's fitness is based on more than a single score. The trainer rewards policies that score well, survive longer, behave robustly across seeds, and explore useful new behavior.

## Ingredients

- **Mean score:** average pipes passed across training episodes.
- **Minimum score:** rewards policies that are not lucky only once.
- **Mean reward:** includes survival reward, pass reward, and death penalty.
- **Mean steps:** gives partial credit for staying alive.
- **Novelty:** encourages behavior that is different from recent policies.
- **Complexity penalty:** gently prefers smaller circuits when performance is similar.

## Why Not Just Score?

Pure score can be sparse. Early in training, most agents score zero. Survival time and reward shaping provide a smoother signal so evolution can distinguish "failed instantly" from "almost reached the next pipe."

## Robustness

Using multiple seeds matters because pipe sequences vary. A policy that works on one seed but fails everywhere else is brittle. FlappyLearn gives weight to minimum and average performance so robust policies are favored.
