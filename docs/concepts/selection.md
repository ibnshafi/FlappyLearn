# Selection

Selection decides which genomes get more influence over the next generation.

FlappyLearn uses tournament selection. The trainer samples a small group of genomes and chooses the best one in that group as a parent. This repeats until enough parents have been chosen.

## Why Tournament Selection?

Tournament selection is simple, fast, and easy to tune.

- Larger tournaments increase pressure toward the current best genomes.
- Smaller tournaments preserve more diversity.

FlappyLearn also uses elitism, which copies the strongest genomes directly into the next generation. Elitism protects hard-won progress while mutation and crossover continue exploring.
