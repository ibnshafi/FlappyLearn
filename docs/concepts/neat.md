# What Is NEAT?

NEAT stands for NeuroEvolution of Augmenting Topologies. It is a famous approach for evolving neural networks that can grow more complex over time.

Traditional evolutionary neural networks often start with a fixed architecture. NEAT starts small and lets useful structure emerge. Networks can gain new connections and nodes, and the algorithm keeps track of structural history so crossover can combine parents sensibly.

## Core NEAT Ideas

- **Start simple:** begin with small networks instead of overbuilding.
- **Augment topology:** add nodes and connections through mutation.
- **Protect innovation:** group similar genomes so new structures have time to prove themselves.
- **Crossover structure:** combine compatible parts of parent networks.

## Is FlappyLearn A Strict NEAT Implementation?

No. FlappyLearn is NEAT-inspired, not strict NEAT.

FlappyLearn evolves topology and recurrent circuits, but it uses a compact custom genome representation instead of full innovation-number tracking and speciation. That tradeoff keeps the code easier to read for learners while preserving the important idea: the model can change its structure as it evolves.

## Why Mention NEAT?

NEAT is one of the best-known examples of neuroevolution. Understanding it helps readers place FlappyLearn in the broader family of algorithms that evolve both behavior and structure.
