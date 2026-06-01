# How Neural Networks Work

A neural network is a function made from connected units. Each unit receives numbers, combines them with weights, applies a nonlinear activation, and passes the result forward.

In FlappyLearn, the network receives game observations such as bird height, velocity, pipe distance, and gap position. It outputs one decision: flap or do nothing.

## Weights

Weights control how strongly one value affects another. A positive weight can make a signal more likely to increase the output. A negative weight can suppress it.

## Biases

A bias shifts a unit before activation. Biases help the network represent behavior even when inputs are near zero.

## Activations

Activations add nonlinearity. Without them, a network would mostly behave like a single linear equation. FlappyLearn uses simple activation functions such as `tanh`, `relu`, and `elu`.

## Recurrence And Memory

Flappy Bird is about timing. A single frame does not tell the whole story. FlappyLearn uses recurrent memory so the policy can carry information from previous steps into the next decision.

That memory helps the agent learn rhythms such as "wait a little longer before flapping" or "recover after a recent flap."

## Output Threshold

The network produces a score-like output called a logit. If the logit crosses the genome's threshold, the policy flaps. Otherwise, it waits.
