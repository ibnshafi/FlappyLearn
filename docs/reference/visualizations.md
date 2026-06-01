# Visualization Guide

FlappyLearn generates browser-friendly artifacts without requiring a frontend build step.

## Metrics HTML

Command:

```bash
python -m flappylearn visualize --metrics runs/smoke/latest/metrics.jsonl --output runs/smoke/latest/metrics.html
```

The metrics page shows:

- Best evaluation score.
- Population mean score.
- Recent generation table.
- Novelty weight and curriculum phase.

Use it to compare whether training is improving, stagnating, or becoming unstable.

## Replay HTML

Command:

```bash
python -m flappylearn replay --checkpoint runs/smoke/latest/best.json --output runs/smoke/latest/best_replay.html
```

The replay page shows the saved agent playing one seeded episode on a canvas. It is useful for debugging behavior that a score alone cannot explain.

## Benchmark JSON

Command:

```bash
python -m flappylearn benchmark --checkpoint runs/smoke/latest/best.json --episodes 100 --output runs/smoke/latest/benchmark.json
```

The benchmark compares the learned agent against random and no-op baselines on the same seeds.

## Shareable Demo Recommendation

For launch, record a 12-second GIF:

1. First 3 seconds: metrics curve rising.
2. Next 6 seconds: best replay passing multiple pipes.
3. Final 3 seconds: network diagram or checkpoint stats.

Keep the GIF under 8 MB so it loads quickly on GitHub.
