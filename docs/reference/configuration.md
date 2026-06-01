# Configuration Reference

FlappyLearn configs are JSON files with three top-level sections: `run`, `environment`, and `trainer`.

## Run

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | string | `flappylearn` | Output folder under `runs/`. |
| `output_dir` | string | `runs` | Root directory for generated artifacts. |
| `seed` | integer | `42` | Base random seed for reproducibility. |

## Environment

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `width` | integer | `288` | Game width in pixels. |
| `height` | integer | `512` | Game height in pixels. |
| `bird_x` | number | `64` | Horizontal bird position. |
| `bird_radius` | number | `12` | Collision radius. |
| `gravity` | number | `0.45` | Downward velocity added each step. |
| `flap_velocity` | number | `-7.5` | Upward velocity applied on flap. |
| `max_velocity` | number | `10.0` | Velocity clamp. |
| `pipe_width` | number | `52` | Width of each pipe. |
| `pipe_gap` | number | `110` | Vertical opening size. |
| `pipe_spacing` | number | `155` | Horizontal distance between pipes. |
| `scroll_speed` | number | `2.5` | Pipe movement speed. |
| `spawn_x` | number | `320` | New pipe spawn location. |
| `alive_reward` | number | `0.01` | Per-step survival reward. |
| `pass_reward` | number | `1.0` | Reward for passing a pipe. |
| `death_penalty` | number | `-1.0` | Terminal collision penalty. |

## Trainer

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `population_size` | integer | `96` | Genomes per generation. |
| `generations` | integer | `120` | Maximum generations. |
| `hidden_units` | integer | `10` | Initial hidden units per genome. |
| `max_hidden_units` | integer | `64` | Structural growth limit. |
| `elites` | integer | `8` | Top genomes copied unchanged. |
| `tournament_size` | integer | `5` | Parent selection pressure. |
| `episodes_per_genome` | integer | `4` | Training episodes per genome. |
| `evaluation_episodes` | integer | `12` | Best-genome validation episodes. |
| `max_steps` | integer | `12000` | Per-episode step cap. |
| `novelty_weight` | number | `0.15` | Behavioral novelty bonus. |
| `complexity_penalty` | number | `0.0008` | Penalty for larger circuits. |
| `checkpoint_interval` | integer | `5` | Generations between checkpoints. |
| `target_score` | integer | `200` | Early stopping threshold. |
| `curriculum` | boolean | `true` | Enable easier-to-harder training. |
| `replay_best` | boolean | `true` | Export replay artifacts for best genome. |

## CLI Overrides

```bash
python -m flappylearn train \
  --config configs/default.json \
  --seed 7 \
  --generations 50 \
  --population 64 \
  --max-steps 8000 \
  --run-name experiment-7
```
