from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_metrics_html(metrics_path: str | Path, output_path: str | Path) -> None:
    records = read_metrics(metrics_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    best = [float(record.get("eval_score_mean", record.get("best_score_mean", 0.0))) for record in records]
    mean = [float(record.get("population_score_mean", 0.0)) for record in records]
    svg = _line_chart([("Best eval score", best, "#2563eb"), ("Population mean", mean, "#16a34a")])
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FlappyLearn Metrics</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    main {{ max-width: 980px; margin: auto; }}
    .panel {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; text-align: right; padding: 8px; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
<main>
  <h1>FlappyLearn Metrics</h1>
  <div class="panel">{svg}</div>
  {_metrics_table(records[-20:])}
</main>
</body>
</html>
"""
    output.write_text(body, encoding="utf-8")


def write_replay_html(replay_path: str | Path, output_path: str | Path) -> None:
    replay_file = Path(replay_path)
    replay = json.loads(replay_file.read_text(encoding="utf-8"))
    frames = replay.get("frames", [])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(frames)
    title = f"Score {replay.get('score', 0)} in {replay.get('steps', 0)} steps"
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FlappyLearn Replay</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #0f172a;
      color: white;
      font-family: system-ui, sans-serif;
    }}
    main {{ display: grid; gap: 12px; justify-items: center; }}
    canvas {{
      background: linear-gradient(#7dd3fc, #e0f2fe 62%, #86efac 62%);
      border: 1px solid #334155;
      border-radius: 8px;
    }}
    .meta {{ color: #dbeafe; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <canvas id="game" width="288" height="512"></canvas>
  <div class="meta" id="meta"></div>
</main>
<script>
const frames = {data};
const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const meta = document.getElementById("meta");
let i = 0;
function draw(frame) {{
  const cfg = frame.config || {{ width: 288, height: 512, pipe_width: 52, pipe_gap: 110 }};
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#7dd3fc";
  ctx.fillRect(0, 0, canvas.width, canvas.height * 0.62);
  ctx.fillStyle = "#bbf7d0";
  ctx.fillRect(0, canvas.height * 0.62, canvas.width, canvas.height * 0.38);
  ctx.fillStyle = "#16a34a";
  for (const pipe of frame.pipes || []) {{
    const gapTop = pipe.gap_y - cfg.pipe_gap / 2;
    const gapBottom = pipe.gap_y + cfg.pipe_gap / 2;
    ctx.fillRect(pipe.x, 0, cfg.pipe_width, gapTop);
    ctx.fillRect(pipe.x, gapBottom, cfg.pipe_width, canvas.height - gapBottom);
  }}
  const bird = frame.bird || {{ x: 64, y: 256, radius: 12 }};
  ctx.fillStyle = frame.action ? "#f97316" : "#facc15";
  ctx.beginPath();
  ctx.arc(bird.x, bird.y, bird.radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#111827";
  ctx.font = "16px system-ui";
  ctx.fillText(`Score: ${{frame.score}}`, 12, 24);
  meta.textContent = `step ${{frame.step}} / ${{frames.length - 1}}`;
}}
function tick() {{
  if (frames.length > 0) {{
    draw(frames[i]);
    i = (i + 1) % frames.length;
  }}
  setTimeout(tick, 33);
}}
tick();
</script>
</body>
</html>
"""
    output.write_text(body, encoding="utf-8")


def _line_chart(series: list[tuple[str, list[float], str]]) -> str:
    width = 920
    height = 360
    pad = 44
    values = [value for _, vals, _ in series for value in vals]
    if not values:
        return "<p>No metrics recorded yet.</p>"
    y_min = min(0.0, min(values))
    y_max = max(1.0, max(values))
    x_count = max(1, max(len(vals) for _, vals, _ in series) - 1)

    def point(index: int, value: float) -> tuple[float, float]:
        x = pad + (width - pad * 2) * (index / x_count)
        y = height - pad - (height - pad * 2) * ((value - y_min) / max(1e-9, y_max - y_min))
        return x, y

    paths = []
    for label, vals, color in series:
        if not vals:
            continue
        d = []
        for index, value in enumerate(vals):
            x, y = point(index, value)
            command = "M" if index == 0 else "L"
            d.append(f"{command}{x:.2f},{y:.2f}")
        paths.append(
            f'<path d="{" ".join(d)}" fill="none" stroke="{color}" stroke-width="3">'
            f"<title>{html.escape(label)}</title></path>"
        )
    legend = " ".join(
        f'<span style="color:{color}; margin-right:18px;">{html.escape(label)}</span>' for label, _, color in series
    )
    return f"""
<div>{legend}</div>
<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Learning curves">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#fff" />
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#9ca3af" />
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#9ca3af" />
  <text x="{pad}" y="{pad - 12}" fill="#374151">{y_max:.2f}</text>
  <text x="{pad}" y="{height - pad + 28}" fill="#374151">generation</text>
  {"".join(paths)}
</svg>
"""


def _metrics_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "<p>No records.</p>"
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{int(record.get('generation', 0))}</td>"
            f"<td>{float(record.get('eval_score_mean', 0.0)):.3f}</td>"
            f"<td>{float(record.get('population_score_mean', 0.0)):.3f}</td>"
            f"<td>{float(record.get('best_score_max', 0.0)):.3f}</td>"
            f"<td>{float(record.get('novelty_weight', 0.0)):.3f}</td>"
            f"<td>{float(record.get('curriculum_phase', 0.0)):.3f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Generation</th><th>Eval mean</th><th>Population mean</th>"
        "<th>Best max</th><th>Novelty</th><th>Curriculum</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
