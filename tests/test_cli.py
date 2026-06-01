from flappylearn.cli import build_parser


def test_cli_parser_builds_all_primary_commands():
    parser = build_parser()
    commands = [
        ["train", "--config", "configs/smoke.json"],
        ["eval", "--checkpoint", "best.json", "--episodes", "2"],
        ["benchmark", "--checkpoint", "best.json", "--episodes", "2"],
        ["replay", "--checkpoint", "best.json", "--output", "replay.html"],
        ["visualize", "--metrics", "metrics.jsonl", "--output", "metrics.html"],
        ["profile", "--profile-generations", "1"],
    ]
    for command in commands:
        args = parser.parse_args(command)
        assert callable(args.func)
