# Security Policy

FlappyLearn is an educational machine-learning project. The current package is local-first, has no network service, and does not intentionally collect user data. Security still matters because contributors may run training commands, open generated HTML artifacts, and install development dependencies.

## Supported Versions

| Version | Supported |
| --- | --- |
| `0.1.x` | Yes |

Only the latest minor release receives security fixes before the project reaches `1.0.0`.

## Reporting A Vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub private vulnerability reporting when it is enabled for the repository. If private reporting is unavailable, contact the maintainer team through the repository owner profile and include:

- A description of the issue.
- Steps to reproduce it.
- Affected files, commands, or generated artifacts.
- Expected impact.
- Any suggested fix.

Maintainers will acknowledge valid reports within seven days when the project is actively maintained and will coordinate a fix, release, and disclosure note.

## Security Boundaries

In scope:

- Unsafe handling of local files or paths.
- Generated HTML that could execute untrusted content.
- Dependency vulnerabilities that affect normal install or development workflows.
- Reproducible denial-of-service issues from malformed configs or checkpoints.

Out of scope:

- Malicious code intentionally added by a local user to their own checkout.
- Attacks requiring arbitrary local filesystem write access before FlappyLearn runs.
- Game strategy exploits that only improve score without affecting user security.

## Current Security Practices

- Core runtime dependency surface is intentionally small.
- Generated replay and metrics HTML escape user-visible text.
- Config loading rejects unknown dataclass keys.
- CI runs tests and linting on every pull request.
- Dependabot monitors GitHub Actions and Python package dependencies.

## Maintainer Checklist

When handling a security fix:

1. Reproduce privately.
2. Add a regression test when feasible.
3. Patch the smallest affected surface.
4. Release a patch version.
5. Credit the reporter unless they prefer not to be named.
6. Publish a concise advisory after users have an upgrade path.
