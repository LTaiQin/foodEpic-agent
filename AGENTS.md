# Project Operating Rules

This repository is bound to:

```text
https://github.com/LTaiQin/foodEpic-agent.git
```

Local data directories are intentionally not tracked:

```text
data/
data-test/
annotations/
outputs/
```

Workflow for future code changes:

1. Make the requested code or documentation change.
2. Run the relevant verification command.
3. If verification passes, commit the change automatically.
4. Push only when GitHub credentials are available or the user explicitly asks.

Recommended helper:

```bash
scripts/verify_and_commit.sh "commit message" python -m compileall food_agent scripts
```

If the verification command fails, do not commit until the failure is fixed or explicitly accepted.

