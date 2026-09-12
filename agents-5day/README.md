# 5-Day AI Agents Intensive

The eleven notebooks of the [5-Day AI Agents Intensive](https://www.kaggle.com/learn-guide/5-day-agents),
pulled from `kaggle5daysofai`. `sources/` holds each one exactly as published,
with its `kernel-metadata.json`.

    day-0  troubleshooting-and-faqs
    day-1a from-prompt-to-action          day-1b agent-architectures
    day-2a agent-tools                    day-2b agent-tools-best-practices
    day-3a agent-sessions                 day-3b agent-memory
    day-4a agent-observability            day-4b agent-evaluation
    day-5a agent2agent-communication      day-5b agent-deployment

## These are codelabs, not exercises

No `learntools`, no `____`, no `check()` anywhere in the 167 code cells. There
is nothing to answer and nothing that can be graded -- unlike the ten courses in
`../learn/`, which are exercise sets with a grader behind them. The value here
is reading and running the code.

## What running them needs

Every notebook does

    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")

so a Kaggle Secret named exactly `GOOGLE_API_KEY` has to exist and be attached
to the notebook. There is no API for that -- the Kaggle CLI has no secrets
command -- so it is done in the browser, per notebook, under Add-ons > Secrets.
Internet must also be on.

Imports across the set: `google` (genai and ADK), `vertexai`, `mcp`,
`requests`, `kaggle_secrets`. No pip installs; the packages are preinstalled in
Kaggle's image, which is the main reason to run these on Kaggle rather than
locally.
