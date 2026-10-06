# LLM seeds: how they work and why none has run on real data

## How the proposer works

`src/alpha_gp_lab/llm_seed.py`. A prompt containing a short research
brief and the grammar (never data) asks for N expressions. Responses are cached in a JSON
replay file keyed by the SHA-256 of the exact prompt. Every proposed line is parsed; invalid
lines are logged, rejected and counted (a line starting `- ` is rejected too, because a list
bullet and a minus sign cannot be told apart). The demo and the tests read only the replay file.
([Diagram: replay or one live request, and every refusal](DIAGRAMS.md#4-the-llm-seed-path-replay-or-one-live-request).)
`seeds --live` makes one request to a free hosted open-weight model on OpenRouter, pinned as
`qwen/qwen3.8-27b:free` (weights: Hugging Face `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, Apache-2.0; created on Hugging Face 2026-08-05,
listed by OpenRouter 2026-08-14; sources: [Hugging Face model API](https://huggingface.co/api/models/Qwen/Qwen3.8-27B),
[OpenRouter model list](https://openrouter.ai/api/v1/models)). The seed proposer uses no
Anthropic or OpenAI model, by Oscar's decision. The request is temperature 0 with
reasoning at effort low (the smallest the listing offers; the dates, revision and effort list are in the committed
[model evidence](model-evidence/README.md); the free-tier limits are from OpenRouter's
[limits page](https://openrouter.ai/docs/api-reference/limits), read 2026-10-05) and excluded from the response, and
`max_tokens` 8192, which the hidden reasoning also counts against.
Any non-200 status (429 is the rate limit), error field, unfinished answer, empty content,
oversized body or response from another model is refused and nothing is saved; a refusal is not
retried, since the free tier allows 20 requests a minute and 50 a day without purchased credits.
An accepted answer is saved labelled **REAL LLM OUTPUT** with the response's model, provider,
response id and date. The key is read from `OPENROUTER_API_KEY` and never saved. A run with
`use_seeds: false` reads no LLM output at all. The one live attempt so far (through `claude -p`,
before the switch) failed before reaching a model (see below). The only shipped replay entry is
labelled **HAND-WRITTEN FIXTURE (not real LLM output)**. It was written while building this repo,
by someone who knew how the synthetic data is generated, and it deliberately contains one
duplicate and three invalid lines. It is used by the SYNTHETIC demo only.

## Status on real data: not run

**Live LLM seeds pending: CLI not signed in on 2026-10-03.** The one live call made (of 3 allowed),
`PYTHONPATH=src python3.11 -m alpha_gp_lab seeds --live --config fixtures/binance_daily_config.json`,
exited 1 before reaching a model (0 input and 0 output tokens; `results/llm_live_attempt.json`).
No retry was made. The seeded config (`fixtures/binance_daily_seeded_config.json`, pre-registered
in `4f9c276`) has not been run, so there is no seeded-vs-ablation comparison on real data; the main
run is the ablation without seeds. The live path now uses the pinned open-weight model
`qwen/qwen3.8-27b:free` on OpenRouter, by Oscar's decision on 2026-10-05.

*Note added 2026-10-05, before any live call reached a model:* the `4f9c276` pre-registration
described the live seeds as coming from `claude -p`. The proposer model has changed to the pinned
open-weight model above; the seeded config's bytes (brief, n, splits, GP settings) are unchanged.
