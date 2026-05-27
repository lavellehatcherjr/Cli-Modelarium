<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cli-modelarium-wordmark-dark.svg">
  <img alt="cli modelarium" src="docs/assets/cli-modelarium-wordmark-light.svg" width="420">
</picture>

Read this in other languages: [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [한국어](README.ko.md) | [中文](README.zh.md) | [Deutsch](README.de.md) | [Português](README.pt.md) | [Italiano](README.it.md)

> Compare LLM outputs side-by-side from your terminal - 8 cloud providers + local models, with parallel streaming, batch evaluation, LLM-as-judge scoring, hallucination detection, and CI/CD-ready assertions.

[![PyPI](https://img.shields.io/pypi/v/cli-modelarium)](https://pypi.org/project/cli-modelarium/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Platforms](https://img.shields.io/badge/platforms-Mac%20%7C%20Windows%20%7C%20Linux-lightgrey)](#)

## What it does

**Cli Modelarium** is a polished command-line tool for comparing LLM outputs across providers, models, system prompts, and temperatures - with live parallel streaming, batch evaluation, deterministic testing, and quality scoring built in.

Useful for evaluating which model fits your specific task, running prompt regression tests in CI/CD, comparing local models against cloud APIs, or building evaluation datasets - all from a single terminal command.

## System requirements

- Python 3.11 or higher (Python 3.10 users: install `cli-modelarium==0.1.1`)
- ~150 MB disk space (including scipy and numpy)
- macOS (Apple Silicon and Intel), Windows 10+ (x64 and ARM), Linux (x64 and ARM)
- Internet access for the first install (PyPI wheel download)

## Quick start

```bash
pip install cli-modelarium

# Configure API keys (saves securely to your OS keychain)
cli-modelarium configure

# Run your first comparison
cli-modelarium "Explain quantum computing in one sentence" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro \
  --temperatures 0,0.7
```

That's it. You'll see all three models stream their responses live in parallel, with latency, token counts, and cost displayed in a clean comparison table.

## Features

### 🤖 Providers (8 cloud + unlimited local)

- **Cloud providers:** OpenAI, Anthropic, Google (Gemini), xAI (Grok), DeepSeek, Mistral, Groq, OpenRouter
- **Local models:** Ollama, LM Studio, vLLM, llama.cpp - any OpenAI-compatible local server
- Mix-and-match local and cloud models in the same comparison
- Configurable model selection per call (no hardcoded lists)

### ⚡ Parallel streaming

- Live token-by-token display across all models simultaneously
- Time-to-First-Token (TTFT) tracking per model
- See which model finishes first, watch outputs diverge in real time
- Streams from all 8 providers (SSE under the hood)

### 📊 Multiple comparison modes

- **Single prompt vs. multiple models** - quick "which is best?" comparisons
- **Single prompt vs. multiple temperatures** - see how randomness affects output
- **Multiple system prompts vs. one user prompt** - A/B test prompt engineering
- **Batch mode** - multi-prompt × multi-model for real evaluation work
- **Local vs. cloud comparisons** - quantify the gap (or lack thereof)

### 🧪 Evaluation features

- **Statistical reproducibility analysis** - `--runs N` runs each configuration N times and reports mean/median/stdev/CV of latency and tokens, output frequency, mode output, and output diversity. Combine with `--check-hallucination` to measure hallucination rate across runs.
- **Deterministic assertions** - 10 assertion types (`contains`, `regex`, `json_valid`, `json_schema`, `max_length_chars`, `latency_under`, `cost_under`, and more) with pass/fail output and CI exit codes
- **LLM-as-a-judge scoring** - Use one LLM to score outputs from others on quality criteria
- **Judge panels** - Multiple judges average scores for less biased evaluation
- **Hallucination detection preset** - Ready-to-use criteria for factual accuracy checking
- **Custom criteria** - Define your own scoring rubrics
- **Self-evaluation auto-skip** - Judge models automatically skipped when also being judged

### 💾 Output formats

- **Live terminal** - Rich-powered panels with progress bars and streaming display
- **CSV** - Spreadsheet-friendly (open in Excel, Google Sheets, pandas)
- **JSON** - Structured for scripts and pipelines
- **Markdown** - Pretty tables for blog posts and reports
- **Exit codes** - 0/1/2 reflecting pass/fail status for CI/CD

### 💰 Cost transparency

- Per-call cost shown from each provider's reported usage
- Total cost summary per comparison
- Judge cost shown separately when LLM-as-judge is enabled
- Local models displayed as "Free"
- `--max-cost` flag to prevent surprise bills

### 🔒 Security

- API keys stored in OS-native keychain via `keyring` (Mac Keychain, Windows Credential Manager, Linux Secret Service)
- Format validation catches paste errors before storage
- Error message redaction prevents key leakage in tracebacks
- Localhost-only validation for local model URLs
- `SECURITY.md` with responsible disclosure policy

### 🛡️ Rate limit handling

- Per-provider concurrency limits (default 5) respect all tier baselines
- Automatic 429 retry with exponential backoff
- Anthropic 529 "overloaded" handled separately from rate limits
- `--concurrency` flag for power users on higher tiers
- Graceful per-model failure (other models continue)

### 🌐 Cross-platform

- Works identically on macOS, Windows (10+ and ARM), and Linux
- All file I/O uses `pathlib` + explicit UTF-8 encoding
- CSV writing uses `newline=""` for Windows compatibility
- Python 3.11+ required

### 📋 Developer experience

- **Single CLI binary** - `pip install cli-modelarium` and you're done
- **Polished Rich-based UI** - Claude Code-level terminal polish
- **JSON output** - Pipe into anything (`jq`, scripts, monitoring)
- **CI/CD ready** - Exit codes, structured output, GitHub Actions example included
- **Apache 2.0 licensed** - Use in any project, commercial or otherwise

## Examples

### Compare 3 models on a coding task

```bash
cli-modelarium "Write a Python function to find the longest palindromic substring" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro
```

### Reproducibility analysis - run N times and see variance

```bash
cli-modelarium "What is quantum computing?" \
  --models gpt-5.5,claude-opus-4-7 \
  --runs 5
```

Each model gets called 5 times in parallel. The output shows mean/stdev of
latency, coefficient of variation, mode answer, and output diversity per
model. Combine with `--check-hallucination` and `--judge` to measure the
hallucination rate across runs.

### Statistical significance testing

When you run two or more models with `--runs > 1`, cli-modelarium automatically
computes pairwise statistical significance tests (Welch's t-test by default)
with Bonferroni correction and Cohen's d effect sizes. The math is delegated
to [scipy](https://scipy.org/) so results match the scientific Python ecosystem.

```bash
cli-modelarium "Solve this math problem step by step" \
  --models gpt-5.5,claude-opus-4-7 \
  --runs 20 \
  --judge gemini-3.1-pro
```

The output adds a "Statistical Significance Tests" block with pairwise
p-values (corrected), Cohen's d effect sizes, and a significance verdict at
the chosen threshold. The default metric is the judge `score` when judging is
on, otherwise `latency_ms`.

Customise the analysis:

```bash
cli-modelarium "Q" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro \
  --runs 30 \
  --judge mistral-large \
  --significance-test mann-whitney \
  --correction holm \
  --significance-threshold 0.01
```

Opt out:

```bash
cli-modelarium "Q" --models gpt-5.5,claude --runs 10 --no-significance
```

### Bootstrap confidence intervals (v0.1.3)

Every reported mean comes with a bootstrap confidence interval showing
measurement uncertainty. CIs are auto-enabled whenever `--runs > 1`, and the
default method is BCa (bias-corrected and accelerated) — the publication-grade
standard.

```bash
cli-modelarium "Q" \
  --models gpt-5.5,claude-opus-4-7 \
  --runs 30 \
  --bootstrap-seed 42
```

For publication, **always set `--bootstrap-seed`**. Without a seed, CIs vary
slightly across invocations because the bootstrap resampling is stochastic.

Customise:

```bash
cli-modelarium "Q" --models gpt-5.5,claude --runs 30 \
  --ci-level 0.99 \
  --ci-method percentile \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 42
```

Disable entirely:

```bash
cli-modelarium "Q" --models gpt-5.5,claude --runs 30 --no-confidence-intervals
```

### Paired tests for same-prompt comparisons (v0.1.3)

When the same prompts are run on multiple models, **paired** tests have more
statistical power than independent-sample tests because they exploit the
within-prompt correlation. Pick `paired-t` for roughly-normal score
differences and `wilcoxon-signed` for ordinal or non-normal data.

```bash
cli-modelarium "Q" --models gpt-5.5,claude --runs 30 \
  --significance-test paired-t
```

```bash
cli-modelarium "Q" --models gpt-5.5,claude --runs 30 \
  --significance-test wilcoxon-signed
```

Paired tests automatically align observations by `run_index`, so they handle
asymmetric failures correctly (if model A succeeded runs `[0,1,2,4,5]` and
model B succeeded `[0,1,3,4,5]`, only the intersection `[0,1,4,5]` is used).

### McNemar's test for hallucination significance (v0.1.3)

When `--check-hallucination` is set with `--runs > 1` and 2+ models, McNemar's
test automatically compares hallucination pass/fail outcomes between every
pair of models. The implementation uses the exact binomial test for small
discordant counts (`n_discordant < 25`) and Edwards continuity-corrected
chi-square otherwise.

```bash
cli-modelarium "Q" --models gpt-5.5,claude --runs 30 \
  --check-hallucination --expected-facts facts.txt \
  --bootstrap-seed 42
```

The output adds a "Binary Outcome Significance (McNemar)" block alongside the
standard significance tests.

### Batch evaluation with assertions

Create `eval.json`:

```json
[
  {
    "id": "math-1",
    "prompt": "What is 2 + 2?",
    "assertions": [
      {"type": "contains", "value": "4"},
      {"type": "max_length_chars", "value": 100}
    ]
  },
  {
    "id": "json-1",
    "prompt": "List 3 colors in JSON array format",
    "assertions": [
      {"type": "json_valid"}
    ]
  }
]
```

Run it:

```bash
cli-modelarium batch eval.json \
  --models gpt-5.5,claude-opus-4-7 \
  --output results.csv
```

### Score outputs with an LLM judge

```bash
cli-modelarium "Explain recursion in one paragraph" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro,local/llama-3.3-70b \
  --judge claude-opus-4-7 \
  --judge-criteria "accuracy,clarity,brevity"
```

### Detect hallucinations against known facts

```bash
cli-modelarium "Tell me about the Eiffel Tower" \
  --models gpt-5.5,claude-opus-4-7 \
  --judge claude-opus-4-7 \
  --check-hallucination \
  --expected-facts "Built 1887-1889,Located in Paris France,Designed by Gustave Eiffel"
```

### Compare local model against cloud APIs

```bash
# Start Ollama first: ollama run llama3.3
cli-modelarium "Summarize the key features of microservices architecture" \
  --models local/llama-3.3-70b,gpt-5.5,claude-opus-4-7
```

### Run in CI/CD (GitHub Actions example)

```yaml
- name: Run LLM evaluation
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    cli-modelarium batch ./eval/test_suite.json \
      --models gpt-5.5,claude-opus-4-7 \
      --output eval_results.json \
      --min-pass-rate 0.90
```

The command exits with code 1 if pass rate drops below 90%, failing the build.

## Configuration

### API keys

Cli Modelarium stores API keys in your OS-native keychain (Mac Keychain, Windows Credential Manager, or Linux Secret Service via `keyring`). Keys never touch disk in plain text.

```bash
# Interactive setup (recommended)
cli-modelarium configure

# Or set individually
cli-modelarium keys set openai
cli-modelarium keys set anthropic
cli-modelarium keys set google

# Check which keys are configured
cli-modelarium keys list

# Remove a key
cli-modelarium keys delete openai
```

You can also use environment variables (useful for CI/CD):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

Environment variables take precedence over keychain storage.

### Headless Linux servers

On Linux servers without a desktop environment (no `gnome-keyring`, KWallet, or other Secret Service backend), the OS keyring may not be available — common on CI/CD runners, cloud VMs, and Docker containers. In that case, skip `cli-modelarium configure` and `keys set` entirely, and use environment variables instead:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."        # or GEMINI_API_KEY
export MISTRAL_API_KEY="..."
export XAI_API_KEY="xai-..."
export DEEPSEEK_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
export OPENROUTER_API_KEY="sk-or-..."
```

`cli-modelarium` checks environment variables before the OS keyring, so this works out of the box. If you prefer a keyring on Linux, install `gnome-keyring` (GNOME), KWallet (KDE), or `keyrings.alt` (file-based fallback).

### Local models (Ollama, LM Studio, etc.)

Local models work via OpenAI-compatible endpoints - no API keys needed. The tool auto-detects the default Ollama port.

```bash
# Default: assumes Ollama at localhost:11434
cli-modelarium "test" --models local/llama-3.3

# Use LM Studio instead
cli-modelarium "test" --models local/qwen-3-32b --local-url http://localhost:1234/v1

# Save a custom local URL as default
cli-modelarium keys set local --base-url http://localhost:1234/v1
```

## Supported providers

| Provider | API Keys Needed | Streaming | Cost Tracking |
|----------|-----------------|-----------|---------------|
| OpenAI (GPT-5, GPT-5 mini, o3, o4-mini, etc.) | ✅ | ✅ | ✅ |
| Anthropic (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5, etc.) | ✅ | ✅ | ✅ |
| Google (Gemini 3.1 Pro, Gemini 3 Flash, etc.) | ✅ | ✅ | ✅ |
| xAI (Grok 4.1, etc.) | ✅ | ✅ | ✅ |
| DeepSeek (V3, R1) | ✅ | ✅ | ✅ |
| Mistral (Large, Medium, Small) | ✅ | ✅ | ✅ |
| Groq (Llama, Mixtral, etc.) | ✅ | ✅ | ✅ |
| OpenRouter (any model on the platform) | ✅ | ✅ | ✅ |
| **Local: Ollama** | ❌ | ✅ | Free |
| **Local: LM Studio** | ❌ | ✅ | Free |
| **Local: vLLM** | ❌ | ✅ | Free |
| **Local: llama.cpp server** | ❌ | ✅ | Free |

Run `cli-modelarium list-models` to see all currently supported models.

## How it works

Cli Modelarium uses a modular provider abstraction layer that hides the API differences between OpenAI's `messages` array, Anthropic's top-level `system` parameter, Google's `system_instruction`, and others. Every provider implements the same async streaming interface, so the CLI can run them all in parallel with `asyncio.gather()`.

Cost calculations come from each provider's reported `usage` field (input tokens, output tokens, cached tokens) multiplied by current pricing constants. Pricing data was verified from official provider documentation on **May 25, 2026** - see [Notes & Limitations](#notes--limitations) for caveats.

For local models, the same OpenAI Python SDK is used with a custom `base_url`, since Ollama, LM Studio, vLLM, and llama.cpp all expose OpenAI-compatible REST endpoints.

## Notes & Limitations

### Pricing data

All pricing built into Cli Modelarium was verified from official provider documentation on **May 25, 2026**. LLM pricing changes frequently (sometimes monthly). The tool displays the `pricing_as_of` date in every output. Always verify against each provider's official pricing page before relying on cost calculations for budgeting or production decisions.

### Rate limits

Rate limit handling and the default per-provider concurrency settings are based on provider rate limits verified **May 25, 2026**. Your specific tier's limits may differ from the defaults assumed here. Verify your current limits against the provider's official dashboard before building production capacity assumptions.

### Model availability

Models supported by Cli Modelarium reflect what providers offered on **May 25, 2026**. Providers regularly release new models, deprecate older ones, and adjust capabilities. If a model in the registry no longer works, run `cli-modelarium list-models` and check the provider's documentation.

### Not a production-grade gateway

Cli Modelarium is designed for evaluation and comparison - running ad-hoc side-by-side tests across providers from a developer terminal. It is NOT a production inference gateway. If you need production-scale routing, load balancing, fallback chains, or SLA-managed inference, look for tools specifically built for that purpose.

### Token count comparisons across providers

Token counts shown in results are reported by each provider's API. Different providers use different tokenizers, so "output tokens" is not directly comparable across providers for the same text. If you're comparing cost efficiency for production use, run real prompts in your actual workload - don't rely solely on per-token math across providers.

### LLM-as-a-Judge usage

Cli Modelarium includes optional LLM-as-a-judge scoring (enabled with the `--judge` flag), which uses one LLM to evaluate outputs from other LLMs. This is standard benchmarking methodology and is permitted under the Terms of Service of all supported providers as evaluation/benchmarking activity.

When using `--judge`, you are responsible for following the Terms of Service of each provider whose models you use. Each provider's ToS applies to both the models being judged and the judge model itself.

**Judge bias notice:** LLM judges have documented biases (self-preference, same-family preference, verbosity preference). Judge scores are useful signal, not ground truth. Use judge panels (`--judges` with multiple models) to reduce bias.

### Hallucination detection

The hallucination detection preset is a useful comparison signal between models, not a ground-truth validation. Detection accuracy varies based on the judge model used, the domain knowledge required, and whether reference facts are provided via `--expected-facts`. Use it for relative quality comparison, not absolute correctness verification.

### Comparison methodology

LLMs are non-deterministic at temperature > 0 - re-running the same prompt may produce different outputs. A single comparison run shows you ONE sample from each model, not a definitive quality verdict.

To draw more reliable conclusions:
- Use `--runs 5` (or higher) to automatically run each comparison N times and see statistical summaries: mean/median latency, coefficient of variation, mode output, and output diversity. Coefficient of variation below 0.05 indicates stable model behavior across runs.
- For hallucination consistency analysis, combine `--runs` with `--check-hallucination` to see how often the model produces hallucinations across multiple runs (the hallucination rate).
- Use `--temperatures 0` for more deterministic outputs (where supported)
- Compare across multiple prompts, not just one
- Use the `--output json` flag to save runs for systematic analysis (with `--runs > 1` the JSON includes per-cell `stats_by_cell` aggregates)

## About the author

Cli Modelarium was built by **[Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)**.

### Connect

- 💼 LinkedIn: [linkedin.com/in/lavellehatcherjr](https://linkedin.com/in/lavellehatcherjr)
- 🐙 GitHub: [github.com/lavellehatcherjr](https://github.com/lavellehatcherjr)
- 💬 Questions about this project: [open an issue](../../issues)
- 📩 Collaboration/opportunities: reach out via LinkedIn

## Why I built this

Comparing LLM outputs across providers is tedious - different SDKs, different auth patterns, different response shapes, no easy way to see them side-by-side with cost and latency data. The polished cloud playgrounds only show one provider at a time, and the available open source options either focus on production routing or are full evaluation platforms optimized for teams.

Cli Modelarium is the small, focused CLI tool that does one thing well: side-by-side comparison with quality scoring, assertions, batch mode, and streaming - all designed for the terminal-first developer workflow.

It's intentionally focused: no production routing, no agent orchestration, no fine-tuning, no GUI. Just clean, fast comparison from the command line.

Built with a modular provider abstraction, parallel execution, transparent cost calculation, and secure key storage via OS keychain systems for local users.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

For security issues, please see [SECURITY.md](SECURITY.md) - do not file public issues for security concerns.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

See the [NOTICE](NOTICE) file for attribution requirements.

---

Built by [Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)

Licensed under Apache 2.0. Issues, PRs, and conversations welcome.
