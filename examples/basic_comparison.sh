#!/usr/bin/env bash
# Basic LLM comparison example.
# Compares the same prompt across OpenAI, Anthropic, and Google models.

set -euo pipefail

cli-modelarium "Explain quantum entanglement in 2 sentences." \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro-preview \
  --temperatures 0.7
