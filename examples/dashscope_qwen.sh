#!/usr/bin/env bash
# Compare Alibaba/DashScope Qwen models (needs DASHSCOPE_API_KEY).
#
# DashScope uses the International/Singapore endpoint. Set your key first:
#   export DASHSCOPE_API_KEY=sk-...   (or: cli-modelarium keys set dashscope)

set -euo pipefail

cli-modelarium "Write a haiku about databases." \
  --models qwen3.7-max,qwen3.6-flash \
  --temperatures 0.7
