#!/usr/bin/env bash
# Score outputs with a panel of judges (averaged) to reduce single-judge bias.
#
# --judges runs multiple judge models and averages their scores. A model never
# judges its own output (self-evaluation is skipped automatically).

set -euo pipefail

cli-modelarium "Explain TCP vs UDP in 3 sentences." \
  --models gpt-5.5,claude-opus-4-7 \
  --judges claude-opus-4-7,gemini-3.1-pro-preview
