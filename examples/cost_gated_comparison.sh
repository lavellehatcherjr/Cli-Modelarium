#!/usr/bin/env bash
# Refuse a run if the estimated cost exceeds a ceiling (--max-cost).
#
# The gate estimates cost BEFORE any API call and refuses if it exceeds the
# ceiling (judge cost is tracked separately). The estimate for these two
# models is ~$0.0325/run, which exceeds 0.01 - so this command demonstrates
# the refusal ("Estimated cost ... exceeds --max-cost 0.01. Refusing to run.").
# Raise the ceiling (e.g. --max-cost 0.10) to let it run.

set -euo pipefail

cli-modelarium "Explain monads simply." \
  --models gpt-5.5,claude-opus-4-7 \
  --max-cost 0.01
