#!/usr/bin/env bash
# Compare all models from a running local server (--models all-local).
# No API key needed.
#
# Discovers models from a running local server via its /v1/models endpoint;
# the default is Ollama on localhost:11434. For LM Studio / vLLM / llama.cpp,
# point at their port with --local-url, e.g.:
#   --local-url http://localhost:1234/v1
# If no local server is reachable, you get a clear message (no crash).

set -euo pipefail

cli-modelarium "Explain recursion in one paragraph." --models all-local
