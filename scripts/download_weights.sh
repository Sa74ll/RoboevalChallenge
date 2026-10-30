#!/usr/bin/env bash
# Download the LiftPot ACT weights (the bundle's only learned model, 137 MB) from Hugging Face into
# submission/checkpoint/LiftPot/ and check them against the published SHA-1 list.
# Needs the Hugging Face CLI: pip install -U huggingface_hub
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=Sa74ll/LocalModels-ACT-bimanual
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
hf download "$REPO" --include "submission/checkpoint/LiftPot/*" --include SHA1SUMS.txt --local-dir "$TMP"
(cd "$TMP" && grep '  submission/checkpoint/LiftPot/' SHA1SUMS.txt | sha1sum -c -)
cp "$TMP"/submission/checkpoint/LiftPot/* submission/checkpoint/LiftPot/
echo "LiftPot weights are in submission/checkpoint/LiftPot/"
