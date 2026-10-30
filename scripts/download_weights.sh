#!/usr/bin/env bash
# Download the LiftPot ACT weights (the bundle's only learned model, 137 MB) from Hugging Face into
# submission/checkpoint/LiftPot/ and check them against the published SHA-1 list.
# Needs the Hugging Face CLI: pip install -U huggingface_hub
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=Sa74ll/LocalModels-ACT-bimanual
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
hf download "$REPO" --include "pick22/checkpoint/LiftPot/*" --include "pick22/SHA1SUMS_pick22_released.txt" --local-dir "$TMP"
(cd "$TMP/pick22" && grep '  \./checkpoint/LiftPot/' SHA1SUMS_pick22_released.txt | sha1sum -c -)
cp "$TMP"/pick22/checkpoint/LiftPot/* submission/checkpoint/LiftPot/
echo "LiftPot weights are in submission/checkpoint/LiftPot/"
