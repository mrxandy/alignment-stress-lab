#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/AI/alignment-stress-lab
input_path=runtime/outputs/baseline-9b-dev-full.jsonl
if [[ ! -f "$input_path" ]]; then
    echo 'Baseline response file is missing' >&2
    exit 1
fi
if [[ "$(wc -l < "$input_path")" -ne 24 ]]; then
    echo 'Expected exactly 24 completed baseline responses' >&2
    exit 1
fi
if [[ -e runtime/outputs/baseline-9b-dev-full.judged.jsonl ]]; then
    echo 'Judge output already exists; refusing to overwrite it' >&2
    exit 1
fi

# Authorize Docker before reading the API key, so sudo cannot consume stdin.
sudo -v
IFS= read -r -s -p 'DeepSeek API key: ' judge_api_key
printf '\n' >&2
if [[ -z "$judge_api_key" ]]; then
    echo 'DeepSeek API key is empty' >&2
    exit 1
fi

echo 'Sending 24 baseline prompt-response pairs to DeepSeek deepseek-chat.' >&2
printf '%s\n' "$judge_api_key" | sudo docker compose \
    -f containers/dgx-spark/compose.yaml exec -T \
    -e JUDGE_MODEL=deepseek-chat \
    -e JUDGE_BASE_URL=https://api.deepseek.com/v1 \
    alignment-stress-lab sh -c '
        IFS= read -r JUDGE_API_KEY
        export JUDGE_API_KEY
        exec python3 -m alignment_stress_lab.score_baseline \
            --config configs/judge.deepseek.yaml \
            --input /workspace/outputs/baseline-9b-dev-full.jsonl \
            --output /workspace/outputs/baseline-9b-dev-full.judged.jsonl
    '
unset judge_api_key
