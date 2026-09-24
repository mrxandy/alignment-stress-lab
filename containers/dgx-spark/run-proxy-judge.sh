#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/AI/alignment-stress-lab
input_path=runtime/outputs/proxy-9b-dev-group-1.jsonl
output_path=runtime/outputs/proxy-9b-dev-group-1.judged.jsonl
if [[ ! -f "$input_path" ]]; then
    echo 'Proxy rollout response file is missing' >&2
    exit 1
fi
if [[ "$(wc -l < "$input_path")" -ne 8 ]]; then
    echo 'Expected exactly 8 proxy rollout responses' >&2
    exit 1
fi
if [[ -e "$output_path" ]]; then
    echo 'Proxy Judge output already exists; refusing to overwrite it' >&2
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

echo 'Sending up to 8 fictional proxy prompt-response pairs to DeepSeek deepseek-chat.' >&2
printf '%s\n' "$judge_api_key" | sudo docker compose \
    -f containers/dgx-spark/compose.yaml exec -T \
    -e JUDGE_MODEL=deepseek-chat \
    -e JUDGE_BASE_URL=https://api.deepseek.com/v1 \
    alignment-stress-lab sh -c '
        IFS= read -r JUDGE_API_KEY
        export JUDGE_API_KEY
        exec python3 -m alignment_stress_lab.score_proxy_group \
            --config configs/judge.proxy.deepseek.yaml \
            --input /workspace/outputs/proxy-9b-dev-group-1.jsonl \
            --output /workspace/outputs/proxy-9b-dev-group-1.judged.jsonl
    '
unset judge_api_key
