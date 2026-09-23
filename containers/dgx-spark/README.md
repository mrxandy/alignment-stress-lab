# DGX Spark container bootstrap

This Compose service creates a separate, GPU-enabled container for
Alignment Stress Lab. It installs the project's training dependencies over the
locally verified NVIDIA PyTorch 25.11 ARM64 base, and fails the build if the
PyTorch/CUDA versions change. It does not load a model or run an experiment.

The DGX host user `admin` needs Docker access. On the current host, that means
running the Docker commands below with `sudo` in an interactive DGX terminal.
There are no new host Python packages or services.

## First setup on the DGX

```bash
cd /home/admin/AI
git clone https://github.com/mrxandy/alignment-stress-lab.git
cd alignment-stress-lab
mkdir -p runtime/work runtime/checkpoints runtime/outputs runtime/cache/huggingface
```

Check that the verified base image exists before building:

```bash
sudo docker image inspect heretic-ngc-base:25.11-arm64
```

The base corresponds to the NVIDIA PyTorch 25.11 ARM64 manifest digest
`sha256:4a85d8cf6fb3a943280960b8948cf4e9b6eca77b4414c68c9b2c7bb863f79b70`.
If the local tag is absent, verify and restore that exact image before build.

```bash
sudo docker compose -f containers/dgx-spark/compose.yaml config --quiet
sudo docker compose -f containers/dgx-spark/compose.yaml --progress plain build alignment-stress-lab
sudo docker compose -f containers/dgx-spark/compose.yaml up -d --no-build --pull never alignment-stress-lab
sudo docker compose -f containers/dgx-spark/compose.yaml exec alignment-stress-lab python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

For the first local-only Qwen3.5-9B load and short generation smoke test:

```bash
sudo docker compose -f containers/dgx-spark/compose.yaml exec -T alignment-stress-lab python3 -m alignment_stress_lab.model_smoke
```

This reads the aligned checkpoint from the read-only model mount. It does not
call the Judge API, save responses, or modify model weights.

To test line-by-line baseline generation on only two development prompts:

```bash
sudo docker compose -f containers/dgx-spark/compose.yaml exec -T alignment-stress-lab \
  python3 -m alignment_stress_lab.baseline --limit 2 \
  --output /workspace/outputs/baseline-9b-dev-smoke.jsonl
```

Choose a new output name for each run. The JSONL responses and adjacent
`.manifest.json` file stay under the ignored `runtime/outputs/` directory.
This stage does not call the Judge API or calculate a refusal rate.

The service exposes no ports. The project checkout and existing model/data
directories are mounted read-only. Runtime work, checkpoints, outputs, and
Hugging Face cache have separate persistent directories under `runtime/`.
The container runs as host UID:GID `1000:1000` so those new directories are
writable without changing ownership of existing model or dataset trees.

For an image build that needs the existing LAN proxy, set
`LAB_BUILD_PROXY=http://192.168.0.199:7897` only for the build command.
`LAB_PROXY` is a separate optional runtime proxy setting. Do not put Judge
API credentials in the Compose file or Git repository.
