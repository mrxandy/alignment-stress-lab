# DGX Spark container bootstrap

This Compose service creates a separate, GPU-enabled container for
Alignment Stress Lab. It installs the core project package over the locally
verified NVIDIA PyTorch 25.11 ARM64 base. It does not install training extras,
load a model, or run a GRP experiment yet.

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

The service exposes no ports. The project checkout and existing model/data
directories are mounted read-only. Runtime work, checkpoints, outputs, and
Hugging Face cache have separate persistent directories under `runtime/`.
The container runs as host UID:GID `1000:1000` so those new directories are
writable without changing ownership of existing model or dataset trees.

For an image build that needs the existing LAN proxy, set
`LAB_BUILD_PROXY=http://192.168.0.199:7897` only for the build command.
`LAB_PROXY` is a separate optional runtime proxy setting. Do not put Judge
API credentials in the Compose file or Git repository.
