# Reuse the ARM64 NVIDIA PyTorch image already validated on this DGX Spark.
ARG BASE_IMAGE=heretic-ngc-base:25.11-arm64
FROM ${BASE_IMAGE}

USER root
ENV PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/alignment-stress-lab
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install the training extras without changing the NGC CUDA/PyTorch stack.
RUN torch_stack_before="$(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')" \
 && python3 -m pip install --no-cache-dir '.[train]' pynvml \
 && python3 -m pip check \
 && test "$torch_stack_before" = "$(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')" \
 && python3 -c "import alignment_stress_lab, torch; print(alignment_stress_lab.__version__, torch.__version__, torch.version.cuda)"

USER 1000:1000
WORKDIR /workspace/project
CMD ["sleep", "infinity"]
