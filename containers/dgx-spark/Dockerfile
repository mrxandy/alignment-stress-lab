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

# Install only the project's core dependencies. The NGC CUDA/PyTorch packages
# remain in the base image; requirements-train.txt is a later, verified step.
RUN python3 -m pip install --no-cache-dir . \
 && python3 -m pip check \
 && python3 -c "import alignment_stress_lab, torch; print(alignment_stress_lab.__version__, torch.__version__, torch.version.cuda)"

USER 1000:1000
WORKDIR /workspace/project
CMD ["sleep", "infinity"]
