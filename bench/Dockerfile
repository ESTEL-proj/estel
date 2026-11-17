# Benchmark image (Part I, §6.3; task T2.11, issue #89): the parity container with ESTEL
# installed, so that ESTEL and STL are measured in the same environment.
#
# ESTEL goes into the `master` environment of the parity image (tools/stl_parity/README.md), the
# one that measures STL for 2D data: the same interpreter, NumPy and PyTorch, the version of
# ESTEL's uv.lock. It adds ESTEL's only other dependency, array-api-compat, at the version of
# uv.lock. `make bench` builds this image from the working tree before each container run; the
# benchmark scripts of bench/ are mounted, not copied. Build context: the repository root,
# filtered by bench/Dockerfile.dockerignore.
#
#   make parity-image && make bench

ARG PARITY_IMAGE=estel-stl-parity
FROM ${PARITY_IMAGE}

RUN uv pip install --quiet --python /opt/venv/master/bin/python array-api-compat==1.16.0

COPY pyproject.toml AGENTS.md LICENSE NOTICE /opt/estel/
COPY src/estel /opt/estel/src/estel
RUN uv pip install --quiet --python /opt/venv/master/bin/python --no-deps /opt/estel

WORKDIR /mnt
