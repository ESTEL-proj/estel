# Parity harness container (Part II, §14.2; task T0.6).
#
# One image, three STL environments, one per frozen STL-Dev commit of Part II, §9.1:
#   /opt/venv/master    STL-Dev `master`        8eaf769 (v1.6.0)  2D cross spectra
#   /opt/venv/2d1d      STL-Dev `2D+1D`         8f51143           1D, 2D, tensor products
#   /opt/venv/healpix   STL-Dev `HealpixUpdate` 90eb1a9           HEALPix, with healpix-analyse aac5e26
# Every Python dependency is pinned in requirements/*.txt; PyTorch is 2.15.0+cpu in all three.
#
# Build and run from the repository root: make parity-image, then make parity.

FROM --platform=linux/amd64 python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_NO_CACHE=1 \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    UV_INDEX_STRATEGY=unsafe-best-match \
    PYTHONDONTWRITEBYTECODE=1

ARG STL_REPOSITORY=https://github.com/STL-Proj/STL-Dev.git
RUN git clone --quiet "$STL_REPOSITORY" /opt/stl/repo

COPY requirements/ /opt/stl_parity/requirements/

# One environment per STL commit: the pinned dependencies, then STL itself at the full commit
# hash, without dependencies. The check on `git rev-parse` fails the build if a commit moved.
RUN set -eu; \
    for spec in \
        master:8eaf7691d55f3c3441cde9b95f1f80eb17aa8014 \
        2d1d:8f51143c5f8ebc259328830f246894d7031acae5 \
        healpix:90eb1a954e71abb7af621b10e70b0456e8a86b4e; \
    do \
        env="${spec%%:*}"; commit="${spec#*:}"; \
        git -C /opt/stl/repo worktree add --quiet --detach "/opt/stl/$env" "$commit"; \
        test "$(git -C "/opt/stl/$env" rev-parse HEAD)" = "$commit"; \
        uv venv --quiet --python /usr/local/bin/python3.11 "/opt/venv/$env"; \
        uv pip install --quiet --python "/opt/venv/$env/bin/python" \
            -r "/opt/stl_parity/requirements/$env.txt"; \
        uv pip install --quiet --python "/opt/venv/$env/bin/python" --no-deps \
            --build-constraints /opt/stl_parity/requirements/build.txt "/opt/stl/$env"; \
    done

COPY smoke.py run_smoke.sh /opt/stl_parity/

# Run from a directory that holds no STL checkout, so each interpreter imports its installed STL.
WORKDIR /work
CMD ["/bin/sh", "/opt/stl_parity/run_smoke.sh"]
