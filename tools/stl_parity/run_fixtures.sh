#!/bin/sh
# Writes the STL parity fixtures (make_fixtures.py) in each of the three STL environments of the
# image. Run by `make parity-fixtures`, which mounts tools/stl_parity at /mnt/stl_parity and
# tests/parity/fixtures at /mnt/fixtures, and sets PARITY_IMAGE_ID. Arguments: case names
# (default: every case). Exits non-zero if any case fails.
set -u
: "${PARITY_IMAGE_ID:?set PARITY_IMAGE_ID to the ID of this image: run make parity-fixtures}"
status=0
for env in master 2d1d healpix; do
    echo "== STL environment: $env"
    "/opt/venv/$env/bin/python" /mnt/stl_parity/make_fixtures.py "$env" \
        --out /mnt/fixtures --image-id "$PARITY_IMAGE_ID" "$@" || status=1
done
if [ "$status" -eq 0 ]; then
    echo "Fixtures written to tests/parity/fixtures/: commit them with MANIFEST.json."
else
    echo "At least one case failed: read its errors above." >&2
fi
exit "$status"
