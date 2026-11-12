#!/bin/sh
# Runs the float64 smoke check (smoke.py) in each of the three STL environments of the image.
# Default command of the parity container; exits non-zero if any environment fails.
set -u
status=0
for env in master 2d1d healpix; do
    echo "== STL environment: $env"
    "/opt/venv/$env/bin/python" /opt/stl_parity/smoke.py "$env" || status=1
done
if [ "$status" -eq 0 ]; then
    echo "STL runs in float64 in all three environments."
else
    echo "At least one STL environment failed: read its errors above." >&2
fi
exit "$status"
