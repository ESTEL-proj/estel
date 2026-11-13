"""tools/check_workflows.py fails on workflows that break review rule E5."""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import REPO, run_tool, write

CHECKOUT = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0"
SETUP_UV = "astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e # v6.8.0"
REUSABLE = "org/repo/.github/workflows/build.yml@0123456789abcdef0123456789abcdef01234567 # v1.0.0"
DIGEST = "sha256:" + "a" * 64

WORKFLOW = f"""name: ci
on: [push]

# Least privilege.
permissions:
  contents: read

jobs:
  check:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: read
    steps:
      - uses: {CHECKOUT}
        with:
          fetch-depth: 0
          persist-credentials: false

      - name: Second checkout, keys in another order
        with:
          persist-credentials: "false"
          path: other
        uses: {CHECKOUT}

      - uses: {SETUP_UV}
        with:
          enable-cache: true

      - uses: ./.github/actions/local

      - uses: docker://alpine@{DIGEST} # 3.20

      - name: A script that mentions uses and permissions
        run: |
          echo "uses: actions/checkout@v4"
          # permissions: none
        env:
          A: b

  reusable:
    uses: {REUSABLE}
"""

# The `with:` keys of the first checkout, after its `with:` line.
CHECKOUT_WITH = "          fetch-depth: 0\n          persist-credentials: false\n"

GOOD = {
    ".github/workflows/ci.yml": WORKFLOW,
    ".github/dependabot.yml": "version: 2\n",  # not a workflow: never checked
}


def check(root: Path) -> tuple[int, str]:
    result = run_tool("check_workflows.py", "--root", str(root))
    return result.returncode, result.stderr


def test_repository_passes() -> None:
    code, err = check(REPO)
    assert code == 0, err


def test_compliant_workflow_passes(tmp_path: Path) -> None:
    write(tmp_path, GOOD)
    code, err = check(tmp_path)
    assert code == 0, err


@pytest.mark.parametrize(
    ("workflow", "message"),
    [
        pytest.param(
            WORKFLOW.replace(SETUP_UV, "astral-sh/setup-uv@v6"),
            "`uses: astral-sh/setup-uv@v6` is not pinned to a full commit SHA",
            id="tag",
        ),
        pytest.param(
            WORKFLOW.replace(SETUP_UV, "astral-sh/setup-uv@d0cc045 # v6.8.0"),
            "`uses: astral-sh/setup-uv@d0cc045` is not pinned to a full commit SHA",
            id="short-sha",
        ),
        pytest.param(
            WORKFLOW.replace(REUSABLE, "org/repo/.github/workflows/build.yml@main"),
            "`uses: org/repo/.github/workflows/build.yml@main` is not pinned",
            id="reusable-workflow-on-branch",
        ),
        pytest.param(
            WORKFLOW.replace(SETUP_UV, SETUP_UV.split(" #")[0]),
            "has no version comment",
            id="no-version-comment",
        ),
        pytest.param(
            WORKFLOW.replace(f"alpine@{DIGEST}", "alpine:3.20"),
            "`uses: docker://alpine:3.20` is not pinned to an image digest",
            id="docker-tag",
        ),
        pytest.param(
            WORKFLOW.replace(CHECKOUT_WITH, ""),
            "ci.yml:14: this actions/checkout step keeps the token",
            id="checkout-without-with",
        ),
        pytest.param(
            WORKFLOW.replace("          persist-credentials: false\n", ""),
            "ci.yml:14: this actions/checkout step keeps the token",
            id="checkout-without-persist-credentials",
        ),
        pytest.param(
            WORKFLOW.replace("persist-credentials: false", "persist-credentials: true"),
            "ci.yml:14: this actions/checkout step keeps the token",
            id="checkout-persisting-credentials",
        ),
        pytest.param(
            WORKFLOW.replace('persist-credentials: "false"', "persist-credentials: true"),
            "ci.yml:23: this actions/checkout step keeps the token",
            id="checkout-step-not-starting-with-uses",
        ),
        pytest.param(
            WORKFLOW.replace(CHECKOUT_WITH, "").replace(
                "          enable-cache: true\n", "          persist-credentials: false\n"
            ),
            "ci.yml:14: this actions/checkout step keeps the token",
            id="persist-credentials-in-another-step",
        ),
        pytest.param(
            WORKFLOW.replace("permissions:\n  contents: read\n", ""),
            "ci.yml has no top-level `permissions:` block",
            id="only-job-permissions",
        ),
    ],
)
def test_planted_violation_fails(tmp_path: Path, workflow: str, message: str) -> None:
    write(tmp_path, {**GOOD, ".github/workflows/ci.yml": workflow})
    code, err = check(tmp_path)
    assert code == 1
    assert message in err


def test_yaml_extension_is_checked(tmp_path: Path) -> None:
    write(tmp_path, {**GOOD, ".github/workflows/other.yaml": "on: [push]\njobs: {}\n"})
    code, err = check(tmp_path)
    assert code == 1
    assert "other.yaml has no top-level `permissions:` block" in err
