"""tools/check_layers.py fails on class-name tests, ``__class__`` writes and in-place updates."""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import REPO, run_tool, write

CLEAN = '''
"""Functional L2 code: compute, collect, assemble."""


def level(x, plan, xp):
    values = [xp.mean(x * w) for w in plan.weights]
    y = xp.stack(values, axis=-1)
    y = y + 1
    first = y[0]
    table = {j: first for j in plan.scales}
    obj = type(x)
    name = type(x).__name__
    kind = x.__class__
    total = 0
    total += 1
    x.__init_subclass__()
    data = x.downsample(inplace=False)
    return y, table, obj, name, kind, total, data
'''


def fake_repo(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    write(
        tmp_path,
        {
            "src/estel/__init__.py": "",
            "src/estel/scattering/_loop.py": CLEAN,
            "src/estel/backend/torch/_ops.py": "def add(x, y):\n    return x.add_(y)\n",
            **(files or {}),
        },
    )
    return tmp_path


def check(root: Path) -> tuple[int, str, str]:
    result = run_tool("check_layers.py", "--root", str(root))
    return result.returncode, result.stdout, result.stderr


def test_repository_passes() -> None:
    code, out, err = check(REPO)
    assert code == 0, err
    assert "check_layers: OK" in out


def test_clean_repository_passes(tmp_path: Path) -> None:
    code, _, err = check(fake_repo(tmp_path))
    assert code == 0, err


@pytest.mark.parametrize(
    ("path", "source", "message"),
    [
        pytest.param(
            "src/estel/data/_base.py",
            "ok = type(x).__name__ == 'STL_2D_Kernel_Torch'\n",
            "compares a class name as a string (`type(x).__name__`)",
            id="type-name-eq",
        ),
        pytest.param(
            "src/estel/backend/_registry.py",
            "ok = 'Tensor' != type(x).__name__\n",
            "Use `isinstance(x, Cls)`",
            id="type-name-ne-outside-l1-l2",
        ),
        pytest.param(
            "src/estel/_spec/_planar.py",
            "ok = x.__class__.__name__ in ('A', 'B')\n",
            "compares a class name as a string (`x.__class__.__name__`)",
            id="class-name-in-oracle",
        ),
        pytest.param(
            "src/estel/data/tensor_prod.py",
            "self.__class__ = Block\n",
            "assigns to `__class__`",
            id="class-assign",
        ),
        pytest.param(
            "src/estel/engines/_python.py",
            "obj.__class__, y = Block, 1\n",
            "Compose instead",
            id="class-assign-tuple-outside-l1-l2",
        ),
        pytest.param(
            "src/estel/__init__.py",
            "setattr(obj, '__class__', Block)\n",
            "assigns to `__class__`",
            id="class-setattr",
        ),
        pytest.param(
            "src/estel/data/_base.py",
            "object.__setattr__(obj, '__class__', Block)\n",
            "assigns to `__class__`",
            id="class-object-setattr",
        ),
        pytest.param(
            "src/estel/scattering/_loop.py",
            "s4[:, :, j1] = cov(u1, u2)\n",
            "writes into an array in place (`s4[...] = ...`)",
            id="subscript-assign",
        ),
        pytest.param(
            "src/estel/statistics/_reduce.py",
            "a, b[0] = 1, 2\n",
            "writes into an array in place (`b[...] = ...`)",
            id="subscript-assign-in-tuple",
        ),
        pytest.param(
            "src/estel/functional/_scattering.py",
            "out: Array = x\nout[mask] = 0\n",
            "collect them in a list and `stack` them",
            id="subscript-assign-l2-functional",
        ),
        pytest.param(
            "src/estel/functional.py",
            "for x[0] in range(3):\n    pass\n",
            "writes into an array in place (`x[...] = ...`)",
            id="subscript-for-target-module-layer",
        ),
        pytest.param(
            "src/estel/wavelets/_kernels.py",
            "kernel[..., 0] += shift\n",
            "updates an array in place (`kernel[...] += ...`)",
            id="subscript-augassign",
        ),
        pytest.param(
            "src/estel/synthesis/_driver.py",
            "xp.add(a, b, out=a)\n",
            "passes `out=` to `xp.add`",
            id="out-keyword",
        ),
        pytest.param(
            "src/estel/data/planar.py",
            "x.add_(y)\n",
            "calls `.add_()`, an in-place method",
            id="inplace-method",
        ),
        pytest.param(
            "src/estel/synthesis/_driver.py",
            "x.detach().requires_grad_(True)\n",
            "calls `.requires_grad_()`",
            id="inplace-method-chained",
        ),
        pytest.param(
            "src/estel/data/_base.py",
            "data.downsample(inplace=True)\n",
            "passes `inplace=True` to `data.downsample`",
            id="inplace-keyword",
        ),
        pytest.param(
            "src/estel/data/_base.py",
            "data.downsample(inplace=flag)\n",
            "passes `inplace=flag`",
            id="inplace-keyword-variable",
        ),
        pytest.param(
            "src/estel/scattering/_loop.py",
            "def f(:\n",
            "cannot be parsed",
            id="syntax-error",
        ),
    ],
)
def test_crafted_violation_fails(tmp_path: Path, path: str, source: str, message: str) -> None:
    root = fake_repo(tmp_path, {path: source})
    code, _, err = check(root)
    assert code == 1
    assert message in err
    assert f"check_layers: {path}:" in err


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("s4[:, j1] = cov(u1, u2)\n", id="subscript-assign"),
        pytest.param("kernel[0] += 1\n", id="subscript-augassign"),
        pytest.param("torch.add(a, b, out=a)\n", id="out-keyword"),
        pytest.param("x.copy_(y)\n", id="inplace-method"),
        pytest.param("f(x, inplace=True)\n", id="inplace-keyword"),
    ],
)
def test_in_place_allowed_outside_l1_l2(tmp_path: Path, source: str) -> None:
    for path in ("src/estel/backend/torch/_conv.py", "src/estel/_spec/_planar.py"):
        write(tmp_path, {path: source})
    code, _, err = check(fake_repo(tmp_path))
    assert code == 0, err


def test_every_violation_is_reported(tmp_path: Path) -> None:
    source = "x[0] = 1\nx.add_(y)\nf(out=x)\n"
    code, _, err = check(fake_repo(tmp_path, {"src/estel/data/_base.py": source}))
    assert code == 1
    for line in (1, 2, 3):
        assert f"src/estel/data/_base.py:{line}:" in err
    assert "3 violation(s)" in err


def test_missing_package_fails(tmp_path: Path) -> None:
    code, _, err = check(tmp_path)
    assert code == 1
    assert "Pass the repository root with --root" in err
