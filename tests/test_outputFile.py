"""
End-to-end output comparison for residual, transport-theta and transport-press.

Runs each CLI with the shipped config files (which point at the sample input
data under tests/data/sample_input/), saves output to pytest tmp directories,
then compares every produced .nc file against the corresponding local baseline
under tests/data/baseline_output/.

"""

import re
import subprocess
import numpy as np
import pytest
import xarray as xr
from pathlib import Path

_TESTS_DIR  = Path(__file__).parent
_REPO_ROOT  = _TESTS_DIR.parent
_BASELINE   = _TESTS_DIR / "data" / "baseline_output"
_THETA_CFG   = _REPO_ROOT / "config" / "tracerTransportTheta_config.toml"
_PRESS_CFG   = _REPO_ROOT / "config" / "tracerTransportPress_config.toml"
_RESCIRC_CFG = _REPO_ROOT / "config" / "residualCirc_config.toml"

BASELINE_THETA   = _BASELINE / "tTransport_theta"
BASELINE_PRESS   = _BASELINE / "tTransport_press"
BASELINE_RESCIRC = _BASELINE / "residual_circulation"

_theta_baseline_files   = sorted(f.name for f in BASELINE_THETA.glob("*.nc"))
_press_baseline_files   = sorted(f.name for f in BASELINE_PRESS.glob("*.nc"))
_rescirc_baseline_files = sorted(f.name for f in BASELINE_RESCIRC.glob("*.nc"))


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def _assert_datasets_match(ds_base, ds_out, rtol=3e-3, atol=2e-7):
    """
    Compare two datasets variable by variable on finite elements only.
    NaN and ±inf positions are treated as fill values; their masks must
    be identical and all finite pairs must agree within rtol/atol.
    """
    assert set(ds_base.data_vars) == set(ds_out.data_vars), (
        f"Variable mismatch: baseline={set(ds_base.data_vars)}, "
        f"output={set(ds_out.data_vars)}"
    )
    for var in ds_base.data_vars:
        b = np.array(ds_base[var])
        o = np.array(ds_out[var])
        fill_b = ~np.isfinite(b)
        fill_o = ~np.isfinite(o)
        assert np.array_equal(fill_b, fill_o), (
            f"{var}: fill-value mask differs "
            f"(baseline fill={fill_b.sum()}, output fill={fill_o.sum()})"
        )
        finite = ~fill_b
        np.testing.assert_allclose(
            b[finite], o[finite], rtol=rtol, atol=atol,
            err_msg=f"{var}: finite values differ",
        )


# ---------------------------------------------------------------------------
# Helper: run a CLI with a patched config that redirects output to tmp_dir
# ---------------------------------------------------------------------------

def _run_cli(command, config_src, tmp_dir):
    """
    Copy config_src to tmp_dir, patch outputDirectory to tmp_dir,
    Disable outDirSkip, then run `pixi run <command> <patched_config>`.
    Returns the tmp_dir path.
    """
    cfg_path = tmp_dir / config_src.name
    text = config_src.read_text()
    text = re.sub(
        r"^outputDirectory\s*=\s*'[^']*'",
        lambda _: f"outputDirectory = '{tmp_dir.as_posix()}'",
        text,
        flags=re.MULTILINE,
    )
    text = text.replace("outDirSkip = true", "outDirSkip = false")
    cfg_path.write_text(text)

    result = subprocess.run(
        ["pixi", "run", command, str(cfg_path)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{command} exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return tmp_dir


# ---------------------------------------------------------------------------
# Module-scoped fixtures — run each CLI once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def theta_output_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("theta_output")
    return _run_cli("transport-theta", _THETA_CFG, tmp)


@pytest.fixture(scope="module")
def press_output_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("press_output")
    return _run_cli("transport-press", _PRESS_CFG, tmp)


@pytest.fixture(scope="module")
def rescirc_output_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rescirc_output")
    return _run_cli("residual", _RESCIRC_CFG, tmp)


# ===========================================================================
# transport-theta tests
# ===========================================================================

def test_theta_cli_produces_output_files(theta_output_dir):
    assert list(theta_output_dir.glob("*.nc")), \
        f"No .nc files produced in {theta_output_dir}"


def test_theta_cli_produces_correct_file_count(theta_output_dir):
    produced = sorted(f.name for f in theta_output_dir.glob("*.nc"))
    assert len(produced) == len(_theta_baseline_files), (
        f"Expected {len(_theta_baseline_files)} files, got {len(produced)}.\n"
        f"Missing: {set(_theta_baseline_files) - set(produced)}\n"
        f"Extra:   {set(produced) - set(_theta_baseline_files)}"
    )


@pytest.mark.parametrize("filename", _theta_baseline_files)
def test_theta_output_matches_baseline(theta_output_dir, filename):
    produced = theta_output_dir / filename
    assert produced.exists(), f"CLI did not produce {filename}"
    with xr.open_dataset(BASELINE_THETA / filename) as ds_base, \
         xr.open_dataset(produced) as ds_out:
        _assert_datasets_match(ds_base, ds_out)


# ===========================================================================
# transport-press tests
# ===========================================================================

def test_press_cli_produces_output_files(press_output_dir):
    assert list(press_output_dir.glob("*.nc")), \
        f"No .nc files produced in {press_output_dir}"


def test_press_cli_produces_correct_file_count(press_output_dir):
    produced = sorted(f.name for f in press_output_dir.glob("*.nc"))
    assert len(produced) == len(_press_baseline_files), (
        f"Expected {len(_press_baseline_files)} files, got {len(produced)}.\n"
        f"Missing: {set(_press_baseline_files) - set(produced)}\n"
        f"Extra:   {set(produced) - set(_press_baseline_files)}"
    )


@pytest.mark.parametrize("filename", _press_baseline_files)
def test_press_output_matches_baseline(press_output_dir, filename):
    produced = press_output_dir / filename
    assert produced.exists(), f"CLI did not produce {filename}"
    with xr.open_dataset(BASELINE_PRESS / filename) as ds_base, \
         xr.open_dataset(produced) as ds_out:
        _assert_datasets_match(ds_base, ds_out)


# ===========================================================================
# residual tests
# ===========================================================================

def test_rescirc_cli_produces_output_files(rescirc_output_dir):
    assert list(rescirc_output_dir.glob("*.nc")), \
        f"No .nc files produced in {rescirc_output_dir}"


def test_rescirc_cli_produces_correct_file_count(rescirc_output_dir):
    produced = sorted(f.name for f in rescirc_output_dir.glob("*.nc"))
    assert len(produced) == len(_rescirc_baseline_files), (
        f"Expected {len(_rescirc_baseline_files)} files, got {len(produced)}.\n"
        f"Missing: {set(_rescirc_baseline_files) - set(produced)}\n"
        f"Extra:   {set(produced) - set(_rescirc_baseline_files)}"
    )


@pytest.mark.parametrize("filename", _rescirc_baseline_files)
def test_rescirc_output_matches_baseline(rescirc_output_dir, filename):
    produced = rescirc_output_dir / filename
    assert produced.exists(), f"CLI did not produce {filename}"
    with xr.open_dataset(BASELINE_RESCIRC / filename) as ds_base, \
         xr.open_dataset(produced) as ds_out:
        _assert_datasets_match(ds_base, ds_out)


# ===========================================================================
# Wave-summation tests: sum over waveN dimension ≈ 2-D total variable
#
# The Fourier decomposition normalises each wavenumber so that summing across
# all stored wavenumber bins recovers the zonal-mean eddy quantity (Parseval).
# Data are stored as float32, so tolerances account for float32 accumulation.
# ===========================================================================

def _assert_waveN_sum_matches_total(ds, waveN_var, total_var, wave_dim,
                                    rtol=5e-3, atol=1e-8):
    """Sum ds[waveN_var] over wave_dim and compare finite elements to ds[total_var]."""
    summed = np.array(ds[waveN_var].sum(dim=wave_dim).squeeze())
    total = np.array(ds[total_var].squeeze())
    finite = np.isfinite(summed) & np.isfinite(total)
    np.testing.assert_allclose(
        summed[finite], total[finite], rtol=rtol, atol=atol,
        err_msg=f"sum({waveN_var}, dim={wave_dim!r}) does not match {total_var}",
    )


# ---------------------------------------------------------------------------
# residual wave-summation tests
# ---------------------------------------------------------------------------

_RESCIRC_WAVE_PAIRS = [
    ("EPFVert_WaveN",     "EPF_vert"),
    ("EPFLat_WaveN",      "EPF_lat"),
    ("divEPFVert_WaveN",  "div_EPF_vert"),
    ("divEPFLat_WaveN",   "div_EPF_lat"),
    ("divEPF_WaveN",      "div_EPF")
]


@pytest.mark.parametrize("waveN_var,total_var", _RESCIRC_WAVE_PAIRS)
@pytest.mark.parametrize("filename", _rescirc_baseline_files)
def test_rescirc_waveN_sum_matches_total(rescirc_output_dir, filename,
                                         waveN_var, total_var):
    produced = rescirc_output_dir / filename
    assert produced.exists(), f"CLI did not produce {filename}"
    with xr.open_dataset(produced) as ds:
        if waveN_var not in ds.data_vars:
            pytest.skip(f"{waveN_var} not present in {filename}")
        _assert_waveN_sum_matches_total(ds, waveN_var, total_var, "waveNumber")


# ---------------------------------------------------------------------------
# transport-theta wave-summation tests
# ---------------------------------------------------------------------------

_THETA_WAVE_PAIRS = [
    ("BA_m_lat_WN",      "BA_m_lat"),
    ("BA_m_theta_WN",    "BA_m_theta"),
    ("BA_divm_lat_WN",   "BA_divm_lat"),
    ("BA_divm_theta_WN", "BA_divm_theta"),
]


@pytest.mark.parametrize("waveN_var,total_var", _THETA_WAVE_PAIRS)
@pytest.mark.parametrize("filename", _theta_baseline_files)
def test_theta_waveN_sum_matches_total(theta_output_dir, filename,
                                       waveN_var, total_var):
    produced = theta_output_dir / filename
    assert produced.exists(), f"CLI did not produce {filename}"
    with xr.open_dataset(produced) as ds:
        _assert_waveN_sum_matches_total(ds, waveN_var, total_var, "waveN")


@pytest.mark.parametrize("filename", _theta_baseline_files)
def test_theta_divm_waveN_sum_matches_total(theta_output_dir, filename):
    """BA_divm_WN summed over waveN must equal BA_divm_lat + BA_divm_theta."""
    produced = theta_output_dir / filename
    assert produced.exists(), f"CLI did not produce {filename}"
    with xr.open_dataset(produced) as ds:
        summed = np.array(ds["BA_divm_WN"].sum(dim="waveN").squeeze())
        total = np.array((ds["BA_divm_lat"] + ds["BA_divm_theta"]).squeeze())
        finite = np.isfinite(summed) & np.isfinite(total)
        np.testing.assert_allclose(
            summed[finite], total[finite], rtol=5e-3, atol=1e-8,
            err_msg="sum(BA_divm_WN) does not match BA_divm_lat + BA_divm_theta",
        )


# ---------------------------------------------------------------------------
# transport-press wave-summation tests
# ---------------------------------------------------------------------------

_PRESS_WAVE_PAIRS = [
    ("BA_m_lat_WN",    "BA_m_lat"),
    ("BA_m_z_WN",      "BA_m_z"),
    ("BA_divm_lat_WN", "BA_divm_lat"),
    ("BA_divm_z_WN",   "BA_divm_z"),
]


@pytest.mark.parametrize("waveN_var,total_var", _PRESS_WAVE_PAIRS)
@pytest.mark.parametrize("filename", _press_baseline_files)
def test_press_waveN_sum_matches_total(press_output_dir, filename,
                                       waveN_var, total_var):
    produced = press_output_dir / filename
    assert produced.exists(), f"CLI did not produce {filename}"
    with xr.open_dataset(produced) as ds:
        _assert_waveN_sum_matches_total(ds, waveN_var, total_var, "waveN")


@pytest.mark.parametrize("filename", _press_baseline_files)
def test_press_divm_waveN_sum_matches_total(press_output_dir, filename):
    """BA_divm_WN summed over waveN must equal BA_divm_lat + BA_divm_z."""
    produced = press_output_dir / filename
    assert produced.exists(), f"CLI did not produce {filename}"
    with xr.open_dataset(produced) as ds:
        summed = np.array(ds["BA_divm_WN"].sum(dim="waveN").squeeze())
        total = np.array((ds["BA_divm_lat"] + ds["BA_divm_z"]).squeeze())
        finite = np.isfinite(summed) & np.isfinite(total)
        np.testing.assert_allclose(
            summed[finite], total[finite], rtol=5e-3, atol=1e-8,
            err_msg="sum(BA_divm_WN) does not match BA_divm_lat + BA_divm_z",
        )
