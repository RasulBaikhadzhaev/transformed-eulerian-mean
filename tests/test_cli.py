"""
Tests for cli.py entry-point functions.

Each test patches sys.argv so that parse_args() sees the real sample config,
then overrides outputDirectory to a pytest tmp_path so tests are hermetic.
The CLIs are called directly (no subprocess) so coverage is measured.
"""
import re
import sys
import pytest
import pandas as pd
import xarray as xr
from pathlib import Path
from unittest.mock import patch, MagicMock

from tem_pkg.cli import run_residual, run_tracer_transport_theta, run_tracer_transport_press

_TESTS_DIR  = Path(__file__).parent
_REPO_ROOT  = _TESTS_DIR.parent
_THETA_CFG   = _REPO_ROOT / "config" / "tracerTransportTheta_config.toml"
_PRESS_CFG   = _REPO_ROOT / "config" / "tracerTransportPress_config.toml"
_RESCIRC_CFG = _REPO_ROOT / "config" / "residualCirc_config.toml"


def _patched_config(src_cfg: Path, output_dir: Path) -> Path:
    """Write a copy of src_cfg with outputDirectory → output_dir and processNumber = 1."""
    text = src_cfg.read_text()
    text = re.sub(
        r"^outputDirectory\s*=\s*'[^']*'",
        lambda _: f"outputDirectory = '{output_dir.as_posix()}'",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^processNumber\s*=.*$",
        "processNumber = 1",
        text,
        flags=re.MULTILINE,
    )
    text = text.replace("outDirSkip = true", "outDirSkip = false")
    cfg_path = output_dir / src_cfg.name
    cfg_path.write_text(text)
    return cfg_path


# ---------------------------------------------------------------------------
# Module-scoped fixtures — run each CLI once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rescirc_tmp(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rescirc")
    cfg = _patched_config(_RESCIRC_CFG, tmp)
    with patch.object(sys, "argv", ["tem-residual", str(cfg)]):
        run_residual()
    return tmp


@pytest.fixture(scope="module")
def theta_tmp(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("theta")
    cfg = _patched_config(_THETA_CFG, tmp)
    with patch.object(sys, "argv", ["tem-tTransport-theta", str(cfg)]):
        run_tracer_transport_theta()
    return tmp


@pytest.fixture(scope="module")
def press_tmp(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("press")
    cfg = _patched_config(_PRESS_CFG, tmp)
    with patch.object(sys, "argv", ["tem-tTransport-press", str(cfg)]):
        run_tracer_transport_press()
    return tmp


# ---------------------------------------------------------------------------
# run_residual
# ---------------------------------------------------------------------------

def test_rescirc_produces_nc_files(rescirc_tmp):
    assert list(rescirc_tmp.glob("*.nc")), "run_residual produced no .nc files"


def test_rescirc_correct_file_count(rescirc_tmp):
    n = len(list(rescirc_tmp.glob("*.nc")))
    assert n == 12, f"Expected 12 output files (4 per day × 3 days), got {n}"


def test_rescirc_output_has_expected_variables(rescirc_tmp):
    sample = sorted(rescirc_tmp.glob("*.nc"))[0]
    with xr.open_dataset(sample) as ds:
        for var in ["V_RES_STD", "W_RES_STD", "div_EPF"]:
            assert var in ds.data_vars, f"Missing variable {var!r}"


def test_rescirc_all_core_flag_accepted(tmp_path):
    """processNumber='all cores' string triggers cpu_count() branch, no crash."""
    text = _RESCIRC_CFG.read_text()
    text = re.sub(
        r"^outputDirectory\s*=\s*'[^']*'",
        f"outputDirectory = '{tmp_path}'",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^processNumber\s*=.*$",
        "processNumber = 'all cores'",
        text,
        flags=re.MULTILINE,
    )
    text = text.replace("outDirSkip = true", "outDirSkip = false")
    cfg = tmp_path / "all_cores.toml"
    cfg.write_text(text)
    with patch.object(sys, "argv", ["tem-residual", str(cfg)]):
        run_residual()
    assert list(tmp_path.glob("*.nc"))


def test_rescirc_bad_output_dir_exits(tmp_path):
    """Non-existent outputDirectory must trigger sys.exit(1)."""
    cfg = tmp_path / "bad.toml"
    text = _RESCIRC_CFG.read_text()
    text = re.sub(
        r"^outputDirectory\s*=\s*'[^']*'",
        "outputDirectory = '/nonexistent_dir_xyz'",
        text,
        flags=re.MULTILINE,
    )
    cfg.write_text(text)
    with patch.object(sys, "argv", ["tem-residual", str(cfg)]):
        with pytest.raises(SystemExit) as exc:
            run_residual()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# run_tracer_transport_theta
# ---------------------------------------------------------------------------

def test_theta_produces_nc_files(theta_tmp):
    assert list(theta_tmp.glob("*.nc")), "run_tracer_transport_theta produced no .nc files"


def test_theta_correct_file_count(theta_tmp):
    n = len(list(theta_tmp.glob("*.nc")))
    assert n == 3, f"Expected 3 output files (one per tracer-data day), got {n}"


def test_theta_output_has_expected_variables(theta_tmp):
    sample = sorted(theta_tmp.glob("*.nc"))[0]
    with xr.open_dataset(sample) as ds:
        for var in ["BA_m_lat", "BA_m_theta", "BA_divm_lat", "BA_divm_theta"]:
            assert var in ds.data_vars, f"Missing variable {var!r}"


def test_theta_bad_output_dir_exits(tmp_path):
    cfg = tmp_path / "bad.toml"
    text = _THETA_CFG.read_text()
    text = re.sub(
        r"^outputDirectory\s*=\s*'[^']*'",
        "outputDirectory = '/nonexistent_dir_xyz'",
        text,
        flags=re.MULTILINE,
    )
    cfg.write_text(text)
    with patch.object(sys, "argv", ["tem-tTransport-theta", str(cfg)]):
        with pytest.raises(SystemExit) as exc:
            run_tracer_transport_theta()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# run_tracer_transport_press
# ---------------------------------------------------------------------------

def test_press_produces_nc_files(press_tmp):
    assert list(press_tmp.glob("*.nc")), "run_tracer_transport_press produced no .nc files"


def test_press_correct_file_count(press_tmp):
    n = len(list(press_tmp.glob("*.nc")))
    assert n == 3, f"Expected 3 output files (one per tracer-data day), got {n}"


def test_press_output_has_expected_variables(press_tmp):
    sample = sorted(press_tmp.glob("*.nc"))[0]
    with xr.open_dataset(sample) as ds:
        for var in ["BA_m_lat", "BA_m_z", "BA_divm_lat", "BA_divm_z"]:
            assert var in ds.data_vars, f"Missing variable {var!r}"


def test_press_bad_output_dir_exits(tmp_path):
    cfg = tmp_path / "bad.toml"
    text = _PRESS_CFG.read_text()
    text = re.sub(
        r"^outputDirectory\s*=\s*'[^']*'",
        "outputDirectory = '/nonexistent_dir_xyz'",
        text,
        flags=re.MULTILINE,
    )
    cfg.write_text(text)
    with patch.object(sys, "argv", ["tem-tTransport-press", str(cfg)]):
        with pytest.raises(SystemExit) as exc:
            run_tracer_transport_press()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# run_tracer_transport: {tracerNames} substitution in outPrefix
# ---------------------------------------------------------------------------

def test_theta_tracer_names_substituted_in_prefix(theta_tmp):
    """outPrefix contains '{tracerNames}' → output files must start with 'BA'."""
    names = [f.name for f in theta_tmp.glob("*.nc")]
    assert all(n.startswith("BA") for n in names), \
        f"Expected all output files to start with 'BA', got: {names}"


def test_press_tracer_names_substituted_in_prefix(press_tmp):
    names = [f.name for f in press_tmp.glob("*.nc")]
    assert all(n.startswith("BA") for n in names), \
        f"Expected all output files to start with 'BA', got: {names}"


def test_tracer_names_not_a_list_exits(tmp_path):
    """tracerNames must be a list; a bare string should trigger sys.exit(1)."""
    cfg = tmp_path / "bad.toml"
    text = _PRESS_CFG.read_text()
    text = re.sub(
        r"^outputDirectory\s*=\s*'[^']*'",
        f"outputDirectory = '{tmp_path}'",
        text,
        flags=re.MULTILINE,
    )
    # Replace list with a bare string
    text = re.sub(r"tracerNames\s*=\s*\[[^\]]*\]", "tracerNames = 'BA'", text)
    cfg.write_text(text)
    with patch.object(sys, "argv", ["tem-tTransport-press", str(cfg)]):
        with pytest.raises(SystemExit) as exc:
            run_tracer_transport_press()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# run_residual: reqVars branch when verticalDimensionType != 'other'
# ---------------------------------------------------------------------------

def _toml_value(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)


def _rescirc_cfg_with_overrides(tmp_path, **overrides):
    """Write a copy of the residualCirc config with arbitrary key overrides."""
    text = _RESCIRC_CFG.read_text()
    text = re.sub(
        r"^outputDirectory\s*=\s*'[^']*'",
        f"outputDirectory = '{tmp_path}'",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^processNumber\s*=.*$", "processNumber = 1", text, flags=re.MULTILINE)
    text = text.replace("outDirSkip = true", "outDirSkip = false")
    for key, value in overrides.items():
        text = re.sub(rf"^{key}\s*=.*$", f"{key} = {_toml_value(value)}", text, flags=re.MULTILINE)
    cfg = tmp_path / "custom.toml"
    cfg.write_text(text)
    return cfg


def _run_residual_with_mock_pool(cfg):
    """Run run_residual() with a no-op Pool and a no-op reporter thread."""
    mock_thread = MagicMock()
    mock_thread.daemon = True
    with patch.object(sys, "argv", ["tem-residual", str(cfg)]):
        with patch('multiprocessing.Pool') as mock_pool:
            mock_pool.return_value.__enter__ = lambda s: s
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)
            mock_pool.return_value.starmap = MagicMock()
            with patch('threading.Thread', return_value=mock_thread):
                run_residual()


def test_rescirc_reqvars_non_other_vertdim(tmp_path):
    """verticalDimensionType != 'other' → reqVars omits pressureName (lines 50-51)."""
    cfg = _rescirc_cfg_with_overrides(tmp_path, verticalDimensionType='log-pressure')
    _run_residual_with_mock_pool(cfg)
    mock_pool_call = True  # reached without error


def test_rescirc_reqvars_missing_vertwind(tmp_path):
    """verticalWindType='missing' → reqVars omits verticalWindName (lines 52-54)."""
    cfg = _rescirc_cfg_with_overrides(tmp_path, verticalWindType='missing')
    _run_residual_with_mock_pool(cfg)


def test_rescirc_reqvars_non_other_missing_wind(tmp_path):
    """verticalDimensionType != 'other' AND verticalWindType='missing' → else branch (lines 55-57)."""
    cfg = _rescirc_cfg_with_overrides(
        tmp_path, verticalDimensionType='log-pressure', verticalWindType='missing'
    )
    _run_residual_with_mock_pool(cfg)


# ---------------------------------------------------------------------------
# run_residual: monthly / daily outputTemporalMean chunking (lines 85, 87)
# ---------------------------------------------------------------------------

def test_rescirc_monthly_temporal_mean(tmp_path):
    """outputTemporalMean='monthly' exercises line 85 (groupby 'MS')."""
    cfg = _rescirc_cfg_with_overrides(tmp_path, outputTemporalMean='monthly')
    with patch.object(sys, "argv", ["tem-residual", str(cfg)]):
        run_residual()
    produced = list(tmp_path.glob("*.nc"))
    # sample data all in Jan → 1 monthly output file
    assert produced, "Expected at least one monthly output file"


def test_rescirc_daily_temporal_mean(tmp_path):
    """outputTemporalMean='daily' exercises line 87 (groupby 'D')."""
    cfg = _rescirc_cfg_with_overrides(tmp_path, outputTemporalMean='daily')
    with patch.object(sys, "argv", ["tem-residual", str(cfg)]):
        run_residual()
    produced = list(tmp_path.glob("*.nc"))
    # sample data has 3 days → 3 daily output files
    assert produced, "Expected at least one daily output file"


# ---------------------------------------------------------------------------
# run_residual: missing timestamps print block (lines 76-79)
# ---------------------------------------------------------------------------

def test_rescirc_missing_timestamps_printed(tmp_path, capsys):
    """When collectFileNames returns a non-empty missingTimeStamps, lines 76-79 run."""
    import tem_pkg.cli as cli_module
    missing_ts = pd.DatetimeIndex(['2000-01-02 00:00:00'])
    fake_paths = pd.DataFrame(
        {'Path': [str(_REPO_ROOT / 'tests/data/sample_input/ERA5/era5_sample_00010100.nc')]},
        index=pd.DatetimeIndex(['2000-01-01T00:00:00']),
    )
    cfg = _rescirc_cfg_with_overrides(tmp_path)
    mock_thread = MagicMock()
    mock_thread.daemon = True

    with patch.object(sys, "argv", ["tem-residual", str(cfg)]):
        with patch.object(cli_module, 'collectFileNames',
                          return_value=(fake_paths, missing_ts, '6h')):
            with patch('multiprocessing.Pool') as mock_pool:
                mock_pool.return_value.__enter__ = lambda s: s
                mock_pool.return_value.__exit__ = MagicMock(return_value=False)
                mock_pool.return_value.starmap = MagicMock()
                with patch('threading.Thread', return_value=mock_thread):
                    run_residual()

    out = capsys.readouterr().out
    assert "Missing timestamps" in out


# ---------------------------------------------------------------------------
# run_residual: Pool exception → sys.exit(1) (lines 111-113)
# ---------------------------------------------------------------------------

def test_rescirc_pool_exception_exits(tmp_path):
    """If Pool.starmap raises, run_residual prints error and exits with code 1."""
    cfg = _rescirc_cfg_with_overrides(tmp_path)
    mock_thread = MagicMock()
    mock_thread.daemon = True

    with patch.object(sys, "argv", ["tem-residual", str(cfg)]):
        with patch('multiprocessing.Pool') as mock_pool:
            mock_pool.return_value.__enter__ = lambda s: s
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)
            mock_pool.return_value.starmap = MagicMock(side_effect=RuntimeError("pool boom"))
            with patch('threading.Thread', return_value=mock_thread):
                with pytest.raises(SystemExit) as exc:
                    run_residual()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# run_tracer_transport_theta: reqVars branches (lines 257, 261)
# ---------------------------------------------------------------------------

def _theta_cfg_with_overrides(tmp_path, **overrides):
    text = _THETA_CFG.read_text()
    text = re.sub(
        r"^outputDirectory\s*=\s*'[^']*'",
        f"outputDirectory = '{tmp_path}'",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^processNumber\s*=.*$", "processNumber = 1", text, flags=re.MULTILINE)
    text = text.replace("outDirSkip = true", "outDirSkip = false")
    for key, value in overrides.items():
        text = re.sub(rf"^{key}\s*=.*$", f"{key} = {_toml_value(value)}", text, flags=re.MULTILINE)
    cfg = tmp_path / "custom_theta.toml"
    cfg.write_text(text)
    return cfg


def _run_theta_with_mock_pool(cfg):
    """Run run_tracer_transport_theta() with a no-op Pool and reporter thread."""
    mock_thread = MagicMock()
    mock_thread.daemon = True
    with patch.object(sys, "argv", ["tem-tTransport-theta", str(cfg)]):
        with patch('multiprocessing.Pool') as mock_pool:
            mock_pool.return_value.__enter__ = lambda s: s
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)
            mock_pool.return_value.starmap = MagicMock()
            with patch('threading.Thread', return_value=mock_thread):
                run_tracer_transport_theta()


def test_theta_reqvars_theta_vertdim(tmp_path):
    """verticalDimensionType='theta' → reqVars omits thetaName (lines 257-258)."""
    cfg = _theta_cfg_with_overrides(tmp_path, verticalDimensionType='theta', vertDim='theta')
    _run_theta_with_mock_pool(cfg)


def test_theta_tracer_in_met_reqvars(tmp_path):
    """tracerDataInMetFiles=True → tracerNames appended to reqVars (line 261)."""
    cfg = _theta_cfg_with_overrides(tmp_path, tracerDataInMetFiles=True)
    _run_theta_with_mock_pool(cfg)


# ---------------------------------------------------------------------------
# run_tracer_transport_press: reqVars branches (lines 282, 286)
# ---------------------------------------------------------------------------

def _press_cfg_with_overrides(tmp_path, **overrides):
    text = _PRESS_CFG.read_text()
    text = re.sub(
        r"^outputDirectory\s*=\s*'[^']*'",
        f"outputDirectory = '{tmp_path}'",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^processNumber\s*=.*$", "processNumber = 1", text, flags=re.MULTILINE)
    text = text.replace("outDirSkip = true", "outDirSkip = false")
    for key, value in overrides.items():
        text = re.sub(rf"^{key}\s*=.*$", f"{key} = {_toml_value(value)}", text, flags=re.MULTILINE)
    cfg = tmp_path / "custom_press.toml"
    cfg.write_text(text)
    return cfg


def _run_press_with_mock_pool(cfg):
    """Run run_tracer_transport_press() with a no-op Pool and reporter thread."""
    mock_thread = MagicMock()
    mock_thread.daemon = True
    with patch.object(sys, "argv", ["tem-tTransport-press", str(cfg)]):
        with patch('multiprocessing.Pool') as mock_pool:
            mock_pool.return_value.__enter__ = lambda s: s
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)
            mock_pool.return_value.starmap = MagicMock()
            with patch('threading.Thread', return_value=mock_thread):
                run_tracer_transport_press()


def test_press_reqvars_non_other_vertdim(tmp_path):
    """verticalDimensionType='log-pressure' → reqVars omits pressureName (lines 282-283)."""
    cfg = _press_cfg_with_overrides(tmp_path, verticalDimensionType='log-pressure')
    _run_press_with_mock_pool(cfg)


def test_press_tracer_in_met_reqvars(tmp_path):
    """tracerDataInMetFiles=True → tracerNames appended to reqVars (line 286)."""
    cfg = _press_cfg_with_overrides(tmp_path, tracerDataInMetFiles=True)
    _run_press_with_mock_pool(cfg)


# ---------------------------------------------------------------------------
# run_tracer_transport: tracerDataInMetFiles=False branch (lines 172-204, 224-230)
# The default theta config already uses tracerDataInMetFiles=False, so the
# module-scoped theta_tmp fixture exercises this path end-to-end.
# Here we exercise the missing-timestamps print sub-branch (lines 191-199).
# ---------------------------------------------------------------------------

def test_tracer_transport_separate_missing_printed(tmp_path, capsys):
    """tracerDataInMetFiles=False: missing timestamps print block (lines 191-199)."""
    import tem_pkg.cli as cli_module
    missing_ts = ['2000-01-02 00:00:00']
    met_paths = pd.DataFrame(
        {'Path': [str(_REPO_ROOT / 'tests/data/sample_input/ERA5/era5_sample_00010100.nc')]},
        index=pd.DatetimeIndex(['2000-01-01T00:00:00']),
    )
    tracer_paths = met_paths.copy()
    mock_thread = MagicMock()
    mock_thread.daemon = True
    cfg = _theta_cfg_with_overrides(tmp_path, tracerDataInMetFiles=False)

    with patch.object(sys, "argv", ["tem-tTransport-theta", str(cfg)]):
        with patch.object(cli_module, 'collectFileNamesTTransport',
                          side_effect=[(met_paths, missing_ts, '6h'),
                                       (tracer_paths, missing_ts, '6h')]):
            with patch.object(cli_module, 'chunkMetFilesPathsForBinning',
                              return_value={pd.Timestamp('2000-01-01'): (str(_REPO_ROOT / 'tests/data/sample_input/ERA5/era5_sample_00010100.nc'), [], [])}):
                with patch('multiprocessing.Pool') as mock_pool:
                    mock_pool.return_value.__enter__ = lambda s: s
                    mock_pool.return_value.__exit__ = MagicMock(return_value=False)
                    mock_pool.return_value.starmap = MagicMock()
                    with patch('threading.Thread', return_value=mock_thread):
                        run_tracer_transport_theta()

    out = capsys.readouterr().out
    assert "Missing timestamps" in out


# ---------------------------------------------------------------------------
# run_tracer_transport: tracerDataInMetFiles=True Pool branch (lines 153-170, 216-222)
# ---------------------------------------------------------------------------

def test_theta_tracer_in_met_pool_branch(tmp_path):
    """tracerDataInMetFiles=True uses the pathsAndTime Pool branch (lines 215-222)."""
    cfg = _theta_cfg_with_overrides(tmp_path, tracerDataInMetFiles=True)
    _run_theta_with_mock_pool(cfg)


def test_tracer_transport_in_met_missing_printed(tmp_path, capsys):
    """tracerDataInMetFiles=True: missing timestamps print block (lines 163-167)."""
    import tem_pkg.cli as cli_module
    missing_ts = pd.DatetimeIndex(['2000-01-02 00:00:00'])
    fake_paths = pd.DataFrame(
        {'Path': [str(_REPO_ROOT / 'tests/data/sample_input/ERA5/era5_sample_00010100.nc')]},
        index=pd.DatetimeIndex(['2000-01-01T00:00:00']),
    )
    cfg = _theta_cfg_with_overrides(tmp_path, tracerDataInMetFiles=True)
    mock_thread = MagicMock()
    mock_thread.daemon = True

    with patch.object(sys, "argv", ["tem-tTransport-theta", str(cfg)]):
        with patch.object(cli_module, 'collectFileNames',
                          return_value=(fake_paths, missing_ts, '6h')):
            with patch('multiprocessing.Pool') as mock_pool:
                mock_pool.return_value.__enter__ = lambda s: s
                mock_pool.return_value.__exit__ = MagicMock(return_value=False)
                mock_pool.return_value.starmap = MagicMock()
                with patch('threading.Thread', return_value=mock_thread):
                    run_tracer_transport_theta()

    out = capsys.readouterr().out
    assert "Missing timestamps" in out


# ---------------------------------------------------------------------------
# run_tracer_transport: Pool exception → sys.exit(1) (lines 232-234)
# ---------------------------------------------------------------------------

def test_tracer_transport_pool_exception_exits(tmp_path):
    """If Pool.starmap raises, run_tracer_transport exits with code 1."""
    cfg = _theta_cfg_with_overrides(tmp_path, tracerDataInMetFiles=True)
    mock_thread = MagicMock()
    mock_thread.daemon = True

    with patch.object(sys, "argv", ["tem-tTransport-theta", str(cfg)]):
        with patch('multiprocessing.Pool') as mock_pool:
            mock_pool.return_value.__enter__ = lambda s: s
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)
            mock_pool.return_value.starmap = MagicMock(side_effect=RuntimeError("pool boom"))
            with patch('threading.Thread', return_value=mock_thread):
                with pytest.raises(SystemExit) as exc:
                    run_tracer_transport_theta()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# run_tracer_transport_theta: 'all cores' flag (line 250)
# ---------------------------------------------------------------------------

def test_theta_all_cores_flag(tmp_path):
    """processNumber='all cores' triggers cpu_count() branch (line 250)."""
    text = _THETA_CFG.read_text()
    text = re.sub(
        r"^outputDirectory\s*=\s*'[^']*'",
        f"outputDirectory = '{tmp_path}'",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^processNumber\s*=.*$", "processNumber = 'all cores'", text, flags=re.MULTILINE)
    text = text.replace("outDirSkip = true", "outDirSkip = false")
    cfg = tmp_path / "all_cores.toml"
    cfg.write_text(text)
    with patch.object(sys, "argv", ["tem-tTransport-theta", str(cfg)]):
        run_tracer_transport_theta()
    assert list(tmp_path.glob("*.nc"))
