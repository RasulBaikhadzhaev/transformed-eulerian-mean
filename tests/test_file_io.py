"""
Unit tests for tem_pkg.file_io covering previously uncovered paths.

Groups:
  - extractTimeFromFileNames: error path, non-wildcard YY prefix
  - _load_and_filter_file_paths (via collectFileNames): all filtering branches
  - collectFileNames: frequency-inference edge cases, outDirSkip
  - collectFileNamesTTransport: basic path, outDirSkip
  - chunkMetFilesPathsForBinning: auto same-freq no-overlap, int N, invalid
  - saveOut: no-Fourier, specific waves, 'all' waves, filename encoding, alt/theta vert coord
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tem_pkg.file_io import (
    _ALT_VERT_COORD,
    _THETA_VERT_COORD,
    chunkMetFilesPathsForBinning,
    collectFileNames,
    collectFileNamesTTransport,
    extractTimeFromFileNames,
    saveOut,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch(tmp_path, names):
    for name in names:
        (tmp_path / name).touch()


def _df(timestamps, paths=None):
    if paths is None:
        paths = [Path(f"f{i}.nc") for i in range(len(timestamps))]
    return pd.DataFrame({"Path": paths}, index=pd.DatetimeIndex(timestamps))


# ---------------------------------------------------------------------------
# extractTimeFromFileNames — error + non-wildcard YY paths
# ---------------------------------------------------------------------------

def test_extracttime_no_year_token_exits():
    with pytest.raises(SystemExit):
        extractTimeFromFileNames([Path("data_20230115.nc")], "*MMDD????")


def test_extracttime_yy_prefix_pre50():
    """Non-wildcard YY template: year < 50 → 20YY."""
    result = extractTimeFromFileNames([Path("230115_data.nc")], "YYMMDD")
    assert result[0].year == 2023


def test_extracttime_yy_prefix_post50():
    """Non-wildcard YY template: year >= 50 → 19YY."""
    result = extractTimeFromFileNames([Path("781220_data.nc")], "YYMMDD")
    assert result[0].year == 1978


# ---------------------------------------------------------------------------
# _load_and_filter_file_paths — via collectFileNames
# ---------------------------------------------------------------------------

def test_nonexistent_directory_exits(tmp_path):
    with pytest.raises(SystemExit):
        collectFileNames(str(tmp_path / "missing"), "*.nc", "*YYYYMMDD???")


def test_empty_directory_exits(tmp_path):
    with pytest.raises(SystemExit):
        collectFileNames(str(tmp_path), "*.nc", "*YYYYMMDD???")


_SAMPLE_ERA5_DIR = Path(__file__).parent / "data" / "sample_input" / "ERA5"
_ERA5_PATTERN    = "*YYYYMMDDHH???"


def test_txt_input_path_type(tmp_path):
    files = [tmp_path / f"era5_202001{d:02d}.nc" for d in [1, 2, 3]]
    for f in files:
        f.touch()
    txt = tmp_path / "list.txt"
    txt.write_text("\n".join(str(f) for f in files) + "\n")
    result, _, _ = collectFileNames(str(txt), "*.nc", "*YYYYMMDD???", inputPathType=".txt")
    assert len(result) == 3


def test_txt_empty_file_exits(tmp_path):
    txt = tmp_path / "empty.txt"
    txt.write_text("")
    with pytest.raises(SystemExit):
        collectFileNames(str(txt), "*.nc", "*YYYYMMDD???", inputPathType=".txt")


def test_txt_with_sample_era5_files(tmp_path):
    """Use real sample input files listed in a .txt — covers the txt success path.

    ERA5 sample filenames encode the date as YYMMDDHH (8 digits), e.g.
    era5_sample_00010100.nc = YY=00 → 2000, MM=01, DD=01, HH=00.
    """
    era5_files = sorted(_SAMPLE_ERA5_DIR.glob("era5_sample_*.nc"))
    txt = tmp_path / "era5_list.txt"
    txt.write_text("\n".join(str(f) for f in era5_files) + "\n")
    result, _, _ = collectFileNames(
        str(txt), "*.nc", "*YYMMDDHH???", inputPathType=".txt")
    assert len(result) == len(era5_files)


def test_date_start_filter(tmp_path):
    _touch(tmp_path, ["era5_20200101.nc", "era5_20200201.nc", "era5_20200301.nc"])
    result, _, _ = collectFileNames(str(tmp_path), "*.nc", "*YYYYMMDD???",
                                    dateStart="2020-02-01")
    assert len(result) == 2
    assert all(result.index >= pd.Timestamp("2020-02-01"))


def test_date_end_filter(tmp_path):
    _touch(tmp_path, ["era5_20200101.nc", "era5_20200201.nc", "era5_20200301.nc"])
    result, _, _ = collectFileNames(str(tmp_path), "*.nc", "*YYYYMMDD???",
                                    dateEnd="2020-02-01")
    assert len(result) == 2
    assert all(result.index <= pd.Timestamp("2020-02-01"))


def test_date_both_filters(tmp_path):
    _touch(tmp_path, ["era5_20200101.nc", "era5_20200201.nc", "era5_20200301.nc"])
    result, _, _ = collectFileNames(str(tmp_path), "*.nc", "*YYYYMMDD???",
                                    dateStart="2020-02-01", dateEnd="2020-02-01")
    assert len(result) == 1


def test_date_filter_empty_exits(tmp_path):
    _touch(tmp_path, ["era5_20200101.nc", "era5_20200201.nc"])
    with pytest.raises(SystemExit):
        collectFileNames(str(tmp_path), "*.nc", "*YYYYMMDD???",
                         dateStart="2021-01-01")


def test_hours_to_keep_filter(tmp_path):
    _touch(tmp_path, [
        "era5_2020010100.nc", "era5_2020010106.nc",
        "era5_2020010112.nc", "era5_2020010118.nc",
    ])
    result, _, _ = collectFileNames(str(tmp_path), "*.nc", "*YYYYMMDDHH???",
                                    hoursToKeep=[0, 12])
    assert sorted(result.index.hour.tolist()) == [0, 12]


def test_hours_to_keep_empty_exits(tmp_path):
    _touch(tmp_path, ["era5_2020010106.nc", "era5_2020010118.nc"])
    with pytest.raises(SystemExit):
        collectFileNames(str(tmp_path), "*.nc", "*YYYYMMDDHH???",
                         hoursToKeep=[0, 12])


# ---------------------------------------------------------------------------
# collectFileNames — frequency-inference edge cases
# ---------------------------------------------------------------------------

def test_fewer_than_3_files_returns_empty_freq(tmp_path):
    _touch(tmp_path, ["era5_20200101.nc", "era5_20200201.nc"])
    _, missing, freq = collectFileNames(str(tmp_path), "*.nc", "*YYYYMMDD???")
    assert freq == ""
    assert isinstance(missing, pd.DataFrame)


def test_monthly_files_infers_non_empty_freq(tmp_path):
    _touch(tmp_path, [
        "era5_20200101.nc", "era5_20200201.nc", "era5_20200301.nc",
        "era5_20200401.nc", "era5_20200501.nc",
    ])
    _, _, freq = collectFileNames(str(tmp_path), "*.nc", "*YYYYMMDD???")
    assert freq is not None and freq != ""


# ---------------------------------------------------------------------------
# collectFileNamesTTransport — basic path (no outDirSkip)
# ---------------------------------------------------------------------------

def test_collect_ttransport_basic(tmp_path):
    _touch(tmp_path, [
        "clams_20200101.nc", "clams_20200201.nc", "clams_20200301.nc",
    ])
    result, missing, freq = collectFileNamesTTransport(
        str(tmp_path), "*.nc", "*YYYYMMDD???")
    assert len(result) == 3
    assert isinstance(freq, pd.Timedelta)


# ---------------------------------------------------------------------------
# collectFileNames — outDirSkip
# ---------------------------------------------------------------------------

def test_collect_outDirSkip_removes_already_processed(tmp_path):
    """outDirSkip=True: timestamps that already have output files are excluded."""
    inp_dir = tmp_path / "inp"
    inp_dir.mkdir()
    _touch(inp_dir, ["era5_20200101.nc", "era5_20200201.nc", "era5_20200301.nc"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "myprefix_2020_01_01_00_00.nc").touch()

    result, _, _ = collectFileNames(
        str(inp_dir), "*.nc", "*YYYYMMDD???",
        outputDir=str(out_dir), outPrefix="myprefix_",
        outDirSkip=True,
    )
    assert len(result) == 2
    assert pd.Timestamp("2020-01-01") not in result.index


def test_collect_outDirSkip_all_done_exits(tmp_path):
    """outDirSkip=True: SystemExit when all timestamps are already processed."""
    inp_dir = tmp_path / "inp"
    inp_dir.mkdir()
    _touch(inp_dir, ["era5_20200101.nc", "era5_20200201.nc", "era5_20200301.nc"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for ts in ["2020_01_01_00_00", "2020_02_01_00_00", "2020_03_01_00_00"]:
        (out_dir / f"pfx_{ts}.nc").touch()

    with pytest.raises(SystemExit):
        collectFileNames(
            str(inp_dir), "*.nc", "*YYYYMMDD???",
            outputDir=str(out_dir), outPrefix="pfx_",
            outDirSkip=True,
        )


def test_collect_outDirSkip_ignored_for_monthly_mean(tmp_path):
    """outDirSkip is bypassed when outputTemporalMean='monthly'."""
    inp_dir = tmp_path / "inp"
    inp_dir.mkdir()
    _touch(inp_dir, ["era5_20200101.nc", "era5_20200201.nc", "era5_20200301.nc"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for ts in ["2020_01_01_00_00", "2020_02_01_00_00", "2020_03_01_00_00"]:
        (out_dir / f"pfx_{ts}.nc").touch()

    result, _, _ = collectFileNames(
        str(inp_dir), "*.nc", "*YYYYMMDD???",
        outputDir=str(out_dir), outPrefix="pfx_",
        outDirSkip=True, outputTemporalMean="monthly",
    )
    assert len(result) == 3


# ---------------------------------------------------------------------------
# collectFileNamesTTransport — outDirSkip
# ---------------------------------------------------------------------------

def test_collect_ttransport_outDirSkip_removes_already_processed(tmp_path):
    """outDirSkip=1: timestamps already in output dir are excluded."""
    inp_dir = tmp_path / "inp"
    inp_dir.mkdir()
    _touch(inp_dir, ["clams_20200101.nc", "clams_20200201.nc", "clams_20200301.nc"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "pfx_2020_01_01_00_00.nc").touch()

    result, _, _ = collectFileNamesTTransport(
        str(inp_dir), "*.nc", "*YYYYMMDD???",
        outputDir=str(out_dir), outPrefix="pfx_",
        outDirSkip=1,
    )
    assert len(result) == 2
    assert pd.Timestamp("2020-01-01") not in result.index


def test_collect_ttransport_outDirSkip_all_done_exits(tmp_path):
    """outDirSkip=1: SystemExit when all timestamps are already processed."""
    inp_dir = tmp_path / "inp"
    inp_dir.mkdir()
    _touch(inp_dir, ["clams_20200101.nc", "clams_20200201.nc", "clams_20200301.nc"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for ts in ["2020_01_01_00_00", "2020_02_01_00_00", "2020_03_01_00_00"]:
        (out_dir / f"pfx_{ts}.nc").touch()

    with pytest.raises(SystemExit):
        collectFileNamesTTransport(
            str(inp_dir), "*.nc", "*YYYYMMDD???",
            outputDir=str(out_dir), outPrefix="pfx_",
            outDirSkip=1,
        )


# ---------------------------------------------------------------------------
# chunkMetFilesPathsForBinning
# ---------------------------------------------------------------------------

def test_chunk_auto_same_freq_overlapping():
    """auto + same freq + shared timestamps → 1-to-1 join with weight=1."""
    times  = pd.date_range("2020-01-01", periods=3, freq="6h")
    tracer = _df(times)
    met    = _df(times, [Path(f"met{i}.nc") for i in range(3)])
    freq   = pd.Timedelta("6h")
    result = chunkMetFilesPathsForBinning(met, tracer, "auto", freq, freq)
    assert len(result) == 3
    for ts in times:
        assert result[ts][2] == 1


def test_chunk_auto_same_freq_no_overlap_two_met_files():
    """auto + same freq + no shared timestamps: straddling met files get weights.

    The code path (lines 227-234) requires tracer to have ≥ 2 timestamps so that
    tracerFilesPaths.index[1] is valid when computing the second weight.
    Met files straddle a tracer midpoint so exactly 2 fall inside the half-period window.
    """
    # tracer at T+0h and T+6h; met at T+3h and T+9h — no index overlap, same 6h freq
    tracer_times = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 06:00"])
    met_times    = pd.DatetimeIndex(["2020-01-01 03:00", "2020-01-01 09:00"])
    tracer = _df(tracer_times)
    met    = _df(met_times, [Path("met0.nc"), Path("met1.nc")])
    freq   = pd.Timedelta("6h")
    result = chunkMetFilesPathsForBinning(met, tracer, "auto", freq, freq)
    # The first tracer timestamp (00:00) has met0 (03:00) within ±3h window → size==1, skipped
    # The second tracer timestamp (06:00) has met0 (03:00) and met1 (09:00) within ±3h → size==2
    ts = pd.Timestamp("2020-01-01 06:00")
    assert ts in result
    assert len(result[ts][1]) == 2


def test_chunk_auto_different_freq_empty_window_skipped():
    """auto + different freq: tracer timestamps with no met in window are skipped."""
    tracer_times = pd.date_range("2020-01-01", periods=2, freq="D")
    met_times    = pd.date_range("2020-01-10", periods=4, freq="6h")
    tracer = _df(tracer_times)
    met    = _df(met_times, [Path(f"met{i}.nc") for i in range(4)])
    result = chunkMetFilesPathsForBinning(
        met, tracer, "auto", pd.Timedelta("1D"), pd.Timedelta("6h"))
    assert len(result) == 0


def test_chunk_int_n_nearest(tmp_path):
    """int MetDataBinningTime selects N nearest met files with equal weights."""
    tracer_times = pd.date_range("2020-01-01 06:00", periods=2, freq="6h")
    met_times    = pd.date_range("2020-01-01", periods=8, freq="2h")
    tracer = _df(tracer_times)
    met    = _df(met_times, [Path(f"met{i}.nc") for i in range(8)])
    result = chunkMetFilesPathsForBinning(
        met, tracer, 3, pd.Timedelta("6h"), pd.Timedelta("2h"))
    assert len(result) == 2
    for ts in tracer_times:
        assert len(result[ts][1]) == 3
        np.testing.assert_allclose(sum(result[ts][2]), 1.0)


def test_chunk_invalid_binning_time_exits():
    times  = pd.date_range("2020-01-01", periods=2, freq="6h")
    tracer = _df(times)
    met    = _df(times)
    with pytest.raises(SystemExit):
        chunkMetFilesPathsForBinning(
            met, tracer, "invalid", pd.Timedelta("6h"), pd.Timedelta("6h"))


# ---------------------------------------------------------------------------
# saveOut
# ---------------------------------------------------------------------------

@pytest.fixture
def out_cfg(tmp_path):
    return {"outputDirectory": str(tmp_path), "outPrefix": "test_", "Waves": [1, 2]}


def _scalar_data(ntheta=4, nlat=5):
    arr = np.arange(ntheta * nlat, dtype=np.float64).reshape(ntheta, nlat)
    return {"chi_bar": (arr, "chi long name", "m/s"),
            "psi_bar": (arr * 2.0, "psi long name", "m2/s")}


def test_saveout_creates_nc_file(tmp_path, out_cfg):
    lats   = np.linspace(-60, 60, 5)
    thetas = np.array([300.0, 350.0, 400.0, 450.0])
    saveOut(_scalar_data(), out_cfg, pd.Timestamp("2020-01-01 06:00"), lats, thetas)
    assert len(list(tmp_path.glob("*.nc"))) == 1


def test_saveout_variables_coords_attrs(tmp_path, out_cfg):
    lats   = np.linspace(-60, 60, 5)
    thetas = np.array([300.0, 350.0, 400.0, 450.0])
    saveOut(_scalar_data(), out_cfg, pd.Timestamp("2020-01-01 06:00"), lats, thetas)
    with xr.open_dataset(list(tmp_path.glob("*.nc"))[0]) as ds:
        assert {"chi_bar", "psi_bar"} <= set(ds.data_vars)
        assert {"theta", "lat", "time"} <= set(ds.coords)
        assert ds["chi_bar"].attrs["units"] == "m/s"
        assert ds["chi_bar"].attrs["long_name"] == "chi long name"
        assert ds.theta.attrs["units"] == "K"
        assert ds.lat.attrs["units"] == "degree_N"


def test_saveout_values_stored_as_float32(tmp_path, out_cfg):
    lats   = np.linspace(-60, 60, 5)
    thetas = np.array([300.0, 350.0, 400.0, 450.0])
    arr    = np.arange(20, dtype=np.float64).reshape(4, 5)
    saveOut({"chi_bar": (arr, "n", "u")}, out_cfg,
            pd.Timestamp("2020-01-01"), lats, thetas)
    with xr.open_dataset(list(tmp_path.glob("*.nc"))[0]) as ds:
        np.testing.assert_allclose(ds["chi_bar"].values, arr.astype(np.float32),
                                   rtol=1e-6)


def test_saveout_fourier_specific_waves(tmp_path, out_cfg):
    lats   = np.linspace(-60, 60, 5)
    thetas = np.array([300.0, 350.0, 400.0, 450.0])
    data   = {
        "chi_bar": (np.ones((4, 5)), "chi", "m/s"),
        "Fourier": {"chi_WN": (np.ones((4, 5, 2)), "chi wn", "m/s")},
    }
    saveOut(data, out_cfg, pd.Timestamp("2020-02-15"), lats, thetas)
    with xr.open_dataset(list(tmp_path.glob("*.nc"))[0]) as ds:
        assert "chi_WN" in ds.data_vars
        assert list(ds.coords["waveN"].values) == [1, 2]
        assert ds.waveN.attrs["long_name"] == "wave number"


def test_saveout_fourier_all_waves(tmp_path):
    cfg    = {"outputDirectory": str(tmp_path), "outPrefix": "out_", "Waves": ["all"]}
    lats   = np.linspace(-60, 60, 5)
    thetas = np.array([300.0, 350.0, 400.0, 450.0])
    data   = {
        "chi_bar": (np.ones((4, 5)), "chi", "m/s"),
        "Fourier": {"chi_WN": (np.ones((4, 5, 3)), "chi wn", "m/s")},
    }
    saveOut(data, cfg, pd.Timestamp("2020-03-01 12:00"), lats, thetas)
    with xr.open_dataset(list(tmp_path.glob("*.nc"))[0]) as ds:
        assert list(ds.coords["waveN"].values) == [1, 2, 3]


def test_saveout_filename_encodes_timestamp(tmp_path, out_cfg):
    lats   = np.linspace(-60, 60, 5)
    thetas = np.array([300.0, 350.0, 400.0, 450.0])
    saveOut(_scalar_data(), out_cfg, pd.Timestamp("2020-06-15 18:00"), lats, thetas)
    names = [f.name for f in tmp_path.glob("*.nc")]
    assert any("2020_06_15_18_00" in n for n in names)


def test_saveout_alt_vert_coord(tmp_path, out_cfg):
    lats = np.linspace(-60, 60, 5)
    alts = np.array([0.0, 737.5, 1137.6, 2000.0])
    saveOut(_scalar_data(), out_cfg, pd.Timestamp("2020-01-01"), lats, alts, _ALT_VERT_COORD)
    with xr.open_dataset(list(tmp_path.glob("*.nc"))[0]) as ds:
        assert "alt" in ds.coords
        assert "theta" not in ds.coords
        assert ds.alt.attrs["long_name"] == "Log-pressure altitude"
        assert ds.alt.attrs["units"] == "m"
        assert ds["chi_bar"].dims[0] == "alt"


def test_saveout_default_vert_coord_is_theta(tmp_path, out_cfg):
    lats   = np.linspace(-60, 60, 5)
    thetas = np.array([300.0, 350.0, 400.0, 450.0])
    saveOut(_scalar_data(), out_cfg, pd.Timestamp("2020-01-01"), lats, thetas)
    with xr.open_dataset(list(tmp_path.glob("*.nc"))[0]) as ds:
        assert "theta" in ds.coords
        assert ds.theta.attrs["units"] == "K"
