import pytest

from transformed_eulerian_mean.parser import residual_circ_parser, tTransport_theta_parser, tTransport_press_parser

_CFG = "from config file"


# ── residual_circ_parser ──────────────────────────────────────────────────────

def test_residual_parser_returns_parser():
    p = residual_circ_parser()
    assert p is not None


def test_residual_parser_config_file_required():
    p = residual_circ_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_residual_parser_config_file_positional():
    p = residual_circ_parser()
    args = p.parse_args(["my_config.toml"])
    assert args.configFile == "my_config.toml"


def test_residual_parser_defaults_to_cfg_sentinel():
    p = residual_circ_parser()
    args = p.parse_args(["cfg.toml"])
    assert args.outputDirectory == _CFG
    assert args.inFileNames == _CFG
    assert args.processNumber == _CFG
    assert args.startDate == _CFG
    assert args.endDate == _CFG


def test_residual_parser_override_output_directory():
    p = residual_circ_parser()
    args = p.parse_args(["cfg.toml", "--outputDirectory", "/some/dir"])
    assert args.outputDirectory == "/some/dir"


def test_residual_parser_override_process_number():
    p = residual_circ_parser()
    args = p.parse_args(["cfg.toml", "--processNumber", "4"])
    assert args.processNumber == "4"


def test_residual_parser_override_dates():
    p = residual_circ_parser()
    args = p.parse_args(["cfg.toml", "--startDate", "2010-01-01-00", "--endDate", "2010-12-31-18"])
    assert args.startDate == "2010-01-01-00"
    assert args.endDate == "2010-12-31-18"


def test_residual_parser_residual_specific_args_default():
    p = residual_circ_parser()
    args = p.parse_args(["cfg.toml"])
    assert args.inputPath == _CFG
    assert args.inputPathType == _CFG
    assert args.saveEddyTerms == _CFG
    assert args.saveInterpolatedZonalMean == _CFG
    assert args.saveZonalMean == _CFG
    assert args.outputTemporalMean == _CFG
    assert args.hoursToKeep == _CFG


def test_residual_parser_override_save_eddy_terms():
    p = residual_circ_parser()
    args = p.parse_args(["cfg.toml", "--saveEddyTerms", "true"])
    assert args.saveEddyTerms == "true"


def test_residual_parser_override_output_temporal_mean():
    p = residual_circ_parser()
    args = p.parse_args(["cfg.toml", "--outputTemporalMean", "monthly"])
    assert args.outputTemporalMean == "monthly"


def test_residual_parser_vertical_dimension_default():
    p = residual_circ_parser()
    args = p.parse_args(["cfg.toml"])
    assert args.verticalDimensionType == _CFG


def test_residual_parser_all_common_args_default():
    p = residual_circ_parser()
    args = p.parse_args(["cfg.toml"])
    for attr in ["timeInfoInFileNames", "outPrefix", "targetLevels", "FourierTransform",
                 "Waves", "outDirSkip", "temperatureType", "temperatureName",
                 "verticalWindType", "pressureName", "zonalWindName",
                 "meridionalWindName", "verticalWindName", "latDim", "lonDim",
                 "vertDim", "timeDim"]:
        assert getattr(args, attr) == _CFG


# ── tTransport_theta_parser ───────────────────────────────────────────────────

def test_theta_parser_returns_parser():
    p = tTransport_theta_parser()
    assert p is not None


def test_theta_parser_config_file_required():
    p = tTransport_theta_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_theta_parser_config_file_positional():
    p = tTransport_theta_parser()
    args = p.parse_args(["cfg.toml"])
    assert args.configFile == "cfg.toml"


def test_theta_parser_common_args_default():
    p = tTransport_theta_parser()
    args = p.parse_args(["cfg.toml"])
    assert args.outputDirectory == _CFG
    assert args.processNumber == _CFG


def test_theta_parser_tracer_args_default():
    p = tTransport_theta_parser()
    args = p.parse_args(["cfg.toml"])
    assert args.tracerDataInMetFiles == _CFG
    assert args.inputDirectory == _CFG
    assert args.tracerNames == _CFG
    assert args.sinksSources == _CFG
    assert args.MetDataBinningTime == _CFG
    assert args.tracerInputDirectory == _CFG
    assert args.tracerInFileNames == _CFG
    assert args.tracerTimeInfoInFileNames == _CFG
    assert args.binningLat == _CFG
    assert args.binningLon == _CFG


def test_theta_parser_override_tracer_names():
    p = tTransport_theta_parser()
    args = p.parse_args(["cfg.toml", "--tracerNames", "O3"])
    assert args.tracerNames == "O3"


def test_theta_parser_override_binning():
    p = tTransport_theta_parser()
    args = p.parse_args(["cfg.toml", "--binningLat", "2", "--binningLon", "4"])
    assert args.binningLat == "2"
    assert args.binningLon == "4"


# ── tTransport_press_parser ───────────────────────────────────────────────────

def test_press_parser_returns_parser():
    p = tTransport_press_parser()
    assert p is not None


def test_press_parser_config_file_required():
    p = tTransport_press_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_press_parser_config_file_positional():
    p = tTransport_press_parser()
    args = p.parse_args(["cfg.toml"])
    assert args.configFile == "cfg.toml"


def test_press_parser_common_args_default():
    p = tTransport_press_parser()
    args = p.parse_args(["cfg.toml"])
    assert args.outputDirectory == _CFG
    assert args.processNumber == _CFG


def test_press_parser_tracer_args_default():
    p = tTransport_press_parser()
    args = p.parse_args(["cfg.toml"])
    assert args.tracerDataInMetFiles == _CFG
    assert args.tracerNames == _CFG
    assert args.sinksSources == _CFG


def test_press_parser_override_output_directory():
    p = tTransport_press_parser()
    args = p.parse_args(["cfg.toml", "--outputDirectory", "/out"])
    assert args.outputDirectory == "/out"


def test_press_parser_override_fourier():
    p = tTransport_press_parser()
    args = p.parse_args(["cfg.toml", "--FourierTransform", "true", "--Waves", "1-3"])
    assert args.FourierTransform == "true"
    assert args.Waves == "1-3"


def test_press_parser_tracer_vertical_args_default():
    p = tTransport_press_parser()
    args = p.parse_args(["cfg.toml"])
    assert args.tracerVerticalDimensionType == _CFG
    assert args.tracerPressureName == _CFG
    assert args.tracerLatDim == _CFG
    assert args.tracerLonDim == _CFG
    assert args.tracerVertDim == _CFG
    assert args.tracerTimeDim == _CFG
