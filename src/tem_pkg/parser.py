from __future__ import annotations

import argparse

_CFG = 'from config file'


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Register CLI arguments shared by all three entry-point tools.

    Adds arguments for output directory, input file discovery, time filtering,
    vertical coordinate configuration, variable names, Fourier options, and
    parallelism. All arguments default to ``'from config file'`` so that the
    TOML config is the primary source and CLI flags are optional overrides.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to which arguments are added in-place.
    """
    parser.add_argument("--outputDirectory",
                        default=_CFG,
                        help="directory where results are stored.")

    parser.add_argument("--inFileNames",
                        default=_CFG,
                        help="names of input files; rglob is used, e.g. '*[0-9].nc'.")

    parser.add_argument("--timeInfoInFileNames",
                        default=_CFG,
                        help="position of time information in file names. "
                             "Accepts YYYY (or YY), MM, DD, HH, mm, ss. "
                             "Examples: '*YYMMDDHH???' for era5_10120100.nc")

    parser.add_argument("--outPrefix",
                        default=_CFG,
                        help="prefix prepended to every output file name.")

    parser.add_argument("--targetLevels",
                        default=_CFG,
                        help="vertical levels for interpolation "
                             "(K for theta-coordinates; km for log-pressure). "
                             "Homogeneous array required (all floats or all integers).")

    parser.add_argument("--FourierTransform",
                        default=_CFG,
                        help="perform Fourier decomposition of eddy flux terms. "
                             "Accepts true or false.")

    parser.add_argument("--Waves",
                        default=_CFG,
                        help="wavenumber bands to save after Fourier transformation, "
                             "e.g. ['1', '2', '6-10', '21-end']. "
                             "Only used if FourierTransform = true.")

    parser.add_argument("--processNumber",
                        default=_CFG,
                        help="number of CPU cores to use. "
                             "Accepts 'all cores' or a positive integer.")

    parser.add_argument("--startDate",
                        default=_CFG,
                        help="exclude files before this date. "
                             "Accepts YYYY-MM-DD-HH format, or '' to disable.")

    parser.add_argument("--endDate",
                        default=_CFG,
                        help="exclude files after this date (inclusive). "
                             "Same format as startDate.")

    parser.add_argument("--outDirSkip",
                        default=_CFG,
                        help="skip timestamps for which an output file already exists. "
                             "Accepts true or false.")

    parser.add_argument("--verticalDimensionType",
                        default=_CFG,
                        help="vertical coordinate type of the input files. "
                             "Accepts 'pressure', 'log-pressure', or 'other'.")

    parser.add_argument("--temperatureType",
                        default=_CFG,
                        help="temperature variable type in the input files. "
                             "Accepts 'theta' or 'temperature'.")

    parser.add_argument("--temperatureName",
                        default=_CFG,
                        help="name of the temperature variable in the input file.")

    parser.add_argument("--verticalWindType",
                        default=_CFG,
                        help="vertical wind variable type in the input files. "
                             "Accepts 'W', 'omega', or 'missing'.")

    parser.add_argument("--pressureName",
                        default=_CFG,
                        help="name of the 3-D pressure variable in the input file. "
                             "Not required if verticalDimensionType is 'pressure' or 'log-pressure'.")

    parser.add_argument("--zonalWindName",
                        default=_CFG,
                        help="name of the zonal wind variable in the input file.")

    parser.add_argument("--meridionalWindName",
                        default=_CFG,
                        help="name of the meridional wind variable in the input file.")

    parser.add_argument("--verticalWindName",
                        default=_CFG,
                        help="name of the vertical wind variable in the input file.")

    parser.add_argument("--latDim",
                        default=_CFG,
                        help="name of the latitude dimension in the input file.")

    parser.add_argument("--lonDim",
                        default=_CFG,
                        help="name of the longitude dimension in the input file.")

    parser.add_argument("--vertDim",
                        default=_CFG,
                        help="name of the vertical dimension in the input file.")

    parser.add_argument("--timeDim",
                        default=_CFG,
                        help="name of the time dimension in the input file. "
                             "Set to '' if files have no time dimension.")


def _add_tracer_args(parser: argparse.ArgumentParser) -> None:
    """
    Register CLI arguments shared by both tracer-transport entry-point tools.

    Adds arguments for tracer variable names, sink/source terms, met-data
    temporal binning, and separate tracer file paths and dimension names.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to which arguments are added in-place.
    """
    parser.add_argument("--tracerDataInMetFiles",
                        default=_CFG,
                        help="set to true if tracer and meteorological data are in the same files, "
                             "false if they are in separate files.")

    parser.add_argument("--inputDirectory",
                        default=_CFG,
                        help="directory where input (meteorological) data is located.")

    parser.add_argument("--binningLat",
                        default=_CFG,
                        help="number of adjacent latitude points to average into one. "
                             "Set to 1 to keep the original resolution.")

    parser.add_argument("--binningLon",
                        default=_CFG,
                        help="number of adjacent longitude points to average into one. "
                             "Set to 1 to keep the original resolution.")

    parser.add_argument("--tracerNames",
                        default=_CFG,
                        help="names of the tracers as they appear in the input data files.")

    parser.add_argument("--sinksSources",
                        default=_CFG,
                        help="sinks and sources for each tracer. "
                             "List entries correspond to tracerNames in order.")

    parser.add_argument("--MetDataBinningTime",
                        default=_CFG,
                        help="temporal binning of meteorological data to match tracer data frequency. "
                             "Accepts 'auto' or a positive integer.")

    parser.add_argument("--tracerVerticalDimensionType",
                        default=_CFG,
                        help="vertical coordinate type of the tracer input files. "
                             "Accepts 'pressure', 'log-pressure', or 'other'.")

    parser.add_argument("--tracerPressureName",
                        default=_CFG,
                        help="name of the pressure variable in the tracer input file. "
                             "Not required if tracerVerticalDimensionType is 'pressure' or 'log-pressure'.")

    parser.add_argument("--tracerLatDim",
                        default=_CFG,
                        help="name of the latitude dimension in the tracer input file.")

    parser.add_argument("--tracerLonDim",
                        default=_CFG,
                        help="name of the longitude dimension in the tracer input file.")

    parser.add_argument("--tracerVertDim",
                        default=_CFG,
                        help="name of the vertical dimension in the tracer input file.")

    parser.add_argument("--tracerTimeDim",
                        default=_CFG,
                        help="name of the time dimension in the tracer input file.")

    parser.add_argument("--tracerInputDirectory",
                        default=_CFG,
                        help="directory where tracer input data is located.")

    parser.add_argument("--tracerInFileNames",
                        default=_CFG,
                        help="names of tracer input files (same rglob rules as inFileNames).")

    parser.add_argument("--tracerTimeInfoInFileNames",
                        default=_CFG,
                        help="position of time information in tracer file names "
                             "(same format as timeInfoInFileNames).")


def residual_circ_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the residual-circulation tool.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``tem-residual-circ``, covering common arguments plus
        options specific to the residual-circulation calculation (input path
        type, eddy-term saving, temporal averaging).
    """
    parser = argparse.ArgumentParser(
        description="Compute residual mean meridional circulation and Eliassen-Palm flux diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("configFile",
                        help="path to the TOML configuration file.")

    _add_common_args(parser)

    parser.add_argument("--inputPath",
                        default=_CFG,
                        help="path where input data is located.")

    parser.add_argument("--inputPathType",
                        default=_CFG,
                        help="type of input path. Accepts 'directory' or 'txt'.")

    parser.add_argument("--inputDataDescription",
                        default=_CFG,
                        help="description of the input data, saved as a file attribute.")

    parser.add_argument("--hoursToKeep",
                        default=_CFG,
                        help="restrict input to these hours of the day, e.g. [0, 6, 12, 18]. "
                             "Leave as [] to keep all available timestamps.")

    parser.add_argument("--saveEddyTerms",
                        default=_CFG,
                        help="save eddy flux terms to the output file. "
                             "Some terms are 3-D and require significant storage. "
                             "Accepts true or false.")

    parser.add_argument("--saveInterpolatedZonalMean",
                        default=_CFG,
                        help="interpolate to targetLevels and save the zonal mean of these input variables.")

    parser.add_argument("--saveZonalMean",
                        default=_CFG,
                        help="save the zonal mean of these 2-D input variables directly.")

    parser.add_argument("--outputTemporalMean",
                        default=_CFG,
                        help="compute a temporal mean before saving output. "
                             "Accepts 'monthly', 'daily', or false.")

    return parser


def tTransport_theta_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the theta-coordinate tracer-transport tool.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``tem-tracer-transport-theta``, covering common arguments
        and tracer-specific arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute tracer transport diagnostics in potential temperature (theta) coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("configFile",
                        help="path to the TOML configuration file.")

    _add_common_args(parser)
    _add_tracer_args(parser)

    return parser


def tTransport_press_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the log-pressure tracer-transport tool.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``tem-tracer-transport-press``, covering common arguments
        and tracer-specific arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute tracer transport diagnostics in log-pressure altitude coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("configFile",
                        help="path to the TOML configuration file.")

    _add_common_args(parser)
    _add_tracer_args(parser)

    return parser


def wave_decomp_press_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the log-pressure stationary/transient wave decomposition tool.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``wave-decomp-press``, covering common arguments,
        tracer-specific arguments, and the EP-flux decomposition flag.
    """
    parser = argparse.ArgumentParser(
        description="Decompose eddy fluxes into stationary and transient contributions "
                    "in log-pressure altitude coordinates, covering EP-flux and tracer transport.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("configFile",
                        help="path to the TOML configuration file.")

    _add_common_args(parser)
    _add_tracer_args(parser)

    parser.add_argument("--computeEPF",
                        default=_CFG,
                        help="compute stationary/transient decomposition of the EP flux "
                             "and eddy heat flux. Requires zonalWindName to be set. "
                             "Accepts true or false.")

    return parser


def wave_decomp_theta_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the isentropic stationary/transient wave decomposition tool.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``wave-decomp-theta``, covering common arguments and
        tracer-specific arguments for isentropic coordinates.
    """
    parser = argparse.ArgumentParser(
        description="Decompose tracer eddy fluxes into stationary and transient contributions "
                    "in isentropic (theta) coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("configFile",
                        help="path to the TOML configuration file.")

    _add_common_args(parser)
    _add_tracer_args(parser)

    return parser
