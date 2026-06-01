from __future__ import annotations

import multiprocessing
import sys
import threading
import time
import warnings
from collections.abc import Callable
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd

from .file_io import chunkMetFilesPathsForBinning, collectFileNames, collectFileNamesTTransport
from .utils import (
    is_equal_or_shorter_than_day,
    is_equal_or_shorter_than_month,
    load_and_merge_config,
    progress_reporter,
)

# Suppress expected warnings. 
# The warning "mean of empty slice" happens when zonal mean is calculated, it occurs due to absence of data at every
# longitudinal point in places like 1000 hPa level in antarctica. Other warning occurs during interpolation to pressure
# coordinates, if target pressure level is beyond input pressure values, ex when lowest pressure in input data is 0.0012 
# hPa and targeted is 0.001. In both cases nan value is assigned for the result.
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Interpolation point out of data bounds')
warnings.filterwarnings('ignore', message='divide by zero encountered in divide') 
np.seterr(divide='ignore')
# Ignore all 'overflow encountered' and 'invalid value encountered' warnings from NumPy
warnings.filterwarnings('ignore', message='overflow encountered')
warnings.filterwarnings('ignore', message='invalid value encountered') 

def run_residual() -> None:
    """
    Entry point for the ``tem-residual-circ`` CLI command.

    Parses CLI arguments, merges them with the TOML config, collects and
    filters input file paths, then dispatches per-chunk TEM calculations
    to a multiprocessing pool. Progress is reported to stdout via a
    daemon thread. Exits with code 1 on any unhandled exception.
    """
    timeStart = time.time()
    from .parser import residual_circ_parser
    from .residual_circulation import init_worker, mainCalcs

    # counter for progress
    sharedCounter = multiprocessing.Value('i', 0)

    parserArgs = residual_circ_parser().parse_args()
    config = load_and_merge_config(parserArgs)

    if config['processNumber'] in ['all cores', 'all', 'All']:
        config['processNumber'] = multiprocessing.cpu_count()
            
    if not Path(config['outputDirectory']).is_dir():
        print(f"ERROR: Directory '{config['outputDirectory']}' does not exist.\n\n"
                "Please specify existing directory to store output files")
        sys.exit(1)
    
    # create list of required variables    
    if config['verticalDimensionType'] == 'other' and config['verticalWindType'] != 'missing':
        reqVars = [config['pressureName'], config['temperatureName'], config['zonalWindName'],
                config['meridionalWindName'], config['verticalWindName']]
    elif config['verticalDimensionType'] != 'other' and config['verticalWindType'] != 'missing':
        reqVars = [config['temperatureName'], config['zonalWindName'],
                config['meridionalWindName'], config['verticalWindName']]
    elif config['verticalDimensionType'] == 'other' and config['verticalWindType'] == 'missing':
        reqVars = [config['pressureName'], config['temperatureName'], config['zonalWindName'],
                config['meridionalWindName']]
    else:
        reqVars = [config['temperatureName'], config['zonalWindName'],
                config['meridionalWindName']]
           
    saveInterpolatedZonalMeanVars = config['saveInterpolatedZonalMean']
    saveZonalMeanVars = config['saveZonalMean']

    pathsAndTime, missingTimeStamps, expectedFrequency = collectFileNames(inputDir=config['inputPath'], 
                                                                        fileNames=config['inFileNames'], 
                                                                        timeInfoInFileNames=config['timeInfoInFileNames'],  
                                                                        outputDir=config['outputDirectory'], 
                                                                        dateStart=config['startDate'], 
                                                                        dateEnd=config['endDate'], 
                                                                        outPrefix=config['outPrefix'], 
                                                                        outDirSkip=config['outDirSkip'],
                                                                        inputPathType=config['inputPathType'],
                                                                        hoursToKeep = config['hoursToKeep'],
                                                                        outputTemporalMean = config['outputTemporalMean'])

    if not missingTimeStamps.empty:
        print(f"Missing timestamps which are estimated from expected data frequency of '{expectedFrequency}':")
        for timeStamp in missingTimeStamps:
            print(timeStamp)
        print(f"Number of missing files which are expected from typical data frequency: {len(missingTimeStamps)}")
    
    numOfFiles = len(pathsAndTime)

    # chunk pathAndTime by month, day or instance.
    if str(config['outputTemporalMean']).lower() in ['monthly', 'month'] and is_equal_or_shorter_than_month(expectedFrequency):
        pathsAndTimeChunked = {date: group for date, group in pathsAndTime.groupby(pd.Grouper(freq='MS'))}
    elif str(config['outputTemporalMean']).lower() in ['daily', 'day'] and is_equal_or_shorter_than_day(expectedFrequency):
        pathsAndTimeChunked = {date: group for date, group in pathsAndTime.groupby(pd.Grouper(freq='D'))}
    else:
        pathsAndTimeChunked = {index: group for index, group in pathsAndTime.groupby(pathsAndTime.index)}

    reporter = threading.Thread(
            target=progress_reporter, 
            args=(sharedCounter, numOfFiles, timeStart)
        )
    reporter.daemon = True # Allows the program to exit if the thread is stuck
    reporter.start()

    try: 
        with multiprocessing.Pool(
            processes=config['processNumber'], 
            initializer=init_worker, 
            initargs=(sharedCounter, )
            ) as p:
            
            p.starmap(mainCalcs, zip(pathsAndTimeChunked.values(), 
                                    repeat(reqVars), 
                                    repeat(config), 
                                    repeat(saveInterpolatedZonalMeanVars), 
                                    repeat(saveZonalMeanVars)))
    except Exception as e:
        print(f"\nAn error occurred: {repr(e)}")
        sys.exit(1)

    else:
        reporter.join()


def run_tracer_transport(mainCalcs: Callable, init_worker: Callable, tomlConfig: dict, reqVars: list[str]) -> None:
    """
    Shared driver for both tracer-transport entry points.

    Handles file discovery (combined or separate met/tracer files), spawns a
    progress-reporter thread, and dispatches per-file calculations to a
    multiprocessing pool. Exits with code 1 on any unhandled exception.

    Parameters
    ----------
    mainCalcs : Callable
        Worker function dispatched to the pool (theta or press variant).
    init_worker : Callable
        Pool initializer that stores the shared counter in each worker.
    tomlConfig : dict
        Merged configuration dict (TOML + CLI overrides).
    reqVars : list[str]
        Names of meteorological variables to load from each input file.
    """
    timeStart = time.time()
    # counter for progress
    sharedCounter = multiprocessing.Value('i', 0)

    # if '{tracerNames}' exists in outPrefix replace it with names of the tracers.
    if '{tracerNames}' in tomlConfig['outPrefix']:
        if isinstance(tomlConfig['tracerNames'], list):
            tomlConfig['outPrefix'] = tomlConfig['outPrefix'].replace('{tracerNames}', '_'.join(tomlConfig['tracerNames']))
        else:
            print("ERROR: option 'tracerNames' should be list of strings.")
            sys.exit(1)
    
    if not Path(tomlConfig['outputDirectory']).is_dir():
        print(f"ERROR: Directory '{tomlConfig['outputDirectory']}' does not exist.\n\n"
                "Please specify existing directory to store output files")
        sys.exit(1)

    if tomlConfig['tracerDataInMetFiles']:
        # tracer and met data are in the same files
        pathsAndTime, missingTimeStamps, expectedFrequency = collectFileNames(tomlConfig['inputDirectory'],
                                                        tomlConfig['inFileNames'],
                                                        tomlConfig['timeInfoInFileNames'],
                                                        tomlConfig['outputDirectory'],
                                                        dateStart=tomlConfig['startDate'],
                                                        dateEnd=tomlConfig['endDate'],
                                                        outPrefix=tomlConfig['outPrefix'],
                                                        outDirSkip=tomlConfig['outDirSkip'])
        if not missingTimeStamps.empty:
            print(f"Missing timestamps which are estimated from expected data frequency of '{expectedFrequency}':")
            for timeStamp in missingTimeStamps:
                print(timeStamp)
            print(f"Number of missing files which are expected from typical data frequency: {len(missingTimeStamps)}")

        numOfFiles = len(pathsAndTime)
        numbers = list(range(len(pathsAndTime)))

    else:
        # met and tracer data are in different files; met data expected at same or higher temporal frequency
        metPathsAndTime, metMissing, metExpFreq = collectFileNamesTTransport(tomlConfig['inputDirectory'],
                                                        tomlConfig['inFileNames'],
                                                        tomlConfig['timeInfoInFileNames'],
                                                        tomlConfig['outputDirectory'],
                                                        dateStart=tomlConfig['startDate'],
                                                        dateEnd=tomlConfig['endDate'],
                                                        outPrefix=tomlConfig['outPrefix'],
                                                        outDirSkip=tomlConfig['outDirSkip'])

        tracerPathsAndTime, tracerMissing, tracerExpFreq = collectFileNamesTTransport(tomlConfig['tracerInputDirectory'],
                                                                tomlConfig['tracerInFileNames'],
                                                                tomlConfig['tracerTimeInfoInFileNames'],
                                                                tomlConfig['outputDirectory'],
                                                                dateStart=tomlConfig['startDate'],
                                                                dateEnd=tomlConfig['endDate'],
                                                                outPrefix=tomlConfig['outPrefix'],
                                                                outDirSkip=tomlConfig['outDirSkip'])
        if tracerMissing or metMissing:
            print(f"Missing timestamps which are estimated from expected data frequency of '{tracerExpFreq}':")
            for timeStamp in tracerMissing:
                print(timeStamp)
            print(f"Missing timestamps which are estimated from expected data frequency of '{metExpFreq}':")
            for timeStamp in metMissing:
                print(timeStamp)
            print(f"Number of missing tracer files which are expected from typical data frequency: {len(tracerMissing)}")
            print(f"Number of missing met data files which are expected from typical data frequency: {len(metMissing)}")

        pathDictionary = chunkMetFilesPathsForBinning(metPathsAndTime, tracerPathsAndTime, tomlConfig['MetDataBinningTime'], tracerExpFreq, metExpFreq)

        if not pathDictionary:
            print("ERROR: No tracer timestamps could be matched to any met files.\n\n"
                  "Check that met and tracer file date ranges overlap and that "
                  "MetDataBinningTime is wide enough to capture at least one met file per tracer timestamp.")
            sys.exit(1)

        unmatched = [ts for ts in tracerPathsAndTime.index if ts not in pathDictionary]
        if unmatched:
            print(f"WARNING: {len(unmatched)} tracer timestamp(s) could not be matched to any met files and will be skipped:")
            for ts in unmatched:
                print(f"  {ts}")

        numOfFiles = len(pathDictionary)
        numbers = list(range(len(pathDictionary)))


    reporter = threading.Thread(
            target=progress_reporter, 
            args=(sharedCounter, numOfFiles, timeStart)
        )
    reporter.daemon = True # Allows the program to exit if the thread is stuck
    reporter.start()

    try: 
        if tomlConfig['tracerDataInMetFiles']:
            with multiprocessing.Pool(
                processes=tomlConfig['processNumber'], 
                initializer=init_worker, 
                initargs=(sharedCounter, )
                ) as p:
                p.starmap(mainCalcs, zip(repeat(tomlConfig), numbers,
                                    repeat(pathsAndTime), repeat(reqVars), repeat(''), repeat(''))) # reqVars includes tracers
        else:
            with multiprocessing.Pool(
                processes=tomlConfig['processNumber'], 
                initializer=init_worker, 
                initargs=(sharedCounter, )
                ) as p:
                p.starmap(mainCalcs, zip(repeat(tomlConfig), numbers, 
                                    repeat(''), repeat(''), repeat(pathDictionary), repeat(reqVars))) # no tracer names in reqVars
    
    except Exception as e:
        print(f"\nAn error occurred: {repr(e)}")
        sys.exit(1)

    else:
        reporter.join()
    print('Processing complete!')


def run_tracer_transport_theta() -> None:
    """
    Entry point for the ``tem-tracer-transport-theta`` CLI command.

    Parses arguments, assembles the required-variable list for
    theta-coordinate inputs, and delegates to :func:`run_tracer_transport`.
    """
    from .parser import tTransport_theta_parser
    from .tracer_transport_theta import init_worker, mainCalcs
    parserArgs = tTransport_theta_parser().parse_args()
    
    # combine parser and config file settings
    tomlConfig = load_and_merge_config(parserArgs)
    if tomlConfig['processNumber'] in ['all cores', 'all', 'All']:
        tomlConfig['processNumber'] = multiprocessing.cpu_count()

    # create list of required variables    
    if tomlConfig['verticalDimensionType'] == 'other':
        reqVars = [tomlConfig['thetaName'], tomlConfig['pressureName'], 
                tomlConfig['meridionalWindName'], tomlConfig['verticalWindName']]
    else:
        reqVars = [tomlConfig['pressureName'],
                tomlConfig['meridionalWindName'], tomlConfig['verticalWindName']]
        
    if tomlConfig['tracerDataInMetFiles']:
        reqVars.extend(tomlConfig['tracerNames'])

    run_tracer_transport(mainCalcs, init_worker, tomlConfig, reqVars)



def run_tracer_transport_press() -> None:
    """
    Entry point for the ``tem-tracer-transport-press`` CLI command.

    Parses arguments, assembles the required-variable list for
    log-pressure-coordinate inputs, and delegates to :func:`run_tracer_transport`.
    """
    from .parser import tTransport_press_parser
    from .tracer_transport_press import init_worker, mainCalcs
    parserArgs = tTransport_press_parser().parse_args()

    # combine parser and config file settings
    tomlConfig = load_and_merge_config(parserArgs)
    if tomlConfig['processNumber'] in ['all cores', 'all', 'All']:
        tomlConfig['processNumber'] = multiprocessing.cpu_count()

    # create list of required variables
    if tomlConfig['verticalDimensionType'] == 'other':
        reqVars = [tomlConfig['pressureName'], tomlConfig['temperatureName'],
                tomlConfig['meridionalWindName'], tomlConfig['verticalWindName']]
    else:
        reqVars = [tomlConfig['temperatureName'],
                tomlConfig['meridionalWindName'], tomlConfig['verticalWindName']]

    if tomlConfig['tracerDataInMetFiles']: # if met and tracer data are in the same files.
        reqVars.extend(tomlConfig['tracerNames'])

    run_tracer_transport(mainCalcs, init_worker, tomlConfig, reqVars)


# def run_wave_decomp(coord: str) -> None:
#     """
#     Shared driver for the stationary/transient wave decomposition commands.

#     Reads all input files for the configured time period into memory, then
#     calls the decomposition function once with the full list of datasets.
#     Unlike the per-timestep transport commands, no multiprocessing pool is
#     used: the decomposition is inherently a whole-period operation.

#     Parameters
#     ----------
#     coord : str
#         ``'press'`` for log-pressure coordinates, ``'theta'`` for isentropic.
#     """
#     timeStart = time.time()

#     if coord == 'press':
#         from .parser import wave_decomp_press_parser
#         from .temporal_wave_decomp import waveDecompPress as decompFn
#         from .interpolation import interpolateToPressureAndCombineData, interpolateToLogPressure
#         parserArgs = wave_decomp_press_parser().parse_args()
#     else:
#         from .parser import wave_decomp_theta_parser
#         from .temporal_wave_decomp import waveDecompTheta as decompFn
#         from .interpolation import interpolateToThetaAndCombineData, interpolateToTheta
#         parserArgs = wave_decomp_theta_parser().parse_args()

#     from .file_io import readAndTransposeData, readDataAndGetWeightedAverage, saveOut
#     from .utils import binData

#     tomlConfig = load_and_merge_config(parserArgs)
#     if tomlConfig['processNumber'] in ['all cores', 'all', 'All']:
#         tomlConfig['processNumber'] = multiprocessing.cpu_count()

#     if '{tracerNames}' in tomlConfig['outPrefix']:
#         if isinstance(tomlConfig['tracerNames'], list):
#             tomlConfig['outPrefix'] = tomlConfig['outPrefix'].replace('{tracerNames}', '_'.join(tomlConfig['tracerNames']))
#         else:
#             print("ERROR: option 'tracerNames' should be list of strings.")
#             sys.exit(1)

#     if not Path(tomlConfig['outputDirectory']).is_dir():
#         print(f"ERROR: Directory '{tomlConfig['outputDirectory']}' does not exist.\n\n"
#               "Please specify existing directory to store output files")
#         sys.exit(1)

#     # --- build required variable lists ---
#     tracers = tomlConfig.get('tracerNames', [])
#     if coord == 'press':
#         met_vars = [tomlConfig['temperatureName'], tomlConfig['meridionalWindName']]
#         if tomlConfig['verticalDimensionType'] == 'other':
#             met_vars.insert(0, tomlConfig['pressureName'])
#         if tomlConfig['verticalWindType'].lower() != 'missing':
#             met_vars.append(tomlConfig['verticalWindName'])
#         if tomlConfig.get('computeEPF', True):
#             met_vars.append(tomlConfig['zonalWindName'])
#     else:
#         met_vars = [tomlConfig['pressureName'], tomlConfig['meridionalWindName'],
#                     tomlConfig['verticalWindName']]
#         if tomlConfig['verticalDimensionType'] == 'other':
#             met_vars.insert(0, tomlConfig['thetaName'])
#     met_vars = list(dict.fromkeys(met_vars))  # deduplicate, preserve order

#     # --- collect file paths ---
#     if tomlConfig['tracerDataInMetFiles']:
#         pathsAndTime, missingTS, expFreq = collectFileNamesTTransport(
#             tomlConfig['inputDirectory'], tomlConfig['inFileNames'],
#             tomlConfig['timeInfoInFileNames'],
#             dateStart=tomlConfig['startDate'], dateEnd=tomlConfig['endDate'])
#         if missingTS:
#             print(f"WARNING: {len(missingTS)} missing timestamp(s) estimated from "
#                   f"expected frequency '{expFreq}'.")
#     else:
#         metPathsAndTime, metMissing, metExpFreq = collectFileNamesTTransport(
#             tomlConfig['inputDirectory'], tomlConfig['inFileNames'],
#             tomlConfig['timeInfoInFileNames'],
#             dateStart=tomlConfig['startDate'], dateEnd=tomlConfig['endDate'])
#         tracerPathsAndTime, tracerMissing, tracerExpFreq = collectFileNamesTTransport(
#             tomlConfig['tracerInputDirectory'], tomlConfig['tracerInFileNames'],
#             tomlConfig['tracerTimeInfoInFileNames'],
#             dateStart=tomlConfig['startDate'], dateEnd=tomlConfig['endDate'])
#         pathDictionary = chunkMetFilesPathsForBinning(
#             metPathsAndTime, tracerPathsAndTime,
#             tomlConfig['MetDataBinningTime'], tracerExpFreq, metExpFreq)
#         if not pathDictionary:
#             print("ERROR: No tracer timestamps could be matched to any met files.\n\n"
#                   "Check that met and tracer file date ranges overlap and that "
#                   "MetDataBinningTime is wide enough to capture at least one met file per tracer timestamp.")
#             sys.exit(1)
#         unmatched = [ts for ts in tracerPathsAndTime.index if ts not in pathDictionary]
#         if unmatched:
#             print(f"WARNING: {len(unmatched)} tracer timestamp(s) could not be matched "
#                   "to any met files and will be skipped.")

#     # --- load and interpolate all timesteps ---
#     numOfFiles = (len(pathsAndTime) if tomlConfig['tracerDataInMetFiles']
#                   else len(pathDictionary))
#     print(f"Loading {numOfFiles} file(s)...")

#     datasets: list = []

#     if tomlConfig['tracerDataInMetFiles']:
#         req_all = list(dict.fromkeys(met_vars + tracers))
#         for path in pathsAndTime.Path:
#             ds = readAndTransposeData(path, req_all,
#                                       tomlConfig['vertDim'], tomlConfig['latDim'], tomlConfig['lonDim'],
#                                       timeDimName=tomlConfig.get('timeDim', ''))
#             if coord == 'press':
#                 ds = interpolateToLogPressure(ds, req_all, tomlConfig['verticalDimensionType'],
#                                               tomlConfig['targetLevels'], tomlConfig['vertDim'],
#                                               tomlConfig['latDim'], tomlConfig['lonDim'],
#                                               tomlConfig['pressureName'])
#             else:
#                 ds = interpolateToTheta(ds, req_all, tomlConfig)
#             datasets.append(binData(ds, tomlConfig['binningLat'], tomlConfig['binningLon']))
#     else:
#         for ts, entry in pathDictionary.items():
#             tracerPath, metPaths, metWeights = entry
#             tracerDs = readAndTransposeData(tracerPath, tracers,
#                                             tomlConfig['tracerVertDim'], tomlConfig['tracerLatDim'],
#                                             tomlConfig['tracerLonDim'],
#                                             timeDimName=tomlConfig.get('tracerTimeDim', ''))
#             metDs = readDataAndGetWeightedAverage(metPaths, metWeights, met_vars,
#                                                   tomlConfig['vertDim'], tomlConfig['latDim'],
#                                                   tomlConfig['lonDim'])
#             if coord == 'press':
#                 ds = interpolateToPressureAndCombineData(tracerDs, metDs, met_vars, tomlConfig)
#             else:
#                 ds = interpolateToThetaAndCombineData(tracerDs, metDs, met_vars, tomlConfig)
#             datasets.append(binData(ds, tomlConfig['binningLat'], tomlConfig['binningLon']))

#     print(f"Computing decomposition over {len(datasets)} timestep(s)...")

#     # --- run decomposition and save ---
#     dataToSave, lats, vertCoord, wave_numbers = decompFn(datasets, tomlConfig)

#     # Build a single output filename using start/end of the period
#     timestamps = (list(pathsAndTime.index) if tomlConfig['tracerDataInMetFiles']
#                   else list(pathDictionary.keys()))
#     t0 = pd.Timestamp(timestamps[0])
#     t1 = pd.Timestamp(timestamps[-1])
#     fnout = (tomlConfig['outputDirectory'] + '/' + tomlConfig['outPrefix']
#              + f"{t0.year}_{t0.month:02d}_{t0.day:02d}"
#              + f"_to_{t1.year}_{t1.month:02d}_{t1.day:02d}.nc")

#     import xarray as xr
#     dsOut = xr.Dataset()
#     vert_dim = 'alt' if coord == 'press' else 'theta'
#     for var, (data, long_name, unit) in dataToSave.items():
#         dsOut[var] = ((vert_dim, 'lat', 'waveN'), np.single(data))
#         dsOut[var].attrs['long_name'] = long_name
#         dsOut[var].attrs['units'] = unit
#     dsOut.coords[vert_dim] = vertCoord
#     if coord == 'press':
#         dsOut[vert_dim].attrs['long_name'] = 'log-pressure altitude (z = -H·ln(p/ps), H=7 km, ps=1000 hPa)'
#         dsOut[vert_dim].attrs['units'] = 'm'
#     else:
#         dsOut[vert_dim].attrs['long_name'] = 'potential temperature'
#         dsOut[vert_dim].attrs['units'] = 'K'
#     dsOut.coords['lat'] = lats
#     dsOut.lat.attrs['long_name'] = 'latitude'
#     dsOut.lat.attrs['units'] = 'degree_N'
#     dsOut.coords['waveN'] = wave_numbers
#     dsOut.waveN.attrs['long_name'] = 'wavenumber (number of waves per 360 degrees of longitude)'
#     dsOut.attrs['period_start'] = str(t0)
#     dsOut.attrs['period_end'] = str(t1)
#     dsOut.attrs['n_timesteps'] = len(datasets)
#     dsOut.to_netcdf(fnout)

#     elapsed = time.time() - timeStart
#     print(f"Done. Output written to {fnout}  ({elapsed:.1f}s)")


# def run_wave_decomp_press() -> None:
#     """Entry point for the ``wave-decomp-press`` CLI command."""
#     run_wave_decomp('press')


# def run_wave_decomp_theta() -> None:
#     """Entry point for the ``wave-decomp-theta`` CLI command."""
#     run_wave_decomp('theta')
