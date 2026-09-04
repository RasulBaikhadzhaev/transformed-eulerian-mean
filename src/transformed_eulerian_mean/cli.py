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
    init_spinner,
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

    spinnerStop = threading.Event()
    spinnerThread = threading.Thread(target=init_spinner, args=(spinnerStop, timeStart), daemon=True)
    spinnerThread.start()

    # chunk pathAndTime by month, day or instance.
    if str(config['outputTemporalMean']).lower() in ['monthly', 'month'] and is_equal_or_shorter_than_month(expectedFrequency):
        pathsAndTimeChunked = {date: group for date, group in pathsAndTime.groupby(pd.Grouper(freq='MS'))}
    elif str(config['outputTemporalMean']).lower() in ['daily', 'day'] and is_equal_or_shorter_than_day(expectedFrequency):
        pathsAndTimeChunked = {date: group for date, group in pathsAndTime.groupby(pd.Grouper(freq='D'))}
    else:
        pathsAndTimeChunked = {index: group for index, group in pathsAndTime.groupby(pathsAndTime.index)}

    spinnerStop.set()
    spinnerThread.join()
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

    if str(config['outputTemporalMean']).lower() in ['monthly', 'month', 'daily', 'day']:
        maxtasksperchild = 1
    else:
        maxtasksperchild = 10

    try:
        with multiprocessing.Pool(
            processes=config['processNumber'],
            initializer=init_worker,
            initargs=(sharedCounter, ),
            maxtasksperchild=maxtasksperchild,
            ) as p:

            reporter = threading.Thread(
                target=progress_reporter,
                args=(sharedCounter, numOfFiles, timeStart, p)
            )
            reporter.daemon = True
            reporter.start()

            p.starmap(mainCalcs, zip(pathsAndTimeChunked.values(),
                                    repeat(reqVars),
                                    repeat(config),
                                    repeat(saveInterpolatedZonalMeanVars),
                                    repeat(saveZonalMeanVars)), chunksize=1)
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

    spinnerStop = threading.Event()
    spinnerThread = threading.Thread(target=init_spinner, args=(spinnerStop, timeStart), daemon=True)

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
        spinnerThread.start()

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

        spinnerThread.start()
        pathDictionary = chunkMetFilesPathsForBinning(metPathsAndTime, tracerPathsAndTime, tomlConfig['MetDataBinningTime'], tracerExpFreq, metExpFreq)

        if not pathDictionary:
            spinnerStop.set()
            print("\nERROR: No tracer timestamps could be matched to any met files.\n\n"
                  "Check that met and tracer file date ranges overlap and that "
                  "MetDataBinningTime is wide enough to capture at least one met file per tracer timestamp.")
            sys.exit(1)

        unmatched = [ts for ts in tracerPathsAndTime.index if ts not in pathDictionary]
        if unmatched:
            spinnerStop.set()
            spinnerThread.join()
            print(f"\nWARNING: {len(unmatched)} tracer timestamp(s) could not be matched to any met files and will be skipped:")
            for ts in unmatched:
                print(f"  {ts}")
            spinnerStop.clear()
            spinnerThread = threading.Thread(target=init_spinner, args=(spinnerStop, timeStart), daemon=True)
            spinnerThread.start()

        numOfFiles = len(pathDictionary)

    spinnerStop.set()
    spinnerThread.join()
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

    try:
        if tomlConfig['tracerDataInMetFiles']:
            # Pre-extract per-task path so the full DataFrame is not pickled for every task
            task_paths = [(ts, pathsAndTime['Path'].iloc[i]) for i, ts in enumerate(pathsAndTime.index)]
            with multiprocessing.Pool(
                processes=tomlConfig['processNumber'],
                initializer=init_worker,
                initargs=(sharedCounter, ),
                maxtasksperchild=10,
                ) as p:
                reporter = threading.Thread(
                    target=progress_reporter,
                    args=(sharedCounter, numOfFiles, timeStart, p)
                )
                reporter.daemon = True
                reporter.start()
                p.starmap(mainCalcs, zip(repeat(tomlConfig), task_paths,
                                    repeat(reqVars), repeat(''), repeat('')), chunksize=1)
        else:
            # Pre-extract per-task entries so the full dict is not pickled for every task
            task_entries = [(ts, v[0], v[1], v[2]) for ts, v in pathDictionary.items()]
            with multiprocessing.Pool(
                processes=tomlConfig['processNumber'],
                initializer=init_worker,
                initargs=(sharedCounter, ),
                maxtasksperchild=10,
                ) as p:
                reporter = threading.Thread(
                    target=progress_reporter,
                    args=(sharedCounter, numOfFiles, timeStart, p)
                )
                reporter.daemon = True
                reporter.start()
                p.starmap(mainCalcs, zip(repeat(tomlConfig), repeat(''), repeat(''),
                                    task_entries, repeat(reqVars)), chunksize=1)

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
