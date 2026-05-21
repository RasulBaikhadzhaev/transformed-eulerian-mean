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
# The warning "mean of empty slice" happens when zonal mean is calculated, it accures due to absence of data at every 
# longtitudal point in places like 1000 hPa level in antarctica. Other warning occures during interpolation to pressure 
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
    # if config['saveZonalMean']:
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

    # numOfChunks = len(pathsAndTimeChunked)
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


    # with multiprocessing.Pool(
    #     processes=config['processNumber'], 
    #     initializer=init_worker, 
    #     initargs=(sharedCounter, )
    #     ) as p:
        
    #     p.starmap(mainCalcs, zip(pathsAndTimeChunked.values(), 
    #                             repeat(reqVars), 
    #                             repeat(config), 
    #                             repeat(saveInterpolatedZonalMeanVars), 
    #                             repeat(saveZonalMeanVars)))


def run_tracer_transport(mainCalcs: Callable, init_worker: Callable, tomlConfig: dict, reqVars: list[str]) -> None:

    timeStart = time.time()
    # counter for progress
    sharedCounter = multiprocessing.Value('i', 0)

    # if '{tracerNames}' exists in outPrefix replace it with names of the tracers.
    if '{tracerNames}' in tomlConfig['outPrefix']:
        if isinstance(tomlConfig['tracerNames'], list):
            tomlConfig['outPrefix'] = tomlConfig['outPrefix'].replace('{tracerNames}', '_'.join(tomlConfig['tracerNames']))
        # elif isinstance(tomlConfig['tracerNames'], str):
        #     tomlConfig['outPrefix'] = tomlConfig['outPrefix'].replace('{tracerNames}', tomlConfig['tracerNames'])
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

        numOfFiles = len(tracerPathsAndTime)
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
