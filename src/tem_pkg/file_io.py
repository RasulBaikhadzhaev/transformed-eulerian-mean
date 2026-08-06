import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def extractTimeFromFileNames(filesPaths, timeInfoInFileNames):
    '''
    Parses timestamps from file names based on a user-provided template.

    Accepts 'YYYY' (or 'YY' if century information is not in file names) for
    year, 'MM' for month, 'DD' for day, 'HH' for hour, 'mm' for minute, and
    'ss' for second. At least the year position must be given; include all
    other available time fields. Only one '*' is allowed and it must be the
    first or last character; all non-time characters must be represented by '?'.

    Examples:
        '*YYMMDDHH???' for era5_10120100.nc
        'mm?HH?MM?DD?YYYY*' for 50_18_10_25_1990_data.nc
    '''
    if 'YYYY' in timeInfoInFileNames:
        yearString = 'YYYY'
    elif 'YY' in timeInfoInFileNames:
        yearString = 'YY'
    else:
        print("ERROR: Impossible to extract time information from file names.\n\n"
                "At least position of year (YYYY or YY) must be provided in config.")
        sys.exit(1)

    timeStrings = {'monthString': 'MM', 'dayString': 'DD', 'hourString': 'HH', 'minuteString': 'mm', 'secondString': 'ss'}
    timeInfo = {}

    if timeInfoInFileNames.startswith('*'):
        yearInfoStart = len(timeInfoInFileNames) - timeInfoInFileNames.find(yearString) - len(yearString)
        yearInfo = [str(file)[-(yearInfoStart + len(yearString)):-yearInfoStart] for file in filesPaths]
        if yearString == 'YY':
            yearInfo = ['20' + year if int(year[0:2]) < 50 else '19' + year for year in yearInfo]

        for string in timeStrings.keys():
            if timeStrings[string] in timeInfoInFileNames:
                timeInfoStart = len(timeInfoInFileNames) - timeInfoInFileNames.find(timeStrings[string]) - len(timeStrings[string])
                timeInfo[string] = [str(file)[-(timeInfoStart + len(timeStrings[string])):-timeInfoStart] for file in filesPaths]
            else:
                timeInfo[string] = ['00'] * len(filesPaths)

    else:
        yearInfoStart = timeInfoInFileNames.find(yearString)
        yearInfo = [str(file.name)[yearInfoStart: (yearInfoStart + len(yearString))] for file in filesPaths]
        if yearString == 'YY':
            yearInfo = ['20' + year if int(year[0:2]) < 50 else '19' + year for year in yearInfo]

        for string in timeStrings.keys():
            if timeStrings[string] in timeInfoInFileNames:
                timeInfoStart = timeInfoInFileNames.find(timeStrings[string])
                timeInfo[string] = [str(file.name)[timeInfoStart: (timeInfoStart + len(timeStrings[string]))] for file in filesPaths]
            else:
                timeInfo[string] = ['00'] * len(filesPaths)

    dateAndtime = pd.to_datetime([yearInfo[i] + '-' + timeInfo['monthString'][i] + '-' + timeInfo['dayString'][i] + ' ' +
                                    timeInfo['hourString'][i] + ':' +  timeInfo['minuteString'][i] + ':' +
                                    timeInfo['secondString'][i] for i in range(len(filesPaths))], format='%Y-%m-%d %H:%M:%S')

    return dateAndtime


def _load_and_filter_file_paths(inputDir, fileNames, timeInfoInFileNames,
                                dateStart, dateEnd, hoursToKeep, inputPathType):
    """
    Shared file-discovery and filtering logic for both collectFileNames variants.

    Returns filesPathAndTime (DataFrame with Path column, time index), sorted and
    filtered by date range and hoursToKeep.
    """
    if inputPathType in ('directory', 'Directory'):
        if not Path(inputDir).is_dir():
            print(f"ERROR: Directory '{inputDir}' does not exist.\n\n"
                  "Please check if the path is correct. There should be no spaces before or after the path")
            sys.exit(1)
        filesPaths = list(Path(inputDir).rglob(fileNames))
        if not filesPaths:
            print(f"ERROR: File '{fileNames}' does not exist in '{inputDir}'.\n\n"
                  "Please specify NetCDF filename which contains data for TEM calculations")
            sys.exit(1)
    elif inputPathType == '.txt':
        with open(inputDir) as txtFile:
            filesPaths = [Path(line.strip()) for line in txtFile]
        if not filesPaths:
            print("ERROR: No input files found. Please check if the input .txt file exists and\n\n"
                  "includes path(s) to exising NetCDF file(s) with data for TEM calculations")
            sys.exit(1)

    time = extractTimeFromFileNames(filesPaths, timeInfoInFileNames)
    filesPathAndTime = pd.DataFrame(data={'Path': filesPaths}, index=time)

    if dateStart != '' and dateEnd != '':
        filesPathAndTime = filesPathAndTime[(filesPathAndTime.index >= pd.to_datetime(dateStart)) &
                                            (filesPathAndTime.index <= pd.to_datetime(dateEnd))]
    elif dateStart != '':
        filesPathAndTime = filesPathAndTime[filesPathAndTime.index >= pd.to_datetime(dateStart)]
    elif dateEnd != '':
        filesPathAndTime = filesPathAndTime[filesPathAndTime.index <= pd.to_datetime(dateEnd)]

    if filesPathAndTime.empty:
        print(f"ERROR: After filtering by start and end date no suitable files '{fileNames}' exist in '{inputDir}'.\n\n"
              "Please adjust dates or specify NetCDF filename which contains data for TEM calculations")
        sys.exit(1)

    if hoursToKeep:
        filesPathAndTime = filesPathAndTime[filesPathAndTime.index.hour.isin(hoursToKeep)]
    if filesPathAndTime.empty:
        print(f"ERROR: After filtering by hours to keep parameter no suitable files '{fileNames}' exist in '{inputDir}'.\n\n"
              "Please adjust hoursToKeep parameter in the configuration file")
        sys.exit(1)

    return filesPathAndTime.sort_index()


def collectFileNames(inputDir, fileNames, timeInfoInFileNames, outputDir='', dateStart='', dateEnd='',
                     outPrefix='', outDirSkip=0, inputPathType='directory', hoursToKeep=[], outputTemporalMean=''):
    '''
    Gathers and filters input file paths based on time ranges and existence of output files.

    Arguments:
        inputDir: input directory
        fileNames: file names; rglob is used to find files with matching names
        outputDir: output directory, only used to filter already-processed timestamps
        dateStart: YYYY-MM-DD-HH format; files before this date are excluded
        dateEnd: YYYY-MM-DD-HH format; files after this date are excluded
        outPrefix: prefix given to output files
        outDirSkip: skip timestamps for which an output file already exists
        inputPathType: 'directory' or '.txt'
        hoursToKeep: restrict to these hours of day; all hours kept if empty
        outputTemporalMean: 'monthly', 'daily', or falsy

    Returns:
        filesPathsAndTime: DataFrame with Path and time index
        missingTimeStamps: timestamps absent from the expected regular grid
        expectedFrequency: modal time step inferred from the file list
    '''
    filesPathAndTime = _load_and_filter_file_paths(
        inputDir, fileNames, timeInfoInFileNames, dateStart, dateEnd, hoursToKeep, inputPathType)

    if len(filesPathAndTime) >= 3:
        expectedFrequency = pd.infer_freq(filesPathAndTime.index)

        if not expectedFrequency:
            all_diffs = filesPathAndTime.index.to_series().diff().dropna().dt.days
            min_gap = all_diffs.min()

            if (28 <= min_gap <= 31) and (len(filesPathAndTime.index.day.unique()) == 1 or filesPathAndTime.index.is_month_end.all()):
                expectedFrequency = pd.DateOffset(months=1)
            else:
                expectedFrequency = filesPathAndTime.index.to_series().diff().dropna().mode()[0]

        completeRange = pd.date_range(
            start=filesPathAndTime.index.min(),
            end=filesPathAndTime.index.max(),
            freq=expectedFrequency
        )
        missingTimeStamps = completeRange.difference(filesPathAndTime.index)
    else:
        expectedFrequency = ''
        missingTimeStamps = pd.DataFrame()

    if outDirSkip and str(outputTemporalMean).lower() not in ['monthly', 'month', 'daily', 'day']:
        filesPaths_inOutDir = list(Path(outputDir).rglob(f'{outPrefix}*.nc'))
        times_InOuttDir = extractTimeFromFileNames(filesPaths_inOutDir, '*YYYY_MM_DD_HH_mm???')
        filesPathAndTime_InOutDir = pd.DataFrame(data={'Path': filesPaths_inOutDir}, index=times_InOuttDir)
        filesPathAndTime = filesPathAndTime[~filesPathAndTime.index.isin(filesPathAndTime_InOutDir.index)]
        if filesPathAndTime.empty:
            print(f"ERROR: Files with prefix '{outPrefix}' are already present in the output directory for all selected timestamps.")
            sys.exit(1)

    return filesPathAndTime, missingTimeStamps, expectedFrequency


def collectFileNamesTTransport(inputDir, fileNames, timeInfoInFileNames, outputDir='', dateStart='', dateEnd='',
                     outPrefix='', outDirSkip=0, inputPathType='directory', hoursToKeep=[]):
    '''
    Gathers and filters input file paths for tracer transport tools.

    Same filtering logic as collectFileNames but without outputTemporalMean
    support, and with a simpler missing-timestamp estimate based on the modal
    time delta rather than pd.infer_freq.
    '''
    filesPathAndTime = _load_and_filter_file_paths(
        inputDir, fileNames, timeInfoInFileNames, dateStart, dateEnd, hoursToKeep, inputPathType)

    timeDiffs = filesPathAndTime.index.to_series().diff().dropna()
    expectedFrequency = timeDiffs.mode()[0]
    completeRange = pd.date_range(start=filesPathAndTime.index.min(), end=filesPathAndTime.index.max(), freq=expectedFrequency)
    missingTimeStamps = [ts for ts in completeRange if ts not in set(filesPathAndTime.index)]

    if outDirSkip == 1:
        filesPaths_inOutDir = list(Path(outputDir).rglob(f'{outPrefix}*.nc'))
        times_InOuttDir = extractTimeFromFileNames(filesPaths_inOutDir, '*YYYY_MM_DD_HH_mm???')
        filesPathAndTime_InOutDir = pd.DataFrame(data={'Path': filesPaths_inOutDir}, index=times_InOuttDir)
        filesPathAndTime = filesPathAndTime[~filesPathAndTime.index.isin(filesPathAndTime_InOutDir.index)]
        if filesPathAndTime.empty:
            print(f"ERROR: Files with prefix '{outPrefix}' are already present in the output directory for all selected timestamps.")
            sys.exit(1)

    return filesPathAndTime, missingTimeStamps, expectedFrequency


def chunkMetFilesPathsForBinning(metFilesPaths, tracerFilesPaths, MetDataBinningTime, tracerExpectedFrequency, metExpectedFrequency):
    """
    Pairs met data files with tracer files, potentially averaging multiple met
    files to match the lower temporal frequency of the tracer data.
    """
    pathDictionary = {}

    if MetDataBinningTime == 'auto':
        if tracerExpectedFrequency == metExpectedFrequency:
            if set(tracerFilesPaths.index) & set(metFilesPaths.index):
                pathAndTime = tracerFilesPaths.join(metFilesPaths.rename(columns={'Path': 'metFilesPath'}))
                pathAndTime.dropna(inplace=True)
                pathAndTime['weight'] = 1
                timestamps = pathAndTime.index
                for timestamp in timestamps:
                    pathDictionary[timestamp] = [pathAndTime.loc[timestamp].Path, pathAndTime.loc[timestamp].metFilesPath, pathAndTime.loc[timestamp].weight]
            else:
                timestamps = tracerFilesPaths.index
                for timestamp in timestamps:
                    metDataPathOfTimestamp = metFilesPaths[(metFilesPaths.index >= timestamp - tracerExpectedFrequency / 2) &
                                                            (metFilesPaths.index <= timestamp + tracerExpectedFrequency / 2)]
                    if np.array(metDataPathOfTimestamp).size == 2:
                        weight1stFile = np.abs((metDataPathOfTimestamp.index[0] - timestamp) / tracerExpectedFrequency)
                        weight2ndFile = np.abs((metDataPathOfTimestamp.index[1] - timestamp) / tracerExpectedFrequency)
                        pathDictionary[timestamp] = [tracerFilesPaths.loc[timestamp].Path, np.array(metDataPathOfTimestamp), [weight1stFile, weight2ndFile]]

        else:
            timestamps = tracerFilesPaths.index
            for timestamp in timestamps:
                metDataPathOfTimestamp = metFilesPaths[(metFilesPaths.index >= timestamp - tracerExpectedFrequency / 2) &
                                                        (metFilesPaths.index <= timestamp + tracerExpectedFrequency / 2)].copy()

                if metDataPathOfTimestamp.empty:
                    continue

                metDataPathOfTimestamp.loc[:, 'hour'] = metDataPathOfTimestamp.index.hour
                hourCounts = metDataPathOfTimestamp['hour'].value_counts()
                totalWeight = 1.0
                uniqueHours = len(hourCounts)
                hourlyWeight = totalWeight / uniqueHours
                weightsPerRow = hourlyWeight / hourCounts
                metDataPathOfTimestamp['weight'] = metDataPathOfTimestamp['hour'].map(weightsPerRow)

                pathDictionary[timestamp] = [tracerFilesPaths.loc[timestamp].Path, np.array(metDataPathOfTimestamp.Path),
                                            np.array(metDataPathOfTimestamp.weight)]

    elif isinstance(MetDataBinningTime, int):
        timestamps = tracerFilesPaths.index
        for timestamp in timestamps:
            timeDiffs = abs(metFilesPaths.index - timestamp)
            closestRows = metFilesPaths.iloc[np.argsort(timeDiffs)[:MetDataBinningTime]]
            pathDictionary[timestamp] = [tracerFilesPaths.loc[timestamp].Path, np.array(closestRows.Path),
                                         np.zeros(MetDataBinningTime) + 1/MetDataBinningTime]

    else:
        print(f"ERROR: MetDataBinningTime in is set to '{MetDataBinningTime}'.\n\n"
              "it can only be set to 'auto' or be integer, please check the parameter in the configuration file")
        sys.exit(1)

    return pathDictionary


def _replace_fill_values(dataset, fillValues):
    '''Replace user-specified fill values with NaN in all data variables.'''
    if not fillValues:
        return dataset
    for var in dataset.data_vars:
        mask = True
        for fv in fillValues:
            mask = mask & (dataset[var] != fv)
        dataset[var] = dataset[var].where(mask)
    return dataset


def readAndTransposeData(filePath, reqVars, vertDimName, latDimName, lonDimName, saveInterpolatedZonalMeanVars=[], saveZonalMeanVars=[], fillValues=[]):
    '''
    Reads a NetCDF file and standardizes dimension order to [Vertical, Latitude, Longitude].
    '''
    dataset = xr.open_dataset(filePath)[reqVars + saveInterpolatedZonalMeanVars + saveZonalMeanVars].squeeze()
    dataset = _replace_fill_values(dataset, fillValues)
    if 'time' in dataset.dims:
        dataset = dataset.transpose('time', vertDimName, latDimName, lonDimName)
    else:
        dataset = dataset.transpose(vertDimName, latDimName, lonDimName)
    return dataset


def readDataAndGetWeightedAverage(filesPaths, weights, reqVars, vertDimName, latDimName, lonDimName, fillValues=[]):
    '''Computes a weighted average of multiple NetCDF files (e.g., for time-binning met data).'''
    for index, path in enumerate(filesPaths):
        dataset = xr.open_dataset(path)[reqVars].squeeze()
        dataset = _replace_fill_values(dataset, fillValues)
        if index == 0:
            weightedMeanDataset = dataset * weights[index]
        else:
            weightedMeanDataset = weightedMeanDataset + (dataset * weights[index])
    for variable in reqVars:
        weightedMeanDataset[variable].attrs = dataset[variable].attrs
        weightedMeanDataset[variable] = weightedMeanDataset[variable].transpose(vertDimName, latDimName, lonDimName)
    return weightedMeanDataset


def saveOut(dataToSave, tomlConfig, timeStamp, lats, thetaLevels):
    """
    Formats the final results and Fourier components into an xarray dataset
    and saves it to a NetCDF file.
    """
    fnout = tomlConfig['outputDirectory'] + '/' + tomlConfig['outPrefix'] + str(timeStamp)[:-3].replace('-', '_').replace(' ', '_').replace(':', '_') + '.nc'

    if 'Fourier' in dataToSave.keys():
        Fourier = dataToSave['Fourier']
        FourierToSave = True
        del dataToSave['Fourier']
    else:
        FourierToSave = False

    dsOut = xr.Dataset()

    for variable in dataToSave.keys():
        dsOut[variable] = (('theta', 'lat'), np.single(dataToSave[variable][0]))
        getattr(dsOut, variable).attrs['long_name'] = dataToSave[variable][1]
        getattr(dsOut, variable).attrs['units'] = dataToSave[variable][2]

    if FourierToSave:
        for variable in Fourier.keys():
            dsOut[variable] = (('theta', 'lat', 'waveN'),
                               np.single(Fourier[variable][0]))
            getattr(dsOut, variable).attrs['long_name'] = Fourier[variable][1]
            getattr(dsOut, variable).attrs['units'] = Fourier[variable][2]

        if tomlConfig['Waves'] == ['all'] or tomlConfig['Waves'] == ['All']:
            waveNumbers = list(range(1, Fourier[list(Fourier.keys())[0]][0].shape[2] + 1))
        else:
            waveNumbers = tomlConfig['Waves']

        dsOut.coords['waveN'] = waveNumbers
        dsOut.waveN.attrs['long_name'] = 'wave number'

    dsOut.coords['theta'] = thetaLevels
    dsOut.theta.attrs['long_name'] = 'Potential temperature levels'
    dsOut.theta.attrs['units'] = 'K'
    dsOut.coords['lat'] = lats
    dsOut.lat.attrs['long_name'] = 'latitude'
    dsOut.lat.attrs['units'] = 'degree_N'
    dsOut.coords['time'] = [timeStamp]

    dsOut.to_netcdf(fnout)
