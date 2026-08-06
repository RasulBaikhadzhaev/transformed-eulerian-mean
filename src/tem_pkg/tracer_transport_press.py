import numpy as np
from metpy.units import units
from scipy.integrate import cumulative_trapezoid

from .constants import P0, Cp, R, Ts, gEarth, rEarth
from .file_io import readAndTransposeData, readDataAndGetWeightedAverage, saveOut
from .interpolation import alt2press, interpolateToLogPressure, interpolateToPressureAndCombineData
from .utils import addRatioUnits, binData, nanGradient


def tracerTransport(interpolatedDataset, tomlConfig):

    addRatioUnits()

    lats = np.array(interpolatedDataset.lat)
    lons = np.array(interpolatedDataset.lon)
    latsR = (lats * units.degrees).to('radian').magnitude
    cosFi = np.cos(latsR)[np.newaxis, :]

    #find basic density
    altitudes = (np.array(interpolatedDataset.alt) * units(interpolatedDataset.alt.units)).to('m')
    pressureLevels = alt2press(altitudes)
    densBasic = (pressureLevels / (R * Ts)).to('kg/m^3')
    densBasic2D = densBasic[:, np.newaxis]
    densBasic3D = densBasic[:, np.newaxis, np.newaxis]

    v = (np.array(interpolatedDataset[tomlConfig['meridionalWindName']]) * units(interpolatedDataset[tomlConfig['meridionalWindName']].units)).to('m/s')
    if tomlConfig['verticalWindType'] == 'omega':
        w = (-np.array(interpolatedDataset[tomlConfig['verticalWindName']]) * 
                units(interpolatedDataset[tomlConfig['verticalWindName']].units) / (densBasic3D * gEarth)).to('m/s')
    else:
        w = (np.array(interpolatedDataset[tomlConfig['verticalWindName']]) * units(interpolatedDataset[tomlConfig['verticalWindName']].units)).to('m/s')

    if tomlConfig['temperatureType'] != 'theta':
        theta = ((np.array(interpolatedDataset[tomlConfig['temperatureName']]) * units(interpolatedDataset[tomlConfig['temperatureName']].units)) *
                    (P0 / pressureLevels)[:, np.newaxis, np.newaxis] ** (R/Cp)).to('kelvin')
    else:
        theta = (np.array(interpolatedDataset[tomlConfig['temperatureName']]) * units(interpolatedDataset[tomlConfig['temperatureName']].units)).to('kelvin')

    vBar = np.nanmean(v, 2)
    wBar = np.nanmean(w, 2)
    thetaBar = np.nanmean(theta, 2)

    vPrime = v - vBar[:, :, np.newaxis]
    wPrime = w - wBar[:, :, np.newaxis]
    thetaPrime = theta - thetaBar[:, :, np.newaxis]

    vPrimeThetaPrimeBar = np.nanmean(vPrime * thetaPrime, 2)
    DThetaBarDZ = nanGradient(thetaBar, altitudes.to('m'), axis=0)

    vBarStar = vBar - (1 / densBasic2D) * nanGradient((densBasic2D * vPrimeThetaPrimeBar) / DThetaBarDZ, altitudes.to('m'), axis=0)
    wBarStar = wBar + 1 / (rEarth * cosFi) * nanGradient(cosFi * vPrimeThetaPrimeBar / DThetaBarDZ, latsR, axis=1)

    dataToSave = {}
    if tomlConfig['massSF']:
        massSF = -cosFi * np.flip(cumulative_trapezoid(y=np.flip(densBasic2D * np.nan_to_num(vBarStar), axis=0),
                                                        x=np.flip(altitudes, axis=0), axis=0, initial=0), axis=0)
        massSF = massSF * units('kg/(m*s)')
        dataToSave['massSF'] = [massSF, 'Mass stream function', str(massSF.units)]

    if len(tomlConfig['sinksSources']) != len(tomlConfig['tracerNames']):
        raise ValueError(
            f"sinksSources has {len(tomlConfig['sinksSources'])} entries but tracerNames has "
            f"{len(tomlConfig['tracerNames'])}; they must be the same length"
        )
    for i, ss in enumerate(tomlConfig['sinksSources']):
        if 'half life' in ss and len(ss.split(', ')) != 3:
            raise ValueError(
                f"sinksSources[{i}] = '{ss}' is invalid; expected format: 'half life, N, unit'"
            )

    FourierToSave = {}

    for index, tracer in enumerate(tomlConfig['tracerNames']):
        if units(interpolatedDataset[tracer].units).dimensionality == '[time]': # convert from time units to seconds, mostly due to Age of Air being in years
            chi = (np.array(interpolatedDataset[tracer]) * units(interpolatedDataset[tracer].units)).to_base_units()
        else: # keep original units, because mixing ratio becomes unitsless if base units are taken and info if it is mass or volume mixing ratio is lost
            chi = (np.array(interpolatedDataset[tracer]) * units(interpolatedDataset[tracer].units))
        chiBar = np.nanmean(chi, 2)
        chiPrime = chi - chiBar[:, :, np.newaxis]
        vPrimeChiPrimeBar = np.nanmean(vPrime * chiPrime, 2)
        wPrimeChiPrimeBar = np.nanmean(wPrime * chiPrime, 2)
        DChiBarDZ = nanGradient(chiBar, altitudes.to('m'), axis=0)
        DChiBarDFI = nanGradient(chiBar / rEarth, latsR, axis=1)
        
        MFi = -densBasic2D * (vPrimeChiPrimeBar - DChiBarDZ * vPrimeThetaPrimeBar / DThetaBarDZ)
        Mz = -densBasic2D * (wPrimeChiPrimeBar + DChiBarDFI * vPrimeThetaPrimeBar / DThetaBarDZ)
        divMFi = 1 / (rEarth * cosFi) * nanGradient(MFi * cosFi, latsR, axis=1)
        divMz = nanGradient(Mz, altitudes.to('m'), axis=0)
        # divM = divMFi + divMz
        
        sinkSource = 0 * chi.units / units('s')
        if str.isdigit(tomlConfig['sinksSources'][index]): # if sinkSource is integer
            sinkSource = int(tomlConfig['sinksSources'][index]) * chi.units / units('s')
        elif 'half life' in tomlConfig['sinksSources'][index]:
            halfLife = float(tomlConfig['sinksSources'][index].split(', ')[1])
            halfLifeUnits = units(tomlConfig['sinksSources'][index].split(', ')[2])
            sinkSource = -chiBar * np.log(2) / (halfLife * halfLifeUnits).to_base_units()
        
        chi_bar = chiBar
        divm_z = divMz / densBasic2D
        divm_lat = divMFi / densBasic2D
        m_z = Mz / densBasic2D
        m_lat = MFi / densBasic2D
        adv_z = -wBarStar * DChiBarDZ
        adv_lat = -vBarStar * DChiBarDFI
        dt_sum = sinkSource + divm_z + divm_lat + adv_lat + adv_z
        
        
        dataToSave[f'{tracer}_chi_bar'] = [chi_bar, f'zonal mean value of {tracer}', str(chi_bar.units)]
        dataToSave[f'{tracer}_dt_sum'] = [dt_sum, f'temporal derivative of {tracer} estimated from sum of other components', str(dt_sum.units)]
        dataToSave[f'{tracer}_divm_z'] = [divm_z, 'divergence of vertical eddy flux vector divided by basic density', str(divm_z.units)]
        dataToSave[f'{tracer}_divm_lat'] = [divm_lat, 'divergence of meridional eddy flux vector divided by basic density', str(divm_lat.units)]
        dataToSave[f'{tracer}_m_z'] = [m_z, 'vertical eddy flux vector divided by basic density', str(m_z.units)]
        dataToSave[f'{tracer}_m_lat'] = [m_lat, 'meridional eddy flux vector divided by basic density', str(m_lat.units)]
        dataToSave[f'{tracer}_adv_z'] = [adv_z, f'negated vertical advection of {tracer}', str(adv_z.units)]
        dataToSave[f'{tracer}_adv_lat'] = [adv_lat, f'negated meridional advection of {tracer}', str(adv_lat.units)]
        

        
        if tomlConfig['FourierTransform']:
            # Fourier decomposition of eddy tracer transport component
            ChiPrimeFFT = np.fft.rfft(np.nan_to_num(chiPrime.magnitude), axis=2)
            vPrimeFFT = np.fft.rfft(np.nan_to_num(vPrime.magnitude), axis=2)
            ThetaPrimeFFT = np.fft.rfft(np.nan_to_num(thetaPrime.magnitude), axis=2)
            wPrimeFFT = np.fft.rfft(np.nan_to_num(wPrime.magnitude), axis=2)

            # n_valid: number of finite longitude samples per (alt, lat) point for each cross-product pair.
            # Using n_valid instead of lons.size in the Parseval normalisation ensures
            # that nan_to_num (which zeros NaN positions) stays consistent with the
            # real-space nanmean (which ignores NaN positions).
            _N = lons.size
            n_valid_vc = np.sum(np.isfinite(vPrime.magnitude * chiPrime.magnitude), axis=2, keepdims=True)
            n_valid_vt = np.sum(np.isfinite(vPrime.magnitude * thetaPrime.magnitude), axis=2, keepdims=True)
            n_valid_wc = np.sum(np.isfinite(wPrime.magnitude * chiPrime.magnitude), axis=2, keepdims=True)

            FourTNDBD = {}
            with np.errstate(invalid='ignore'):
                FourTNDBD[f'{tracer}_m_lat_WN'] = (-densBasic3D.magnitude * (
                    np.real(vPrimeFFT * np.conj(ChiPrimeFFT)) / (n_valid_vc * _N / 2)
                    - DChiBarDZ[:, :, np.newaxis].magnitude / DThetaBarDZ[:, :, np.newaxis].magnitude *
                      np.real(vPrimeFFT * np.conj(ThetaPrimeFFT)) / (n_valid_vt * _N / 2)
                )) * units(str(MFi.units))

                FourTNDBD[f'{tracer}_m_z_WN'] = (-densBasic3D.magnitude * (
                    np.real(wPrimeFFT * np.conj(ChiPrimeFFT)) / (n_valid_wc * _N / 2)
                    + DChiBarDFI[:, :, np.newaxis].magnitude / DThetaBarDZ[:, :, np.newaxis].magnitude *
                      np.real(vPrimeFFT * np.conj(ThetaPrimeFFT)) / (n_valid_vt * _N / 2)
                )) * units(str(Mz.units))
            
            FourTNDBD[f'{tracer}_divm_lat_WN'] = (1 / (rEarth * cosFi[:, :, np.newaxis]) * nanGradient(FourTNDBD[f'{tracer}_m_lat_WN'] * cosFi[:, :, np.newaxis], latsR, axis=1))
            
            FourTNDBD[f'{tracer}_divm_z_WN'] = nanGradient(FourTNDBD[f'{tracer}_m_z_WN'], altitudes.to('m'), axis=0)
            
            FourTNDBD[f'{tracer}_divm_WN'] = FourTNDBD[f'{tracer}_divm_z_WN'] + FourTNDBD[f'{tracer}_divm_lat_WN']            
            
            # it probably would be better to remove densBasic3D from FourTNDBD['MFi_WN'] and ['MZ_WN'] and get results divided by basic density
            # but to follow equation in the book, scaling by basc density is done here
            FourT = {}
            for variable in FourTNDBD.keys():
                FourT[variable] = FourTNDBD[variable] / densBasic3D

            # The Nyquist component (k=N//2) appears only once in the DFT (not twice like k=1..N//2-1),
            # so it was over-normalised by a factor of 2; correct that here.
            _nyq = lons.size // 2
            for variable in FourT:
                FourT[variable][:, :, _nyq] = FourT[variable][:, :, _nyq] / 2

            if len(tomlConfig['Waves']) == 1 and tomlConfig['Waves'][0].lower() == 'all':
                FShape = np.zeros((FourT[f'{tracer}_m_lat_WN'].shape[0], FourT[f'{tracer}_m_lat_WN'].shape[1], FourT[f'{tracer}_m_lat_WN'].shape[2] - 1)).shape
                Fourier = {f'{tracer}_m_lat_WN': [np.zeros((FShape)), 'Fourier transform of meridional eddy flux vector divided by basic density', str(m_lat.units)], 
                           f'{tracer}_m_z_WN': [np.zeros((FShape)), 'Fourier transform of vertical eddy flux vector divided by basic density', str(m_z.units)],
                           f'{tracer}_divm_lat_WN': [np.zeros((FShape)), 'Fourier transform of divergence of meridional eddy flux vector divided by basic density', str(divm_lat.units)], 
                           f'{tracer}_divm_z_WN': [np.zeros((FShape)), 'Fourier transform of divergence of vertical eddy flux vector divided by basic density', str(divm_z.units)],
                           f'{tracer}_divm_WN': [np.zeros((FShape)), 'Fourier transform of divergence of eddy flux vector divided by basic density', str(divm_lat.units)]}

                for variable in Fourier.keys():
                    Fourier[variable][0][:, :, :] = FourT[variable][:, :, 1:]
                    
                
            else:
                '''Saving entire Fourier transform result usually significantly increases size of the output file.
                There is an option to save only some waves which are stored in args dictionary as args["Waves"]. 
                args["Waves"] is expected to be a string "all" or a list of strings like "5" or "6-10" where in
                case of "6-10" sum of waves from 6 to 10 will be saved as a single 2d field'''
            
                FShape = np.zeros((FourT[f'{tracer}_m_lat_WN'].shape[0], FourT[f'{tracer}_m_lat_WN'].shape[1], len(tomlConfig['Waves']))).shape
                Fourier = {f'{tracer}_m_lat_WN': [np.zeros((FShape)), 'Fourier transform of meridional eddy flux vector divided by basic density', str(m_lat.units)], 
                           f'{tracer}_m_z_WN': [np.zeros((FShape)), 'Fourier transform of vertical eddy flux vector divided by basic density', str(m_z.units)],
                           f'{tracer}_divm_lat_WN': [np.zeros((FShape)), 'Fourier transform of divergence of meridional eddy flux vector divided by basic density', str(divm_lat.units)], 
                           f'{tracer}_divm_z_WN': [np.zeros((FShape)), 'Fourier transform of divergence of vertical eddy flux vector divided by basic density', str(divm_z.units)],
                           f'{tracer}_divm_WN': [np.zeros((FShape)), 'Fourier transform of divergence of eddy flux vector divided by basic density', str(divm_lat.units)]}

                for variable in Fourier.keys():
                    for i, wave in enumerate(tomlConfig['Waves']):
                        if '-' not in wave:  # if it is a single wave
                            wave = int(wave)
                            Fourier[variable][0][:, :, i] = FourT[variable][:, :, wave]
                        elif 'end' not in wave and '-' in wave:  # if it is a range but not to the last wave
                            waveStart = int(wave.split('-')[0])
                            waveEnd = int(wave.split('-')[1])
                            Fourier[variable][0][:, :, i] = np.nansum(FourT[variable][:, :, waveStart:waveEnd + 1], 2)
                        else:  # if it is a range to the last wave
                            waveStart = int(wave.split('-')[0])
                            Fourier[variable][0][:, :, i] = np.nansum(FourT[variable][:, :, waveStart:], 2)
            
            FourierToSave.update(Fourier)
    
    
    dataToSave['Fourier'] = FourierToSave

    return dataToSave, lats, altitudes




def init_worker(shared_counter):
    ''' store the counter for later use to calculate percent done'''
    global counter
    counter = shared_counter


def mainCalcs(tomlConfig, count, pathsAndTime='', reqVarsWithTracers='', pathDictionary='', reqVars=''):   
    try:
        if tomlConfig['tracerDataInMetFiles']: # if met and tracer data are in the same files.
            timeStamp = list(pathsAndTime.index)[count]
            
            dataset = readAndTransposeData(pathsAndTime['Path'][count], reqVarsWithTracers, tomlConfig['vertDim'], 
                                        tomlConfig['latDim'], tomlConfig['lonDim'],
                                        fillValues=tomlConfig.get('fillValues', []))
            
            interpolatedDataset = interpolateToLogPressure(dataset, reqVarsWithTracers, tomlConfig['verticalDimensionType'], tomlConfig['targetLevels'], 
                                                        tomlConfig['vertDim'], tomlConfig['latDim'], tomlConfig['lonDim'], tomlConfig['pressureName'],)
            
        else:
            timeStamp = list(pathDictionary.keys())[count]

            tracerFilePath = pathDictionary[list(pathDictionary.keys())[count]][0]
            metFilePaths = pathDictionary[list(pathDictionary.keys())[count]][1]
            metFilesWeights = pathDictionary[list(pathDictionary.keys())[count]][2]

            tracerDataset = readAndTransposeData(tracerFilePath, tomlConfig['tracerNames'],
                                                tomlConfig['tracerVertDim'], tomlConfig['tracerLatDim'], tomlConfig['tracerLonDim'],
                                                fillValues=tomlConfig.get('fillValues', []))
            
            metDataset = readDataAndGetWeightedAverage(metFilePaths, metFilesWeights, reqVars,
                                                    tomlConfig['vertDim'], tomlConfig['latDim'], tomlConfig['lonDim'],
                                                    fillValues=tomlConfig.get('fillValues', []))

            interpolatedDataset = interpolateToPressureAndCombineData(tracerDataset, metDataset, reqVars, tomlConfig)



        interpolatedDataset = binData(interpolatedDataset, tomlConfig['binningLat'], tomlConfig['binningLon'])
        dataToSave, lats, thetaLevels = tracerTransport(interpolatedDataset, tomlConfig)
        saveOut(dataToSave, tomlConfig, timeStamp, lats, thetaLevels)
        
        ''' increment the global counter and display percent done '''
        global counter
        # += operation is not atomic, so get a lock:
        with counter.get_lock():
            counter.value += 1

    except Exception as e:
            # Raise am exception that includes the path context
            raise type(e)(f"[pathsAndTime: {pathsAndTime['Path'][count]}] {str(e)}").with_traceback(e.__traceback__)