from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from metpy.units import units
from scipy.integrate import cumulative_trapezoid

from .constants import gEarth, rEarth
from .file_io import readAndTransposeData, readDataAndGetWeightedAverage, saveOut
from .interpolation import interpolateToTheta, interpolateToThetaAndCombineData
from .utils import addRatioUnits, binData, nanGradient


def tracerTransport(interpolatedDataset: xr.Dataset, tomlConfig: dict) -> tuple[dict, np.ndarray, Any]:
    """
    Compute tracer transport diagnostics in isentropic (theta) coordinates.

    Calculates the residual circulation tendencies, eddy flux vectors, and
    their divergences for each tracer listed in ``tomlConfig['tracerNames']``.
    Optionally computes a Fourier decomposition of the eddy flux terms.

    Parameters
    ----------
    interpolatedDataset : xr.Dataset
        Combined tracer and met dataset on a common theta grid with dimensions
        (``'theta'``, ``'lat'``, ``'lon'``).
    tomlConfig : dict
        Configuration dict. Uses ``tracerNames``, ``sinksSources``,
        ``meridionalWindName``, ``verticalWindName``, ``pressureName``,
        ``FourierTransform``, ``Waves``, and ``saveEddyTerms`` keys.

    Returns
    -------
    dataToSave : dict
        Mapping ``{var_name: [data_array, long_name, units]}`` for all output
        variables, including an optional ``'Fourier'`` sub-dict.
    lats : np.ndarray
        Latitude coordinate values in degrees.
    thetaLevels : pint Quantity
        Potential temperature levels (K).
    """
    addRatioUnits()
    lats = np.array(interpolatedDataset.lat)
    lons = np.array(interpolatedDataset.lon)
    latsR = (lats * units.degrees).to('radian').magnitude
    cosFi = np.cos(latsR)[np.newaxis, :]

    thetaLevels = np.array(interpolatedDataset.theta) * units(interpolatedDataset.theta.units)
    pressure = np.array(interpolatedDataset[tomlConfig['pressureName']]) * units(interpolatedDataset[tomlConfig['pressureName']].units)
    pressureBar = np.nanmean(pressure, 2)
    sigmaDensity = (- 1/gEarth * nanGradient(pressure, thetaLevels, axis=0)).to('kg/m/m/K')  # density in isentropic coordinates
    sigmaDensityBar = np.nanmean(sigmaDensity, 2)

    v = (np.array(interpolatedDataset[tomlConfig['meridionalWindName']]) * units(interpolatedDataset[tomlConfig['meridionalWindName']].units)).to('m/s')
    q = (np.array(interpolatedDataset[tomlConfig['verticalWindName']]) * units(interpolatedDataset[tomlConfig['verticalWindName']].units)).to('K/s') 

    vBarStar = np.nanmean(sigmaDensity * v, axis=2) / np.nanmean(sigmaDensity, axis=2)
    qBarStar = np.nanmean(sigmaDensity * q, axis=2) / np.nanmean(sigmaDensity, axis=2)

    sigmaDensity_v_prime =  (sigmaDensity * v) - np.nanmean((sigmaDensity * v), axis=2)[:, :, np.newaxis]
    sigmaDensity_q_prime =  (sigmaDensity * q) - np.nanmean((sigmaDensity * q), axis=2)[:, :, np.newaxis]

    FourierToSave = {}
    if tomlConfig['FourierTransform']:
        sigmaDensity_v_primeFFT = np.fft.rfft(np.nan_to_num(sigmaDensity_v_prime.magnitude), axis=2)
        sigmaDensity_q_primeFFT = np.fft.rfft(np.nan_to_num(sigmaDensity_q_prime.magnitude), axis=2)


    dataToSave = {}
    dataToSave['PRESS'] = [pressureBar, 'zonal mean pressure', str(pressureBar.units)]

    if tomlConfig['massSF']:
        massSF = ((cosFi / gEarth) * (np.flip(cumulative_trapezoid(y=np.flip(np.nan_to_num(vBarStar), axis=0), x=np.flip(pressureBar, axis=0), axis=0, initial=0), 
                                                    axis=0) * units(str(vBarStar.units)) * units(str(pressureBar.units)))).to('kg/m/s')
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

    for index, tracer in enumerate(tomlConfig['tracerNames']):
        if units(interpolatedDataset[tracer].units).dimensionality == '[time]': # convert from time units to seconds, mostly due to Age of Air being in years
            chi = (np.array(interpolatedDataset[tracer]) * units(interpolatedDataset[tracer].units)).to_base_units()
        else: # keep original units, because mixing ratio becomes unitsless if base units are taken and info if it is mass or volume mixing ratio is lost
            chi = (np.array(interpolatedDataset[tracer]) * units(interpolatedDataset[tracer].units))
        chiBar = np.nanmean(chi, 2)
        chiPrime = chi - chiBar[:, :, np.newaxis]
        DChiBarDTheta = nanGradient(chiBar, thetaLevels, axis=0)
        DChiBarDPhi = nanGradient(chiBar, latsR, axis=1)
        
        M_phi = -np.nanmean(sigmaDensity_v_prime * chiPrime, 2)
        M_theta = -np.nanmean(sigmaDensity_q_prime * chiPrime, 2)
        
        div_M_phi = 1 / (rEarth * cosFi) * nanGradient(M_phi * cosFi, latsR, axis=1)
        div_M_theta = nanGradient(M_theta, thetaLevels, axis=0)
                
        sinkSource = 0 * chi.units / units('s')
        if str.isdigit(tomlConfig['sinksSources'][index]): # if sinkSource is integer
            sinkSource = int(tomlConfig['sinksSources'][index]) * chi.units / units('s')
        elif 'half life' in tomlConfig['sinksSources'][index]:
            halfLife = float(tomlConfig['sinksSources'][index].split(', ')[1])
            halfLifeUnits = units(tomlConfig['sinksSources'][index].split(', ')[2])
            sinkSource = -chiBar * np.log(2) / (halfLife * halfLifeUnits).to_base_units()

        divm_theta = div_M_theta / sigmaDensityBar
        divm_lat = div_M_phi / sigmaDensityBar
        m_theta = M_theta / sigmaDensityBar
        m_lat = M_phi / sigmaDensityBar
        adv_theta = -qBarStar * DChiBarDTheta
        adv_lat = -vBarStar/rEarth * DChiBarDPhi
        dt_sum = sinkSource + divm_theta + divm_lat + adv_lat + adv_theta
        

        dataToSave[f'{tracer}_chi_bar'] = [chiBar, f'zonal mean value of {tracer}', str(chiBar.units)]
        dataToSave[f'{tracer}_dt_sum'] = [dt_sum, f'temporal derivative of {tracer} estimated from sum of sinkSource, circulation and eddy mixing tendencies', str(dt_sum.units)]
        dataToSave[f'{tracer}_divm_theta'] = [divm_theta, f'vertical eddy mixing tendency of {tracer}', str(divm_theta.units)]
        dataToSave[f'{tracer}_divm_lat'] = [divm_lat, f'horizontal eddy mixing tendency of {tracer}', str(divm_lat.units)]
        dataToSave[f'{tracer}_m_theta'] = [m_theta, f'vertical eddy flux vector of {tracer} divided by basic density', str(m_theta.units)]
        dataToSave[f'{tracer}_m_lat'] = [m_lat, f'meridional eddy flux vector of {tracer} divided by basic density', str(m_lat.units)]
        dataToSave[f'{tracer}_adv_theta'] = [adv_theta, f'vertical residual circulation tendency of {tracer}', str(adv_theta.units)]
        dataToSave[f'{tracer}_adv_lat'] = [adv_lat, f'meridional residual circulation tendency of {tracer}', str(adv_lat.units)]

        if tomlConfig['FourierTransform']:
            ChiPrimeFFT = np.fft.rfft(np.nan_to_num(chiPrime.magnitude), axis=2)

            # n_valid: number of finite longitude samples per (theta, lat) point.
            # Using n_valid instead of lons.size in the Parseval normalisation ensures
            # that nan_to_num (which zeros NaN positions) stays consistent with the
            # real-space nanmean (which ignores NaN positions).
            n_valid_v = np.sum(np.isfinite(sigmaDensity_v_prime.magnitude), axis=2, keepdims=True)
            n_valid_q = np.sum(np.isfinite(sigmaDensity_q_prime.magnitude), axis=2, keepdims=True)
            _N = lons.size  # full FFT length

            # / (n_valid * N / 2) ensures sum of _WN over all wave indices equals the real-space eddy flux
            FourTNDBD = {}
            with np.errstate(invalid='ignore'):
                FourTNDBD[f'{tracer}_m_lat_WN'] = (-np.real(sigmaDensity_v_primeFFT * np.conj(ChiPrimeFFT)) /
                                                    (n_valid_v * _N / 2)) * units(str(m_lat.units))

                FourTNDBD[f'{tracer}_m_theta_WN'] = (-np.real(sigmaDensity_q_primeFFT * np.conj(ChiPrimeFFT)) /
                                                      (n_valid_q * _N / 2)) * units(str(m_theta.units))

                FourTNDBD[f'{tracer}_divm_lat_WN'] = (1 / (rEarth * cosFi[:, :, np.newaxis]) * nanGradient(FourTNDBD[f'{tracer}_m_lat_WN'] * cosFi[:, :, np.newaxis], latsR, axis=1))

                FourTNDBD[f'{tracer}_divm_theta_WN'] = nanGradient(FourTNDBD[f'{tracer}_m_theta_WN'], thetaLevels, axis=0)

                FourTNDBD[f'{tracer}_divm_WN'] = FourTNDBD[f'{tracer}_divm_theta_WN'] + FourTNDBD[f'{tracer}_divm_lat_WN']

            FourT = {}
            for variable in FourTNDBD.keys():
                FourT[variable] = FourTNDBD[variable] / sigmaDensityBar[:,:,np.newaxis]

            # The Nyquist component (k=N//2) appears only once in the DFT (not twice like k=1..N//2-1),
            # so it was over-normalised by a factor of 2; correct that here.
            _nyq = _N // 2
            for variable in FourT:
                FourT[variable][:, :, _nyq] = FourT[variable][:, :, _nyq] / 2

            if len(tomlConfig['Waves']) == 1 and tomlConfig['Waves'][0].lower() == 'all':
                FShape = np.zeros((FourT[f'{tracer}_m_lat_WN'].shape[0], FourT[f'{tracer}_m_lat_WN'].shape[1], FourT[f'{tracer}_m_lat_WN'].shape[2] - 1)).shape
                Fourier = {f'{tracer}_m_lat_WN': [np.zeros((FShape)), f'Fourier transform of meridional eddy flux vector of {tracer} divided by basic density', str(m_lat.units)], 
                            f'{tracer}_m_theta_WN': [np.zeros((FShape)), f'Fourier transform of vertical eddy flux vector of {tracer} divided by basic density', str(m_theta.units)],
                            f'{tracer}_divm_lat_WN': [np.zeros((FShape)), f'Fourier transform of horizontal eddy mixing tendency of {tracer}', str(divm_lat.units)], 
                            f'{tracer}_divm_theta_WN': [np.zeros((FShape)), f'Fourier transform of vertical eddy mixing tendency of {tracer}', str(divm_theta.units)],
                            f'{tracer}_divm_WN': [np.zeros((FShape)), f'Fourier transform of eddy mixing tendency of {tracer}', str(divm_lat.units)]}

                for variable in Fourier.keys():
                    Fourier[variable][0][:, :, :] = FourT[variable][:, :, 1:]
                    
                
            else:
                # Save only the wavenumber bands listed in tomlConfig['Waves'].
                # Each entry is either a single wave (e.g. "5") or a range (e.g. "6-10"),
                # where a range is summed into one 2-D field.
                FShape = np.zeros((FourT[f'{tracer}_m_lat_WN'].shape[0], FourT[f'{tracer}_m_lat_WN'].shape[1], len(tomlConfig['Waves']))).shape
                Fourier = {f'{tracer}_m_lat_WN': [np.zeros((FShape)), f'Fourier transform of meridional eddy flux vector of {tracer} divided by basic density', str(m_lat.units)], 
                            f'{tracer}_m_theta_WN': [np.zeros((FShape)), f'Fourier transform of vertical eddy flux vector of {tracer} divided by basic density', str(m_theta.units)],
                            f'{tracer}_divm_lat_WN': [np.zeros((FShape)), f'Fourier transform of horizontal eddy mixing tendency of {tracer}', str(divm_lat.units)], 
                            f'{tracer}_divm_theta_WN': [np.zeros((FShape)), f'Fourier transform of vertical eddy mixing tendency of {tracer}', str(divm_theta.units)],
                            f'{tracer}_divm_WN': [np.zeros((FShape)), f'Fourier transform of eddy mixing tendency of {tracer}', str(divm_lat.units)]}

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


    return dataToSave, lats, thetaLevels


def init_worker(shared_counter: Any) -> None:
    """
    Initialise a pool worker by storing the shared progress counter.

    Parameters
    ----------
    shared_counter : multiprocessing.Value
        Shared integer incremented after each file is processed.
    """
    global counter
    counter = shared_counter


def mainCalcs(tomlConfig: dict, count: int, pathsAndTime: Any = '', reqVarsWithTracers: Any = '', pathDictionary: Any = '', reqVars: Any = '') -> None:
    """
    Process one tracer-transport time step in theta coordinates and save output.

    Reads input file(s), interpolates to the theta grid, runs
    :func:`tracerTransport`, and writes results to a NetCDF file via
    :func:`~tem_pkg.file_io.saveOut`.

    Called by the multiprocessing pool; increments the shared ``counter`` on
    success. Re-raises exceptions with the offending file path prepended to
    the message.

    Parameters
    ----------
    tomlConfig : dict
        Configuration dict.
    count : int
        Index into *pathsAndTime* or *pathDictionary* for this worker.
    pathsAndTime : pd.DataFrame or ``''``
        Combined met+tracer file paths; used when ``tracerDataInMetFiles`` is
        True.
    reqVarsWithTracers : list of str or ``''``
        Variable names including tracers; used with *pathsAndTime*.
    pathDictionary : dict or ``''``
        Mapping of tracer timestamps to ``[tracer_path, met_paths, weights]``;
        used when met and tracer files are separate.
    reqVars : list of str or ``''``
        Met-only variable names; used with *pathDictionary*.
    """
    try:
        if tomlConfig['tracerDataInMetFiles']: # if met and tracer data are in the same files.
            timeStamp = list(pathsAndTime.index)[count]
            
            dataset = readAndTransposeData(pathsAndTime['Path'].iloc[count], reqVarsWithTracers, tomlConfig['vertDim'],
                                        tomlConfig['latDim'], tomlConfig['lonDim'],
                                        timeDimName=tomlConfig.get('timeDim', ''))
            interpolatedDataset = interpolateToTheta(dataset, reqVarsWithTracers, tomlConfig)
            
        else:
            timeStamp = list(pathDictionary.keys())[count]

            tracerFilePath = pathDictionary[list(pathDictionary.keys())[count]][0]
            metFilePaths = pathDictionary[list(pathDictionary.keys())[count]][1]
            metFilesWeights = pathDictionary[list(pathDictionary.keys())[count]][2]

            tracerDataset = readAndTransposeData(tracerFilePath, tomlConfig['tracerNames'],
                                                tomlConfig['tracerVertDim'], tomlConfig['tracerLatDim'], tomlConfig['tracerLonDim'],
                                                timeDimName=tomlConfig.get('tracerTimeDim', ''))
            
            metDataset = readDataAndGetWeightedAverage(metFilePaths, metFilesWeights, reqVars,
                                                    tomlConfig['vertDim'], tomlConfig['latDim'], tomlConfig['lonDim'])

            interpolatedDataset = interpolateToThetaAndCombineData(tracerDataset, metDataset, reqVars, tomlConfig)



        interpolatedDataset = binData(interpolatedDataset, tomlConfig['binningLat'], tomlConfig['binningLon'])
        dataToSave, lats, thetaLevels = tracerTransport(interpolatedDataset, tomlConfig)
        saveOut(dataToSave, tomlConfig, timeStamp, lats, thetaLevels)
        
        global counter
        # += operation is not atomic, so get a lock:
        with counter.get_lock():
            counter.value += 1

    except Exception as e:
            # Raise an exception that includes the path context
            raise type(e)(f"[pathsAndTime: {pathsAndTime['Path'].iloc[count]}] {str(e)}").with_traceback(e.__traceback__)