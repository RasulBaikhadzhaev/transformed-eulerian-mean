from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from metpy.units import units
from scipy.integrate import cumulative_trapezoid

from .constants import P0, Cp, R, Ts, gEarth, rEarth
from .file_io import _ALT_VERT_COORD, readAndTransposeData, readDataAndGetWeightedAverage, saveOut
from .interpolation import alt2press, interpolateToLogPressure, interpolateToPressureAndCombineData
from .utils import addRatioUnits, apply_waves_banding, binData, nanGradient


def tracerTransport(interpolatedDataset: xr.Dataset, tomlConfig: dict) -> tuple[dict, np.ndarray, Any]:
    """
    Compute tracer transport diagnostics in log-pressure altitude coordinates.

    Calculates the residual circulation tendencies, eddy flux vectors, and
    their divergences for each tracer in ``tomlConfig['tracerNames']``.
    Optionally computes a Fourier decomposition of the eddy flux terms.

    Parameters
    ----------
    interpolatedDataset : xr.Dataset
        Combined tracer and met dataset on a common log-pressure altitude grid
        with dimensions (``'alt'``, ``'lat'``, ``'lon'``).
    tomlConfig : dict
        Configuration dict. Uses ``tracerNames``, ``sinksSources``,
        ``meridionalWindName``, ``verticalWindName``, ``verticalWindType``,
        ``temperatureName``, ``temperatureType``, ``FourierTransform``,
        ``Waves``, and ``saveEddyTerms`` keys.

    Returns
    -------
    dataToSave : dict
        Mapping ``{var_name: [data_array, long_name, units]}`` for all output
        variables, including an optional ``'Fourier'`` sub-dict.
    lats : np.ndarray
        Latitude coordinate values in degrees.
    altitudes : pint Quantity
        Log-pressure altitude levels in metres.
    """
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

        sinkSource = 0 * chi.units / units('s')
        if tomlConfig['sinksSources'][index].isdigit():
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

            # Compute ρ-weighted cross-spectra as plain numpy arrays (no units yet).
            # Dividing by ρ₀ is the step that gives the correct physical units;
            # attaching units beforehand would require tracking ρv′χ′ units through
            # gradient operations and is error-prone.
            rho3 = densBasic3D.magnitude
            dcdz = DChiBarDZ[:, :, np.newaxis].magnitude
            dtdz = DThetaBarDZ[:, :, np.newaxis].magnitude
            dcdf = DChiBarDFI[:, :, np.newaxis].magnitude
            altitudes_m = altitudes.to('m').magnitude
            with np.errstate(invalid='ignore'):
                m_lat_spec = -rho3 * (
                    np.real(vPrimeFFT * np.conj(ChiPrimeFFT)) / (n_valid_vc * _N / 2)
                    - dcdz / dtdz * np.real(vPrimeFFT * np.conj(ThetaPrimeFFT)) / (n_valid_vt * _N / 2)
                )
                m_z_spec = -rho3 * (
                    np.real(wPrimeFFT * np.conj(ChiPrimeFFT)) / (n_valid_wc * _N / 2)
                    + dcdf / dtdz * np.real(vPrimeFFT * np.conj(ThetaPrimeFFT)) / (n_valid_vt * _N / 2)
                )
            divm_lat_spec = (1 / (rEarth.magnitude * cosFi[:, :, np.newaxis]) *
                             nanGradient(m_lat_spec * cosFi[:, :, np.newaxis], latsR, axis=1))
            divm_z_spec = nanGradient(m_z_spec, altitudes_m, axis=0)
            divm_spec = divm_lat_spec + divm_z_spec

            FourT = {
                f'{tracer}_m_lat_WN': (m_lat_spec / rho3) * m_lat.units,
                f'{tracer}_m_z_WN': (m_z_spec / rho3) * m_z.units,
                f'{tracer}_divm_lat_WN': (divm_lat_spec / rho3) * divm_lat.units,
                f'{tracer}_divm_z_WN': (divm_z_spec / rho3) * divm_z.units,
                f'{tracer}_divm_WN': (divm_spec / rho3) * divm_lat.units,
            }

            # The Nyquist component (k=N//2) appears only once in the DFT (not twice like k=1..N//2-1),
            # so it was over-normalised by a factor of 2; correct that here.
            _nyq = _N // 2
            for variable in FourT:
                FourT[variable][:, :, _nyq] = FourT[variable][:, :, _nyq] / 2

            waves_cfg = tomlConfig['Waves']
            if len(waves_cfg) == 1 and waves_cfg[0].lower() == 'all':
                waves_cfg = [str(k) for k in range(1, _N // 2 + 1)]

            long_names = {
                f'{tracer}_m_lat_WN': 'Fourier transform of meridional eddy flux vector divided by basic density',
                f'{tracer}_m_z_WN': 'Fourier transform of vertical eddy flux vector divided by basic density',
                f'{tracer}_divm_lat_WN': 'Fourier transform of divergence of meridional eddy flux vector divided by basic density',
                f'{tracer}_divm_z_WN': 'Fourier transform of divergence of vertical eddy flux vector divided by basic density',
                f'{tracer}_divm_WN': 'Fourier transform of divergence of eddy flux vector divided by basic density',
            }
            unit_strs = {
                f'{tracer}_m_lat_WN': str(m_lat.units),
                f'{tracer}_m_z_WN': str(m_z.units),
                f'{tracer}_divm_lat_WN': str(divm_lat.units),
                f'{tracer}_divm_z_WN': str(divm_z.units),
                f'{tracer}_divm_WN': str(divm_lat.units),
            }
            # Strip k=0 from FourT before banding (k=0 is the zonal mean, not an eddy)
            FourT_stripped = {var: arr[:, :, 1:] for var, arr in FourT.items()}
            Fourier = {}
            for variable, arr in FourT_stripped.items():
                banded = apply_waves_banding(arr.magnitude, waves_cfg)
                Fourier[variable] = [banded, long_names[variable], unit_strs[variable]]
            
            FourierToSave.update(Fourier)
            dataToSave['Fourier'] = FourierToSave

    return dataToSave, lats, altitudes




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


def mainCalcs(tomlConfig: dict, task_path: Any = '', reqVarsWithTracers: Any = '', task_entry: Any = '', reqVars: Any = '') -> None:
    """
    Process one tracer-transport time step in log-pressure coordinates and save output.

    Reads input file(s), interpolates to the log-pressure grid, runs
    :func:`tracerTransport`, and writes results to a NetCDF file via
    :func:`~transformed_eulerian_mean.file_io.saveOut`.

    Called by the multiprocessing pool; increments the shared ``counter`` on
    success. Re-raises exceptions with the offending file path prepended to
    the message.

    Parameters
    ----------
    tomlConfig : dict
        Configuration dict.
    task_path : tuple (timestamp, path) or ``''``
        Pre-extracted (timestamp, file path) for this task; used when
        ``tracerDataInMetFiles`` is True.
    reqVarsWithTracers : list of str or ``''``
        Variable names including tracers; used with *task_path*.
    task_entry : tuple (timestamp, tracer_path, met_paths, weights) or ``''``
        Pre-extracted entry for this task; used when met and tracer files are
        separate.
    reqVars : list of str or ``''``
        Met-only variable names; used with *task_entry*.
    """
    try:
        if tomlConfig['tracerDataInMetFiles']:
            timeStamp, filePath = task_path

            dataset = readAndTransposeData(filePath, reqVarsWithTracers, tomlConfig['vertDim'],
                                        tomlConfig['latDim'], tomlConfig['lonDim'],
                                        timeDimName=tomlConfig.get('timeDim', ''),
                                        fillValues=tomlConfig.get('fillValues', []))

            interpolatedDataset = interpolateToLogPressure(dataset, reqVarsWithTracers, tomlConfig['verticalDimensionType'], tomlConfig['targetLevels'],
                                                        tomlConfig['vertDim'], tomlConfig['latDim'], tomlConfig['lonDim'], tomlConfig['pressureName'],)

        else:
            timeStamp, tracerFilePath, metFilePaths, metFilesWeights = task_entry

            tracerDataset = readAndTransposeData(tracerFilePath, tomlConfig['tracerNames'],
                                                tomlConfig['tracerVertDim'], tomlConfig['tracerLatDim'], tomlConfig['tracerLonDim'],
                                                timeDimName=tomlConfig.get('tracerTimeDim', ''),
                                                fillValues=tomlConfig.get('fillValues', []))

            metDataset = readDataAndGetWeightedAverage(metFilePaths, metFilesWeights, reqVars,
                                                    tomlConfig['vertDim'], tomlConfig['latDim'], tomlConfig['lonDim'],
                                                    fillValues=tomlConfig.get('fillValues', []))

            interpolatedDataset = interpolateToPressureAndCombineData(tracerDataset, metDataset, reqVars, tomlConfig)

        interpolatedDataset = binData(interpolatedDataset, tomlConfig['binningLat'], tomlConfig['binningLon'])
        dataToSave, lats, altLevels = tracerTransport(interpolatedDataset, tomlConfig)
        saveOut(dataToSave, tomlConfig, timeStamp, lats, altLevels, _ALT_VERT_COORD)

        global counter
        # += operation is not atomic, so get a lock:
        with counter.get_lock():
            counter.value += 1

    except Exception as e:
        try:
            path_ctx = task_path[1] if task_path != '' else task_entry[1]
        except Exception:
            path_ctx = '?'
        raise type(e)(f"[path: {path_ctx}] {str(e)}").with_traceback(e.__traceback__)