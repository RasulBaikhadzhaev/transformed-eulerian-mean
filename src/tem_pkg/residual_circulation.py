import warnings

import numpy as np
import pandas as pd
import xarray as xr
from metpy.units import units
from scipy.integrate import cumulative_trapezoid

from .constants import P0, Cp, R, Ts, angVeloEarth, gEarth, rEarth
from .file_io import readAndTransposeData
from .interpolation import alt2press, interpolateToLogPressure
from .utils import nanGradient


def TEMCalcs(tomlConfig, datasetLogPress, time):
    """
    Description:
    Calculates residual mean meridional flow (VBarStar, WBarStar), Eliassen-Pulm flux, its vertical and latitudinal
    components, its divergence and residual mean meridional stream function. Can also perform fourier decomposition of EPF

    Parameters
    __________
    tomlConfig       : dictionary containing parameters from toml configuration file
    datasetLogPress  : unput dataset in log pressure coordinates 
    time             : time of the input dataset

    Returns
    _______
    dsOut       : xarray dataset containing results of the calculation
    """


    # calculations
    # some values are nan at every longtitudal point, so mean gives warning Mean of empty slice
    # warnings.filterwarnings('ignore')  # ignore all warnings

    if 'deg N' in datasetLogPress.lat.units:
        datasetLogPress.lat.attrs['units'] = datasetLogPress.lat.units.replace('deg N', 'degree')
    latsR = (np.array(datasetLogPress.lat) * units(datasetLogPress.lat.units)).to('radian')
    cosFi = np.cos(latsR)[np.newaxis, :]
    coriolisParameter = 2 * angVeloEarth * np.sin(latsR)[np.newaxis, :]

    altitudes = (np.array(datasetLogPress.alt) * units(datasetLogPress.alt.units)).to('m')
    pressureLevels = alt2press(altitudes)
    densBasic = (pressureLevels / (R * Ts)).to('kg/m^3')
    densBasic2D = densBasic[:, np.newaxis]
    densBasic3D = densBasic[:, np.newaxis, np.newaxis]

    u = (np.array(datasetLogPress[tomlConfig['zonalWindName']]) * units(datasetLogPress[tomlConfig['zonalWindName']].units)).to('m/s')
    v = (np.array(datasetLogPress[tomlConfig['meridionalWindName']]) * units(datasetLogPress[tomlConfig['meridionalWindName']].units)).to('m/s')
    if tomlConfig['temperatureType'] != 'theta':
        # if temperature is given, estimate potential temperature theta from temperature and pressure
        theta = ((np.array(datasetLogPress[tomlConfig['temperatureName']]) * units(datasetLogPress[tomlConfig['temperatureName']].units)) * 
                    (P0 / pressureLevels)[:, np.newaxis, np.newaxis] ** (R/Cp)).to('kelvin')
    else:
        theta = (np.array(datasetLogPress[tomlConfig['temperatureName']]) * units(datasetLogPress[tomlConfig['temperatureName']].units)).to('kelvin')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        uBar = np.nanmean(u, 2)
        vBar = np.nanmean(v, 2)
        thetaBar = np.nanmean(theta, 2)

        uPrime = u - uBar[:, :, np.newaxis]
        vPrime = v - vBar[:, :, np.newaxis]
        thetaPrime = theta - thetaBar[:, :, np.newaxis]

        DUBarDZ = nanGradient(uBar, altitudes, axis=0)
        DThetaBarDZ = nanGradient(thetaBar, altitudes, axis=0)
        vPrimeThetaPrimeBar = np.nanmean(vPrime * thetaPrime, 2)
        vPrimeUPrimeBar = np.nanmean(vPrime * uPrime, 2)
        uBarCosFiDFI = nanGradient(uBar * cosFi, latsR, axis=1)

        vBarStar = vBar - (1 / densBasic2D) * nanGradient((densBasic2D * vPrimeThetaPrimeBar) / DThetaBarDZ, altitudes, axis=0)
        EPFLat = densBasic2D * rEarth * cosFi * (DUBarDZ * vPrimeThetaPrimeBar / DThetaBarDZ - vPrimeUPrimeBar)

        massSF = -cosFi * np.flip(cumulative_trapezoid(y=np.flip(densBasic2D * np.nan_to_num(vBarStar), axis=0),
                                                        x=np.flip(altitudes, axis=0), axis=0, initial=0), axis=0)
        massSF = massSF * units('kg/(m*s)')
        # While calculating mass stream function [np.nan_to_num(vBarStar)] is conveting all nan values
        # of vBarStar to 0, it could be done here because we only have nan values at top and bottom edges.
        # At the top nan values appear when there is no information about this level or above and at the
        # bottom when there is no information about this level or levels below thus converting nan to zero should
        # not affect integratoin using cummulative trapezoid

        if tomlConfig['verticalWindType'].lower() != 'missing': # if vertical wind data is present in input dataset
            if tomlConfig['verticalWindType'] in ['W', 'w']:
                w = (np.array(datasetLogPress[tomlConfig['verticalWindName']]) * units(datasetLogPress[tomlConfig['verticalWindName']].units)).to('m/s')
            else:
                # convert OMEGA to W assuming hydrostatic balance
                w = ((-np.array(datasetLogPress[tomlConfig['verticalWindName']]) * units(datasetLogPress[tomlConfig['verticalWindName']].units)) / 
                    (densBasic * gEarth)[:, np.newaxis, np.newaxis]).to('m/s')
            wBar = np.nanmean(w, 2)  # [m/s]
            wPrime = w - wBar[:, :, np.newaxis]  # [m/s]
            wBarStar = wBar + 1 / (rEarth * cosFi) * nanGradient(cosFi * vPrimeThetaPrimeBar / DThetaBarDZ, latsR, axis=1)
            wPrimeUPrimeBar = np.nanmean(wPrime * uPrime, 2)
            EPFVert = densBasic2D * rEarth * cosFi * ((coriolisParameter - 1 / (rEarth * cosFi) * uBarCosFiDFI) *
                                                vPrimeThetaPrimeBar / DThetaBarDZ - wPrimeUPrimeBar)
        else: # if vertical wind data is not present in the input dataset
            DPsyDFi = nanGradient(massSF, latsR, axis=1)
            wBarStar = (1 / (rEarth * densBasic2D * cosFi)) * DPsyDFi
            EPFVert = densBasic2D * rEarth * cosFi * ((coriolisParameter - 1 / (rEarth * cosFi) * uBarCosFiDFI) *
                                                vPrimeThetaPrimeBar / DThetaBarDZ)

    divEPFLat = 1 / (rEarth * cosFi) * nanGradient(EPFLat * cosFi, latsR, axis=1)
    divEPFVert = nanGradient(EPFVert, altitudes, axis=0)

    # divEPFLat = divEPFLat2/densB*rEarth*cosFi[np.newaxis,:]   # rescale data for better ploting results
    # divEPFVert = divEPFVert2/densB*rEarth*cosFi[np.newaxis,:]

    divEPF = divEPFLat + divEPFVert


    # create output xarray dataset
    dsOut = xr.Dataset()
    dsOut.attrs['Title'] = 'Transformed Eulerian Mean computed from ' + tomlConfig['inputDataDescription']

    # prepare data and its attributes to be written to the output dataset
    dataToSave = {}
    dataToSave = {   # A dictionary containing variables names, data, full names, and units
            'THETA': (thetaBar, 'potential temperature', str(thetaBar.units)),
            'U': (uBar, 'zonal wind', str(uBar.units)),
            'V': (vBar, 'meridional wind', str(vBar.units)),
            'V_RES_STD': (vBarStar.magnitude, 'V component of residual mean meridonal circulation', str(vBarStar.units)),
            'W_RES_STD': (wBarStar.magnitude, 'W component of residual mean meridonal circulation', str(wBarStar.units)),
            'EPF_vert': (EPFVert.magnitude, 'vertical component of Eliassen-Pulm flux', str(EPFVert.units)),
            'EPF_lat': (EPFLat.magnitude, 'latitudinal component of Eliassen-Pulm flux', str(EPFLat.units)),
            'div_EPF_vert': (divEPFVert.magnitude, 'vertical component of Eliassen-Pulm flux divergence', str(divEPFVert.units)),
            'div_EPF_lat': (divEPFLat.magnitude, 'zonal component of Eliassen-Pulm flux divergence', str(divEPFLat.units)),
            'div_EPF': (divEPF.magnitude, 'Eliassen-Pulm flux divergence', str(divEPF.units)),
            'MASS_SF_RES_STD': (massSF.magnitude, 'stream function of the (barStar) residual flow', str(massSF.units))
        }
    if tomlConfig['verticalWindType'].lower() != 'missing':
        dataToSave['W'] = (wBar, 'vertical wind', str(wBar.units))
        
    if tomlConfig['saveEddyTerms']:
        dataToSave['vPrimeUPrimeBar'] = (vPrimeUPrimeBar, 'zonal mean horizontal eddy momentum flux', str(vPrimeUPrimeBar.units))
        dataToSave['vPrimeThetaPrimeBar'] = (vPrimeThetaPrimeBar, 'zonal mean horizontal eddy heat flux', str(vPrimeThetaPrimeBar.units))


        vPrimeThetaPrime = vPrime * thetaPrime
        vPrimeUPrime = vPrime * uPrime
        dataToSaveWithLongitude = {
            'vPrime': (vPrime, 'deviation of meridional wind from zonal mean value', str(vPrime.units)),
            'uPrime': (uPrime, 'deviation of zonal wind from zonal mean value', str(uPrime.units)),
            'thetaPrime': (thetaPrime, 'deviation of potential temperature from zonal mean value', str(thetaPrime.units)),
            'vPrimeThetaPrime': (vPrimeThetaPrime, 'horizontal eddy heat flux', str(vPrimeThetaPrime.units)),
            'vPrimeUPrime': (vPrimeUPrime, 'horizontal eddy momentum flux', str(vPrimeUPrime.units))
            } 

        if tomlConfig['verticalWindType'].lower() != 'missing': # if vertical wind data is present in input dataset
            dataToSave['wPrimeUPrimeBar'] = (wPrimeUPrimeBar, 'zonal mean vertical eddy momentum flux', str(wPrimeUPrimeBar.units))
            wPrimeUPrime = wPrime * uPrime
            dataToSaveWithLongitude['wPrime'] = (wPrime, 'deviation of vertical wind from zonal mean value', str(wPrime.units))
            dataToSaveWithLongitude['wPrimeUPrime'] = (wPrimeUPrime, 'mean vertical eddy momentum flux', str(wPrimeUPrime.units))

        # add results with logitude dimension to the output dataset CHECK IF TIME ATTRIBUTE IS STILL REQUIRED!!!!
        for variable in dataToSaveWithLongitude.keys():
            dsOut[variable] = (('alt', 'lat', 'lon'), np.single(dataToSaveWithLongitude[variable][0]))
            getattr(dsOut, variable).attrs['long_name'] = dataToSaveWithLongitude[variable][1]
            getattr(dsOut, variable).attrs['units'] = dataToSaveWithLongitude[variable][2]
            
        dsOut.coords['lon'] = np.array(datasetLogPress.lon)
        dsOut.lon.attrs['long_name'] = 'longitude'
        dsOut.lon.attrs['units'] = str(datasetLogPress.lon.units)
        
    # add main results to output dataset CHECK IF TIME ATTRIBUTE IS STILL REQUIRED!!!!
    for variable in dataToSave.keys():
        dsOut[variable] = (('alt', 'lat'), np.single(dataToSave[variable][0]))
        getattr(dsOut, variable).attrs['long_name'] = dataToSave[variable][1]
        getattr(dsOut, variable).attrs['units'] = dataToSave[variable][2]

    dsOut.coords['alt'] = np.array(datasetLogPress.alt)
    dsOut.alt.attrs['long_name'] = 'log-pressure vertical coordinate defined by z=-H*ln(p/ps), H=7 km, ps=1000 hPa'
    dsOut.alt.attrs['units'] = str(datasetLogPress.alt.units)
    dsOut.coords['lat'] = np.array(datasetLogPress.lat)
    dsOut.lat.attrs['long_name'] = 'latitude'
    dsOut.lat.attrs['units'] = str(datasetLogPress.lat.units)
    dsOut.coords['time'] = [time]


    if tomlConfig['FourierTransform'] and tomlConfig['verticalWindType'].lower() != 'missing':
        # Fourier decomposition of EPF
        uPrimeFFT = np.fft.rfft(np.nan_to_num(uPrime.magnitude), axis=2)
        vPrimeFFT = np.fft.rfft(np.nan_to_num(vPrime.magnitude), axis=2)
        thetaPrimeFFT = np.fft.rfft(np.nan_to_num(thetaPrime.magnitude), axis=2)
        wPrimeFFT = np.fft.rfft(np.nan_to_num(wPrime.magnitude), axis=2)

        # n_valid: number of finite longitude samples per (alt, lat) point for each cross-product pair.
        # Using n_valid instead of lons.size in the Parseval normalisation ensures
        # that nan_to_num (which zeros NaN positions) stays consistent with the
        # real-space nanmean (which ignores NaN positions).
        _N = datasetLogPress.lon.size
        n_valid_vt = np.sum(np.isfinite(vPrime.magnitude * thetaPrime.magnitude), axis=2, keepdims=True)
        n_valid_uv = np.sum(np.isfinite(uPrime.magnitude * vPrime.magnitude), axis=2, keepdims=True)
        n_valid_uw = np.sum(np.isfinite(uPrime.magnitude * wPrime.magnitude), axis=2, keepdims=True)

        FourT = {}

        with np.errstate(divide='ignore', invalid='ignore'): # ignore warnings about empty slices in Antarctica.

            FourT['EPFLat_WaveN'] = (densBasic3D.magnitude * rEarth.magnitude * cosFi[:, :, np.newaxis].magnitude *
                                (DUBarDZ[:, :, np.newaxis].magnitude / DThetaBarDZ[:, :, np.newaxis].magnitude *
                                np.real(vPrimeFFT * np.conj(thetaPrimeFFT)) / (n_valid_vt * _N / 2) -
                                np.real(uPrimeFFT * np.conj(vPrimeFFT)) / (n_valid_uv * _N / 2))) * units(str(EPFLat.units))

            FourT['EPFVert_WaveN'] = (densBasic3D.magnitude * rEarth.magnitude * cosFi[:, :, np.newaxis].magnitude *
                                ((coriolisParameter[:, :, np.newaxis].magnitude - 1 /
                                        (rEarth.magnitude * cosFi[:, :, np.newaxis].magnitude) *
                                        uBarCosFiDFI[:, :, np.newaxis].magnitude) / DThetaBarDZ[:, :, np.newaxis].magnitude *
                                        np.real(vPrimeFFT * np.conj(thetaPrimeFFT)) / (n_valid_vt * _N / 2) -
                                        np.real(uPrimeFFT * np.conj(wPrimeFFT)) / (n_valid_uw * _N / 2))) * units(str(EPFVert.units))

        FourT['divEPFLat_WaveN'] = (1 / (rEarth.magnitude * cosFi[:, :, np.newaxis].magnitude) *
                                nanGradient(FourT['EPFLat_WaveN'] * cosFi[:, :, np.newaxis].magnitude,
                                                latsR, axis=1)).magnitude * units(str(divEPFLat.units))
        
        FourT['divEPFVert_WaveN'] = nanGradient(FourT['EPFVert_WaveN'], altitudes, axis=0).magnitude * units(str(divEPFVert.units))
        
        FourT['divEPF_WaveN'] = FourT['divEPFVert_WaveN'] + FourT['divEPFLat_WaveN']

        # The Nyquist component (k=N//2) appears only once in the DFT (not twice like k=1..N//2-1),
        # so it was over-normalised by a factor of 2; correct that here.
        _nyq = datasetLogPress.lon.size // 2
        for _var in FourT:
            FourT[_var][:, :, _nyq] = FourT[_var][:, :, _nyq] / 2

        if tomlConfig['saveEddyTerms']: # SHOULD FFT OR SOMETHING ELSE BE SAVED HERE?????
            vPrimeThetaPrimeWaveN = np.real(vPrimeFFT * np.conj(thetaPrimeFFT)) / (n_valid_vt * _N / 2)
            vPrimeUPrimeWaveN = np.real(vPrimeFFT * np.conj(uPrimeFFT)) / (n_valid_uv * _N / 2)
            wPrimeUPrimeWaveN = np.real(wPrimeFFT * np.conj(uPrimeFFT)) / (n_valid_uw * _N / 2)
        
            eddyTermsFour = {'vPrimeThetaPrimeWaveN': vPrimeThetaPrimeWaveN, 'vPrimeUPrimeWaveN': vPrimeUPrimeWaveN,
                                    'wPrimeUPrimeWaveN': wPrimeUPrimeWaveN}
            for _et in eddyTermsFour:
                eddyTermsFour[_et][:, :, _nyq] = eddyTermsFour[_et][:, :, _nyq] / 2
            
        if len(tomlConfig['Waves']) == 1 and tomlConfig['Waves'][0].lower() == 'all':
            FShape = np.zeros((FourT['EPFVert_WaveN'].shape[0], FourT['EPFVert_WaveN'].shape[1], FourT['EPFVert_WaveN'].shape[2] - 1)).shape
            Fourier = {"EPFLat_WaveN": np.zeros((FShape)), "EPFVert_WaveN": np.zeros((FShape)),
                        "divEPFVert_WaveN": np.zeros((FShape)), "divEPFLat_WaveN": np.zeros((FShape)),
                        'divEPF_WaveN': np.zeros((FShape))}

            for variable in Fourier.keys():
                Fourier[variable][:, :, :] = FourT[variable][:, :, 1:]
        
            waveNumbers = list(range(1, Fourier['divEPF_WaveN'].shape[2] + 1))    
            
            if tomlConfig['saveEddyTerms']:
                for variable in eddyTermsFour.keys():
                    Fourier[variable] = eddyTermsFour[variable][:, :, 1:]
                    
        else:
            # Saving entire Fourier transform result usually significantly increases size of the output file.
            # There is an option to save only some waves which are stored in tomlConfig dictionary as tomlConfig["Waves"]. 
            # tomlConfig["Waves"] is expected to be a string "all" or a list of strings like "5" or "6-10" where in
            # case of "6-10" sum of waves from 6 to 10 will be saved as a single 2d field
            
            FShape = np.zeros((FourT['EPFVert_WaveN'].shape[0], FourT['EPFVert_WaveN'].shape[1], len(tomlConfig['Waves']))).shape
            Fourier = {"EPFLat_WaveN": np.zeros((FShape)), "EPFVert_WaveN": np.zeros((FShape)),
                        "divEPFVert_WaveN": np.zeros((FShape)), "divEPFLat_WaveN": np.zeros((FShape)),
                        'divEPF_WaveN': np.zeros((FShape))}

            for variable in Fourier.keys():
                for i, wave in enumerate(tomlConfig['Waves']):
                    if '-' not in wave:  # if it is a single wave
                        wave = int(wave)
                        Fourier[variable][:, :, i] = FourT[variable][:, :, wave]
                    elif 'end' not in wave and '-' in wave:  # if it is a range but not to the last wave
                        waveStart = int(wave.split('-')[0])
                        waveEnd = int(wave.split('-')[1])
                        Fourier[variable][:, :, i] = np.nansum(FourT[variable][:, :, waveStart:waveEnd + 1], 2)
                    else:  # if it is a range to the last wave
                        waveStart = int(wave.split('-')[0])
                        Fourier[variable][:, :, i] = np.nansum(FourT[variable][:, :, waveStart:], 2)

            waveNumbers = tomlConfig['Waves']

            if tomlConfig['saveEddyTerms']:
                Fourier.update({'vPrimeThetaPrimeWaveN': np.zeros((FShape)), 'vPrimeUPrimeWaveN': np.zeros((FShape)),
                                'wPrimeUPrimeWaveN': np.zeros((FShape))})
                for variable in eddyTermsFour.keys():
                    for i, wave in enumerate(tomlConfig['Waves']):
                        if '-' not in wave:  # if it is a single wave
                            wave = int(wave)
                            Fourier[variable][:, :, i] = eddyTermsFour[variable][:, :, wave]
                        elif 'end' not in wave and '-' in wave:  # if it is a range but not to the last wave
                            waveStart = int(wave.split('-')[0])
                            waveEnd = int(wave.split('-')[1])
                            Fourier[variable][:, :, i] = np.nansum(eddyTermsFour[variable][:, :, waveStart:waveEnd + 1], 2)
                        else:  # if it is a range to the last wave
                            waveStart = int(wave.split('-')[0])
                            Fourier[variable][:, :, i] = np.nansum(eddyTermsFour[variable][:, :, waveStart:], 2)

        dataToSaveFourier = {
            'EPFVert_WaveN': (Fourier['EPFVert_WaveN'], 'vertical component of Eliassen-Pulm flux', str(EPFVert.units)),
            'EPFLat_WaveN': (Fourier['EPFLat_WaveN'], 'latitudinal component of Eliassen-Pulm flux', str(EPFLat.units)),
            'divEPFVert_WaveN': (Fourier['divEPFVert_WaveN'],
                        'vertical component of Eliassen-Pulm flux divergence', str(divEPFVert.units)),
            'divEPFLat_WaveN': (Fourier['divEPFLat_WaveN'],
                        'zonal component of Eliassen-Pulm flux divergence', str(divEPFLat.units)),
            'divEPF_WaveN': (Fourier['divEPF_WaveN'], 'Eliassen-Pulm flux divergence', str(divEPF.units)),
        }

        if tomlConfig['saveEddyTerms']:
            dataToSaveFourier['vPrimeThetaPrimeWaveN'] = (Fourier['vPrimeThetaPrimeWaveN'], 'wave decomposition of horizontal eddy heat flux', str((vPrime * thetaPrime).units)) # CHECK UNITS
            dataToSaveFourier['vPrimeUPrimeWaveN'] = (Fourier['vPrimeUPrimeWaveN'], 'wave decomposition of horizontal eddy momentum flux', str((vPrime * uPrime).units))
            dataToSaveFourier['wPrimeUPrimeWaveN'] = (Fourier['wPrimeUPrimeWaveN'], 'wave decomposition of vertical eddy momentum flux', str((wPrime * uPrime).units))
            
        
        for variable in dataToSaveFourier.keys():
            dsOut[variable] = (('alt', 'lat', 'waveNumber'),
                                np.single(dataToSaveFourier[variable][0]))
            getattr(dsOut, variable).attrs['long_name'] = dataToSaveFourier[variable][1]
            getattr(dsOut, variable).attrs['units'] = dataToSaveFourier[variable][2] 
            
        dsOut.coords['waveNumber'] = waveNumbers
        dsOut.waveNumber.attrs['long_name'] = 'wave size, as number of waves per 360 degrees of logitude'

    return dsOut


def _run_tem_and_attach_vars(tomlConfig, datasetLogPress, timestamp, saveInterpolatedZonalMeanVars, saveZonalMeanVars):
    dsOut = TEMCalcs(tomlConfig, datasetLogPress, timestamp)
    if saveInterpolatedZonalMeanVars:
        dsOut[saveInterpolatedZonalMeanVars] = datasetLogPress[saveInterpolatedZonalMeanVars]
    if saveZonalMeanVars:
        dsOut[saveZonalMeanVars] = datasetLogPress[saveZonalMeanVars]
    return dsOut


def _finalize_mean(dsAccum, dsU, count, time1st, last_instance):
    dsOut = dsAccum / count
    dsOut.attrs = last_instance.attrs
    for variable in dsOut.data_vars:
        dsOut[variable].attrs = last_instance[variable].attrs
    dU_dt = np.array(dsU.U.differentiate(coord='time', datetime_unit='s').mean(dim='time'))
    dsOut['dU_dt'] = (('alt', 'lat'), np.single(dU_dt))
    dsOut.dU_dt.attrs['long_name'] = 'acceleration of zonal wind'
    dsOut.dU_dt.attrs['units'] = str(dsOut.U.units) + ' / s'
    dsOut.coords['time'] = time1st
    return dsOut


def _build_output_filename(tomlConfig, outTime):
    prefix = f"{tomlConfig['outputDirectory']}/{tomlConfig['outPrefix']}"
    mean = str(tomlConfig['outputTemporalMean']).lower()
    if mean in ['monthly', 'month']:
        return f"{prefix}_{tomlConfig['outputTemporalMean']}Mean_{outTime.year}_{outTime.month:02d}.nc"
    elif mean in ['daily', 'day']:
        return f"{prefix}_{tomlConfig['outputTemporalMean']}Mean_{outTime.year}_{outTime.month:02d}_{outTime.day:02d}.nc"
    else:
        return f"{prefix}_{outTime.year}_{outTime.month:02d}_{outTime.day:02d}_{outTime.hour:02d}_{outTime.minute:02d}.nc"


def _accumulate(dsOut, dsU, dsInstance):
    """Add dsInstance into the running sum dsAccum; return updated (dsAccum, dsU)."""
    dsU_instance = dsInstance.U.expand_dims(time=dsInstance.coords['time'])
    dsU = xr.merge([dsU, dsU_instance], compat='override', join='outer')
    return dsOut + dsInstance.squeeze(), dsU


def init_worker(shared_counter):
    ''' store the counter for later use to calculate percent done'''
    global counter
    counter = shared_counter


def mainCalcs(pathsAndTimeChunk, reqVars, tomlConfig, saveInterpolatedZonalMeanVars=[], saveZonalMeanVars=[]):

    global counter
    timeDim = tomlConfig.get('timeDim', '')

    def _interp(ds, timestamp=None):
        sel = ds if timestamp is None else ds.sel({timeDim: timestamp})
        return interpolateToLogPressure(
            sel, reqVars, tomlConfig['verticalDimensionType'], tomlConfig['targetLevels'],
            tomlConfig['vertDim'], tomlConfig['latDim'], tomlConfig['lonDim'], tomlConfig['pressureName'],
            saveInterpolatedZonalMeanVars, saveZonalMeanVars,
        )

    def _process_group(group_ds, timestamps):
        """Average TEMCalcs over a sequence of timestamps; return finalized dsOut and fnout."""
        if len(timestamps) == 1:
            ts = timestamps[0].values if hasattr(timestamps[0], 'values') else timestamps[0]
            dlp = _interp(group_ds.squeeze() if hasattr(group_ds, 'squeeze') else group_ds)
            dsOut = _run_tem_and_attach_vars(tomlConfig, dlp, ts, saveInterpolatedZonalMeanVars, saveZonalMeanVars)
            return dsOut, _build_output_filename(tomlConfig, pd.Timestamp(dsOut.time.values[0]))

        first = True
        for ts in timestamps:
            ts = ts if not hasattr(ts, 'values') else ts.values
            dlp = _interp(group_ds, ts)
            dsInstance = _run_tem_and_attach_vars(tomlConfig, dlp, ts, saveInterpolatedZonalMeanVars, saveZonalMeanVars)
            if first:
                time1st = dsInstance.coords['time']
                dsU = dsInstance.U.expand_dims(time=dsInstance.coords['time'])
                dsAccum = dsInstance.squeeze()
                first = False
            else:
                dsAccum, dsU = _accumulate(dsAccum, dsU, dsInstance)
        dsOut = _finalize_mean(dsAccum, dsU, len(timestamps), time1st, dsInstance)
        return dsOut, _build_output_filename(tomlConfig, pd.Timestamp(dsOut.time.values[0]))

    if len(pathsAndTimeChunk) == 1:
        dataset = readAndTransposeData(
            pathsAndTimeChunk.Path.iloc[0], reqVars,
            tomlConfig['vertDim'], tomlConfig['latDim'], tomlConfig['lonDim'],
            saveInterpolatedZonalMeanVars, saveZonalMeanVars,
        )
        if timeDim and timeDim in dataset.dims:
            mean = str(tomlConfig['outputTemporalMean']).lower()
            if mean in ['monthly', 'month'] and np.max(dataset[timeDim]) - np.min(dataset[timeDim]) >= np.timedelta64(27, 'D'):
                for _, group in dataset.resample({timeDim: "MS"}):
                    dsOut, fnout = _process_group(group, list(group[timeDim]))
                    dsOut.to_netcdf(fnout)
            elif mean in ['daily', 'day'] and np.max(dataset[timeDim]) - np.min(dataset[timeDim]) >= np.timedelta64(1, 'D'):
                for _, group in dataset.resample({timeDim: "D"}):
                    dsOut, fnout = _process_group(group, list(group[timeDim]))
                    dsOut.to_netcdf(fnout)
            else:
                for time in dataset[timeDim]:
                    dlp = _interp(dataset, time)
                    dsOut = _run_tem_and_attach_vars(tomlConfig, dlp, pathsAndTimeChunk.index[0], saveInterpolatedZonalMeanVars, saveZonalMeanVars)
                    dsOut.coords['time'] = [time.values]
                    fnout = _build_output_filename(tomlConfig, pd.Timestamp(dsOut.time.values[0]))
                    dsOut.to_netcdf(fnout)
        else:
            dlp = interpolateToLogPressure(
                dataset, reqVars, tomlConfig['verticalDimensionType'], tomlConfig['targetLevels'],
                tomlConfig['vertDim'], tomlConfig['latDim'], tomlConfig['lonDim'], tomlConfig['pressureName'],
                saveInterpolatedZonalMeanVars, saveZonalMeanVars,
            )
            dsOut = _run_tem_and_attach_vars(tomlConfig, dlp, pathsAndTimeChunk.index[0], saveInterpolatedZonalMeanVars, saveZonalMeanVars)
            fnout = _build_output_filename(tomlConfig, pd.Timestamp(dsOut.time.values[0]))
            dsOut.to_netcdf(fnout)

        # += operation is not atomic, so we need to get a lock:
        with counter.get_lock():
            counter.value += 1

    else:  # multiple files — calculate temporal mean across all of them
        first = True
        for path, time in zip(pathsAndTimeChunk.Path, pathsAndTimeChunk.index):
            dataset = readAndTransposeData(
                path, reqVars, tomlConfig['vertDim'], tomlConfig['latDim'], tomlConfig['lonDim'],
                saveInterpolatedZonalMeanVars, saveZonalMeanVars,
            )
            if timeDim and timeDim in dataset.dims:
                for time2 in dataset[timeDim]:
                    dlp = _interp(dataset, time2)
                    dsInstance = _run_tem_and_attach_vars(tomlConfig, dlp, time, saveInterpolatedZonalMeanVars, saveZonalMeanVars)
                    if first:
                        time1st = dsInstance.coords['time']
                        dsU = dsInstance.U.expand_dims(time=dsInstance.coords['time'])
                        dsAccum = dsInstance.squeeze()
                        first = False
                    else:
                        dsAccum, dsU = _accumulate(dsAccum, dsU, dsInstance)
            else:
                dlp = interpolateToLogPressure(
                    dataset, reqVars, tomlConfig['verticalDimensionType'], tomlConfig['targetLevels'],
                    tomlConfig['vertDim'], tomlConfig['latDim'], tomlConfig['lonDim'], tomlConfig['pressureName'],
                    saveInterpolatedZonalMeanVars, saveZonalMeanVars,
                )
                dsInstance = _run_tem_and_attach_vars(tomlConfig, dlp, time, saveInterpolatedZonalMeanVars, saveZonalMeanVars)
                if first:
                    time1st = dsInstance.coords['time']
                    dsU = dsInstance.U.expand_dims(time=dsInstance.coords['time'])
                    dsAccum = dsInstance.squeeze()
                    first = False
                else:
                    dsAccum, dsU = _accumulate(dsAccum, dsU, dsInstance)

            # += operation is not atomic, so we need to get a lock:
            with counter.get_lock():
                counter.value += 1

        dsOut = _finalize_mean(dsAccum, dsU, len(pathsAndTimeChunk), time1st, dsInstance)
        fnout = _build_output_filename(tomlConfig, pd.Timestamp(dsOut.time.values[0]))
        dsOut.to_netcdf(fnout)
        




