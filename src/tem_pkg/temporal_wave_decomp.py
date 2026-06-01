from __future__ import annotations

import numpy as np
import xarray as xr
from metpy.units import units

from .constants import P0, Cp, R, Ts, angVeloEarth, gEarth, rEarth
from .interpolation import alt2press
from .utils import addRatioUnits, apply_waves_banding, nanGradient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band_all(dataToSave: dict, waves_config: list) -> tuple[dict, list]:
    """Apply ``apply_waves_banding`` to every array in *dataToSave*."""
    banded = {}
    for key, val in dataToSave.items():
        arr, long_name, unit = val
        banded[key] = [apply_waves_banding(arr, waves_config), long_name, unit]
    return banded, list(waves_config)


# ---------------------------------------------------------------------------
# Log-pressure (pressure) coordinates
# ---------------------------------------------------------------------------

def waveDecompPress(
    datasets: list[xr.Dataset],
    tomlConfig: dict,
) -> tuple[dict, np.ndarray, np.ndarray, list]:
    """
    Per-wavenumber stationary/transient decomposition of EP-flux and tracer
    eddy transport tendencies in log-pressure altitude coordinates.

    For each zonal wavenumber k the eddy covariance splits exactly as:

        time_mean( Re[A_k · B_k*] ) = Re[<A_k> · <B_k>*]  +  residual
                total_k                   stationary_k          transient_k

    where <A_k> = time_mean(A_k) is the time-mean complex Fourier coefficient
    (the stationary wave at wavenumber k) and the residual is the transient
    eddy contribution. All per-wavenumber fluxes and tendencies are computed
    from these spectra and then optionally summed into wavenumber bands.

    Parameters
    ----------
    datasets : list of xr.Dataset
        Per-timestep 3-D datasets on a common log-pressure altitude grid with
        dimensions (``'alt'``, ``'lat'``, ``'lon'``).
    tomlConfig : dict
        Configuration dict.  Uses the same keys as the transport calculators:
        ``meridionalWindName``, ``zonalWindName``, ``verticalWindName``,
        ``verticalWindType``, ``temperatureName``, ``temperatureType``,
        ``tracerNames``, ``computeEPF``, ``Waves``.

    Returns
    -------
    dataToSave : dict
        ``{var_name: [data_array, long_name, units_str]}``.
        Arrays have shape ``(alt, lat, n_bands)`` where ``n_bands`` equals the
        number of entries in ``Waves`` (or ``N_lon // 2`` when ``Waves='all'``).
        Variable naming convention: ``{quantity}_WN_stat`` / ``{quantity}_WN_trans``.
    lats : np.ndarray
        Latitude coordinate values in degrees.
    altitudes_m : np.ndarray
        Log-pressure altitude levels in metres.
    wave_numbers : list
        Wavenumber band labels for the output ``waveN`` coordinate.
    """
    addRatioUnits()
    if not datasets:
        raise ValueError("datasets list is empty")

    ds0 = datasets[0]
    lats = np.array(ds0.lat)
    latsR = (lats * units.degrees).to('radian').magnitude
    cosFi = np.cos(latsR)[np.newaxis, :]
    N_lon = ds0.lon.size
    N_t = len(datasets)
    _norm = N_lon / 2   # Parseval normalisation: sum over WN == zonal-mean covariance
    _nyq = N_lon // 2   # Nyquist index in rfft output

    altitudes = (np.array(ds0.alt) * units(ds0.alt.units)).to('m')
    altitudes_m = altitudes.magnitude
    pressureLevels = alt2press(altitudes)
    densBasic = (pressureLevels / (R * Ts)).to('kg/m^3').magnitude   # (alt,)
    rho0 = densBasic[:, np.newaxis]                                   # (alt, 1)

    vName = tomlConfig['meridionalWindName']
    uName = tomlConfig['zonalWindName']
    wName = tomlConfig['verticalWindName']
    tName = tomlConfig['temperatureName']
    tracers = tomlConfig.get('tracerNames', [])
    compute_epf = tomlConfig.get('computeEPF', True)
    has_w = tomlConfig['verticalWindType'].lower() != 'missing'

    # ------------------------------------------------------------------
    # Single-pass accumulation
    #
    # fft_sums[key]    : sum of complex rfft coefficients across timesteps
    #                    shape (alt, lat, N_lon//2+1)
    # xspec_sums[key]  : sum of Re[A_k · B_k*] (unnormalised) across timesteps
    #                    shape (alt, lat, N_lon//2+1)
    # nvalid_sums[key] : sum of n_valid (finite-pair count) across timesteps
    #                    shape (alt, lat, 1)   — same for all wavenumbers of a pair
    # tmean3d[name]    : sum of 3-D real-space fields (for time-mean gradients)
    #                    shape (alt, lat, N_lon)
    #
    # Normalisation mirrors the existing transport calculators exactly:
    #   covariance_k = Re[A_k · B_k*] / (n_valid * N_lon / 2)
    # where n_valid counts finite longitude samples of the product a'·b' at
    # each (alt, lat) point.  Summing over all wavenumbers recovers nanmean(a'·b').
    # ------------------------------------------------------------------
    fft_sums: dict[str, np.ndarray] = {}
    xspec_sums: dict[str, np.ndarray] = {}
    nvalid_sums: dict[str, np.ndarray] = {}
    tmean3d: dict[str, np.ndarray] = {}

    def _acc(key: str, arr: np.ndarray, store: dict) -> None:
        store[key] = store.get(key, np.zeros_like(arr)) + arr

    for ds in datasets:
        def _get3d(name: str, to_unit: str) -> np.ndarray:
            return (np.array(ds[name]) * units(ds[name].units)).to(to_unit).magnitude

        v = _get3d(vName, 'm/s')
        if tomlConfig['temperatureType'] != 'theta':
            theta = (np.array(ds[tName]) * units(ds[tName].units) *
                     (P0 / pressureLevels)[:, np.newaxis, np.newaxis] ** (R / Cp)).to('kelvin').magnitude
        else:
            theta = _get3d(tName, 'kelvin')
        if has_w:
            w = (_get3d(wName, 'm/s') if tomlConfig['verticalWindType'] in ['W', 'w'] else
                 (-np.array(ds[wName]) * units(ds[wName].units) /
                  (densBasic[:, np.newaxis, np.newaxis] * units('kg/m^3') * gEarth)).to('m/s').magnitude)
        if compute_epf:
            u = _get3d(uName, 'm/s')

        # 3-D real-space sums for time-mean gradients
        _acc(tName, theta, tmean3d)
        if compute_epf:
            _acc(uName, u, tmean3d)

        # per-timestep zonal eddies
        vp = v - np.nanmean(v, axis=2)[:, :, np.newaxis]
        tp = theta - np.nanmean(theta, axis=2)[:, :, np.newaxis]
        if has_w:
            wp = w - np.nanmean(w, axis=2)[:, :, np.newaxis]
        if compute_epf:
            up = u - np.nanmean(u, axis=2)[:, :, np.newaxis]

        # FFT of eddies (NaN → 0 before transform, matching transport code)
        vFFT = np.fft.rfft(np.nan_to_num(vp), axis=2)
        tFFT = np.fft.rfft(np.nan_to_num(tp), axis=2)
        if has_w:
            wFFT = np.fft.rfft(np.nan_to_num(wp), axis=2)
        if compute_epf:
            uFFT = np.fft.rfft(np.nan_to_num(up), axis=2)

        # accumulate complex FFT coefficients (time-mean → stationary wave)
        _acc('v', vFFT, fft_sums)
        _acc('t', tFFT, fft_sums)
        if has_w:
            _acc('w', wFFT, fft_sums)
        if compute_epf:
            _acc('u', uFFT, fft_sums)

        # n_valid is computed from the real-space eddy products (before FFT),
        # matching exactly what the transport calculator does.
        nv_vt = np.sum(np.isfinite(vp * tp), axis=2, keepdims=True)
        _acc('vp_tp', np.real(vFFT * np.conj(tFFT)), xspec_sums)
        _acc('vp_tp', nv_vt, nvalid_sums)

        if compute_epf:
            nv_vu = np.sum(np.isfinite(vp * up), axis=2, keepdims=True)
            _acc('vp_up', np.real(vFFT * np.conj(uFFT)), xspec_sums)
            _acc('vp_up', nv_vu, nvalid_sums)
            if has_w:
                nv_wu = np.sum(np.isfinite(wp * up), axis=2, keepdims=True)
                _acc('wp_up', np.real(wFFT * np.conj(uFFT)), xspec_sums)
                _acc('wp_up', nv_wu, nvalid_sums)

        for tracer in tracers:
            chi = np.array(ds[tracer]) * units(ds[tracer].units)
            chi = (chi.to_base_units() if chi.dimensionality == units('s').dimensionality
                   else chi).magnitude
            _acc(tracer, chi, tmean3d)
            chiP = chi - np.nanmean(chi, axis=2)[:, :, np.newaxis]
            chiFFT = np.fft.rfft(np.nan_to_num(chiP), axis=2)
            _acc(f'chi_{tracer}', chiFFT, fft_sums)

            nv_vc = np.sum(np.isfinite(vp * chiP), axis=2, keepdims=True)
            _acc(f'vp_chip_{tracer}', np.real(vFFT * np.conj(chiFFT)), xspec_sums)
            _acc(f'vp_chip_{tracer}', nv_vc, nvalid_sums)
            if has_w:
                nv_wc = np.sum(np.isfinite(wp * chiP), axis=2, keepdims=True)
                _acc(f'wp_chip_{tracer}', np.real(wFFT * np.conj(chiFFT)), xspec_sums)
                _acc(f'wp_chip_{tracer}', nv_wc, nvalid_sums)

    # ------------------------------------------------------------------
    # Finalise time means and compute stationary/transient spectra
    # ------------------------------------------------------------------
    for k in tmean3d:
        tmean3d[k] /= N_t

    # time-mean complex FFT coefficients (stationary wave at each k)
    fft_mean = {k: v / N_t for k, v in fft_sums.items()}

    # total per-wavenumber covariance, normalised identically to transport code:
    #   sum_t Re[A_k B_k*] / (n_valid_total * N_lon/2)
    # n_valid_total is the sum of per-timestep finite-pair counts.
    xspec_total = {k: xspec_sums[k] / (nvalid_sums[k] * _norm)
                   for k in xspec_sums}

    # stationary covariance: Re[<A_k> · <B_k>*] / (n_valid_total/N_t * N_lon/2)
    # We use the same n_valid denominator so stat+trans = total exactly.
    xspec_stat: dict[str, np.ndarray] = {}
    xspec_stat['vp_tp'] = np.real(fft_mean['v'] * np.conj(fft_mean['t'])) * N_t / (nvalid_sums['vp_tp'] * _norm)
    if compute_epf:
        xspec_stat['vp_up'] = np.real(fft_mean['v'] * np.conj(fft_mean['u'])) * N_t / (nvalid_sums['vp_up'] * _norm)
        if has_w:
            xspec_stat['wp_up'] = np.real(fft_mean['w'] * np.conj(fft_mean['u'])) * N_t / (nvalid_sums['wp_up'] * _norm)
    for tracer in tracers:
        nv_vc = nvalid_sums[f'vp_chip_{tracer}']
        xspec_stat[f'vp_chip_{tracer}'] = np.real(fft_mean['v'] * np.conj(fft_mean[f'chi_{tracer}'])) * N_t / (nv_vc * _norm)
        if has_w:
            nv_wc = nvalid_sums[f'wp_chip_{tracer}']
            xspec_stat[f'wp_chip_{tracer}'] = np.real(fft_mean['w'] * np.conj(fft_mean[f'chi_{tracer}'])) * N_t / (nv_wc * _norm)

    # transient = total − stationary
    xspec_trans = {k: xspec_total[k] - xspec_stat[k] for k in xspec_stat}

    for d in (xspec_total, xspec_stat, xspec_trans):
        for arr in d.values():
            arr[:, :, _nyq] /= 2

    # Strip k=0 (zonal-mean squared, not an eddy); keep k=1..(N//2)
    # Resulting shape: (alt, lat, N//2)
    for d in (xspec_total, xspec_stat, xspec_trans):
        for key in list(d.keys()):
            d[key] = d[key][:, :, 1:]

    # ------------------------------------------------------------------
    # Time-mean gradients (2-D: alt × lat)
    # ------------------------------------------------------------------
    theta_bar = np.nanmean(tmean3d[tName], axis=2)
    DThetaBarDZ = nanGradient(theta_bar, altitudes_m, axis=0)

    # ------------------------------------------------------------------
    # Per-wavenumber EP-flux diagnostics
    # ------------------------------------------------------------------
    dataToSave: dict = {}

    if compute_epf:
        u_bar = np.nanmean(tmean3d[uName], axis=2)
        DUBarDZ = nanGradient(u_bar, altitudes_m, axis=0)[:, :, np.newaxis]
        uBarCosFiDFi = nanGradient(u_bar * cosFi, latsR, axis=1)[:, :, np.newaxis]
        coriolisP = ((2 * angVeloEarth * np.sin(latsR * units.radian)).to('1/s').magnitude
                     [np.newaxis, :, np.newaxis])
        DTZ3 = DThetaBarDZ[:, :, np.newaxis]
        rho3 = rho0[:, :, np.newaxis]
        cf3 = cosFi[:, :, np.newaxis]

        for sfx, xd in [('stat', xspec_stat), ('trans', xspec_trans)]:
            vp_tp = xd['vp_tp']    # (alt, lat, N//2)
            vp_up = xd['vp_up']

            EPFlat = rho3 * rEarth.magnitude * cf3 * (DUBarDZ * vp_tp / DTZ3 - vp_up)
            divEPFlat = (1 / (rEarth.magnitude * cf3) *
                         nanGradient(EPFlat * cf3, latsR, axis=1))

            dataToSave[f'EPFlat_WN_{sfx}'] = [EPFlat, f'{sfx}-wave meridional EP flux per wavenumber', 'kg / s^2']
            dataToSave[f'divEPFlat_WN_{sfx}'] = [divEPFlat, f'{sfx}-wave meridional EP flux divergence per wavenumber', 'kg / m / s^2']

            if has_w:
                wp_up = xd['wp_up']
                EPFvert = rho3 * rEarth.magnitude * cf3 * (
                    (coriolisP - 1 / (rEarth.magnitude * cf3) * uBarCosFiDFi) *
                    vp_tp / DTZ3 - wp_up)
                divEPFvert = nanGradient(EPFvert, altitudes_m, axis=0)

                dataToSave[f'EPFvert_WN_{sfx}'] = [EPFvert, f'{sfx}-wave vertical EP flux per wavenumber', 'kg / s^2']
                dataToSave[f'divEPFvert_WN_{sfx}'] = [divEPFvert, f'{sfx}-wave vertical EP flux divergence per wavenumber', 'kg / m / s^2']
                dataToSave[f'divEPF_WN_{sfx}'] = [divEPFlat + divEPFvert, f'{sfx}-wave total EP flux divergence per wavenumber', 'kg / m / s^2']

            dataToSave[f'vPrimeThetaPrimeBar_WN_{sfx}'] = [vp_tp, f'{sfx}-wave meridional eddy heat flux per wavenumber', 'm K / s']

    # ------------------------------------------------------------------
    # Per-wavenumber tracer eddy flux and mixing tendency
    # ------------------------------------------------------------------
    for tracer in tracers:
        chi_bar = np.nanmean(tmean3d[tracer], axis=2)
        chi_u = units(datasets[0][tracer].units)
        flux_units = str((units('m/s') * chi_u).units)
        tend_units = str((units('m/s') * chi_u / units('m')).units)

        DChiBarDZ = nanGradient(chi_bar, altitudes_m, axis=0)[:, :, np.newaxis]
        DChiBarDFi = nanGradient(chi_bar / rEarth.magnitude, latsR, axis=1)[:, :, np.newaxis]
        DTZ3 = DThetaBarDZ[:, :, np.newaxis]
        cf3 = cosFi[:, :, np.newaxis]

        for sfx, xd in [('stat', xspec_stat), ('trans', xspec_trans)]:
            vp_tp = xd['vp_tp']
            vp_cp = xd[f'vp_chip_{tracer}']

            # Stokes-corrected meridional flux / rho0
            m_lat = -(vp_cp - DChiBarDZ * vp_tp / DTZ3)
            divm_lat = (1 / (rEarth.magnitude * cf3) *
                        nanGradient(m_lat * cf3, latsR, axis=1))

            dataToSave[f'{tracer}_m_lat_WN_{sfx}'] = [m_lat, f'{sfx}-wave meridional eddy flux of {tracer} per wavenumber / rho0', flux_units]
            dataToSave[f'{tracer}_divm_lat_WN_{sfx}'] = [divm_lat, f'{sfx}-wave meridional eddy mixing tendency of {tracer} per wavenumber', tend_units]

            if has_w:
                wp_cp = xd[f'wp_chip_{tracer}']
                m_z = -(wp_cp + DChiBarDFi * vp_tp / DTZ3)
                divm_z = nanGradient(m_z, altitudes_m, axis=0)

                dataToSave[f'{tracer}_m_z_WN_{sfx}'] = [m_z, f'{sfx}-wave vertical eddy flux of {tracer} per wavenumber / rho0', flux_units]
                dataToSave[f'{tracer}_divm_z_WN_{sfx}'] = [divm_z, f'{sfx}-wave vertical eddy mixing tendency of {tracer} per wavenumber', tend_units]
                dataToSave[f'{tracer}_divm_WN_{sfx}'] = [divm_lat + divm_z, f'{sfx}-wave total eddy mixing tendency of {tracer} per wavenumber', tend_units]

    # ------------------------------------------------------------------
    # Wavenumber banding
    # ------------------------------------------------------------------
    waves_cfg = tomlConfig.get('Waves', ['all'])
    if len(waves_cfg) == 1 and waves_cfg[0].lower() == 'all':
        wave_numbers = [str(k) for k in range(1, N_lon // 2 + 1)]
    else:
        dataToSave, wave_numbers = _band_all(dataToSave, waves_cfg)

    return dataToSave, lats, altitudes_m, wave_numbers


# ---------------------------------------------------------------------------
# Isentropic (theta) coordinates
# ---------------------------------------------------------------------------

def waveDecompTheta(
    datasets: list[xr.Dataset],
    tomlConfig: dict,
) -> tuple[dict, np.ndarray, np.ndarray, list]:
    """
    Per-wavenumber stationary/transient decomposition of tracer eddy transport
    tendencies in isentropic (theta) coordinates.

    Applies the same spectral identity as :func:`waveDecompPress` but with
    isentropic density σ = −g⁻¹ ∂p/∂θ as the weighting factor.  The
    Fourier coefficients of (σv)′ and (σQ)′ are accumulated; their time means
    define the stationary eddy at each wavenumber.

    Parameters
    ----------
    datasets : list of xr.Dataset
        Per-timestep 3-D datasets on a common theta grid with dimensions
        (``'theta'``, ``'lat'``, ``'lon'``).
    tomlConfig : dict
        Configuration dict.  Uses ``meridionalWindName``, ``verticalWindName``
        (diabatic heating Q, K/s), ``pressureName``, ``tracerNames``, ``Waves``.

    Returns
    -------
    dataToSave : dict
        ``{var_name: [data_array, long_name, units_str]}``.
        Arrays have shape ``(theta, lat, n_bands)``.
    lats : np.ndarray
        Latitude coordinate values in degrees.
    thetaLevels_K : np.ndarray
        Potential temperature levels in K.
    wave_numbers : list
        Wavenumber band labels for the output ``waveN`` coordinate.
    """
    addRatioUnits()
    if not datasets:
        raise ValueError("datasets list is empty")

    ds0 = datasets[0]
    lats = np.array(ds0.lat)
    latsR = (lats * units.degrees).to('radian').magnitude
    cosFi = np.cos(latsR)[np.newaxis, :]
    N_lon = ds0.lon.size
    N_t = len(datasets)
    _norm = N_lon / 2
    _nyq = N_lon // 2

    thetaLevels = (np.array(ds0.theta) * units(ds0.theta.units)).to('K')
    thetaLevels_K = thetaLevels.magnitude

    vName = tomlConfig['meridionalWindName']
    qName = tomlConfig['verticalWindName']
    pName = tomlConfig['pressureName']
    tracers = tomlConfig.get('tracerNames', [])

    fft_sums: dict[str, np.ndarray] = {}
    xspec_sums: dict[str, np.ndarray] = {}
    nvalid_sums: dict[str, np.ndarray] = {}
    sigma_sum: np.ndarray | None = None
    chi_tmean: dict[str, np.ndarray] = {}

    def _acc(key: str, arr: np.ndarray, store: dict) -> None:
        store[key] = store.get(key, np.zeros_like(arr)) + arr

    for ds in datasets:
        p = np.array(ds[pName]) * units(ds[pName].units)
        sigma = (-1 / gEarth * nanGradient(p, thetaLevels, axis=0)).to('kg/m/m/K').magnitude
        v = (np.array(ds[vName]) * units(ds[vName].units)).to('m/s').magnitude
        q = (np.array(ds[qName]) * units(ds[qName].units)).to('K/s').magnitude

        sigma_sum = sigma if sigma_sum is None else sigma_sum + sigma

        # sigma-weighted eddy flux fields
        sv = sigma * v
        sq = sigma * q
        sv_prime = sv - np.nanmean(sv, axis=2)[:, :, np.newaxis]
        sq_prime = sq - np.nanmean(sq, axis=2)[:, :, np.newaxis]

        svFFT = np.fft.rfft(np.nan_to_num(sv_prime), axis=2)
        sqFFT = np.fft.rfft(np.nan_to_num(sq_prime), axis=2)

        _acc('sv', svFFT, fft_sums)
        _acc('sq', sqFFT, fft_sums)

        for tracer in tracers:
            chi = np.array(ds[tracer]) * units(ds[tracer].units)
            chi = (chi.to_base_units() if chi.dimensionality == units('s').dimensionality
                   else chi).magnitude
            chi_tmean[tracer] = chi_tmean.get(tracer, np.zeros_like(chi)) + chi
            chiP = chi - np.nanmean(chi, axis=2)[:, :, np.newaxis]
            chiFFT = np.fft.rfft(np.nan_to_num(chiP), axis=2)
            _acc(f'chi_{tracer}', chiFFT, fft_sums)

            nv_svc = np.sum(np.isfinite(sv_prime * chiP), axis=2, keepdims=True)
            _acc(f'sv_chip_{tracer}', np.real(svFFT * np.conj(chiFFT)), xspec_sums)
            _acc(f'sv_chip_{tracer}', nv_svc, nvalid_sums)

            nv_sqc = np.sum(np.isfinite(sq_prime * chiP), axis=2, keepdims=True)
            _acc(f'sq_chip_{tracer}', np.real(sqFFT * np.conj(chiFFT)), xspec_sums)
            _acc(f'sq_chip_{tracer}', nv_sqc, nvalid_sums)

    sigma_bar_tmean = np.nanmean(sigma_sum / N_t, axis=2)  # (theta, lat)
    fft_mean = {k: v / N_t for k, v in fft_sums.items()}
    xspec_total = {k: xspec_sums[k] / (nvalid_sums[k] * _norm) for k in xspec_sums}

    xspec_stat: dict[str, np.ndarray] = {}
    for tracer in tracers:
        xspec_stat[f'sv_chip_{tracer}'] = np.real(fft_mean['sv'] * np.conj(fft_mean[f'chi_{tracer}'])) * N_t / (nvalid_sums[f'sv_chip_{tracer}'] * _norm)
        xspec_stat[f'sq_chip_{tracer}'] = np.real(fft_mean['sq'] * np.conj(fft_mean[f'chi_{tracer}'])) * N_t / (nvalid_sums[f'sq_chip_{tracer}'] * _norm)

    xspec_trans = {k: xspec_total[k] - xspec_stat[k] for k in xspec_stat}

    for d in (xspec_total, xspec_stat, xspec_trans):
        for arr in d.values():
            arr[:, :, _nyq] /= 2

    # Strip k=0
    for d in (xspec_total, xspec_stat, xspec_trans):
        for key in list(d.keys()):
            d[key] = d[key][:, :, 1:]

    # ------------------------------------------------------------------
    # Per-wavenumber tracer eddy flux and mixing tendency
    # ------------------------------------------------------------------
    dataToSave: dict = {}

    for tracer in tracers:
        chi_u = units(datasets[0][tracer].units)
        flux_lat_units = str((units('m/s') * chi_u).units)
        flux_theta_units = str((units('K/s') * chi_u).units)
        tend_units = str((units('m/s') * chi_u / units('m')).units)

        sigma_bar3 = sigma_bar_tmean[:, :, np.newaxis]
        cf3 = cosFi[:, :, np.newaxis]

        for sfx, xd in [('stat', xspec_stat), ('trans', xspec_trans)]:
            sv_cp = xd[f'sv_chip_{tracer}']
            sq_cp = xd[f'sq_chip_{tracer}']

            # Take gradients BEFORE dividing by sigma_bar to match transport code:
            #   divm = gradient(M_unnorm) / sigma_bar
            # Dividing first introduces a spurious d(1/sigma_bar)/daxis term.
            M_phi_unnorm = -sv_cp
            M_theta_unnorm = -sq_cp

            divM_phi = (1 / (rEarth.magnitude * cf3) *
                        nanGradient(M_phi_unnorm * cf3, latsR, axis=1)) / sigma_bar3
            divM_theta = nanGradient(M_theta_unnorm, thetaLevels_K, axis=0) / sigma_bar3

            M_phi = M_phi_unnorm / sigma_bar3
            M_theta = M_theta_unnorm / sigma_bar3

            dataToSave[f'{tracer}_m_lat_WN_{sfx}'] = [M_phi, f'{sfx}-wave meridional eddy flux of {tracer} per wavenumber / sigma_bar', flux_lat_units]
            dataToSave[f'{tracer}_m_theta_WN_{sfx}'] = [M_theta, f'{sfx}-wave vertical eddy flux of {tracer} per wavenumber / sigma_bar', flux_theta_units]
            dataToSave[f'{tracer}_divm_lat_WN_{sfx}'] = [divM_phi, f'{sfx}-wave meridional eddy mixing tendency of {tracer} per wavenumber', tend_units]
            dataToSave[f'{tracer}_divm_theta_WN_{sfx}'] = [divM_theta, f'{sfx}-wave vertical eddy mixing tendency of {tracer} per wavenumber', tend_units]
            dataToSave[f'{tracer}_divm_WN_{sfx}'] = [divM_phi + divM_theta, f'{sfx}-wave total eddy mixing tendency of {tracer} per wavenumber', tend_units]

    # ------------------------------------------------------------------
    # Wavenumber banding
    # ------------------------------------------------------------------
    waves_cfg = tomlConfig.get('Waves', ['all'])
    if len(waves_cfg) == 1 and waves_cfg[0].lower() == 'all':
        wave_numbers = [str(k) for k in range(1, N_lon // 2 + 1)]
    else:
        dataToSave, wave_numbers = _band_all(dataToSave, waves_cfg)

    return dataToSave, lats, thetaLevels_K, wave_numbers
