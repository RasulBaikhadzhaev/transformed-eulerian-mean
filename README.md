# Transformed Eulerian Mean

> **Before publishing:** CHECK EQUATIONS IN THE README SOME ARE NOT DISPLAYED PROPERLY Update the Zenodo DOI badge below and in `CITATION.cff` (ALSO AT THE END OF README IN HOW TO CITE SECTION) with the real DOI after uploading to Zenodo. Also update the title, authors, and journal details in `CITATION.cff`.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

A Python package for Transformed Eulerian Mean (TEM) diagnostics of atmospheric circulation.
TEM separates the meridional circulation into a residual mean flow and eddy-driven components,
making it a standard framework for studying stratospheric dynamics and tracer transport.

The package provides three calculators:

| Calculator | Command | What it computes |
|---|---|---|
| Residual circulation | `residual` | Residual mean flow ($\bar{v}^*$, $\bar{w}^*$), Eliassen–Palm flux and its divergence, mass stream function |
| Tracer transport (pressure coords) | `transport-press` | Zonal-mean tracer budget: residual advection, eddy mixing tendencies, eddy flux vectors |
| Tracer transport (theta coords) | `transport-theta` | Same as above but in isentropic (potential temperature) coordinates |

All three read NetCDF input files and write NetCDF output files.

---

## Prerequisites

**Pixi** is required to manage dependencies, including Python itself.
See the [Pixi installation guide](https://pixi.sh/latest/#installation).

---

## Installation

```bash
git clone https://github.com/RasulBaikhadzhaev/transformed-eulerian-mean.git
cd TEM_pkg
pixi install
```

---

## Sample data and quick verification

The repository includes low-resolution sample input files (ERA5-like and CLaMS-like NetCDF) in `tests/data/sample_input/`. These are intended for testing and verification only — the spatial and temporal resolution is too coarse for scientific use.

To generate sample output using the shipped config files:

```bash
pixi run residual          config/residualCirc_config.toml
pixi run transport-press   config/tracerTransportPress_config.toml
pixi run transport-theta   config/tracerTransportTheta_config.toml
```

Output is written to `tests/data/sample_output/residual_circulation/`, `tests/data/sample_output/tTransport_press/`, and `tests/data/sample_output/tTransport_theta/` respectively.

---

## Scientific background

TEM diagnostics follow the Andrews and McIntyre (1976) and Andrews et al. (1987) formalism. The core equations implemented by this package are summarised below.

### Log-pressure altitude coordinate

`residual` and `transport-press` use log-pressure altitude as the vertical coordinate:

$$z = -H \ln(p / p_s)$$

where $H = 7$ km is the scale height and $p_s = 1000$ hPa is the reference pressure.

### Residual mean circulation

The residual mean meridional and vertical velocities are (Andrews et al., 1987):

$$\bar{v}^* = \bar{v} - \rho_0^{-1} \partial_z\!\left(\rho_0 \frac{\overline{v'\theta'}}{\partial_z\bar{\theta}}\right), \qquad \bar{w}^* = \bar{w} + \frac{1}{a\cos\phi} \partial_\phi\!\left(\cos\phi\, \frac{\overline{v'\theta'}}{\partial_z\bar{\theta}}\right)$$

where $\rho_0$ is the reference density, $a$ is Earth's radius, $\phi$ is latitude, $\theta$ is potential temperature, and primes denote deviations from the zonal mean.

### Eliassen–Palm flux

The EP flux quantifies the meridional propagation of wave activity:

$$F^{(\phi)} = \rho_0 a \cos\phi \left(\frac{\partial_z\bar{u}}{\partial_z\bar{\theta}}\,\overline{v'\theta'} - \overline{v'u'}\right)$$

$$F^{(z)} = \rho_0 a \cos\phi \left\{\left[f - \frac{\partial_\phi(\bar{u}\cos\phi)}{a\cos\phi}\right] \frac{\overline{v'\theta'}}{\partial_z\bar{\theta}} - \overline{w'u'}\right\}$$

Its divergence $\nabla \cdot F$ is the wave forcing on the zonal mean flow and is saved in the output alongside the individual components.

### Residual mass stream function

The `residual` and `transport-press` calculators compute the TEM residual mass stream function by integrating $\bar{v}^*$ downward from the model top:

$$\psi^*(\phi, z) = -\cos\phi \int_z^\infty \rho_0\, \bar{v}^*(\phi, z')\, \mathrm{d}z'$$

### Tracer transport in log-pressure coordinates

The zonal-mean tracer continuity equation within the TEM framework is (Andrews et al., 1987, Eq. 9.4.13):

$$\partial_t \bar{\chi} = \bar{S} - \frac{\bar{v}^*}{a} \partial_\phi \bar{\chi} - \bar{w}^* \partial_z \bar{\chi} + \rho_0^{-1} \nabla \cdot M$$

where $\chi$ is the tracer mixing ratio, $\bar{S}$ represents chemical sources and sinks, and $M$ is the eddy flux vector with components:

$$M^{(\phi)} = -\rho_0\!\left(\overline{v'\chi'} - \overline{v'\theta'}\,\frac{\partial_z \bar{\chi}}{\partial_z \bar{\theta}}\right), \qquad M^{(z)} = -\rho_0\!\left(\overline{w'\chi'} + \overline{v'\theta'}\,\frac{\partial_\phi \bar{\chi}}{a\,\partial_z \bar{\theta}}\right)$$

### Tracer transport in isentropic coordinates

When potential temperature $\theta$ is the vertical coordinate, the analogue of the residual circulation is the diabatic mean circulation:

$$\bar{v}^* = \overline{\sigma v}/\overline{\sigma}, \qquad \overline{Q^*} = \overline{\sigma Q}/\overline{\sigma}$$

where $\sigma = -g^{-1}\partial_\theta p$ is the isentropic density and $Q = \mathrm{d}\theta/\mathrm{d}t$ is the diabatic heating rate. The tracer transport equation in isentropic coordinates is (Andrews et al., 1987, Eq. 9.4.21):

$$\partial_t \bar{\chi} = \bar{S} - \frac{\bar{v}^*}{a}\partial_\phi \bar{\chi} - \overline{Q^*}\partial_\theta \bar{\chi} + \frac{1}{\bar{\sigma}}\!\left[\nabla \cdot M - \partial_t\!\left(\overline{\sigma'\chi'}\right)\right]$$

with eddy flux vector components:

$$M^{(\phi)} = -\overline{(\sigma v)'\chi'}, \qquad M^{(\theta)} = -\overline{(\sigma Q)'\chi'}$$

---

## Quick start

Each calculator is driven by a TOML configuration file.
Template configs for all three calculators are in the `config/` directory.

1. Copy the relevant template and edit it for your data:

   ```bash
   cp config/residualCirc_config.toml my_run.toml
   # edit my_run.toml
   ```

2. Run:

   ```bash
   pixi run residual my_run.toml
   pixi run transport-press my_run.toml
   pixi run transport-theta my_run.toml
   ```

---

## Configuration

Navigate to `config/` and open the relevant template in a text editor:

| Calculator | Template file |
|---|---|
| Residual circulation | `residualCirc_config.toml` |
| Tracer transport (pressure) | `tracerTransportPress_config.toml` |
| Tracer transport (theta) | `tracerTransportTheta_config.toml` |

Each file is fully commented. At minimum you must set the input/output paths and the variable names that match your NetCDF files.

The sections below document the non-obvious settings that are most likely to require attention.

---

### File naming pattern (`timeInfoInFileNames`)

The calculators extract timestamps from file names using a pattern string. Every character in the file name must be accounted for:

- Use `YYYY` (or `YY` if the century is absent), `MM`, `DD`, `HH`, `mm`, `ss` for time fields.
- Use `?` for every non-time character (underscores, letters, etc.).
- Use `*` as a wildcard for a variable-length prefix or suffix — exactly one `*` is allowed and it must be either the first or last character.

Year is mandatory; include all other fields that are present in your file names.

```
File name:  era5_sample_20000101_00.nc
Pattern:    *YYYYMMDD?HH??.nc
```
```
File name:  2000010100_era5.nc
Pattern:    YYYYMMDDHH*
```

---

### Vertical coordinates (`verticalDimensionType`, `targetLevels`)

All calculations are performed on a fixed vertical grid. How the input data is mapped to that grid depends on `verticalDimensionType`:

| Value | Meaning |
|---|---|
| `'pressure'` | Input vertical axis is pressure (1-D); interpolated to log-pressure altitude levels (km) |
| `'log-pressure'` | Input vertical axis is already log-pressure altitude; no interpolation needed |
| `'other'` | Input is on model levels (sigma, hybrid, etc.); a 3-D pressure variable is required (`pressureName`) |
| `'theta'` | `transport-theta` only: input vertical axis is potential temperature; no interpolation needed |

`verticalWindType` must also be set to match the vertical wind variable available in your data:

| Value | Meaning |
|---|---|
| `'omega'` | Vertical wind is pressure velocity (Pa s⁻¹); converted to m s⁻¹ internally |
| `'W'` | Vertical wind is already in m s⁻¹ |
| `'missing'` | No vertical wind available; W\* is diagnosed from V\* via the residual mass stream function (residual only) |
| `'thetaDot'` | Diabatic heating rate dθ/dt (K s⁻¹); required for `transport-theta` |

`targetLevels` sets the output vertical grid. For `residual` and `transport-press` it is in km (log-pressure altitude); for `transport-theta` it is in K (potential temperature).

When the input is already on the target coordinate system (`'pressure'`, `'log-pressure'`, or `'theta'`), re-interpolation to a fixed grid can be skipped by setting `targetLevels = 'skip'`. The native vertical levels are then kept as-is. This is not supported when `verticalDimensionType = 'other'`, where a numeric `targetLevels` array is always required.

---

### Wavenumber decomposition (`FourierTransform`, `Waves`)

Set `FourierTransform = true` to enable per-wavenumber output. The `Waves` list controls which wavenumber bands are saved:

| Entry | Effect |
|---|---|
| `'1'`, `'2'`, … | Save that individual wavenumber |
| `'3-5'` | Sum wavenumbers 3 through 5 into a single band |
| `'21-end'` | Sum from wavenumber 21 to the last resolved one |
| `['all']` | Expand to every individual wavenumber from 1 to N_lon/2 |

Wavenumber 0 (the zonal mean) is always excluded. The decomposition requires vertical wind data and is disabled automatically when `verticalWindType = 'missing'`.

---

### Missing vertical wind (`verticalWindType = 'missing'`, residual only)

If no vertical wind data is available, set `verticalWindType = 'missing'`. $\bar{w}^*$ is then estimated diagnostically from $\bar{v}^*$ via the residual mass stream function, and the EP flux vertical component is computed without the w′u′ term. Fourier decomposition of the EP flux requires vertical wind and is disabled automatically.

---

### Temporal averaging of output (`outputTemporalMean`, residual only)

Controls whether individual timesteps or time-averaged fields are written:

| Value | Behaviour |
|---|---|
| `false` | One output file per input timestep |
| `'daily'` | Timesteps within each calendar day are averaged; one output file per day |
| `'monthly'` | Timesteps within each calendar month are averaged; one output file per month |

Monthly and daily means also include `dU_dt` (acceleration of the zonal wind) estimated from the time series within the averaging period.

---

### Saving additional input variables (residual only)

Two optional lists let you piggyback variables from the input files into the output without extra processing:

- `saveInterpolatedZonalMean` — interpolates these variables to `targetLevels` and saves their zonal mean. Useful for variables like temperature, geopotential height, or specific humidity.
- `saveZonalMean` — saves the zonal mean of 2-D (latitude-only) input variables directly without any vertical interpolation. Useful for tropopause diagnostics.

Set either to `[]` to save nothing.

---

### Tracer and met data in separate files (`tracerDataInMetFiles`, transport only)

The transport calculators support two input layouts:

- `tracerDataInMetFiles = true` — tracer mixing ratios and meteorological fields are in the same files.
- `tracerDataInMetFiles = false` — tracer and met data are in separate directories and can have different temporal frequencies. In this case, `tracerInputDirectory`, `tracerInFileNames`, and `tracerTimeInfoInFileNames` point to the tracer files, while `inputDirectory` points to the (typically higher-frequency) met files.

---

### Met data temporal binning (`MetDataBinningTime`, transport only, separate files only)

When tracer and met data are in separate files, `MetDataBinningTime` controls how met files are matched to each tracer timestep:

- `'auto'` — selects all met files within ±½ tracer timestep and weights them by the inverse frequency of their hour-of-day to account for tidal aliases. For example, daily tracer data at 12:00 with 6-hourly met data will use the 00, 06, 12, 18, and 00 (next day) files with weights 0.125, 0.25, 0.25, 0.25, 0.125.
- Integer `N` — selects the N met files closest in time with equal weights, regardless of temporal distance. Use with care if the met data period does not fully overlap the tracer period.

---

### Tracer sinks and sources (`sinksSources`, transport only)

One entry per tracer (same order as `tracerNames`). Three formats are accepted:

| Format | Effect |
|---|---|
| `'0'` | No sink or source |
| `'N'` (digit string) | Constant sink rate of N per second |
| `'half life, N, unit'` | First-order exponential decay with half-life N in the given unit (e.g. `'half life, 90, days'`) |

---

### Spatial binning (`binningLat`, `binningLon`, transport only)

`binningLat` and `binningLon` coarsen the output by averaging that many adjacent grid points into one along each axis. Set to `1` to keep the original resolution. When tracer and met data are on different horizontal grids, met data is interpolated to the tracer grid first, then binning is applied.

---

### Eddy term output (`saveEddyTerms`, residual only)

Set `saveEddyTerms = true` to additionally save:
- The zonal-mean eddy covariances v′θ′, v′u′, and w′u′ on the (alt, lat) grid.
- The full three-dimensional anomaly fields v′, u′, θ′, w′, v′θ′, v′u′, w′u′ on the (alt, lat, lon) grid.

When `FourierTransform = true`, the per-wavenumber covariances v′θ′_k, v′u′_k, and w′u′_k are also written.

---

### Hours filter (`hoursToKeep`, residual only)

`hoursToKeep` restricts input files to specific UTC hours before any other filtering. For example, `hoursToKeep = [0, 6, 12, 18]` keeps only 6-hourly snapshots even if more frequent files are present. Set to `[]` to keep all hours.

---

### Input data description (`inputDataDescription`)

A free-text string that is written as the global `Title` attribute in every output NetCDF file. Useful for recording the source dataset:

```toml
inputDataDescription = 'ERA5.1 reanalysis, ECMWF L137 model levels'
```

---

### Time dimension name (`timeDim`)

Controls how multi-timestep files are handled:

- `timeDim = 'time'` — the input file has a named time dimension; the calculator iterates over each timestep and writes one output file per timestep.
- `timeDim = ''` — each input file contains exactly one timestep with no time dimension.

Note: `transport-press` and `transport-theta` require one NetCDF file per timestep and do not support the multi-timestep mode.

---

### Mass stream function (`massSF`, transport only)

Set `massSF = true` to compute and save the residual mass stream function alongside the tracer budget terms. For `transport-press` this is the log-pressure $\psi^*$ (same formula as the `residual` calculator); for `transport-theta` it is the isentropic stream function $\psi_\theta$, computed by integrating $\bar{v}^*$ over pressure rather than altitude.

---

### Age-of-air and time-unit tracers

Tracers whose units are a time quantity (e.g. age of air in years) are automatically converted to SI base units (seconds) before the budget computation. All output tendency terms for such tracers are in s s⁻¹ (i.e. dimensionless). No configuration is needed — the conversion is triggered by the physical dimensions of the `units` attribute in the input NetCDF file.

---

## Input and output notes

**Input file format**

- `transport-press` and `transport-theta` require **one NetCDF file per timestep**.
- `residual` also accepts files that contain **multiple timesteps** (with a time dimension); it iterates over each timestep internally and writes one output file per timestep (or per month/day when `outputTemporalMean` is set).

**Pointing to input files**

For the residual circulation calculator, `inputPath` can be either a directory (default) or a path to a plain-text `.txt` file that lists absolute input file paths one per line. Set `inputPathType = 'txt'` to use the latter. This is useful when data is spread across multiple directories.

**Output directory must exist before running**

The calculator will exit with an error if `outputDirectory` does not exist. Create the directory manually before running:

```bash
mkdir -p /data/results/2005
pixi run residual my_run.toml --outputDirectory /data/results/2005
```

**Resuming an interrupted run**

Setting `outDirSkip = true` in the config (or `--outDirSkip true` on the command line) causes the calculator to skip any timestamp for which an output file already exists in `outputDirectory`. This lets you safely resume a run that was interrupted without reprocessing completed files.

**`{tracerNames}` placeholder in `outPrefix`**

For tracer transport, the string `{tracerNames}` in `outPrefix` is replaced at runtime with the actual tracer name(s) joined by underscores. For example, with `tracerNames = ['O3', 'N2O']` and `outPrefix = '{tracerNames}_Transport_'`, output files will be named `O3_N2O_Transport_...nc`. This ensures distinct file names when looping over tracers with `--tracerNames`.

---

## Overriding config values on the command line

Any parameter from the TOML file can be overridden by passing it as a `--flag` after the config path.
Command-line values take precedence over the file, everything else falls back to the file.

```bash
pixi run residual my_run.toml --outputDirectory /data/results/2005 --startDate 2005-01-01-00 --endDate 2005-12-31-18
```

This is useful for running the same base config across multiple date ranges or tracers from a bash script:

```bash
for year in 2000 2001 2002 2003; do
    pixi run residual base_config.toml \
        --startDate ${year}-01-01-00 \
        --endDate   ${year}-12-31-18 \
        --outputDirectory /data/results/${year}
done
```

```bash
# tracerNames and sinksSources must be kept in sync — one entry each per tracer
declare -A sinks=( [E90]="half life, 90, days" [N2O]="0" [BA]="1" )

for tracer in E90 N2O BA; do
    pixi run transport-press base_config.toml \
        --tracerNames "[${tracer}]" \
        --sinksSources "['${sinks[$tracer]}']" \
        --outputDirectory /data/results/${tracer}
done
```

All overridable flags are listed by running the calculator with `--help`:

```bash
pixi run residual --help
pixi run transport-press --help
pixi run transport-theta --help
```

---

## Output variables

### Residual circulation (`residual`)

| Variable | Description |
|---|---|
| `V_RES_STD`, `W_RES_STD` | Residual mean meridional and vertical circulation ($\bar{v}^*$, $\bar{w}^*$) |
| `EPF_lat`, `EPF_vert` | Meridional and vertical components of the Eliassen–Palm flux |
| `div_EPF_lat`, `div_EPF_vert`, `div_EPF` | EP flux divergence (components and total) |
| `MASS_SF_RES_STD` | Mass stream function of the residual flow |
| `U`, `V`, `W`, `THETA` | Zonal-mean winds and potential temperature |

Optional eddy terms (`vPrimeUPrimeBar`, `vPrimeThetaPrimeBar`, etc.) and Fourier decompositions by wavenumber are saved when enabled in the config.

### Tracer transport in pressure coordinates (`transport-press`)

| Variable | Description |
|---|---|
| `{tracer}_chi_bar` | Zonal-mean tracer mixing ratio |
| `{tracer}_adv_lat`, `{tracer}_adv_z` | Meridional and vertical residual advection tendencies |
| `{tracer}_divm_lat`, `{tracer}_divm_z` | Meridional and vertical eddy mixing tendencies |
| `{tracer}_m_lat`, `{tracer}_m_z` | Meridional and vertical eddy flux vectors |
| `{tracer}_dt_sum` | Estimated total temporal tendency (sum of all components) |
| `massSF` | Mass stream function (optional) |

### Tracer transport in theta coordinates (`transport-theta`)

Same variables as above, with `_theta` replacing `_z` for the vertical components, plus `PRESS` (zonal-mean pressure).

## Project structure

```
TEM_pkg/
├── config/                          # TOML configuration templates
│   ├── residualCirc_config.toml
│   ├── tracerTransportPress_config.toml
│   └── tracerTransportTheta_config.toml
├── src/tem_pkg/                     # Package source code
│   ├── cli.py                       # Entry points for the three calculators
│   ├── residual_circulation.py      # Residual circulation (V*, W*, EP flux)
│   ├── tracer_transport_press.py    # Tracer transport in pressure coordinates
│   ├── tracer_transport_theta.py    # Tracer transport in theta coordinates
│   ├── file_io.py                   # NetCDF reading/writing and file discovery
│   ├── interpolation.py             # Vertical coordinate interpolation
│   ├── parser.py                    # CLI argument parsers
│   ├── utils.py                     # Shared utilities (config loading, binning, …)
│   └── constants.py                 # Atmospheric constants (via MetPy)
├── tests/
│   ├── data/
│   │   ├── sample_input/            # Small ERA5 and CLaMS sample files for tests
│   │   │   ├── ERA5/
│   │   │   ├── CLaMS_Press/
│   │   │   └── CLaMS_Theta/
│   │   ├── sample_output/           # Output written during test runs
│   │   └── baseline_output/         # Reference output for regression tests
│   └── test_*.py                    # Test modules (one per source module)
├── scripts/
│   └── create_sample_files.py       # Helper to regenerate test input data
└── pyproject.toml                   # Build config, dependencies, Pixi tasks
```

---

## Testing

The package includes unit tests, integration tests, and regression tests against pre-computed reference output. Run the full test suite with coverage:

```bash
pixi run coverage
```

To run lint and tests together (as in CI):

```bash
pixi run ci
```

Sample input files and baseline reference output are in `tests/data/`.

---

## Planned work

### Separation of stationary and transient waves

The Fourier decomposition currently computes waves from instantaneous (daily or sub-daily) fields. A planned extension will separate stationary and transient wave contributions by additionally computing the decomposition on monthly-mean fields. Waves present in the monthly mean represent the stationary component; the difference between the instantaneous and monthly-mean wave fields gives the transient component. This will allow users to attribute tracer transport tendencies and EP flux to stationary versus transient eddies without any additional input data.

### RAM usage optimisation

For large datasets the peak memory footprint during processing can be significant. Planned work includes performing intermediate calculations in 32-bit floating-point precision where the loss of precision is acceptable, and explicitly releasing arrays from memory as soon as they are no longer needed. Together these changes should substantially reduce per-worker memory usage and allow larger grids or longer time series to be processed within typical memory budgets.

---

## How to cite

If you use `tem_pkg` in published work, please cite the archived code release:

> Baikhadzhaev, R. (2025). *tem_pkg: A Python package for Transformed Eulerian Mean diagnostics* (v0.8.0). Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

A `CITATION.cff` file is included in the repository for reference managers that support it. Update the DOI badge and `CITATION.cff` with the real Zenodo DOI after uploading.

---
