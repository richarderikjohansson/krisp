# Naming Conventions

This document describes the naming conventions used in krisp for all files and
directories it produces or depends on. Consistency across these conventions is
what enables full traceability from a retrieval result back to the raw
measurement and every processing step in between.

---

## Configuration and Log Directory

krisp stores all logs under a directory created in the user's home directory on
first use:

```
~/.krisp/
└── logs/
```

The directory is created automatically the first time any logging occurs. No
manual setup is required.

---

## Log Files

Log files are named by the time the krisp process started, in UTC, formatted as
ISO 8601 with colons removed for filesystem compatibility:

```
~/.krisp/logs/<run_ts>.log
```

| Component | Description |
|---|---|
| `run_ts` | UTC timestamp of process startup, format `YYYY-MM-DDTHHMMSS` |

**Example:**
```
~/.krisp/logs/2024-03-15T021400.log
```

Log files are named by *processing time*, not measurement time. The link
between a log file and a specific measurement is established through the
`id` field present on every log line (see [Log Line Format](#log-line-format)).

### Log Line Format

Every log line follows the same structure:

```
<iso_timestamp> | <LEVEL>    | <logger>:<function>:<line> - <message> | key=value ...
```

**Example lines:**
```
2024-03-15T02:14:03+00:00 | INFO     | krisp.arts.retrieval:run:42 - Retrieval started | id=1710460800 instrument=KIMRA
2024-03-15T02:14:05+00:00 | INFO     | krisp.arts.retrieval:run:87 - Converged | id=1710460800 chi2=0.0023 iterations=14
2024-03-15T02:14:06+00:00 | WARNING  | krisp.arts.retrieval:run:91 - Chi2 elevated | id=1710460800 chi2=0.0023 threshold=0.001
2024-03-15T02:14:07+00:00 | ERROR    | krisp.daemon.watcher:dispatch:33 - File not found | id=1710461500
```

The `id` field is the UNIX timestamp of the measurement midpoint and
is included on every log line that relates to a specific measurement. This
allows the complete processing history of any measurement to be retrieved with
a single command:

```bash
grep "id=1710460800" ~/.krisp/logs/*.log
```

---

## Measurement Files

Measurement files are produced and stored on the server `databear` at IRF (Kiruna) by radex and consumed by krisp. They are HDF5
files organised under a directory hierarchy rooted at the configured data directory:

```
databear:/home/data/mwr_data/<Instrument>/<Spectrometer>/<Year>/<Month>/<Day>/
    <instrument>_<id>_<iso_id>.h5
```


| Component | Description |
|---|---|
| `Instrument` | Instrument name, capitalised, e.g. `KIMRA` |
| `Spectrometer` | Spectrometer name, capitalised, e.g. `RPGFFTS` |
| `Year` | Four-digit UTC year, e.g. `2018` |
| `Month` | Two-digit UTC month, e.g. `06` |
| `Day` | Two-digit UTC day, e.g. `20` |
| `instrument` | Instrument name, lowercase, e.g. `kimra` |
| `id` | Unix timestamp of the measurement midpoint, integer, UTC |
| `iso_id` | ISO 8601 of the same midpoint, e.g. `2024-03-15T02:00:00` |

**Example:**
```
KIMRA/RPGFFTS/2024/03/15/kimra_1710460800_2024-03-15T02:00:00.h5
```

`id` is the canonical unique key for a measurement. It is used as
the primary identifier throughout krisp and in all log lines. The ISO
timestamp in the filename is provided for human readability when browsing
the directory tree and always refers to the same moment as `id`.

---

## HDF5 File Structure

Each measurement file contains four top-level groups:

```
kimra_1710460800_2024-03-15T02:00:00.h5
├── /provenance/
├── /measurement/
├── /auxiliary/
└── /retrievals/
    ├── /ozone_v1/
    └── /ozone_v2/
```

### `/provenance`

Written by radex at file creation time and never modified afterward. Contains
all metadata required to identify and reproduce the measurement.

| Attribute | Type | Description |
|---|---|---|
| `id` | int | Unix timestamp of measurement midpoint, UTC |
| `meas_ts_start` | int | Unix timestamp of measurement start, UTC |
| `meas_ts_end` | int | Unix timestamp of measurement end, UTC |
| `meas_duration_s` | float | Duration of measurement in seconds |
| `iso_id` | str | ISO 8601 of measurement midpoint, UTC |
| `instrument` | str | Instrument name, e.g. `KIMRA` |
| `spectrometer` | str | Spectrometer name, e.g. `RPGFFTS` |
| `mode` | str | Instrument mode, e.g. `CO_O3` |
| `azimuth` | float | Azimuth angle in degrees |
| `elevation` | float | Elevation angle in degrees |
| `zenith` | float | Zenith angle in degrees |
| `latitude` | float | Station latitude in degrees north |
| `longitude` | float | Station longitude in degrees east |
| `altitude_station` | float | Station altitude in metres above sea level |
| `source` | str | Absolute path to the raw instrument database file |
| `source_type` | str | `database` |
| `radex_version` | str | Version of radex that produced the file |
| `radex_log` | str | Filename of the radex log for this processing run |
| `created_ts` | int | Unix timestamp of when this file was written, UTC |

### `/measurement`

Contains the calibrated spectrum and a priori atmospheric profiles as an
xarray-compatible dataset.

**Dimensions:**

| Dimension | Size | Description |
|---|---|---|
| `level` | 137 | Atmospheric layer index |
| `fb` | 32768 | Frequency backend |

**Coordinates:**

| Name | Dimension | Units | Description |
|---|---|---|---|
| `level` | `level` | `1` | Atmospheric layer index |
| `p` | `level` | `Pa` | Pressure at layer midpoint |
| `z` | `level` | `m` | Geometric altitude at layer midpoint |
| `fb` | `fb` | `Hz` | Frequency backend |

`p` and `z` are both coordinates on the same `level` dimension.
They describe the same 137 atmospheric layers expressed in two different
vertical coordinate systems obtained from ECMWF.

**Data variables:**

| Name | Dimension | Units | Description |
|---|---|---|---|
| `T` | `level` | `K` | Atmospheric temperature |
| `h2o` | `level` | `VMR` | Water vapour volume mixing ratio |
| `o3` | `level` | `VMR` | Ozone volume mixing ratio |
| `y` | `fb` | `K` | Observed brightness temperature spectrum |

Each coordinate and data variable carries `description` and `units` attributes.
`units` follows CF conventions throughout.

### `/auxiliary`

Stores references to the CDS auxiliary data files used during parsing.

| Attribute | Type | Description |
|---|---|---|
| `cds_products` | list[str] | Names of CDS products fetched |
| `<product>_<id>`.nc | int | `id` linking to the corresponding `.nc` file |

**Example:**
```
cds_products:       ["ecmwf_coordinate", "ecmwf_profile"]
ecmwf_coordinate_ts:   1710460800
ecmwf_profile_ts:     1710460800
```

### `/retrievals/<type>_v<n>`

Written by krisp. Each retrieval is stored under a versioned group name of the
form `<retrieval_type>_v<n>`, for example `ozone_v1`. Reprocessing a
measurement increments the version number, when started from watcher. Thus existing groups are never
overwritten.

**Group attributes:**

**This is not decided yet**
| Attribute | Type | Description |
|---|---|---|
| `null` | null | null |
---

**Datasets:**

**This is not decided yet**
| Name | Dimension | Units | Description |
|---|---|---|---|
| `null` | `null` | `null` | null|
---

## Traceability

Every artifact produced by radex and krisp is linked to every other artifact
through `id` as the common key, for example:

```
id=1710460800
    │
    ├── databear:/home/data/mwr_data/KIMRA/RPGFFTS/2024/03/15/kimra_1710460800_2024-03-15T02:00:00.h5
    │       /provenance       → radex_version, radex_log
    │       /measurement      → raw spectrum and a priori profiles
    │       /auxiliary        → links to .nc files by id
    │       /retrievals/ozone_v1 → krisp_version, krisp_log
    │
    ├── databear:/home/data/.radex/ecmwf/ecmwf_coordinate_1710460800.nc
    ├── databear:/home/data/.radex/ecmwf/ecmwf_ozone_1710460800.nc
    │
    ├── databear:/home/data/.radex/logs/2024-03-15T015800.log
    │       grep id=1710460800 → all parsing events
    │
    └── ~/.krisp/logs/2024-03-15T021400.log
            grep id=1710460800 → all retrieval events
```

The `.h5` file is the canonical hub. It holds explicit references to the log
files that produced it and the auxiliary data it depends on, making the file
fully self-documenting without requiring any external index or database.
