# Redownload Instructions

Data files were deleted on 2026-07-10 to free disk space. All data is
publicly available and can be re-downloaded as described below.

## Virtual environment

```bash
uv venv && uv sync
```

## data/cdc_case_surveillance/cdc_full.csv (~12 GB)

CDC COVID-19 Case Surveillance 12-element Public Use Data (dataset ID: `vbim-akqf`).
Covers tens of millions of individual COVID-19 cases with demographics and outcomes.
Reporting discontinued July 1, 2024; historical data through May 2023 remains available.

```bash
mkdir -p data/cdc_case_surveillance
curl -L -o data/cdc_case_surveillance/cdc_full.csv \
  "https://data.cdc.gov/api/views/vbim-akqf/rows.csv?accessType=DOWNLOAD"
```

See also: COVID19_INDIVIDUAL_DATA_SOURCES.md for alternative/extended datasets.

## data/mexico_covid/ (~2.1 GB)

Mexico federal open COVID-19 data from Direccion General de Epidemiologia.

- `220103COVID19MEXICO.csv` (2.0 GB) — snapshot from January 3, 2022

**No download needed.** The source archive is kept at
`~/Documents/backup/ecological-bias-covid/datos_abiertos_covid19_03.01.2022.zip`
(237 MB, sha256 `63993520348a74ba...f272dc`, the hash the correction addendum records).
It holds exactly the 2,014,503,018-byte CSV the loader expects:

```bash
unzip -o ~/Documents/backup/ecological-bias-covid/datos_abiertos_covid19_03.01.2022.zip \
  -d data/mexico_covid/
```

The loader hashes the result and refuses to return data unless it is the registered
snapshot, so a wrong or truncated restore fails loudly rather than quietly. The same
file is also on the Modal volume `jamia-mexico-snapshot`, which is where the analyses
read it; that volume is not a backup.
- `COVID19MEXICO.csv` (38 MB) — smaller extract
- `datos_abiertos_covid19_2022.zip` (249 MB) — raw archive
- `datos_abiertos_covid19.zip` (4 MB) — raw archive

Source: https://www.gob.mx/salud/documentos/datos-abiertos-152127
(historical archives may be available via Wayback Machine or academic mirrors).

## data/zenodo_ny_hospitals/ (~12 MB)

Anonymized COVID-19 patient data from SUNY Downstate and Maimonides Medical Center
(n=1,540 patients, Feb-May 2020). Zenodo record 6771834.

```bash
mkdir -p data/zenodo_ny_hospitals
curl -L -o data/zenodo_ny_hospitals/demographics_both_hospitals.csv \
  "https://zenodo.org/records/6771834/files/demographics_both_hospitals.csv"
curl -L -o data/zenodo_ny_hospitals/dynamics_clean_both_hospitals.csv \
  "https://zenodo.org/records/6771834/files/dynamics_clean_both_hospitals.csv"
```

## Computed results (small, not deleted)

- `data/cdc_case_surveillance/results/` (12 KB) — analysis outputs
- `data/geometric_results/` (64 KB) — geometric analysis outputs
- `data/mexico_covid/mexico_ecological_results.json` (14 KB) — Mexico analysis output

These are regenerable by running the analysis scripts in each data subdirectory.
