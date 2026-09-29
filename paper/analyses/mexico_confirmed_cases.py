"""
Mexico confirmed-case loader for S2, S3 and S9.

Applies the definitions of the primary analysis
(data/mexico_covid/analyze_mexico_ecological.py) to the 3 January 2022 snapshot:
confirmed cases are CLASIFICACION_FINAL 1-3, death is a recorded FECHA_DEF, the
site is the state of the treating medical unit (ENTIDAD_UM), elderly is age 70+,
male is SEXO 2, and comorbidity is any of nine flags. Unknown sex and unknown
comorbidity codes fall in the reference category, as they do in the primary
analysis; the audit file counts them.

`load()` refuses to return data that do not reproduce the primary analysis: the
file hash, the confirmed-case, death and site counts, and every site's size,
mortality rate, elderly, male and comorbidity share.
"""

import hashlib
import importlib.util
import json
from datetime import datetime

import numpy as np
import pandas as pd

from paths import (EXPECTED_SNAPSHOT_SHA256, MEXICO_DIR, MEXICO_MAIN_RESULTS,
                   MEXICO_SNAPSHOT_CSV, RESULTS, write_result)

COMORBIDITY_COLS = [
    "DIABETES", "HIPERTENSION", "OBESIDAD", "CARDIOVASCULAR",
    "RENAL_CRONICA", "EPOC", "ASMA", "INMUSUPR", "TABAQUISMO",
]
SITE_COLS = ["ENTIDAD_UM", "ENTIDAD_RES", "MUNICIPIO_RES", "SECTOR"]
USECOLS = (["ID_REGISTRO", "SEXO", "EDAD", "FECHA_DEF", "FECHA_SINTOMAS", "CLASIFICACION_FINAL"]
           + SITE_COLS + COMORBIDITY_COLS)
AGE_THRESHOLD = 70
ALIVE_CODE = "9999-99-99"
# Sentinel codes, read from the Secretaría de Salud catalogue shipped with the open data
# (diccionario_datos_abiertos.zip, sheet "Catálogo MUNICIPIOS" and "Catálogo SECTOR"). Real
# municipality keys run to 570; real sector codes to 15.
MUNICIPALITY_SENTINELS = {997: "NO APLICA", 998: "SE IGNORA", 999: "NO ESPECIFICADO"}
VALID_STATES = range(1, 33)          # the catalogue's 32 federal entities
VALID_MUNICIPALITIES = range(1, 571)  # municipality codes within a state
SECTOR_SENTINELS = {99: "NO ESPECIFICADO"}
DICTIONARY = ("diccionario_datos_abiertos.zip, Secretaría de Salud, catalogue dated 2024-07-08; "
              "archived beside the snapshot")
ONSET_RANGE = ("2020-01-01", "2022-01-03")
EXPECTED_CSV_SHA256 = EXPECTED_SNAPSHOT_SHA256
SOURCE_URL = ("https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/"
              "historicos/2022/01/datos_abiertos_covid19_03.01.2022.zip")
TOLERANCE = 1e-9

_spec = importlib.util.spec_from_file_location(
    "analyze_mexico_ecological", MEXICO_DIR / "analyze_mexico_ecological.py")
_primary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_primary)
STATE_NAMES = _primary.STATE_NAMES


class LoaderMismatchError(ValueError):
    """The loaded data do not reproduce the primary analysis."""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _counts(series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def _site_table(df) -> pd.DataFrame:
    table = df.groupby("site").agg(
        n=("died", "size"),
        deaths=("died", "sum"),
        mortality_rate=("died", "mean"),
        prop_elderly=("elderly", "mean"),
        prop_male=("male", "mean"),
        prop_comorbidity=("has_comorbidity", "mean"),
    )
    table.index = [STATE_NAMES[code] for code in table.index]
    return table.sort_index()


def _compare_with_primary(df, site_table) -> tuple[dict, list[str]]:
    primary = json.load(open(MEXICO_MAIN_RESULTS))
    expected = {"n": primary["n_total"], "deaths": primary["n_deaths"], "sites": primary["n_sites"]}
    observed = {"n": len(df), "deaths": int(df["died"].sum()), "sites": int(df["site"].nunique())}

    problems = [f"{k}: loader {observed[k]}, primary analysis {expected[k]}"
                for k in expected if observed[k] != expected[k]]

    primary_sites = primary["ecological_regression"]["site_details"]
    missing = set(primary_sites) ^ set(site_table.index)
    if missing:
        problems.append(f"site names differ from the primary analysis: {sorted(missing)}")
    else:
        for name, row in site_table.iterrows():
            for field in ["n", "mortality_rate", "prop_elderly", "prop_male", "prop_comorbidity"]:
                if not np.isclose(row[field], primary_sites[name][field], rtol=0, atol=TOLERANCE):
                    problems.append(f"{name}.{field}: loader {row[field]}, "
                                     f"primary analysis {primary_sites[name][field]}")
    return {"expected_from_primary_analysis": expected, "observed": observed}, problems


def records(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per confirmed case, from the snapshot's raw columns.

    Sentinel codes carry no value rather than a wrong one: a municipality coded 997, 998
    or 999, and a sector coded 99, become missing, so those records fall outside the site
    definition that uses them instead of forming a pseudo-site of their own.
    """
    onset = pd.to_datetime(raw["FECHA_SINTOMAS"], format="%Y-%m-%d", errors="coerce")
    onset_valid = onset.between(*ONSET_RANGE)

    # A code outside its catalogue carries no site, in the same way a sentinel does, so
    # the cohort is what the catalogue admits rather than what the file happens to hold.
    treating = raw["ENTIDAD_UM"].astype(int)
    residence = raw["ENTIDAD_RES"].astype(int)
    municipality_code = raw["MUNICIPIO_RES"].astype(int)
    sector_code = raw["SECTOR"].astype(int)
    residence_valid = residence.isin(VALID_STATES)
    municipality_valid = (residence_valid & municipality_code.isin(VALID_MUNICIPALITIES))

    df = pd.DataFrame({
        # A treating-unit state outside the catalogue would change the primary cohort, so
        # the loader refuses the file rather than quietly dropping the record; residence
        # and municipality only feed alternative site definitions, so they carry no value.
        "site": treating,
        "site_residence": residence.where(residence_valid).astype("Int64"),
        "municipality": np.where(municipality_valid,
                                 residence * 1000 + municipality_code, np.nan),
        "sector": np.where(sector_code.isin(SECTOR_SENTINELS), np.nan, sector_code),
        "died": (raw["FECHA_DEF"] != ALIVE_CODE).astype(int),
        "age": raw["EDAD"].astype(int),
        "elderly": (raw["EDAD"] >= AGE_THRESHOLD).astype(int),
        "male": (raw["SEXO"] == 2).astype(int),
        "has_comorbidity": ((raw[COMORBIDITY_COLS] == 1).sum(axis=1) > 0).astype(int),
        "onset": onset.where(onset_valid),
        # `astype(str)` renders a missing period as the string "NaT" in some pandas
        # versions, which `dropna` then keeps as a calendar-month category, so the mask
        # is applied after the conversion rather than relied on through it.
        "onset_month": onset.dt.to_period("M").astype("string").where(onset_valid),
    })
    for col in COMORBIDITY_COLS:
        df[col.lower()] = (raw[col] == 1).astype(int)
    df["comorbidity_unknown"] = (~raw[COMORBIDITY_COLS].isin([1, 2])).all(axis=1).astype(int)
    return df


def load() -> pd.DataFrame:
    """Return one row per confirmed case: site, died, elderly, male, has_comorbidity."""
    csv_hash = sha256(MEXICO_SNAPSHOT_CSV)
    if csv_hash != EXPECTED_CSV_SHA256:
        raise LoaderMismatchError(
            f"{MEXICO_SNAPSHOT_CSV.name} hashes to {csv_hash}; the snapshot this analysis "
            f"is registered against hashes to {EXPECTED_CSV_SHA256} (source: {SOURCE_URL})")

    total_rows = 0
    class_counts: dict[str, int] = {}
    frames = []
    for chunk in pd.read_csv(MEXICO_SNAPSHOT_CSV, usecols=USECOLS, chunksize=2_000_000,
                             encoding="latin-1", low_memory=False):
        total_rows += len(chunk)
        for k, v in _counts(chunk["CLASIFICACION_FINAL"]).items():
            class_counts[k] = class_counts.get(k, 0) + v
        frames.append(chunk[chunk["CLASIFICACION_FINAL"].isin([1, 2, 3])])
    raw = pd.concat(frames, ignore_index=True)

    df = records(raw)

    site_table = _site_table(df)
    counts, problems = _compare_with_primary(df, site_table)

    unknown_states = sorted(set(df["site"]) - set(STATE_NAMES))
    if unknown_states:
        problems.append(f"state codes outside 1-32: {unknown_states}")

    death_dates = raw.loc[raw["FECHA_DEF"] != ALIVE_CODE, "FECHA_DEF"]
    unparseable_deaths = int(pd.to_datetime(death_dates, format="%Y-%m-%d", errors="coerce").isna().sum())
    municipality_codes = raw["MUNICIPIO_RES"].astype(int)
    unknown_municipality = int(municipality_codes.isin(MUNICIPALITY_SENTINELS).sum())
    outside_catalogue = {
        "ENTIDAD_UM_outside_1_32": int((~raw["ENTIDAD_UM"].astype(int).isin(VALID_STATES)).sum()),
        "ENTIDAD_RES_outside_1_32": int((~raw["ENTIDAD_RES"].astype(int).isin(VALID_STATES)).sum()),
        "MUNICIPIO_RES_outside_1_570_and_not_a_sentinel": int((
            ~municipality_codes.isin(list(VALID_MUNICIPALITIES) + list(MUNICIPALITY_SENTINELS))
        ).sum()),
        "records_with_no_municipality": int(df["municipality"].isna().sum()),
        "records_with_no_residence_state": int(df["site_residence"].isna().sum()),
        "records_with_no_treating_state": int(df["site"].isna().sum()),
        "records_with_no_onset_month": int(df["onset_month"].isna().sum()),
    }
    sector_codes = raw["SECTOR"].astype(int)
    duplicate_ids = int(raw["ID_REGISTRO"].duplicated().sum())

    audit = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_url": SOURCE_URL,
        "csv": MEXICO_SNAPSHOT_CSV.name,
        "csv_sha256": csv_hash,
        "total_rows": total_rows,
        "clasificacion_final_counts": class_counts,
        "definitions": {
            "confirmed": "CLASIFICACION_FINAL in {1, 2, 3}",
            "death": f"FECHA_DEF != '{ALIVE_CODE}'",
            "site": "ENTIDAD_UM (state of the treating medical unit)",
            "elderly": f"EDAD >= {AGE_THRESHOLD}",
            "male": "SEXO == 2 (1 is female, 99 unknown; unknown falls in the reference group)",
            "comorbidity": (f"any of {COMORBIDITY_COLS} == 1 (2 is no, 97/98/99 unknown; "
                             "unknown falls in the reference group)"),
            "municipality": (f"ENTIDAD_RES x 1000 + MUNICIPIO_RES; codes "
                              f"{dict(MUNICIPALITY_SENTINELS)} carry no municipality"),
            "sector": f"SECTOR; codes {dict(SECTOR_SENTINELS)} carry no sector",
            "catalogue": DICTIONARY,
        },
        "codes_among_confirmed_cases": {
            "EDAD_min": int(raw["EDAD"].min()),
            "EDAD_max": int(raw["EDAD"].max()),
            "EDAD_over_120": int((raw["EDAD"] > 120).sum()),
            "EDAD_missing": int(raw["EDAD"].isna().sum()),
            "SEXO": _counts(raw["SEXO"]),
            "ENTIDAD_UM": _counts(raw["ENTIDAD_UM"]),
            "ENTIDAD_RES": _counts(raw["ENTIDAD_RES"]),
            "SECTOR": _counts(raw["SECTOR"]),
            "MUNICIPIO_RES_sentinels": {
                f"{code} ({label})": int((municipality_codes == code).sum())
                for code, label in MUNICIPALITY_SENTINELS.items()},
            "MUNICIPIO_RES_codes_above_570": {
                str(code): int((municipality_codes == code).sum())
                for code in sorted(set(municipality_codes[municipality_codes > 570]))},
            "SECTOR_sentinels": {f"{code} ({label})": int((sector_codes == code).sum())
                                  for code, label in SECTOR_SENTINELS.items()},
            "outside_catalogue": outside_catalogue,
            "MUNICIPIO_RES_sentinel_records": unknown_municipality,
            "MUNICIPIO_RES_distinct_valid": int(df["municipality"].nunique()),
            "MUNICIPIO_RES_max_sentinel_pseudo_site": int(
                raw.loc[municipality_codes.isin(MUNICIPALITY_SENTINELS)]
                .groupby("ENTIDAD_RES").size().max()) if unknown_municipality else 0,
            "FECHA_SINTOMAS_min": str(onset.min()),
            "FECHA_SINTOMAS_max": str(onset.max()),
            "FECHA_SINTOMAS_invalid_or_out_of_range": int((~onset_valid).sum()),
            "onset_months": int(df["onset_month"].nunique()),
            "FECHA_DEF_alive_code": int((raw["FECHA_DEF"] == ALIVE_CODE).sum()),
            "FECHA_DEF_missing": int(raw["FECHA_DEF"].isna().sum()),
            "FECHA_DEF_unparseable_dates": unparseable_deaths,
            "comorbidity_flags": {c: _counts(raw[c]) for c in COMORBIDITY_COLS},
        },
        "record_identifier": {
            "field": "ID_REGISTRO",
            "duplicated_values": duplicate_ids,
            "unique": duplicate_ids == 0,
            "note": ("a record identifier, not a person identifier: uniqueness here means one row "
                      "per record, and the same person may appear in more than one record"),
        },
        "unknown_in_reference_group": {
            "sex_unknown": int((~raw["SEXO"].isin([1, 2])).sum()),
            "all_comorbidity_flags_unknown": int(
                (~raw[COMORBIDITY_COLS].isin([1, 2])).all(axis=1).sum()),
        },
        **counts,
        "site_table": json.loads(site_table.to_json(orient="index")),
        "matches_primary_analysis": not problems,
        "problems": problems,
    }
    RESULTS.mkdir(exist_ok=True)
    write_result(audit, RESULTS / "mexico_loader_audit.json")

    if problems:
        raise LoaderMismatchError("; ".join(problems))
    return df


def collapsed_cells(df) -> pd.DataFrame:
    """Deaths and patients per site x elderly x male x comorbidity cell.

    The patient-level covariates are all binary, so a binomial fit on these cells
    has the same likelihood as the patient-level fit.
    """
    cells = df.groupby(["site", "elderly", "male", "has_comorbidity"]).agg(
        deaths=("died", "sum"), n=("died", "size")).reset_index()
    cells["alive"] = cells["n"] - cells["deaths"]
    return cells
