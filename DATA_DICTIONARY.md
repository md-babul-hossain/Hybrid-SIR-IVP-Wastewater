# U.S. state–week panel: hospital admissions + wastewater (Jan 2022 – Nov 2026)

File: `us_state_week_panel.csv` — 13,005 rows = 51 jurisdictions (50 states + DC) × 255 weeks.
One row per state per week. `week_end` is the Saturday ending the CDC (MMWR) week.
Built by `build_panel.py` from four CDC downloads (data.cdc.gov, accessed 20 Nov 2026).

## Hospital data — CDC NHSN Hospital Respiratory Data (HRD)
| Column | Meaning |
|---|---|
| `covid_adm`, `flu_adm`, `rsv_adm` | Total new hospital admissions in the week (lab-confirmed) |
| `covid_occ`, `flu_occ`, `rsv_occ` | Total patients hospitalized (occupancy) |
| `inpatient_beds`, `inpatient_beds_occ` | Inpatient beds available / occupied |
| `icu_beds`, `icu_beds_occ` | ICU beds available / occupied |
| `n_hosp_reporting` | Number of hospitals reporting that week |
| `reporting_frac` | `n_hosp_reporting` ÷ the state's median in Jan–Apr 2024 (mandatory period) |

RSV admissions/occupancy are essentially only available from October 2024 onward.

## Wastewater — CDC NWSS
**WVAL (Wastewater Viral Activity Level, CDC-standardised; available for all three pathogens)**
| Column | Meaning |
|---|---|
| `ww_<p>_wval_median` | Median site WVAL in the state that week |
| `ww_<p>_wval_popwt` | Population-served-weighted mean site WVAL |
| `ww_<p>_n_sites` | Number of sites contributing |
| `ww_<p>_pop_served` | Total population served by those sites |

`<p>` = `covid` (SARS-CoV-2), `flu` (Influenza A), `rsv`.

**Raw concentration (SARS-CoV-2 and RSV only)**
| Column | Meaning |
|---|---|
| `ww_<p>_log10conc_rel` | Population-weighted mean of each site's log10(copies/L + 1) minus that site's own median. 0 = typical level for the site; +1 ≈ 10× higher |
| `ww_<p>_raw_n_sites` | Number of sites contributing |

Only samples in copies/L wastewater are used; dry-sludge (copies/g) and log10-unit records were dropped because they cannot be mixed. Multiple samples per site-week are averaged.

## Quality flags
| Column | Meaning |
|---|---|
| `voluntary_period` | 1 for weeks 11 May – 2 Nov 2024, when hospital reporting was voluntary and ~60% of hospitals stopped reporting |
| `low_reporting` | 1 if `voluntary_period` = 1 or `reporting_frac` < 0.8 |
| `preliminary` | 1 for the last 2 weeks (CDC data still being revised) |
| `use_in_model` | 1 if `low_reporting` = 0 and `preliminary` = 0 → 11,455 rows recommended for fitting/evaluation |

## Notes for the SIR-IVP model
- State population (N) is **not** in these files; add U.S. Census Bureau state population estimates.
- Influenza wastewater is available only as WVAL (no raw-concentration file was downloaded).
- Missing wastewater weeks (≈6% for COVID WVAL, ≈27% for flu/RSV WVAL among `use_in_model` rows) are left blank — interpolate or mask them; do not fill with 0.
- Sanity check: log raw SARS-CoV-2 concentration vs. log COVID admissions correlate at r = 0.93 (CA), 0.71 (TX), 0.52 (NY).

## Sources
- NHSN HRD: https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data
- NWSS wastewater: https://www.cdc.gov/nwss/about-data.html (datasets on data.cdc.gov: SARS-CoV-2 `j9g8-acpt`, RSV `45cq-cw4i`, WVAL `atcp-73re`)
