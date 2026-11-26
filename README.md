# Code and data for: A Hybrid SIR-IVP Model With Machine Learning and Wastewater Data

**Paper:** Hossain, M. B., Uddin, M. S., & Miah, M. M. (2026). *A Hybrid SIR-IVP Model With Machine Learning and Wastewater Data: Forecasting COVID-19 and Influenza Hospital Admissions and Allocating Surge Capacity Across Ten U.S. States.* Manuscript submitted for publication.

**Authors:** Md Babul Hossain (corresponding author; Department of Mathematics and Statistics, Florida Atlantic University, Boca Raton, FL 33431, USA; mdhossain2024@fau.edu), Md. Sharif Uddin (Department of Mathematics, SUNY Broome Community College, Binghamton, NY 13905, USA; uddinm@sunybroome.edu) and Md. Musa Miah (Department of Mathematics, Mawlana Bhashani Science and Technology University, Tangail-1902, Bangladesh; mmusa@mbstu.ac.bd)

## Contents
| File | Description |
|---|---|
| `us_state_week_panel.csv` | Processed state-week panel (51 jurisdictions, Jan 2022 - Nov 2026): NHSN hospital admissions/occupancy and CDC NWSS wastewater indicators, with data-quality flags |
| `DATA_DICTIONARY.md` | Definition of every column in the panel |
| `build_panel.py` | Builds the panel from the four raw CDC downloads |
| `hybrid_sir_ivp_experiment.py` | Transmission-rate reconstruction, hybrid SIR-IVP forecasts and benchmarks (routine forecasting) |
| `outage_experiment.py` | Simulated reporting outages, the 2024 reporting gap, and decision-time forecasts for allocation |
| `early_warning.py` | Wave-onset early-warning analysis |
| `allocation.py` | Surge-capacity allocation linear program |
| `rsv_supplement.py` | Supplementary RSV analysis |
| `block_bootstrap.py` | Moving-block bootstrap confidence intervals |
| `probabilistic.py` | Calibrated interval forecasts, weighted interval score and coverage |
| `build_vintages.py` | Collects archived real-time data vintages of hospital admissions from the two CDC hubs |
| `vintages_10states.csv` | The resulting vintages for the ten states (COVID-19: 101, influenza: 104 as-of dates) |
| `realtime_vintage.py` | Real-time evaluation: the hybrid model run on admissions as first published |
| `hub_comparison.py` | Comparison with the CDC COVID-19 Forecast Hub and FluSight ensembles |
| `synthetic_validation.py` | Numerical checks of the theorems on synthetic data |
| `verify_error_propagation.py` | Check of Theorem 3 and Corollary 2 on every real test forecast |
| `calibration.py` | PIT histograms, WIS decomposition, relative WIS by state |
| `conditional_calibration.py` | Regime-conditional calibration (supplementary negative result) |
| `prob_allocation.py` | Allocation with predictive distributions (Theorem 5) versus point forecasts |
| `prob_allocation_paired.py` | Same, with COVID-19 and influenza errors drawn as dependent pairs (sensitivity analysis) |
| `matched_ablation.py` | Matched ablation: hybrid SIR-IVP vs direct admissions model with identical learner, origins and predictors |
| `sensitivity.py` | Sensitivity to the recovery rate, hyperparameters and predictor groups; permutation importance |
| `q1_figs.py` | Figures 2-6 and Supplementary Figures S1-S3 |
| `schematic/fig1_schematic.tex` | Figure 1 (TikZ; compile with pdflatex) |
| `make_figs.py` | Figures of an earlier version (not used in the paper; kept for completeness) |
| `results/` | All result files and figures reported in the paper |

## Raw data (public, CDC)
Raw files are not redistributed here; download them from data.cdc.gov into one folder:
- Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction, NHSN (`ua7e-t2fy`)
- CDC Wastewater Data for SARS-CoV-2 (`j9g8-acpt`, save as `export.csv`)
- CDC Wastewater Data for RSV (`45cq-cw4i`)
- CDC Wastewater Viral Activity Level for SARS-CoV-2, Influenza A and RSV (`atcp-73re`)

Data used in the paper were accessed on 20 November 2026. Later downloads may include revisions.

For the ensemble comparison, clone the two public CDC hubs (only `model-output/<ensemble>`, `model-output/<baseline>` and `target-data` are needed; `build_vintages.py` additionally needs the full git history of the COVID-19 hub, e.g. `git fetch --filter=blob:none --unshallow`), accessed 24 November 2026:
- https://github.com/CDCgov/covid19-forecast-hub (CovidHub-ensemble, CovidHub-baseline)
- https://github.com/cdcepi/FluSight-forecast-hub (FluSight-ensemble, FluSight-baseline)

## Reproducing the results
Requirements: Python 3.11+, numpy, pandas, scipy, scikit-learn, matplotlib (install with `pip install -r requirements.txt`; exact versions used are listed there). Run the steps in order, one at a time (parallel runs oversubscribe the CPU); total runtime about 2 hours (the sensitivity analyses take about 1 hour).

| Step | Command | Paper output |
|---|---|---|
| Panel (optional; panel is included) | `python build_panel.py <raw_data_folder> us_state_week_panel.csv` | Section 2 |
| Routine forecasts | `python hybrid_sir_ivp_experiment.py us_state_week_panel.csv results` | Table 1; Suppl. Table S2 |
| Outage tracking and 2024 gap | `python outage_experiment.py us_state_week_panel.csv results` | Fig. 5; Suppl. Table S6, Fig. S3 |
| Allocation forecasts (2-week cutoff) | `python outage_experiment.py us_state_week_panel.csv results alloc 0` | Table 4 |
| Allocation forecasts (+1-week publication lag) | `python outage_experiment.py us_state_week_panel.csv results alloc 1` | Table 4 sensitivity |
| Allocation (point forecasts) | `python allocation.py us_state_week_panel.csv results` | Table 4, Fig. 6b |
| Early warning | `python early_warning.py us_state_week_panel.csv results` | Suppl. Section S3, Table S3 |
| RSV | `python rsv_supplement.py us_state_week_panel.csv results` | Suppl. Section S4 |
| Bootstrap | `python block_bootstrap.py results` | Confidence intervals, Section 5 |
| Probabilistic forecasts | `python probabilistic.py results` | Table 2 |
| Data vintages (optional; file is included) | `python build_vintages.py <covid19-forecast-hub> <FluSight-forecast-hub> vintages_10states.csv` | Section 3.6 |
| Real-time forecasts | `python realtime_vintage.py us_state_week_panel.csv results vintages_10states.csv` | Table 3 (real-time rows) |
| CDC ensemble comparison | `python hub_comparison.py us_state_week_panel.csv results <covid19-forecast-hub> <FluSight-forecast-hub>` | Table 3 |
| Calibration diagnostics | `python calibration.py results` | Fig. 4; Suppl. Table S4 |
| Regime-conditional calibration | `python conditional_calibration.py results` | Suppl. Section S5 |
| Allocation with predictive distributions | `python prob_allocation.py us_state_week_panel.csv results` | Section 5.7, Fig. 6a |
| Allocation with dependent (paired) errors | `python prob_allocation_paired.py us_state_week_panel.csv results` | Section 5.7; Suppl. Section S10, Table S8 |
| Matched ablation (hybrid vs direct, same inputs) | `python matched_ablation.py us_state_week_panel.csv results` (about 4 minutes) | Section 5.2; Suppl. Section S10, Table S7 |
| Error propagation check | `python verify_error_propagation.py us_state_week_panel.csv results` | Section 5.5 |
| Sensitivity analyses | `python sensitivity.py us_state_week_panel.csv results {gamma,hyper,ablation,importance} {covid,flu}` (8 runs) | Section 5.5; Suppl. Section S6, Table S5, Fig. S2 |
| Theory checks | `python synthetic_validation.py results us_state_week_panel.csv` | Section 4.7; Suppl. Section S1, Table S1 |
| Figures | `python q1_figs.py results figs`; `pdflatex schematic/fig1_schematic.tex` | Figs. 1-6 (files fig1_schematic, fig2_reproduction_heatmap, fig3_accuracy, fig4_calibration, fig6_outage = Fig. 5, fig8_allocation = Fig. 6); Suppl. Figs. S1 (fig5_probabilistic), S2 (figS2_sensitivity), S3 (fig7_gap2024) |

Random seeds are fixed. A clean rerun reproduced every number in the allocation table (Table 4) exactly; individual forecasts can differ in the last digits between machines because gradient-boosting sums are multithreaded.

## License
Code: MIT License (see `LICENSE`). Processed data and results: CC BY 4.0. Raw data are U.S. government works from the CDC.
