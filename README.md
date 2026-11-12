# Immediate versus Midnight Screening for E-commerce Fulfillment Risk under Daily Capacity Limits

Reproduce the core results of manuscript v1.1.1 from fitted weights or refit its models on the original historical windows. The release contains the prepared training and evaluation records, reference tables and the order-level replay. All computations run locally on CPU.

## Run

Use Python 3.12 in a virtual environment. From the repository root, run:

```sh
python -m pip install -r requirements.txt
python -B reproduce.py --output rebuilt
python -B verify_results.py rebuilt
```

Use a new output directory for each run. Before loading weights, the script checks the released files against `SHA256.json`. The reference environment used Python 3.12.10 on CPU. The trained Python estimators are serialized with joblib; load model files only from a trusted release.

To refit all ten Olist deployments and both DataCo endpoints, run:

```sh
python -B train.py --output refitted
python -B reproduce.py --weights refitted --output rebuilt_refit
python -B verify_results.py rebuilt_refit
```

Both routes use the same historical admission thresholds. The training script checks Olist raw and Platt predictions for every evaluation order against the released fits, and verifies DataCo coefficients and lookup values. The comparison script checks the 14 CSVs in `reference/`, requiring exact integer counts. To regenerate the published Figure 5 from a verified result directory:

```sh
python -B plot_figure5.py rebuilt/Figure5.csv figures
```

| Rebuilt CSV | Paper result |
|---|---|
| Table1, Table2, Table3 | Cohort sizes and events, matched policy comparisons, and the effect of relabeling the same DataCo selections. |
| Table4, Figure5 | Three slots per day at an assumed eight-hour reporting lag; paired immediate-minus-midnight differences at all five assumed lags. |
| Figure3, Figure4 | Monthly and pooled Olist differences for three scorers; the orders removed, retained and added to the DataCo midnight list by service mode. |
| Fixed_budget, Fixed_pairs | Counts for three policies under both endpoints and all five reporting lags, with paired missing-outcome bounds. |
| Olist_frozen, Olist_selected | May 2018 fit applied to May–June, including raw and Platt performance and forecast counts on adjudicable selected deliveries. |
| S21 | Original DataCo flag by cancellation status and whether actual integer days exceed scheduled days, across 65,752 unique orders. |
| S22, S18_clock | South Asian retailer audit, including missing-time bounds and the hypothetical clock shift. |

Three checks against the manuscript: Olist unrestricted matched review gives 386 versus 409 confirmed events (difference −23, bounds −95 to +7). DataCo recorded-flag LR gives +43 unrestricted and −179 after refilling from pending orders. With three daily slots and an assumed eight-hour lag, the elapsed-time endpoint gives −31 across all selected orders and +124 before recorded dispatch.

## Weights and replay inputs

- **Olist.** Ten monthly bundles, September 2017 through June 2018. Each holds the fitted preprocessing pipeline, classifier and Platt map for logistic regression, CatBoost and LightGBM, plus the input column order. These bundles were reconstructed using the original S1 training and calibration windows, fixed settings and seed 417. After saving and reloading, raw and calibrated probabilities for each model were checked against the archived predictions by order. The May bundle also serves the frozen May–June analysis. The verification counts and maximum absolute differences are in `weights/olist_verification.csv`.
- **DataCo.** `weights/dataco_lr.json` contains the archived feature order, centering, scaling, coefficients, intercept and Platt coefficients for each endpoint. The lookup file stores fitted service-by-placement-hour rates. For a feature row x, the raw LR probability is expit(((x − center) / scale) · coef + intercept). The historical Platt map acts on logit(raw), clipped at probabilities 1e-12 and 1 − 1e-12. The replay checks recalculated LR probabilities against the two frozen score columns to absolute tolerance 2e-15, then uses the archived precision for threshold equality and tie ordering.
- **Policy inputs.** The test cohorts contain the predictors, decision clocks, outcomes, carrier/dispatch information and an integer rank preserving the original order-ID tie order. The thresholds in `data/thresholds.npz` come from the earlier calibration windows. Replaying these inputs reconstructs screening lists; the candidate files do not contain the published hit counts or policy lists.
- **Training windows.** The Olist history contains 65,591 linked orders approved from January 2017 through March 2018. For deployment month D, training uses approvals before D minus four months, and calibration uses approvals from D minus four months up to D minus two months. Both require an adjudicable delivery recorded before D. A month can supply evaluation cases for one deployment and historical observations for a later deployment. The DataCo history contains 50,406 orders. Each endpoint uses 36,434 fit orders with a recorded shipment before November 2016 and 11,404 calibration orders with a recorded shipment before June 2017.
- **Label and timestamp audits.** The DataCo label file contains one de-identified record per order and preserves the raw flag for canceled orders. Policy evaluation treats their outcomes as unresolved. The South Asian audit starts with 100,000 item lines. An anonymous grouping key links lines of the same order. The replay checks placement consistency, counts missing shipment times, and reconstructs first and last recorded dispatch for the original calendar day.

The starting point for Olist and DataCo is the prepared order record. Raw transaction joins, supplementary resampling analyses and historical model searches are outside this repository. The released thresholds are held fixed in both replay routes. Figures 3 and 4 are reproduced as numerical data and Figure 5 can also be rendered as SVG and PNG.

## Source data and attribution

- Olist, *Brazilian E-Commerce Public Dataset by Olist* (2018), [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). The Olist-derived training and test data are distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). It uses filtered order, product, seller and carrier data with derived predictors and integer tie ranks.
- F. Constante, F. Silva and A. Pereira, *DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS*, v5, [Mendeley Data](https://doi.org/10.17632/8gx2fvg2k6.5), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The projections retain historical and evaluation orders, the published flag, derived outcomes and the predictors used in the models.
- V. Bansal, *E-Commerce Dataset*, v1 (2020), [Mendeley Data](https://doi.org/10.17632/ggbkd8ck3x.1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The source credits Debjit Roy and Vishal Bansal. The projection retains only a replacement order grouping key and the placement and dispatch times for the timestamp audit.
