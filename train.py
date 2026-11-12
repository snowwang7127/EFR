"""Refit the manuscript models and calibrators on their historical windows."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from reproduce import ROOT, MODELS, olist_scores, verify_release
from training.estimators import CLOCK, STATIC, fit_model, lp, score


def platt(raw, y):
    return LogisticRegression(C=np.inf, tol=1e-9, max_iter=500).fit(lp(raw), y)


def train_olist(dest):
    frame = pd.read_parquet(ROOT / "training/olist_history.parquet")
    candidates = pd.read_parquet(ROOT / "data/olist.parquet")
    cols = CLOCK + STATIC
    for dep in pd.date_range("2017-09-01", "2018-06-01", freq="MS"):
        tag = dep.strftime("%Y-%m")
        start = dep - pd.DateOffset(months=4)
        end = dep - pd.DateOffset(months=2)
        known = frame.pure.notna() & frame.delivered.lt(dep)
        train = frame.loc[known & frame.approved.lt(start)].assign(y=lambda df: df.pure)
        cal = frame.loc[frame.approved.between(start, end, inclusive="left")]
        available = known.loc[cal.index].to_numpy()
        horizon = 2 if tag == "2018-05" else 1
        test = candidates.loc[candidates.approved.between(dep, dep + pd.DateOffset(months=horizon), inclusive="left")]
        frozen = joblib.load(ROOT / "weights/olist" / (tag + ".joblib"))
        states = {}
        for name in MODELS:
            prep, est = fit_model(name, cols, train)
            raw = score(prep, est, cal, cols)
            mapping = platt(raw[available], cal.loc[available, "pure"])
            states[name] = {"preprocessor": prep, "classifier": est, "calibrator": mapping}
            for fitted, archived in zip(olist_scores(states[name], cols, test, name),
                                        olist_scores(frozen["models"][name], cols, test, name)):
                np.testing.assert_allclose(fitted, archived, atol=1e-8, rtol=0)
        joblib.dump({"features": cols, "models": states}, dest / "olist" / (tag + ".joblib"), compress=3)
        print(f"{tag}: {len(train)} fit, {available.sum()} calibration, {len(test)} predictions checked per model", flush=True)


def train_dataco(dest):
    frame = pd.read_parquet(ROOT / "training/dataco_history.parquet")
    frozen = json.loads((ROOT / "weights/dataco_lr.json").read_text())
    recorded_cells = json.loads((ROOT / "weights/dataco_lookup.json").read_text())
    cols = ["hour", "weekday", "plan", "lines", "quantity", "item_value"]
    weights, lookup = {}, {}
    for endpoint in ("exact", "recorded"):
        known = frame[endpoint].notna()
        train = frame.loc[known & frame.split.eq("fit") & frame.shipped.lt("2016-11-01")]
        cal = frame.loc[known & frame.split.eq("cal") & frame.shipped.lt("2017-06-01")]
        assert train.approved.max() < pd.Timestamp("2016-11-01") and cal.approved.max() < pd.Timestamp("2017-06-01")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.25, max_iter=600, random_state=417))
        model.fit(train[cols], train[endpoint])
        mapping = platt(model.predict_proba(cal[cols])[:, 1], cal[endpoint])
        params = dict(features=cols, center=model[0].mean_.tolist(), scale=model[0].scale_.tolist(),
                      coef=model[1].coef_[0].tolist(), intercept=model[1].intercept_[0],
                      platt_slope=mapping.coef_[0, 0], platt_intercept=mapping.intercept_[0])
        for key in params.keys() - {"features"}:
            np.testing.assert_allclose(params[key], frozen[endpoint][key], atol=1e-9, rtol=0)
        weights[endpoint] = params
        cells = train.groupby(["mode", "hour"])[endpoint].mean()
        lookup[endpoint] = {f"{mode}/{hour}": rate for (mode, hour), rate in cells.items()}
        assert lookup[endpoint] == recorded_cells[endpoint]
        print(f"DataCo {endpoint}: {len(train)} fit, {len(cal)} calibration, {len(cells)} lookup cells verified", flush=True)
    for filename, data in [("dataco_lr.json", weights), ("dataco_lookup.json", lookup)]:
        (dest / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("refitted"))
    dest = parser.parse_args().output
    verify_release()
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "olist").mkdir()
    with threadpool_limits(limits=6):
        train_olist(dest)
        train_dataco(dest)
