"""Score the held-out orders and rebuild the core numerical results (IJCIS v1.1.1)."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
MODELS = ("LR", "CatBoost", "LightGBM")


def contrast(a, b, y):
    unknown = np.isnan(y)
    ha, hb = int(np.nansum(y[a])), int(np.nansum(y[b]))
    delta = ha - hb
    return dict(used_A=int(a.sum()), used_B=int(b.sum()), hit_A=ha, hit_B=hb, delta=delta,
                id_lower=delta-int((b & ~a & unknown).sum()), id_upper=delta+int((a & ~b & unknown).sum()))


def entry_list(scores, rules, bins, blocks, allowed, cap, proportional=False):
    selected = np.zeros(len(scores), dtype=bool)
    for indices in blocks:
        used = 0
        for n, i in enumerate(indices):
            left = (n//20 + 1 if proportional else cap) - used
            if allowed[i] and left > 0 and scores[i] >= rules[bins[i], min(left, rules.shape[1]-1)]:
                selected[i] = True
                used += 1
    return selected


def midnight_list(scores, ties, blocks, eligible, entry, cap=None):
    selected = np.zeros(len(scores), dtype=bool)
    for ix in blocks:
        pool = ix[eligible[ix]]
        ranked = pool[np.lexsort((ties[pool], -scores[pool]))]
        quota = int(entry[ix].sum()) if cap is None else cap
        selected[ranked[:quota]] = True
    return selected


def clock(frame):
    t = frame.approved
    return (list(frame.groupby(t.dt.normalize()).indices.values()),
            ((t.dt.hour*60+t.dt.minute)//15).to_numpy(), frame.tie.to_numpy())


def olist_scores(state, features, f, name):
    x = state["preprocessor"].transform(f[features]).astype("float32")
    est = state["classifier"]
    raw = est.booster_.predict(x,num_threads=6) if name == "LightGBM" else est.predict_proba(x)[:,1]
    raw = np.asarray(raw, dtype=float)
    mapped = state["calibrator"].predict_proba(logit(np.clip(raw,1e-12,1-1e-12)).reshape(-1,1))[:,1]
    return raw, mapped


def olist(rules, output, weights):
    f = pd.read_parquet(ROOT/"data/olist.parquet")
    month = f.approved.dt.strftime("%Y-%m")
    y = f.component.map({"on_time":0, "delivered_late":1, "canceled":0, "unavailable":0}).to_numpy(dtype=float)
    pure = f.component.map({"on_time":0, "delivered_late":1})
    output["Table1"].append(dict(dataset="Olist",candidates=len(f),assessed=pure.count(),confirmed=int(pure.sum()),endpoint="late delivery"))
    cols = ["original_matched","pending_matched","pending_matched_to_eligible_online"]
    masks = {key:[np.zeros(len(f),dtype=bool) for _ in range(2)] for key in cols}
    by_model = {m:[np.zeros(len(f),dtype=bool) for _ in range(2)] for m in MODELS}
    for tag, ix in f.groupby(month).indices.items():
        orders = f.iloc[ix]
        daily, bins, ties = clock(orders)
        available = np.ones(len(ix),dtype=bool)
        state = joblib.load(weights/"olist"/(tag+".joblib"))
        for m in MODELS:
            raw, _ = olist_scores(state["models"][m], state["features"], orders, m)
            cut = rules[tag+"_"+m]
            a = entry_list(raw, cut, bins, daily, available, 11)
            b = midnight_list(raw,ties,daily,available,a)
            by_model[m][0][ix], by_model[m][1][ix] = a, b
            output["Figure3"].append(dict(model=m,month=tag,**contrast(a,b,y[ix])))
            if m == "LR":
                early = entry_list(raw,cut,bins,daily,~orders.seen_entry.to_numpy(),11)
                for key, arm, pool in zip(cols,(a,a,early),(available,~orders.seen_close.to_numpy(),~orders.seen_close.to_numpy())):
                    masks[key][0][ix]=arm
                    masks[key][1][ix]=midnight_list(raw,ties,daily,pool,arm)
    for m,(a,b) in by_model.items():
        output["Figure3"].append(dict(model=m,month="TOTAL",**contrast(a,b,y)))
    for policy,(a,b) in masks.items():
        output["Table2"].append(dict(dataset="Olist",label="late delivery",policy=policy,**contrast(a,b,y)))
    # May's frozen fit evaluated through June, separately from the monthly deployments.
    sub=f.loc[month.isin(["2018-05","2018-06"])]
    observed=pure.loc[sub.index].to_numpy(dtype=float)
    known=np.isfinite(observed)
    state=joblib.load(weights/"olist/2018-05.joblib")
    days,_,ties=clock(sub)
    for m in MODELS:
        raw,pl=olist_scores(state["models"][m],state["features"],sub,m)
        for mapping,score in [("raw",raw),("platt",pl)]:
            output["Olist_frozen"].append(dict(model=m,mapping=mapping,assessed=int(known.sum()),positives=int(np.nansum(observed)),
                AP=average_precision_score(observed[known],score[known]),AUC=roc_auc_score(observed[known],score[known]),
                Brier=brier_score_loss(observed[known],score[known])))
        if m=="LR":
            full=midnight_list(raw,ties,days,np.ones(len(sub),bool),None,cap=11)
            assessed=full&known
            output["Olist_selected"].append(dict(selected=int(full.sum()),assessed=int(assessed.sum()),late=int(np.nansum(observed[assessed])),
                unknown=int((full & sub.component.eq("unknown").to_numpy()).sum()),raw_sum=raw[assessed].sum(),platt_sum=pl[assessed].sum()))


def fixed_budget(t, y, scores, ties, blocks, entry, arrivals, shipped, output):
    names=("entry","rolling","midnight")
    for lag in (0,4,8,10,12):
        notice=shipped+lag*60
        ticks={name:np.full(len(y),-1,dtype=np.int64) for name in names}
        ticks["entry"][entry]=arrivals[entry]
        for ix in blocks:
            start=(arrivals[ix[0]]//1440)*1440
            for name,schedule in [("rolling",[(4,1),(8,1),(12,2),(16,2),(20,3),(24,3)]),("midnight",[(24,3)])]:
                used=0
                for hour,cap in schedule:
                    tick=start+60*hour
                    pool=ix[(arrivals[ix]<tick)&(notice[ix]>tick)&(ticks[name][ix]<0)]
                    chosen=pool[np.lexsort((ties[pool],-scores[pool]))[:cap-used]]
                    ticks[name][chosen]=tick
                    used+=len(chosen)
        scopes={}
        for name,at in ticks.items():
            selected=at>=0
            pending=selected & (shipped>at)
            scopes[name]={"all":selected,"pending":pending}
            hits=int(np.nansum(y[pending]))
            output["Fixed_budget"].append(dict(endpoint=t,lag_hours=lag,policy=name,offered=3*len(blocks),used=int(selected.sum()),
                full_hits=int(np.nansum(y[selected])),pending_hits=hits,pending_hits_per_100_offered=round(100*hits/(3*len(blocks)),1)))
        for a,b in combinations(names,2):
            for scope in ("all","pending"):
                output["Fixed_pairs"].append(dict(target=t,lag_h=lag,A=a,B=b,scope=scope,**contrast(scopes[a][scope],scopes[b][scope],y)))


def dataco(rules, output, weights):
    f=pd.read_parquet(ROOT/"data/dataco.parquet")
    pars=json.loads((weights/"dataco_lr.json").read_text())
    lookup=json.loads((weights/"dataco_lookup.json").read_text())
    blocks,bins,ties=clock(f)
    arrivals=f.approved.to_numpy(dtype="datetime64[m]").astype("int64")
    shipped=f.shipped.to_numpy(dtype="datetime64[m]").astype("int64")
    eligible=shipped>(arrivals//1440+1)*1440
    all_orders=np.ones(len(f),bool)
    rows={}
    for t in ("exact","recorded"):
        y=f[t].to_numpy(dtype=float)
        output["Table1"].append(dict(dataset="DataCo",candidates=len(f),assessed=int(np.isfinite(y).sum()),confirmed=int(np.nansum(y)),endpoint=t))
        par=pars[t]
        model_score=expit(((f[par["features"]].to_numpy()-par["center"])/par["scale"]) @ par["coef"]+par["intercept"])
        scores=f["raw_"+t].to_numpy()
        np.testing.assert_allclose(model_score,scores,rtol=0,atol=2e-15)
        cap_key=next(k for k in rules.files if k.startswith("dc_"+t+"_") and not k.endswith("_3"))
        for scorer in ("LR","Lookup"):
            if scorer=="Lookup" and t=="exact":continue
            pred=scores if scorer=="LR" else np.array([lookup[t][m+"/"+str(h)] for m,h in zip(f["mode"],f.hour)])
            key=cap_key if scorer=="LR" else "lookup_"+t
            a=entry_list(pred,rules[key],bins,blocks,all_orders,0,proportional=True)
            for name,pool in [("original_matched",all_orders),("pending_matched",eligible)]:
                b=midnight_list(pred,ties,blocks,pool,a)
                rows[t,scorer,name]=a,b
                if scorer=="LR":
                    output["Table2"].append(dict(dataset="DataCo",label=t,policy=name,**contrast(a,b,y)))
        a,b=rows[t,"LR","original_matched"]
        _,p=rows[t,"LR","pending_matched"]
        for change,group in [("Removed",b&~p),("Retained",b&p),("Added",p&~b)]:
            for mode in sorted(f.loc[group,"mode"].unique()):
                sel=group & f["mode"].eq(mode).to_numpy()
                output["Figure4"].append(dict(endpoint=t,change=change,mode=mode,orders=int(sel.sum()),hits=int(np.nansum(y[sel])),unknown=int(np.isnan(y[sel]).sum())))
        entry=entry_list(scores,rules["dc_"+t+"_3"],bins,blocks,all_orders,3)
        fixed_budget(t,y,scores,ties,blocks,entry,arrivals,shipped,output)
    for fit,score_as,model,b in [("exact","exact","LR","original_matched"),("exact","recorded","LR","original_matched"),
            ("recorded","recorded","LR","original_matched"),("recorded","recorded","LR","pending_matched"),
            ("recorded","recorded","Lookup","original_matched"),("recorded","recorded","Lookup","pending_matched")]:
        a,bmask=rows[fit,model,b]
        output["Table3"].append(dict(fit=fit,evaluated=score_as,model=model,policy=b,**contrast(a,bmask,f[score_as].to_numpy())))


def south_asia(output):
    lines = pd.read_parquet(ROOT / "data/south_asia_lines.parquet")
    f = lines.groupby("order_group").agg(lines=("placed","size"), observed=("ship","count"),
        placements=("placed","nunique"), placed=("placed","min"), first=("ship","min"), last=("ship","max"))
    midnight = f.placed.dt.normalize()
    ok = f.placements.eq(1) & midnight.le("2020-11-23") & ~f["first"].lt(f.placed)
    for column in ("placed","first","last"):
        f[column] = (f[column] - midnight).dt.total_seconds()
    complete=ok & f.observed.eq(f.lines)
    any_before=f["first"].lt(86400).fillna(False)
    all_before=f["last"].lt(86400).fillna(False)
    for name,yes,no in [("any_dispatch",ok&any_before,complete&~any_before),("all_recorded_lines_dispatched",complete&all_before,ok&f["last"].ge(86400).fillna(False))]:
        for scope,pop in [("complete_orders",complete),("eligible_including_missing",ok)]:
            n=int(pop.sum()); h=int((yes&pop).sum()); u=int((pop&~yes&~no).sum())
            output["S22"].append(dict(endpoint=name,scope=scope,denominator=n,confirmed_yes=h,unknown=u,lower_pct=100*h/n,upper_pct=100*(h+u)/n))
    boundary=((f.placed+19800)//86400+1)*86400-19800
    output["S18_clock"].append(dict(complete=int(complete.sum()),archived_first=int((complete & any_before).sum()),
        shifted_first=int((complete & f["first"].lt(boundary)).sum())))


def verify_release():
    manifest=json.loads((ROOT/"SHA256.json").read_text())
    for path,digest in manifest.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Checksum mismatch: {path}")


def main(dest, weights):
    verify_release()
    names=["Table1","Table2","Table3","Figure3","Figure4","Fixed_budget","Fixed_pairs","Olist_frozen","Olist_selected","S22","S18_clock"]
    tables={name:[] for name in names}
    with np.load(ROOT/"data/thresholds.npz") as rules, threadpool_limits(limits=1):
        olist(rules,tables,weights)
        dataco(rules,tables,weights)
    south_asia(tables)
    frames={k:pd.DataFrame(v) for k,v in tables.items()}
    labels = pd.read_parquet(ROOT / "data/dataco_label_audit.parquet")
    exceeds = labels.actual_days.gt(labels.scheduled_days).astype(int).rename("actual_exceeds_scheduled")
    cells = labels.groupby([labels.cancelled, exceeds, labels.source_flag]).size().unstack(fill_value=0)
    frames["S21"] = cells.rename(columns={0:"flag_0", 1:"flag_1"}).reset_index()
    frames["Table4"]=frames["Fixed_budget"].query("lag_hours==8")
    frames["Figure5"]=frames["Fixed_pairs"].query("A=='entry' and B=='midnight'")
    dest.mkdir(parents=True,exist_ok=False)
    for name,table in frames.items():
        table.to_csv(dest/(name+".csv"),index=False,float_format="%.12g")
    print(f"Rebuilt {len(frames)} numerical tables in {dest.resolve()}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("rebuilt"))
    parser.add_argument("--weights", type=Path, default=ROOT / "weights", help="Released weights or directory produced by train.py")
    args = parser.parse_args()
    main(args.output, args.weights)
