# -*- coding: utf-8 -*-
"""
20A_RUN_TABULAR_BASELINES_REPAIRED_V2.py
========================================
Formal repaired-V2 rerun for BASE-01..BASE-04 only.

DATA:
<PROJECT_ROOT>\\data\\07_FINAL_REPAIRED_DATASET_V2

FORMAL SOURCE:
<PROJECT_ROOT>\\modeling\\BASE\\BASE-00_contract

Protocols preserved:
- seeds = [7,11,21]
- test=outer; validation=(outer+1)%5; train=remaining 3 folds
- preprocessing fitted on training folds only
- no outer-test selection
- no post-selection refit
- role training weights: positive=1.0, control=0.5
- pair ties are wrong (strict positive > control)
- bootstrap cluster = pair_set_id, 2000 reps, seed=20260706

BASE-01:
  exact executed protocol: 10 last-step rainfall fields and train
  quantiles 0.05..0.95.

BASE-02:
  exact executed protocol: LogisticRegression elasticnet/SAGA,
  C=[.01,.1,1,10], l1_ratio=[.1,.5,.9], max_iter=10000.

BASE-03:
  official XGBoostBaseline + frozen 20 configs.

BASE-04:
  official LightGBMBaseline + frozen 20 configs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "07_FINAL_REPAIRED_DATASET_V2"
OUT = PROJECT / "experiments" / "FORMAL_BASELINES_REPAIRED_V2_V1"

SAMPLE = DATA / "04_FINAL_SAMPLE_INDEX.parquet"
PAIR = DATA / "03_REPAIRED_PAIR_SETS.parquet"
STATIC = DATA / "06_STATIC_92_RAW_BY_SAMPLE.parquet"
DYNAMIC = DATA / "07_DYNAMIC_70x10_FLOAT32.npy"
DYN_TIME = DATA / "07A_DYNAMIC_70_TIMESTAMP_REGISTRY.csv"
DYN_ORDER = DATA / "07B_DYNAMIC_10_FIELD_ORDER.csv"
PAIR_TENSOR = DATA / "08_PAIR_TENSOR_INDEX_INT64.npy"
PAIR_REGISTRY = DATA / "08A_PAIR_TENSOR_REGISTRY.parquet"
CLOSED = DATA / "DATA_SECTION_CLOSED.flag"
FINAL_DECISION = DATA / "99_FINAL_DATASET_DECISION.json"

OLD_D2 = PROJECT / "data" / "05_final_dataset_assembly" / "02_model_ready_inputs"
STATIC_SCHEMA = OLD_D2 / "dataset_02_static_feature_schema.json"
DYNAMIC_SCHEMA = OLD_D2 / "dataset_02_dynamic_feature_schema.json"

BASE00 = PROJECT / "modeling" / "BASE" / "BASE-00_contract"
if str(BASE00) not in sys.path:
    sys.path.insert(0, str(BASE00))

from baseline_tuning import candidate_configs
from baseline_xgboost import XGBoostBaseline
from baseline_lightgbm import LightGBMBaseline

SEEDS = [7, 11, 21]
RAIN_FIELDS = [
    "rain_30m_mm","rain_1h_mm","rain_3h_mm","rain_6h_mm","rain_12h_mm",
    "rain_24h_mm","rain_48h_mm","rain_72h_mm","rain_120h_mm",
    "api_k090_step30m_120h",
]
REF = pd.Timestamp("2018-07-06T11:00:00Z")
N = 15168
NP = 5056
NC = 10112
BOOT_SEED = 20260706
BASE01_Q = [float(round(x,2)) for x in np.arange(.05,1.0,.05)]
BASE02_CS = [.01,.1,1.,10.]
BASE02_L1 = [.1,.5,.9]

FORBIDDEN = [
    "centroid_x","centroid_y","centroid_lon","centroid_lat","label","y_pair",
    "pair_set_id","pair_edge_id","outer_fold","spatial_group","sample_role",
    "control_rank","matching","inventory","gsi_only","future",
]


def log(x):
    OUT.mkdir(parents=True, exist_ok=True)
    line=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {x}"
    print(line,flush=True)
    with (OUT/"20A_RUN.log").open("a",encoding="utf-8") as f: f.write(line+"\n")


def jread(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def jwrite(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")


def onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore",sparse_output=False,dtype=np.float64)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore",sparse=False,dtype=np.float64)


class Bundle:
    pass


def preflight():
    req=[
        CLOSED,FINAL_DECISION,SAMPLE,PAIR,STATIC,DYNAMIC,DYN_TIME,DYN_ORDER,
        PAIR_TENSOR,PAIR_REGISTRY,STATIC_SCHEMA,DYNAMIC_SCHEMA,
        BASE00/"baseline_xgboost.py",BASE00/"baseline_lightgbm.py",
        BASE00/"baseline_tuning.py",
    ]
    miss=[str(p) for p in req if not p.exists()]
    if miss: raise RuntimeError("MISSING_ASSETS:\n"+"\n".join(miss))
    if "PASS" not in CLOSED.read_text(encoding="utf-8-sig"):
        raise RuntimeError("DATA_SECTION_CLOSED_NOT_PASS")
    if jread(FINAL_DECISION).get("status")!="PASS":
        raise RuntimeError("FINAL_DATASET_DECISION_NOT_PASS")

    b=Bundle()
    b.sample=pd.read_parquet(SAMPLE).sort_values("sample_index",kind="stable").reset_index(drop=True)
    if len(b.sample)!=N or not np.array_equal(b.sample.sample_index.to_numpy(np.int64),np.arange(N)):
        raise RuntimeError("SAMPLE_INDEX_CONTRACT")
    if b.sample.pair_set_id.nunique()!=NP or int(b.sample.y_pair.sum())!=NP or int((b.sample.y_pair==0).sum())!=NC:
        raise RuntimeError("SAMPLE_COUNTS")
    if not b.sample.groupby("pair_set_id").size().eq(3).all() or not b.sample.groupby("pair_set_id").y_pair.sum().eq(1).all():
        raise RuntimeError("PAIR_1_TO_2_CONTRACT")
    if b.sample.groupby("spatial_group_id").outer_fold_id.nunique().gt(1).any():
        raise RuntimeError("SPATIAL_GROUP_CROSS_FOLD")

    b.pair=pd.read_parquet(PAIR)
    if len(b.pair)!=NP: raise RuntimeError("PAIR_ROWS")

    b.pt=np.asarray(np.load(PAIR_TENSOR,mmap_mode="r"),dtype=np.int64)
    b.pr=pd.read_parquet(PAIR_REGISTRY)
    if b.pt.shape!=(NP,3) or len(b.pr)!=NP:
        raise RuntimeError("PAIR_TENSOR_SHAPE")
    rt=b.pr[["positive_sample_index","control1_sample_index","control2_sample_index"]].to_numpy(np.int64)
    if not np.array_equal(rt,b.pt): raise RuntimeError("PAIR_REGISTRY_MISMATCH")

    roles=b.sample.sample_role.astype(str).to_numpy()
    if not np.all(roles[b.pt[:,0]]=="MATCHED_POSITIVE") or not np.all(roles[b.pt[:,1:]]=="HARD_CONTROL"):
        raise RuntimeError("PAIR_POSITION_SEMANTICS")
    folds=b.sample.outer_fold_id.to_numpy(np.int64)
    pf=folds[b.pt]
    if not np.all(pf==pf[:,:1]): raise RuntimeError("PAIR_CROSS_FOLD")
    b.pair_fold=pf[:,0]
    if set(np.unique(folds))!=set(range(5)): raise RuntimeError("FOLD_IDS")

    b.static=pd.read_parquet(STATIC).sort_values("sample_index",kind="stable").reset_index(drop=True)
    if len(b.static)!=N or not np.array_equal(b.static.sample_index.to_numpy(np.int64),np.arange(N)):
        raise RuntimeError("STATIC_ALIGNMENT")

    b.dynamic=np.load(DYNAMIC,mmap_mode="r")
    if b.dynamic.shape!=(N,70,10): raise RuntimeError(f"DYNAMIC_SHAPE={b.dynamic.shape}")
    if not np.isfinite(np.asarray(b.dynamic)).all() or (np.asarray(b.dynamic)<0).any():
        raise RuntimeError("DYNAMIC_VALUES")

    times=pd.to_datetime(pd.read_csv(DYN_TIME).timestamp_utc,utc=True)
    if len(times)!=70 or times.iloc[-1]!=REF: raise RuntimeError("REFERENCE_TIME")
    order=pd.read_csv(DYN_ORDER).sort_values("dynamic_index").field_name.astype(str).tolist()
    if order!=RAIN_FIELDS: raise RuntimeError("DYNAMIC_ORDER")

    b.ss=jread(STATIC_SCHEMA); b.ds=jread(DYNAMIC_SCHEMA)
    allc=list(b.ss["continuous_fields_in_order"])
    cats=list(b.ss["categorical_fields_in_order"])
    if len(allc)!=91 or len(cats)!=1: raise RuntimeError("STATIC_SCHEMA_91_PLUS_1")
    if "geology_metamorphic_rock_fraction" not in allc:
        raise RuntimeError("FORBIDDEN_GEOLOGY_FIELD_EXPECTED_IN_RAW_SCHEMA")
    b.cont=[f for f in allc if f!="geology_metamorphic_rock_fraction"]
    if len(b.cont)!=90: raise RuntimeError("MODEL_RAW_CONTINUOUS_NOT_90")
    b.cat=cats[0]
    b.soil=[f for f in allc if f.startswith("soil_")]
    if len(b.soil)!=18: raise RuntimeError(f"SOIL_FIELDS={len(b.soil)}")
    needed=set(allc+[b.cat])
    if not needed.issubset(b.static.columns):
        raise RuntimeError("FINAL_STATIC_MISSING_FIELDS:"+",".join(sorted(needed-set(b.static.columns))))
    b.splan=dict(b.ss.get("transform_plan_by_field",{}))
    if list(b.ds["feature_fields_in_order"])!=RAIN_FIELDS: raise RuntimeError("DYNAMIC_SCHEMA_ORDER")
    b.dlog={f:bool(b.ds["log1p_for_model_branch"][f]) for f in RAIN_FIELDS}

    names=b.cont+["soil_missing_indicator"]+RAIN_FIELDS+[b.cat]
    bad=[n for n in names if any(t in n.lower() for t in FORBIDDEN)]
    if bad: raise RuntimeError("FORBIDDEN_X:"+",".join(bad))

    cfg=candidate_configs()
    if len(cfg["BASE-03"])!=20 or len(cfg["BASE-04"])!=20:
        raise RuntimeError("TREE_CANDIDATE_BUDGET_CHANGED")

    payload={
        "status":"PASS","dataset":str(DATA),"samples":N,"pairs":NP,"controls":NC,
        "spatial_groups":int(b.sample.spatial_group_id.nunique()),
        "pair_cross_fold":0,"spatial_group_cross_fold":0,
        "static_raw_fields":92,"model_raw_continuous":90,"soil_indicator":1,
        "categorical_field":b.cat,"dynamic_shape":[N,70,10],
        "dynamic_fields":RAIN_FIELDS,"reference_time_utc":str(REF),
        "future_steps_read":0,"seeds":SEEDS,
        "split_rule":"test=outer; validation=(outer+1)%5; train=remaining_three",
        "BASE03_candidates":20,"BASE04_candidates":20,
    }
    jwrite(OUT/"00_PREFLIGHT.json",payload)
    return b


def srows(b,folds):
    return np.flatnonzero(np.isin(b.sample.outer_fold_id.to_numpy(np.int64),list(folds)))


def prows(b,folds):
    return np.flatnonzero(np.isin(b.pair_fold,list(folds)))


def pair_tables(b,rows,score,mid,seed,outer):
    t=b.pt[rows]; s3=np.asarray(score[t],float)
    if not np.isfinite(s3).all(): raise RuntimeError("NONFINITE_PAIR_SCORE")
    gaps=s3[:,:1]-s3[:,1:]
    ids=b.pr.iloc[rows].pair_set_id.to_numpy()
    units=b.sample.unit_id.astype(str).to_numpy()
    p=pd.DataFrame({
        "experiment_id":mid,"seed":seed,"outer_fold":outer,"pair_set_id":ids,
        "positive_unit_id":units[t[:,0]],"control_unit_id_1":units[t[:,1]],
        "control_unit_id_2":units[t[:,2]],"positive_score":s3[:,0],
        "control_1_score":s3[:,1],"control_2_score":s3[:,2],
        "gap_1":gaps[:,0],"gap_2":gaps[:,1],
    })
    p["min_gap"]=p[["gap_1","gap_2"]].min(axis=1)
    p["mean_gap"]=p[["gap_1","gap_2"]].mean(axis=1)
    p["pair_concordant"]=((p.gap_1>0)&(p.gap_2>0)).astype(np.int8)
    e1=p[["experiment_id","seed","outer_fold","pair_set_id","positive_unit_id","control_unit_id_1","positive_score","control_1_score","gap_1"]].rename(columns={"control_unit_id_1":"control_unit_id","control_1_score":"control_score","gap_1":"risk_gap"})
    e1["control_rank"]=1
    e2=p[["experiment_id","seed","outer_fold","pair_set_id","positive_unit_id","control_unit_id_2","positive_score","control_2_score","gap_2"]].rename(columns={"control_unit_id_2":"control_unit_id","control_2_score":"control_score","gap_2":"risk_gap"})
    e2["control_rank"]=2
    e=pd.concat([e1,e2],ignore_index=True)
    e["edge_correct"]=(e.risk_gap>0).astype(np.int8)
    e["pair_edge_id"]=e.pair_set_id.astype(str)+"::C"+e.control_rank.astype(str)
    return p,e


def rmetrics(p,e):
    return {
        "pair_set_concordance":float(p.pair_concordant.mean()),
        "StrictPair":float(p.pair_concordant.mean()),
        "edge_wise_ranking_accuracy":float(e.edge_correct.mean()),
        "Edge":float(e.edge_correct.mean()),
        "gap_mean":float(e.risk_gap.mean()),"gap_median":float(e.risk_gap.median()),
        "gap_q1":float(e.risk_gap.quantile(.25)),"gap_q3":float(e.risk_gap.quantile(.75)),
        "positive_risk_gap_fraction":float((e.risk_gap>0).mean()),
        "min_gap_mean":float(p.min_gap.mean()),"min_gap_median":float(p.min_gap.median()),
    }


def cmetrics(y,score,prob,pred=None):
    y=np.asarray(y,np.int8); score=np.asarray(score,float); prob=np.asarray(prob,float)
    if pred is None: pred=(prob>=.5).astype(np.int8)
    pred=np.asarray(pred,np.int8)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
    return {
        "AUROC":float(roc_auc_score(y,score)),
        "AUPRC":float(average_precision_score(y,score)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "F1":float(f1_score(y,pred,zero_division=0)),
        "Brier_score":float(brier_score_loss(y,prob)),
        "sensitivity":float(sens),"specificity":float(spec),
        "Youden_J":float(sens+spec-1),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
    }


def metrics(p,e,y,score,prob,pred=None):
    return {**rmetrics(p,e),**cmetrics(y,score,prob,pred)}


def sample_frame(b,idx,mid,seed,outer,score,prob,pred,extra=None):
    s=b.sample.iloc[idx][["sample_index","unit_id","pair_set_id","sample_role","control_rank","y_pair","outer_fold_id","spatial_group_id"]].copy()
    s.insert(0,"experiment_id",mid); s["seed"]=seed; s["outer_fold"]=outer
    s["y_true"]=s.pop("y_pair").astype(np.int8)
    s["risk_score"]=np.asarray(score,float); s["probability"]=np.asarray(prob,float)
    s["predicted_label"]=np.asarray(pred,np.int8); s["is_oof"]=True
    if extra:
        for k,v in extra.items(): s[k]=v
    return s


def robust(x):
    z=np.asarray(x,float); z=z[np.isfinite(z)]
    if len(z)==0: return 0.,1.,True
    med=float(np.median(z)); q1,q3=np.quantile(z,[.25,.75]); iqr=float(q3-q1)
    zero=not np.isfinite(iqr) or iqr<=0
    return med,(1. if zero else iqr),bool(zero)


def lg(x,yes):
    z=np.asarray(x,float)
    if yes:
        finite=np.isfinite(z)
        if (z[finite] < -1).any(): raise RuntimeError("LOG1P_VALUE_LT_MINUS1")
        z=np.log1p(z)
    return z


def soil_indicator(b,idx):
    x=b.static.iloc[idx][b.soil].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    c=(~np.isfinite(x)).sum(1)
    if np.any((c!=0)&(c!=18)): raise RuntimeError("UNEXPECTED_PARTIAL_SOIL_MISSINGNESS")
    return (c==18).astype(float)


def fitprep(b,train_idx,scale):
    sp=[]
    for f in b.cont:
        x=pd.to_numeric(b.static[f],errors="coerce").to_numpy(float)[train_idx]
        use=b.splan.get(f)=="LOG1P_AT_MODEL_STAGE"; x=lg(x,use); med,iqr,z=robust(x)
        sp.append((f,med,iqr,use,z))
    dp=[]
    for j,f in enumerate(RAIN_FIELDS):
        x=np.asarray(b.dynamic[train_idx,-1,j],float); use=b.dlog[f]; x=lg(x,use); med,iqr,z=robust(x)
        dp.append((f,med,iqr,use,z))
    enc=onehot()
    cat=b.static[b.cat].fillna("MISSING").astype(str)
    enc.fit(cat.iloc[train_idx].to_frame(name=b.cat))
    return {"static":sp,"dynamic":dp,"encoder":enc,"scale":scale,"cat":b.cat}


def transform(b,idx,pre):
    cols=[]; names=[]
    for f,med,iqr,use,_ in pre["static"]:
        x=pd.to_numeric(b.static[f],errors="coerce").to_numpy(float)[idx]; x=lg(x,use)
        x=np.where(np.isfinite(x),x,med)
        if pre["scale"]: x=(x-med)/iqr
        cols.append(x); names.append(f)
    cols.append(soil_indicator(b,idx)); names.append("soil_missing_indicator")
    for j,(f,med,iqr,use,_) in enumerate(pre["dynamic"]):
        x=np.asarray(b.dynamic[idx,-1,j],float); x=lg(x,use); x=np.where(np.isfinite(x),x,med)
        if pre["scale"]: x=(x-med)/iqr
        cols.append(x); names.append(f)
    cat=b.static[pre["cat"]].fillna("MISSING").astype(str).iloc[idx].to_frame(name=pre["cat"])
    oh=pre["encoder"].transform(cat)
    try: ohn=pre["encoder"].get_feature_names_out([pre["cat"]]).tolist()
    except Exception: ohn=[f"{pre['cat']}_OH_{i}" for i in range(oh.shape[1])]
    X=np.column_stack(cols+[oh]).astype(np.float64,copy=False); names += ohn
    if X.shape[1]!=len(names) or not np.isfinite(X).all() or len(names)!=len(set(names)):
        raise RuntimeError("TABULAR_TRANSFORM_INTEGRITY")
    bad=[n for n in names if any(t in n.lower() for t in FORBIDDEN)]
    if bad: raise RuntimeError("FORBIDDEN_X:"+",".join(bad))
    return X,names


def wmiddle(v,w):
    o=np.argsort(v,kind="stable"); v=np.asarray(v)[o]; w=np.asarray(w,dtype=np.int64)[o]
    c=np.cumsum(w); n=int(c[-1])
    def kth(k): return float(v[np.searchsorted(c,k,side="left")])
    return kth((n+1)//2) if n%2 else (kth(n//2)+kth(n//2+1))/2


def waucap(s3,c):
    s=s3.ravel(); y=np.tile([1,0,0],len(s3)); w=np.repeat(c,3)
    lev,inv=np.unique(s,return_inverse=True)
    pos=np.bincount(inv[y==1],weights=w[y==1],minlength=len(lev))
    neg=np.bincount(inv[y==0],weights=w[y==0],minlength=len(lev))
    auc=float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/(pos.sum()*neg.sum()))
    tp=np.cumsum(pos[::-1]); fp=np.cumsum(neg[::-1])
    ap=float(np.sum((pos[::-1]/pos.sum())*(tp/(tp+fp))))
    return auc,ap


def bootstrap(p,reps):
    if reps<=0: return pd.DataFrame()
    p=p.sort_values("pair_set_id",kind="stable").reset_index(drop=True)
    s3=p[["positive_score","control_1_score","control_2_score"]].to_numpy(float)
    g=p[["gap_1","gap_2"]].to_numpy(float); mg=p.min_gap.to_numpy(float); con=p.pair_concordant.to_numpy(float)
    n=len(p); rng=np.random.default_rng(BOOT_SEED)
    names=["pair_set_concordance","edge_wise_ranking_accuracy","gap_mean","gap_median","min_gap_mean","min_gap_median","AUROC","AUPRC"]
    vals={k:np.empty(reps) for k in names}
    for i in range(reps):
        c=rng.multinomial(n,np.full(n,1/n))
        vals["pair_set_concordance"][i]=np.average(con,weights=c)
        vals["edge_wise_ranking_accuracy"][i]=np.sum(c[:,None]*(g>0))/(2*n)
        vals["gap_mean"][i]=np.sum(c[:,None]*g)/(2*n)
        vals["gap_median"][i]=wmiddle(g.ravel(),np.repeat(c,2))
        vals["min_gap_mean"][i]=np.average(mg,weights=c)
        vals["min_gap_median"][i]=wmiddle(mg,c)
        vals["AUROC"][i],vals["AUPRC"][i]=waucap(s3,c)
    rows=[]
    for k,v in vals.items():
        lo,hi=np.quantile(v,[.025,.975])
        rows.append({"metric":k,"ci_lower_95":float(lo),"ci_upper_95":float(hi),"bootstrap_replicates":reps,"bootstrap_seed":BOOT_SEED,"cluster_unit":"pair_set_id"})
    return pd.DataFrame(rows)


def finalize(mid,sfs,pfs,efs,foldrows,candrows,selrows,prerows,reps,extra=None):
    d=OUT/mid.replace("-",""); d.mkdir(parents=True,exist_ok=True)
    s=pd.concat(sfs,ignore_index=True); p=pd.concat(pfs,ignore_index=True); e=pd.concat(efs,ignore_index=True)
    for seed in SEEDS:
        if len(s[s.seed==seed])!=N or len(p[p.seed==seed])!=NP or len(e[e.seed==seed])!=NC:
            raise RuntimeError(f"{mid}_OOF_COUNT_SEED_{seed}")
    if s.duplicated(["seed","sample_index"]).any() or p.duplicated(["seed","pair_set_id"]).any() or e.duplicated(["seed","pair_set_id","control_rank"]).any():
        raise RuntimeError(mid+"_DUPLICATE_OOF")
    s.to_parquet(d/"SAMPLE_OOF.parquet",index=False); p.to_parquet(d/"PAIR_OOF.parquet",index=False); e.to_parquet(d/"EDGE_OOF.parquet",index=False)
    pd.DataFrame(foldrows).to_csv(d/"FOLD_SEED_METRICS.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(candrows).to_csv(d/"VALIDATION_CANDIDATES.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(selrows).to_csv(d/"SELECTED_CONFIGS.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(prerows).to_csv(d/"PREPROCESSING_AUDIT.csv",index=False,encoding="utf-8-sig")
    seeds=[]; boots=[]
    for seed in SEEDS:
        ss=s[s.seed==seed]; pp=p[p.seed==seed]; ee=e[e.seed==seed]
        m=metrics(pp,ee,ss.y_true,ss.risk_score,ss.probability,ss.predicted_label)
        seeds.append({"experiment_id":mid,"seed":seed,**m})
        bb=bootstrap(pp,reps)
        if not bb.empty:
            bb.insert(0,"seed",seed); bb.insert(0,"experiment_id",mid); boots.append(bb)
    sd=pd.DataFrame(seeds); sd.to_csv(d/"SEED_METRICS.csv",index=False,encoding="utf-8-sig")
    (pd.concat(boots,ignore_index=True) if boots else pd.DataFrame()).to_csv(d/"BOOTSTRAP_CI.csv",index=False,encoding="utf-8-sig")
    core=["StrictPair","Edge","pair_set_concordance","edge_wise_ranking_accuracy","min_gap_mean","min_gap_median","AUROC","AUPRC","balanced_accuracy","F1","Brier_score"]
    summary={"status":"PASS","experiment_id":mid,"seeds":SEEDS,"seed_mean":{k:float(sd[k].mean()) for k in core},"seed_std":{k:float(sd[k].std(ddof=1)) for k in core},"oof_counts_per_seed":{"samples":N,"pairs":NP,"edges":NC},"outer_test_used_for_selection":False,"refit_after_selection":False,"future_steps_used":0,"bootstrap_replicates":reps}
    if extra: summary.update(extra)
    jwrite(d/"MODEL_SUMMARY.json",summary); (d/"COMPLETED.flag").write_text("PASS\n",encoding="utf-8")
    log(f"{mid} COMPLETE StrictPair={summary['seed_mean']['StrictPair']:.6f} Edge={summary['seed_mean']['Edge']:.6f} AUPRC={summary['seed_mean']['AUPRC']:.6f} AUROC={summary['seed_mean']['AUROC']:.6f}")
    return summary


def unique_thresholds(x):
    g={}
    for q in BASE01_Q: g.setdefault(float(np.quantile(x,q)),[]).append(q)
    return [(t,"|".join(f"{q:.2f}" for q in qs)) for t,qs in sorted(g.items())]


def run01(b,args):
    mid="BASE-01"; d=OUT/"BASE01"
    if (d/"COMPLETED.flag").exists() and not args.force: return jread(d/"MODEL_SUMMARY.json")
    X=np.asarray(b.dynamic[:,-1,:],float)
    sb=[];pb=[];eb=[];fb=[];cand=[];sel=[];prep=[]
    y=b.sample.y_pair.to_numpy(np.int8)
    for outer in range(5):
        val=(outer+1)%5; train=sorted(set(range(5))-{outer,val})
        ti=srows(b,train); vi=srows(b,[val]); xi=srows(b,[outer]); vpr=prows(b,[val]); xpr=prows(b,[outer])
        records=[]; cache={}
        for k,f in enumerate(RAIN_FIELDS):
            tr=X[ti,k]; med=float(np.median(tr)); q1,q3=np.quantile(tr,[.25,.75]); iqr=float(q3-q1)
            if not np.isfinite(iqr) or iqr<=0: continue
            sc=(X[:,k]-med)/iqr; full=np.asarray(sc,float)
            pp,ee=pair_tables(b,vpr,full,mid,7,outer); rm=rmetrics(pp,ee)
            cache[f]={"k":k,"med":med,"iqr":iqr,"score":sc,"rm":rm}
            records.append((rm["pair_set_concordance"],rm["edge_wise_ranking_accuracy"],rm["min_gap_median"],rm["min_gap_mean"],-k,f))
        if not records: raise RuntimeError("BASE01_ALL_FIELDS_INVALID")
        sf=max(records)[-1]; c=cache[sf]; k=c["k"]
        th=[]
        for t,qs in unique_thresholds(X[ti,k]):
            pr=(X[vi,k]>=t).astype(np.int8); cm=cmetrics(y[vi],c["score"][vi],pr.astype(float),pr)
            th.append({"threshold":t,"sources":qs,**cm})
        chosen=max(th,key=lambda r:(r["balanced_accuracy"],r["F1"],r["Youden_J"],r["threshold"]))
        for f,fc in cache.items():
            fk=fc["k"]
            for t,qs in unique_thresholds(X[ti,fk]):
                pr=(X[vi,fk]>=t).astype(np.int8); cm=cmetrics(y[vi],fc["score"][vi],pr.astype(float),pr)
                cand.append({"experiment_id":mid,"outer_fold":outer,"seed":7,"field":f,"field_index":fk,"threshold":t,"quantile_sources":qs,"validation_pair_set_concordance":fc["rm"]["pair_set_concordance"],"validation_edge_wise_ranking_accuracy":fc["rm"]["edge_wise_ranking_accuracy"],"validation_min_gap_median":fc["rm"]["min_gap_median"],"validation_min_gap_mean":fc["rm"]["min_gap_mean"],"validation_AUPRC":cm["AUPRC"],"validation_balanced_accuracy":cm["balanced_accuracy"],"validation_F1":cm["F1"],"validation_Youden_J":cm["Youden_J"],"outer_test_used_for_selection":"NO"})
        t=chosen["threshold"]; score=c["score"]; pred=(X[xi,k]>=t).astype(np.int8); prob=pred.astype(float)
        full=np.full(N,np.nan); full[xi]=score[xi]; pp,ee=pair_tables(b,xpr,full,mid,7,outer)
        ss=sample_frame(b,xi,mid,7,outer,score[xi],prob,pred,{"selected_rainfall_field":sf,"selected_threshold":t,"raw_rainfall_value":X[xi,k]})
        mm=metrics(pp,ee,ss.y_true,ss.risk_score,ss.probability,ss.predicted_label)
        sb.append(ss);pb.append(pp);eb.append(ee);fb.append({"experiment_id":mid,"outer_fold":outer,"seed":7,"selected_rainfall_field":sf,"selected_threshold":t,**mm})
        sel.append({"experiment_id":mid,"outer_fold":outer,"seed":7,"training_folds":"|".join(map(str,train)),"validation_fold":val,"test_fold":outer,"selected_rainfall_field":sf,"selected_threshold":t,"validation_pair_set_concordance":c["rm"]["pair_set_concordance"],"validation_edge_wise_ranking_accuracy":c["rm"]["edge_wise_ranking_accuracy"],"validation_min_gap_median":c["rm"]["min_gap_median"],"validation_min_gap_mean":c["rm"]["min_gap_mean"],"validation_balanced_accuracy":chosen["balanced_accuracy"],"refit_performed":"NO","outer_test_used_for_selection":"NO"})
        prep.append({"experiment_id":mid,"outer_fold":outer,"fit_scope":"TRAINING_FOLDS_ONLY","static_features_used":0,"dynamic_fields_considered":10,"dynamic_values_read_per_sample":1,"future_steps_used":0,"training_samples":len(ti),"validation_samples":len(vi),"test_samples":len(xi)})
        log(f"BASE-01 fold={outer} field={sf} threshold={t:.6g} pair={mm['StrictPair']:.4f} edge={mm['Edge']:.4f}")
    # deterministic seed replication
    sfs=[];pfs=[];efs=[];fr=[];sr=[];cr=[]
    for seed in SEEDS:
        for x in sb: z=x.copy();z["seed"]=seed;sfs.append(z)
        for x in pb: z=x.copy();z["seed"]=seed;pfs.append(z)
        for x in eb: z=x.copy();z["seed"]=seed;efs.append(z)
        for x in fb: z=dict(x);z["seed"]=seed;fr.append(z)
        for x in sel: z=dict(x);z["seed"]=seed;sr.append(z)
        for x in cand: z=dict(x);z["seed"]=seed;cr.append(z)
    return finalize(mid,sfs,pfs,efs,fr,cr,sr,prep,args.bootstrap,{"baseline_type":"DETERMINISTIC_RAINFALL_RULE","model_trained":False,"quantiles":BASE01_Q,"deterministic_across_seeds":True})


def vkey(m,ci):
    return (m["pair_set_concordance"],m["edge_wise_ranking_accuracy"],m["min_gap_median"],m["AUPRC"],-ci)


def b2key(r):
    m=r["m"]; cfg=r["cfg"]
    return (m["pair_set_concordance"],m["edge_wise_ranking_accuracy"],m["min_gap_median"],m["min_gap_mean"],m["AUPRC"],-cfg["C"],cfg["l1_ratio"])


def fitcand(mid,cfg,ci,seed,Xtr,ytr,wtr,Xv,yv,vi,vpr,b):
    t0=time.perf_counter(); warn=""
    if mid=="BASE-02":
        model=LogisticRegression(penalty="elasticnet",solver="saga",C=cfg["C"],l1_ratio=cfg["l1_ratio"],fit_intercept=True,max_iter=10000,tol=1e-4,warm_start=False,n_jobs=1,random_state=seed,class_weight=None)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always",ConvergenceWarning); model.fit(Xtr,ytr,sample_weight=wtr)
        warn="|".join(str(w.message) for w in ws if issubclass(w.category,ConvergenceWarning))
        conv=(warn=="" and int(model.n_iter_[0])<10000); score=model.decision_function(Xv); prob=model.predict_proba(Xv)[:,1]
    elif mid=="BASE-03":
        model=XGBoostBaseline(**cfg).initialize(seed); model.model.fit(Xtr,ytr,sample_weight=wtr); score=model.risk_score(Xv); prob=np.asarray(score,float); conv=True
    else:
        model=LightGBMBaseline(**cfg).initialize(seed); model.model.fit(Xtr,ytr,sample_weight=wtr); score=model.risk_score(Xv); prob=np.asarray(score,float); conv=True
    score=np.asarray(score,float);prob=np.asarray(prob,float);pred=(prob>=.5).astype(np.int8)
    full=np.full(N,np.nan);full[vi]=score;pp,ee=pair_tables(b,vpr,full,mid,seed,-1)
    mm=metrics(pp,ee,yv,score,prob,pred)
    return {"model":model,"cfg":cfg,"ci":ci,"m":mm,"converged":conv,"warning":warn,"seconds":time.perf_counter()-t0}


def run_tab(b,args,mid):
    d=OUT/mid.replace("-","")
    if (d/"COMPLETED.flag").exists() and not args.force: return jread(d/"MODEL_SUMMARY.json")
    if mid=="BASE-02":
        configs=[{"C":C,"l1_ratio":l1,"max_iter":10000} for C in BASE02_CS for l1 in BASE02_L1]; scale=True
    else:
        configs=candidate_configs()[mid];scale=False
    sfs=[];pfs=[];efs=[];fr=[];cr=[];sr=[];prep=[]
    y=b.sample.y_pair.to_numpy(np.int8); roles=b.sample.sample_role.astype(str).to_numpy()
    for outer in range(5):
        val=(outer+1)%5; train=sorted(set(range(5))-{outer,val})
        ti=srows(b,train);vi=srows(b,[val]);xi=srows(b,[outer]);vpr=prows(b,[val]);xpr=prows(b,[outer])
        pr=fitprep(b,ti,scale);Xtr,names=transform(b,ti,pr);Xv,nv=transform(b,vi,pr);Xt,nt=transform(b,xi,pr)
        if names!=nv or names!=nt: raise RuntimeError(mid+"_FEATURE_ORDER_CHANGED")
        ytr=y[ti];yv=y[vi];yt=y[xi];wtr=np.where(roles[ti]=="MATCHED_POSITIVE",1.,.5)
        prep.append({"experiment_id":mid,"outer_fold":outer,"fit_scope":"TRAINING_FOLDS_ONLY","fit_samples":len(ti),"validation_fit_rows":0,"test_fit_rows":0,"static_raw_fields":90,"soil_missing_indicator":1,"dynamic_fields":10,"categorical_raw_fields":1,"one_hot_columns":Xtr.shape[1]-101,"final_X_dimension":Xtr.shape[1],"scaling":scale,"future_steps_used":0,"feature_order_sha256":hashlib.sha256("\n".join(names).encode()).hexdigest()})
        for seed in SEEDS:
            log(f"{mid} outer={outer} seed={seed} fitting {len(configs)} candidates")
            indexed=list(enumerate(configs,start=1))
            def one(item):
                ci,cfg=item;return fitcand(mid,cfg,ci,seed,Xtr,ytr,wtr,Xv,yv,vi,vpr,b)
            fits=Parallel(n_jobs=args.jobs,prefer="threads")(delayed(one)(it) for it in indexed) if args.jobs>1 else [one(it) for it in indexed]
            for r in fits:
                cr.append({"experiment_id":mid,"outer_fold":outer,"seed":seed,"candidate_index":r["ci"],"configuration":json.dumps(r["cfg"],sort_keys=True),"converged":r["converged"],"convergence_warning":r["warning"],"fit_seconds":r["seconds"],"outer_test_used_for_selection":"NO",**{"validation_"+k:v for k,v in r["m"].items()}})
            ok=[r for r in fits if r["converged"]]
            if not ok: raise RuntimeError(mid+"_ALL_CANDIDATES_FAILED")
            best=max(ok,key=b2key) if mid=="BASE-02" else max(ok,key=lambda r:vkey(r["m"],r["ci"]))
            model=best["model"]
            if mid=="BASE-02": score=model.decision_function(Xt);prob=model.predict_proba(Xt)[:,1]
            else: score=model.risk_score(Xt);prob=np.asarray(score,float)
            score=np.asarray(score,float);prob=np.asarray(prob,float);pred=(prob>=.5).astype(np.int8)
            full=np.full(N,np.nan);full[xi]=score;pp,ee=pair_tables(b,xpr,full,mid,seed,outer)
            ss=sample_frame(b,xi,mid,seed,outer,score,prob,pred,{"candidate_index":best["ci"],"selected_configuration":json.dumps(best["cfg"],sort_keys=True)})
            mm=metrics(pp,ee,yt,score,prob,pred)
            sfs.append(ss);pfs.append(pp);efs.append(ee);fr.append({"experiment_id":mid,"outer_fold":outer,"seed":seed,"candidate_index":best["ci"],"selected_configuration":json.dumps(best["cfg"],sort_keys=True),**mm})
            sr.append({"experiment_id":mid,"outer_fold":outer,"seed":seed,"training_folds":"|".join(map(str,train)),"validation_fold":val,"test_fold":outer,"candidate_index":best["ci"],"selected_configuration":json.dumps(best["cfg"],sort_keys=True),"validation_pair_set_concordance":best["m"]["pair_set_concordance"],"validation_edge_wise_ranking_accuracy":best["m"]["edge_wise_ranking_accuracy"],"validation_min_gap_median":best["m"]["min_gap_median"],"validation_AUPRC":best["m"]["AUPRC"],"refit_performed":"NO","outer_test_used_for_selection":"NO"})
            log(f"{mid} fold={outer} seed={seed} C{best['ci']:02d} pair={mm['StrictPair']:.4f} edge={mm['Edge']:.4f} AUPRC={mm['AUPRC']:.4f}")
    return finalize(mid,sfs,pfs,efs,fr,cr,sr,prep,args.bootstrap,{"candidate_count_per_fold_seed":len(configs),"continuous_scaling":scale,"role_weights":{"MATCHED_POSITIVE":1.0,"HARD_CONTROL":0.5},"reference_time_utc":str(REF)})


def update_compare():
    rows=[]
    for mid in ["BASE-01","BASE-02","BASE-03","BASE-04"]:
        p=OUT/mid.replace("-","")/"MODEL_SUMMARY.json"
        if not p.exists(): continue
        s=jread(p)
        if s.get("status")!="PASS": continue
        m=s["seed_mean"];sd=s["seed_std"]
        rows.append({"experiment_id":mid,"StrictPair_mean":m["StrictPair"],"StrictPair_std":sd["StrictPair"],"Edge_mean":m["Edge"],"Edge_std":sd["Edge"],"AUPRC_mean":m["AUPRC"],"AUPRC_std":sd["AUPRC"],"AUROC_mean":m["AUROC"],"AUROC_std":sd["AUROC"],"balanced_accuracy_mean":m["balanced_accuracy"]})
    df=pd.DataFrame(rows)
    if not df.empty: df=df.sort_values(["StrictPair_mean","Edge_mean","AUPRC_mean"],ascending=False)
    df.to_csv(OUT/"FORMAL_BASELINE_COMPARISON.csv",index=False,encoding="utf-8-sig")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models",default="BASE-01,BASE-02,BASE-03,BASE-04")
    ap.add_argument("--jobs",type=int,default=4)
    ap.add_argument("--bootstrap",type=int,default=2000)
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--preflight-only",action="store_true")
    args=ap.parse_args()
    if args.jobs<1 or args.bootstrap<0: raise ValueError("invalid jobs/bootstrap")
    models=[x.strip().upper() for x in args.models.split(",") if x.strip()]
    bad=[x for x in models if x not in {"BASE-01","BASE-02","BASE-03","BASE-04"}]
    if bad: raise ValueError("20A supports BASE-01..04 only: "+",".join(bad))
    OUT.mkdir(parents=True,exist_ok=True);log("20A formal repaired-V2 tabular baseline run started")
    b=preflight();log("PREFLIGHT PASS")
    if args.preflight_only:
        jwrite(OUT/"20A_FINAL_STATUS.json",{"status":"PASS_PREFLIGHT_ONLY"});print("FINAL_STATUS=PASS_PREFLIGHT_ONLY");return 0
    try:
        summaries={}
        for mid in models:
            summaries[mid]=run01(b,args) if mid=="BASE-01" else run_tab(b,args,mid)
            update_compare()
        status={"status":"PASS_TABULAR_BASELINES_COMPLETED","requested":models,"completed":[m for m in models if (OUT/m.replace("-","")/"COMPLETED.flag").exists()],"comparison":str(OUT/"FORMAL_BASELINE_COMPARISON.csv")}
        jwrite(OUT/"20A_FINAL_STATUS.json",status)
        print("\n"+"="*110);print("20A FORMAL TABULAR BASELINES COMPLETE");print("="*110)
        comp=pd.read_csv(OUT/"FORMAL_BASELINE_COMPARISON.csv")
        if not comp.empty: print(comp[["experiment_id","StrictPair_mean","Edge_mean","AUPRC_mean","AUROC_mean"]].to_string(index=False))
        print("\nFINAL_STATUS:",status["status"]);print("\nPlease send:");print(OUT/"FORMAL_BASELINE_COMPARISON.csv");print(OUT/"20A_FINAL_STATUS.json")
        for mid in models:
            print(OUT/mid.replace("-","")/"MODEL_SUMMARY.json")
            print(OUT/mid.replace("-","")/"FOLD_SEED_METRICS.csv")
            print(OUT/mid.replace("-","")/"SELECTED_CONFIGS.csv")
        return 0
    except Exception as e:
        err={"status":"FAIL","error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc(),"completed":[m for m in ["BASE-01","BASE-02","BASE-03","BASE-04"] if (OUT/m.replace("-","")/"COMPLETED.flag").exists()]}
        jwrite(OUT/"20A_FINAL_STATUS.json",err);traceback.print_exc();return 1


if __name__=="__main__":
    raise SystemExit(main())
