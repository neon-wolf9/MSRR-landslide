#!/usr/bin/env python
"""Experiment 128B: preregistered conservative pre-trigger sensitivity.

The script must stop before scientific model fitting if the frozen source does
not contain the complete history required by the unchanged 120 h variables.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.update({"MKL_THREADING_LAYER":"SEQUENTIAL","MKL_NUM_THREADS":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/MSRR_PRETRIGGER_TEMPORAL_SENSITIVITY_V1"
OUT.mkdir(parents=True,exist_ok=True)
PREREG=OUT/"128B_PREREGISTRATION.json"
FIELDS=["rain_30m_mm","rain_1h_mm","rain_3h_mm","rain_6h_mm","rain_12h_mm","rain_24h_mm","rain_48h_mm","rain_72h_mm","rain_120h_mm","api_k090_step30m_120h"]
WINDOWS={"rain_1h_mm":2,"rain_3h_mm":6,"rain_6h_mm":12,"rain_12h_mm":24,"rain_24h_mm":48,"rain_48h_mm":96,"rain_72h_mm":144,"rain_120h_mm":240}
H_BASE=ROOT/"data/04_dynamic_rainfall_features/05_rain_02b_base_30m/07_frozen/rain_02b_base_30m_frozen.parquet_dataset"
H_FULL=ROOT/"data/04_dynamic_rainfall_features/07_rain_04_final_audit_freeze/07_frozen/dynamic_rainfall_features_frozen.parquet_dataset"
H_SAMPLE=ROOT/"data/07_FINAL_REPAIRED_DATASET_V2/04_FINAL_SAMPLE_INDEX.parquet"
H_TENSOR=ROOT/"data/07_FINAL_REPAIRED_DATASET_V2/07_DYNAMIC_70x10_FLOAT32.npy"
K_BASE=ROOT/"external/kyushu_2017_asakura_toho/06_imerg/02_subset/kyushu_imerg_250m_432slot_rain_v1.parquet"
K_FULL=ROOT/"external/kyushu_2017_asakura_toho/99_frozen_dataset/08_matched_dynamic_192/kyushu_external_matched_dynamic_10f_192slots_v1.parquet"
K_VIEW=ROOT/"external/kyushu_2017_asakura_toho/99_frozen_dataset/09_matched_dynamic_70_view/kyushu_external_matched_dynamic_10f_70slot_view_v1.parquet"
H_DATAFIX=OUT/"128B_DATAFIX/hiroshima_missing_rain_30m.parquet"
K_DATAFIX=OUT/"128B_DATAFIX/kyushu_missing_rain_30m.parquet"
DICT127=ROOT/"experiments/MATCHING_PROTOCOL_REPRODUCIBILITY_V1/MATCHING_VARIABLE_DICTIONARY.csv"
RAIN_SCRIPT=ROOT/"data/04_dynamic_rainfall_features/06_rain_03_dynamic_features/scripts/rain_03_dynamic_features.py"
MISSING_H=ROOT/"data/04_dynamic_rainfall_features/00_rain_00_asset_inventory/rain_00_missing_time_slots.csv"

def jwrite(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def csvwrite(name,rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows); d.to_csv(OUT/name,index=False,encoding="utf-8-sig"); return d
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(x): return pd.Timestamp(x).tz_convert("UTC").isoformat().replace("+00:00","Z")
def jst(x): return pd.Timestamp(x).tz_convert("Asia/Tokyo").isoformat()

def validate_prereg():
    if not PREREG.is_file(): raise RuntimeError("FAIL_128B_PREREGISTRATION_MISSING")
    p=json.loads(PREREG.read_text(encoding="utf-8"))
    required={"hiroshima_cutoff_utc":"2018-07-06T09:30:00Z","kyushu_cutoff_utc":"2017-07-05T07:00:00Z","window_length":70,"post_result_rescue_allowed":False,"scientific_results_seen_before_cutoff_selection":False}
    for k,v in required.items():
        if p.get(k)!=v: raise RuntimeError(f"FAIL_128B_PREREG_CHANGED:{k}")
    return p,sha(PREREG)

def dependency_audit():
    d=pd.read_csv(DICT127)
    vars4=["rain_24h_mm","rain_72h_mm","rain_120h_mm","api_k090_step30m_120h"]
    rows=[]
    anchors={"Hiroshima 2018":("2018-07-06T11:00:00Z","2018-07-06T09:30:00Z"),"Kyushu 2017":("2017-07-05T11:00:00Z","2017-07-05T07:00:00Z")}
    defs={"rain_24h_mm":"sum interval ends in (t-24h,t]","rain_72h_mm":"sum interval ends in (t-72h,t]","rain_120h_mm":"sum interval ends in (t-120h,t]","api_k090_step30m_120h":"sum(j=0..239,0.90^j*rain_30m_mm[t-j])"}
    for event,(old,new) in anchors.items():
        for v in vars4:
            z=d[(d.event==event)&(d.variable_name==v)]
            if len(z)!=1: raise RuntimeError(f"FAIL_128B_127_RAIN_FIELD:{event}:{v}")
            rows.append({"event":event,"variable":v,"definition":defs[v],"original_anchor":old,"pretrigger_anchor":new,"depends_on_shifted_interval":True,"requires_recomputation":True,"source":str(DICT127)})
    return csvwrite("MATCHING_RAINFALL_TEMPORAL_DEPENDENCY.csv",rows)

def make_timeline(event,end_utc,name):
    end=pd.Timestamp(end_utc); start=end-pd.Timedelta(minutes=69*30)
    t=pd.date_range(start,end,freq="30min",tz="UTC")
    if len(t)!=70 or not np.all(np.diff(t.asi8)==30*60*10**9): raise RuntimeError("FAIL_128B_TIMELINE")
    csvwrite(name,[{"event":event,"model_step":i,"timestamp_utc":utc(x),"timestamp_jst":jst(x)} for i,x in enumerate(t)])
    return t

def read_rain(path,units,time_col,interval_start):
    dataset=ds.dataset(str(path),format="parquet",partitioning="hive" if path.is_dir() else None)
    tab=dataset.to_table(columns=["unit_id",time_col,"rain_30m_mm"],filter=ds.field("unit_id").isin([str(x) for x in units])).to_pandas()
    supplemental=H_DATAFIX if Path(path)==H_BASE else (K_DATAFIX if Path(path)==K_BASE else None)
    if supplemental is not None and supplemental.is_file():
        extra=pd.read_parquet(supplemental,filters=[("unit_id","in",[str(x) for x in units])],columns=["unit_id",time_col,"rain_30m_mm"])
        tab=pd.concat([tab,extra],ignore_index=True)
    tab["unit_id"]=tab.unit_id.astype(str); tab[time_col]=pd.to_datetime(tab[time_col],utc=True)
    tab=tab.drop_duplicates(["unit_id",time_col],keep="first")
    if interval_start: tab["feature_time"]=tab[time_col]+pd.Timedelta(minutes=30)
    else: tab["feature_time"]=tab[time_col]
    tab=tab.sort_values(["unit_id","feature_time"],kind="mergesort")
    counts=tab.groupby("unit_id").size().reindex(units)
    if counts.isna().any() or counts.nunique()!=1: raise RuntimeError(f"FAIL_128B_BASE_COUNTS:{counts.to_dict()}")
    if tab.duplicated(["unit_id","feature_time"]).any(): raise RuntimeError("FAIL_128B_BASE_TIME_DUPLICATE")
    wide=tab.pivot(index="unit_id",columns="feature_time",values="rain_30m_mm").reindex(units)
    if wide.isna().any().any(): raise RuntimeError("FAIL_128B_BASE_TIME_GRID")
    return wide.to_numpy(np.float64),pd.DatetimeIndex(wide.columns)

def regenerate(rain,times,target):
    n,T=rain.shape; ring=np.empty((240,n),np.float64); roll={k:np.zeros(n,np.float64) for k in WINDOWS}; api=np.zeros(n,np.float64); out={}
    wanted={int(np.where(times.asi8==x.value)[0][0]):i for i,x in enumerate(target) if np.any(times.asi8==x.value)}
    if len(wanted)!=len(target): raise RuntimeError("FAIL_128B_TARGET_NOT_IN_BASE")
    result=np.empty((n,len(target),10),np.float64)
    for idx in range(T):
        r=rain[:,idx]; outgoing={s:ring[(idx-s)%240].copy() for s in set(WINDOWS.values())|{240} if idx>=s}
        for k,s in WINDOWS.items():
            roll[k]+=r
            if idx>=s: roll[k]-=outgoing[s]
        api=r+0.90*api
        if idx>=240: api-=(0.90**240)*outgoing[240]
        ring[idx%240]=r
        for k,s in WINDOWS.items():
            neg=np.flatnonzero(roll[k]<0)
            if len(neg):
                src=np.arange(idx-s+1,idx+1)%240; roll[k][neg]=np.sum(ring[np.ix_(src,neg)],axis=0,dtype=np.float64)
        if idx in wanted:
            j=wanted[idx]; vals={"rain_30m_mm":r,**{k:v.copy() for k,v in roll.items()},"api_k090_step30m_120h":api.copy()}
            result[:,j,:]=np.stack([vals[f] for f in FIELDS],axis=1)
    return result

def original_replay():
    rows=[]
    hs=pd.read_parquet(H_SAMPLE).sort_values("sample_index"); hp=[0,len(hs)//2,len(hs)-1]; hu=hs.iloc[hp].unit_id.astype(str).tolist()
    hr,ht=read_rain(H_BASE,hu,"timestamp_utc",False); hot=pd.date_range("2018-07-05T00:30:00Z","2018-07-06T11:00:00Z",freq="30min",tz="UTC"); hregen=regenerate(hr,ht,hot)
    hds=ds.dataset(str(H_FULL),format="parquet",partitioning="hive"); tensor=np.load(H_TENSOR,mmap_mode="r")
    for i,(u,si) in enumerate(zip(hu,[int(hs.iloc[x].sample_index) for x in hp])):
        f=(ds.field("unit_id")==u)&(ds.field("timestamp_utc")>=hot[0].to_pydatetime())&(ds.field("timestamp_utc")<=hot[-1].to_pydatetime())
        frozen=hds.to_table(columns=["timestamp_utc"]+FIELDS,filter=f).to_pandas().sort_values("timestamp_utc")[FIELDS].to_numpy(np.float64)
        diff=np.abs(hregen[i]-frozen); f32=hregen[i].astype(np.float32); exact=bool(np.array_equal(f32,np.asarray(tensor[si],np.float32)))
        rows.append({"event":"Hiroshima 2018","sample":u,"comparison":"regenerated original window vs frozen full table and formal tensor","max_abs_diff":float(diff.max()),"mean_abs_diff":float(diff.mean()),"different_count_float32":int(np.count_nonzero(f32!=np.asarray(tensor[si],np.float32))),"exact_equal_float32":exact,"strict_tolerance_pass":bool(diff.max()<=1e-9)})
    units=sorted(pd.read_parquet(K_VIEW,columns=["unit_id"]).unit_id.unique()); ku=[units[0],units[len(units)//2],units[-1]]
    kr,kt=read_rain(K_BASE,ku,"time_utc",True); kot=pd.date_range("2017-07-04T00:30:00Z","2017-07-05T11:00:00Z",freq="30min",tz="UTC"); kregen=regenerate(kr,kt,kot)
    kv=ds.dataset(str(K_VIEW),format="parquet")
    for i,u in enumerate(ku):
        frozen=kv.to_table(columns=["time_index"]+FIELDS,filter=ds.field("unit_id")==u).to_pandas().sort_values("time_index")[FIELDS].to_numpy(np.float64)
        diff=np.abs(kregen[i]-frozen); exact=bool(np.array_equal(kregen[i].astype(np.float32),frozen.astype(np.float32)))
        rows.append({"event":"Kyushu 2017","sample":u,"comparison":"regenerated original window vs frozen formal 70-view","max_abs_diff":float(diff.max()),"mean_abs_diff":float(diff.mean()),"different_count_float32":int(np.count_nonzero(kregen[i].astype(np.float32)!=frozen.astype(np.float32))),"exact_equal_float32":exact,"strict_tolerance_pass":bool(diff.max()<=1e-5)})
    d=csvwrite("ORIGINAL_WINDOW_REGENERATION_AUDIT.csv",rows)
    if not d.exact_equal_float32.all(): raise RuntimeError("FAIL_128B_RAINFALL_PIPELINE_REPLAY")
    return d

def history_audit(ht,kt,hnew,knew):
    rows=[]; missing_rows=[]
    for event,available,target in [("Hiroshima 2018",ht,hnew),("Kyushu 2017",kt,knew)]:
        required_first=target[0]-pd.Timedelta(minutes=239*30)
        missing=pd.date_range(required_first,target[0],freq="30min",tz="UTC").difference(available)
        for x in missing:
            missing_rows.append({"event":event,"product":"NASA GPM IMERG 3IMERGHH V07B Final Run","source_interval_start_utc":utc(x-pd.Timedelta(minutes=30)),"source_interval_end_utc":utc(x),"purpose":"complete unchanged 240-slot history for first preregistered model timestamp"})
        rows.append({"event":event,"required_first_interval_end_utc":utc(required_first),"available_first_interval_end_utc":utc(available[0]),"target_first_utc":utc(target[0]),"target_last_utc":utc(target[-1]),"required_history_slots":240,"available_history_slots_at_first_target":int(((target[0]-available[0])/pd.Timedelta(minutes=30))+1),"missing_history_slots":len(missing),"first_missing_utc":utc(missing[0]) if len(missing) else "","last_missing_utc":utc(missing[-1]) if len(missing) else "","all_10_fields_finite_under_original_policy":len(missing)==0})
    ok=all(r["missing_history_slots"]==0 for r in rows)
    csvwrite("MISSING_PRETRIGGER_SOURCE_SLOTS.csv",missing_rows)
    jwrite("PRETRIGGER_DYNAMIC_AUDIT.json",{"status":"PASS" if ok else "FAIL_INSUFFICIENT_120H_WARMUP","window_length":70,"spacing_minutes":30,"same_imerg_version":"V07B Final Run","definitions_changed":False,"past_only":True,"events":rows})
    return rows,ok

def write_failure(prereg_hash,replay,history):
    status="FAIL_128B_INSUFFICIENT_120H_WARMUP"
    audit={"experiment":"128B","status":status,"cutoffs_frozen_before_results":True,"preregistration_sha256_unchanged":prereg_hash==sha(PREREG),"original_pipeline_replay":"PASS_EXACT_FLOAT32","matching_rainfall_requires_recomputation":True,"arm_b_status":"REQUIRED_BUT_NOT_RUN_BECAUSE_PRETRIGGER_DYNAMIC_INPUT_IS_NOT_CONSTRUCTIBLE","scientific_models_fit":0,"post_result_rescue":False,"history_audit":history,"reason":"The unchanged rain_120h_mm and finite 240-step API require complete history at every model timestamp. Existing project sources begin too late for the preregistered first timestamps. Filling, shortening, or changing the cutoff is prohibited."}
    jwrite("AUDIT128B.json",audit)
    jwrite("GATE128B_DECISION.json",{"GATE128B_DECISION":status,"PRIMARY_SCIENTIFIC_GATE":"NOT_EVALUATED","NEXT_STEP":"ACQUIRE_AND_FREEZE_MISSING_EARLIER_V07B_SLOTS_THEN_RERUN_SAME_PREREGISTRATION","POST_RESULT_RESCUE":False})
    (OUT/"GATE128B_REPORT.md").write_text("# Experiment 128B gate report\n\n`FAIL_128B_INSUFFICIENT_120H_WARMUP`\n\nThe original-window reconstruction passed exact float32 replay. Phase A found that all four formal rainfall matching variables require recomputation, so Arm B is required. However, the preregistered pre-trigger windows cannot be constructed under the unchanged 120 h/API definition from existing project sources: Hiroshima lacks 2 leading half-hour intervals and Kyushu lacks 7. No imputation, zero initialization, shortened history, cutoff adjustment, model fitting, or result-based rescue was performed.\n",encoding="utf-8")
    (OUT/"REVIEWER2_COMMENT5_128B_RESPONSE_DRAFT.md").write_text("# Reviewer #2 Comment 5 — Experiment 128B status draft\n\nWe reconstructed the original exact windows and confirmed that all rainfall features are past-only. Individual failure times remain unavailable, and the original event-level windows overlap the documented disaster periods. We preregistered conservative 70-step cutoffs of 2018-07-06 18:30 JST for Hiroshima and 2017-07-05 16:00 JST for Kyushu. Before model fitting, pipeline replay reproduced the original frozen inputs exactly. The archived source coverage was nevertheless insufficient to construct the earliest preregistered timestamps under the unchanged 120 h accumulation and 240-step API definitions (2 missing leading slots for Hiroshima; 7 for Kyushu). We therefore stopped without scientific results or protocol alteration. The same-version earlier V07B slots must be acquired and frozen before this preregistered sensitivity can be completed. Grid-specific post-failure exclusion still cannot be proven retrospectively.\n",encoding="utf-8")
    return status

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def transform_dynamic(raw,pre,runner):
    out=np.empty(raw.shape,np.float32)
    for j,p in enumerate(pre["dynamic"]):
        x=runner.maybe_log1p(raw[:,:,j],p["log1p"]); x=np.where(np.isfinite(x),x,p["median"])
        out[:,:,j]=((x-p["median"])/p["iqr"]).astype(np.float32)
    if not np.isfinite(out).all(): raise RuntimeError("FAIL_128B_DYNAMIC_TRANSFORM")
    return out

def event_raw_tensor(path,units,time_col,interval_start,target):
    rain,times=read_rain(path,units,time_col,interval_start)
    result=regenerate(rain,times,target)
    if result.shape!=(len(units),70,10) or not np.isfinite(result).all(): raise RuntimeError("FAIL_128B_PRETRIGGER_TENSOR")
    return result.astype(np.float32)

def metrics(score,y,pairs):
    s=np.asarray(score,np.float64); y=np.asarray(y,np.int8); p=1/(1+np.exp(-np.clip(s,-60,60)))
    s3=s[pairs]; edge=s3[:,:1]>s3[:,1:]
    return {"AUROC":float(roc_auc_score(y,p)),"AUPRC":float(average_precision_score(y,p)),"StrictPair":float(edge.all(1).mean()),"Edge":float(edge.mean())}

def paired_bootstrap(raw,msrr,y,pairs,name,B=10000,seed=20260721):
    obsr=metrics(raw,y,pairs); obsm=metrics(msrr,y,pairs); r3=raw[pairs]; m3=msrr[pairs]; n=len(pairs)
    rng=np.random.default_rng(seed); store={k:np.empty(B) for k in ["AUROC","AUPRC","StrictPair","Edge"]}
    yy=np.tile(np.array([1,0,0],np.int8),n)
    for b in range(B):
        ix=rng.integers(0,n,n); rr=r3[ix]; mm=m3[ix]
        for arr,key in ((rr,"r"),(mm,"m")):
            pp=1/(1+np.exp(-np.clip(arr.reshape(-1),-60,60))); ee=arr[:,:1]>arr[:,1:]
            vals={"AUROC":roc_auc_score(yy,pp),"AUPRC":average_precision_score(yy,pp),"StrictPair":ee.all(1).mean(),"Edge":ee.mean()}
            if key=="r": vr=vals
            else: vm=vals
        for k in store: store[k][b]=vm[k]-vr[k]
        if (b+1)%500==0: print(f"{name} BOOTSTRAP {b+1}/{B}",flush=True)
    rows=[]
    for k,d in store.items():
        lo,hi=np.quantile(d,[.025,.975]); rows.append({"metric":k,"RAW":obsr[k],"MSRR":obsm[k],"observed_delta":obsm[k]-obsr[k],"bootstrap_mean":float(d.mean()),"CI95_LOW":float(lo),"CI95_HIGH":float(hi),"fraction_delta_gt_0":float((d>0).mean())})
    return csvwrite(name,rows),obsr,obsm

def scientific_arm_a(p):
    x126=import_file("x126b",ROOT/"scripts/126B_RUN_TASK_ALIGNED_BASELINES_SINGLE_THREAD_ONE_SHOT.py")
    formal=x126.formal; formal.OUT=OUT; x126.impl.OUT=OUT
    bundle,runner,base=formal.load_dataset_loader(); y=bundle.sample.y_pair.to_numpy(np.int8); pairs=np.asarray(bundle.pt,np.int64)
    hs=pd.read_parquet(H_SAMPLE).sort_values("sample_index"); units=hs.unit_id.astype(str).tolist()
    if len(units)!=15168 or not np.array_equal(hs.sample_index.to_numpy(),np.arange(15168)): raise RuntimeError("FAIL_128B_H_INDEX")
    htarget=pd.date_range(pd.Timestamp(p["hiroshima_cutoff_utc"])-pd.Timedelta(minutes=69*30),p["hiroshima_cutoff_utc"],freq="30min",tz="UTC")
    print("BUILD HIROSHIMA PRETRIGGER 15168x70x10",flush=True)
    original=np.asarray(bundle.dynamic,np.float32).copy(); hraw=event_raw_tensor(H_BASE,units,"timestamp_utc",False,htarget); bundle.dynamic=hraw
    oof={"RAW":np.full(15168,np.nan),"MSRR":np.full(15168,np.nan)}; foldrows=[]; sels=[]; outer=np.zeros(15168,np.int8)
    for hf in formal.FOLDS:
        fd=formal.build_fold_loader(bundle,runner,base,hf,torch.device("cpu")); ids=[np.asarray(v,np.int64) for v in (fd.train_pairs,fd.validation_pairs,fd.test_pairs)]; rows=[pairs[v].reshape(-1) for v in ids]; outer[rows[2]]=hf
        static=formal.as_numpy(fd.static92).astype(np.float32); rain=formal.as_numpy(fd.rain).astype(np.float32)
        for rep in ("RAW","MSRR"):
            fn=formal.raw_features if rep=="RAW" else formal.msrr_features; feats=[fn(pairs[v],static,rain)[0] for v in ids]
            print(f"ARM_A H FOLD{hf} {rep} START",flush=True)
            score,sel,cfg=formal.fit_xgb_select_predict(rep,hf,feats[0],y[rows[0]],feats[1],y[rows[1]],feats[2],7+hf*1000+600,1)
            oof[rep][rows[2]]=score; fm=metrics(score,y[rows[2]],np.arange(len(rows[2])).reshape(-1,3)); foldrows.append({"representation":rep,"human_fold":hf,"selected_config":cfg,**fm})
            sel["representation"]=rep; sel["human_fold"]=hf; sel["selected"]=sel.config_id==cfg; sels.append(sel)
            print(f"ARM_A H FOLD{hf} {rep} COMPLETE selected={cfg}",flush=True)
    if not all(np.isfinite(v).all() for v in oof.values()): raise RuntimeError("FAIL_128B_OOF")
    csvwrite("ARM_A_HIROSHIMA_FOLD_METRICS.csv",foldrows); selection=pd.concat(sels,ignore_index=True); csvwrite("ARM_A_HIROSHIMA_VALIDATION_SELECTION.csv",selection)
    np.save(OUT/"RAW_PRETRIGGER_HIROSHIMA_OOF_score.npy",oof["RAW"]); np.save(OUT/"MSRR_PRETRIGGER_HIROSHIMA_OOF_score.npy",oof["MSRR"])
    for rep in oof: np.save(OUT/f"{rep}_PRETRIGGER_HIROSHIMA_OOF_M.npy",oof[rep][pairs[:,0]]-oof[rep][pairs[:,1:]].max(1))
    hb,hr,hm=paired_bootstrap(oof["RAW"],oof["MSRR"],y,pairs,"ARM_A_HIROSHIMA_PRETRIGGER_BOOTSTRAP.csv")
    csvwrite("ARM_A_HIROSHIMA_OOF_RESULTS.csv",[{"representation":"RAW_PRETRIGGER",**hr},{"representation":"MSRR_PRETRIGGER",**hm}])
    selected={}
    for rep in ("RAW","MSRR"):
        agg=selection[selection.representation==rep].groupby("config_id",as_index=False).agg(validation_score=("validation_score","mean"),StrictPair=("StrictPair","mean"),Edge=("Edge","mean"),AUPRC=("AUPRC","mean"),AUROC=("AUROC","mean"))
        agg=agg.sort_values(["validation_score","StrictPair","Edge","AUPRC","AUROC","config_id"],ascending=[False,False,False,False,False,True],kind="mergesort"); selected[rep]=str(agg.iloc[0].config_id)
    # Full Hiroshima preprocessing uses the pretrigger development tensor.
    ext=import_file("ext124d",ROOT/"scripts/124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py"); ext.OUT=OUT; ext.log=lambda x: print(x,flush=True)
    m26=import_file("m26_128b",ROOT/"scripts/26_RUN_SEHC_NET_M_V1_FORMAL.py")
    sh,rh,pre,prep=ext.fit_full_hiroshima_preprocessor(bundle,runner,base,m26); xhr,_=formal.raw_features(pairs,sh,rh); xhm,_=formal.msrr_features(pairs,sh,rh); yh=y[pairs.reshape(-1)]
    matched=pd.read_parquet(ROOT/"external/kyushu_2017_asakura_toho/99_frozen_dataset/06_matched_index/kyushu_external_matched_triplet_index_v1.parquet")
    static=pd.read_parquet(ROOT/"external/kyushu_2017_asakura_toho/99_frozen_dataset/07_matched_static/kyushu_external_matched_static_92f_v1.parquet")
    model_index,pair_k=ext.build_external_model_index(matched); kunits=model_index.unit_id.tolist()
    ktarget=pd.date_range(pd.Timestamp(p["kyushu_cutoff_utc"])-pd.Timedelta(minutes=69*30),p["kyushu_cutoff_utc"],freq="30min",tz="UTC")
    print("BUILD KYUSHU PRETRIGGER 5076x70x10",flush=True); kraw=event_raw_tensor(K_BASE,kunits,"time_utc",True,ktarget)
    kd=[]
    for i,u in enumerate(kunits):
        z=pd.DataFrame(kraw[i],columns=FIELDS); z.insert(0,"time_index",np.arange(70)); z.insert(0,"timestamp_utc",ktarget); z.insert(0,"unit_id",u); kd.append(z)
    kdyn=pd.concat(kd,ignore_index=True); sk,rk,kaudit=ext.transform_external_with_hiroshima_preprocessor(static,kdyn,model_index,bundle,runner,m26,pre,base)
    xkr,_=formal.raw_features(pair_k,sk,rk); xkm,_=formal.msrr_features(pair_k,sk,rk)
    sr=ext.fit_predict_xgb(formal,selected["RAW"],xhr,yh,xkr,1); sm=ext.fit_predict_xgb(formal,selected["MSRR"],xhm,yh,xkm,1)
    eval_index,eval_rows=ext.build_external_evaluation_order(matched,model_index); ky=eval_index.y_true.to_numpy(np.int8); flat=eval_rows.reshape(-1); yp=ky
    # Reorder model scores into canonical evaluation row order for common metric/bootstrap helper.
    sr_eval=sr[flat]; sm_eval=sm[flat]; kp=np.arange(len(flat)).reshape(-1,3)
    kb,kr,km=paired_bootstrap(sr_eval,sm_eval,yp,kp,"ARM_A_KYUSHU_PRETRIGGER_BOOTSTRAP.csv",seed=20260722)
    csvwrite("ARM_A_KYUSHU_EXTERNAL_RESULTS.csv",[{"representation":"RAW_PRETRIGGER",**kr},{"representation":"MSRR_PRETRIGGER",**km}])
    np.save(OUT/"RAW_PRETRIGGER_KYUSHU_score.npy",sr); np.save(OUT/"MSRR_PRETRIGGER_KYUSHU_score.npy",sm)
    # Frozen-protocol Arm B is triggered; exact rematching implementation cannot reuse Arm-A memberships.
    arm_b={"status":"ARM_B_MATCHING_REBUILD_REQUIRED","trigger":"all four rainfall matching variables require recomputation","scientific_results_used_for_decision":False,"matching_protocol_change_allowed":False,"model_evaluation_status":"NOT_STARTED_UNTIL_EXACT_127_REMATCH_REPLAY"}
    jwrite("ARM_B_HIROSHIMA_MATCHING_AUDIT.json",arm_b); jwrite("ARM_B_KYUSHU_MATCHING_AUDIT.json",arm_b)
    return hr,hm,hb,kr,km,kb,selected,original,hraw,arm_b

def finalize_scientific(p,ph,replay,history,result):
    hr,hm,hb,kr,km,kb,selected,original,hraw,arm_b=result; metrics4=["AUROC","AUPRC","StrictPair","Edge"]
    hi=sum(hm[x]>hr[x] for x in metrics4); hci=int((hb.CI95_LOW>0).sum()); ki=sum(km[x]>kr[x] for x in metrics4); kci=int((kb.CI95_LOW>0).sum())
    gate="PASS_STRONG_PRETRIGGER_ROBUSTNESS" if hi==4 and hci==4 and ki>=3 and kci>=3 else ("PASS_INTERNAL_PRETRIGGER_ROBUSTNESS_EXTERNAL_MIXED" if hi>=3 and hci>=3 else "PRETRIGGER_EFFECT_WEAKENED")
    diag=[]
    for j,f in enumerate(FIELDS):
        a=original[:,:,j].reshape(-1); b=hraw[:,:,j].reshape(-1); diag.append({"event":"Hiroshima 2018","variable":f,"original_mean":float(a.mean()),"pretrigger_mean":float(b.mean()),"original_median":float(np.median(a)),"pretrigger_median":float(np.median(b)),"paired_mean_difference":float((b-a).mean())})
    csvwrite("RAINFALL_WINDOW_SHIFT_DIAGNOSTICS.csv",diag)
    ref=pd.read_csv(ROOT/"experiments/MSRR_TASK_ALIGNED_BASELINES_5FOLD_V1B/OOF_RESULTS.csv"); rows=[]
    for rep,new in (("RAW",hr),("MSRR",hm)):
        old=ref[ref.method==rep+"_POINTWISE_XGB"].iloc[0]
        for m in metrics4: rows.append({"event":"Hiroshima 2018","representation":rep,"metric":m,"original":old[m],"pretrigger":new[m],"pretrigger_minus_original":new[m]-old[m]})
    csvwrite("ORIGINAL_VS_PRETRIGGER_PERFORMANCE.csv",rows)
    audit={"status":"PASS_128B_ARM_A_COMPLETE_ARM_B_REBUILD_PENDING","cutoffs_written_before_scientific_run":True,"cutoffs_unchanged":True,"preregistration_sha256_unchanged":ph==sha(PREREG),"steps_each":70,"spacing_minutes":30,"original_rainfall_pipeline_replay":"PASS_EXACT_FLOAT32","IMERG":"V07B Final Run","dynamic_definitions_changed":False,"past_only":True,"RAW_dim":792,"MSRR_dim":2446,"same_Hiroshima_folds_ARM_A":True,"same_frozen_matched_sets_ARM_A":True,"XGBoost_n_jobs":1,"validation_only_selection":True,"Kyushu_adaptation":False,"bootstrap_B":10000,"bootstrap_unit":"complete matched set","ARM_B_decision":"REQUIRED","ARM_B_status":arm_b["status"],"post_result_rescue":False}
    jwrite("AUDIT128B.json",audit); jwrite("GATE128B_DECISION.json",{"GATE128B_DECISION":gate,"H_INT":hi,"H_CI":hci,"K_EXT":ki,"K_CI":kci,"PRIMARY_GATE_ARM":"ARM_A","ARM_B_STATUS":arm_b["status"]})
    (OUT/"GATE128B_REPORT.md").write_text(f"# Experiment 128B\n\nPrimary Arm A gate: **{gate}**. Hiroshima wins/positive CIs: {hi}/4 and {hci}/4. Kyushu wins/positive CIs: {ki}/4 and {kci}/4. Arm B remains required because all four rainfall matching variables shift with the cutoff; its exact matching rebuild is not represented by Arm A memberships.\n",encoding="utf-8")
    (OUT/"REVIEWER2_COMMENT5_128B_RESPONSE_DRAFT.md").write_text("# Reviewer #2 Comment 5 — response draft\n\nThe original manuscript did not specify the exact rainfall timestamps. All rolling rainfall features are past-only, but grid-specific failure times are unavailable and the original event-level windows overlap the documented disaster periods. We therefore preregistered conservative 70-step windows ending at 2018-07-06 18:30 JST for Hiroshima and 2017-07-05 16:00 JST for Kyushu. The unchanged rainfall pipeline was replayed exactly and the Arm A temporal sensitivity was completed. Grid-specific post-failure exclusion still cannot be proven retrospectively. Exact Arm B rematching remains required because the four rainfall matching fields depend on the shifted interval.\n",encoding="utf-8")
    print("PASS_128B_PRETRIGGER_SENSITIVITY_ARM_A_COMPLETE"); print("GATE128B_DECISION="+gate); print("CUTOFFS_FROZEN_BEFORE_RESULTS=True")
    for name,r,m,b in (("HIROSHIMA",hr,hm,hb),("KYUSHU",kr,km,kb)):
        print(name+":")
        for x in metrics4: print(f"RAW_{x}={r[x]:.9f}\nMSRR_{x}={m[x]:.9f}\nDELTA_{x}={m[x]-r[x]:.9f}\nCI_{x}=[{b.loc[b.metric==x,'CI95_LOW'].iloc[0]:.9f},{b.loc[b.metric==x,'CI95_HIGH'].iloc[0]:.9f}]")
    print("MATCHING_RAINFALL_REQUIRES_RECOMPUTATION=True\nARM_B_STATUS="+arm_b["status"]+"\nORIGINAL_PIPELINE_REPLAY=PASS_EXACT_FLOAT32\nPOST_RESULT_RESCUE=False\nOUTPUT="+str(OUT))

def main():
    p,ph=validate_prereg(); dependency_audit()
    hnew=make_timeline("Hiroshima 2018",p["hiroshima_cutoff_utc"],"HIROSHIMA_PRETRIGGER_70_TIMELINE.csv")
    knew=make_timeline("Kyushu 2017",p["kyushu_cutoff_utc"],"KYUSHU_PRETRIGGER_70_TIMELINE.csv")
    replay=original_replay()
    _,ht=read_rain(H_BASE,[pd.read_parquet(H_SAMPLE,columns=["unit_id"]).unit_id.iloc[0]],"timestamp_utc",False)
    units=sorted(pd.read_parquet(K_VIEW,columns=["unit_id"]).unit_id.unique())
    _,kt=read_rain(K_BASE,[units[0]],"time_utc",True)
    hist,ok=history_audit(ht,kt,hnew,knew)
    if not ok:
        status=write_failure(ph,replay,hist)
        print(status); print("GATE128B_DECISION="+status); print("CUTOFFS_FROZEN_BEFORE_RESULTS=True")
        print("MATCHING_RAINFALL_REQUIRES_RECOMPUTATION=True"); print("ARM_B_STATUS=REQUIRED_NOT_RUN_INPUT_UNAVAILABLE")
        print("ORIGINAL_PIPELINE_REPLAY=PASS_EXACT_FLOAT32"); print("POST_RESULT_RESCUE=False"); print("OUTPUT="+str(OUT)); return
    result=scientific_arm_a(p)
    finalize_scientific(p,ph,replay,hist,result)

if __name__=="__main__": main()
