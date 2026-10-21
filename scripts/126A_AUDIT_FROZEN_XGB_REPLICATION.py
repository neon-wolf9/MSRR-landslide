#!/usr/bin/env python
"""Read-only protocol forensics; only Fold 1 RAW XGB technical reruns are fitted."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import inspect
import itertools
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import sklearn
import scipy
import joblib

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/MSRR_TASK_ALIGNED_BASELINES_5FOLD_V1"
OUT = PARENT / "126A_REPLICATION_AUDIT"
REF = ROOT / "experiments/MSRR_CROSS_BACKBONE_5FOLD_V1B"
S120 = ROOT / "scripts/120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py"
S126 = ROOT / "scripts/126_RUN_TASK_ALIGNED_BASELINES_5FOLD_ONE_SHOT.py"
sys.path.insert(0, str(ROOT))
SCORE_ATOL = 1e-5
METRIC_ATOL = 1e-6


def pearsonr(a,b):
    """Coefficient only; elementwise reduction avoids native corrcoef failure."""
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    ac,bc=a-a.mean(),b-b.mean()
    return float(np.sum(ac*bc)/np.sqrt(np.sum(ac*ac)*np.sum(bc*bc))), None


def spearmanr(a,b):
    # Spearman is Pearson on average ranks, including tied scores.
    return pearsonr(pd.Series(a).rank(method="average").to_numpy(),
                    pd.Series(b).rank(method="average").to_numpy())


def parameter_equal(a,b):
    # XGBoost's shared default missing=np.nan is semantically equal to itself.
    if isinstance(a,(float,np.floating)) and isinstance(b,(float,np.floating)):
        if np.isnan(a) and np.isnan(b): return True
    return a==b


def write(name, value):
    def convert(v):
        if v is pd.NA or v is pd.NaT: return None
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert)+"\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def log(message):
    with (OUT/"126A_RUN.log").open("a", encoding="utf-8") as f:
        f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ")+message+"\n")


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summary(a):
    a=np.asarray(a)
    if a.dtype.hasobject:
        data=json.dumps(a.tolist(),ensure_ascii=False,default=str,separators=(",",":")).encode()
        encoding="JSON UTF-8 ordered values (no object pointer bytes)"
    else:
        data=np.ascontiguousarray(a).tobytes(); encoding="contiguous ndarray bytes"
    return dict(shape=list(a.shape),dtype=str(a.dtype),SHA256=hashlib.sha256(data).hexdigest(),
                hash_encoding=encoding,first20=a.ravel()[:20].tolist(),last20=a.ravel()[-20:].tolist())


def compare(name,a,b):
    a,b=np.asarray(a),np.asarray(b)
    same_shape=a.shape==b.shape
    d=np.abs(a.astype(np.float64)-b.astype(np.float64)) if same_shape else np.array([np.inf])
    return dict(item=name,array_equal=bool(np.array_equal(a,b)),dtype_equal=a.dtype==b.dtype,
        bytes_equal=a.dtype==b.dtype and summary(a)["SHA256"]==summary(b)["SHA256"],
        max_abs_diff=float(d.max()),mean_abs_diff=float(d.mean()),
        different_element_count=int(np.sum(a!=b)) if same_shape else -1)


def csv(name,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT/name,index=False,encoding="utf-8-sig")
    return frame


def main_assignment(source,target):
    main=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="main")
    for node in ast.walk(main):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in node.targets):
            return node
    raise RuntimeError("Assignment not found: "+target)


def eval_assignment(source,target,env):
    node=main_assignment(source,target)
    env[target]=eval(compile(ast.Expression(node.value),"<audited-source>","eval"),env)
    return env[target]


def source_audit(a,b,t120,t126):
    entries=[]
    for name in ("load_dataset_loader","build_fold_loader","raw_features","class_weights","make_xgb",
                 "fit_xgb_select_predict","metrics_from_ordered_rows","val_scalar","choose_best_index","prob_to_margin"):
        fa,fb=getattr(a,name),getattr(b.formal,name)
        sa,sb=inspect.getsource(fa),inspect.getsource(fb)
        file_a=inspect.getsourcefile(fa); line=inspect.getsourcelines(fa)[1]
        entries.append(dict(logic=name,code120=f"{file_a}:{line}\n{sa}",
            code126=f"126 imports formal120b at {S126}:35 and calls formal.{name}; resolved code: {inspect.getsourcefile(fb)}:{inspect.getsourcelines(fb)[1]}",
            status="IDENTICAL" if sa==sb else "DIFFERENT",note="Exact imported function body"))
    pairs=[("tr_pair_ids","ids"),("va_pair_ids","ids"),("te_pair_ids","ids"),
           ("tr_pairs","rows"),("va_pairs","rows"),("te_pairs","rows"),("fold_seed","seed")]
    for left,right in pairs:
        na,nb=main_assignment(t120,left),main_assignment(t126,right)
        entries.append(dict(logic=left,code120=f"{S120}:{na.lineno}\n{ast.get_source_segment(t120,na)}",
            code126=f"{S126}:{nb.lineno}\n{ast.get_source_segment(t126,nb)}",status="IDENTICAL",
            note="Equivalent evaluated protocol; syntax differs. Pair-array vs flattened-row intermediates are compared in IMPLEMENTATION_EQUIVALENCE.csv. Fold seed comparison includes +600 passed at 120B XGB call."))
    entries.append(dict(logic="XGB_CANDIDATES",code120=json.dumps(a.XGB_CANDIDATES),code126="formal.XGB_CANDIDATES (same source)",
        status="IDENTICAL" if a.XGB_CANDIDATES==b.formal.XGB_CANDIDATES else "DIFFERENT",note="No copied candidate grid"))
    entries.append(dict(logic="candidate-specific seed",code120="120B:1220 seed + ci * 37",code126="126:290 calls formal.fit_xgb_select_predict, same seed + ci * 37",
        status="IDENTICAL",note="D4/D6/D8 seeds evaluated separately in seed audit"))
    report="# Source protocol comparison\n\nScope: Fold 1 RAW_POINTWISE_XGB only. IDENTICAL means effective protocol, not character-for-character orchestration syntax.\n\n"
    report+="The entire 120B source hash is checked against its historical manifest. Historical dependency and feature-byte snapshots were not recorded there; current dual-path equivalence cannot prove historical byte identity.\n\n"
    report+="Execution context difference: 120B trains HistGradientBoosting before XGB, whereas 126 fits XGB first; explicit XGB seeds and arguments are unchanged. This is recorded, not assumed causal or silently repaired.\n\n"
    for e in entries:
        report+=f"## {e['logic']}: {e['status']}\n\n120B code:\n```python\n{e['code120']}\n```\n126 code:\n```python\n{e['code126']}\n```\n{e['note']}\n\n"
    report+="No actual change to the scoped effective pointwise protocol was identified. No source was repaired. CLI default jobs differ (120B=4,126=8), but requested runs use explicit --xgb-jobs 8; historical actual thread count is not recorded in 120B manifest.\n"
    (OUT/"SOURCE_PROTOCOL_DIFF.md").write_text(report,encoding="utf-8")
    return all(e["status"]=="IDENTICAL" for e in entries)


def search_history():
    cached=OUT/"TECHNICAL_ATTEMPT_01/HISTORICAL_RUNTIME_SEARCH.json"
    if cached.exists():
        result=json.loads(cached.read_text(encoding="utf-8"))
        result["reused_search_after_JSON_serialization_bug"]=str(cached)
        write("HISTORICAL_RUNTIME_SEARCH.json",result)
        return result,result["candidate_prediction_artifact_candidates"]
    # Entire project inventory, including ignored files; no network or external paths.
    proc=subprocess.run(["rg","--files","--hidden","--no-ignore","-g","!.git/**","-g","!**/__pycache__/**",str(ROOT)],capture_output=True,text=True,encoding="utf-8",errors="replace")
    paths=[Path(p) for p in proc.stdout.splitlines()]
    evidence=[]; version_evidence=[]; candidates=[]; scanned=[]; errors=[]
    version_rx=re.compile(r"(?:xgboost|py-xgboost|libxgboost)[\s\"':=<>~v-]{1,35}([0-9]+\.[0-9]+(?:\.[0-9]+)?)",re.I)
    for p in paths:
        if OUT in p.parents: continue
        low=p.name.lower()
        if p.suffix.lower() in (".npy",".npz",".parquet",".csv",".ubj",".json") and (
                any(s in str(p).lower() for s in ("120b","cross_backbone")) and
                any(s in low for s in ("pred","score","margin","model","d4","d6","d8"))):
            candidates.append(str(p))
        eligible=(p.suffix.lower() in (".log",".txt",".yml",".yaml") or
                  p.suffix.lower() in (".json",".md",".csv") and any(s in low for s in (
                      "manifest","provenance","environment","runtime","conda","freeze","console","audit","dependency","summary")))
        if not eligible: continue
        try:
            scanned.append(str(p))
            with p.open(encoding="utf-8-sig",errors="replace") as f:
                for i,line in enumerate(f,1):
                    if re.search(r"xgboost|pip freeze|conda list",line,re.I):
                        record=dict(file=str(p),line=i,text=line.rstrip()[:3000])
                        evidence.append(record)
                        match=version_rx.search(line)
                        if match: version_evidence.append({**record,"version":match.group(1),"attribution":"unverified for actual 120B run"})
        except OSError as e: errors.append(dict(file=str(p),error=str(e)))
    # Only run-specific provenance can establish the historical 120B runtime.
    direct=[v for v in version_evidence if REF in Path(v["file"]).parents]
    found=bool(direct)
    versions=sorted(set(v["version"] for v in direct))
    result=dict(HISTORICAL_XGBOOST_VERSION_FOUND=found,HISTORICAL_XGBOOST_VERSION=versions[0] if len(versions)==1 else None,
        scope=str(ROOT),internet_used=False,scanned_file_count=len(scanned),scanned_files=scanned,
        rg_returncode=proc.returncode,inventory_errors=proc.stderr,read_errors=errors,
        version_evidence=version_evidence,keyword_evidence=evidence,candidate_prediction_artifact_candidates=candidates,
        limitation="Versions in other experiments (including same-environment conda snapshots) are contextual evidence, not proof of the actual 120B process version. Permission-denied paths cannot be ruled out.")
    write("HISTORICAL_RUNTIME_SEARCH.json",result)
    return result,candidates


def main():
    if OUT.exists() and any(p.name!="TECHNICAL_ATTEMPT_01" for p in OUT.iterdir()):
        raise RuntimeError("Refusing to overwrite forensic audit")
    OUT.mkdir(parents=True,exist_ok=True)
    protected=[S120,S126,ROOT/"scripts/121_RECOVER_OOF_AND_PAIRED_BOOTSTRAP.py",ROOT/"scripts/TRAIN_SEHC_V2_TEMPLATE.py"]
    for directory in (REF,ROOT/"experiments/MSRR_121_PAIRED_BOOTSTRAP",PARENT):
        protected.extend(p for p in directory.rglob("*") if p.is_file() and OUT not in p.parents)
    protected_hashes={str(p):sha(p) for p in protected}
    write("126A_RUN_MANIFEST.json",dict(task="technical forensic audit only",source_sha256=sha(__file__),
        protected_hashes=protected_hashes,fold=1,representation="frozen RAW 792",scientific_baselines_run=False,
        fits="D4/D6/D8 jobs8 once; jobs1 once; each jobs1 three additional consecutive repeats",total_fits=15,
        score_atol=SCORE_ATOL,metric_atol=METRIC_ATOL,rtol=0,selection="validation only, exact 120B tie break",
        diagnostics_precedence=["INPUT_OR_PROTOCOL_DRIFT_CONFIRMED","CURRENT_RUNTIME_NONDETERMINISM","THREAD_COUNT_SENSITIVITY","LIKELY_SOFTWARE_RUNTIME_DRIFT","CAUSE_UNRESOLVED"],
        historical_identity_caveat="Current dual-path identity is not a historical feature snapshot.",network=False,package_changes=False))
    a=load("audit120b",S120); b=load("audit126",S126)
    t120,t126=S120.read_text(encoding="utf-8"),S126.read_text(encoding="utf-8")
    protocol_identical=source_audit(a,b,t120,t126)
    ref_manifest=json.loads((REF/"RUN_MANIFEST.json").read_text())
    historical_source_identical=sha(S120)==ref_manifest["source_sha256"]
    runtime=dict(Python=sys.version,executable=sys.executable,numpy=np.__version__,pandas=pd.__version__,
        scikit_learn=sklearn.__version__,xgboost=xgb.__version__,joblib=joblib.__version__,scipy=scipy.__version__,
        OS=platform.platform(),Windows=platform.win32_ver(),CPU=platform.processor(),cpu_count=os.cpu_count(),
        xgboost_file=xgb.__file__,build_info=xgb.build_info() if hasattr(xgb,"build_info") else None,
        thread_environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")})
    write("CURRENT_RUNTIME.json",runtime)
    log("Source comparison complete; searching historical project provenance")
    history,pred_candidates=search_history()
    # Redirect only a legacy preflight report's destination in freshly imported modules.
    # Dataset paths, preflight logic, preprocessing and fold construction are unchanged.
    import scripts.TRAIN_SEHC_V2_TEMPLATE as template
    original_import=template._import_file
    def local_import(name,path):
        module=original_import(name,path)
        if Path(path)==template.BASE06_RUNNER:
            module.base.OUT=OUT/"LOADER_PREFLIGHT"
        return module
    template._import_file=local_import
    a.set_seed(a.SEED+1000)
    bundle,runner,base=a.load_dataset_loader()
    fd=a.build_fold_loader(bundle,runner,base,1,torch.device("cpu"))
    a.set_seed(a.SEED+1000)
    bundle2,runner2,base2=b.formal.load_dataset_loader()
    fd2=b.formal.build_fold_loader(bundle2,runner2,base2,1,torch.device("cpu"))
    template._import_file=original_import
    pairs=np.asarray(bundle.pt,dtype=np.int64); y=bundle.sample.y_pair.to_numpy(np.int8)
    structure={"bundle.pt":summary(pairs),"y_all":summary(y),"sample_row_order":summary(bundle.sample.index.to_numpy()),
               "scope":"Current execution of unchanged 120B source; not a recovered historical hash"}
    for name in ("sample_index","unit_id","pair_set_id","sample_role","control_rank"):
        if name in bundle.sample: structure["sample."+name]=summary(bundle.sample[name].to_numpy())
    structure["sample_table_ordered_hash"]=summary(pd.util.hash_pandas_object(bundle.sample,index=True).to_numpy())
    env=dict(np=np,fd=fd,all_pair_rows=pairs)
    left={}; right={}
    for name in ("tr_pair_ids","va_pair_ids","te_pair_ids","tr_pairs","va_pairs","te_pairs","tr_idx","va_idx","te_idx"):
        left[name]=eval_assignment(t120,name,env); structure[name]=summary(left[name])
    env2=dict(np=np,fd=fd2,pairs=np.asarray(bundle2.pt,dtype=np.int64))
    ids=eval_assignment(t126,"ids",env2); rows=eval_assignment(t126,"rows",env2)
    equivalence=[compare("bundle.pt",pairs,bundle2.pt),compare("y_all",y,bundle2.sample.y_pair.to_numpy(np.int8))]
    for i,prefix in enumerate(("tr","va","te")):
        right[prefix+"_pair_ids"]=ids[i]; right[prefix+"_pairs"]=rows[i].reshape(-1,3); right[prefix+"_idx"]=rows[i]
    equivalence.extend(compare(k,left[k],right[k]) for k in left)
    with np.load(PARENT/"FOLD1_FROZEN_SPLITS.npz") as saved:
        structure["matches_failed_126_saved_splits"]=all(np.array_equal(ids[i],saved[k]) for i,k in enumerate(("train","validation","test")))
    write("FOLD1_STRUCTURE_HASHES.json",structure)
    labels={}; features={}; data={}; feature_hashes={}
    static=a.as_numpy(fd.static92).astype(np.float32,copy=False); rain=a.as_numpy(fd.rain).astype(np.float32,copy=False)
    static2=b.formal.as_numpy(fd2.static92).astype(np.float32,copy=False); rain2=b.formal.as_numpy(fd2.rain).astype(np.float32,copy=False)
    for prefix in ("tr","va","te"):
        yy=y[left[prefix+"_idx"]]; yy2=bundle2.sample.y_pair.to_numpy(np.int8)[right[prefix+"_idx"]]
        if not np.array_equal(yy.reshape(-1,3),np.tile([1,0,0],(len(yy)//3,1))):
            raise RuntimeError("FAIL_126A_PAIR_ORDER_OR_LABEL_MISMATCH")
        labels["y"+prefix]={**summary(yy),"sum":int(yy.sum()),"class_counts":{str(v):int((yy==v).sum()) for v in np.unique(yy)}}
        equivalence.append(compare("y"+prefix,yy,yy2))
        xx,anchors=a.raw_features(left[prefix+"_pairs"],static,rain)
        xx2,anchors2=b.formal.raw_features(right[prefix+"_pairs"],static2,rain2)
        if xx.shape[1]!=792 or not np.array_equal(anchors,anchors2): raise RuntimeError("FAIL_126A_FEATURE_ORDER")
        equivalence.append(compare("x"+prefix,xx,xx2))
        feature_hashes["x"+prefix]={**summary(xx),"C_CONTIGUOUS":bool(xx.flags.c_contiguous),"F_CONTIGUOUS":bool(xx.flags.f_contiguous),
            "min":float(xx.min()),"max":float(xx.max()),"mean":float(xx.mean(dtype=np.float64)),"std":float(xx.std(dtype=np.float64)),
            "NaN_count":int(np.isnan(xx).sum()),"Inf_count":int(np.isinf(xx).sum()),
            "float32_contiguous_SHA256":hashlib.sha256(np.ascontiguousarray(xx,dtype=np.float32).tobytes()).hexdigest(),
            "first_row_first20":xx[0,:20].tolist(),"last_row_last20":xx[-1,-20:].tolist()}
        data["x"+prefix]=xx; data["y"+prefix]=yy
    write("FOLD1_LABEL_HASHES.json",labels); write("FOLD1_RAW_FEATURE_HASHES.json",feature_hashes)
    eq=csv("IMPLEMENTATION_EQUIVALENCE.csv",equivalence)
    sw=a.class_weights(data["ytr"]); sw2=b.formal.class_weights(data["ytr"])
    weight_equal=np.array_equal(sw,sw2)
    write("FOLD1_WEIGHT_AUDIT.json",dict(**summary(sw),unique_values=np.unique(sw),counts={str(v):int((sw==v).sum()) for v in np.unique(sw)},
        positive_weight=np.unique(sw[data["ytr"]==1]),control_weight=np.unique(sw[data["ytr"]==0]),comparison=compare("weights",sw,sw2)))
    foldseed=eval_assignment(t120,"fold_seed",dict(SEED=a.SEED,hf=1))
    xseed=eval_assignment(t126,"seed",dict(formal=b.formal,hf=1))
    seeds={cfg["config_id"]:dict(seed120B=foldseed+600+i*37,seed126=xseed+i*37) for i,cfg in enumerate(a.XGB_CANDIDATES)}
    write("FOLD1_SEED_AUDIT.json",dict(SEED=a.SEED,fold_seed=foldseed,xgb_base_seed120B=foldseed+600,xgb_base_seed126=xseed,candidates=seeds))
    params={}; paramdiff={}
    for cfg in a.XGB_CANDIDATES:
        cid=cfg["config_id"]
        ma=a.make_xgb(cfg,seeds[cid]["seed120B"],8); mb=b.formal.make_xgb(cfg,seeds[cid]["seed126"],8)
        pa,pb=ma.get_params(deep=True),mb.get_params(deep=True)
        params[cid]=dict(get_params=pa,get_xgb_params=ma.get_xgb_params())
        paramdiff[cid]=dict(params120B=pa,params126=pb,differences={k:[pa.get(k),pb.get(k)] for k in set(pa)|set(pb) if not parameter_equal(pa.get(k),pb.get(k))})
    write("XGB_EFFECTIVE_PARAMETERS.json",params); write("XGB_120B_VS_126_PARAM_DIFF.json",paramdiff)
    # Read and verify every frozen OOF file, not only XGBoost.
    oofaudit=[]
    ref_oof=pd.read_csv(REF/"OOF_RESULTS.csv")
    for p in sorted(REF.glob("*_OOF_margin.npy")):
        arr=np.load(p,allow_pickle=False)
        values=a.metrics_from_full_oof(arr,y,pairs)
        matches=[r for _,r in ref_oof.iterrows() if f"{r.backbone}_{r.representation}_OOF_margin.npy".replace("-","_")==p.name]
        r=matches[0]
        oofaudit.append(dict(file=str(p),file_sha256=sha(p),**summary(arr),metrics=values,
                            matches_historical_metrics=all(abs(values[k]-r[k])<1e-9 for k in a.METRICS4)))
    write("HISTORICAL_OOF_AUDIT.json",oofaudit)
    write("HISTORICAL_CANDIDATE_PREDICTIONS.json",dict(status="HISTORICAL_CANDIDATE_PREDICTIONS_UNAVAILABLE",
        inspected_candidates=pred_candidates,available="Frozen selected-model pooled OOF scores only, including Fold 1 RAW D6 test scores; no per-candidate validation score files located."))
    drift=not(protocol_identical and historical_source_identical and eq.array_equal.all() and eq.bytes_equal.all() and weight_equal and
              all(v["seed120B"]==v["seed126"] for v in seeds.values()) and not any(v["differences"] for v in paramdiff.values()))
    log("Input/parameter audits complete. Starting 15 fixed technical fits.")
    predictions={}; results={}; modelconfigs={}
    def fit(cfg,jobs,repeat=0):
        cid=cfg["config_id"]; seed=seeds[cid]["seed120B"]
        log(f"FIT {cid} jobs={jobs} repeat={repeat}")
        a.set_seed(foldseed)
        model=a.make_xgb(cfg,seed,jobs)
        model.fit(data["xtr"],data["ytr"],sample_weight=sw)
        val=a.prob_to_margin(model.predict_proba(data["xva"])[:,1])
        test=a.prob_to_margin(model.predict_proba(data["xte"])[:,1])
        vm=a.metrics_from_ordered_rows(val,data["yva"]); tm=a.metrics_from_ordered_rows(test,data["yte"])
        tag=f"FOLD1_{cid}_NJOBS{jobs}"+(f"_REPEAT{repeat}" if repeat else "")
        np.save(OUT/(tag+"_VAL_SCORE.npy"),val); np.save(OUT/(tag+"_TEST_SCORE.npy"),test)
        modelconfigs[tag]=json.loads(model.get_booster().save_config())
        predictions[(cid,jobs,repeat)]=(val,test)
        rec=dict(config_id=cid,n_jobs=jobs,repeat=repeat,actual_seed=seed,
            **{f"validation_{k}":v for k,v in vm.items()},validation_score=a.val_scalar(vm),
            **{f"test_{k}":v for k,v in tm.items()})
        results[(cid,jobs,repeat)]=rec
        return rec
    for jobs in (8,1): csv(f"FOLD1_XGB_NJOBS{jobs}.csv",[fit(cfg,jobs) for cfg in a.XGB_CANDIDATES])
    for cfg in a.XGB_CANDIDATES:
        for repeat in (1,2,3): fit(cfg,1,repeat)
    write("XGB_TRAINED_BOOSTER_CONFIGS.json",modelconfigs)
    def predcompare(cid,leftkey,rightkey,scope):
        recs=[]
        for i,part in enumerate(("validation","test")):
            av,bv=predictions[leftkey][i],predictions[rightkey][i]
            am=a.metrics_from_ordered_rows(av,data["yva" if i==0 else "yte"])
            bm=a.metrics_from_ordered_rows(bv,data["yva" if i==0 else "yte"])
            d=np.abs(av-bv)
            recs.append(dict(config_id=cid,split=part,comparison=scope,max_abs_score_diff=float(d.max()),mean_abs_score_diff=float(d.mean()),
                exact_equal=bool(np.array_equal(av,bv)),within_frozen_score_tolerance=bool(np.allclose(av,bv,atol=SCORE_ATOL,rtol=0)),
                Pearson=float(pearsonr(av,bv)[0]),Spearman=float(spearmanr(av,bv)[0]),
                **{f"delta_{k}":bm[k]-am[k] for k in am}))
        return recs
    threading=[]; repeated=[]
    for cfg in a.XGB_CANDIDATES:
        cid=cfg["config_id"]
        threading.extend(predcompare(cid,(cid,8,0),(cid,1,0),"jobs8 vs jobs1"))
        for i,j in itertools.combinations((0,1,2,3),2):
            repeated.extend(predcompare(cid,(cid,1,i),(cid,1,j),f"repeat{i} vs repeat{j}"))
    threads=csv("NJOBS_DETERMINISM.csv",threading); reps=csv("REPEATED_RUN_DETERMINISM.csv",repeated)
    deterministic=bool(reps.within_frozen_score_tolerance.all()); threads_equal=bool(threads.within_frozen_score_tolerance.all())
    selection=pd.read_csv(REF/"VALIDATION_SELECTION.csv")
    historical=selection[(selection.human_fold==1)&(selection.backbone=="XGBoost")&(selection.representation=="RAW")]
    def selected(frame,prefix=""):
        rr=[dict(config_id=r["config_id"],validation_score=r["validation_score"],**{k:r[prefix+k] for k in a.METRICS4}) for r in frame]
        return rr[a.choose_best_index(rr)]["config_id"]
    selected_old=selected(historical.to_dict("records"))
    selected8=selected([results[(c["config_id"],8,0)] for c in a.XGB_CANDIDATES],"validation_")
    selected1=selected([results[(c["config_id"],1,0)] for c in a.XGB_CANDIDATES],"validation_")
    comp=[]
    for _,r in historical.iterrows():
        for metric in a.METRICS4+["validation_score"]:
            key=metric if metric=="validation_score" else "validation_"+metric
            v8=results[(r.config_id,8,0)][key]; v1=results[(r.config_id,1,0)][key]
            comp.append(dict(config=r.config_id,metric=metric,**{"120B_value":float(r[metric])},
                current_jobs8_value=v8,current_jobs1_value=v1,delta_jobs8=v8-r[metric],delta_jobs1=v1-r[metric]))
    comptable=csv("FOLD1_120B_VS_CURRENT_VALIDATION.csv",comp)
    mismatch=bool((comptable.delta_jobs8.abs()>METRIC_ATOL).any())
    # Same D6 selected-model scores are available for a valid historical test comparison.
    oldtest=np.load(REF/"XGBoost_RAW_OOF_margin.npy")[left["te_idx"]]
    htest=[]
    for jobs in (8,1):
        new=predictions[(selected_old,jobs,0)][1]
        htest.append({**compare(f"historical_{selected_old}_test_vs_current_jobs{jobs}",oldtest,new),
            "Pearson":float(pearsonr(oldtest,new)[0]),"Spearman":float(spearmanr(oldtest,new)[0])})
    csv("HISTORICAL_SELECTED_TEST_SCORE_COMPARISON.csv",htest)
    previous=np.load(PARENT/"FOLD1_RAW_POINTWISE_XGB_score.npy")
    csv("FAILED126_VS_CURRENT_SELECTED_TEST.csv",[compare("failed126_vs_current_jobs8",previous,predictions[(selected8,8,0)][1])])
    if drift: diagnosis="INPUT_OR_PROTOCOL_DRIFT_CONFIRMED"
    elif not deterministic: diagnosis="CURRENT_RUNTIME_NONDETERMINISM"
    elif not threads_equal: diagnosis="THREAD_COUNT_SENSITIVITY"
    elif mismatch and (not history["HISTORICAL_XGBOOST_VERSION_FOUND"] or history["HISTORICAL_XGBOOST_VERSION"]!=xgb.__version__):
        diagnosis="LIKELY_SOFTWARE_RUNTIME_DRIFT"
    else: diagnosis="CAUSE_UNRESOLVED"
    unchanged=all(sha(p)==h for p,h in protected_hashes.items())
    if not unchanged: raise RuntimeError("FAIL_126A_PROTECTED_FILE_CHANGED")
    flags=dict(FOLD_IDS_IDENTICAL=bool(eq[eq.item.str.endswith("pair_ids")].array_equal.all()),
        PAIR_ORDER_IDENTICAL=True,LABELS_IDENTICAL=bool(eq[eq.item.str.startswith("y")].array_equal.all()),
        RAW_FEATURE_BYTES_IDENTICAL=bool(eq[eq.item.str.startswith("x")].bytes_equal.all()),WEIGHTS_IDENTICAL=bool(weight_equal),
        SEEDS_IDENTICAL=all(v["seed120B"]==v["seed126"] for v in seeds.values()),EXPLICIT_PARAMS_IDENTICAL=not any(v["differences"] for v in paramdiff.values()),
        CURRENT_RUNTIME_DETERMINISTIC=deterministic,NJOBS1_VS_8_IDENTICAL=threads_equal,
        HISTORICAL_XGBOOST_VERSION_FOUND=history["HISTORICAL_XGBOOST_VERSION_FOUND"],
        HISTORICAL_XGBOOST_VERSION=history["HISTORICAL_XGBOOST_VERSION"],CURRENT_XGBOOST_VERSION=xgb.__version__,
        **{"120B_FOLD1_SELECTED":selected_old},CURRENT_NJOBS8_SELECTED=selected8,CURRENT_NJOBS1_SELECTED=selected1)
    decision=dict(status="PASS_126A_FORENSIC_AUDIT_COMPLETE",DIAGNOSIS=diagnosis,**flags,
        identity_scope="120B and 126 source paths executed now on independently loaded data; does not prove historical 120B feature bytes",
        historical_feature_bytes_verified=False,historical_dependency_versions_verified=False,
        threading_status="THREADING_NONDETERMINISM_NOT_SUPPORTED" if threads_equal else "THREADING_NONDETERMINISM_DETECTED",
        repeated_status="CURRENT_RUNTIME_DETERMINISTIC" if deterministic else "CURRENT_RUNTIME_NONDETERMINISTIC",
        protected_files_unchanged=unchanged,scientific_baselines_run=False,
        technical_amendment="Evidence can support drafting a prospective amendment with explicit provenance limitations; does not authorize running 126B or declaring frozen replication passed.")
    write("126A_DIAGNOSTIC_DECISION.json",decision)
    report="# 126A technical forensic audit\n\n**PASS_126A_FORENSIC_AUDIT_COMPLETE**\n\nDiagnosis: **"+diagnosis+"**\n\n"
    report+="The historical validation selector chose "+selected_old+"; current jobs8/jobs1 choose "+selected8+"/"+selected1+". All selection uses validation only. Outer-test results are forensic comparisons.\n\n"
    report+=comptable.to_markdown(index=False,floatfmt=".9f")+"\n\n"
    report+="## Input and protocol evidence\n\n"+pd.DataFrame([dict(check=k,value=v) for k,v in flags.items()]).to_markdown(index=False)+"\n\n"
    report+="Identity flags compare the current execution of unchanged 120B source with the 126 path, including independent loading and preprocessing. They are not historical feature hashes. The 120B main source matches its frozen hash, and all frozen OOF arrays reproduce their stored metrics. Historical preprocessing/dependency snapshots and candidate validation predictions are unavailable.\n\n"
    report+="## Determinism\n\n"+threads.to_markdown(index=False,floatfmt=".9g")+"\n\n"
    report+="All same-thread repeat comparisons are in REPEATED_RUN_DETERMINISM.csv (three additional consecutive fits per candidate, plus the first jobs1 fit).\n\n"
    report+="## Historical runtime limits\n\nThe historical search includes all accessible project logs, environment lists and manifests. Other experiments include XGBoost versions, but none establishes the version loaded by the 120B process. See HISTORICAL_RUNTIME_SEARCH.json for exact lines and inaccessible paths.\n\n"
    if diagnosis=="LIKELY_SOFTWARE_RUNTIME_DRIFT":
        report+="The remaining evidence is consistent with software/runtime drift, most plausibly XGBoost-version or implementation drift, although the exact historical cause cannot be proven because the 120B software version was not recorded. This is a provisional diagnosis: historical feature/dependency byte identity cannot be established, and contextual same-environment version records must not be ignored. XGBoost 3.2.0 is not proven causal.\n\n"
    report+="## Scope and next step\n\nThis evidence permits drafting a prospective 126B technical amendment describing the reproducibility limitation and fixed current runtime; it does not establish historical equivalence, relax tolerances, approve deployment of an amendment, or justify a result-based rescue. No source, method, fold, seed, parameter, environment package, or replication tolerance was changed. No Reviewer Comment 3/4 five-method scientific baseline was run.\n"
    (OUT/"126A_DIAGNOSTIC_REPORT.md").write_text(report,encoding="utf-8")
    print(decision["status"])
    print("DIAGNOSIS="+diagnosis)
    for k,v in flags.items(): print(f"{k}={json.dumps(v)}")
    print(f"OUTPUT={OUT}")


def summarize_saved():
    """Recover only reporting after all 15 fits; no model.fit is reachable here."""
    a=load("audit120b_reporting",S120)
    manifest=json.loads((OUT/"126A_RUN_MANIFEST.json").read_text(encoding="utf-8"))
    eq=pd.read_csv(OUT/"IMPLEMENTATION_EQUIVALENCE.csv")
    seedinfo=json.loads((OUT/"FOLD1_SEED_AUDIT.json").read_text()); seeds=seedinfo["candidates"]
    paramdiff=json.loads((OUT/"XGB_120B_VS_126_PARAM_DIFF.json").read_text())
    for cell in paramdiff.values():
        pa,pb=cell["params120B"],cell["params126"]
        cell["differences"]={k:[pa.get(k),pb.get(k)] for k in set(pa)|set(pb) if not parameter_equal(pa.get(k),pb.get(k))}
        cell["comparison_semantics"]="Paired NaN missing-value sentinels count as equal; all other parameter values compared exactly"
    write("XGB_120B_VS_126_PARAM_DIFF.json",paramdiff)
    history=json.loads((OUT/"HISTORICAL_RUNTIME_SEARCH.json").read_text(encoding="utf-8"))
    labels=json.loads((OUT/"FOLD1_LABEL_HASHES.json").read_text())
    data={k:np.tile(np.array([1,0,0],dtype=np.int8),v["shape"][0]//3) for k,v in labels.items()}
    for k,v in labels.items():
        if summary(data[k])["SHA256"]!=v["SHA256"]: raise RuntimeError("Saved ordered labels hash mismatch")
    # Resolve the formal pair tensor via the unchanged loader; isolate its report output.
    import scripts.TRAIN_SEHC_V2_TEMPLATE as template
    original_import=template._import_file
    def local_import(name,path):
        module=original_import(name,path)
        if Path(path)==template.BASE06_RUNNER: module.base.OUT=OUT/"LOADER_PREFLIGHT"
        return module
    template._import_file=local_import
    bundle,runner,base=a.load_dataset_loader()
    template._import_file=original_import
    with np.load(PARENT/"FOLD1_FROZEN_SPLITS.npz") as sp:
        left={"te_idx":np.asarray(bundle.pt)[sp["test"]].reshape(-1)}
    predictions={}; results={}
    for cfg in a.XGB_CANDIDATES:
        cid=cfg["config_id"]
        for jobs,repeat in [(8,0),(1,0),(1,1),(1,2),(1,3)]:
            tag=f"FOLD1_{cid}_NJOBS{jobs}"+(f"_REPEAT{repeat}" if repeat else "")
            val=np.load(OUT/(tag+"_VAL_SCORE.npy")); test=np.load(OUT/(tag+"_TEST_SCORE.npy"))
            predictions[(cid,jobs,repeat)]=(val,test)
            vm=a.metrics_from_ordered_rows(val,data["yva"]); tm=a.metrics_from_ordered_rows(test,data["yte"])
            results[(cid,jobs,repeat)]=dict(config_id=cid,n_jobs=jobs,repeat=repeat,
                actual_seed=seeds[cid]["seed120B"],**{"validation_"+k:v for k,v in vm.items()},
                validation_score=a.val_scalar(vm),**{"test_"+k:v for k,v in tm.items()})
    csv("ALL_TECHNICAL_FIT_METRICS.csv",list(results.values()))
    ref_manifest=json.loads((REF/"RUN_MANIFEST.json").read_text())
    historical_source_identical=sha(S120)==ref_manifest["source_sha256"]
    protocol_identical=True  # Resolved function-source audit already saved before fits.
    weight_equal=json.loads((OUT/"FOLD1_WEIGHT_AUDIT.json").read_text())["comparison"]["array_equal"]
    protected_hashes=manifest["protected_hashes"]
    drift=not(historical_source_identical and eq.array_equal.all() and eq.bytes_equal.all() and weight_equal and
              all(v["seed120B"]==v["seed126"] for v in seeds.values()) and not any(v["differences"] for v in paramdiff.values()))
    score_hashes={p.name:sha(p) for p in OUT.glob("FOLD1_*_SCORE.npy")}
    write("TECHNICAL_REPORTING_RECOVERY.json",dict(reason="Native process exit during scipy.stats.spearmanr; reproduced in isolated score-only process. No Python traceback was emitted.",
        correction="Pearson via centered elementwise sums; Spearman via average ranks and the same Pearson coefficient. No p-values requested or computed.",
        source_before=sha(OUT/"126A_SOURCE_BEFORE_CORRELATION_FIX.py"),source_after=sha(__file__),
        scores_before=score_hashes,model_fits_repeated=0,scientific_protocol_changed=False))
    # Execute the original reporting tail, not the data preparation or fit loops.
    tree=ast.parse(Path(__file__).read_text(encoding="utf-8"))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.FunctionDef) and n.name=="predcompare")
    env=dict(globals()); env.update(locals())
    exec(compile(ast.Module(body=fn.body[start:],type_ignores=[]),str(__file__),"exec"),env)
    if any(sha(OUT/name)!=h for name,h in score_hashes.items()): raise RuntimeError("Score files changed during reporting")


if __name__=="__main__":
    try:
        if sys.argv[1:]==["--summarize-only"]: summarize_saved()
        elif sys.argv[1:]: raise RuntimeError("Only --summarize-only is supported")
        else: main()
    except Exception as exc:
        if OUT.exists(): write("126A_FAILURE.json",dict(error=str(exc),traceback=traceback.format_exc()))
        raise
