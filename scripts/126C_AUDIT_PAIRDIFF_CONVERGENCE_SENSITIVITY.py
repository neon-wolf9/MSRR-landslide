#!/usr/bin/env python
"""Post-hoc technical convergence sensitivity for frozen 126B PairDiff."""
from __future__ import annotations
import os
os.environ.update(MKL_THREADING_LAYER="SEQUENTIAL",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
import gc, hashlib, importlib.util, json, random, sys, time, traceback, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT = Path(__file__).resolve().parents[1]
SRC=ROOT/"experiments/MSRR_TASK_ALIGNED_BASELINES_5FOLD_V1B"
OUT=SRC/"126C_PAIRDIFF_CONVERGENCE_AUDIT"
S126B=ROOT/"scripts/126B_RUN_TASK_ALIGNED_BASELINES_SINGLE_THREAD_ONE_SHOT.py"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("frozen126b",S126B); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
formal=b.formal; old126=b.impl; boot=b.impl.boot121
CS=[0.1,1.0,10.0]; STAGES=[6000,12000]; B=10000; METRICS=formal.METRICS4
AUDIT={}; PROTECTED={}

def sha(p): return boot.sha256_file(Path(p))
def write(name,obj): formal.jwrite(OUT/name,obj)
def csv(name,rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows); d.to_csv(OUT/name,index=False,encoding="utf-8-sig"); return d
def log(s):
    line=time.strftime("[%Y-%m-%d %H:%M:%S] ")+s; print(line,flush=True)
    with (OUT/"RUN.log").open("a",encoding="utf-8") as f:f.write(line+"\n")
def check(x,code):
    if not x:raise RuntimeError(code)
def smetrics(score,y): return old126.score_metrics(score,y)
def safe_corr(a,c):
    a=np.asarray(a,float);c=np.asarray(c,float);x=a-a.mean();y=c-c.mean()
    return float(np.sum(x*y)/np.sqrt(np.sum(x*x)*np.sum(y*y)))
def ranks(x): return pd.Series(x).rank(method="average").to_numpy(float)

def fit_candidate(hf,cap,c,ci,xtr,ytr,xva,yva,xte,seed):
    model=LogisticRegression(C=c,penalty="elasticnet",l1_ratio=.5,solver="saga",max_iter=cap,
        fit_intercept=False,tol=1e-4,random_state=seed+ci*31,n_jobs=1,verbose=1)
    log(f"FOLD{hf} CAP={cap} C={c:g} START; native SAGA epoch progress follows")
    t=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning);model.fit(xtr,ytr)
    conv=any(issubclass(w.category,ConvergenceWarning) for w in caught)
    va=model.decision_function(xva).astype(float);te=model.decision_function(xte).astype(float)
    m=smetrics(va,yva);coef=float(np.linalg.norm(model.coef_))
    tag=f"FOLD{hf}_MAXITER{cap}_C{c:g}".replace(".","p")
    np.save(OUT/(tag+"_VAL_score.npy"),va);np.save(OUT/(tag+"_TEST_score.npy"),te);np.save(OUT/(tag+"_coef.npy"),model.coef_)
    rec=dict(human_fold=hf,C=c,max_iter=cap,n_iter=int(model.n_iter_.max()),convergence_warning=conv,
             coef_norm=coef,**m,validation_score=formal.val_scalar(m),fit_seconds=time.perf_counter()-t,
             seed=seed+ci*31,n_jobs=1,solver="saga",penalty="elasticnet",l1_ratio=.5,tol=1e-4,fit_intercept=False)
    log(f"FOLD{hf} CAP={cap} C={c:g} COMPLETE n_iter={rec['n_iter']} warning={conv} val={rec['validation_score']:.6f}")
    return rec

def bootstrap_c6(msrr,pairdiff,y,pairs):
    msrr_prob=formal.sigmoid(msrr); pd_prob=pairdiff
    ms,me=boot.pair_success_arrays(msrr,pairs);ps,pe=boot.pair_success_arrays(pairdiff,pairs)
    rowpair=np.empty(len(y),int);rowpair[pairs.ravel()]=np.repeat(np.arange(len(pairs)),3)
    obs_m=formal.metrics_from_full_oof(msrr,y,pairs);obs_p=smetrics(pairdiff[pairs.ravel()],y[pairs.ravel()])
    draws=np.empty((B,4));rng=np.random.default_rng(boot.BOOTSTRAP_SEED)
    for i in range(B):
        count=np.bincount(rng.integers(0,len(pairs),len(pairs)),minlength=len(pairs)).astype(float);w=count[rowpair]
        draws[i]=[roc_auc_score(y,msrr_prob,sample_weight=w)-roc_auc_score(y,pd_prob,sample_weight=w),
          average_precision_score(y,msrr_prob,sample_weight=w)-average_precision_score(y,pd_prob,sample_weight=w),
          np.dot(count,ms-ps)/len(pairs),np.dot(count,me-pe)/(2*len(pairs))]
        if (i+1)%500==0:log(f"C6 BOOTSTRAP {i+1}/{B}")
    np.save(OUT/"C6_BOOTSTRAP_DRAWS.npy",draws)
    rows=[]
    for j,k in enumerate(METRICS):
        lo,hi=np.quantile(draws[:,j],[.025,.975]);rows.append(dict(metric=k,observed_delta=obs_m[k]-obs_p[k],
            bootstrap_mean=draws[:,j].mean(),CI95_LOW=lo,CI95_HIGH=hi,fraction_delta_gt_0=(draws[:,j]>0).mean(),B=B,seed=boot.BOOTSTRAP_SEED))
    return csv("C6_CONVERGENCE_BOOTSTRAP.csv",rows),obs_m,obs_p

def main():
    check(not OUT.exists() or not any(OUT.iterdir()),"FAIL_126C_OUTPUT_EXISTS")
    OUT.mkdir(parents=True,exist_ok=True)
    global PROTECTED
    PROTECTED={str(p):sha(p) for p in SRC.rglob("*") if p.is_file() and OUT not in p.parents}
    check(json.loads((SRC/"GATE126B_DECISION.json").read_text())["status"]=="PASS_126B_TASK_ALIGNED_BASELINES_COMPLETE","FAIL_126C_126B_NOT_COMPLETE")
    rules={"stage2":"Run all 5 folds x all 3 C from scratch at max_iter=12000 iff any stage1 candidate has convergence_warning=True; performance is irrelevant",
      "final_cap":"6000 if every stage1 candidate converged, otherwise 12000; never exceed 12000",
      "A":"ROBUST_NO_SELECTION_CHANGE iff all final candidates converged, 5/5 selections unchanged, and C6 4/4 observed>0 and CI low>0",
      "B":"ROBUST_SELECTION_CHANGED_BUT_CONCLUSION_UNCHANGED iff >=1 selection changed and C6 4/4 observed>0 and CI low>0",
      "C":"PAIRDIFF_RESULT_AFFECTS_REVIEWER_CONCLUSION iff any C6 observed<=0 or CI low<=0",
      "D":"PAIRDIFF_CONVERGENCE_UNRESOLVED iff any candidate still warns at 12000; D takes precedence"}
    manifest=dict(audit_type="post-hoc technical convergence sensitivity",reason="C=10 reached max_iter=3000 with convergence_warning=True in all five 126B folds",
      original_max_iter=3000,audit_max_iter_stage1=6000,audit_max_iter_stage2=12000,
      scientific_method_changed=False,solver_changed=False,tol_changed=False,C_grid_changed=False,fold_changed=False,
      representation_changed=False,**{"126B_outputs_modified":False},decision_rules=rules,C_grid=CS,solver="saga",penalty="elasticnet",
      l1_ratio=.5,tol=1e-4,fit_intercept=False,n_jobs=1,seed_rule="126B fold_seed+100 then ci*31",B=B,
      bootstrap_unit="complete matched set",MSRR_POINTWISE_source=str(SRC/"MSRR_POINTWISE_XGB_OOF_score.npy"),
      source_sha256=sha(__file__),source_126B_sha256=sha(S126B),protected_126B_hashes=PROTECTED,
      scientific_results_seen_before_manifest=False,post_result_rescue_allowed=False,created=time.strftime("%Y-%m-%d %H:%M:%S"))
    write("126C_RUN_MANIFEST.json",manifest);manifest_hash=sha(OUT/"126C_RUN_MANIFEST.json")
    formal.log=log
    # Redirect legacy preflight report only; loader/splits/preprocessing unchanged.
    import scripts.TRAIN_SEHC_V2_TEMPLATE as template
    oi=template._import_file
    def local_import(name,path):
        m=oi(name,path)
        if Path(path)==template.BASE06_RUNNER:m.base.OUT=OUT/"LOADER_PREFLIGHT"
        return m
    template._import_file=local_import
    bundle,runner,base=formal.load_dataset_loader();template._import_file=oi
    y=bundle.sample.y_pair.to_numpy(np.int8);pairs=np.asarray(bundle.pt,dtype=np.int64)
    check(len(y)==15168 and pairs.shape==(5056,3),"FAIL_126C_BENCHMARK")
    formal.verify_pair_order(pairs,y,"GLOBAL")
    folddata={};feature_audit=[]
    for hf in formal.FOLDS:
        fd=formal.build_fold_loader(bundle,runner,base,hf,torch.device("cpu"));ids=[np.asarray(v,int) for v in (fd.train_pairs,fd.validation_pairs,fd.test_pairs)]
        rows=[pairs[i].reshape(-1) for i in ids]
        with np.load(SRC/f"FOLD{hf}_FROZEN_SPLITS.npz") as old:check(all(np.array_equal(v,old[k]) for v,k in zip(ids,("train","validation","test"))),"FAIL_126C_SPLIT_DRIFT")
        x=[formal.raw_features(pairs[i],formal.as_numpy(fd.static92).astype(np.float32),formal.as_numpy(fd.rain).astype(np.float32))[0] for i in ids]
        mu,sd=formal.fit_standardizer(x[0]);old=np.load(SRC/f"FOLD{hf}_PAIRDIFF_STANDARDIZER.npz")
        check(np.array_equal(mu,old["mu"]) and np.array_equal(sd,old["sd"]),"FAIL_126C_RAW_OR_STANDARDIZER_DRIFT")
        z=[formal.apply_standardizer(xx,mu,sd).astype(np.float64) for xx in x]
        tri=z[0].reshape(-1,3,792);d1,d2=tri[:,0]-tri[:,1],tri[:,0]-tri[:,2]
        xp=np.stack([d1,d2,-d1,-d2],axis=1).reshape(-1,792);yp=np.tile([1,1,0,0],len(tri))
        feature_audit.append(dict(human_fold=hf,train_raw_sha256=hashlib.sha256(np.ascontiguousarray(x[0]).tobytes()).hexdigest(),
          validation_raw_sha256=hashlib.sha256(np.ascontiguousarray(x[1]).tobytes()).hexdigest(),test_raw_sha256=hashlib.sha256(np.ascontiguousarray(x[2]).tobytes()).hexdigest(),
          standardizer_matches_126B=True,split_matches_126B=True,raw_dim=792))
        folddata[hf]=(xp,yp,z[1],y[rows[1]],z[2],y[rows[2]],rows[2]);del fd,x,z,tri,d1,d2;gc.collect()
    csv("FOLD_RAW_FEATURE_AUDIT.csv",feature_audit)
    stages={}
    for cap in STAGES:
        if cap==12000 and not stages[6000].convergence_warning.any():
            write("STAGE2_STATUS.json",{"status":"STAGE2_NOT_REQUIRED_ALL_CONVERGED_AT_6000"});break
        records=[]
        for hf in formal.FOLDS:
            xtr,ytr,xva,yva,xte,yte,teidx=folddata[hf];base_seed=formal.SEED+hf*1000+100
            for ci,c in enumerate(CS):
                records.append(fit_candidate(hf,cap,c,ci,xtr,ytr,xva,yva,xte,base_seed));csv(f"STAGE{1 if cap==6000 else 2}_MAXITER{cap}_VALIDATION.csv",records)
        stages[cap]=pd.DataFrame(records)
    finalcap=6000 if 12000 not in stages else 12000;final=stages[finalcap]
    original=pd.concat([pd.read_csv(SRC/f"FOLD{hf}_RAW_PAIRDIFF_LOGIT_SELECTION.csv") for hf in formal.FOLDS],ignore_index=True)
    original["C"]=original.config_id.str.replace("C=","",regex=False).astype(float)
    comparison=[];stability=[];oof=np.full(15168,np.nan);cover=np.zeros(15168,np.int8);identity=[]
    for hf in formal.FOLDS:
        old=original[original.human_fold==hf];new=final[final.human_fold==hf]
        oi=formal.choose_best_index(old.to_dict("records"));ni=formal.choose_best_index(new.to_dict("records"));oc=float(old.iloc[oi].C);nc=float(new.iloc[ni].C)
        stability.append(dict(human_fold=hf,original_selected_C=oc,convergence_safe_selected_C=nc,selection_changed=oc!=nc,
          final_selected_converged=not bool(new.iloc[ni].convergence_warning),final_audit_cap=finalcap))
        for c in CS:
            a=old[np.isclose(old.C,c)].iloc[0];q=new[np.isclose(new.C,c)].iloc[0]
            comparison.append(dict(human_fold=hf,C=c,original_max_iter=3000,original_n_iter=a.n_iter,original_convergence_warning=a.convergence_warning,
              **{"original_"+k:a[k] for k in METRICS},original_validation_score=a.validation_score,audit_max_iter=finalcap,audit_n_iter=q.n_iter,
              audit_convergence_warning=q.convergence_warning,**{"audit_"+k:q[k] for k in METRICS},audit_validation_score=q.validation_score,
              delta_validation_score=q.validation_score-a.validation_score))
        tag=f"FOLD{hf}_MAXITER{finalcap}_C{nc:g}".replace(".","p");score=np.load(OUT/(tag+"_TEST_score.npy"));coef=np.load(OUT/(tag+"_coef.npy"))
        zte=folddata[hf][4];s=score.reshape(-1,3);zt=zte.reshape(-1,3,792);errs=[]
        for k in (1,2):errs.append(np.max(np.abs(s[:,0]-s[:,k]-(zt[:,0]-zt[:,k])@coef.ravel())))
        err=float(max(errs));check(err<1e-10,"FAIL_126C_PAIRDIFF_SCORE_IDENTITY");identity.append(dict(human_fold=hf,max_abs_error=err))
        idx=folddata[hf][6];oof[idx]=score;cover[idx]+=1
    csv("ORIGINAL_VS_CONVERGED_VALIDATION.csv",comparison);stab=csv("PAIRDIFF_SELECTION_STABILITY.csv",stability)
    check(np.all(cover==1) and np.isfinite(oof).all(),"FAIL_126C_OOF")
    np.save(OUT/"RAW_PAIRDIFF_LOGIT_CONVERGENCE_AUDIT_OOF_score.npy",oof);margin=oof[pairs[:,0]]-oof[pairs[:,1:]].max(axis=1)
    np.save(OUT/"RAW_PAIRDIFF_LOGIT_CONVERGENCE_AUDIT_OOF_M.npy",margin)
    original_score=np.load(SRC/"RAW_PAIRDIFF_LOGIT_OOF_score.npy");oldm=smetrics(original_score[pairs.ravel()],y[pairs.ravel()]);newm=smetrics(oof[pairs.ravel()],y[pairs.ravel()])
    pooled=csv("PAIRDIFF_POOLED_SENSITIVITY.csv",[dict(metric=k,original126B=oldm[k],convergence_audit=newm[k],delta=newm[k]-oldm[k]) for k in METRICS])
    scorestab=[]
    for hf in formal.FOLDS:
        idx=folddata[hf][6];a=original_score[idx];q=oof[idx];pa=pairs[np.asarray(np.load(SRC/f"FOLD{hf}_FROZEN_SPLITS.npz")["test"],int)]
        ma=original_score[pa[:,0]]-original_score[pa[:,1:]].max(axis=1);mq=oof[pa[:,0]]-oof[pa[:,1:]].max(axis=1);d=mq-ma
        scorestab.append(dict(human_fold=hf,Pearson=safe_corr(a,q),Spearman=safe_corr(ranks(a),ranks(q)),max_abs_score_diff=np.max(np.abs(q-a)),
          mean_abs_score_diff=np.mean(np.abs(q-a)),median_margin_change=np.median(d),mean_margin_change=np.mean(d),fraction_positive_change=np.mean(d>0)))
    csv("PAIRDIFF_SCORE_STABILITY.csv",scorestab)
    msrr=np.load(SRC/"MSRR_POINTWISE_XGB_OOF_score.npy");bs,msrrm,pdm=bootstrap_c6(msrr,oof,y,pairs)
    c6=csv("C6_CONVERGENCE_SENSITIVITY.csv",[dict(metric=k,MSRR_POINTWISE_XGB=msrrm[k],PairDiff_convergence_audit=pdm[k],delta=msrrm[k]-pdm[k]) for k in METRICS])
    allconv=not bool(final.convergence_warning.any());changed=int(stab.selection_changed.sum());positive=bool((c6.delta>0).all());cis=bool((bs.CI95_LOW>0).all())
    if not allconv:gate="PAIRDIFF_CONVERGENCE_UNRESOLVED"
    elif positive and cis and changed==0:gate="ROBUST_NO_SELECTION_CHANGE"
    elif positive and cis and changed>0:gate="ROBUST_SELECTION_CHANGED_BUT_CONCLUSION_UNCHANGED"
    else:gate="PAIRDIFF_RESULT_AFFECTS_REVIEWER_CONCLUSION"
    check(sha(OUT/"126C_RUN_MANIFEST.json")==manifest_hash,"FAIL_126C_MANIFEST_CHANGED")
    check(all(sha(p)==h for p,h in PROTECTED.items()),"FAIL_126C_126B_OUTPUT_CHANGED")
    AUDIT.update(status="PASS_126C_PAIRDIFF_CONVERGENCE_AUDIT_COMPLETE",five_folds_complete=True,matched_sets=5056,candidate_rows=15168,
      train_validation_test_folds_identical_to_126B=True,RAW_features_identical_to_126B=True,standardizer_train_only=True,C_grid_unchanged=True,
      solver_unchanged=True,penalty_unchanged=True,l1_ratio_unchanged=True,tol_unchanged=True,fit_intercept=False,n_jobs=1,seed_unchanged=True,
      only_max_iter_changed=True,outer_test_not_used_for_selection=True,PairDiff_candidate_score_identity="PASS",identity_checks=identity,
      **{"126B_outputs_unchanged":True},MSRR_POINTWISE_reused_frozen_126B_predictions=True,bootstrap_unit="matched set",bootstrap_B=B,
      final_audit_cap=finalcap,all_selected_models_converged=allconv,all_candidates_converged=allconv,selection_changed_folds=stab[stab.selection_changed].human_fold.tolist(),post_result_rescue=False)
    write("AUDIT126C.json",AUDIT);decision=dict(status=AUDIT["status"],GATE126C_DECISION=gate,FINAL_AUDIT_MAX_ITER=finalcap,
      ALL_SELECTED_MODELS_CONVERGED=bool(stab.final_selected_converged.all()),ALL_CANDIDATES_CONVERGED=allconv,N_FOLDS_SELECTION_CHANGED=changed,
      C6_ALL_OBSERVED_DELTAS_GT_0=positive,C6_ALL_CI95_LOW_GT_0=cis,reviewer_comment_3_pairdiff_convergence_sensitivity_complete=True,
      **{"126B_outputs_modified":False});write("GATE126C_DECISION.json",decision)
    report="# Experiment 126C: PairDiff convergence sensitivity\n\n**"+AUDIT["status"]+"**\n\nDecision: **"+gate+"**\n\n"
    report+=stab.to_markdown(index=False)+"\n\n## Pooled PairDiff sensitivity\n\n"+pooled.to_markdown(index=False,floatfmt=".6f")
    report+="\n\n## C6 convergence sensitivity\n\n"+c6.to_markdown(index=False,floatfmt=".6f")+"\n\n"+bs.to_markdown(index=False,floatfmt=".6f")
    report+="\n\nStage 2 was triggered only by Stage 1 convergence warnings. Selection used validation metrics only. The frozen 126B MSRR pointwise scores were reused; no MSRR or other 126B method was retrained. No 126B file was modified.\n"
    (OUT/"GATE126C_REPORT.md").write_text(report,encoding="utf-8")
    print("PASS_126C_PAIRDIFF_CONVERGENCE_AUDIT_COMPLETE");print(f"FINAL_AUDIT_MAX_ITER={finalcap}");print(f"ALL_SELECTED_MODELS_CONVERGED={bool(stab.final_selected_converged.all())}");print(f"N_FOLDS_SELECTION_CHANGED={changed}")
    for r in stab.itertuples():print(f"FOLD{r.human_fold} original_C={r.original_selected_C:g} audit_C={r.convergence_safe_selected_C:g}")
    for label,m in (("PAIRDIFF_ORIGINAL",oldm),("PAIRDIFF_CONVERGENCE_AUDIT",newm)):
        print(label+":");[print(f"{k}={m[k]:.9f}") for k in METRICS]
    print("C6_CONVERGENCE_AUDIT:");[print(f"{r.metric} delta={r.observed_delta:.9f} CI=[{r.CI95_LOW:.9f}, {r.CI95_HIGH:.9f}]") for r in bs.itertuples()]
    print(f"GATE126C_DECISION={gate}");print(f"OUTPUT={OUT}")

if __name__=="__main__":
    try:main()
    except Exception as e:
        if OUT.exists():write("126C_FAILURE.json",dict(error=str(e),traceback=traceback.format_exc()))
        raise
