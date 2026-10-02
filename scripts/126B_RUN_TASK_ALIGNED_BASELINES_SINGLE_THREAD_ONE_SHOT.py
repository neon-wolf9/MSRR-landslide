#!/usr/bin/env python
"""Prospective single-thread amendment; contemporaneous frozen five-method run."""
from __future__ import annotations
import os
# Process-local, frozen before numerical library initialization; no package changes.
THREAD_ENV = dict(MKL_THREADING_LAYER="SEQUENTIAL", MKL_NUM_THREADS="1",
                  OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
os.environ.update(THREAD_ENV)
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import msvcrt
import platform
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

ROOT = Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/MSRR_TASK_ALIGNED_BASELINES_5FOLD_V1B"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("frozen126",ROOT/"scripts/126_RUN_TASK_ALIGNED_BASELINES_5FOLD_ONE_SHOT.py")
impl=importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)
formal=impl.formal
REF=impl.REFERENCE
METHODS=impl.METHODS
METRICS=impl.METRICS
CONTRASTS=impl.CONTRASTS
XGB_N_JOBS=1
GATES={"STRONG_REPRESENTATION_BEYOND_OBJECTIVE":"C3 wins >=3 AND C5 wins >=3 AND C6 wins >=3",
       "REPRESENTATION_OBJECTIVE_COMPLEMENTARY":"C3 wins >=3 AND previous strong condition fails",
       "OBJECTIVE_DOMINANT_OR_MIXED":"otherwise"}
AUDIT={}


def check(ok,code):
    if not ok: raise RuntimeError(code)


def sha(p): return impl.boot121.sha256_file(Path(p))


def write(name,value): formal.jwrite(OUT/name,value)


def log(message):
    line=time.strftime("[%Y-%m-%d %H:%M:%S] ")+message
    print(line,flush=True)
    with (OUT/"RUN.log").open("a",encoding="utf-8") as f: f.write(line+"\n")


def csv(name,records):
    frame=records if isinstance(records,pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(OUT/name,index=False,encoding="utf-8-sig")
    return frame


def make_manifest():
    authority=ROOT/"experiments/MSRR_TASK_ALIGNED_BASELINES_5FOLD_V1/126A_REPLICATION_AUDIT/126A_DIAGNOSTIC_DECISION.json"
    diagnosis=json.loads(authority.read_text(encoding="utf-8"))
    check(diagnosis["status"]=="PASS_126A_FORENSIC_AUDIT_COMPLETE" and diagnosis["DIAGNOSIS"]=="THREAD_COUNT_SENSITIVITY",
          "FAIL_126B_AMENDMENT_AUTHORITY")
    protected=[Path(impl.__file__),Path(formal.__file__),Path(impl.boot121.__file__),
               ROOT/"scripts/TRAIN_SEHC_V2_TEMPLATE.py",ROOT/"scripts/126A_AUDIT_FROZEN_XGB_REPLICATION.py"]
    for directory in (REF,ROOT/"experiments/MSRR_121_PAIRED_BOOTSTRAP",ROOT/"experiments/MSRR_TASK_ALIGNED_BASELINES_5FOLD_V1"):
        protected.extend(p for p in directory.rglob("*") if p.is_file())
    source_hashes={str(p):sha(p) for p in protected}
    amendment=dict(reason="126A detected material thread-count sensitivity in current XGBoost predictions.",
        prospective_fix="All XGBoost-based models in Experiment 126B use n_jobs=1.",
        rationale="Single-thread execution is fixed because repeated single-thread runs were deterministic, whereas jobs1 and jobs8 produced materially different predictions under otherwise identical data, folds, seeds, weights and explicit model parameters.",
        scientific_results_seen_before_amendment=False,post_result_rescue=False,
        historical_120B_runtime_reconstructed=False,historical_120B_results_modified=False,
        replication_tolerance_relaxed=False,
        historical_reference_policy="Historical 120B scores are not a pass/fail gate for 126B; original historical replication failure and tolerances remain unchanged. Five methods are refitted contemporaneously.",
        forensic_observation_only="Current single-thread selector and historical selector both chose D6 in fold 1; this coincidence is not the reason for the amendment and does not reconstruct historical scores.",
        process_thread_environment=THREAD_ENV,
        process_runtime_note="Default MKL threading-library loading failed in a pre-scientific matrix-operation smoke check; MKL sequential mode passed. Settings are process-local, fixed before imports and all scientific fitting; no packages changed.",
        gate=GATES,gate_authority="User's prospective 126B specification: complementary if C3 wins >=3 and strong condition fails",
        authority=str(authority),authority_sha256=sha(authority),created=time.strftime("%Y-%m-%d %H:%M:%S"))
    write("126B_TECHNICAL_AMENDMENT.json",amendment)
    (OUT/"126B_TECHNICAL_AMENDMENT.md").write_text("# Prospective 126B technical amendment\n\n"+
        amendment["reason"]+"\n\n"+amendment["rationale"]+"\n\n"+amendment["prospective_fix"]+"\n\n"+
        amendment["historical_reference_policy"]+"\n\n"+amendment["forensic_observation_only"]+"\n\n"+
        amendment["process_runtime_note"]+"\n\nProcess settings: `"+json.dumps(THREAD_ENV)+"`.\n\n"+
        "No task-aligned scientific baseline results were seen before this amendment. No post-result rescue. No historical result or original replication tolerance is changed.\n\n"+
        "The current user-specified gate is frozen in RUN_MANIFEST.json before training. The complementary branch requires C3 wins >=3 and failure of the strong condition.\n",encoding="utf-8")
    mode="qid" if "qid" in inspect.signature(xgb.XGBRanker.fit).parameters else "group"
    manifest=dict(experiment="126B",methods=METHODS,metrics=METRICS,folds=formal.FOLDS,
        dataset="Hiroshima 2018",protocol="PRIMARY_BALANCED_SPATIAL_GROUPED_5FOLD",matched_sets=5056,rows=15168,
        fold_sizes=[1011,1011,1011,1011,1012],pair_order=["P","C1","C2"],RAW_dim=792,MSRR_dim=2446,
        loader="Unchanged 120B load_dataset_loader/build_fold_loader, bundle.pt and fd.train_pairs/validation_pairs/test_pairs",
        features="Imported unchanged 120B raw_features, msrr_features, dynsum7",
        learners="Imported unchanged 126 PairDiff and RankPair functions; imported unchanged 120B pointwise fitter",
        seed=formal.SEED,fold_seed="7+1000*fold",xgb_base_seed="fold_seed+600",xgb_candidate_seed="base_seed+37*candidate_index",
        pairdiff_base_seed="fold_seed+100",pairdiff_candidate_seed="base_seed+31*candidate_index",
        xgb_candidates=formal.XGB_CANDIDATES,xgb_n_jobs=1,xgboost_version=xgb.__version__,
        xgboost_build_info=xgb.build_info() if hasattr(xgb,"build_info") else None,ranking_API_mode=mode,
        thread_policy="prospectively frozen after 126A forensic audit and before scientific baseline results",
        process_thread_environment=THREAD_ENV,
        runtime=dict(Python=sys.version,executable=sys.executable,numpy=np.__version__,pandas=pd.__version__,
                     sklearn=sklearn.__version__,scipy=scipy.__version__,torch=torch.__version__,OS=platform.platform()),
        pointwise_common_params=dict(objective="binary:logistic",eval_metric="logloss",tree_method="hist",max_bin=256,
                                     verbosity=0,n_jobs=1),pointwise_weights=dict(positive=1.0,control=0.5),
        ranking_changes=dict(objective="rank:pairwise",eval_metric="ndcg@3",group_size=3,group_weights="uniform"),
        pairdiff=dict(penalty="elasticnet",l1_ratio=0.5,solver="saga",max_iter=3000,fit_intercept=False,n_jobs=1,
                      tol=1e-4,C=[0.1,1.0,10.0],standardization="train candidate rows only; sd<1e-6 ->1",
                      edges="d1,d2,-d1,-d2; labels 1,1,0,0",score="w^T z"),
        selection="validation mean(AUROC,AUPRC,StrictPair,Edge); ties StrictPair,Edge,AUPRC,AUROC,earliest config",
        score_policy="Unchanged 126: pointwise logit discrimination score evaluated through frozen 120B metric routine; other methods raw scalar scores",
        contrasts=CONTRASTS,bootstrap_B=10000,bootstrap_seed=impl.boot121.BOOTSTRAP_SEED,
        bootstrap_unit="complete matched set",same_bootstrap_draws_all_methods=True,
        gate=GATES,historical_reference_is_descriptive_only=True,historical_reference_used_for_gate=False,
        historical_OOF_used_as_current_predictions=False,outer_test_used_for_selection=False,
        scientific_results_seen_before_technical_amendment=False,post_result_rescue_allowed=False,
        scientific_definition_changes=False,source_sha256=sha(__file__),protected_hashes=source_hashes,
        amendment_sha256=sha(OUT/"126B_TECHNICAL_AMENDMENT.json"),
        technical_resume_policy="Only completed same-manifest fold-method checkpoints may be reused after a technical interruption; no historical predictions, no result-dependent changes")
    write("RUN_MANIFEST.json",manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgb-jobs",type=int,choices=[1],default=1)
    parser.add_argument("--resume-technical",action="store_true",help="Resume same frozen run after technical interruption only")
    args=parser.parse_args()
    existing=OUT.exists() and any(OUT.iterdir())
    check(not existing or args.resume_technical,"FAIL_126B_OUTPUT_EXISTS")
    OUT.mkdir(parents=True,exist_ok=True)
    # OS lock is released even if the process crashes; prevent concurrent instances.
    lock=(OUT/"RUN_LOCK").open("a+b")
    if lock.tell()==0: lock.write(b"0"); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    impl.OUT=OUT; impl.log=log; impl.AUDIT=AUDIT; formal.log=log
    if args.resume_technical:
        manifest=json.loads((OUT/"RUN_MANIFEST.json").read_text(encoding="utf-8"))
        check(manifest["source_sha256"]==sha(__file__),"FAIL_126B_RESUME_SOURCE_CHANGED")
        check(manifest["xgboost_version"]==xgb.__version__ and manifest["process_thread_environment"]==THREAD_ENV,
              "FAIL_126B_RESUME_RUNTIME_CHANGED")
    else: manifest=make_manifest()
    frozen_hashes={name:sha(OUT/name) for name in ("RUN_MANIFEST.json","126B_TECHNICAL_AMENDMENT.json","126B_TECHNICAL_AMENDMENT.md")}
    runtime_checks=[]
    original_make=formal.make_xgb
    def single_thread_make(cfg,seed,jobs):
        check(jobs==1,"FAIL_126B_XGB_THREADS")
        model=original_make(cfg,seed,1)
        check(model.get_params()["n_jobs"]==1,"FAIL_126B_XGB_THREADS")
        runtime_checks.append(dict(config_id=cfg["config_id"],seed=seed,n_jobs=1))
        return model
    formal.make_xgb=single_thread_make
    # Isolate the legacy preflight JSON; all input and feature routines unchanged.
    import scripts.TRAIN_SEHC_V2_TEMPLATE as template
    original_import=template._import_file
    def audit_import(name,path):
        module=original_import(name,path)
        if Path(path)==template.BASE06_RUNNER: module.base.OUT=OUT/"LOADER_PREFLIGHT"
        return module
    template._import_file=audit_import
    formal.set_seed(formal.SEED)
    bundle,runner,base=formal.load_dataset_loader()
    template._import_file=original_import
    y=bundle.sample.y_pair.to_numpy(np.int8); pairs=np.asarray(bundle.pt,dtype=np.int64)
    check(len(y)==int(base.N)==15168 and pairs.shape==(5056,3),"FAIL_126B_BENCHMARK_SIZE")
    check(np.array_equal(np.sort(pairs.ravel()),np.arange(15168)),"FAIL_126B_PAIR_COVERAGE")
    formal.verify_pair_order(pairs,y,"GLOBAL")
    oof={m:np.full(15168,np.nan) for m in METHODS}; coverage={m:np.zeros(15168,dtype=np.int8) for m in METHODS}
    outer=np.zeros(15168,dtype=np.int8)
    fold_records=[]; selection_tables=[]; split_records=[]
    log("START 126B CONTEMPORANEOUS FIVE-METHOD RUN; all XGB/PairDiff n_jobs=1; amendment frozen")
    for hf in formal.FOLDS:
        fold_seed=formal.SEED+hf*1000
        formal.set_seed(fold_seed)
        fd=formal.build_fold_loader(bundle,runner,base,hf,torch.device("cpu"))
        ids=[np.asarray(v,dtype=np.int64) for v in (fd.train_pairs,fd.validation_pairs,fd.test_pairs)]
        rows=[pairs[v].reshape(-1) for v in ids]
        for i in range(3):
            check(len(np.unique(ids[i]))==len(ids[i]),"FAIL_126B_DUPLICATE_SPLIT")
            formal.verify_pair_order(pairs[ids[i]],y,f"FOLD{hf}_SPLIT{i}")
            for j in range(i):
                check(not np.intersect1d(ids[i],ids[j]).size,"FAIL_126B_SET_OVERLAP")
                check(not np.intersect1d(rows[i],rows[j]).size,"FAIL_126B_ROW_OVERLAP")
        check(np.array_equal(np.sort(np.concatenate(ids)),np.arange(5056)),"FAIL_126B_SPLIT_COVERAGE")
        check(len(ids[2])==manifest["fold_sizes"][hf-1],"FAIL_126B_FOLD_SIZE")
        check(np.all(outer[rows[2]]==0),"FAIL_126B_OUTER_OVERLAP")
        outer[rows[2]]=hf
        splitfile=OUT/f"FOLD{hf}_FROZEN_SPLITS.npz"
        if splitfile.exists():
            with np.load(splitfile) as old:
                check(all(np.array_equal(v,old[k]) for v,k in zip(ids,("train","validation","test"))),"FAIL_126B_RESUME_SPLIT_CHANGED")
        else: np.savez(splitfile,train=ids[0],validation=ids[1],test=ids[2])
        split_records.append(dict(human_fold=hf,train=len(ids[0]),validation=len(ids[1]),test=len(ids[2]),sha256=sha(splitfile)))
        static=formal.as_numpy(fd.static92).astype(np.float32,copy=False)
        rain=formal.as_numpy(fd.rain).astype(np.float32,copy=False)
        for rep in ("RAW","MSRR"):
            feature_fn=formal.raw_features if rep=="RAW" else formal.msrr_features
            feats=[feature_fn(pairs[v],static,rain) for v in ids]
            check(all(np.array_equal(a,r) for (_,a),r in zip(feats,rows)),"FAIL_126B_FEATURE_ORDER")
            xtr,xva,xte=[x for x,a in feats]
            check(all(x.shape[1]==(792 if rep=="RAW" else 2446) for x,a in feats),"FAIL_126B_FEATURE_DIM")
            common=(xtr,y[rows[0]],xva,y[rows[1]],xte)
            todo=[f"{rep}_POINTWISE_XGB"]+(["RAW_PAIRDIFF_LOGIT"] if rep=="RAW" else [])+[f"{rep}_RANKPAIR_XGB"]
            for method in todo:
                prefix=f"FOLD{hf}_{method}"
                checkpoint=OUT/(prefix+"_COMPLETE.json")
                scorefile=OUT/(prefix+"_score.npy")
                selfile=OUT/(prefix+"_SELECTION.csv")
                if args.resume_technical and checkpoint.exists():
                    saved=json.loads(checkpoint.read_text())
                    check(saved["manifest_sha256"]==frozen_hashes["RUN_MANIFEST.json"] and saved["score_sha256"]==sha(scorefile),"FAIL_126B_CHECKPOINT_HASH")
                    score=np.load(scorefile); sel=pd.read_csv(selfile); cfg=saved["selected_config"]
                    if "pairdiff_identity" in saved: AUDIT.setdefault("pairdiff_identity",[]).append(saved["pairdiff_identity"])
                    log(f"RESUME VERIFIED {prefix}")
                else:
                    log(f"START {prefix}")
                    if "POINTWISE" in method:
                        score,sel,cfg=formal.fit_xgb_select_predict(rep,hf,*common,fold_seed+600,1)
                    elif "PAIRDIFF" in method:
                        score,sel,cfg=impl.pairdiff(*common,fold_seed+100,hf,1)
                    else:
                        score,sel,cfg=impl.rankpair(rep,hf,*common,fold_seed+600,1,manifest["ranking_API_mode"])
                    np.save(scorefile,score)
                    sel["method"]=method; sel["human_fold"]=hf; sel["selected"]=sel.config_id==cfg
                    csv(selfile.name,sel)
                    saved=dict(manifest_sha256=frozen_hashes["RUN_MANIFEST.json"],score_sha256=sha(scorefile),selected_config=cfg,n_jobs=1)
                    if "PAIRDIFF" in method: saved["pairdiff_identity"]=AUDIT["pairdiff_identity"][-1]
                    write(checkpoint.name,saved)
                check(score.shape==rows[2].shape and np.isfinite(score).all(),"FAIL_126B_INVALID_SCORES")
                check(np.all(coverage[method][rows[2]]==0),"FAIL_126B_DUPLICATE_OOF")
                oof[method][rows[2]]=score; coverage[method][rows[2]]+=1
                evaluator=formal.metrics_from_ordered_rows if "POINTWISE" in method else impl.score_metrics
                fm=evaluator(score,y[rows[2]])
                fold_records.append(dict(method=method,human_fold=hf,selected_config=cfg,**fm))
                selection_tables.append(sel)
                csv("FOLD_METRICS.csv",fold_records)
                csv("VALIDATION_SELECTION.csv",pd.concat(selection_tables,ignore_index=True))
                write("TRAINING_PROGRESS.json",dict(completed_cells=len(fold_records),total_cells=25,last_completed=prefix))
                log(f"COMPLETE {prefix} selected={cfg} ({len(fold_records)}/25)")
                gc.collect()
            del feats,xtr,xva,xte,common
            gc.collect()
        del fd,static,rain
        gc.collect()
    check(all(np.all(c==1) for c in coverage.values()),"FAIL_126B_OOF_COVERAGE")
    check(all(np.isfinite(s).all() for s in oof.values()),"FAIL_126B_NONFINITE_OOF")
    row_pair=np.empty(15168,dtype=np.int64); row_pair[pairs.ravel()]=np.repeat(np.arange(5056),3)
    predictions=pd.DataFrame(dict(row_index=np.arange(15168),matched_set_index=row_pair,human_fold=outer,y_pair=y))
    observed={}
    for method in METHODS:
        score=oof[method]
        observed[method]=(formal.metrics_from_full_oof(score,y,pairs) if "POINTWISE" in method else
                          impl.score_metrics(score[pairs.ravel()],y[pairs.ravel()]))
        predictions[method+"_score"]=score
        np.save(OUT/f"{method}_OOF_score.npy",score)
        np.save(OUT/f"{method}_OOF_M.npy",score[pairs[:,0]]-score[pairs[:,1:]].max(axis=1))
        np.save(OUT/f"{method}_OOF_edge_margins.npy",score[pairs[:,:1]]-score[pairs[:,1:]])
    csv("OOF_PREDICTIONS.csv",predictions)
    results=csv("OOF_RESULTS.csv",[dict(method=m,**observed[m]) for m in METHODS])
    matrix=results.copy()
    matrix["Objective"]=["pointwise","pairwise-difference","rank:pairwise","pointwise","rank:pairwise"]
    matrix["Representation"]=["RAW","RAW difference","RAW","MSRR","MSRR"]
    csv("REPRESENTATION_OBJECTIVE_MATRIX.csv",matrix)
    contrasts=csv("KEY_COMPARISONS.csv",[dict(contrast=cid,description=desc,method_a=a,method_b=b,
        **{"delta_"+k:observed[a][k]-observed[b][k] for k in METRICS}) for cid,a,b,desc in CONTRASTS])
    historical=pd.read_csv(REF/"OOF_RESULTS.csv")
    ref_rows=[]
    for rep in ("RAW","MSRR"):
        old=historical[(historical.backbone=="XGBoost")&(historical.representation==rep)].iloc[0]
        method=rep+"_POINTWISE_XGB"
        for k in METRICS:
            ref_rows.append(dict(method=method,metric=k,historical_120B_metric=old[k],current_126B_metric=observed[method][k],
                                 difference=observed[method][k]-old[k],used_for_pass_fail=False))
    reference=csv("126B_VS_120B_REFERENCE.csv",ref_rows)
    log("All 25 fold-method fits complete. Starting fixed 10000 complete-matched-set bootstrap draws.")
    bs=impl.bootstrap(oof,y,pairs,observed)
    wins={cid:sum(observed[a][k]>observed[b][k] for k in METRICS) for cid,a,b,_ in CONTRASTS}
    if wins["C3"]>=3 and wins["C5"]>=3 and wins["C6"]>=3: gate="STRONG_REPRESENTATION_BEYOND_OBJECTIVE"
    elif wins["C3"]>=3: gate="REPRESENTATION_OBJECTIVE_COMPLEMENTARY"
    else: gate="OBJECTIVE_DOMINANT_OR_MIXED"
    check(all(sha(p)==h for p,h in manifest["protected_hashes"].items()),"FAIL_126B_PROTECTED_FILE_CHANGED")
    check(all(sha(OUT/p)==h for p,h in frozen_hashes.items()),"FAIL_126B_PREREGISTRATION_CHANGED")
    check(len(AUDIT.get("pairdiff_identity",[]))==5,"FAIL_126B_PAIRDIFF_AUDIT_MISSING")
    AUDIT.update(status="PASS_126B_TASK_ALIGNED_BASELINES_COMPLETE",matched_sets=5056,rows=15168,positive=5056,controls=10112,
        pair_ordering_correct=True,RAW_dim=792,MSRR_dim=2446,five_folds_complete=True,
        no_train_val_test_row_overlap=True,no_matched_set_overlap=True,splits=split_records,
        train_only_standardization=True,all_XGB_n_jobs_1=True,xgb_constructor_checks=runtime_checks,
        PairDiff_n_jobs=1,PairDiff_fit_intercept=False,PairDiff_score_consistency="PASS",
        ranker_qid_not_predictor=True,pair_set_id_not_predictor=True,sample_role_not_predictor=True,control_rank_not_predictor=True,
        outer_test_never_selects_config=True,all_OOF_finite=True,one_OOF_score_per_candidate_per_method=True,
        all_methods_contemporaneous=True,historical_reference_is_descriptive_only=True,all_protected_files_unchanged=True,
        preregistration_unchanged=True,scientific_results_seen_before_technical_amendment=False,post_result_rescue_allowed=False)
    write("AUDIT126B.json",AUDIT)
    decision=dict(status=AUDIT["status"],GATE126B_DECISION=gate,representation_pairwise_wins=wins["C3"],
        msrr_pointwise_vs_raw_rank_wins=wins["C5"],pairdiff_beaten=wins["C6"],all_contrast_win_counts=wins,
        xgboost_version=xgb.__version__,xgb_n_jobs=1,reviewer_comment_3_experiment_complete=True,
        reviewer_comment_4_experiment_complete=True,post_result_rescue=False)
    write("GATE126B_DECISION.json",decision)
    report="# Experiment 126B: task-aligned baselines\n\n"+AUDIT["status"]+"\n\n"
    report+="All five methods were refitted in the same current runtime with n_jobs=1. The prospective amendment and gate were written before scientific fitting.\n\n"
    report+=matrix.to_markdown(index=False,floatfmt=".6f")+"\n\n"
    report+=contrasts.drop(columns=["method_a","method_b"]).to_markdown(index=False,floatfmt=".6f")+"\n\n"
    report+=bs.drop(columns=["method_a","method_b"]).to_markdown(index=False,floatfmt=".6f")+"\n\n"
    report+=f"Pre-registered gate: **{gate}**. C3/C5/C6 wins: {wins['C3']}/{wins['C5']}/{wins['C6']} of 4.\n\n"
    report+="Historical reference (descriptive only; no pass/fail use):\n\n"+reference.to_markdown(index=False,floatfmt=".6f")+"\n\n"
    report+="Scores are discrimination/ranking scores, not landslide probabilities. StrictPair and Edge use strict positive margins; no threshold tuning or calibration. Bootstrap uses 10,000 identical complete-matched-set draws across methods, conditional on fitted OOF models. Folds, metrics and draws are not independent scientific replications. No unfavorable results were removed; no method or gate changed after results.\n"
    (OUT/"GATE126B_REPORT.md").write_text(report,encoding="utf-8")
    log(AUDIT["status"]); log(results.to_string(index=False)); log(bs.to_string(index=False))
    log(f"GATE126B_DECISION={gate}"); log(f"XGBOOST_VERSION={xgb.__version__}"); log("XGB_N_JOBS=1"); log(f"OUTPUT={OUT}")


if __name__=="__main__":
    try: main()
    except Exception as exc:
        if OUT.exists() and str(exc)!="FAIL_126B_OUTPUT_EXISTS":
            write("FAILURE126B.json",dict(status="FAIL_126B_TECHNICAL_OR_INTEGRITY_ERROR",error=str(exc),traceback=traceback.format_exc()))
        raise
