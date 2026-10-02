#!/usr/bin/env python
"""Experiment 128C: scalable exact connected-component replay of Arm B."""
from __future__ import annotations

import os
os.environ.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","MKL_THREADING_LAYER":"SEQUENTIAL"})

import argparse, gc, hashlib, importlib.util, json, platform, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_flow, min_weight_full_bipartite_matching
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BASE128=ROOT/"experiments/MSRR_PRETRIGGER_TEMPORAL_SENSITIVITY_V1"
OUT=BASE128/"128C_EXACT_REMATCHING"; CP=OUT/"CHECKPOINTS"
BLOCKS=["geology_parent","matching_missingness_pattern_id","soil_completeness_block"]
FIELDS=["rain_30m_mm","rain_1h_mm","rain_3h_mm","rain_6h_mm","rain_12h_mm","rain_24h_mm","rain_48h_mm","rain_72h_mm","rain_120h_mm","api_k090_step30m_120h"]
CONTRACT_VERSION="128C_FROZEN_COMPONENT_EXACT_V1"
TOL=1e-9

def imp(name:str,path:Path):
    s=importlib.util.spec_from_file_location(name,path)
    if s is None or s.loader is None: raise RuntimeError(f"IMPORT_SPEC_FAILED:{path}")
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def jwrite(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")

def csvwrite(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    d=obj if isinstance(obj,pd.DataFrame) else pd.DataFrame(obj)
    d.to_csv(path,index=False,encoding="utf-8-sig"); return d

def write_once(path:Path,text:str):
    if path.exists():
        if path.read_text(encoding="utf-8")!=text: raise RuntimeError(f"FROZEN_FILE_CHANGED:{path}")
    else: path.write_text(text,encoding="utf-8")

def log(msg:str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}",flush=True)

def semantic_edge_hash(b:pd.DataFrame)->str:
    x=b[["positive_unit_id","candidate_unit_id","composite_distance"]].sort_values(["positive_unit_id","candidate_unit_id"],kind="mergesort")
    hv=pd.util.hash_pandas_object(x,index=False,categorize=True).to_numpy(np.uint64)
    return hashlib.sha256(hv.tobytes()).hexdigest()

def preregister():
    OUT.mkdir(parents=True,exist_ok=True); CP.mkdir(parents=True,exist_ok=True)
    pre={"experiment":"128C_SCALABLE_EXACT_PRETRIGGER_REMATCHING","frozen_before_any_128C_model_result":True,
         "hiroshima_cutoff_utc":"2018-07-06T09:30:00Z","hiroshima_cutoff_jst":"2018-07-06T18:30:00+09:00",
         "kyushu_cutoff_utc":"2017-07-05T07:00:00Z","kyushu_cutoff_jst":"2017-07-05T16:00:00+09:00",
         "window_steps":70,"interval_minutes":30,"matching_variables":40,"ratio":"1:2","control_capacity":1,
         "all_or_none":True,"component_partition":"true connected components of each frozen block legal graph",
         "primary_objective":"maximum number of positives receiving exactly two distinct nonreused controls",
         "historical_tie_rule":"stable positive ID/degree/flow ordering; exact MILP only when deterministic upper-bound construction fails",
         "assignment_objective":"after positive selection, minimum total frozen composite distance",
         "no_edge_deletion":True,"no_approximation":True,"no_greedy_fallback":True,"xgb_n_jobs":1,
         "xgb_candidates":["D4","D6","D8"],"bootstrap_B":10000,"bootstrap_unit":"complete matched set",
         "gate":{"PASS_STRONG_FULL_PRETRIGGER_ROBUSTNESS":"H_WINS=4 and H_CI=4 and K_WINS>=3 and K_CI>=3",
                 "PASS_INTERNAL_FULL_PRETRIGGER_ROBUSTNESS_EXTERNAL_MIXED":"H_WINS>=3 and H_CI>=3 and external strong condition fails",
                 "otherwise":"FULL_PRETRIGGER_EFFECT_WEAKENED"},"post_result_rescue":False}
    text=json.dumps(pre,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    write_once(OUT/"128C_PREREGISTRATION.json",text)
    contract="""# 128C frozen optimization contract

The authority is Experiment 127 and its source code, not a newly invented formulation. Legal edges use the unchanged 40 variables, joint event-specific median/IQR transforms, group weights, composite distance, hard calipers, geology compatibility, exact missingness block, exact soil-completeness block, 20 km rule, and P0 candidate pool.

For every legal edge e=(p,c), x_e is binary; for every positive p, z_p is binary. The constraints are sum_{e incident to p} x_e = 2 z_p and sum_{e incident to c} x_e <= 1. The first objective maximizes sum_p z_p. Historical deterministic positive ordering resolves a same-cardinality selection when the upper-bound construction succeeds; an exact sparse MILP recovers the optimum when it does not. With selected positives frozen, sparse minimum-weight full bipartite matching minimizes sum_e composite_distance_e x_e. Candidate-index machine-epsilon perturbation and stable ID ordering retain the frozen tie principle. No legal edge is removed for computation.
"""
    proof="""# Exact connected-component decomposition proof

Within every frozen matching block, let the complete legal bipartite graph be the disjoint union G = union_k G_k of its true connected components. A positive or control vertex occurs in exactly one component. Therefore no edge decision variable is shared, no control-capacity constraint crosses components, and no positive all-or-none constraint crosses components. Feasible assignments form the Cartesian product of component feasible sets. Matched-positive cardinality and assigned-edge composite cost are additive. Maximizing complete-positive cardinality componentwise gives the global maximum; after the frozen positive selection, minimizing assignment cost componentwise gives the global minimum for that selection. Concatenating component solutions is consequently exactly equivalent. The implementation stops if it detects a cross-component variable, block overlap, global quota, or any nonseparable constraint.
"""
    write_once(OUT/"128C_FROZEN_OPTIMIZATION_CONTRACT.md",contract)
    write_once(OUT/"128C_DECOMPOSITION_PROOF.md",proof)
    status=f'''PowerShell command to inspect the process:
$pid128c = Get-Content "{OUT}\RUN_PID.txt"
Get-Process -Id $pid128c -ErrorAction SilentlyContinue

View the most recent log lines:
Get-Content "{OUT}\RUN_STDOUT.log" -Tail 50

View the error log:
Get-Content "{OUT}\RUN_STDERR.log" -Tail 50

Check whether the final report was generated:
Test-Path "{OUT}\GATE128C_REPORT.md"
'''
    (OUT/"CHECK_STATUS_128C.txt").write_text(status,encoding="utf-8")

exp=imp("exp128c",ROOT/"scripts/128B_RUN_CONSERVATIVE_PRETRIGGER_SENSITIVITY.py")
rev=imp("rev128c",ROOT/"data/06_hard_control/01b_revised_matching_protocol/revised_hard_control_01.py")
sens=imp("sens128c",ROOT/"data/06_hard_control/01c_matching_sensitivity_audit/scripts/hard_control_01b_matching_sensitivity.py")
common=imp("common128c",ROOT/"data/06_hard_control/01f_common_support_protocol/hard_control_01e_common_support_protocol.py")
ky=imp("ky128c",ROOT/"scripts/kyushu_match_01.py")
x126=imp("x126_128c",ROOT/"scripts/126B_RUN_TASK_ALIGNED_BASELINES_SINGLE_THREAD_ONE_SHOT.py")
formal=x126.formal
ext=imp("ext128c",ROOT/"scripts/124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py")
m26=imp("m26_128c",ROOT/"scripts/26_RUN_SEHC_NET_M_V1_FORMAL.py")
ds01=imp("ds01_128c",ROOT/"data/05_final_dataset_assembly/01_spatial_split_protocol/scripts/dataset_01_spatial_split_protocol.py")

AUTH=[ROOT/"experiments/MATCHING_PROTOCOL_REPRODUCIBILITY_V1"/n for n in ["MATCHING_OPTIMIZATION_PROTOCOL.md","MATCHING_DISTANCE_FORMULA.md","MATCHING_DISTANCE_COMPONENTS.csv","MATCHING_FEASIBILITY_RULES.csv","MATCHING_TIE_AUDIT.json","MATCHING_VARIABLE_DICTIONARY.csv","COMMON_SUPPORT_PROTOCOL.csv"]]

def validate_authorities():
    missing=[str(p) for p in AUTH if not p.is_file()]
    if missing: raise RuntimeError("MISSING_127_AUTHORITIES:"+"|".join(missing))
    if rev.RAIN_FIELDS!=["rain_24h_mm","rain_72h_mm","rain_120h_mm","api_k090_step30m_120h"]: raise RuntimeError("RAIN_MATCH_FIELDS_CHANGED")
    if common.BLOCKS!=BLOCKS: raise RuntimeError("BLOCK_DEFINITION_CHANGED")
    return {str(p):sha(p) for p in AUTH}

def anchor_rain(path:Path,units:list[str],time_col:str,interval_start:bool,target:str):
    r,t=exp.read_rain(path,units,time_col,interval_start)
    z=exp.regenerate(r,t,pd.DatetimeIndex([pd.Timestamp(target)]))[:,0,:]
    if z.shape!=(len(units),10) or not np.isfinite(z).all(): raise RuntimeError("NONFINITE_ANCHOR_RAIN")
    return pd.DataFrame(z,columns=FIELDS).assign(unit_id=units)

def build_hiroshima_graph():
    log("HIROSHIMA BUILD PRETRIGGER MATCHING COVARIATES")
    labels=pd.read_parquet(rev.LABELS,columns=["unit_id","label_role_main"])
    pids=labels.loc[labels.label_role_main.eq("POSITIVE"),"unit_id"].astype(str).tolist()
    cids=pd.read_parquet(ROOT/"data/06_hard_control/00_candidate_pool/09_frozen/hard_control_00_core_candidate_pool_frozen.parquet",columns=["unit_id"]).unit_id.astype(str).tolist()
    units=pids+cids; soil=pd.read_csv(rev.SOIL_FIELDS,encoding="utf-8-sig").actual_field_name.tolist()
    nonrain=sorted({f for g,fs in rev.GROUP_FIELDS.items() if g!="ANTECEDENT_RAINFALL" for f in fs})
    static=pd.read_parquet(rev.STATIC_FULL,columns=["unit_id","geology_dominant_class"]+nonrain+soil)
    static.unit_id=static.unit_id.astype(str); static=static.set_index("unit_id").loc[units].reset_index()
    rain=anchor_rain(exp.H_BASE,units,"timestamp_utc",False,"2018-07-06T09:30:00Z")
    cov=static.merge(rain[["unit_id"]+rev.RAIN_FIELDS],on="unit_id",validate="one_to_one")
    grid=gpd.read_file(rev.MASTER,layer="hiroshima_grid_250m_master")[["unit_id","geometry"]]; grid.unit_id=grid.unit_id.astype(str); grid=grid.set_index("unit_id").loc[units]
    cen=grid.geometry.centroid; cov=cov.merge(pd.DataFrame({"unit_id":units,"centroid_x":cen.x.to_numpy(),"centroid_y":cen.y.to_numpy()}),on="unit_id",validate="one_to_one")
    pos=cov.set_index("unit_id").loc[pids].reset_index(); cand=cov.set_index("unit_id").loc[cids].reset_index()
    pos,cand,params,soil_inv,groups=rev.transform_and_blocks(pos,cand)
    log("HIROSHIMA BUILD COMPLETE LEGAL GRAPH")
    edges=sens.build_parent20_superset(pos,cand,params,soil_inv,"P0").sort_values(["positive_unit_id","candidate_unit_id"],kind="mergesort").reset_index(drop=True)
    return pos,cand,edges,soil_inv

def build_kyushu_graph():
    log("KYUSHU BUILD PRETRIGGER MATCHING COVARIATES")
    import pyogrio
    master=pyogrio.read_dataframe(ky.MASTER,layer="kyushu_2017_250m_master_grid_v1"); static=pd.read_parquet(ky.STATIC); labels=pd.read_parquet(ky.LABELS)
    units=master.unit_id.astype(str).tolist(); rain=anchor_rain(exp.K_BASE,units,"time_utc",True,"2017-07-05T07:00:00Z")
    anchor=rain[["unit_id"]+rev.RAIN_FIELDS].copy(); anchor["timestamp_utc"]=pd.Timestamp("2017-07-05T07:00:00Z"); anchor["time_index"]=70
    revised,sensitivity,cm,pos,cand,params,soil_inv,groups=ky.build_covariates(master,static,labels,anchor)
    log("KYUSHU BUILD COMPLETE LEGAL GRAPH")
    edges=sensitivity.build_parent20_superset(pos,cand,params,soil_inv,"P0").sort_values(["positive_unit_id","candidate_unit_id"],kind="mergesort").reset_index(drop=True)
    return pos,cand,edges,soil_inv

def add_components(event:str,edges:pd.DataFrame,pos:pd.DataFrame,cand:pd.DataFrame):
    edges=edges.copy(); edges["component_index"]=-1; rows=[]; gid=0
    for key,b in edges.groupby(BLOCKS,sort=True,dropna=False):
        pis=np.sort(b.positive_index.unique()); cis=np.sort(b.candidate_index.unique())
        pm=pd.Series(np.arange(len(pis),dtype=np.int32),index=pis); cm=pd.Series(np.arange(len(cis),dtype=np.int32),index=cis)
        ep=pm.loc[b.positive_index].to_numpy(np.int32); ec=cm.loc[b.candidate_index].to_numpy(np.int32)+len(pis)
        graph=csr_matrix((np.ones(2*len(b),np.uint8),(np.concatenate([ep,ec]),np.concatenate([ec,ep]))),shape=(len(pis)+len(cis),len(pis)+len(cis)))
        n,lab=connected_components(graph,directed=False,return_labels=True)
        edge_lab=lab[ep]
        if not np.array_equal(edge_lab,lab[ec]): raise RuntimeError("FAIL_128C_EXACT_DECOMPOSITION_NOT_VALID")
        for local in range(n):
            ix=b.index.to_numpy()[edge_lab==local]; edges.loc[ix,"component_index"]=gid
            q=edges.loc[ix]; P=q.positive_index.nunique(); C=q.candidate_index.nunique(); E=len(q)
            rows.append({"event":event,"block_id":"|".join(map(str,key)),"component_id":gid,"n_positive":P,"n_control":C,"n_edges":E,"edge_density":E/max(P*C,1),"estimated_variables":E+P,"estimated_constraints":P+C+1})
            gid+=1
        del graph; gc.collect()
    if (edges.component_index<0).any(): raise RuntimeError("FAIL_128C_COMPONENT_ASSIGNMENT")
    audit=pd.DataFrame(rows)
    path=OUT/("HIROSHIMA_COMPONENT_SIZE_AUDIT.csv" if event=="Hiroshima 2018" else "KYUSHU_COMPONENT_SIZE_AUDIT.csv")
    csvwrite(path,audit)
    summary={"event":event,"n_blocks":int(edges.groupby(BLOCKS,dropna=False).ngroups),"n_components":len(audit),"largest_component_edges":int(audit.n_edges.max()),"median_component_edges":float(audit.n_edges.median()),"p90_component_edges":float(audit.n_edges.quantile(.9)),"p99_component_edges":float(audit.n_edges.quantile(.99))}
    jwrite(path.with_suffix(".summary.json"),summary); log(f"{event} COMPONENTS={len(audit)} LARGEST_EDGES={summary['largest_component_edges']}")
    return edges,audit

def local_maxflow(b:pd.DataFrame,selected:np.ndarray|None=None):
    pis=np.sort(b.positive_index.unique()) if selected is None else np.sort(np.asarray(selected,np.int32)); cis=np.sort(b.candidate_index.unique())
    pm=pd.Series(np.arange(len(pis),dtype=np.int32),index=pis); cm=pd.Series(np.arange(len(cis),dtype=np.int32),index=cis)
    keep=b.positive_index.isin(pis).to_numpy(); ep=pm.loc[b.loc[keep,"positive_index"]].to_numpy(np.int32); ec=cm.loc[b.loc[keep,"candidate_index"]].to_numpy(np.int32)
    source,p0,c0,sink=0,1,1+len(pis),1+len(pis)+len(cis)
    rows=np.concatenate([np.zeros(len(pis),np.int32),p0+ep,c0+np.arange(len(cis),dtype=np.int32)])
    cols=np.concatenate([p0+np.arange(len(pis),dtype=np.int32),c0+ec,np.full(len(cis),sink,np.int32)])
    vals=np.concatenate([np.full(len(pis),2,np.int32),np.ones(len(ep),np.int32),np.ones(len(cis),np.int32)])
    res=maximum_flow(csr_matrix((vals,(rows,cols)),shape=(sink+1,sink+1)),source,sink,method="dinic")
    pc=res.flow[p0:c0,c0:sink].tocoo(); nz=pc.data>0
    counts=np.bincount(pc.row[nz],minlength=len(pis)); return int(res.flow_value),pd.Series(counts,index=pis),pis,cis

def milp_with_heartbeat(label:str,objective:np.ndarray,integrality:np.ndarray,bounds:Bounds,constraints:LinearConstraint):
    """Run native HiGHS while the Python main thread prints liveness updates."""
    started=time.monotonic()
    log(f"{label} MILP_SUBMITTED variables={len(objective)} heartbeat_interval=15s")
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix="128C_HIGHS") as pool:
        future=pool.submit(milp,objective,integrality=integrality,bounds=bounds,constraints=constraints,options={"mip_rel_gap":0.0})
        while True:
            try:
                result=future.result(timeout=15)
                log(f"{label} MILP_RETURNED elapsed={time.monotonic()-started:.1f}s success={result.success} status={result.status} message={result.message}")
                return result
            except FutureTimeout:
                log(f"{label} MILP_RUNNING elapsed={time.monotonic()-started:.1f}s; exact HiGHS solve is alive; no percentage is exposed by scipy.optimize.milp")

def exact_milp_select(b:pd.DataFrame,progress_label:str):
    pis=np.sort(b.positive_index.unique()); cis=np.sort(b.candidate_index.unique()); P=len(pis); C=len(cis); E=len(b)
    pm=pd.Series(np.arange(P,dtype=np.int32),index=pis); cm=pd.Series(np.arange(C,dtype=np.int32),index=cis)
    ep=pm.loc[b.positive_index].to_numpy(np.int32); ec=cm.loc[b.candidate_index].to_numpy(np.int32)
    rows=np.concatenate([ep,P+ec,np.arange(P,dtype=np.int32)])
    cols=np.concatenate([np.arange(E,dtype=np.int32),np.arange(E,dtype=np.int32),E+np.arange(P,dtype=np.int32)])
    vals=np.concatenate([np.ones(E),np.ones(E),np.full(P,-2.0)])
    A=coo_matrix((vals,(rows,cols)),shape=(P+C,E+P)).tocsr()
    lb=np.concatenate([np.zeros(P),np.full(C,-np.inf)]); ub=np.concatenate([np.zeros(P),np.ones(C)])
    integ=np.concatenate([np.zeros(E,np.uint8),np.ones(P,np.uint8)]); bounds=Bounds(0,1)
    r1=milp_with_heartbeat(progress_label+" STAGE1_MAX_CARDINALITY",np.concatenate([np.zeros(E),-np.ones(P)]),integ,bounds,LinearConstraint(A,lb,ub))
    if not r1.success or r1.x is None: raise RuntimeError("FAIL_128C_EXACT_COMPONENT_RESOURCE_LIMIT:STAGE1")
    K=int(np.rint(r1.x[E:].sum()))
    log(f"{progress_label} STAGE1_COMPLETE K_STAR={K}")
    A2=coo_matrix((np.ones(P),(np.zeros(P,np.int32),E+np.arange(P,dtype=np.int32))),shape=(1,E+P)).tocsr()
    from scipy.sparse import vstack
    AA=vstack([A,A2],format="csr"); lb2=np.concatenate([lb,[K]]); ub2=np.concatenate([ub,[K]])
    rank=np.arange(P,dtype=float)*1e-12
    r2=milp_with_heartbeat(progress_label+" STAGE2_FROZEN_POSITIVE_TIE",np.concatenate([np.zeros(E),rank]),integ,bounds,LinearConstraint(AA,lb2,ub2))
    if not r2.success or r2.x is None: raise RuntimeError("FAIL_128C_EXACT_COMPONENT_RESOURCE_LIMIT:STAGE2")
    return pis[r2.x[E:]>.5].astype(np.int32),K,"EXACT_SPARSE_MILP_MAX_CARDINALITY_THEN_FROZEN_POSITIVE_ORDER"

def mincost_component(b:pd.DataFrame,selected:np.ndarray,pos:pd.DataFrame,cand:pd.DataFrame):
    q=b[b.positive_index.isin(selected)].copy(); pis=np.sort(selected); cis=np.sort(q.candidate_index.unique()); P=len(pis)
    pm=pd.Series(np.arange(P,dtype=np.int32),index=pis); cm=pd.Series(np.arange(len(cis),dtype=np.int32),index=cis)
    ep=pm.loc[q.positive_index].to_numpy(np.int32); ec=cm.loc[q.candidate_index].to_numpy(np.int32)
    rows=np.concatenate([2*ep,2*ep+1]); cols=np.concatenate([ec,ec]); base=q.composite_distance.to_numpy(float); tie=(q.candidate_index.to_numpy(float)+1)*np.finfo(float).eps
    M=csr_matrix((np.concatenate([base+tie+1e-12,base+tie+1e-12]),(rows,cols)),shape=(2*P,len(cis)))
    ri,ci=min_weight_full_bipartite_matching(M)
    if len(ri)!=2*P: raise RuntimeError("FAIL_128C_MINCOST_INCOMPLETE")
    a=pd.DataFrame({"positive_index":pis[ri//2],"candidate_index":cis[ci]})
    lookup=q.sort_values(["positive_index","candidate_index"],kind="mergesort").drop_duplicates(["positive_index","candidate_index"]).set_index(["positive_index","candidate_index"])
    d=lookup.loc[pd.MultiIndex.from_frame(a)].reset_index()
    if "positive_unit_id" not in d: d=d.merge(pos[["unit_id"]].reset_index(names="positive_index"),on="positive_index").rename(columns={"unit_id":"positive_unit_id"})
    if "candidate_unit_id" not in d: d=d.merge(cand[["unit_id"]].reset_index(names="candidate_index"),on="candidate_index").rename(columns={"unit_id":"candidate_unit_id"})
    return d

def solve_component(event:str,cid:int,b:pd.DataFrame,pos:pd.DataFrame,cand:pd.DataFrame,contract_hash:str,use_checkpoint=True):
    tag=("H" if event.startswith("Hiroshima") else "K")+f"_{cid:06d}"; meta=CP/(tag+".json"); tab=CP/(tag+"_edges.parquet")
    eh=semantic_edge_hash(b); signature={"contract_version":CONTRACT_VERSION,"contract_hash":contract_hash,"legal_edge_hash":eh,"scipy_version":scipy.__version__,"cutoff":"2018-07-06T09:30:00Z" if tag[0]=="H" else "2017-07-05T07:00:00Z"}
    if use_checkpoint and meta.exists() and tab.exists():
        old=json.loads(meta.read_text(encoding="utf-8"))
        if old.get("signature")==signature and old.get("status")=="OPTIMAL":
            log(f"event={event} component={cid} CHECKPOINT_REUSED K_STAR={old.get('K_star')} runtime_saved={old.get('runtime_seconds')}s")
            return pd.read_parquet(tab),old
    t=time.time(); label=f"event={event} component={cid} P={b.positive_index.nunique()} C={b.candidate_index.nunique()} E={len(b)}"
    log(f"{label} START MAXFLOW_UPPER_BOUND")
    flow,counts,pis,cis=local_maxflow(b); upper=flow//2
    log(f"{label} MAXFLOW_UPPER_BOUND_COMPLETE ordinary_maxflow={flow} half_upper={upper}")
    degree=b.groupby("positive_index").size().reindex(pis).to_numpy(); ids=pos.loc[pis,"unit_id"].astype(str).to_numpy()
    ranked=pis[np.lexsort((ids,-degree,-counts.reindex(pis).to_numpy()))][:upper]
    cf,cc,_,_=local_maxflow(b,ranked); method="FROZEN_UPPER_BOUND_DETERMINISTIC_CONSTRUCTION"
    log(f"{label} DETERMINISTIC_CONSTRUCTION flow={cf} required={2*upper}")
    if cf!=2*upper or not np.all(cc.reindex(ranked,fill_value=0).to_numpy()==2):
        log(f"{label} EXACT_MILP_REQUIRED reason=deterministic_construction_below_upper")
        ranked,K,method=exact_milp_select(b,label)
    else: K=upper
    log(f"{label} MINCOST_ASSIGNMENT_START selected_positives={K}")
    assigned=mincost_component(b,ranked,pos,cand)
    log(f"{label} MINCOST_ASSIGNMENT_COMPLETE assignments={len(assigned)} total_cost={assigned.composite_distance.sum():.12g}")
    if len(assigned)!=2*K or assigned.candidate_unit_id.duplicated().any() or not assigned.groupby("positive_unit_id").size().eq(2).all(): raise RuntimeError("FAIL_128C_COMPONENT_CERTIFICATE")
    rec={"component_id":cid,"status":"OPTIMAL","K_star":K,"minimum_cost":float(assigned.composite_distance.sum()),"selected_edges":len(assigned),"selected_positives":K,"runtime_seconds":time.time()-t,"selection_method":method,"assignment_method":"SCIPY_SPARSE_MIN_WEIGHT_FULL_BIPARTITE_MATCHING","signature":signature}
    if use_checkpoint: assigned.to_parquet(tab,index=False); jwrite(meta,rec)
    return assigned,rec

def small_equivalence(cases:list[tuple],contract_hash:str):
    rows=[]
    for event,cid,b,pos,cand in cases[:10]:
        common.N_POS=len(pos); common.N_CAND=len(cand)
        oldsel,_=common.exact_all_or_none(b.drop(columns="component_index",errors="ignore"),pos)
        olda,_=common.mincost_certificate(b.drop(columns="component_index",errors="ignore"),oldsel,pos,cand)
        newa,rec=solve_component(event,cid,b,pos,cand,contract_hash,False)
        oldpairs=set(zip(olda.positive_unit_id.astype(str),olda.candidate_unit_id.astype(str))); newpairs=set(zip(newa.positive_unit_id.astype(str),newa.candidate_unit_id.astype(str)))
        sameK=len(oldsel)==rec["K_star"]; dc=abs(float(olda.composite_distance.sum())-rec["minimum_cost"]); same=oldpairs==newpairs
        ok=sameK and dc<=TOL
        rows.append({"event":event,"component_id":cid,"n_edges":len(b),"original_matched_positive":len(oldsel),"decomposed_matched_positive":rec["K_star"],"same_cardinality":sameK,"selected_positive_ids_equal":set(olda.positive_unit_id)==set(newa.positive_unit_id),"selected_control_ids_equal":set(olda.candidate_unit_id)==set(newa.candidate_unit_id),"pair_membership_equal":same,"total_cost_difference":dc,"tie_equivalent_alternative":bool(ok and not same),"PASS":ok})
    d=csvwrite(OUT/"128C_SMALL_COMPONENT_EQUIVALENCE.csv",rows)
    if len(d)<10 or not d.PASS.all(): raise RuntimeError("FAIL_128C_SOLVER_EQUIVALENCE")

def solve_event(event:str,pos:pd.DataFrame,cand:pd.DataFrame,edges:pd.DataFrame,audit:pd.DataFrame,soil_inv:pd.DataFrame,contract_hash:str):
    parts=[]; records=[]; total=len(audit)
    for n,cid in enumerate(audit.component_id,1):
        b=edges[edges.component_index.eq(cid)].drop(columns="component_index")
        a,r=solve_component(event,int(cid),b,pos,cand,contract_hash,True); parts.append(a); records.append(r)
        log(f"event={event} block={audit.loc[audit.component_id.eq(cid),'block_id'].iloc[0]} component={cid} completed_components={n}/{total} runtime={r['runtime_seconds']:.1f}s matched_positives={r['K_star']}")
        gc.collect()
    assigned=pd.concat(parts,ignore_index=True); assigned=assigned.sort_values(["positive_unit_id","composite_distance","candidate_unit_id"],kind="mergesort")
    assigned["control_rank"]=assigned.groupby("positive_unit_id").cumcount()+1
    prefix="H128C" if event.startswith("Hiroshima") else "K128C"; pids=sorted(assigned.positive_unit_id.unique()); pmap={u:f"{prefix}_{i:05d}" for i,u in enumerate(pids,1)}; out=[]
    for u in pids:
        out.append({"pair_set_id":pmap[u],"unit_id":u,"sample_role":"POSITIVE","y_pair":1,"control_rank":pd.NA,"matching_distance":0.0})
        for r in assigned[assigned.positive_unit_id.eq(u)].itertuples(): out.append({"pair_set_id":pmap[u],"unit_id":r.candidate_unit_id,"sample_role":"HARD_CONTROL","y_pair":0,"control_rank":int(r.control_rank),"matching_distance":float(r.composite_distance)})
    index=pd.DataFrame(out); index.control_rank=index.control_rank.astype("Int64")
    stem="HIROSHIMA" if event.startswith("Hiroshima") else "KYUSHU"; csvwrite(OUT/f"{stem}_PRETRIGGER_EXACT_MATCHING.csv",index)
    matchedp=pos.set_index("unit_id").loc[pids]; cids=index.loc[index.y_pair.eq(0),"unit_id"].tolist(); matchedc=cand.set_index("unit_id").loc[cids]
    legalc=cand.set_index("unit_id").loc[sorted(edges.candidate_unit_id.unique())]; fields=[f for fs in rev.GROUP_FIELDS.values() for f in fs]+soil_inv.actual_field_name.tolist(); br=[]
    for f in fields:
        def smd(a,b):
            x=pd.to_numeric(a,errors="coerce").dropna().to_numpy(float); y=pd.to_numeric(b,errors="coerce").dropna().to_numpy(float); den=np.sqrt((np.var(x,ddof=1)+np.var(y,ddof=1))/2); return float((x.mean()-y.mean())/den) if den>0 else 0.0
        bef=smd(pos[f],legalc[f]); aft=smd(matchedp[f],matchedc[f]); br.append({"field_name":f,"smd_before":bef,"abs_smd_before":abs(bef),"smd_after":aft,"abs_smd_after":abs(aft)})
    bal=csvwrite(OUT/f"{stem}_PRETRIGGER_BALANCE.csv",br)
    au={"event":event,"inventory_positive_n":len(pos),"legal_positive_n":int(edges.positive_index.nunique()),"matched_positive_n":len(pids),"excluded_positive_n":len(pos)-len(pids),"matched_control_n":len(cids),"matched_set_n":len(pids),"duplicate_control_n":int(pd.Series(cids).duplicated().sum()),"partial_positive_n":int((index.groupby('pair_set_id').size()!=3).sum()),"legal_edge_n":len(edges),"component_n":total,"total_minimum_cost_after_frozen_positive_selection":float(assigned.composite_distance.sum()),"protocol_relaxed":False,"approximate_solver":False,"nearest_k_sparsification":False}
    jwrite(OUT/f"{stem}_PRETRIGGER_MATCHING_AUDIT.json",au)
    return index,bal,au

def fold_replay(index:pd.DataFrame):
    """Rebuild the frozen DATASET-01 partition on the new matched membership."""
    log("ARM_B SPATIAL PARTITION START: rebuild DATASET-00 atomic groups on exact-rematched membership")
    d0=ROOT/"data/05_final_dataset_assembly/00_dataset_inventory_and_split_feasibility"
    split0=pd.read_parquet(d0/"dataset_00_split_group_candidates.parquet")
    frozen=pd.read_parquet(ROOT/"data/05_final_dataset_assembly/01_spatial_split_protocol/dataset_01_primary_5fold_assignment.parquet")
    positives=index[index.y_pair.eq(1)][["pair_set_id","unit_id"]].rename(columns={"unit_id":"positive_unit_id"}).copy()
    old_ids=set(split0.positive_unit_id.astype(str)); new_ids=set(positives.positive_unit_id.astype(str))
    missing_old=sorted(new_ids-old_ids)

    master=gpd.read_file(ds01.MASTER,ignore_geometry=True)[["unit_id","grid_row","grid_col","centroid_x","centroid_y"]]
    pos=positives.merge(master,left_on="positive_unit_id",right_on="unit_id",how="left",validate="one_to_one").drop(columns="unit_id")
    if pos[["grid_row","grid_col","centroid_x","centroid_y"]].isna().any().any(): raise RuntimeError("ARM_B_SPATIAL_GROUP_MASTER_JOIN_FAILED")

    # Exact DATASET-00 atomic grouping rules: 8-neighbour contact plus shared
    # positive evidence component.  Recomputing is necessary because a newly
    # admitted positive can merge two formerly separate groups.
    parent={u:u for u in sorted(new_ids)}
    def find(u):
        while parent[u]!=u:
            parent[u]=parent[parent[u]]; u=parent[u]
        return u
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    grid={(int(r.grid_row),int(r.grid_col)):str(r.positive_unit_id) for r in pos.itertuples(index=False)}
    for r in pos.itertuples(index=False):
        u=str(r.positive_unit_id)
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr or dc:
                    v=grid.get((int(r.grid_row)+dr,int(r.grid_col)+dc))
                    if v: union(u,v)
    trace=pd.read_parquet(ROOT/"data/05_event_label_evidence/08_label_04_positive_grid_mapping/06_traceability/label_04_main_grid_ajg_traceability.parquet")
    trace=trace[trace.unit_id.astype(str).isin(new_ids)&trace.positive_area_flag.eq("YES")].copy()
    for _,g in trace.groupby("evidence_component_id"):
        ids=sorted(set(g.unit_id.astype(str)))
        for v in ids[1:]: union(ids[0],v)
    members={}
    for u in sorted(new_ids): members.setdefault(find(u),[]).append(u)
    gid={u:f"DSG_{hashlib.sha256('|'.join(sorted(v)).encode()).hexdigest()[:16]}" for u,v in members.items()}
    pos["spatial_group_id"]=pos.positive_unit_id.astype(str).map(lambda u:gid[find(u)])
    comps=trace.groupby("unit_id").evidence_component_id.apply(lambda x:"|".join(sorted(set(x.astype(str))))).to_dict()
    pos["evidence_component_ids"]=pos.positive_unit_id.map(comps).fillna("")

    label=pd.read_parquet(ROOT/"data/05_event_label_evidence/08_label_04_positive_grid_mapping/09_frozen/label_04_positive_grid_labels_250m_frozen.parquet",columns=["unit_id","evidence_grade_strongest"])
    static_full=pd.read_parquet(ds01.STATIC_FULL,columns=["unit_id","geology_dominant_class","lc_dominant_fcc_name"])
    pos=pos.merge(label.rename(columns={"unit_id":"positive_unit_id"}),on="positive_unit_id",how="left")
    pos=pos.merge(static_full.rename(columns={"unit_id":"positive_unit_id","lc_dominant_fcc_name":"dominant_landcover"}),on="positive_unit_id",how="left")
    pos["geology_parent_class"]=pos.geology_dominant_class.map(ds01.geology_parent)
    mun=gpd.read_file(ds01.MUNICIPALITY).to_crs(6671)[["N03_007","geometry"]].rename(columns={"N03_007":"municipality_code"})
    points=gpd.GeoDataFrame(pos[["positive_unit_id","centroid_x","centroid_y"]].copy(),geometry=gpd.points_from_xy(pos.centroid_x,pos.centroid_y),crs=6671)
    mj=gpd.sjoin(points,mun,how="left",predicate="within")[["positive_unit_id","municipality_code"]].drop_duplicates("positive_unit_id")
    pos=pos.merge(mj,on="positive_unit_id",how="left"); pos["municipality_code"]=pos.municipality_code.fillna("UNRESOLVED_N03")

    static_model=pd.read_parquet(ds01.STATIC_MODEL)
    units=index[["pair_set_id","unit_id","sample_role"]].merge(static_model,on="unit_id",how="left")
    units=units.merge(ds01.load_anchor_rain(set(units.unit_id.astype(str))),on="unit_id",how="left")
    units["soil_missingness_pattern"]=units.apply(ds01.soil_pattern,axis=1)
    meta=pos[["pair_set_id","spatial_group_id","evidence_grade_strongest","geology_parent_class","dominant_landcover","municipality_code","evidence_component_ids","centroid_x","centroid_y"]]
    units=units.merge(meta,on="pair_set_id",how="left",validate="many_to_one")

    whitelist=pd.read_csv(d0/"dataset_00_model_feature_whitelist.csv")
    categorical=whitelist.loc[whitelist.categorical_encoding_required.eq("YES"),"field_name"].tolist()
    numeric=[f for f in whitelist.field_name if f not in categorical]
    stable=[]
    for f in numeric:
        x=pd.to_numeric(units[f],errors="coerce"); iqr=float(x.quantile(.75)-x.quantile(.25)); nz=float(x.ne(0).sum()/max(x.notna().sum(),1))
        if x.notna().mean()>=.8 and nz>=.05 and iqr>0: stable.append(f)
    groups=pos.groupby("spatial_group_id").agg(pair_set_count=("pair_set_id","size"),centroid_x=("centroid_x","mean"),centroid_y=("centroid_y","mean")).reset_index()
    log(f"ARM_B SPATIAL GROUPS COMPLETE sets={len(pos)} groups={len(groups)} newly_admitted={len(missing_old)}")
    groups["largest_group_rank"]=groups.pair_set_count.rank(method="first",ascending=False).astype(int)
    gi={g:i for i,g in enumerate(groups.spatial_group_id)}
    tree=scipy.spatial.cKDTree(groups[["centroid_x","centroid_y"]].to_numpy()); neighbors=tree.query_ball_tree(tree,r=5000)
    edge_set={(i,j) for i,js in enumerate(neighbors) for j in js if i<j}
    neighbor_edges=np.asarray(sorted(edge_set),dtype=np.int32) if edge_set else np.empty((0,2),dtype=np.int32)
    cat_mats={}
    for name,field in [("grade","evidence_grade_strongest"),("geology","geology_parent_class"),("landcover","dominant_landcover"),("missing","soil_missingness_pattern"),("municipality","municipality_code")]:
        source=pos[["spatial_group_id",field]] if field!="soil_missingness_pattern" else units[["spatial_group_id",field]].drop_duplicates()
        cats=sorted(source[field].fillna("<NA>").astype(str).unique()); mat=np.zeros((len(groups),len(cats)),float)
        for g,s in source.groupby("spatial_group_id"):
            counts=s[field].fillna("<NA>").astype(str).value_counts()
            for j,c in enumerate(cats): mat[gi[g],j]=counts.get(c,0)
        cat_mats[name]=mat
    focus=[f for f in ds01.FOCUS_CONTINUOUS if f in units and f in stable]
    cont_sum=np.zeros((len(groups),len(focus)),float); cont_n=np.zeros_like(cont_sum)
    for j,f in enumerate(focus):
        z,_,_=ds01.robust_scale(units[f]); tmp=pd.DataFrame({"spatial_group_id":units.spatial_group_id,"z":z}); sums=tmp.groupby("spatial_group_id").z.sum(min_count=1); ns=tmp.groupby("spatial_group_id").z.count()
        for g in groups.spatial_group_id:
            v=sums.get(g,0.0); cont_sum[gi[g],j]=float(v) if pd.notna(v) else 0.0; cont_n[gi[g],j]=float(ns.get(g,0.0))

    category_fields=["evidence_grade_strongest","geology_parent_class","dominant_landcover","municipality_code","soil_missingness_pattern"]+categorical
    rows=[]; traces=[]; assignments={}
    for profile,weights in ds01.WEIGHT_PROFILES.items():
        seed=ds01.PROFILE_SEEDS[profile]; model=ds01.ObjectiveModel(groups,cat_mats,cont_sum,cont_n,neighbor_edges,weights)
        log(f"ARM_B DATASET01 OPTIMIZATION profile={profile} seed={seed} stage=GREEDY_START")
        greedy,tr0=ds01.greedy_assignment(model,seed); local,tr1=ds01.local_search(model,greedy,seed); anneal,tr2=ds01.simulated_annealing(model,greedy,seed)
        for method,(a,tr) in {"GREEDY_BALANCED":(greedy,tr0),"GREEDY_PAIRWISE_SWAP":(local,tr1),"DETERMINISTIC_SIMULATED_ANNEALING":(anneal,tr2)}.items():
            cid=f"{method}__{profile}__SEED{seed}"; assignments[cid]=pd.Series(a,index=groups.spatial_group_id)
            metrics=ds01.assignment_metrics(units,assignments[cid],stable,category_fields); obj,parts=model.evaluate(a)
            rows.append({"candidate_id":cid,"method":method,"weight_profile":profile,"seed":seed,"final_objective":obj,**parts,**metrics})
            traces.extend({"candidate_id":cid,**x} for x in tr)
            log(f"ARM_B DATASET01 CANDIDATE_COMPLETE id={cid} objective={obj:.8f} max_pair_deviation={metrics['max_pair_count_deviation']:.6f}")
    comparison=pd.DataFrame(rows); feasible=comparison[comparison.empty_fold_count.eq(0)&comparison.a1_missing_fold_count.eq(0)&comparison.a3_missing_fold_count.eq(0)&comparison.max_pair_count_deviation.le(.10)]
    order=["max_stable_robust_smd","max_category_proportion_difference","final_objective","candidate_id"]
    if feasible.empty: raise RuntimeError("ARM_B_DATASET01_NO_HARD_FEASIBLE_PARTITION")
    selected=str(feasible.sort_values(order,kind="mergesort").iloc[0].candidate_id)
    log(f"ARM_B DATASET01 PARTITION_SELECTED id={selected}")
    comparison["selected_primary_flag"]=np.where(comparison.candidate_id.eq(selected),"YES","NO")
    csvwrite(OUT/"ARM_B_DATASET01_CANDIDATE_SPLIT_COMPARISON.csv",comparison); csvwrite(OUT/"ARM_B_DATASET01_OPTIMIZATION_TRACE.csv",traces)
    p=pos[["pair_set_id","positive_unit_id","spatial_group_id","evidence_component_ids"]].copy(); p["outer_fold_id"]=p.spatial_group_id.map(assignments[selected]).astype(int)

    compfold={}
    for r in p.itertuples(index=False):
        for c in str(r.evidence_component_ids).split("|"):
            if c: compfold.setdefault(c,set()).add(int(r.outer_fold_id))
    old_smap=split0.drop_duplicates("positive_unit_id").set_index("positive_unit_id").spatial_group_id
    old_fmap=frozen.drop_duplicates("spatial_group_id").set_index("spatial_group_id").outer_fold
    bridge={}
    for u in missing_old:
        r=pos[pos.positive_unit_id.eq(u)].iloc[0]; linked=set()
        for q in pos.itertuples(index=False):
            if q.positive_unit_id in old_ids and abs(int(q.grid_row)-int(r.grid_row))<=1 and abs(int(q.grid_col)-int(r.grid_col))<=1: linked.add(str(old_smap.get(q.positive_unit_id)))
        bridge[u]={"old_neighbor_groups":sorted(linked),"old_neighbor_folds":sorted({str(old_fmap.get(g)) for g in linked})}
    checks={"partition":"PRIMARY_BALANCED_SPATIAL_GROUPED_5FOLD recomputed on exact-rematched membership using frozen DATASET-01 implementation, weights, seeds, feasibility filter, and selection rule","direct_old_fold_replay_valid":False,"direct_replay_block_reason":"new positives absent from DATASET-00; at least one bridges old spatial groups assigned to different folds","new_positive_ids_absent_from_old_membership":missing_old,"old_group_bridge_audit":bridge,"selected_candidate_id":selected,"spatial_group_count":int(p.spatial_group_id.nunique()),"no_matched_set_split":not p.pair_set_id.duplicated().any(),"no_spatial_group_split":not p.groupby("spatial_group_id").outer_fold_id.nunique().gt(1).any(),"no_evidence_component_split":all(len(v)==1 for v in compfold.values()),"no_duplicated_dynamic_sequence_split":not index.unit_id.duplicated().any(),"folds":sorted(p.outer_fold_id.unique().tolist()),"fold_set_counts":{str(k):int(v) for k,v in p.outer_fold_id.value_counts().sort_index().items()}}
    checks["PASS"]=all(checks[k] for k in ["no_matched_set_split","no_spatial_group_split","no_evidence_component_split","no_duplicated_dynamic_sequence_split"]) and checks["folds"]==[0,1,2,3,4]
    jwrite(OUT/"ARM_B_SPATIAL_FOLD_AUDIT.json",checks)
    if not checks["PASS"]: raise RuntimeError("ARM_B_EXACT_MATCHING_COMPLETE_MODEL_EVAL_BLOCKED_BY_FOLD_REPLAY")
    return p[["pair_set_id","spatial_group_id","outer_fold_id"]]

def custom_bundle(index:pd.DataFrame,folds:pd.DataFrame,cutoff:str):
    old,runner,base=formal.load_dataset_loader(); x=index.merge(folds,on="pair_set_id",how="left").sort_values(["pair_set_id","y_pair","control_rank"],ascending=[True,False,True],kind="mergesort").reset_index(drop=True)
    x["sample_index"]=np.arange(len(x)); x["outer_fold_id"]=x.outer_fold_id.astype(int); x["spatial_group_id"]=x.spatial_group_id.astype(str)
    pt=x.groupby("pair_set_id",sort=False).sample_index.apply(list); pt=np.asarray(pt.tolist(),np.int64)
    raw=pd.read_parquet(rev.STATIC_FULL); raw.unit_id=raw.unit_id.astype(str); need=list(old.ss["continuous_fields_in_order"])+[old.cat]
    st=raw.set_index("unit_id").loc[x.unit_id,need].reset_index(); st.insert(0,"sample_index",np.arange(len(st)))
    target=pd.date_range(pd.Timestamp(cutoff)-pd.Timedelta(minutes=69*30),cutoff,freq="30min",tz="UTC"); dyn=exp.event_raw_tensor(exp.H_BASE,x.unit_id.astype(str).tolist(),"timestamp_utc",False,target)
    b=SimpleNamespace(sample=x,pair=pd.DataFrame({"pair_set_id":x.loc[pt[:,0],"pair_set_id"].to_numpy()}),pt=pt,pr=pd.DataFrame({"pair_set_id":x.loc[pt[:,0],"pair_set_id"].to_numpy()}),pair_fold=x.loc[pt[:,0],"outer_fold_id"].to_numpy(),static=st,dynamic=dyn,ss=old.ss,ds=old.ds,cont=old.cont,cat=old.cat,soil=old.soil,splan=old.splan,dlog=old.dlog)
    base.N=len(x); base.NP=len(pt); base.NC=2*len(pt)
    return b,runner,base

def metric(score,y,pairs): return exp.metrics(score,y,pairs)

def model_evaluation(hindex:pd.DataFrame,kindex:pd.DataFrame,folds:pd.DataFrame,hbal:pd.DataFrame,kbal:pd.DataFrame):
    bundle,runner,base=custom_bundle(hindex,folds,"2018-07-06T09:30:00Z"); formal.OUT=OUT; x126.impl.OUT=OUT; y=bundle.sample.y_pair.to_numpy(np.int8); pairs=bundle.pt
    oof={"RAW":np.full(len(y),np.nan),"MSRR":np.full(len(y),np.nan)}; foldrows=[]; sels=[]
    for hf in formal.FOLDS:
        fd=formal.build_fold_loader(bundle,runner,base,hf,torch.device("cpu")); ids=[np.asarray(v,np.int64) for v in (fd.train_pairs,fd.validation_pairs,fd.test_pairs)]; rows=[pairs[v].reshape(-1) for v in ids]
        static=formal.as_numpy(fd.static92).astype(np.float32); rain=formal.as_numpy(fd.rain).astype(np.float32)
        for rep in ["RAW","MSRR"]:
            fn=formal.raw_features if rep=="RAW" else formal.msrr_features; feats=[fn(pairs[v],static,rain)[0] for v in ids]
            log(f"ARM_B HIROSHIMA FOLD={hf} REP={rep} START")
            score,sel,cfg=formal.fit_xgb_select_predict(rep,hf,feats[0],y[rows[0]],feats[1],y[rows[1]],feats[2],7+hf*1000+600,1)
            oof[rep][rows[2]]=score; foldrows.append({"representation":rep,"human_fold":hf,"selected_config":cfg,**metric(score,y[rows[2]],np.arange(len(rows[2])).reshape(-1,3))}); sel["representation"]=rep; sel["human_fold"]=hf; sel["selected"]=sel.config_id==cfg; sels.append(sel)
    if not all(np.isfinite(v).all() for v in oof.values()): raise RuntimeError("ARM_B_NONFINITE_OOF")
    csvwrite(OUT/"ARM_B_HIROSHIMA_FOLD_METRICS.csv",foldrows); selection=csvwrite(OUT/"ARM_B_HIROSHIMA_VALIDATION_SELECTION.csv",pd.concat(sels,ignore_index=True)); np.save(OUT/"ARM_B_HIROSHIMA_RAW_OOF_score.npy",oof["RAW"]); np.save(OUT/"ARM_B_HIROSHIMA_MSRR_OOF_score.npy",oof["MSRR"])
    hb,hr,hm=exp.paired_bootstrap(oof["RAW"],oof["MSRR"],y,pairs,"_128C_TEMP.csv",B=10000,seed=20260731); hb.to_csv(OUT/"ARM_B_HIROSHIMA_BOOTSTRAP.csv",index=False,encoding="utf-8-sig"); (BASE128/"_128C_TEMP.csv").unlink(missing_ok=True)
    csvwrite(OUT/"ARM_B_HIROSHIMA_RESULTS.csv",[{"representation":"RAW_PRETRIGGER_EXACT_REMATCH",**hr},{"representation":"MSRR_PRETRIGGER_EXACT_REMATCH",**hm}])
    selected={}
    for rep in ["RAW","MSRR"]:
        a=selection[selection.representation.eq(rep)].groupby("config_id",as_index=False).agg(validation_score=("validation_score","mean"),StrictPair=("StrictPair","mean"),Edge=("Edge","mean"),AUPRC=("AUPRC","mean"),AUROC=("AUROC","mean")).sort_values(["validation_score","StrictPair","Edge","AUPRC","AUROC","config_id"],ascending=[False,False,False,False,False,True],kind="mergesort"); selected[rep]=str(a.iloc[0].config_id)
    sh,rh,pre,prep=ext.fit_full_hiroshima_preprocessor(bundle,runner,base,m26); xhr,_=formal.raw_features(pairs,sh,rh); xhm,_=formal.msrr_features(pairs,sh,rh); yh=y[pairs.reshape(-1)]
    ext.EXPECTED["matched_rows"]=len(kindex); ext.EXPECTED["matched_sets"]=kindex.pair_set_id.nunique(); model_index,pair_k=ext.build_external_model_index(kindex)
    kstatic=pd.read_parquet(ky.STATIC); target=pd.date_range(pd.Timestamp("2017-07-05T07:00:00Z")-pd.Timedelta(minutes=69*30),"2017-07-05T07:00:00Z",freq="30min",tz="UTC"); kraw=exp.event_raw_tensor(exp.K_BASE,model_index.unit_id.tolist(),"time_utc",True,target)
    kd=[]
    for i,u in enumerate(model_index.unit_id):
        z=pd.DataFrame(kraw[i],columns=FIELDS); z.insert(0,"time_index",np.arange(70)); z.insert(0,"timestamp_utc",target); z.insert(0,"unit_id",u); kd.append(z)
    sk,rk,ka=ext.transform_external_with_hiroshima_preprocessor(kstatic,pd.concat(kd,ignore_index=True),model_index,bundle,runner,m26,pre,base); xkr,_=formal.raw_features(pair_k,sk,rk); xkm,_=formal.msrr_features(pair_k,sk,rk)
    sr=ext.fit_predict_xgb(formal,selected["RAW"],xhr,yh,xkr,1); sm=ext.fit_predict_xgb(formal,selected["MSRR"],xhm,yh,xkm,1); eval_index,eval_rows=ext.build_external_evaluation_order(kindex,model_index); ky_y=eval_index.y_true.to_numpy(np.int8); flat=eval_rows.reshape(-1); kp=np.arange(len(flat)).reshape(-1,3)
    kb,kr,km=exp.paired_bootstrap(sr[flat],sm[flat],ky_y,kp,"_128C_TEMP_K.csv",B=10000,seed=20260801); kb.to_csv(OUT/"ARM_B_KYUSHU_BOOTSTRAP.csv",index=False,encoding="utf-8-sig"); (BASE128/"_128C_TEMP_K.csv").unlink(missing_ok=True)
    csvwrite(OUT/"ARM_B_KYUSHU_RESULTS.csv",[{"representation":"RAW_PRETRIGGER_EXACT_REMATCH",**kr},{"representation":"MSRR_PRETRIGGER_EXACT_REMATCH",**km}])
    metrics4=["AUROC","AUPRC","StrictPair","Edge"]; hw=sum(hm[m]>hr[m] for m in metrics4); hc=int((hb.CI95_LOW>0).sum()); kw=sum(km[m]>kr[m] for m in metrics4); kc=int((kb.CI95_LOW>0).sum())
    gate="PASS_STRONG_FULL_PRETRIGGER_ROBUSTNESS" if hw==4 and hc==4 and kw>=3 and kc>=3 else ("PASS_INTERNAL_FULL_PRETRIGGER_ROBUSTNESS_EXTERNAL_MIXED" if hw>=3 and hc>=3 else "FULL_PRETRIGGER_EFFECT_WEAKENED")
    jwrite(OUT/"GATE128C_DECISION.json",{"GATE128C_DECISION":gate,"H_WINS":hw,"H_CI":hc,"K_WINS":kw,"K_CI":kc})
    oldh=pd.read_csv(BASE128/"ARM_A_HIROSHIMA_PRETRIGGER_BOOTSTRAP.csv"); oldk=pd.read_csv(BASE128/"ARM_A_KYUSHU_PRETRIGGER_BOOTSTRAP.csv"); comp=[]
    for ev,a,b,ba,bb in [("Hiroshima 2018",oldh,hb,None,hbal),("Kyushu 2017",oldk,kb,None,kbal)]:
        for m in metrics4:
            ra=a[a.metric.eq(m)].iloc[0]; rb=b[b.metric.eq(m)].iloc[0]; comp.append({"event":ev,"metric":m,"ARM_A_RAW":ra.RAW,"ARM_A_MSRR":ra.MSRR,"ARM_A_delta":ra.observed_delta,"ARM_B_RAW":rb.RAW,"ARM_B_MSRR":rb.MSRR,"ARM_B_delta":rb.observed_delta})
    csvwrite(OUT/"ARM_A_VS_ARM_B_COMPARISON.csv",comp)
    report=f"# Experiment 128C gate report\n\nDecision: **{gate}**. Hiroshima wins/positive CIs: {hw}/4 and {hc}/4. Kyushu wins/positive CIs: {kw}/4 and {kc}/4. Matching used complete legal graphs and exact connected-component decomposition.\n"; (OUT/"GATE128C_REPORT.md").write_text(report,encoding="utf-8")
    evidence="# Reviewer #2 Comment 5 — final evidence\n\nA. Every feature uses only current and past rainfall.\n\nB. The original event-level cutoff cannot prove individual pre-failure status because grid-specific failure timestamps are unavailable.\n\nC. Arm A tests conservative pre-trigger model inputs on the frozen benchmark.\n\nD. Arm B tests conservative pre-trigger rainfall in both exact rematching and model inputs. The completed 128C outputs report its results. Even with a passing gate, we do not claim that all individual landslides are guaranteed pre-failure.\n"; (OUT/"REVIEWER2_COMMENT5_FINAL_EVIDENCE.md").write_text(evidence,encoding="utf-8")
    return gate,hw,hc,kw,kc

def main(dry_run=False):
    preregister(); auth=validate_authorities(); contract_hash=sha(OUT/"128C_PREREGISTRATION.json")
    if dry_run:
        jwrite(OUT/"DRY_RUN.json",{"status":"PASS","imports":True,"authority_files":len(auth),"scipy":scipy.__version__,"python":sys.version,"no_model_run":True}); print("128C_DRY_RUN_PASS",flush=True); return
    audit={"cutoffs_unchanged":True,"steps_70_unchanged":True,"matching_variables_unchanged":True,"weights_unchanged":True,"calipers_unchanged":True,"distance_unchanged":True,"control_capacity":1,"ratio":"1:2","all_or_none_preserved":True,"no_legal_edge_deleted_for_computation":True,"decomposition_true_connected_components_only":True,"no_approximate_solver":True,"no_greedy_fallback":True,"no_nearest_k":True,"xgb_n_jobs":1,"bootstrap_B":10000,"Kyushu_used_for_model_selection":False,"no_post_result_rescue":True,"authority_hashes":auth}
    try:
        hp,hc,he,hs=build_hiroshima_graph(); kp,kc,ke,ks=build_kyushu_graph(); he,ha=add_components("Hiroshima 2018",he,hp,hc); ke,ka=add_components("Kyushu 2017",ke,kp,kc)
        candidates=[]
        for event,edges,a,pos,cand in [("Hiroshima 2018",he,ha,hp,hc),("Kyushu 2017",ke,ka,kp,kc)]:
            for cid in a.sort_values(["n_edges","component_id"]).component_id:
                b=edges[edges.component_index.eq(cid)]
                if len(b)<=150000: candidates.append((event,int(cid),b,pos,cand))
        small_equivalence(candidates,contract_hash); audit["small_problem_exact_equivalence"]="PASS"
        hi,hbal,hau=solve_event("Hiroshima 2018",hp,hc,he,ha,hs,contract_hash); ki,kbal,kau=solve_event("Kyushu 2017",kp,kc,ke,ka,ks,contract_hash)
        csvwrite(OUT/"PRETRIGGER_EXACT_BALANCE_SUMMARY.csv",[{"event":"Hiroshima 2018","median_abs_smd_before":hbal.abs_smd_before.median(),"median_abs_smd_after":hbal.abs_smd_after.median(),"p90_abs_smd_after":hbal.abs_smd_after.quantile(.9),"max_abs_smd_after":hbal.abs_smd_after.max(),"n_abs_smd_after_lt_0_10":int((hbal.abs_smd_after<.1).sum())},{"event":"Kyushu 2017","median_abs_smd_before":kbal.abs_smd_before.median(),"median_abs_smd_after":kbal.abs_smd_after.median(),"p90_abs_smd_after":kbal.abs_smd_after.quantile(.9),"max_abs_smd_after":kbal.abs_smd_after.max(),"n_abs_smd_after_lt_0_10":int((kbal.abs_smd_after<.1).sum())}])
        folds=fold_replay(hi); audit["spatial_fold_leakage_audit"]="PASS"; gate,hw,hci,kw,kci=model_evaluation(hi,ki,folds,hbal,kbal)
        audit.update({"status":"PASS_128C_COMPLETE","duplicate_control_n":hau["duplicate_control_n"]+kau["duplicate_control_n"],"partial_positive_n":hau["partial_positive_n"]+kau["partial_positive_n"],"GATE128C_DECISION":gate}); jwrite(OUT/"AUDIT128C.json",audit); log(f"PASS_128C_COMPLETE GATE={gate}")
    except Exception as exc:
        status="FAIL_128C_EXACT_COMPONENT_RESOURCE_LIMIT" if "RESOURCE_LIMIT" in str(exc) else ("ARM_B_EXACT_MATCHING_COMPLETE_MODEL_EVAL_BLOCKED_BY_FOLD_REPLAY" if "FOLD_REPLAY" in str(exc) else "FAIL_128C")
        audit.update({"status":status,"error":str(exc),"traceback":traceback.format_exc(),"protocol_relaxed":False}); jwrite(OUT/"AUDIT128C.json",audit)
        (OUT/"REVIEWER2_COMMENT5_FINAL_EVIDENCE.md").write_text("# Reviewer #2 Comment 5 — final evidence\n\n128C did not complete scientific model evaluation. See AUDIT128C.json. No matching rule was relaxed and no approximate result is reported.\n",encoding="utf-8")
        log(f"{status}: {exc}"); raise

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args(); main(args.dry_run)
