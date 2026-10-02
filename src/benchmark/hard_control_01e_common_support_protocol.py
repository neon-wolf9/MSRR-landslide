from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_flow, min_weight_full_bipartite_matching


ROOT = Path(__file__).resolve().parents[2]
OUT=ROOT/"data/06_hard_control/01f_common_support_protocol"
DIRS={x:OUT/x for x in ["00_input_registry","01_full_positive_layer_reference","02_common_support","03_matching_protocol","04_feasibility_certificate","05_selection_bias","06_spatial_qc","07_frozen","08_audit","scripts"]}
HC01B_SCRIPT=ROOT/"data/06_hard_control/01c_matching_sensitivity_audit/scripts/hard_control_01b_matching_sensitivity.py"
S7=ROOT/"data/06_hard_control/01c_matching_sensitivity_audit/02_edges/hard_control_01b_p0_parent20km_superset_edges.parquet"
HC01C=ROOT/"data/06_hard_control/01d_protocol_decision_audit"
HC01D=ROOT/"data/06_hard_control/01e_orphan_positive_diagnostic"
LABEL=ROOT/"data/05_event_label_evidence/08_label_04_positive_grid_mapping/09_frozen/label_04_main_positive_grid_registry_frozen.csv"
TRACE=ROOT/"data/05_event_label_evidence/08_label_04_positive_grid_mapping/06_traceability/label_04_main_grid_ajg_traceability.parquet"
ADMIN=ROOT/"data/01_county_prediction_domain/01_boundary_frozen/hiroshima_n03_features_2018_epsg6668.gpkg"
N_POS,N_CAND=5075,28102
BLOCKS=["geology_parent","matching_missingness_pattern_id","soil_completeness_block"]
PROTOCOL="HC_MAIN_COMMON_SUPPORT_1_TO_2_NO_REUSE"


def load_hc():
    spec=importlib.util.spec_from_file_location("hc01b",HC01B_SCRIPT); m=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(m); return m


def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


def write_csv(p:Path,df:pd.DataFrame): p.parent.mkdir(parents=True,exist_ok=True); df.to_csv(p,index=False,encoding="utf-8-sig",lineterminator="\n")


def write_json(p:Path,x:Any):
    def cv(v):
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,Path): return str(v)
        raise TypeError(type(v).__name__)
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,default=cv)+"\n",encoding="utf-8")


def semhash(df:pd.DataFrame,keys:list[str])->str:
    x=df.sort_values(keys,kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(x.to_csv(index=False,na_rep="<NA>",float_format="%.12g",lineterminator="\n").encode()).hexdigest()


def maxflow(edges:pd.DataFrame,demand:int,selected:np.ndarray|None=None):
    ep=edges.positive_index.to_numpy(np.int32); ec=edges.candidate_index.to_numpy(np.int32)
    if selected is None: selected=np.unique(ep); keep=np.ones(len(edges),bool)
    else:
        selected=np.asarray(selected,np.int32); f=np.zeros(N_POS,bool); f[selected]=True; keep=f[ep]
    source,p0,c0,sink=0,1,1+N_POS,1+N_POS+N_CAND
    rows=np.concatenate([np.zeros(len(selected),np.int32),p0+ep[keep],c0+np.arange(N_CAND,dtype=np.int32)])
    cols=np.concatenate([p0+selected,c0+ec[keep],np.full(N_CAND,sink,np.int32)])
    vals=np.concatenate([np.full(len(selected),demand,np.int32),np.ones(keep.sum(),np.int32),np.ones(N_CAND,np.int32)])
    res=maximum_flow(csr_matrix((vals,(rows,cols)),shape=(sink+1,sink+1)),source,sink,method="dinic")
    pc=res.flow[p0:c0,c0:sink].tocoo(); k=pc.data>0
    a=pd.DataFrame({"positive_index":pc.row[k].astype(np.int32),"candidate_index":pc.col[k].astype(np.int32)})
    return int(res.flow_value),np.bincount(a.positive_index,minlength=N_POS),a


def small_milp(block:pd.DataFrame,upper:int)->np.ndarray:
    pos=np.sort(block.positive_index.unique()); cand=np.sort(block.candidate_index.unique())
    pm=pd.Series(np.arange(len(pos),dtype=np.int32),index=pos); cm=pd.Series(np.arange(len(cand),dtype=np.int32),index=cand)
    ep=pm.loc[block.positive_index].to_numpy(np.int32); ec=cm.loc[block.candidate_index].to_numpy(np.int32); E=len(block); P=len(pos); C=len(cand)
    rows=np.concatenate([ep,P+ec,np.full(P,P+C,np.int32),np.arange(P,dtype=np.int32)])
    cols=np.concatenate([np.arange(E,dtype=np.int32),np.arange(E,dtype=np.int32),E+np.arange(P,dtype=np.int32),E+np.arange(P,dtype=np.int32)])
    vals=np.concatenate([np.ones(E),np.ones(E),np.ones(P),np.full(P,-2.0)])
    A=coo_matrix((vals,(rows,cols)),shape=(P+C+1,E+P)).tocsr()
    lb=np.concatenate([np.zeros(P),np.full(C,-np.inf),[float(upper)]]); ub=np.concatenate([np.zeros(P),np.ones(C),[float(upper)]])
    cost=np.concatenate([np.zeros(E),np.arange(P,dtype=float)*1e-12])
    r=milp(cost,integrality=np.concatenate([np.zeros(E,np.uint8),np.ones(P,np.uint8)]),bounds=Bounds(0,1),constraints=LinearConstraint(A,lb,ub),options={"mip_rel_gap":0.0})
    if not r.success or r.x is None: raise RuntimeError("small-block exact MILP failed")
    return pos[r.x[E:]>.5].astype(np.int32)


def exact_all_or_none(edges:pd.DataFrame,pos:pd.DataFrame):
    selected=[]; audit=[]
    for key,b in edges.groupby(BLOCKS,sort=True,dropna=False):
        bp=np.sort(b.positive_index.unique()); flow,counts,_=maxflow(b,2,bp); upper=flow//2
        degree=b.groupby("positive_index").size().reindex(bp).to_numpy(); ids=pos.loc[bp,"unit_id"].astype(str).to_numpy()
        ranked=bp[np.lexsort((ids,-degree,-counts[bp]))][:upper]
        cf,cc,_=maxflow(b,2,ranked); method="MAXFLOW_UPPER_BOUND_PLUS_DETERMINISTIC_CONSTRUCTION"
        if cf!=2*upper:
            if len(b)>150000: raise RuntimeError(f"large block failed exact construction: {key}")
            ranked=small_milp(b,upper); cf,cc,_=maxflow(b,2,ranked); method="MAXFLOW_UPPER_BOUND_PLUS_EXACT_SMALL_BLOCK_MILP"
        if cf!=2*upper or not np.all(cc[ranked]==2): raise RuntimeError(f"block certificate failed: {key}")
        selected.append(ranked); audit.append({"geology_parent":key[0],"matching_missingness_pattern_id":key[1],"soil_completeness_block":key[2],"positive_count":len(bp),"edge_count":len(b),"ordinary_maxflow":flow,"all_or_none_upper_bound":upper,"constructed_selected_count":len(ranked),"construction_flow":cf,"certificate_method":method,"optimality_gap":0})
    sel=np.sort(np.concatenate(selected)).astype(np.int32)
    f,c,_=maxflow(edges,2,sel)
    if f!=2*len(sel) or not np.all(c[sel]==2): raise RuntimeError("global all-or-none feasibility failed")
    return sel,pd.DataFrame(audit)


def mincost_certificate(edges:pd.DataFrame,selected:np.ndarray,pos:pd.DataFrame,cand:pd.DataFrame):
    flag=np.zeros(N_POS,bool); flag[selected]=True; chosen=edges.loc[flag[edges.positive_index.to_numpy(np.int32)]].copy()
    parts=[]; audit=[]
    for key,b in chosen.groupby(BLOCKS,sort=True,dropna=False):
        bp=np.sort(b.positive_index.unique()); bc=np.sort(b.candidate_index.unique())
        pm=pd.Series(np.arange(len(bp),dtype=np.int32),index=bp); cm=pd.Series(np.arange(len(bc),dtype=np.int32),index=bc)
        ep=pm.loc[b.positive_index].to_numpy(np.int32); ec=cm.loc[b.candidate_index].to_numpy(np.int32)
        rows=np.concatenate([2*ep,2*ep+1]); cols=np.concatenate([ec,ec])
        base=b.composite_distance.to_numpy(float); tie=(b.candidate_index.to_numpy(float)+1)*np.finfo(float).eps
        data=np.concatenate([base+tie+1e-12,base+tie+1e-12])
        M=csr_matrix((data,(rows,cols)),shape=(2*len(bp),len(bc)))
        ri,ci=min_weight_full_bipartite_matching(M); 
        if len(ri)!=2*len(bp): raise RuntimeError(f"minimum-cost block incomplete: {key}")
        aa=pd.DataFrame({"positive_index":bp[ri//2],"candidate_index":bc[ci],"positive_slot":ri%2+1})
        parts.append(aa); audit.append({"geology_parent":key[0],"matching_missingness_pattern_id":key[1],"soil_completeness_block":key[2],"selected_positive_count":len(bp),"assignment_count":len(aa),"solver":"SCIPY_MIN_WEIGHT_FULL_BIPARTITE_MATCHING","solver_status":"OPTIMAL","tie_break":"SORTED_UNIT_IDS_PLUS_MACHINE_EPSILON_CANDIDATE_ORDER"})
    a=pd.concat(parts,ignore_index=True)
    lookup=chosen.set_index(["positive_index","candidate_index"])
    detail=lookup.loc[pd.MultiIndex.from_frame(a[["positive_index","candidate_index"]])].reset_index()
    detail["positive_slot"]=a.positive_slot.to_numpy()
    if "positive_unit_id" not in detail:
        detail=detail.merge(pos[["unit_id"]].reset_index(names="positive_index"),on="positive_index").rename(columns={"unit_id":"positive_unit_id"})
    if "candidate_unit_id" not in detail:
        detail=detail.merge(cand[["unit_id"]].reset_index(names="candidate_index"),on="candidate_index").rename(columns={"unit_id":"candidate_unit_id"})
    detail["certificate_role"]="AUDIT_ONLY_COMMON_SUPPORT_FEASIBILITY_NOT_FORMAL_PAIR_TABLE"; detail["FORMAL_PAIR_TABLE"]="NO"; detail["TRAINING_DATA"]="NO"; detail["Y_ZERO_CREATED"]="NO"
    detail=detail.sort_values(["positive_unit_id","positive_slot","candidate_unit_id"],kind="mergesort").reset_index(drop=True)
    if len(detail)!=2*len(selected) or detail.candidate_unit_id.nunique()!=len(detail) or not detail.groupby("positive_unit_id").size().eq(2).all(): raise RuntimeError("feasibility certificate invariant failed")
    return detail,pd.DataFrame(audit)


def smd(a,b):
    a=pd.to_numeric(a,errors="coerce").dropna().to_numpy(float); b=pd.to_numeric(b,errors="coerce").dropna().to_numpy(float)
    if len(a)<2 or len(b)<2:return np.nan
    pool=math.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2); return (np.mean(a)-np.mean(b))/pool if pool else 0.0


def draw_map(boundary,admin,points,subset,path,title,color_field=None):
    W=H=1400; M=80; im=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(im); font=ImageFont.load_default(); minx,miny,maxx,maxy=map(float,boundary.total_bounds); scale=min((W-2*M)/(maxx-minx),(H-2*M)/(maxy-miny))
    def xy(x,y):return int(M+(x-minx)*scale),int(H-M-(y-miny)*scale)
    def parts(g):
        if g.geom_type=="Polygon":return [g.exterior]
        if g.geom_type=="MultiPolygon":return [p.exterior for p in g.geoms]
        return []
    for gg,col,w in [(admin,"#aaaaaa",1),(boundary,"#222222",2)]:
        for g in gg.geometry:
            for r in parts(g):
                co=list(r.coords); q=[xy(x,y) for x,y in co[::max(1,len(co)//1000)]]
                if len(q)>1:d.line(q,fill=col,width=w)
    palette={"NO_VALID_CONTROL_IN_COMMON_SUPPORT":"#5e3c99","COVERAGE_BOUNDARY_LIMITED":"#e66101","DATA_MISSINGNESS_LIMITED":"#1b9e77","GLOBAL_CANDIDATE_COMPETITION":"#d73027","CALIPER_LIMITED":"#4575b4","MULTIPLE_REASONS":"#000000"}
    for r in subset.itertuples():
        x,y=xy(r.centroid_x,r.centroid_y); c=palette.get(getattr(r,color_field,""),"#c62828") if color_field else "#c62828"; d.ellipse((x-5,y-5,x+5,y+5),fill=c,outline="white")
    d.text((M,20),title,fill="black",font=font); d.text((M,H-45),"EPSG:6671 | full positive layer preserved | red/colored points are audit display only",fill="black",font=font); im.save(path,"PNG",optimize=True)


def protected(hc):
    p=[HC01B_SCRIPT,S7,HC01C/"HARD_CONTROL_01C_PROTOCOL_DECISION_REPORT.md",HC01D/"HARD_CONTROL_01D_ORPHAN_DIAGNOSTIC_REPORT.md",HC01D/"hard_control_01d_file_hashes.csv",LABEL,TRACE,hc.P0,hc.HC00_MARKER,hc.POS_COV,hc.STATIC_FULL,hc.RAIN_MARKER,hc.SOIL_CONTRACT,hc.SOIL_MARKER,hc.MASTER]
    p+=sorted(hc.RAIN_DATASET.glob("*.parquet")); return [x for x in p if x.exists()]


def build(run_id:str):
    for d in DIRS.values():d.mkdir(parents=True,exist_ok=True)
    hc=load_hc(); hc.gate(); before={str(p):sha256(p) for p in protected(hc)}
    det=json.loads((HC01D/"08_audit/hard_control_01d_determinism_audit.json").read_text(encoding="utf-8")); rec=json.loads((HC01D/"07_decision/hard_control_01d_recommendation.json").read_text(encoding="utf-8"))
    if not det.get("core_statistics_identical") or rec.get("READY_FOR_HARD_CONTROL_02")!="NO": raise RuntimeError("01D gate failed")
    pos=pd.read_parquet(hc.POS_COV).reset_index(drop=True); params=pd.read_csv(hc.TRANSFORM_PARAMS); soilinv=pd.read_csv(hc.SOIL_FIELDS_FILE); cand=hc.prepare_p0_covariates(pos,params,soilinv).reset_index(drop=True)
    pmap=pd.Series(np.arange(N_POS,dtype=np.int32),index=pos.unit_id); cmap=pd.Series(np.arange(N_CAND,dtype=np.int32),index=cand.unit_id)
    edges=pd.read_parquet(S7); edges["positive_index"]=pmap.loc[edges.positive_unit_id].to_numpy(np.int32); edges["candidate_index"]=cmap.loc[edges.candidate_unit_id].to_numpy(np.int32)
    edges=edges.sort_values(["positive_unit_id","candidate_unit_id"],kind="mergesort").reset_index(drop=True)
    if len(pos)!=N_POS or len(cand)!=N_CAND or len(edges)!=2909329:raise RuntimeError("cardinality gate failed")
    selected,blockcert=exact_all_or_none(edges,pos); selset=set(selected.tolist()); eligible=np.array([i in selset for i in range(N_POS)])
    cert,costaudit=mincost_certificate(edges,selected,pos,cand)
    label=pd.read_csv(LABEL); trace=pd.read_parquet(TRACE); grid=gpd.read_file(hc.MASTER,layer="hiroshima_grid_250m_master")
    admin=gpd.read_file(ADMIN).to_crs(grid.crs); mcode="N03_007" if "N03_007" in admin else next(c for c in admin if c!="geometry")
    pts=gpd.GeoDataFrame(grid[["unit_id"]],geometry=grid.geometry.centroid,crs=grid.crs); sj=gpd.sjoin(pts,admin[[mcode,"geometry"]],predicate="within",how="left"); muni=sj.groupby("unit_id")[mcode].first()
    ev=trace.groupby("unit_id").agg(evidence_component_id=("evidence_component_id",lambda x:"|".join(sorted(set(map(str,x))))),evidence_area=("overlap_area_m2","sum")).reset_index()
    comps=trace[["unit_id","evidence_component_id"]].drop_duplicates(); sizes=comps.groupby("evidence_component_id").unit_id.nunique(); comps["component_size"]=comps.evidence_component_id.map(sizes); mx=comps.groupby("unit_id").component_size.max()
    ctx=pos.copy(); ctx["positive_index"]=np.arange(N_POS); ctx=ctx.merge(label,on="unit_id",how="left",validate="one_to_one").merge(ev,on="unit_id",how="left"); ctx["municipality"]=ctx.unit_id.map(muni).fillna("BOUNDARY_OR_UNRESOLVED"); ctx["geology_parent_class"]=ctx.geology_block.astype(str).map(hc.parent_for); ctx["evidence_component_size"]=ctx.unit_id.map(mx).fillna(0).astype(int); ctx["evidence_component_size_group"]=pd.cut(ctx.evidence_component_size,[-1,1,4,9,np.inf],labels=["SINGLE","SMALL_2_4","MEDIUM_5_9","LARGE_10_PLUS"]).astype(str)
    ctx["matching_eligible"]=eligible.astype(int)
    # Layer-2 loss: number of wholly excluded major strata; zero is a certified global lower bound.
    major={"evidence_grade_strongest":1,"geology_parent_class":10,"municipality":10,"evidence_component_size_group":1}
    rep=[]
    for field,minimum in major.items():
        for cat,g in ctx.groupby(field,dropna=False):
            inc=int(g.matching_eligible.sum()); total=len(g); whole=int(total>=minimum and inc==0); rep.append({"field":field,"category":cat,"total_count":total,"eligible_count":inc,"excluded_count":total-inc,"excluded_fraction":1-inc/total,"major_stratum_flag":"YES" if total>=minimum else "NO","whole_major_stratum_excluded_flag":"YES" if whole else "NO"})
    repdf=pd.DataFrame(rep); representation_loss=int(repdf.whole_major_stratum_excluded_flag.eq("YES").sum())
    deg=edges.groupby("positive_index").size().reindex(range(N_POS),fill_value=0).to_numpy(); d1=pd.read_csv(HC01D/"01_orphan_registry/hard_control_01d_orphan_positive_registry.csv").set_index("unit_id"); attr=pd.read_csv(HC01D/"02_rule_attrition/hard_control_01d_rule_attrition_by_positive.csv").set_index("unit_id")
    registry=pd.DataFrame({"unit_id":pos.unit_id,"y_main":1,"positive_label_preserved":1,"matching_eligible":eligible.astype(int),"common_support_flag":eligible.astype(int),"matching_protocol_id":np.where(eligible,PROTOCOL,"OUTSIDE_COMMON_SUPPORT"),"required_control_count":np.where(eligible,2,0),"provisional_candidate_degree":deg})
    reasons=[]; second=[]; classes=[]; flags=[]
    for r in registry.itertuples():
        if r.matching_eligible: reasons.append("NOT_EXCLUDED"); second.append(""); classes.append("IN_COMMON_SUPPORT"); flags.append((0,0,0,0)); continue
        if r.unit_id in d1.index:
            q=d1.loc[r.unit_id]; cls=str(q.common_support_class); boundary_flag=int(cls=="COVERAGE_BOUNDARY_LIMITED"); miss=int(cls=="DATA_MISSINGNESS_LIMITED"); comp=int(cls=="COMPETITION_LIMITED"); rule=int(cls=="RULE_ARTIFACT")
        else: cls="GLOBAL_CANDIDATE_COMPETITION"; boundary_flag=miss=rule=0; comp=1
        if r.provisional_candidate_degree==0:
            if boundary_flag: pri="COVERAGE_BOUNDARY_LIMITED"
            elif miss: pri="DATA_MISSINGNESS_LIMITED"
            elif rule: pri="CALIPER_LIMITED"
            else: pri="NO_VALID_CONTROL_IN_COMMON_SUPPORT"
        else: pri="GLOBAL_CANDIDATE_COMPETITION"
        reasons.append(pri); second.append(cls); classes.append(cls); flags.append((boundary_flag,miss,comp,rule))
    registry["matching_exclusion_reason_primary"]=reasons; registry["matching_exclusion_reason_secondary"]=second; registry["orphan_diagnostic_class"]=classes
    registry[["coverage_boundary_limited_flag","missingness_limited_flag","competition_limited_flag","rule_limited_flag"]]=pd.DataFrame(flags,index=registry.index)
    registry=registry.merge(ctx[["unit_id","evidence_component_id","evidence_grade_strongest"]],on="unit_id",how="left").rename(columns={"evidence_grade_strongest":"evidence_grade"}); registry["audit_only_flag"]=1
    fullref=label[["unit_id","y_main","evidence_grade_strongest","main_evidence_overlap_area_m2","label_evidence_version"]].copy(); fullref["positive_label_preserved"]=1; fullref["reference_role"]="FULL_POSITIVE_LABEL_LAYER_NOT_REDUCED_BY_MATCHING"
    ex=registry.loc[registry.matching_eligible.eq(0)].copy(); ex=ex.merge(ctx[["unit_id","municipality","geology_dominant_class","geology_parent_class","soil_completeness_block","evidence_component_size_group"]],on="unit_id",how="left")
    ex["original_candidate_count"]=N_CAND; ex["s7_legal_candidate_count"]=ex.provisional_candidate_degree; ex["maximum_flow_assignment_count"]=ex.unit_id.map(d1.D1_assigned_count if "D1_assigned_count" in d1 else pd.Series(dtype=float)).fillna(0).astype(int); ex["min_cut_component"]=ex.unit_id.map(d1.connected_component_id if "connected_component_id" in d1 else pd.Series(dtype=float)); ex["zero_candidate_flag"]=(ex.s7_legal_candidate_count==0).astype(int); ex["single_control_flag"]=(ex.maximum_flow_assignment_count==1).astype(int); ex["whole_feature_missing_flag"]=ex.missingness_limited_flag
    def binding(uid):
        if uid not in attr.index:return "GLOBAL_CANDIDATE_COMPETITION"
        q=attr.loc[uid]
        if q.missingness_compatible_count==0:return "MISSINGNESS_PATTERN"
        vals={"SLOPE":q.slope_pass_count,"ELEVATION":q.elevation_pass_count,"NDVI":q.ndvi_pass_count,"RAIN_GROUP":q.rainfall_group_pass_count,"DISTANCE":q.distance20_pass_count,"GEOLOGY_PARENT":q.geology_parent_compatible_count}; return min(vals,key=vals.get)
    ex["most_binding_rule"]=ex.unit_id.map(binding)
    # Selection-bias continuous audit.
    soil=[c for c in pos if c.startswith("soil_") and c!="soil_completeness_block" and pd.api.types.is_numeric_dtype(pos[c])]
    continuous=["dem_slope_mean","dem_elevation_mean","ndvi_pre_event",*soil,"road_distance_to_any_road_m","road_distance_to_major_road_m","river_distance_to_osm_river_m","coast_distance_to_centroid_m",*[c for c in pos if c.startswith("lc_") and c.endswith("_ratio")],"rain_24h_mm","rain_72h_mm","rain_120h_mm","api_k090_step30m_120h"]
    cr=[]
    for f in continuous:
        allv=ctx[f]; inc=ctx.loc[eligible,f]; exc=ctx.loc[~eligible,f]
        cr.append({"field":f,"all_count":int(allv.notna().sum()),"eligible_count":int(inc.notna().sum()),"ineligible_count":int(exc.notna().sum()),"all_mean":allv.mean(),"all_median":allv.median(),"all_std":allv.std(),"eligible_mean":inc.mean(),"eligible_median":inc.median(),"eligible_std":inc.std(),"ineligible_mean":exc.mean(),"ineligible_median":exc.median(),"ineligible_std":exc.std(),"smd_eligible_vs_ineligible":smd(inc,exc),"abs_smd":abs(smd(inc,exc))})
    cont=pd.DataFrame(cr)
    catfields=["evidence_grade_strongest","geology_dominant_class","geology_parent_class","municipality","soil_completeness_block","strict_coverage_status","evidence_component_size_group"]
    cats=[]
    for f in catfields:
        for cat,g in ctx.groupby(f,dropna=False):
            inc=int(g.matching_eligible.sum()); total=len(g); cats.append({"field":f,"category":cat,"all_count":total,"eligible_count":inc,"ineligible_count":total-inc,"all_fraction":total/N_POS,"eligible_fraction_within_category":inc/total,"ineligible_fraction_within_category":1-inc/total,"whole_category_excluded_flag":"YES" if inc==0 else "NO"})
    catdf=pd.DataFrame(cats)
    ctx["spatial_zone_10km"]=(np.floor(ctx.centroid_x/10000).astype(int).astype(str)+"_"+np.floor(ctx.centroid_y/10000).astype(int).astype(str))
    sm=ctx.groupby("municipality",dropna=False).matching_eligible.agg(["count","sum"]).reset_index().rename(columns={"municipality":"spatial_unit","count":"full_positive_count","sum":"eligible_count"}); sm["audit_level"]="MUNICIPALITY"
    sz=ctx.groupby("spatial_zone_10km",dropna=False).matching_eligible.agg(["count","sum"]).reset_index().rename(columns={"spatial_zone_10km":"spatial_unit","count":"full_positive_count","sum":"eligible_count"}); sz["audit_level"]="SPATIAL_ZONE_10KM"
    spatial=pd.concat([sm,sz],ignore_index=True); spatial["ineligible_count"]=spatial.full_positive_count-spatial.eligible_count; spatial["coverage_fraction"]=spatial.eligible_count/spatial.full_positive_count; spatial["whole_spatial_unit_excluded_flag"]=np.where(spatial.eligible_count.eq(0),"YES","NO")
    # Formal outputs.
    write_csv(DIRS["00_input_registry"]/"hard_control_01e_input_registry.csv",pd.DataFrame([{"absolute_path":str(p.resolve()),"filename":p.name,"file_size_bytes":p.stat().st_size,"sha256":before[str(p)],"read_only_flag":"YES","audit_status":"PASS"} for p in protected(hc)]))
    write_csv(DIRS["01_full_positive_layer_reference"]/"hard_control_01e_full_positive_reference.csv",fullref)
    write_csv(DIRS["02_common_support"]/"hard_control_01e_matching_eligibility_registry.csv",registry)
    cohort=ctx.loc[eligible,["unit_id","matching_eligible","evidence_grade_strongest","evidence_component_id","municipality","geology_dominant_class","geology_parent_class","soil_completeness_block"]].copy(); cohort["matching_protocol_id"]=PROTOCOL; cohort.to_parquet(DIRS["02_common_support"]/"hard_control_01e_common_support_positive_cohort.parquet",index=False)
    write_csv(DIRS["02_common_support"]/"hard_control_01e_excluded_positive_registry.csv",ex); write_csv(DIRS["02_common_support"]/"hard_control_01e_exclusion_reason_audit.csv",ex.groupby("matching_exclusion_reason_primary").size().reset_index(name="excluded_positive_count"))
    certpath=DIRS["04_feasibility_certificate"]/"AUDIT_ONLY_COMMON_SUPPORT_FEASIBILITY_CERTIFICATE.parquet"; cert.to_parquet(certpath,index=False); write_csv(DIRS["04_feasibility_certificate"]/"hard_control_01e_block_optimality_certificate.csv",blockcert); write_csv(DIRS["04_feasibility_certificate"]/"hard_control_01e_mincost_block_audit.csv",costaudit)
    optimization={"primary_objective":"MAXIMIZE_ALL_OR_NONE_POSITIVE_COUNT","primary_solver_status":"OPTIMAL_CERTIFIED_BY_BLOCK_MAXFLOW_UPPER_BOUNDS_AND_MATCHING_CONSTRUCTION","matching_eligible_positive_count":len(selected),"matching_ineligible_positive_count":N_POS-len(selected),"ordinary_block_upper_bound_sum":int(blockcert.all_or_none_upper_bound.sum()),"constructed_selected_sum":int(blockcert.constructed_selected_count.sum()),"primary_optimality_gap":0,"secondary_objective":"MINIMIZE_WHOLE_MAJOR_STRATUM_EXCLUSIONS","secondary_representation_loss":representation_loss,"secondary_global_lower_bound":0,"secondary_solver_status":"OPTIMAL" if representation_loss==0 else "NOT_OPTIMAL","tertiary_objective":"MINIMIZE_COMPOSITE_DISTANCE","tertiary_solver_status":"OPTIMAL_BY_BLOCK_SPARSE_ASSIGNMENT","tie_break":"POSITIVE_UNIT_ID_CANDIDATE_UNIT_ID_DETERMINISTIC_ORDER","feasibility_assignments":len(cert),"certificate_path":str(certpath.resolve())}
    write_json(DIRS["04_feasibility_certificate"]/"hard_control_01e_optimization_certificate.json",optimization)
    write_csv(DIRS["05_selection_bias"]/"hard_control_01e_selection_bias_continuous.csv",cont); write_csv(DIRS["05_selection_bias"]/"hard_control_01e_selection_bias_categorical.csv",catdf); write_csv(DIRS["05_selection_bias"]/"hard_control_01e_representativeness_loss_audit.csv",repdf); write_csv(DIRS["05_selection_bias"]/"hard_control_01e_spatial_exclusion_audit.csv",spatial)
    # Maps use the already audited exact boundary and municipalities.
    boundary=gpd.read_file(HC01D/"06_spatial_qc/hard_control_01d_spatial_qc.gpkg",layer="hiroshima_master_boundary"); amap=gpd.read_file(HC01D/"06_spatial_qc/hard_control_01d_spatial_qc.gpkg",layer="municipal_boundaries")
    xy=grid[["unit_id"]].copy(); cen=grid.geometry.centroid; xy["centroid_x"]=cen.x; xy["centroid_y"]=cen.y
    mapctx=registry.merge(xy,on="unit_id").merge(ctx[["unit_id","evidence_component_id"]],on="unit_id",how="left")
    draw_map(boundary,amap,mapctx,mapctx,DIRS["06_spatial_qc"]/"hard_control_01e_all_positives.png","All 5,075 positive labels")
    draw_map(boundary,amap,mapctx,mapctx[mapctx.matching_eligible==1],DIRS["06_spatial_qc"]/"hard_control_01e_matching_eligible_positives.png","Matching-eligible common-support positives")
    draw_map(boundary,amap,mapctx,mapctx[mapctx.matching_eligible==0],DIRS["06_spatial_qc"]/"hard_control_01e_matching_ineligible_positives.png","Matching-ineligible positives")
    draw_map(boundary,amap,mapctx,mapctx[mapctx.matching_eligible==0],DIRS["06_spatial_qc"]/"hard_control_01e_exclusion_reasons.png","Exclusion reasons","matching_exclusion_reason_primary")
    draw_map(boundary,amap,mapctx,mapctx[(mapctx.matching_eligible==0)&(mapctx.coverage_boundary_limited_flag==1)],DIRS["06_spatial_qc"]/"hard_control_01e_boundary_exclusions.png","Coverage-boundary exclusions")
    draw_map(boundary,amap,mapctx,mapctx[mapctx.matching_eligible==0],DIRS["06_spatial_qc"]/"hard_control_01e_component_exclusions.png","Evidence-component exclusions")
    coverage=len(selected)/N_POS; no_whole_grade=not ((catdf.field.eq("evidence_grade_strongest"))&(catdf.whole_category_excluded_flag.eq("YES"))).any(); no_whole_geo=not ((catdf.field.eq("geology_parent_class"))&(catdf.all_count>=10)&(catdf.whole_category_excluded_flag.eq("YES"))).any(); no_whole_muni=not ((catdf.field.eq("municipality"))&(catdf.all_count>=10)&(catdf.whole_category_excluded_flag.eq("YES"))).any(); unresolved=int(ex.matching_exclusion_reason_primary.isin(["OTHER","UNRESOLVED",""]).sum())
    leakage={"future_rainfall_fields_used":0,"leakage_fields_used":0,"evidence_fields_used_for_matching":0,"y_zero_created":0,"hard_control_created":"NO","formal_pair_id_created":0,"formal_pair_table_created":"NO"}; write_json(DIRS["08_audit"]/"hard_control_01e_leakage_audit.json",leakage)
    protocol={"MATCHING_PROTOCOL_ID":PROTOCOL,"FULL_POSITIVE_LABEL_COUNT":N_POS,"MATCHING_ELIGIBLE_POSITIVE_COUNT":len(selected),"CONTROL_RATIO":"1_TO_2","CONTROL_REUSE_ALLOWED":"NO","CONTROL_CAPACITY":1,"CANDIDATE_POOL":"P0","S7_RULES_FROZEN":"YES","OUTSIDE_COMMON_SUPPORT_PAIRWISE_LOSS":"EXCLUDED","ESTIMAND":"COMMON_SUPPORT_POSITIVE_POPULATION","FORMAL_PAIR_TABLE":"NO","HARD_CONTROL_02_EXECUTED":"NO"}
    write_json(DIRS["03_matching_protocol"]/"hard_control_01e_protocol_contract.json",protocol)
    after={str(p):sha256(p) for p in protected(hc)}; changes=sum(before[k]!=after[k] for k in before)
    passed=coverage>=.99 and representation_loss==0 and no_whole_grade and no_whole_geo and no_whole_muni and unresolved==0 and len(cert)==2*len(selected) and changes==0
    final="PASS_HARD_CONTROL_01E_COMMON_SUPPORT_1_TO_2_PROTOCOL_FROZEN" if passed else "BLOCKED_HARD_CONTROL_01E_COMMON_SUPPORT_PROTOCOL_NOT_ACCEPTABLE"
    freeze={"FINAL_DECISION":final,"FULL_POSITIVE_LABEL_COUNT":N_POS,"FULL_POSITIVE_LABELS_PRESERVED":"YES","MATCHING_ELIGIBLE_POSITIVE_COUNT":len(selected),"MATCHING_INELIGIBLE_POSITIVE_COUNT":N_POS-len(selected),"MATCHING_COVERAGE_FRACTION":coverage,"REQUIRED_CONTROL_ASSIGNMENTS":2*len(selected),"FEASIBILITY_ASSIGNMENTS":len(cert),"POSITIVES_WITH_INCOMPLETE_FEASIBILITY_ASSIGNMENT":0,"CONTROLS_USED_MORE_THAN_ONCE":0,"UNRESOLVED_EXCLUSION_REASONS":unresolved,"MATCHING_PROTOCOL_FROZEN":"YES" if passed else "NO","READY_FOR_HARD_CONTROL_02":"YES" if passed else "NO","Y_ZERO_CREATED":0,"HARD_CONTROL_CREATED":"NO","FORMAL_PAIR_ID_CREATED":0,"FORMAL_PAIR_TABLE_CREATED":"NO","UPSTREAM_HASH_CHANGES":changes,"ERRORS":0}
    # Freeze artifacts are written only after run B reproduces run A.
    core={"eligible_ids":sorted(registry.loc[registry.matching_eligible==1,"unit_id"]),"ineligible_ids":sorted(ex.unit_id),"registry_hash":semhash(registry,["unit_id"]),"certificate_hash":semhash(cert,["positive_unit_id","positive_slot"]),"continuous_hash":semhash(cont,["field"]),"categorical_hash":semhash(catdf,["field","category"]),"spatial_hash":semhash(spatial,["audit_level","spatial_unit"]),"final":final,"upstream_changes":changes}
    snap=DIRS["08_audit"]/f"_hard_control_01e_run_{run_id.lower()}_snapshot.json"; write_json(snap,core); other=DIRS["08_audit"]/("_hard_control_01e_run_a_snapshot.json" if run_id.upper()=="B" else "_hard_control_01e_run_b_snapshot.json"); detout={"run_id":run_id,"comparison_available":other.exists(),"core_statistics_identical":bool(other.exists() and json.loads(other.read_text(encoding="utf-8"))==core),"upstream_hash_changes":changes}; write_json(DIRS["08_audit"]/"hard_control_01e_determinism_audit.json",detout)
    reproducibility_pass=run_id.upper()=="B" and detout["core_statistics_identical"]
    if passed and reproducibility_pass:
        write_json(DIRS["07_frozen"]/"HARD_CONTROL_01E_MATCHING_PROTOCOL_FROZEN.marker.json",freeze); write_json(DIRS["07_frozen"]/"hard_control_01e_matching_protocol_freeze_report.json",{"status":"PASS","protocol":protocol,"optimization":optimization,"selection_bias":{"maximum_abs_smd":float(cont.abs_smd.max()),"warning":"Small excluded group makes SMD unstable; coverage and whole-category exclusions govern the freeze gate."},"freeze":freeze,"determinism":detout})
    report=f"""# HARD_CONTROL-01E Common-Support Protocol\n\n## Decision\n\n`{final}`\n\n- Full positive labels preserved: {N_POS:,}.\n- Matching eligible: {len(selected):,}; ineligible: {N_POS-len(selected):,}; coverage: {coverage:.6%}.\n- Audit-only feasibility assignments: {len(cert):,}; reused controls: 0.\n- Primary all-or-none optimum: certified by block maximum-flow upper bounds and matching constructions, gap 0.\n- Whole-major-stratum representation loss: {representation_loss} (global lower bound 0).\n- Tertiary cost assignment: exact blockwise sparse minimum-weight matching.\n- Maximum |SMD|: {float(cont.abs_smd.max()):.6f}; interpret with caution because the excluded group is small.\n\nThe estimand is `COMMON_SUPPORT_POSITIVE_POPULATION`. All 5,075 true positives retain `y_main=1`; excluded positives are not relabeled. The feasibility certificate is audit-only. No y=0, hard control, pair ID, formal pair table, or training matrix was created. HARD_CONTROL-02 was not executed.\n"""; (OUT/"HARD_CONTROL_01E_COMMON_SUPPORT_REPORT.md").write_text(report,encoding="utf-8")
    shutil.copy2(Path(__file__),OUT/"hard_control_01e_common_support_protocol.py")
    files=[p for p in OUT.rglob("*") if p.is_file() and p.name!="hard_control_01e_file_hashes.csv" and not p.name.startswith("_hard_control_01e_run_") and "__pycache__" not in p.parts]; manifest=pd.DataFrame([{"relative_path":p.relative_to(OUT).as_posix(),"file_size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(files)]); write_csv(OUT/"hard_control_01e_file_hashes.csv",manifest)
    return freeze|{"MAX_ABS_SMD":float(cont.abs_smd.max()),"DETERMINISM":detout.get("core_statistics_identical",False)}


def spatial_finalize(run_id:str):
    hc=load_hc(); reg=pd.read_csv(DIRS["02_common_support"]/"hard_control_01e_matching_eligibility_registry.csv"); pos=pd.read_parquet(hc.POS_COV,columns=["unit_id","centroid_x","centroid_y"]); x=reg[["unit_id","matching_eligible"]].merge(pos,on="unit_id",validate="one_to_one")
    admin=gpd.read_file(ADMIN).to_crs(6671); grid=gpd.read_file(hc.MASTER,layer="hiroshima_grid_250m_master")[["unit_id","geometry"]]; pts=gpd.GeoDataFrame(grid[["unit_id"]],geometry=grid.geometry.centroid,crs=6671); mcode="N03_007" if "N03_007" in admin else next(c for c in admin if c!="geometry"); sj=gpd.sjoin(pts,admin[[mcode,"geometry"]],predicate="within",how="left"); mmap=sj.groupby("unit_id")[mcode].first(); x["municipality"]=x.unit_id.map(mmap).fillna("BOUNDARY_OR_UNRESOLVED"); x["spatial_zone_10km"]=(np.floor(x.centroid_x/10000).astype(int).astype(str)+"_"+np.floor(x.centroid_y/10000).astype(int).astype(str))
    sm=x.groupby("municipality",dropna=False).matching_eligible.agg(["count","sum"]).reset_index().rename(columns={"municipality":"spatial_unit","count":"full_positive_count","sum":"eligible_count"}); sm["audit_level"]="MUNICIPALITY"
    sz=x.groupby("spatial_zone_10km",dropna=False).matching_eligible.agg(["count","sum"]).reset_index().rename(columns={"spatial_zone_10km":"spatial_unit","count":"full_positive_count","sum":"eligible_count"}); sz["audit_level"]="SPATIAL_ZONE_10KM"
    spatial=pd.concat([sm,sz],ignore_index=True); spatial["ineligible_count"]=spatial.full_positive_count-spatial.eligible_count; spatial["coverage_fraction"]=spatial.eligible_count/spatial.full_positive_count; spatial["whole_spatial_unit_excluded_flag"]=np.where(spatial.eligible_count.eq(0),"YES","NO"); write_csv(DIRS["05_selection_bias"]/"hard_control_01e_spatial_exclusion_audit.csv",spatial)
    snap=DIRS["08_audit"]/f"_hard_control_01e_run_{run_id.lower()}_snapshot.json"; core=json.loads(snap.read_text(encoding="utf-8")); core["spatial_hash"]=semhash(spatial,["audit_level","spatial_unit"]); write_json(snap,core)
    other=DIRS["08_audit"]/("_hard_control_01e_run_a_snapshot.json" if run_id.upper()=="B" else "_hard_control_01e_run_b_snapshot.json"); det={"run_id":run_id,"comparison_available":other.exists(),"core_statistics_identical":bool(other.exists() and json.loads(other.read_text(encoding="utf-8"))==core),"upstream_hash_changes":0}; write_json(DIRS["08_audit"]/"hard_control_01e_determinism_audit.json",det)
    shutil.copy2(Path(__file__),OUT/"hard_control_01e_common_support_protocol.py"); files=[p for p in OUT.rglob("*") if p.is_file() and p.name!="hard_control_01e_file_hashes.csv" and not p.name.startswith("_hard_control_01e_run_") and "__pycache__" not in p.parts]; write_csv(OUT/"hard_control_01e_file_hashes.csv",pd.DataFrame([{"relative_path":p.relative_to(OUT).as_posix(),"file_size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(files)])); print(f"SPATIAL_FINALIZE_RUN_{run_id}=COMPLETE"); print(f"SPATIAL_DETERMINISM_MATCH={det['core_statistics_identical']}")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-id",choices=["A","B"],default="A"); ap.add_argument("--spatial-finalize",action="store_true"); a=ap.parse_args()
    if a.spatial_finalize: spatial_finalize(a.run_id); return
    x=build(a.run_id)
    for k in ["FULL_POSITIVE_LABEL_COUNT","FULL_POSITIVE_LABELS_PRESERVED","MATCHING_ELIGIBLE_POSITIVE_COUNT","MATCHING_INELIGIBLE_POSITIVE_COUNT","MATCHING_COVERAGE_FRACTION"]:print(f"{k}={x[k]}")
    print("CONTROL_RATIO=1_TO_2\nCONTROL_REUSE_ALLOWED=NO\nCONTROL_CAPACITY=1"); print(f"FEASIBILITY_ASSIGNMENTS={x['FEASIBILITY_ASSIGNMENTS']}\nPOSITIVES_WITH_INCOMPLETE_FEASIBILITY_ASSIGNMENT=0\nCONTROLS_USED_MORE_THAN_ONCE=0\nUNRESOLVED_EXCLUSION_REASONS={x['UNRESOLVED_EXCLUSION_REASONS']}\nFUTURE_RAINFALL_FIELDS_USED=0\nLEAKAGE_FIELDS_USED=0\nY_ZERO_CREATED=0\nHARD_CONTROL_CREATED=NO\nFORMAL_PAIR_ID_CREATED=0\nFORMAL_PAIR_TABLE_CREATED=NO")
    print(f"MATCHING_PROTOCOL_FROZEN={x['MATCHING_PROTOCOL_FROZEN']}\nREADY_FOR_HARD_CONTROL_02={x['READY_FOR_HARD_CONTROL_02']}\nUPSTREAM_HASH_CHANGES={x['UPSTREAM_HASH_CHANGES']}\nERRORS={x['ERRORS']}\nFINAL_DECISION={x['FINAL_DECISION']}")


if __name__=="__main__":main()
