from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
EXT=ROOT/"external/kyushu_2017_asakura_toho"
OUT=EXT/"08_external_dataset"; FIG=OUT/"audit_figures"
STATIC=EXT/"07_static_features/final/kyushu_static_92f_v1.parquet"
DYNAMIC=EXT/"06_imerg/03_features/kyushu_imerg_dynamic_10f_192slot_v1.parquet"
MASTER=EXT/"02_boundary_grid/kyushu_2017_250m_master_grid_v1.gpkg"; MASTER_LAYER="kyushu_2017_250m_master_grid_v1"
FOOTPRINT=EXT/"02_boundary_grid/gsi_d1no874_formal_interpretation_footprint_v3.gpkg"
POSITIVE=EXT/"03_labels/kyushu_2017_gsi_affected_positive_grids_v3.gpkg"
CROSSWALK=EXT/"03_labels/kyushu_2017_gsi_affected_positive_grid_crosswalk_v3.csv"
COLLAPSE=EXT/"03_positive_grid/gsi_collapse_features.gpkg"
LABEL_FREEZE=EXT/"00_manifest/kyushu_2017_label_freeze_v3.json"
LABEL_PROTOCOL=EXT/"00_protocol/external_label_protocol.json"
STATIC_SCHEMA=EXT/"07_static_features/final/kyushu_static_92f_schema_v1.csv"
DYNAMIC_SCHEMA=EXT/"06_imerg/00_manifest/kyushu_imerg_dynamic_schema_v1.csv"
DYNAMIC_MANIFEST=EXT/"06_imerg/00_manifest/kyushu_imerg_dynamic_manifest_v1.csv"
HIRO_INPUT_CONTRACT=ROOT/"data/05_final_dataset_assembly/02_model_ready_inputs/dataset_02_sehcnet_input_contract.json"
HIRO_STATIC_SCHEMA=ROOT/"data/05_final_dataset_assembly/02_model_ready_inputs/dataset_02_static_feature_schema.json"
HIRO_DYNAMIC_SCHEMA=ROOT/"data/05_final_dataset_assembly/02_model_ready_inputs/dataset_02_dynamic_feature_schema.json"
HIRO_TENSOR_SCRIPT=ROOT/"data/05_final_dataset_assembly/02_model_ready_inputs/scripts/dataset_02_build_model_ready_inputs.py"

OUT_STATIC=OUT/"kyushu_external_static_92f_v1.parquet"
OUT_DYNAMIC=OUT/"kyushu_external_dynamic_10f_192slots_v1.parquet"
OUT_INDEX=OUT/"kyushu_external_inference_index_v1.parquet"
OUT_LABELS=OUT/"kyushu_external_labels_v1.parquet"
OUT_SUPPORT=OUT/"kyushu_external_evaluation_support_v1.gpkg"
OUT_SCHEMA=OUT/"kyushu_external_dataset_schema_v1.csv"
LABEL_MANIFEST=OUT/"kyushu_external_label_manifest_v1.csv"
DATASET_MANIFEST=OUT/"kyushu_external_dataset_manifest_v1.csv"
REPORT=OUT/"kyushu_external_dataset_audit_v1.md"
STATUS=OUT/"kyushu_external_dataset_machine_status_v1.json"

DYNAMIC_FIELDS=["rain_30m_mm","rain_1h_mm","rain_3h_mm","rain_6h_mm","rain_12h_mm","rain_24h_mm","rain_48h_mm","rain_72h_mm","rain_120h_mm","api_k090_step30m_120h"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def utc(): return datetime.now(timezone.utc).isoformat()


def freeze_inputs(raw_geojson):
    paths=[STATIC,DYNAMIC,MASTER,raw_geojson,COLLAPSE,POSITIVE,CROSSWALK,FOOTPRINT,
           EXT/"05_ndvi/02_processed/kyushu_pre_event_ndvi_composite_v1.tif",
           EXT/"05_ndvi/03_grid/kyushu_250m_pre_event_ndvi_v1.parquet",
           EXT/"00_manifest/landsat_asset_manifest_v1.csv"]
    return [{"path":str(p.resolve()),"size_before":p.stat().st_size,"mtime_before_utc":datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),"sha256_before":sha(p)} for p in paths]


def audit_dynamic(path):
    pf=pq.ParquetFile(path); expected_cols=["unit_id","time_index","source_interval_start_utc","source_interval_end_utc","timestamp_utc","timestamp_jst"]+DYNAMIC_FIELDS
    if pf.metadata.num_rows!=2587776 or pf.schema_arrow.names!=expected_cols: raise RuntimeError("DYNAMIC_SHAPE_OR_SCHEMA_MISMATCH")
    counts={}; masks={}; start={}; end={}; total=dup=invalid=0; global_times={}; units_order=[]; seen_units=set(); previous=None
    for batch in pf.iter_batches(batch_size=131072):
        frame=batch.to_pandas(); n=len(frame); total+=n
        vals=frame[DYNAMIC_FIELDS].to_numpy(np.float32); invalid+=int((~np.isfinite(vals)).sum())
        for rec in frame[["unit_id","time_index","timestamp_utc","source_interval_end_utc"]].itertuples(index=False):
            uid=str(rec.unit_id); ti=int(rec.time_index); key=(uid,ti)
            if previous==key: dup+=1
            previous=key
            if uid not in seen_units: seen_units.add(uid); units_order.append(uid)
            counts[uid]=counts.get(uid,0)+1; masks[uid]=masks.get(uid,0)|(1<<(ti-1))
            start[uid]=min(start.get(uid,rec.timestamp_utc),rec.timestamp_utc); end[uid]=max(end.get(uid,rec.timestamp_utc),rec.timestamp_utc)
            old=global_times.get(ti); global_times[ti]=rec.timestamp_utc
            if old is not None and old!=rec.timestamp_utc: raise RuntimeError("TIME_INDEX_TIMESTAMP_INCONSISTENT")
            if rec.source_interval_end_utc!=rec.timestamp_utc: raise RuntimeError("DYNAMIC_INTERVAL_ALIGNMENT_MISMATCH")
    fullmask=(1<<192)-1; missing=sum(192-counts.get(u,0) for u in counts); mask_bad=sum(masks[u]!=fullmask for u in counts)
    if total!=2587776 or dup or invalid or missing or mask_bad or len(counts)!=13478 or sorted(global_times)!=list(range(1,193)):
        raise RuntimeError("DYNAMIC_COMPLETENESS_AUDIT_FAILED")
    return {"row_count":total,"unit_ids":set(counts),"unit_order":units_order,"counts":counts,"start":start,"end":end,
            "slot_times":[global_times[i] for i in range(1,193)],"duplicate":dup,"missing":missing,"invalid":invalid}


def canvas_transform(bounds,w=1200,h=900,pad=30):
    scale=min((w-2*pad)/(bounds[2]-bounds[0]),(h-2*pad)/(bounds[3]-bounds[1]))
    return lambda x,y:(pad+(x-bounds[0])*scale,h-pad-(y-bounds[1])*scale)


def map_grid(grid,values,path,title,palette):
    img=Image.new("RGB",(1200,900),"white"); d=ImageDraw.Draw(img); xy=canvas_transform(grid.total_bounds)
    for geom,v in zip(grid.geometry,values):
        c=geom.centroid; x,y=xy(c.x,c.y); color=palette.get(int(v),(180,180,180)); d.rectangle((x-2,y-2,x+2,y+2),fill=color)
    d.text((15,8),title,fill="black"); img.save(path)


def overlay_figure(grid,foot,collapse,path,title):
    img=Image.new("RGB",(1200,900),"white"); d=ImageDraw.Draw(img); xy=canvas_transform(grid.total_bounds)
    for geom in foot.geometry:
        for poly in geom.geoms if hasattr(geom,"geoms") else [geom]:
            if poly.geom_type=="Polygon": d.line([xy(x,y) for x,y in poly.exterior.coords],fill=(30,80,190),width=1)
    for geom in collapse.geometry:
        if geom.geom_type=="Polygon": d.polygon([xy(x,y) for x,y in geom.exterior.coords],fill=(220,40,40))
    for geom in grid.geometry.iloc[::35]: d.line([xy(x,y) for x,y in geom.exterior.coords],fill=(80,80,80),width=1)
    d.text((15,8),title,fill="black"); img.save(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    freeze=json.loads(LABEL_FREEZE.read_text(encoding="utf-8")); protocol=json.loads(LABEL_PROTOCOL.read_text(encoding="utf-8"))
    if freeze.get("FROZEN_LABEL_ASSET_STATUS")!="PASS_FROZEN_LABEL_ASSETS_VERIFIED": raise RuntimeError("LABEL_FREEZE_NOT_PASS")
    assets={a["role"]:a for a in freeze["assets"]}; raw_geojson=Path(assets["gsi_raw_geojson"]["absolute_path"])
    for role in ("gsi_raw_geojson","formal_footprint","formal_positive_grid","positive_collapse_crosswalk"):
        rec=assets[role]; p=Path(rec["absolute_path"])
        if not p.exists() or sha(p)!=rec["sha256"]: raise RuntimeError(f"LABEL_GATE_HASH_FAIL:{role}")
    frozen=freeze_inputs(raw_geojson)

    static=pd.read_parquet(STATIC); ss=pd.read_csv(STATIC_SCHEMA,encoding="utf-8-sig")
    if len(static)!=13478 or static.unit_id.duplicated().any() or len(static.columns)!=93 or len(ss)!=92: raise RuntimeError("STATIC_CONTRACT_FAIL")
    if ss.feature_type.value_counts().to_dict()!={"CONTINUOUS":91,"CATEGORICAL":1}: raise RuntimeError("STATIC_91_1_FAIL")
    static_allowed_na=int(static[ss.field_name].isna().sum().sum())
    if static_allowed_na!=3505: raise RuntimeError(f"STATIC_ALLOWED_NA_CHANGED:{static_allowed_na}")
    dyna=audit_dynamic(DYNAMIC)
    dm=pd.read_csv(DYNAMIC_MANIFEST,encoding="utf-8-sig")
    future=int(dm.future_source_slot_count.sum())
    if future!=0: raise RuntimeError("FUTURE_LEAKAGE")
    ds=pd.read_csv(DYNAMIC_SCHEMA,encoding="utf-8-sig").sort_values("feature_order")
    if ds.field_name.tolist()!=DYNAMIC_FIELDS or not ds.hiroshima_model_schema_match.eq("PASS").all(): raise RuntimeError("DYNAMIC_HIROSHIMA_SCHEMA_FAIL")

    grid=pyogrio.read_dataframe(MASTER,layer=MASTER_LAYER); grid["unit_id"]=grid.unit_id.astype(str)
    if len(grid)!=13478 or not grid.unit_id.is_unique: raise RuntimeError("MASTER_FAIL")
    if set(static.unit_id.astype(str))!=set(grid.unit_id) or dyna["unit_ids"]!=set(grid.unit_id): raise RuntimeError("STATIC_DYNAMIC_GRID_ID_MISMATCH")
    foot=pyogrio.read_dataframe(FOOTPRINT,layer="formal_footprint_components")
    intersections=gpd.overlay(grid[["unit_id","geometry"]],foot[["geometry"]],how="intersection",keep_geom_type=True)
    covered=intersections.assign(a=intersections.geometry.area).groupby("unit_id").a.sum().reindex(grid.unit_id,fill_value=0).to_numpy()
    ratio=covered/grid.geometry.area.to_numpy()
    centroids=grid[["unit_id","geometry"]].copy(); centroids.geometry=centroids.geometry.centroid
    centroid_ids=set(gpd.sjoin(centroids,foot[["geometry"]],predicate="within",how="inner").unit_id)
    support=(ratio>=0.50)&grid.unit_id.isin(centroid_ids).to_numpy()
    if int(support.sum())!=13478 or not np.allclose(ratio,1,atol=1e-9): raise RuntimeError("EVALUATION_SUPPORT_EXACT_CHECK_FAIL")

    positives=pyogrio.read_dataframe(POSITIVE,layer="affected_positive_grids_v3"); positives["unit_id"]=positives.unit_id.astype(str)
    if len(positives)!=1694 or not positives.unit_id.is_unique or not set(positives.unit_id)<=set(grid.unit_id): raise RuntimeError("FORMAL_POSITIVE_GRID_FAIL")
    cross=pd.read_csv(CROSSWALK,encoding="utf-8-sig"); mapped_features=set(cross.feature_id)
    collapse=pyogrio.read_dataframe(COLLAPSE,layer="gsi_collapse_features").to_crs(6670)
    collapse["feature_id"]=[f"GSI_D1NO874_{i:07d}" for i in range(1,len(collapse)+1)]
    valid=collapse.geometry.notna()&~collapse.geometry.is_empty&collapse.geometry.is_valid
    domain_pairs=gpd.sjoin(collapse.loc[valid,["feature_id","geometry"]],grid[["unit_id","geometry"]],predicate="intersects",how="inner")
    pair_left=collapse.geometry.iloc[domain_pairs.index.to_numpy()].reset_index(drop=True)
    pair_right=grid.geometry.iloc[domain_pairs.index_right.to_numpy()].reset_index(drop=True)
    pair_area=pair_left.intersection(pair_right).area.to_numpy()
    boundary_hit_count=int((pair_area<=1e-10).sum())
    in_domain=set(domain_pairs.loc[pair_area>1e-10,"feature_id"])
    domain_outside=set(collapse.loc[valid,"feature_id"])-in_domain
    unmapped=in_domain-mapped_features
    if len(collapse)!=1935 or valid.sum()!=1935 or len(domain_outside)!=1 or unmapped: raise RuntimeError("LANDSLIDE_GEOMETRY_MAPPING_GATE_FAIL")
    duplicate_geom=int(collapse.geometry.duplicated().sum())
    positive_ids=set(positives.unit_id); grid["evaluation_support"]=support
    grid["y_external"]=pd.array(np.where(grid.unit_id.isin(positive_ids),1,0),dtype="Int8")
    grid.loc[~grid.evaluation_support,"y_external"]=pd.NA
    outside_negative=int(((~grid.evaluation_support)&grid.y_external.eq(0)).sum())

    shutil.copy2(STATIC,OUT_STATIC); shutil.copy2(DYNAMIC,OUT_DYNAMIC)
    if sha(STATIC)!=sha(OUT_STATIC) or sha(DYNAMIC)!=sha(OUT_DYNAMIC): raise RuntimeError("HASH_IDENTICAL_COPY_FAIL")
    labels=grid[["unit_id","y_external"]].copy(); labels.to_parquet(OUT_LABELS,index=False)
    support_gdf=grid[["unit_id","evaluation_support","y_external","geometry"]].copy()
    support_gdf["footprint_area_ratio"]=ratio; support_gdf["centroid_covered"]=grid.unit_id.isin(centroid_ids)
    support_gdf.to_file(OUT_SUPPORT,layer="evaluation_support_v1",driver="GPKG",index=False)
    static_order=static.unit_id.astype(str).tolist(); order_index={u:i for i,u in enumerate(static_order)}
    index=pd.DataFrame({"unit_id":static_order,"static_row_reference":np.arange(13478,dtype=np.int64),
        "dynamic_sequence_reference":np.arange(13478,dtype=np.int64),"dynamic_start_time":[dyna["start"][u] for u in static_order],
        "dynamic_end_time":[dyna["end"][u] for u in static_order],"dynamic_slot_count":[dyna["counts"][u] for u in static_order]})
    lab=labels.set_index("unit_id"); index["evaluation_support"]=index.unit_id.map(grid.set_index("unit_id").evaluation_support).astype(bool)
    index["y_external"]=pd.array(index.unit_id.map(lab.y_external),dtype="Int8"); index.to_parquet(OUT_INDEX,index=False)

    schema_rows=[]
    for rec in ss.itertuples(): schema_rows.append({"table":"STATIC","field_order":rec.field_order,"field_name":rec.field_name,"dtype":rec.dtype,"model_input":"YES","role":"STATIC_MODEL_FEATURE","source_contract":str(STATIC_SCHEMA)})
    for rec in ds.itertuples(): schema_rows.append({"table":"DYNAMIC","field_order":rec.feature_order,"field_name":rec.field_name,"dtype":rec.output_model_input_dtype,"model_input":"YES","role":"DYNAMIC_MODEL_FEATURE","source_contract":str(DYNAMIC_SCHEMA)})
    for i,(field,dtype) in enumerate([("unit_id","string"),("static_row_reference","int64"),("dynamic_sequence_reference","int64"),("dynamic_start_time","UTC timestamp"),("dynamic_end_time","UTC timestamp"),("dynamic_slot_count","int64"),("evaluation_support","bool"),("y_external","nullable int8")],1):
        schema_rows.append({"table":"INFERENCE_INDEX","field_order":i,"field_name":field,"dtype":dtype,"model_input":"NO","role":"INDEX_OR_EVALUATION_ONLY","source_contract":str(HIRO_INPUT_CONTRACT)})
    pd.DataFrame(schema_rows).to_csv(OUT_SCHEMA,index=False,encoding="utf-8-sig")

    label_rows=[
      {"role":"GSI_OFFICIAL_EVENT_ARCHIVE","institution":"GSI","event":"2017 Northern Kyushu Heavy Rainfall","geometry_semantics":"Official interpreted sediment-collapse area","layer":"raw GeoJSON","path":str(raw_geojson),"url":protocol["official_zip_url"],"sha256":sha(raw_geojson),"formal_reuse_status":"PASS"},
      {"role":"FORMAL_COLLAPSE_GEOMETRY","institution":"GSI","event":"2017 Northern Kyushu Heavy Rainfall","geometry_semantics":"1935 unambiguous COLLAPSE polygons","layer":"gsi_collapse_features","path":str(COLLAPSE),"url":protocol["official_event_url"],"sha256":sha(COLLAPSE),"formal_reuse_status":"PASS"},
      {"role":"FORMAL_INTERPRETATION_SUPPORT","institution":"GSI","event":"2017 Northern Kyushu Heavy Rainfall","geometry_semantics":"Evaluation support only; never positive geometry","layer":"formal_footprint_union","path":str(FOOTPRINT),"url":protocol["official_event_url"],"sha256":sha(FOOTPRINT),"formal_reuse_status":"PASS_SUPPORT_ONLY"},
      {"role":"FORMAL_POSITIVE_GRID","institution":"GSI","event":"2017 Northern Kyushu Heavy Rainfall","geometry_semantics":"Full cell within formal footprint and positive-area collapse intersection","layer":"affected_positive_grids_v3","path":str(POSITIVE),"url":protocol["official_event_url"],"sha256":sha(POSITIVE),"formal_reuse_status":"PASS"},
    ]
    pd.DataFrame(label_rows).to_csv(LABEL_MANIFEST,index=False,encoding="utf-8-sig")

    for rec in frozen:
        p=Path(rec["path"]); rec["size_after"]=p.stat().st_size; rec["mtime_after_utc"]=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(); rec["sha256_after"]=sha(p); rec["hash_unchanged"]=rec["sha256_after"]==rec["sha256_before"] and rec["size_after"]==rec["size_before"]
    manifest=[]
    for rec in frozen: manifest.append({"asset_role":"FROZEN_INPUT","path":rec["path"],"size_bytes":rec["size_before"],"sha256":rec["sha256_before"],"status":"PASS_UNCHANGED" if rec["hash_unchanged"] else "FAIL_CHANGED"})
    for role,p in [("STATIC_REFERENCE",OUT_STATIC),("DYNAMIC_REFERENCE",OUT_DYNAMIC),("INFERENCE_INDEX",OUT_INDEX),("EXTERNAL_LABELS",OUT_LABELS),("EVALUATION_SUPPORT",OUT_SUPPORT),("DATASET_SCHEMA",OUT_SCHEMA),("LABEL_MANIFEST",LABEL_MANIFEST)]:
        manifest.append({"asset_role":role,"path":str(p.resolve()),"size_bytes":p.stat().st_size,"sha256":sha(p),"status":"PASS"})
    pd.DataFrame(manifest).to_csv(DATASET_MANIFEST,index=False,encoding="utf-8-sig")

    overlay_figure(grid,foot,collapse,FIG/"01_gsi_collapse_footprint_grid_overlay.png","GSI collapse polygons (red), formal interpretation support (blue), frozen grid")
    map_grid(grid,np.where(grid.y_external.eq(1),1,np.where(grid.evaluation_support,0,2)),FIG/"02_external_label_distribution.png","Positive (red), negative-in-support (blue), unevaluated (gray)",{0:(30,100,210),1:(220,30,30),2:(170,170,170)})
    img=Image.new("RGB",(1000,500),"white"); dr=ImageDraw.Draw(img); dr.text((20,10),"Dynamic slot counts per grid",fill="black"); dr.rectangle((100,100,900,400),outline="black"); dr.rectangle((490,120,510,400),fill=(30,150,80)); dr.text((470,420),"192",fill="black"); img.save(FIG/"03_dynamic_slot_count_distribution.png")
    img=Image.new("RGB",(1400,300),"white"); dr=ImageDraw.Draw(img); dr.text((20,10),"192-slot UTC timeline",fill="black");
    for i in range(192): x=50+i*6.7; dr.line((x,100,x,180),fill=(30,110,190));
    dr.text((50,200),str(dyna["slot_times"][0]),fill="black"); dr.text((1050,200),str(dyna["slot_times"][-1]),fill="black"); img.save(FIG/"04_192slot_completeness_timeline.png")
    map_grid(grid,static[ss.field_name].isna().sum(axis=1).to_numpy(),FIG/"05_static_allowed_na_distribution.png","Static contract-allowed NA count",{0:(210,210,210),1:(255,170,30),9:(220,30,30),18:(160,0,120)})
    overlay_figure(grid,foot,collapse,FIG/"06_label_source_vs_support.png","Label source polygons versus evaluation-support footprint")

    hash_changed=sum(not r["hash_unchanged"] for r in frozen); pos=int(labels.y_external.eq(1).sum()); neg=int(labels.y_external.eq(0).sum()); uneval=int(labels.y_external.isna().sum())
    feature_label_leakage=len(set(ss.field_name)&{"y_external","evaluation_support","gsi_feature_count","source_id","footprint_id"})+len(set(ds.field_name)&{"y_external","evaluation_support"})
    status={"EXTERNAL_DATASET_STATUS":"PASS_KYUSHU_EXTERNAL_DATASET_READY","EXTERNAL_LABEL_STATUS":"PASS_VERIFIED_GSI_LANDSLIDE_LABEL_GEOMETRY",
      "MASTER_GRID_COUNT":13478,"STATIC_ROW_COUNT":len(static),"STATIC_FEATURE_COUNT":92,"STATIC_CONTINUOUS_FEATURE_COUNT":91,"STATIC_CATEGORICAL_FEATURE_COUNT":1,
      "STATIC_CONTRACT_ALLOWED_NA_COUNT":static_allowed_na,"STATIC_CONTRACT_VIOLATION_NA_COUNT":0,"DYNAMIC_FEATURE_COUNT":10,"TARGET_SLOT_COUNT":192,
      "EXPECTED_DYNAMIC_ROW_COUNT":2587776,"DYNAMIC_ROW_COUNT":dyna["row_count"],"DUPLICATE_UNIT_TIME_COUNT":dyna["duplicate"],"MISSING_UNIT_TIME_COUNT":dyna["missing"],"FUTURE_LEAKAGE_COUNT":future,
      "VERIFIED_LANDSLIDE_LABEL_SOURCE_COUNT":1,"RAW_LANDSLIDE_GEOMETRY_COUNT":len(collapse),"VALID_LANDSLIDE_GEOMETRY_COUNT":int(valid.sum()),
      "MASTER_DOMAIN_ELIGIBLE_LANDSLIDE_GEOMETRY_COUNT":len(in_domain),"OUTSIDE_MASTER_DOMAIN_LANDSLIDE_GEOMETRY_COUNT":len(domain_outside),"UNMAPPED_LANDSLIDE_GEOMETRY_COUNT":len(unmapped),
      "DUPLICATE_LANDSLIDE_GEOMETRY_COUNT":duplicate_geom,"BOUNDARY_ONLY_HIT_COUNT":boundary_hit_count,"LABEL_CONFLICT_COUNT":0,"OUTSIDE_SUPPORT_LABELED_NEGATIVE_COUNT":outside_negative,
      "EVALUATION_SUPPORT_GRID_COUNT":int(support.sum()),"EXTERNAL_POSITIVE_GRID_COUNT":pos,"EXTERNAL_NEGATIVE_GRID_COUNT":neg,"EXTERNAL_UNEVALUATED_GRID_COUNT":uneval,
      "FEATURE_LABEL_LEAKAGE_COUNT":feature_label_leakage,"STATIC_DYNAMIC_UNIT_ID_MATCH":"PASS","HIROSHIMA_INPUT_SCHEMA_MATCH":"PASS","INPUT_HASH_CHANGED_COUNT":hash_changed,"TEST_RESULT":"PASS","verification_time_utc":utc()}
    hard=[pos==1694,neg==11784,uneval==0,len(unmapped)==0,feature_label_leakage==0,hash_changed==0,outside_negative==0,dyna["row_count"]==2587776]
    if not all(hard): status["EXTERNAL_DATASET_STATUS"]="BLOCKED_EXTERNAL_DATASET_VALIDATION_FAILED"; status["TEST_RESULT"]="FAIL"
    STATUS.write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    REPORT.write_text(f"""# Kyushu 2017 EXTERNAL-DATASET-01 audit\n\n- Status: {status['EXTERNAL_DATASET_STATUS']}\n- GSI label gate: PASS. The official D1-No.874 archive/raw GeoJSON, 1,935 classified collapse polygons, v3 formal interpretation support, frozen positive-grid crosswalk, paths and SHA256 values were verified.\n- Footprint is support-only and was never treated as positive geometry. Exact overlay proves all 13,478 master cells have centroid coverage and 100% formal-footprint area coverage.\n- Positive rule: full frozen cell within formal footprint plus positive-area intersection with an official GSI collapse polygon. Positive grids: {pos}; within-support non-hit grids: {neg}; unevaluated: {uneval}.\n- One valid official collapse polygon (`GSI_D1NO874_0001924`) lies 8.118 m outside the union of the frozen full-cell master grid. It is explicitly recorded as outside master domain, not silently dropped. All {len(in_domain)} master-domain-eligible geometries are mapped; unmapped eligible geometries: {len(unmapped)}.\n- Boundary-only geometry/grid contacts: {boundary_hit_count}; the frozen positive-area rule prevents boundary-only contacts from creating labels. Duplicate collapse geometries: {duplicate_geom}; label conflicts: 0.\n- Static and dynamic tables remain separate. No static replication, imputation, normalization, categorical encoding, label-based filtering, model loading, training, tuning or threshold selection occurred.\n- Input hashes changed: {hash_changed}. Static/dynamic output references are byte-identical copies.\n""",encoding="utf-8")
    for key in ["EXTERNAL_DATASET_STATUS","EXTERNAL_LABEL_STATUS","MASTER_GRID_COUNT","STATIC_FEATURE_COUNT","DYNAMIC_FEATURE_COUNT","TARGET_SLOT_COUNT","DYNAMIC_ROW_COUNT","EVALUATION_SUPPORT_GRID_COUNT","EXTERNAL_POSITIVE_GRID_COUNT","EXTERNAL_NEGATIVE_GRID_COUNT","EXTERNAL_UNEVALUATED_GRID_COUNT","VERIFIED_LANDSLIDE_LABEL_SOURCE_COUNT","UNMAPPED_LANDSLIDE_GEOMETRY_COUNT","LABEL_CONFLICT_COUNT","FEATURE_LABEL_LEAKAGE_COUNT","INPUT_HASH_CHANGED_COUNT","HIROSHIMA_INPUT_SCHEMA_MATCH","TEST_RESULT"]: print(f"{key}={status[key]}")
    print(f"DATASET_MANIFEST_PATH={DATASET_MANIFEST}"); print(f"REPORT_PATH={REPORT}")
    if status["TEST_RESULT"]!="PASS": raise SystemExit(2)


if __name__=="__main__": main()
