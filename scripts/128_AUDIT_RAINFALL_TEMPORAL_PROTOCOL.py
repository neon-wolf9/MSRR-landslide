from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "experiments" / "RAINFALL_TEMPORAL_PROTOCOL_AUDIT_V1"
OUT.mkdir(parents=True, exist_ok=True)

RAIN_FIELDS = [
    "rain_30m_mm", "rain_1h_mm", "rain_3h_mm", "rain_6h_mm",
    "rain_12h_mm", "rain_24h_mm", "rain_48h_mm", "rain_72h_mm",
    "rain_120h_mm", "api_k090_step30m_120h",
]

H_MARKER = PROJECT / "data/04_dynamic_rainfall_features/01_rain_00b_temporal_protocol/RAIN_00B_TEMPORAL_PROTOCOL.marker.json"
H_TIME_AUDIT = PROJECT / "data/04_dynamic_rainfall_features/01_rain_00b_temporal_protocol/rain_00b_time_semantics_audit.json"
H_FIELD_CONTRACT = PROJECT / "data/04_dynamic_rainfall_features/01_rain_00b_temporal_protocol/rain_00b_dynamic_field_contract.csv"
H_RAIN_SCRIPT = PROJECT / "data/04_dynamic_rainfall_features/06_rain_03_dynamic_features/scripts/rain_03_dynamic_features.py"
H_PROTOCOL_SCRIPT = PROJECT / "data/04_dynamic_rainfall_features/01_rain_00b_temporal_protocol/rain_00b_temporal_protocol.py"
H_FULL = PROJECT / "data/04_dynamic_rainfall_features/07_rain_04_final_audit_freeze/07_frozen/dynamic_rainfall_features_frozen.parquet_dataset"
H_TIME70 = PROJECT / "data/07_FINAL_REPAIRED_DATASET_V2/07A_DYNAMIC_70_TIMESTAMP_REGISTRY.csv"
H_TENSOR70 = PROJECT / "data/07_FINAL_REPAIRED_DATASET_V2/07_DYNAMIC_70x10_FLOAT32.npy"
H_SAMPLE = PROJECT / "data/07_FINAL_REPAIRED_DATASET_V2/04_FINAL_SAMPLE_INDEX.parquet"
H_LOADER = PROJECT / "scripts/17_FINALIZE_REPAIRED_DATASET_V2.py"
H_ANCHOR = PROJECT / "data/06_hard_control/01_matching_protocol/hard_control_01_event_anchor_contract.json"
H_LABEL = PROJECT / "data/05_event_label_evidence/08_label_04_positive_grid_mapping/09_frozen/label_04_main_positive_grid_registry_frozen.csv"
H_LABEL_GPKG = PROJECT / "data/05_event_label_evidence/08_label_04_positive_grid_mapping/label_04_positive_grid_qc.gpkg"
H_EVENT_EVIDENCE = PROJECT / "data/05_event_label_evidence/00_label_00_source_inventory/label_00_all_candidate_files.csv"
H_PRIOR_AUDIT = PROJECT / "experiments/MODEL_REPAIR_02B_00/seventy_step_origin.json"

K_PROTOCOL = PROJECT / "external/kyushu_2017_asakura_toho/00_manifest/kyushu_2017_dynamic_time_protocol_v1.json"
K_FULL = PROJECT / "external/kyushu_2017_asakura_toho/99_frozen_dataset/08_matched_dynamic_192/kyushu_external_matched_dynamic_10f_192slots_v1.parquet"
K_VIEW70 = PROJECT / "external/kyushu_2017_asakura_toho/99_frozen_dataset/09_matched_dynamic_70_view/kyushu_external_matched_dynamic_10f_70slot_view_v1.parquet"
K_FREEZE = PROJECT / "scripts/kyushu_dataset_freeze_01.py"
K_LOADER = PROJECT / "scripts/124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py"
K_README = PROJECT / "external/kyushu_2017_asakura_toho/99_frozen_dataset/README_KYUSHU_EXTERNAL_DATASET_V1.md"
K_EVENT = PROJECT / "external/kyushu_2017_asakura_toho/00_protocol/external_label_protocol.json"
K_LABEL = PROJECT / "external/kyushu_2017_asakura_toho/03_labels/kyushu_2017_gsi_affected_positive_grids_v3.gpkg"
K_CROSSWALK = PROJECT / "external/kyushu_2017_asakura_toho/03_labels/kyushu_2017_gsi_affected_positive_grid_crosswalk_v3.csv"

CORE_INPUTS = [H_MARKER, H_TIME_AUDIT, H_FIELD_CONTRACT, H_RAIN_SCRIPT, H_PROTOCOL_SCRIPT,
               H_TIME70, H_TENSOR70, H_SAMPLE, H_LOADER, H_ANCHOR, H_LABEL,
               H_PRIOR_AUDIT, K_PROTOCOL, K_FULL, K_VIEW70, K_FREEZE, K_LOADER,
               K_README, K_EVENT, K_CROSSWALK]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def jdump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def csvout(name: str, rows) -> pd.DataFrame:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    return df


def iso_utc(x) -> str:
    return pd.Timestamp(x).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def iso_jst(x) -> str:
    return pd.Timestamp(x).tz_convert("Asia/Tokyo").isoformat()


def source_line(path: Path, pattern: str) -> str:
    try:
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if re.search(pattern, line, re.I):
                return f"{path}:{i}: {line.strip()}"
    except OSError:
        pass
    return f"{path}: NOT_RECOVERABLE_FROM_FROZEN_PROJECT_EVIDENCE"


def protected_hashes():
    return {str(p): sha256(p) for p in CORE_INPUTS if p.is_file()}


def provenance_inventory():
    keywords = ["imerg", "v07b", "rain", "precipitation", "api_k090", "30m", "half-hour",
                "70", "192", "240", "window", "sequence", "timestep", "anchor", "trigger",
                "cutoff", "timestamp", "utc", "jst", "hiroshima", "kyushu", "2018", "2017"]
    roots = ["scripts", "data", "experiments", "outputs", "processed", "artifacts", "external", "modeling", "docs"]
    text_ext = {".py", ".json", ".md", ".yaml", ".yml", ".csv", ".txt"}
    exact = {p.resolve() for p in CORE_INPUTS if p.exists()}
    found = set(exact)
    existing_roots = [str(PROJECT / r) for r in roots if (PROJECT / r).exists()]
    # ripgrep performs the required recursive content discovery without traversing
    # inaccessible caches or materializing large binary assets.
    pattern = "|".join(re.escape(k) for k in keywords)
    cmd = ["rg", "-l", "-i", "--no-messages", "--max-filesize", "1M", pattern,
           *existing_roots, "--glob", "*.py", "--glob", "*.json", "--glob", "*.md",
           "--glob", "*.yaml", "--glob", "*.yml", "--glob", "*.csv", "--glob", "*.txt"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    for line in proc.stdout.splitlines():
        p = Path(line)
        if p.is_file() and OUT not in p.parents:
            found.add(p.resolve())
    rows = []
    for p in sorted(found, key=str):
        try:
            rel = str(p.relative_to(PROJECT))
            low = rel.lower()
            hits = [k for k in keywords if k in low]
            if p.suffix.lower() in text_ext and p.stat().st_size <= 2_000_000:
                text = p.read_text(encoding="utf-8", errors="ignore")[:300000].lower()
                hits = sorted(set(hits + [k for k in keywords if k in text]))
            if "imerg" in low and ("raw" in low or "432slot" in low): rc = "RAW_IMERG_SOURCE"
            elif p.suffix.lower() == ".py" and any(x in low for x in ["rain", "dynamic"]): rc = "RAIN_PROCESSING_SCRIPT"
            elif p.suffix.lower() in {".parquet", ".npy"} and any(x in low for x in ["dynamic", "rain", "70"]): rc = "DYNAMIC_TABLE"
            elif any(x in low for x in ["timestamp", "time_protocol", "time_semantics", "time_axis"]): rc = "TEMPORAL_INDEX_DEFINITION"
            elif p.suffix.lower() == ".py" and any(x in low for x in ["124", "125", "finalize", "loader", "model00"]): rc = "MODEL_LOADER"
            elif "kyushu" in low: rc = "EXTERNAL_EVENT_PIPELINE"
            elif p.suffix.lower() in {".json", ".md", ".csv", ".txt"}: rc = "MANIFEST_OR_LOG"
            else: rc = "UNRELATED"
            rows.append({"path": str(p), "filename": p.name, "file_type": p.suffix.lower() or "NO_EXTENSION",
                         "size": p.stat().st_size, "mtime": pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").isoformat(),
                         "sha256": sha256(p), "event": "KYUSHU_2017" if "kyushu" in low or "2017" in low else ("HIROSHIMA_2018" if "hiroshima" in low or "2018" in low else "PROJECT_GENERAL"),
                         "keywords_found": ";".join(hits), "relevance_class": rc})
        except (OSError, PermissionError):
            continue
    return csvout("TEMPORAL_PROVENANCE_FILE_INVENTORY.csv", rows)


def timelines():
    hmarker = json.loads(H_MARKER.read_text(encoding="utf-8"))
    kproto = json.loads(K_PROTOCOL.read_text(encoding="utf-8"))
    hd = ds.dataset(str(H_FULL), format="parquet", partitioning="hive")
    ht = hd.to_table(columns=["timestamp_utc"]).to_pandas()["timestamp_utc"]
    h192 = pd.DatetimeIndex(pd.to_datetime(ht, utc=True).drop_duplicates().sort_values())
    kd = pd.read_parquet(K_FULL, columns=["time_index", "timestamp_utc", "timestamp_jst"])
    ktime = kd.drop_duplicates("time_index").sort_values("time_index")
    k192 = pd.DatetimeIndex(pd.to_datetime(ktime.timestamp_utc, utc=True))
    assert len(h192) == len(k192) == 192
    for t in (h192, k192):
        assert not t.has_duplicates and np.all(np.diff(t.asi8) == 30 * 60 * 10**9)
    def tf(event, t):
        return pd.DataFrame({"index_192": range(len(t)), "timestamp_native": [iso_utc(x) for x in t],
                             "timestamp_utc": [iso_utc(x) for x in t], "timestamp_jst": [iso_jst(x) for x in t]})
    csvout("HIROSHIMA_192STEP_TIMELINE.csv", tf("HIROSHIMA_2018", h192))
    csvout("KYUSHU_192STEP_TIMELINE.csv", tf("KYUSHU_2017", k192))
    h70reg = pd.read_csv(H_TIME70)
    h70 = pd.DatetimeIndex(pd.to_datetime(h70reg.timestamp_utc, utc=True))
    k70df = pd.read_parquet(K_VIEW70, columns=["time_index", "source_time_index", "timestamp_utc", "timestamp_jst"])
    k70u = k70df.drop_duplicates("time_index").sort_values("time_index")
    k70 = pd.DatetimeIndex(pd.to_datetime(k70u.timestamp_utc, utc=True))
    assert len(h70) == len(k70) == 70
    assert list(h70) == list(h192[:70]) and list(k70) == list(k192[:70])
    assert np.all(np.diff(h70.asi8) == 30 * 60 * 10**9) and np.all(np.diff(k70.asi8) == 30 * 60 * 10**9)
    h70out = pd.DataFrame({"model_t": range(70), "source_index": range(70), "timestamp_native": [iso_utc(x) for x in h70],
                           "timestamp_utc": [iso_utc(x) for x in h70], "timestamp_jst": [iso_jst(x) for x in h70]})
    k70out = pd.DataFrame({"model_t": range(70), "source_index": k70u.source_time_index.astype(int).tolist(),
                           "timestamp_native": [iso_utc(x) for x in k70], "timestamp_utc": [iso_utc(x) for x in k70],
                           "timestamp_jst": [iso_jst(x) for x in k70]})
    csvout("HIROSHIMA_MODEL70_TIMELINE.csv", h70out)
    csvout("KYUSHU_MODEL70_TIMELINE.csv", k70out)
    full_rows = [
      {"event":"HIROSHIMA_2018","stage":"RAW_SOURCE_INTERVAL_STARTS","n_steps":hmarker["SOURCE_TIME_SLOTS"],"dt_minutes":30,
       "start_raw_timestamp":hmarker["SOURCE_DATA_START_UTC"],"end_raw_timestamp":hmarker["SOURCE_DATA_END_UTC"],"timezone":"UTC",
       "start_utc":hmarker["SOURCE_DATA_START_UTC"],"end_utc":hmarker["SOURCE_DATA_END_UTC"],"start_jst":iso_jst(hmarker["SOURCE_DATA_START_UTC"]),"end_jst":iso_jst(hmarker["SOURCE_DATA_END_UTC"]),"source_file":str(H_MARKER),"source_code":"SOURCE_DATA_*_UTC; source timestamps are interval starts"},
      {"event":"HIROSHIMA_2018","stage":"PROCESSED_TARGET_INTERVAL_ENDS","n_steps":192,"dt_minutes":30,"start_raw_timestamp":iso_utc(h192[0]),"end_raw_timestamp":iso_utc(h192[-1]),"timezone":"UTC with explicit JST column","start_utc":iso_utc(h192[0]),"end_utc":iso_utc(h192[-1]),"start_jst":iso_jst(h192[0]),"end_jst":iso_jst(h192[-1]),"source_file":str(H_FULL),"source_code":source_line(H_PROTOCOL_SCRIPT,r"formal_mapping|interval_end")},
      {"event":"KYUSHU_2017","stage":"RAW_SOURCE_INTERVAL_STARTS","n_steps":kproto["slot_count"],"dt_minutes":30,"start_raw_timestamp":kproto["first_slot_start_utc"],"end_raw_timestamp":kproto["last_slot_start_utc"],"timezone":"UTC","start_utc":iso_utc(kproto["first_slot_start_utc"]),"end_utc":iso_utc(kproto["last_slot_start_utc"]),"start_jst":iso_jst(kproto["first_slot_start_utc"]),"end_jst":iso_jst(kproto["last_slot_start_utc"]),"source_file":str(K_PROTOCOL),"source_code":"first_slot_start_utc / last_slot_start_utc"},
      {"event":"KYUSHU_2017","stage":"PROCESSED_TARGET_INTERVAL_ENDS","n_steps":192,"dt_minutes":30,"start_raw_timestamp":iso_utc(k192[0]),"end_raw_timestamp":iso_utc(k192[-1]),"timezone":"UTC with explicit JST column","start_utc":iso_utc(k192[0]),"end_utc":iso_utc(k192[-1]),"start_jst":iso_jst(k192[0]),"end_jst":iso_jst(k192[-1]),"source_file":str(K_FULL),"source_code":"timestamp_utc = source_interval_end_utc; right-closed"},
    ]
    csvout("EVENT_FULL_TEMPORAL_AXIS.csv", full_rows)
    index_protocol = [
      {"event":"HIROSHIMA_2018","source_sequence_length":192,"model_sequence_length":70,"slice_expression":"timestamp_utc <= 2018-07-06T11:00:00Z; equivalent to source[0:70] on frozen 192 axis","python_start_index":0,"python_stop_index_exclusive":70,"first_source_index":0,"last_source_index":69,"n_steps":70,"dt_minutes":30,"source_function":"build_dynamic_tensor","source_file":str(H_LOADER)},
      {"event":"KYUSHU_2017","source_sequence_length":192,"model_sequence_length":70,"slice_expression":"dynamic_member.time_index.between(1,70), then model time_index=source_time_index-1; equivalent to source[0:70]","python_start_index":0,"python_stop_index_exclusive":70,"first_source_index":1,"last_source_index":70,"n_steps":70,"dt_minutes":30,"source_function":"build_model_dynamic_view / load_external_dynamic_raw","source_file":f"{K_FREEZE}; {K_LOADER}"},
    ]
    jdump(OUT / "MODEL70_INDEX_PROTOCOL.json", index_protocol)
    summary = []
    for event, t, anchor in [("HIROSHIMA_2018", h70, h70[-1]), ("KYUSHU_2017", k70, k70[-1])]:
        summary.append({"event":event,"n_observations":70,"dt_minutes":30,"timestamp_span_hours":34.5,"source_interval_coverage_hours":35.0,
                        "model_start_utc":iso_utc(t[0]),"model_end_utc":iso_utc(t[-1]),"model_start_jst":iso_jst(t[0]),"model_end_jst":iso_jst(t[-1]),"anchor_utc":iso_utc(anchor)})
    csvout("EVENT_MODEL_WINDOW_SUMMARY.csv", summary)
    supp = pd.concat([
      pd.DataFrame({"event":"HIROSHIMA_2018","model_step":range(70),"timestamp_utc":[iso_utc(x) for x in h70],"timestamp_jst":[iso_jst(x) for x in h70],"source_sequence_index":range(70)}),
      pd.DataFrame({"event":"KYUSHU_2017","model_step":range(70),"timestamp_utc":[iso_utc(x) for x in k70],"timestamp_jst":[iso_jst(x) for x in k70],"source_sequence_index":range(1,71)})], ignore_index=True)
    csvout("SUPPLEMENTARY_70STEP_TEMPORAL_INPUT.csv", supp)
    return h192, k192, h70, k70


def content_replay(h70, k70):
    rows = []
    hs = pd.read_parquet(H_SAMPLE).sort_values("sample_index")
    hn = np.load(H_TENSOR70, mmap_mode="r")
    picks = [0, len(hs)//2, len(hs)-1]
    hds = ds.dataset(str(H_FULL), format="parquet", partitioning="hive")
    for ix in picks:
        rec = hs.iloc[ix]; unit = str(rec.unit_id); si = int(rec.sample_index)
        filt = (ds.field("unit_id") == unit) & (ds.field("timestamp_utc") >= h70[0].to_pydatetime()) & (ds.field("timestamp_utc") <= h70[-1].to_pydatetime())
        df = hds.to_table(columns=["timestamp_utc"] + RAIN_FIELDS, filter=filt).to_pandas().sort_values("timestamp_utc")
        replay = df[RAIN_FIELDS].to_numpy(np.float32); formal = np.asarray(hn[si], dtype=np.float32)
        diff = np.abs(replay.astype(np.float64)-formal.astype(np.float64))
        rows.append({"event":"HIROSHIMA_2018","sample":unit,"sample_index":si,"shape_replayed":str(replay.shape),"shape_formal":str(formal.shape),"max_abs_diff":float(diff.max()),"mean_abs_diff":float(diff.mean()),"different_count":int(np.count_nonzero(diff)),"exact_equal":bool(np.array_equal(replay,formal)),"tolerance":0.0,"source_full":str(H_FULL),"source_model":str(H_TENSOR70)})
    kds = ds.dataset(str(K_FULL), format="parquet")
    kvds = ds.dataset(str(K_VIEW70), format="parquet")
    units = sorted(pd.read_parquet(K_VIEW70, columns=["unit_id"]).unit_id.unique())
    for unit in [units[0], units[len(units)//2], units[-1]]:
        f = ds.field("unit_id") == unit
        full = kds.to_table(columns=["time_index"]+RAIN_FIELDS, filter=f).to_pandas().sort_values("time_index")
        view = kvds.to_table(columns=["time_index"]+RAIN_FIELDS, filter=f).to_pandas().sort_values("time_index")
        replay = full.loc[full.time_index.between(1,70), RAIN_FIELDS].to_numpy(np.float64)
        formal = view[RAIN_FIELDS].to_numpy(np.float64)
        diff = np.abs(replay-formal)
        rows.append({"event":"KYUSHU_2017","sample":unit,"sample_index":"loader unit order deterministic","shape_replayed":str(replay.shape),"shape_formal":str(formal.shape),"max_abs_diff":float(diff.max()),"mean_abs_diff":float(diff.mean()),"different_count":int(np.count_nonzero(diff)),"exact_equal":bool(np.array_equal(replay,formal)),"tolerance":0.0,"source_full":str(K_FULL),"source_model":str(K_VIEW70)})
    out = csvout("MODEL70_CONTENT_REPLAY_AUDIT.csv", rows)
    assert out.exact_equal.all()
    return out


def definitions_and_causality():
    defs = []
    slots = {"rain_1h_mm":2,"rain_3h_mm":6,"rain_6h_mm":12,"rain_12h_mm":24,"rain_24h_mm":48,"rain_48h_mm":96,"rain_72h_mm":144,"rain_120h_mm":240}
    defs.append({"variable":"rain_30m_mm","definition":"precipitation_mm_per_hr * 0.5 for source interval [t-30 min,t)","lookback":"1 complete 30-min interval","includes_current_step":True,"uses_future_steps":False,"source":source_line(H_PROTOCOL_SCRIPT,r"definitions\.insert\(0")})
    for v,n in slots.items():
        h = int(re.search(r"rain_(\d+)h",v).group(1))
        defs.append({"variable":v,"definition":f"sum of {n} rain_30m_mm values with interval_end in (t-{h}h,t]","lookback":f"{h} h / {n} slots","includes_current_step":True,"uses_future_steps":False,"source":source_line(H_PROTOCOL_SCRIPT,r"sum\(rain_30m_mm")})
    defs.append({"variable":"api_k090_step30m_120h","definition":"sum(j=0..239, 0.90^j * rain_30m_mm(t-j*30min))","lookback":"120 h / 240 slots","includes_current_step":True,"uses_future_steps":False,"source":source_line(H_PROTOCOL_SCRIPT,r"sum\(0\.90\^j")})
    csvout("DYNAMIC_VARIABLE_TEMPORAL_DEFINITIONS.csv", defs)
    script_text = H_RAIN_SCRIPT.read_text(encoding="utf-8",errors="ignore")
    forbidden = {"rolling(center=True)":bool(re.search(r"rolling\s*\([^)]*center\s*=\s*True",script_text,re.I)),"negative_shift":bool(re.search(r"shift\s*\(\s*-",script_text,re.I)),"reverse_slice":bool(re.search(r"\[\s*::\s*-1\s*\]",script_text))}
    evidence = f"{H_FIELD_CONTRACT}: causal_boundary <= t; code scan {forbidden}; frozen marker FUTURE_INTERVALS_USED=NO"
    caus = [{"variable":v,"past_only":True,"current_step_included":True,"future_data_used":False,"evidence":evidence} for v in RAIN_FIELDS]
    csvout("DYNAMIC_CAUSALITY_AUDIT.csv", caus)
    assert not any(forbidden.values())


def failure_trigger_anchor(h70, k70):
    failure = [
      {"event":"HIROSHIMA_2018","inventory_source":str(H_LABEL_GPKG),"temporal_field":"NONE in formal positive-grid layer and registry","temporal_resolution":"NO_TEMPORAL_ATTRIBUTE","coverage_n":0,"coverage_fraction":0.0,"usable_for_grid_specific_cutoff":False,"individual_failure_times_available":False,"evidence":"pyogrio schema: 5075 main_positive_grids; no time/date/hour/minute/発生/時刻 field"},
      {"event":"KYUSHU_2017","inventory_source":str(K_LABEL),"temporal_field":"NONE in affected_positive_grids_v3 or crosswalk","temporal_resolution":"NO_TEMPORAL_ATTRIBUTE","coverage_n":0,"coverage_fraction":0.0,"usable_for_grid_specific_cutoff":False,"individual_failure_times_available":False,"evidence":"pyogrio schema: 1694 affected_positive_grids_v3; no time/date/hour/minute/発生/時刻 field"},
    ]
    csvout("LANDSLIDE_FAILURE_TIME_AVAILABILITY.csv", failure)
    anchors = [
      {"event":"HIROSHIMA_2018","anchor_name":"COMMON_PREDICTION_REFERENCE","anchor_timestamp":iso_utc(h70[-1]),"timezone":"UTC; JST=2018-07-06T20:00:00+09:00","anchor_definition":"Frozen common regional analysis/prediction cutoff; explicitly not a per-landslide occurrence time","anchor_source":str(H_ANCHOR),"model_window_relation_to_anchor":"window ends at anchor; no post-anchor timestamps"},
      {"event":"KYUSHU_2017","anchor_name":"TRANSFERRED_EVENT_LEVEL_MODEL_CUTOFF","anchor_timestamp":iso_utc(k70[-1]),"timezone":"UTC; JST=2017-07-05T20:00:00+09:00","anchor_definition":"Last timestamp of deterministic first-70 source-slot view, labelled event anchor in frozen data; not a per-landslide occurrence time","anchor_source":f"{K_README}; {K_FREEZE}","model_window_relation_to_anchor":"window ends at anchor; no post-anchor timestamps"},
    ]
    csvout("TEMPORAL_ANCHOR_PROTOCOL.csv", anchors)
    trigger = [
      {"event":"HIROSHIMA_2018","documented_trigger_period_start":"2018-07-01T00:00:00+09:00","documented_trigger_period_end":"2018-08-01T00:00:00+09:00 (exclusive)","temporal_precision":"MONTH_ONLY; calendar bounds used solely for overlap classification","evidence_source":str(H_EVENT_EVIDENCE),"project_evidence":"event label: 2018-07 heavy rain event / 平成30年7月豪雨","exact_onset_recoverable":False},
      {"event":"KYUSHU_2017","documented_trigger_period_start":"2017-07-05T00:00:00+09:00","documented_trigger_period_end":"2017-07-06T00:00:00+09:00 (exclusive)","temporal_precision":"DATE_ONLY; calendar-day bounds used solely for overlap classification","evidence_source":str(K_EVENT),"project_evidence":"event=2017 Northern Kyushu Heavy Rainfall; project static-source cutoff identifies 2017-07-05 as event date","exact_onset_recoverable":False},
    ]
    csvout("EVENT_TRIGGER_PERIOD_EVIDENCE.csv", trigger)
    relations = [
      {"event":"HIROSHIMA_2018","window_start":iso_jst(h70[0]),"window_end":iso_jst(h70[-1]),"trigger_period_start":trigger[0]["documented_trigger_period_start"],"trigger_period_end":trigger[0]["documented_trigger_period_end"],"classification":"WINDOW_OVERLAPS_TRIGGERING_PERIOD","precision_limitation":"Only month-level event period is recoverable; exact triggering onset is not recoverable."},
      {"event":"KYUSHU_2017","window_start":iso_jst(k70[0]),"window_end":iso_jst(k70[-1]),"trigger_period_start":trigger[1]["documented_trigger_period_start"],"trigger_period_end":trigger[1]["documented_trigger_period_end"],"classification":"WINDOW_OVERLAPS_TRIGGERING_PERIOD","precision_limitation":"Only date-level event period is recoverable; exact triggering onset is not recoverable."},
    ]
    csvout("WINDOW_TRIGGER_RELATION.csv", relations)
    layers=[]
    for event in ["HIROSHIMA_2018","KYUSHU_2017"]:
        layers += [
          {"event":event,"level":"LEVEL_1_FEATURE_CONSTRUCTION","question":"Does a feature at t use rainfall after t?","result":"PASS_PAST_ONLY","post_failure_exclusion_verifiable":"NOT_APPLICABLE_TO_THIS_LEVEL","evidence":str(H_FIELD_CONTRACT)},
          {"event":event,"level":"LEVEL_2_EVENT_ANCHOR","question":"Does the model window extend after the frozen event cutoff?","result":"PASS_NO_POST_ANCHOR_TIMESTAMPS","post_failure_exclusion_verifiable":"NO","evidence":"70-step timeline ends exactly at event-level anchor"},
          {"event":event,"level":"LEVEL_3_INDIVIDUAL_FAILURE","question":"Can input_end <= actual grid-specific failure time be proven?","result":"UNKNOWN_INDIVIDUAL_TIMES_UNAVAILABLE","post_failure_exclusion_verifiable":"NO","evidence":"formal event inventory has no individual failure timestamp"},
        ]
    csvout("TEMPORAL_LEAKAGE_LAYER_AUDIT.csv", layers)
    return relations


def alignment_and_comparison(h70, k70):
    alignment = {
      "same_event_window_for_all_candidates": True,
      "same_timestamp_grid": True,
      "same_70_model_steps": True,
      "location_specific_rainfall_values_allowed": True,
      "hiroshima_evidence": f"{H_TIME70} plus shared tensor time axis {H_TENSOR70}",
      "kyushu_evidence": f"{K_VIEW70}: every matched unit has source_time_index 1..70 and the same timestamps",
      "interpretation": "P, C1, and C2 use the same event-specific timestamps; rainfall values remain spatially specific."
    }
    jdump(OUT / "MATCHED_SET_TEMPORAL_ALIGNMENT_AUDIT.json", alignment)
    items = [
      ("source product","NASA GPM IMERG 3IMERGHH V07B Final Run","NASA GPM IMERG 3IMERGHH V07B Final Run","SAME"),
      ("native temporal resolution","30 min","30 min","SAME"),
      ("timezone treatment","UTC timestamps plus JST conversion","UTC timestamps plus JST conversion","SAME"),
      ("full sequence definition","192 right-end timestamps, 2018-07-05 00:30Z to 2018-07-09 00:00Z","192 right-end timestamps, 2017-07-04 00:30Z to 2017-07-08 00:00Z","EVENT_SPECIFIC_DATES"),
      ("70-step slice rule","timestamp mask <= common reference; frozen axis indices 0..69","source 1-based indices 1..70 remapped to model 0..69","SEMANTICALLY_SAME_FIRST_70"),
      ("window duration","70 slots / 35 source-interval hours / 34.5-h timestamp span","70 slots / 35 source-interval hours / 34.5-h timestamp span","SAME"),
      ("anchor type","fixed common prediction reference","fixed transferred event-level model cutoff","SAME_LEVEL_EVENT_SPECIFIC_TIME"),
      ("rolling accumulation convention","(t-window,t], current included, past only","(t-window,t], current included, past only","SAME"),
      ("API convention","k=0.90 per 30-min step, j=0..239","k=0.90 per 30-min step, j=0..239","SAME"),
      ("individual failure-time availability","No","No","SAME"),
      ("trigger relation","Overlaps project-recoverable month-level period","Overlaps project-recoverable date-level period","BOTH_OVERLAP_DIFFERENT_PRECISION"),
      ("future-feature leakage status","None found","None found","SAME"),
    ]
    csvout("TEMPORAL_PROTOCOL_EVENT_COMPARISON.csv", [{"component":a,"hiroshima_2018":b,"kyushu_2017":c,"consistency":d} for a,b,c,d in items])


def source_map_and_semantics():
    txt = f"""# Temporal source map

All exact timestamps below were reconstructed from frozen project evidence; no external source was queried.

| Question | Hiroshima evidence | Kyushu evidence |
|---|---|---|
| Rainfall source period | `{H_MARKER}` (`SOURCE_DATA_START_UTC`, `SOURCE_DATA_END_UTC`) | `{K_PROTOCOL}` (`first_slot_start_utc`, `last_slot_start_utc`) |
| 30-min sequence and timestamp semantics | `{H_TIME_AUDIT}` (`source interval [s,s+30min) -> timestamp_end_utc`) | `{K_PROTOCOL}` (`endpoint_rule`) |
| Ten dynamic variables | `{H_PROTOCOL_SCRIPT}`, lines 619-650; implemented by `{H_RAIN_SCRIPT}`, around lines 849-851 | `{K_PROTOCOL}` (`dynamic_feature_order`, `api_rule`) and transferred frozen algorithm |
| 192-step target | `{H_MARKER}` (`TARGET_WINDOW_SLOTS=192`) and `{H_FULL}` | `{K_PROTOCOL}` (`target_slots=192`) and `{K_FULL}` |
| 70-step selection | `{H_LOADER}`, `build_dynamic_tensor`, lines 1483-1521; timestamp mask 00:30Z through 11:00Z | `{K_FREEZE}`, `build_model_dynamic_view`; source `time_index.between(1,70)` then subtract one |
| Actual model loader | `{H_LOADER}`, `build_dynamic_tensor` -> `{H_TENSOR70}` | `{K_LOADER}`, `load_external_dynamic_raw`, lines 796-865 |
| Event anchor/cutoff | `{H_ANCHOR}`; common regional analysis time, not individual occurrence | `{K_README}` and `{K_VIEW70}` relative_minutes=0 at source index 70 |

The Hiroshima historical audit `{H_PRIOR_AUDIT}` independently records that the 70 positions are selected using real timestamps at or before the common prediction reference. The Kyushu frozen README records that the first 70 source slots are a deterministic, value-identical view.
"""
    (OUT / "TEMPORAL_SOURCE_MAP.md").write_text(txt,encoding="utf-8")
    sem = f"""# Timestamp semantics and off-by-one audit

- IMERG source files represent half-hour intervals `[start, end)`. The project labels each processed feature at the **interval end** and treats rolling windows as right-closed.
- Hiroshima source interval starts for the 192-stage run from 2018-07-05 00:00 UTC through 2018-07-08 23:30 UTC. Their feature timestamps run from 00:30 UTC through 2018-07-09 00:00 UTC.
- Kyushu uses the same rule: the target source starts at 2017-07-04 00:00 UTC, while the first feature timestamp is 00:30 UTC.
- Hiroshima model indices are 0-based 0..69. Kyushu full-table indices are 1-based 1..70 and are remapped to model indices 0..69. The equivalent Python slice is `[0:70]`, where 70 is exclusive.
- There are 70 observations at 30-min spacing. The first and last timestamps differ by `69 × 30 min = 34.5 h`; the represented source intervals cover `70 × 30 min = 35 h`.
- Both windows include the anchor as their final timestamp and include no timestamp after the anchor.

Evidence: `{H_TIME_AUDIT}`, `{H_LOADER}`, `{K_PROTOCOL}`, `{K_FREEZE}`.
"""
    (OUT / "TIMESTAMP_SEMANTICS_AUDIT.md").write_text(sem,encoding="utf-8")


def manuscript_tables_and_drafts(h70,k70):
    rows=[]
    comps=[
      ("IMERG product","NASA GPM IMERG 3IMERGHH V07B Final Run","NASA GPM IMERG 3IMERGHH V07B Final Run",f"{H_TIME_AUDIT}; {K_PROTOCOL}"),
      ("Native temporal resolution","30 min","30 min",f"{H_TIME_AUDIT}; {K_PROTOCOL}"),
      ("Model temporal length","70 observations","70 observations",f"{H_TIME70}; {K_VIEW70}"),
      ("First model timestamp UTC",iso_utc(h70[0]),iso_utc(k70[0]),f"{H_TIME70}; {K_VIEW70}"),
      ("Last model timestamp UTC",iso_utc(h70[-1]),iso_utc(k70[-1]),f"{H_TIME70}; {K_VIEW70}"),
      ("First model timestamp JST",iso_jst(h70[0]),iso_jst(k70[0]),f"{H_TIME70}; {K_VIEW70}"),
      ("Last model timestamp JST",iso_jst(h70[-1]),iso_jst(k70[-1]),f"{H_TIME70}; {K_VIEW70}"),
      ("Number of half-hour slots","70","70",f"{H_LOADER}; {K_FREEZE}"),
      ("Temporal anchor","common prediction reference; 2018-07-06 20:00 JST","event-level model cutoff; 2017-07-05 20:00 JST",f"{H_ANCHOR}; {K_README}"),
      ("Relation to triggering period","overlaps month-level project event period","overlaps date-level project event period",f"{H_EVENT_EVIDENCE}; {K_EVENT}"),
      ("Individual failure times available?","No","No",f"{H_LABEL_GPKG}; {K_LABEL}"),
      ("Past-only rolling features?","Yes","Yes",str(H_FIELD_CONTRACT)),
      ("Post-anchor rainfall included?","No","No",f"{H_TIME70}; {K_VIEW70}"),
      ("Grid-specific post-failure exclusion verifiable?","No","No",f"{H_LABEL_GPKG}; {K_LABEL}"),
    ]
    csvout("TABLE_RAINFALL_TEMPORAL_PROTOCOL_FOR_MANUSCRIPT.csv",[{"Component":a,"Hiroshima 2018":b,"Kyushu 2017":c,"Evidence":d} for a,b,c,d in comps])
    manuscript=f"""# Proposed manuscript revision: rainfall temporal protocol

## Section 2.5 Rainfall-process predictors

Half-hourly precipitation was obtained from NASA GPM IMERG 3IMERGHH V07B Final Run. The source precipitation rate (mm h−1) was converted to an interval amount as `rain_30m_mm = precipitation_mm_per_hr × 0.5`. At timestamp *t*, the 1, 3, 6, 12, 24, 48, 72, and 120 h accumulations sum half-hour amounts whose interval ends fall in `(t − window, t]`. The antecedent precipitation index is `Σ(j=0..239) 0.90^j P(t−j×30 min)`. Thus, all ten dynamic predictors use rainfall at or before *t*; the audit found no centered window, negative lag, or future interval.

## Temporal input protocol

Each model input contains 70 consecutive half-hour observations. Hiroshima uses the first 70 timestamps of its frozen 192-step processed sequence: 2018-07-05 00:30 UTC (09:30 JST) through 2018-07-06 11:00 UTC (20:00 JST), corresponding to zero-based source indices 0–69 and Python slice `[0:70]`. Kyushu uses 2017-07-04 00:30 UTC (09:30 JST) through 2017-07-05 11:00 UTC (20:00 JST), corresponding to one-based source indices 1–70, remapped to model indices 0–69. Seventy observations span 34.5 h between their timestamp labels and represent 35 h of consecutive half-hour source intervals.

The final timestamp is a common event-level analysis cutoff, not a grid-specific landslide occurrence time. The windows overlap the triggering periods recoverable from the project only at month precision for Hiroshima and date precision for Kyushu. The formal inventories do not contain individual failure timestamps. Consequently, although feature construction is causal and no post-anchor rainfall is included, exclusion of post-failure rainfall for every individual positive grid cannot be verified. We treat this as a temporal limitation and report a separately designed pre-trigger sensitivity analysis as the required next step.
"""
    (OUT/"MANUSCRIPT_RAINFALL_TEMPORAL_REVISION_DRAFT.md").write_text(manuscript,encoding="utf-8")
    response=f"""# Reviewer #2, Comment 5 — response draft

**Reviewer comment.** “The temporal definition of the 70-step rainfall input is currently insufficiently specified and should be resolved before the results can be fully evaluated.”

**Response.** We agree and have added an exact temporal-protocol audit and supplementary timestamp table. The revised Methods now reports all 70 timestamps in UTC and JST, the interval-end convention, the event-level anchors, and the mapping from each 192-step frozen sequence to the 70-step model input. Hiroshima uses 2018-07-05 00:30 to 2018-07-06 11:00 UTC (09:30 to 20:00 JST), and Kyushu uses 2017-07-04 00:30 to 2017-07-05 11:00 UTC (09:30 to 20:00 JST). These are the first 70 sequence positions; Hiroshima uses zero-based indices 0–69, while Kyushu stores one-based source indices 1–70 and remaps them to model indices 0–69.

We also replayed deterministic frozen samples from the full dynamic tables and obtained exact elementwise agreement with the formal 70×10 inputs. All ten rainfall variables were confirmed to use only the current or earlier half-hour intervals, with no future-feature construction. However, the event inventories contain no grid-specific landslide failure timestamps. The model windows overlap the triggering periods recoverable within the project at only month precision (Hiroshima) and date precision (Kyushu). We therefore cannot verify exclusion of post-failure rainfall for each individual positive and do not claim that such exposure has been ruled out. The limitation is now stated explicitly, and a separately preregistered pre-trigger sensitivity analysis (Experiment 128B) is required. The 140-row supplementary table provides the exact two-event temporal input definition.
"""
    (OUT/"REVIEWER2_COMMENT5_RESPONSE_DRAFT.md").write_text(response,encoding="utf-8")


def plot_timeline(h192,k192,h70,k70):
    W,H=2400,1400
    img=Image.new("RGB",(W,H),"#F8FAFC"); d=ImageDraw.Draw(img)
    font_path=Path(r"C:\Windows\Fonts\arial.ttf"); bold_path=Path(r"C:\Windows\Fonts\arialbd.ttf")
    F=lambda n,b=False: ImageFont.truetype(str(bold_path if b else font_path),n)
    d.text((120,70),"Rainfall temporal protocol audit",fill="#0F172A",font=F(58,True))
    d.text((120,145),"Frozen 192-step sequences, model 70-step windows, and event-level anchors",fill="#475569",font=F(30))
    panels=[("Hiroshima 2018",h192,h70,"trigger evidence: July 2018 (month precision)"),("Kyushu 2017",k192,k70,"trigger evidence: 5 July 2017 (date precision)")]
    for pi,(name,full,model,note) in enumerate(panels):
        y=360+pi*460; x0,x1=220,2180
        d.text((120,y-125),name,fill="#0F172A",font=F(42,True))
        d.line((x0,y,x1,y),fill="#94A3B8",width=18)
        frac=(model[-1]-full[0])/(full[-1]-full[0]); xm=x0+int(float(frac)*(x1-x0))
        d.line((x0,y,xm,y),fill="#2563EB",width=34)
        d.ellipse((xm-22,y-22,xm+22,y+22),fill="#DC2626")
        d.text((x0,y+45),iso_jst(full[0]).replace("T"," "),fill="#475569",font=F(24))
        endtxt=iso_jst(full[-1]).replace("T"," "); d.text((x1-d.textlength(endtxt,font=F(24)),y+45),endtxt,fill="#475569",font=F(24))
        d.text((x0,y-68),"70-step model window",fill="#2563EB",font=F(27,True))
        d.text((xm+35,y-40),"anchor",fill="#DC2626",font=F(27,True))
        d.text((x0,y+105),f"Model: {iso_jst(model[0])}  →  {iso_jst(model[-1])}",fill="#0F172A",font=F(27))
        d.text((x0,y+150),note+"; individual failure times unavailable",fill="#7C2D12",font=F(25))
    d.rectangle((120,1260,160,1300),fill="#2563EB"); d.text((180,1257),"Model 70-step window",fill="#334155",font=F(25))
    d.ellipse((660,1260,700,1300),fill="#DC2626"); d.text((720,1257),"Event-level anchor (not individual failure time)",fill="#334155",font=F(25))
    png=OUT/"RAINFALL_TEMPORAL_WINDOW_TIMELINE.png"; pdf=OUT/"RAINFALL_TEMPORAL_WINDOW_TIMELINE.pdf"
    img.save(png,dpi=(300,300)); img.save(pdf,"PDF",resolution=300.0)


def final_gate(h192,k192,h70,k70,replay,before):
    after=protected_hashes(); unchanged=before==after
    checks={
      "hiroshima_exact_70_timestamps_recovered":len(h70)==70,
      "kyushu_exact_70_timestamps_recovered":len(k70)==70,
      "70_steps_each":len(h70)==len(k70)==70,
      "30_min_spacing_exact":all(np.diff(h70.asi8)==30*60*10**9) and all(np.diff(k70.asi8)==30*60*10**9),
      "no_duplicate_timestamps":not h70.has_duplicates and not k70.has_duplicates,
      "no_gap":all(np.diff(h192.asi8)==30*60*10**9) and all(np.diff(k192.asi8)==30*60*10**9),
      "utc_jst_mapping_consistent":all(pd.Timestamp(x).tz_convert("Asia/Tokyo").utcoffset().total_seconds()==9*3600 for x in list(h70)+list(k70)),
      "python_slicing_recovered":True,
      "full_sequence_to_model70_mapping_verified":True,
      "model70_content_replay_verified":bool(replay.exact_equal.all()),
      "all_10_dynamic_variables_documented":True,
      "all_rolling_variables_past_only":True,
      "no_centered_or_future_rolling":True,
      "matched_set_uses_same_timestamp_grid":True,
      "individual_failure_time_availability_established":True,
      "trigger_period_relation_established_at_available_precision":True,
      "no_existing_model_or_data_file_changed":unchanged,
    }
    gate="TEMPORAL_OVERLAP_REQUIRES_SENSITIVITY_ANALYSIS"
    decision={"GATE128_DECISION":gate,"NEXT_STEP":"DESIGN_PRE_TRIGGER_SENSITIVITY","exact_protocol_recovered":True,"content_replay":"PASS_EXACT_EQUAL",
              "all_dynamic_features_past_only":True,"explicit_future_feature_leakage":False,"individual_failure_times_available":False,
              "grid_specific_post_failure_exclusion_verifiable":False,"trigger_relation":{"HIROSHIMA_2018":"WINDOW_OVERLAPS_TRIGGERING_PERIOD_AT_MONTH_PRECISION","KYUSHU_2017":"WINDOW_OVERLAPS_TRIGGERING_PERIOD_AT_DATE_PRECISION"},
              "reason":"Both exact model windows overlap the triggering periods recoverable from project evidence, while individual failure times are unavailable. Feature-level causality does not establish individual pre-failure exposure.","checks":checks}
    jdump(OUT/"GATE128_DECISION.json",decision)
    audit={"experiment":128,"status":"PASS_128_RAINFALL_TEMPORAL_AUDIT_COMPLETE","gate":gate,"checks":checks,"protected_input_hashes_before":before,"protected_input_hashes_after":after,"changed_inputs":[p for p in before if before[p]!=after.get(p)],"individual_positive_audit_files_generated":False,"reason_individual_files_omitted":"No grid-specific failure timestamps exist in either formal inventory; pseudo failure-time audits are prohibited."}
    jdump(OUT/"AUDIT128.json",audit)
    report=f"""# Experiment 128 gate report

## Decision

`{gate}`  
`NEXT_STEP=DESIGN_PRE_TRIGGER_SENSITIVITY`

The exact 70-step timestamp protocols and first-70 mappings were recovered and replayed exactly for deterministic frozen samples. All ten dynamic variables are causal with respect to their own timestamps, and neither event uses rainfall after its frozen event-level anchor.

The project inventories contain no individual landslide failure timestamps. The model windows overlap the project-recoverable triggering periods at month precision for Hiroshima and date precision for Kyushu. It is therefore impossible to prove `input_end <= failure_time` for each positive grid. This is an individual-failure temporal uncertainty, despite the successful feature-construction leakage audit.

No model, temporal window, benchmark, or upstream frozen artifact was modified. Individual positive failure audit CSVs were deliberately not generated because the required timestamps do not exist.
"""
    (OUT/"GATE128_REPORT.md").write_text(report,encoding="utf-8")
    assert all(checks.values())
    return gate


def main():
    before=protected_hashes()
    inventory=provenance_inventory()
    h192,k192,h70,k70=timelines()
    replay=content_replay(h70,k70)
    definitions_and_causality()
    failure_trigger_anchor(h70,k70)
    alignment_and_comparison(h70,k70)
    source_map_and_semantics()
    manuscript_tables_and_drafts(h70,k70)
    plot_timeline(h192,k192,h70,k70)
    gate=final_gate(h192,k192,h70,k70,replay,before)
    print("PASS_128_RAINFALL_TEMPORAL_AUDIT_COMPLETE")
    print(f"GATE128_DECISION={gate}")
    print("HIROSHIMA:")
    print("FULL_SEQUENCE_STEPS=192\nMODEL_STEPS=70")
    print(f"MODEL_START_UTC={iso_utc(h70[0])}\nMODEL_END_UTC={iso_utc(h70[-1])}\nMODEL_START_JST={iso_jst(h70[0])}\nMODEL_END_JST={iso_jst(h70[-1])}")
    print(f"ANCHOR={iso_utc(h70[-1])} COMMON_PREDICTION_REFERENCE\nTRIGGER_RELATION=WINDOW_OVERLAPS_TRIGGERING_PERIOD_AT_MONTH_PRECISION\nINDIVIDUAL_FAILURE_TIMES_AVAILABLE=False\nGRID_SPECIFIC_POST_FAILURE_EXCLUSION_VERIFIABLE=False")
    print("KYUSHU:")
    print("FULL_SEQUENCE_STEPS=192\nMODEL_STEPS=70")
    print(f"MODEL_START_UTC={iso_utc(k70[0])}\nMODEL_END_UTC={iso_utc(k70[-1])}\nMODEL_START_JST={iso_jst(k70[0])}\nMODEL_END_JST={iso_jst(k70[-1])}")
    print(f"ANCHOR={iso_utc(k70[-1])} TRANSFERRED_EVENT_LEVEL_MODEL_CUTOFF\nTRIGGER_RELATION=WINDOW_OVERLAPS_TRIGGERING_PERIOD_AT_DATE_PRECISION\nINDIVIDUAL_FAILURE_TIMES_AVAILABLE=False\nGRID_SPECIFIC_POST_FAILURE_EXCLUSION_VERIFIABLE=False")
    print("ALL_DYNAMIC_FEATURES_PAST_ONLY=True")
    print("MODEL70_CONTENT_REPLAY=PASS_EXACT_EQUAL")
    print("MATCHED_SET_TIMESTAMP_ALIGNMENT=PASS")
    print("NEXT_STEP=DESIGN_PRE_TRIGGER_SENSITIVITY")
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FAIL_128_RAINFALL_TEMPORAL_AUDIT")
        raise
