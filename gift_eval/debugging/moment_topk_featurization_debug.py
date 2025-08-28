# %%
# MOMENT Top-K Retrieval + Segment Featurization Debug (VS Code Interactive)
#
# Validates that:
# - We reuse an existing persistent ChromaDB collection (no reindex)
# - Retrieval returns top_k subsequences per item (when available)
# - Each retrieved subsequence (window tail + horizon) is treated as its own segment
# - Feature generation resets per segment (e.g., running_index starts at 0 for each segment)

from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import time
import logging


# Ensure repository root is on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gift_eval.data import Dataset
from gift_eval.tabpfn_ts_wrapper import TabPFNTSPredictor, TabPFNMode
from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame


def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# %% [markdown]
# Configuration: choose dataset and retrieval settings

# %%
# Pick a dataset that already has a matching persistent collection in ChromaDB
# Examples available in your environment include (names inferred from collection ids):
# - "bizitobs_service" (freq=10S, P=60)
# - "bizitobs_application_10S" (freq=10S, P=60)
# - "SZ_TAXI/H" (freq=H, P=48)
# - "M_DENSE/D" (freq=D, P=30)
# - "LOOP_SEATTLE/D" (freq=D, P=30)
# - "hierarchical_sales/D" (freq=D, P=30)

DATASET_NAME = "M_DENSE/D"
TERM = "short"

# Retrieval hyperparameters
CONTEXT_LENGTH = 30
FEW_SHOT_K = 5

# MOMENT/Chroma settings – ensure these match existing collections
CHROMA_DB_DIR = REPO_ROOT / "gift_eval" / ".moment_chroma"
MOMENT_MODEL_ID = "AutonLab/MOMENT-1-large"
INDEX_STEP = 1
INDEX_TAIL_TRIM_MULTIPLE = 0  # critical: must match existing collections like *_trim_0_*
RETRIEVAL_SCOPE = "per_item"  # or "global"
MOMENT_DEVICE = None  # use multi-GPU list below instead of single device
MOMENT_DEVICES: List[str] = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

# Diagnostics / verbosity
VERBOSE_LOGS = True
USE_TQDM = True  # use tqdm progress inside MOMENT paths


# %% [markdown]
# Utility helpers

# %%
def build_wrapper_input_from_gluonts(entries: List[dict], freq: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in entries:
        e0 = e[0] if isinstance(e, tuple) else e
        out.append({"target": e0["target"], "start": e0["start"], "freq": freq})
    return out


def list_chroma_collections(db_dir: Path) -> List[str]:
    try:
        import chromadb
    except Exception:
        return []
    client = chromadb.PersistentClient(path=str(db_dir))
    return [c.name for c in client.list_collections()]


def detect_segments_via_running_index(item_df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    segs: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    if "running_index" not in item_df.columns:
        return segs
    ridx = item_df["running_index"].to_numpy()
    ts = item_df.index.get_level_values("timestamp").to_numpy()
    if len(ridx) == 0:
        return segs
    start = 0
    for i in range(1, len(ridx)):
        if ridx[i] == 0:  # start of next segment
            segs.append((pd.Timestamp(ts[start]), pd.Timestamp(ts[i - 1]), int(i - start)))
            start = i
    segs.append((pd.Timestamp(ts[start]), pd.Timestamp(ts[-1]), int(len(ridx) - start)))
    return segs


def summarize_segment_lengths(train_tsdf: TimeSeriesDataFrame) -> Dict[int, List[int]]:
    lengths: Dict[int, List[int]] = {}
    for item_id, item_df in train_tsdf.groupby(level="item_id", sort=False):
        segs = detect_segments_via_running_index(item_df)
        lengths[int(item_id)] = [L for _, _, L in segs]
    return lengths


# %% [markdown]
# Step 1: Load dataset and list existing Chroma collections

# %%
section("Step 1: Load dataset and list Chroma collections")
# Match evaluate.py logic for to_univariate selection
_probe = Dataset(name=DATASET_NAME, term=TERM, to_univariate=False, storage_path=REPO_ROOT / "gift_eval" / "data")
to_uni = False if _probe.target_dim == 1 else True
ds = Dataset(name=DATASET_NAME, term=TERM, to_univariate=to_uni, storage_path=REPO_ROOT / "gift_eval" / "data")
print({
    "dataset": ds.name,
    "freq": ds.freq,
    "prediction_length": ds.prediction_length,
    "target_dim": ds.target_dim,
    "to_univariate": to_uni,
})

# Report CUDA availability
try:
    import torch
    print({
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "configured_devices": MOMENT_DEVICES,
    })
except Exception:
    print({"cuda_available": False, "configured_devices": MOMENT_DEVICES})

# Configure logging to surface debug/info from wrapper
if VERBOSE_LOGS:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gift_eval.tabpfn_ts_wrapper").setLevel(logging.INFO)

available = list_chroma_collections(CHROMA_DB_DIR)
print({"chroma_dir": str(CHROMA_DB_DIR), "num_collections": len(available)})
if available:
    print("first few collections:", available[: min(7, len(available))])


# %% [markdown]
# Step 2: Build raw TSDF inputs

# %%
section("Step 2: Build raw TSDF inputs (wrapper input + TSDF)")
test_entries = list(ds.test_data)
print({"test_entries_source": "ds.test_data", "num_entries": len(test_entries)})
test_data_input = build_wrapper_input_from_gluonts(test_entries, ds.freq)
train_tsdf_full = TabPFNTSPredictor.convert_to_timeseries_dataframe(test_data_input)
print({
    "train_tsdf_full_rows": len(train_tsdf_full),
    "items": len(train_tsdf_full.item_ids),
})


# %% [markdown]
# Step 3: Baseline preprocessing (no retrieval)

# %%
section("Step 3: Baseline preprocessing (no retrieval)")
predictor_base = TabPFNTSPredictor(
    ds_prediction_length=ds.prediction_length,
    ds_freq=ds.freq,
    tabpfn_mode=TabPFNMode.MOCK,
    context_length=CONTEXT_LENGTH,
    batch_size=64,
    debug=True,
    few_shot_k=0,
    few_shot_len=0,
)
_t0 = time.perf_counter()
train_base, test_base = predictor_base._preprocess_test_data(test_data_input)
_t1 = time.perf_counter()
print({
    "train_base_rows": len(train_base),
    "test_base_rows": len(test_base),
    "num_items": len(train_base.item_ids),
    "baseline_preprocess_s": round(_t1 - _t0, 3),
})


# %% [markdown]
# Step 4: MOMENT retrieval preprocessing (uses existing persistent Chroma collection)

# %%
section("Step 4: MOMENT retrieval preprocessing (reuse existing collection)")
# Show key retrieval configs before running
print({
    "few_shot_k": FEW_SHOT_K,
    "few_shot_len": CONTEXT_LENGTH + ds.prediction_length,
    "retrieval_scope": RETRIEVAL_SCOPE,
    "moment_index_sample_size": 0,
    "moment_progress(use_tqdm)": USE_TQDM,
    "index_step": INDEX_STEP,
    "index_tail_trim_multiple": INDEX_TAIL_TRIM_MULTIPLE,
})
predictor_moment = TabPFNTSPredictor(
    ds_prediction_length=ds.prediction_length,
    ds_freq=ds.freq,
    tabpfn_mode=TabPFNMode.MOCK,
    context_length=CONTEXT_LENGTH,
    batch_size=4096,
    debug=True,
    few_shot_k=FEW_SHOT_K,
    few_shot_len=CONTEXT_LENGTH + ds.prediction_length,
    retrieval_mode="moment",
    dataset_id=ds.name,
    chroma_db_dir=CHROMA_DB_DIR,
    moment_ctx_len=0,  # 0 => adopt P to match ctx_{P}
    moment_device=MOMENT_DEVICE,
    moment_devices=MOMENT_DEVICES,
    moment_batch_size=4096,
    moment_index_sample_size=0,
    moment_progress=USE_TQDM,
    retrieval_scope=RETRIEVAL_SCOPE,
    index_step=INDEX_STEP,
    index_tail_trim_multiple=INDEX_TAIL_TRIM_MULTIPLE,
    moment_model_id=MOMENT_MODEL_ID,
)

coll_name = predictor_moment._moment_collection_name()
print({"collection": coll_name, "exists": coll_name in set(available)})
_t2 = time.perf_counter()
# Ensure no new embeddings are written: capture count before
_count_before = None
_count_after = None
try:
    import chromadb as _ch
    _client = _ch.PersistentClient(path=str(CHROMA_DB_DIR))
    _coll = _client.get_or_create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})
    _count_before = int(_coll.count())  # type: ignore
except Exception:
    _coll = None
train_mom, test_mom = predictor_moment._preprocess_test_data(test_data_input)
_t3 = time.perf_counter()
print({
    "train_moment_rows": len(train_mom),
    "delta_rows": len(train_mom) - len(train_base),
    "moment_preprocess_s": round(_t3 - _t2, 3),
})
if _coll is not None and _count_before is not None:
    try:
        _count_after = int(_coll.count())  # type: ignore
        print({"collection_count_before": _count_before, "collection_count_after": _count_after, "added": _count_after - _count_before})
    except Exception:
        pass


# %% [markdown]
# Step 5: Inspect retrieval metadata and verify top_k per item

# %%
section("Step 5: Inspect retrieval metadata + verify counts")
retrieval_info = getattr(predictor_moment, "_last_retrieval_info", {})
P = ds.prediction_length

print("items in retrieval_info:", len(retrieval_info))
per_item_k = {int(k): len(v) for k, v in retrieval_info.items()}
print("top-5 per-item retrieved counts:", dict(list(per_item_k.items())[:5]))

# Expected added rows ~= sum_i (k_i * 2P)
expected_added = int(sum(per_item_k.get(int(item), 0) for item in train_mom.item_ids) * (2 * P))
actual_added = int(len(train_mom) - len(train_base))
print({
    "P": P,
    "expected_added_rows_approx": expected_added,
    "actual_added_rows": actual_added,
})


# %% [markdown]
# Step 4R: RANDOM retrieval preprocessing (evaluation-consistent)

# %%
section("Step 4R: RANDOM retrieval preprocessing (evaluation-consistent)")
# To mirror run_random_retrieval_evals.sh/evaluate.py: set few_shot_len=0 so wrapper uses (context_length + P)
print({
    "few_shot_k": FEW_SHOT_K,
    "few_shot_len": 0,
    "retrieval_mode": "random",
    "context_length": CONTEXT_LENGTH,
})
predictor_random = TabPFNTSPredictor(
    ds_prediction_length=ds.prediction_length,
    ds_freq=ds.freq,
    tabpfn_mode=TabPFNMode.MOCK,
    context_length=CONTEXT_LENGTH,
    batch_size=4096,
    debug=True,
    few_shot_k=FEW_SHOT_K,
    few_shot_len=0,
    retrieval_mode="random",
)
_tR0 = time.perf_counter()
train_rand, test_rand = predictor_random._preprocess_test_data(test_data_input)
_tR1 = time.perf_counter()
print({
    "train_random_rows": len(train_rand),
    "delta_rows_random": len(train_rand) - len(train_base),
    "random_preprocess_s": round(_tR1 - _tR0, 3),
})

# Show per-item delta to verify segments were appended
per_item_base = train_base.groupby(level="item_id").size()
per_item_rand = train_rand.groupby(level="item_id").size()
delta_per_item_rand = (per_item_rand - per_item_base).sort_values(ascending=False)
print("Per-item added rows (random) head::\n", delta_per_item_rand.head(10))


# %% [markdown]
# Step 6: Validate feature segmentation via running_index resets

# %%
section("Step 6: Validate feature segmentation via running_index")
seg_lengths = summarize_segment_lengths(train_mom)
sample_item = next(iter(seg_lengths.keys())) if seg_lengths else None
if sample_item is not None:
    print("sample_item:", sample_item, "segment_lengths:", seg_lengths[sample_item])
    # Heuristics: expect one segment ~ CONTEXT_LENGTH and the rest ~ 2P each
    approx_ctx = min(CONTEXT_LENGTH, len(train_base.loc[sample_item])) if sample_item in train_base.item_ids else CONTEXT_LENGTH
    near_ctx = [L for L in seg_lengths[sample_item] if abs(L - approx_ctx) <= max(2, int(0.1 * approx_ctx))]
    near_2P = [L for L in seg_lengths[sample_item] if abs(L - 2 * P) <= max(2, int(0.1 * 2 * P))]
    print({
        "segments_total": len(seg_lengths[sample_item]),
        "segments_near_ctx": len(near_ctx),
        "segments_near_2P": len(near_2P),
    })
else:
    print("No segments detected (unexpected).")


# %% [markdown]
# Step 7: Direct Chroma query check for a sample item

# %%
section("Step 7: Direct Chroma query (metadatas + distances)")
try:
    coll = getattr(predictor_moment, "_chroma_collection", None)
    if coll is not None and len(train_mom.item_ids):
        item_id = int(train_mom.item_ids[0])
        ctx_df = train_mom.loc[item_id].iloc[-min(P, len(train_mom.loc[item_id])) :]
        q_vec = predictor_moment._moment_embed_series(ctx_df["target"].values)
        q_start = ctx_df.index.get_level_values("timestamp")[0]
        where = {"end_ts_int": {"$lt": int(pd.Timestamp(q_start).value)}}
        res = coll.query(query_embeddings=[q_vec], n_results=max(10, FEW_SHOT_K), where=where, include=["metadatas", "distances"])  # type: ignore
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        print("query returned:", len(metas))
        for j, md in enumerate(metas[: min(5, len(metas))]):
            print(f"  j={j} item_id={md.get('item_id')} start={md.get('start_ts')} end={md.get('end_ts')} dist={dists[j] if j < len(dists) else None}")
    else:
        print("No Chroma collection available or no items.")
except Exception as e:
    print("Direct Chroma query failed:", e)


# %% [markdown]
# Optional: Plot baseline context and first few retrieved segments for a sample item

# %%
try:
    import matplotlib.pyplot as plt
    section("Step 8 (optional): Plot baseline + retrieved segments")
    if len(train_mom.item_ids):
        sample_item = int(train_mom.item_ids[0])
        base_item = train_base.loc[sample_item]
        mom_item = train_mom.loc[sample_item]

        # Detect segments for plotting boundaries
        segs = detect_segments_via_running_index(mom_item)

        ts_base = base_item.index.get_level_values("timestamp").to_numpy()
        y_base = base_item["target"].to_numpy()

        plt.figure(figsize=(14, 6))
        plt.plot(ts_base, y_base, label=f"item {sample_item} (baseline context) (len={len(ts_base)})", lw=2)
        cmap = plt.get_cmap("tab10")
        # Overlay first few non-baseline segments (~2P)
        plotted = 0
        for j, (st, en, L) in enumerate(segs):
            if abs(L - 2 * P) <= max(2, int(0.1 * 2 * P)):
                seg = mom_item.loc[st:en, :]
                ts_seg = seg.index.get_level_values("timestamp").to_numpy()
                y_seg = seg["target"].to_numpy()
                plt.plot(ts_seg, y_seg, color=cmap((j + 1) % 10), alpha=0.9, label=f"retrieved #{plotted+1} (len={L})")
                plotted += 1
            if plotted >= min(FEW_SHOT_K, 5):
                break

        plt.title("MOMENT retrieval: baseline context + retrieved segments")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No items to plot.")
except Exception as e:
    print("Plotting skipped:", e)


# %%

