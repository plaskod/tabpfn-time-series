# %%
# VS Code Interactive Debug Script: MOMENT-based retrieval with ChromaDB
# This interactive script walks through the preprocessing and MOMENT+ChromaDB
# retrieval flow used by `TabPFNTSPredictor`.
# Steps overview:
# 1) Load dataset metadata.
# 2) Normalize GluonTS test entries into list[dict] for the wrapper.
# 3) Build a raw `TimeSeriesDataFrame` for reconstruction/plots.
# 4) Run baseline preprocessing (no retrieval).
# 5) Run MOMENT retrieval: for each query, append support segments composed of
#    P points from the retrieved window + P following horizon (P=prediction_length),
#    re-attributed to the querying item_id; segments get unique `segment_id`s
#    so featurization stays independent per segment.
# 6) Inspect captured retrieval metadata.
# 7) Plot baseline context, retrieved windows (solid), and horizons (dashed).

from pathlib import Path
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chromadb

# Ensure repository root is on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gift_eval.data import Dataset
from gift_eval.tabpfn_ts_wrapper import TabPFNTSPredictor, TabPFNMode
from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame


def section(title: str):
    # Pretty step header printer for readability in the interactive output
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# %% [markdown]
# Parameters: choose dataset, term, and retrieval settings (MOMENT + Chroma)

# %%
DATASET_NAME = "bizitobs_service"  # or e.g. "hierarchical_sales/D", "LOOP_SEATTLE/D"
DATASET_STORAGE_PATH = REPO_ROOT / "gift_eval" / "data"
TERM = "short"  # ["short", "medium", "long"]

CONTEXT_LENGTH = 60

# Retrieval (few-shot) params
FEW_SHOT_K = 10       # top_k MOMENT subsequences (0 disables retrieval)
FEW_SHOT_LEN = 128    # length L of each subsequence to append

# MOMENT encoder params
MOMENT_CTX_LEN = 512  # window length for embedding
MOMENT_DEVICE = "cuda"  # e.g., "cuda" or None
MOMENT_DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]  # optional multi-GPU for embeddings
RETRIEVAL_SCOPE = "global"  # "per_item"  # or "global"
CHROMA_DB_DIR = REPO_ROOT / "gift_eval" / ".moment_chroma"
RETRIEVAL_SEPARATION_LEN = 0  # timesteps; set >0 to enforce spacing between retrieved windows


# %% [markdown]
# Load dataset and build GluonTS test instances

# %%
section("Step 1: Load dataset and basic metadata")
# - `to_univariate=True` ensures each series is 1D
# - Provides `freq`, `prediction_length`, `windows`, etc.
ds = Dataset(
    name=DATASET_NAME,
    term=TERM,
    to_univariate=True,
    storage_path=DATASET_STORAGE_PATH,
)

print({
    "dataset": ds.name,
    "freq": ds.freq,
    "prediction_length": ds.prediction_length,
    "target_dim": ds.target_dim,
    "windows": ds.windows,
    "test_data": ds.test_data,
    "test_data_len": len(ds.test_data),
    "train_data": ds.training_dataset,
    "train_data_len": len(ds.training_dataset),
    # "last_train_ts": ds.training_dataset[-1].end,
    # "first_test_ts": ds.test_dataset[0].start,
})


section("Step 2: Convert GluonTS TestData → list[dict] (wrapper input)")
# GluonTS yields tuples in some datasets; normalize to a list of dicts with
# keys: 'target', 'start', 'freq' expected by the wrapper.
test_data_input: List[Dict[str, Any]] = []
for entry in ds.test_data:
    item = entry[0] if isinstance(entry, tuple) else entry
    test_data_input.append({
        "target": item["target"],
        "start": item["start"],
        "freq": ds.freq,
    })
print(f"test_data_input size: {len(test_data_input)}")


# %% [markdown]
# Build train_tsdf_full (as seen by the predictor) for reconstructing retrieved windows

# %%
section("Step 3: Build train_tsdf_full (raw)")
# Raw TSDF without features; used here only for reconstructing/plotting
# retrieved segments by timestamp ranges.
train_tsdf_full = TabPFNTSPredictor.convert_to_timeseries_dataframe(test_data_input)
print("train_tsdf_full: rows=", len(train_tsdf_full), "items=", len(train_tsdf_full.item_ids))
print("train_tsdf_full head:\n", train_tsdf_full.head())


# %% [markdown]
# Baseline preprocessing (no retrieval) for comparison

# %%
section("Step 4: Baseline preprocessing (no retrieval)")
# Internally:
#  - Slices each item to last `context_length` timestamps as the context
#  - Generates test horizon of length `prediction_length`
#  - Marks baseline rows as `segment_id=0` and applies feature transforms
predictor_base = TabPFNTSPredictor(
    ds_prediction_length=ds.prediction_length,
    ds_freq=ds.freq,
    tabpfn_mode=TabPFNMode.MOCK,  # MOCK avoids GPU and heavy init
    context_length=CONTEXT_LENGTH,
    batch_size=64,
    debug=True,
    few_shot_k=0,
    few_shot_len=0,
)

train_base, test_base = predictor_base._preprocess_test_data(test_data_input)

print({
    "train_base_rows": len(train_base),
    "test_base_rows": len(test_base),
    "num_items": len(train_base.item_ids),
    "num_features": len([c for c in train_base.columns if c != "target"]),
})


# %% [markdown]
# Preprocessing with MOMENT retrieval (few-shot augmentation)
print("MOMENT_DEVICE", MOMENT_DEVICE)
# %%
section("Step 5: Preprocessing with MOMENT retrieval (few-shot augmentation)")
# Internally:
#  - Build/refresh a ChromaDB index of eligible windows with MOMENT embeddings
#  - For each query, retrieve top_k windows ending strictly before the query start
#  - For each, create support segment = P window tail + P following horizon
#    (P=prediction_length), re-attributed to the querying item_id
#  - Assign unique `segment_id` per support to keep features independent
#  - Log metadata under `_last_retrieval_info` for debugging/plots
predictor_moment = TabPFNTSPredictor(
    ds_prediction_length=ds.prediction_length,
    ds_freq=ds.freq,
    tabpfn_mode=TabPFNMode.MOCK,
    context_length=CONTEXT_LENGTH,
    batch_size=64,
    debug=True,
    few_shot_k=FEW_SHOT_K,
    few_shot_len=FEW_SHOT_LEN,
    retrieval_mode="moment",
    dataset_id=ds.name,
    chroma_db_dir=CHROMA_DB_DIR,
    moment_ctx_len=MOMENT_CTX_LEN,
    moment_device=MOMENT_DEVICE,
    moment_devices=MOMENT_DEVICES,
    moment_batch_size=128,
    moment_index_sample_size=3000,
    moment_progress=True,
    retrieval_scope=RETRIEVAL_SCOPE,
    retrieval_separation_len=RETRIEVAL_SEPARATION_LEN,
)

train_mom, test_mom = predictor_moment._preprocess_test_data(test_data_input)

print({
    "train_moment_rows": len(train_mom),
    "delta_rows": len(train_mom) - len(train_base),
})

per_item_base = train_base.groupby(level="item_id").size()
per_item_mom = train_mom.groupby(level="item_id").size()
delta_per_item = (per_item_mom - per_item_base).sort_values(ascending=False)
print("Per-item added rows head::\n", delta_per_item.head(10))


# %% [markdown]
# Inspect retrieved windows metadata (top-k per item) and plot

# %%
section("Step 6: Retrieved windows metadata")
# Each metadata dict contains: item_id (source), start_ts, end_ts,
# horizon_end_ts, and similarity distance.
retrieval_info = getattr(predictor_moment, "_last_retrieval_info", {})
if not retrieval_info:
    print("No retrieval info (few_shot_k may be 0 or insufficient history).")
else:
    # Print first item summary
    some_item = list(retrieval_info.keys())[0]
    print("Example item:", some_item)
    for j, md in enumerate(retrieval_info[some_item]):
        print(f"  #{j+1}", md)


# %% [markdown]
# Step 6a: Show item_id inventory and counts

# %%
section("Step 6a: Item inventory and counts")
# Quick sanity check of per-item lengths in the raw input TSDF.
uniq_items = list(train_tsdf_full.item_ids)
print("Num items:", len(uniq_items))
sizes = train_tsdf_full.groupby(level="item_id").size().sort_values(ascending=False)
print("Per-item lengths (top 10):\n", sizes.head(10))


# %% [markdown]
# Step 6b: Rebuild expected where filter and directly query Chroma to inspect item_id

# %%
section("Step 6b: Inspect Chroma query and item_id metadatas")
# Re-run a sample Chroma query to inspect retrieved metadatas and distances.
# Helps verify where filter and retrieval scope.
if len(train_mom.item_ids):
    sample_item = int(list(retrieval_info.keys())[0]) if retrieval_info else train_mom.item_ids[0]
    pred_len = ds.prediction_length
    # Approximate the base_initial_context used for the query
    ctx_df = train_mom.loc[sample_item]
    q_ctx = ctx_df.iloc[-min(pred_len, len(ctx_df)) :]
    q_start_ts = q_ctx.index.get_level_values("timestamp")[0]
    q_start_ts_int = int(pd.Timestamp(q_start_ts).value)
    scope = RETRIEVAL_SCOPE
    if scope == "per_item":
        where = {"$and": [{"item_id": {"$eq": str(sample_item)}}, {"end_ts_int": {"$lt": q_start_ts_int}}]}
    else:
        where = {"end_ts_int": {"$lt": q_start_ts_int}}
    print("retrieval_scope:", scope)
    print("sample_item:", sample_item)
    print("where filter:", where)

    # Query Chroma directly
    try:
        coll = getattr(predictor_moment, "_chroma_collection", None)
        if coll is not None:
            res = coll.query(
                query_embeddings=[predictor_moment._moment_embed_series(q_ctx["target"].values)],
                n_results=20,
                where=where,
                include=["metadatas", "distances"],
            )
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            print("Top-5 metadatas:")
            for j, md in enumerate(metas[:5]):
                print(f"  j={j} item_id={md.get('item_id')} start={md.get('start_ts')} end={md.get('end_ts')} dist={dists[j] if j < len(dists) else None}")
        else:
            print("Chroma collection not available on predictor.")
    except Exception as e:
        print("Direct Chroma query failed:", e)

# %% [markdown]
# Plot baseline context + retrieved windows for a sample item

# %%
section("Step 7: Plot baseline context + retrieved windows")
# - Baseline context: dark solid line
# - Retrieved windows: solid colored lines
# - Horizons: dashed lines with same colors
if len(train_mom.item_ids):
    sample_item = (
        delta_per_item.index[0]
        if (len(delta_per_item) and delta_per_item.iloc[0] > 0)
        else train_mom.item_ids[0]
    )
    base_item = train_base.loc[sample_item]
    mom_item = train_mom.loc[sample_item]

    ts_base = base_item.index.get_level_values("timestamp").to_numpy()
    y_base = base_item["target"].to_numpy()

    plt.figure(figsize=(14, 6))
    plt.plot(ts_base, y_base, label=f"item {sample_item} (baseline context) (len={len(ts_base)})", lw=2)

    # Overlay retrieved windows based on stored metadata
    item_key = str(sample_item)
    if item_key in retrieval_info and len(retrieval_info[item_key]):
        cmap = plt.get_cmap("tab10")
        for j, md in enumerate(retrieval_info[item_key][: FEW_SHOT_K]):
            st = pd.Timestamp(md["start_ts"])  # type: ignore
            en = pd.Timestamp(md["end_ts"])    # type: ignore
            full_df = train_tsdf_full.loc[sample_item]
            seg = full_df.loc[st:en, :]
            ts_seg = seg.index.get_level_values("timestamp").to_numpy()
            y_seg = seg["target"].to_numpy()
            seg_len = len(ts_seg)
            lbl = f"retrieved #{j+1} (len={seg_len}, d={md.get('distance', None)})"
            plt.plot(ts_seg, y_seg, color=cmap((j + 1) % 10), alpha=0.9, label=lbl)

            # Plot dashed horizon following the retrieved subsequence if present
            h_end = md.get("horizon_end_ts")
            if h_end is not None:
                try:
                    h_end_ts = pd.Timestamp(h_end)
                    # Build horizon timestamps at dataset frequency
                    hor_ts = pd.date_range(start=ts_seg[-1], periods=2, freq=ds.freq)[1:]
                    hor_ts = pd.date_range(start=hor_ts[0], end=h_end_ts, freq=ds.freq)
                    hor_vals = full_df.loc[hor_ts, "target"].to_numpy()
                    plt.plot(hor_ts, hor_vals, color=cmap((j + 1) % 10), linestyle="--", alpha=0.9, label=f"horizon #{j+1}")
                except Exception:
                    pass

    plt.title("MOMENT cosine retrieval: baseline context + top-k windows")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No items to plot.")


# import argparse
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt

# from gift_eval.evaluate import construct_evaluation_data
# from gift_eval.tabpfn_ts_wrapper import TabPFNTSPredictor, TabPFNMode


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, required=True)
#     parser.add_argument("--dataset_storage_path", type=str, required=True)
#     parser.add_argument("--context_length", type=int, default=512)
#     parser.add_argument("--few_shot_k", type=int, default=5)
#     parser.add_argument("--few_shot_len", type=int, default=1024)
#     parser.add_argument("--moment_ctx_len", type=int, default=512)
#     parser.add_argument("--chroma_db_dir", type=str, default=str(Path(__file__).parent / ".moment_chroma"))
#     parser.add_argument("--moment_device", type=str, default=None)
#     parser.add_argument("--retrieval_scope", type=str, default="per_item")
#     args = parser.parse_args()

#     [(sub_dataset, ds_meta)] = construct_evaluation_data(
#         dataset_name=args.dataset,
#         dataset_storage_path=Path(args.dataset_storage_path),
#         terms=["short"],
#     )

#     predictor = TabPFNTSPredictor(
#         ds_prediction_length=sub_dataset.prediction_length,
#         ds_freq=sub_dataset.freq,
#         tabpfn_mode=TabPFNMode.LOCAL,
#         context_length=args.context_length,
#         debug=True,
#         batch_size=256,
#         few_shot_k=args.few_shot_k,
#         few_shot_len=args.few_shot_len,
#         retrieval_mode="moment",
#         dataset_id=sub_dataset.name,
#         chroma_db_dir=Path(args.chroma_db_dir),
#         moment_ctx_len=args.moment_ctx_len,
#         moment_device=args.moment_device,
#         retrieval_scope=args.retrieval_scope,
#     )

#     # Build one batch from test_data to trigger retrieval and capture augmented windows
#     test_entries = list(sub_dataset.test_data)
#     batch = test_entries[: min(len(test_entries), 1)]
#     train_tsdf, _ = predictor._preprocess_test_data(batch)

#     # Plot baseline context and retrieved segments per first item
#     item_id = train_tsdf.item_ids[0]
#     df = train_tsdf.loc[item_id]
#     ts = df.index.get_level_values("timestamp").to_numpy()
#     y = df["target"].to_numpy()

#     # Heuristic: segment boundaries were used; just plot the whole with annotations
#     plt.figure(figsize=(14, 5))
#     plt.plot(ts, y, label=f"item {item_id}")
#     plt.title("Baseline context + retrieved windows (MOMENT cosine)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     main()



# %%
