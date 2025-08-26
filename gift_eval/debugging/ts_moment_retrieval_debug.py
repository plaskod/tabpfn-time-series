# %%
# VS Code Interactive Debug Script: MOMENT-based retrieval with ChromaDB

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
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# %% [markdown]
# Parameters: choose dataset, term, and retrieval settings (MOMENT + Chroma)

# %%
DATASET_NAME = "bizitobs_service"  # or e.g. "hierarchical_sales/D", "LOOP_SEATTLE/D"
DATASET_STORAGE_PATH = REPO_ROOT / "gift_eval" / "data"
TERM = "short"  # ["short", "medium", "long"]

CONTEXT_LENGTH = 100

# Retrieval (few-shot) params
FEW_SHOT_K = 3        # top_k MOMENT subsequences (0 disables retrieval)
FEW_SHOT_LEN = 128    # length L of each subsequence to append

# MOMENT encoder params
MOMENT_CTX_LEN = 512  # window length for embedding
MOMENT_DEVICE = "cuda"  # e.g., "cuda" or None
MOMENT_DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]  # optional multi-GPU for embeddings
RETRIEVAL_SCOPE = "per_item"  # or "global"
CHROMA_DB_DIR = REPO_ROOT / "gift_eval" / ".moment_chroma"


# %% [markdown]
# Load dataset and build GluonTS test instances

# %%
section("Step 1: Load dataset and basic metadata")
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
})

section("Step 2: Convert GluonTS TestData → list[dict] (wrapper input)")
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
train_tsdf_full = TabPFNTSPredictor.convert_to_timeseries_dataframe(test_data_input)
print("train_tsdf_full: rows=", len(train_tsdf_full), "items=", len(train_tsdf_full.item_ids))
print("train_tsdf_full head:\n", train_tsdf_full.head())


# %% [markdown]
# Baseline preprocessing (no retrieval) for comparison

# %%
section("Step 4: Baseline preprocessing (no retrieval)")
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
# Plot baseline context + retrieved windows for a sample item

# %%
section("Step 7: Plot baseline context + retrieved windows")
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
    plt.plot(ts_base, y_base, label=f"item {sample_item} (baseline context)", lw=2)

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
            lbl = f"retrieved #{j+1} (d={md.get('distance', None)})"
            plt.plot(ts_seg, y_seg, color=cmap((j + 1) % 10), alpha=0.9, label=lbl)

    plt.title("MOMENT cosine retrieval: baseline context + top-k windows")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No items to plot.")


# %% [markdown]
# Done.

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from gift_eval.evaluate import construct_evaluation_data
from gift_eval.tabpfn_ts_wrapper import TabPFNTSPredictor, TabPFNMode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_storage_path", type=str, required=True)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--few_shot_k", type=int, default=5)
    parser.add_argument("--few_shot_len", type=int, default=1024)
    parser.add_argument("--moment_ctx_len", type=int, default=512)
    parser.add_argument("--chroma_db_dir", type=str, default=str(Path(__file__).parent / ".moment_chroma"))
    parser.add_argument("--moment_device", type=str, default=None)
    parser.add_argument("--retrieval_scope", type=str, default="per_item")
    args = parser.parse_args()

    [(sub_dataset, ds_meta)] = construct_evaluation_data(
        dataset_name=args.dataset,
        dataset_storage_path=Path(args.dataset_storage_path),
        terms=["short"],
    )

    predictor = TabPFNTSPredictor(
        ds_prediction_length=sub_dataset.prediction_length,
        ds_freq=sub_dataset.freq,
        tabpfn_mode=TabPFNMode.LOCAL,
        context_length=args.context_length,
        debug=True,
        batch_size=256,
        few_shot_k=args.few_shot_k,
        few_shot_len=args.few_shot_len,
        retrieval_mode="moment",
        dataset_id=sub_dataset.name,
        chroma_db_dir=Path(args.chroma_db_dir),
        moment_ctx_len=args.moment_ctx_len,
        moment_device=args.moment_device,
        retrieval_scope=args.retrieval_scope,
    )

    # Build one batch from test_data to trigger retrieval and capture augmented windows
    test_entries = list(sub_dataset.test_data)
    batch = test_entries[: min(len(test_entries), 1)]
    train_tsdf, _ = predictor._preprocess_test_data(batch)

    # Plot baseline context and retrieved segments per first item
    item_id = train_tsdf.item_ids[0]
    df = train_tsdf.loc[item_id]
    ts = df.index.get_level_values("timestamp").to_numpy()
    y = df["target"].to_numpy()

    # Heuristic: segment boundaries were used; just plot the whole with annotations
    plt.figure(figsize=(14, 5))
    plt.plot(ts, y, label=f"item {item_id}")
    plt.title("Baseline context + retrieved windows (MOMENT cosine)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


