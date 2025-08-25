# %%
# VS Code Interactive Debug Script: TimeSeriesDataFrame with and without retrieval

from pathlib import Path
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd

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
# Parameters: choose dataset, term, and retrieval settings

# %%
DATASET_NAME = "bizitobs_service"  # or e.g. "hierarchical_sales/D", "LOOP_SEATTLE/D"
DATASET_STORAGE_PATH = REPO_ROOT / "gift_eval" / "data"
TERM = "short"  # ["short", "medium", "long"]

CONTEXT_LENGTH = 100

# Retrieval (few-shot) params
FEW_SHOT_K = 2     # top_k random subsequences per item (0 disables retrieval)
FEW_SHOT_LEN = 64  # length L of each subsequence
FEW_SHOT_SEED = 42


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

section("Step 2: Convert GluonTS TestData → list[dict] (format expected by wrapper)")
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
# Helper: build the original (full) TimeSeriesDataFrame as seen by the wrapper

# %%
section("Step 3: Build train_tsdf_full from test_data_input (exactly like the predictor)")
train_tsdf_full = TabPFNTSPredictor.convert_to_timeseries_dataframe(test_data_input)
print("train_tsdf_full: rows=", len(train_tsdf_full), "items=", len(train_tsdf_full.item_ids))
print("train_tsdf_full head:\n", train_tsdf_full.head())


# %% [markdown]
# Baseline preprocessing (no retrieval)

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

print("Per-item lengths (base) – first 5:\n", train_base.groupby(level="item_id").size().head())

# Show context timestamp range for the first item
first_item = train_base.item_ids[0]
base_item_first = train_base.loc[first_item]
print("First item:", first_item)
print(
    "  context timestamps:",
    base_item_first.index.get_level_values("timestamp").min(),
    "→",
    base_item_first.index.get_level_values("timestamp").max(),
)


# %% [markdown]
# Preprocessing with retrieval (few-shot augmentation)

# %%
section("Step 5: Preprocessing with retrieval (few-shot augmentation)")
predictor_fs = TabPFNTSPredictor(
    ds_prediction_length=ds.prediction_length,
    ds_freq=ds.freq,
    tabpfn_mode=TabPFNMode.MOCK,
    context_length=CONTEXT_LENGTH,
    batch_size=64,
    debug=True,
    few_shot_k=FEW_SHOT_K,
    few_shot_len=FEW_SHOT_LEN,
    few_shot_seed=FEW_SHOT_SEED,
)

train_fs, test_fs = predictor_fs._preprocess_test_data(test_data_input)

print({
    "train_fs_rows": len(train_fs),
    "test_fs_rows": len(test_fs),
    "delta_rows": len(train_fs) - len(train_base),
})

per_item_base = train_base.groupby(level="item_id").size()
per_item_fs = train_fs.groupby(level="item_id").size()
delta_per_item = (per_item_fs - per_item_base).sort_values(ascending=False)
print("Per-item added rows head::\n", delta_per_item.head(10))


# %% [markdown]
# Sanity check: retrieval windows come from earlier than the current context

# %%
section("Step 6: Sanity check that retrieval windows come from earlier history")
def max_context_ts(tsdf: TimeSeriesDataFrame) -> pd.Series:
    return tsdf.reset_index().groupby("item_id")["timestamp"].max()

max_ts_base = max_context_ts(train_base)

# Identify rows that exist only due to retrieval
augmented = train_fs.loc[~train_fs.index.isin(train_base.index)]

if len(augmented):
    first_violation = augmented.reset_index().merge(
        max_ts_base.rename("max_ts"), left_on="item_id", right_index=True
    )
    any_future = (first_violation["timestamp"] > first_violation["max_ts"]).any()
    print("Any retrieval rows after base context?", any_future)
else:
    print("No augmented rows (few_shot_k or few_shot_len likely 0 or insufficient history).")


# %% [markdown]
# Inspect a single item_id timeline before/after (data only)
# Prefer one that actually received retrieval rows if available

# %%
sample_item = (
    delta_per_item.index[0]
    if (len(delta_per_item) and delta_per_item.iloc[0] > 0)
    else train_base.item_ids[0]
)
base_item = train_base.loc[sample_item]
fs_item = train_fs.loc[sample_item]

print("Sample item:", sample_item)
print("Base item rows:", len(base_item), "FS item rows:", len(fs_item))
print("FS-only rows head:\n", fs_item.loc[~fs_item.index.isin(base_item.index)].head())


section("Step 8: What the predictor sees on each split")
print("Short summary per window:")
print("- Baseline inference: for each item_id, context = last CONTEXT_LENGTH rows of train_tsdf_full; test = next prediction_length timestamps.")
print("- Retrieval inference: same context + appended top-k random subsequences from earlier history (per item), features applied to combined table;")
print("  predictions are for exactly the same test timestamps as baseline.")


# %% [markdown]
# Tip: You can also visualize the target + features using matplotlib if desired.
# For clarity in this debug file we print shapes and small slices.


