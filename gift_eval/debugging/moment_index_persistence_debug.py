# %%
# MOMENT + ChromaDB persistent indexing and reuse (VS Code Interactive)
#
# Goal:
# - Build a persistent, on-disk Chroma index of MOMENT embeddings per dataset once
# - Reuse it across runs/evals without recomputing
# - Validate on two datasets: the current dataset and "hierarchical_sales/W"

from pathlib import Path
import sys
import time
from typing import List, Dict, Any, Tuple, Optional, Sequence
import warnings

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
# Configuration

# %%
# Datasets to index (name, term)
DATASETS: List[Tuple[str, str]] = [
    # ("bizitobs_service", "short"),
    # ("hierarchical_sales/W", "short"),
    ("SZ_TAXI/H", "short"),
]

DATASET_STORAGE_PATH = REPO_ROOT / "gift_eval" / "data"

# Common predictor / MOMENT params
CONTEXT_LENGTH = 30 # 60 biz_service
MOMENT_CTX_LEN = 0  # 0 => adopt P (dataset prediction_length)
MOMENT_DEVICE = None  # or None / "cpu"
MOMENT_DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]   # e.g., ["cuda:0", "cuda:1"] for multi-GPU
MOMENT_BATCH_SIZE = 4096 # Batch size of 320712 is greater than max batch size of 5461
MOMENT_PROGRESS = True

# Indexing controls
INDEX_STEP = 1                 # slide windows at this stride
INDEX_TAIL_TRIM_MULT = 2*48       # trim last N*pred_len timestamps from indexing (leakage guard)

# Persistent Chroma location (shared DB with multiple collections)
CHROMA_DB_DIR = REPO_ROOT / "gift_eval" / ".moment_chroma"


# %% [markdown]
# Utilities: build wrapper input, ensure persistent collection, index windows

# %%
def load_dataset_auto(name: str, term: str) -> Dataset:
    """Load dataset, enabling to_univariate only if needed (target_dim > 1)."""
    ds = Dataset(name=name, term=term, to_univariate=False, storage_path=DATASET_STORAGE_PATH)
    if ds.target_dim > 1:
        ds = Dataset(name=name, term=term, to_univariate=True, storage_path=DATASET_STORAGE_PATH)
    return ds


def build_wrapper_inputs_from_training(ds: Dataset) -> List[Dict[str, Any]]:
    """Convert GluonTS data to wrapper input; prefer training split, fallback to full dataset.

    Some datasets with short series can trigger IndexError in the GluonTS training split
    slicing. In that case, we fall back to using the full `gluonts_dataset` as-is.
    """
    try:
        entries = list(ds.training_dataset)
    except Exception:
        entries = list(ds.gluonts_dataset)
    out: List[Dict[str, Any]] = []
    for e in entries:
        out.append({
            "target": e["target"],
            "start": e["start"],
            "freq": ds.freq,
        })
    return out


def ensure_persistent_collection(predictor: TabPFNTSPredictor):
    """Ensure predictor has a PersistentClient and a get_or_create collection with a stable name."""
    import chromadb

    predictor.chroma_db_dir.mkdir(parents=True, exist_ok=True)
    if getattr(predictor, "_chroma_client", None) is None:
        predictor._chroma_client = chromadb.PersistentClient(path=str(predictor.chroma_db_dir))
    if getattr(predictor, "_chroma_collection", None) is None:
        predictor._chroma_collection = predictor._chroma_client.get_or_create_collection(
            name=predictor._moment_collection_name(),
            metadata={"hnsw:space": "cosine"},
        )
    return predictor._chroma_collection


def dir_size_mb(path: Path) -> float:
    total_bytes = 0
    for p in path.rglob("*"):
        if p.is_file():
            total_bytes += p.stat().st_size
    return total_bytes / (1024 * 1024)


def index_dataset_once(dataset_name: str, term: str) -> None:
    section(f"Index dataset (one-time): {dataset_name} [{term}]")
    ds = load_dataset_auto(dataset_name, term)

    # Build predictor with stable, persistent collection naming
    predictor = TabPFNTSPredictor(
        ds_prediction_length=ds.prediction_length,
        ds_freq=ds.freq,
        tabpfn_mode=TabPFNMode.MOCK,
        context_length=CONTEXT_LENGTH,
        batch_size=64,
        debug=True,
        few_shot_k=0,                   # indexing only; no augmentation here
        retrieval_mode="moment",
        dataset_id=ds.name,             # used in collection naming
        chroma_db_dir=CHROMA_DB_DIR,    # persistent on-disk DB
        moment_ctx_len=MOMENT_CTX_LEN,
        moment_device=MOMENT_DEVICE,
        moment_devices=MOMENT_DEVICES,
        moment_batch_size=MOMENT_BATCH_SIZE,
        moment_progress=MOMENT_PROGRESS,
        index_tail_trim_multiple=INDEX_TAIL_TRIM_MULT,
        index_step=INDEX_STEP,
    )

    # Prepare raw TSDF from training split (index entire dataset)
    train_inputs = build_wrapper_inputs_from_training(ds)
    train_tsdf_full = TabPFNTSPredictor.convert_to_timeseries_dataframe(train_inputs)

    coll = ensure_persistent_collection(predictor)
    cname = predictor._moment_collection_name()
    ccount_before = coll.count()  # type: ignore
    size_before = dir_size_mb(CHROMA_DB_DIR)

    t0 = time.time()
    predictor._moment_index_windows(train_tsdf_full)
    dt = time.time() - t0

    ccount_after = coll.count()  # type: ignore
    size_after = dir_size_mb(CHROMA_DB_DIR)

    print({
        "collection": cname,
        "count_before": int(ccount_before),
        "count_after": int(ccount_after),
        "added": int(ccount_after - ccount_before),
        "db_dir": str(CHROMA_DB_DIR),
        "size_mb_before": round(size_before, 2),
        "size_mb_after": round(size_after, 2),
        "delta_mb": round(size_after - size_before, 2),
        "index_time_s": round(dt, 2),
    })


def verify_reuse(dataset_name: str, term: str) -> None:
    section(f"Verify index reuse (no-op if already indexed): {dataset_name} [{term}]")
    ds = load_dataset_auto(dataset_name, term)
    predictor = TabPFNTSPredictor(
        ds_prediction_length=ds.prediction_length,
        ds_freq=ds.freq,
        tabpfn_mode=TabPFNMode.MOCK,
        context_length=CONTEXT_LENGTH,
        batch_size=64,
        debug=True,
        few_shot_k=0,
        retrieval_mode="moment",
        dataset_id=ds.name,
        chroma_db_dir=CHROMA_DB_DIR,
        moment_ctx_len=MOMENT_CTX_LEN,
        moment_device=MOMENT_DEVICE,
        moment_devices=MOMENT_DEVICES,
        moment_batch_size=MOMENT_BATCH_SIZE,
        moment_progress=MOMENT_PROGRESS,
        index_tail_trim_multiple=INDEX_TAIL_TRIM_MULT,
        index_step=INDEX_STEP,
    )

    train_inputs = build_wrapper_inputs_from_training(ds)
    train_tsdf_full = TabPFNTSPredictor.convert_to_timeseries_dataframe(train_inputs)

    coll = ensure_persistent_collection(predictor)
    cname = predictor._moment_collection_name()
    ccount_before = coll.count()  # type: ignore

    predictor._moment_index_windows(train_tsdf_full)

    ccount_after = coll.count()  # type: ignore
    print({
        "collection": cname,
        "count_before": int(ccount_before),
        "count_after": int(ccount_after),
        "added": int(ccount_after - ccount_before),
        "note": "If added==0, persistence + id filtering worked.",
    })


def quick_retrieval_test(dataset_name: str, term: str, few_shot_k: int = 3) -> None:
    section(f"Quick retrieval test (should reuse existing index): {dataset_name} [{term}]")
    ds = load_dataset_auto(dataset_name, term)
    predictor = TabPFNTSPredictor(
        ds_prediction_length=ds.prediction_length,
        ds_freq=ds.freq,
        tabpfn_mode=TabPFNMode.MOCK,
        context_length=CONTEXT_LENGTH,
        batch_size=64,
        debug=True,
        few_shot_k=few_shot_k,
        few_shot_len=CONTEXT_LENGTH + ds.prediction_length,
        retrieval_mode="moment",
        dataset_id=ds.name,
        chroma_db_dir=CHROMA_DB_DIR,
        moment_ctx_len=MOMENT_CTX_LEN,
        moment_device=MOMENT_DEVICE,
        moment_devices=MOMENT_DEVICES,
        moment_batch_size=MOMENT_BATCH_SIZE,
        moment_progress=True,
        index_tail_trim_multiple=INDEX_TAIL_TRIM_MULT,
        index_step=INDEX_STEP,
    )

    # Build minimal test batch (GluonTS test → list[dict])
    test_data_input: List[Dict[str, Any]] = []
    for entry in ds.test_data:
        item = entry[0] if isinstance(entry, tuple) else entry
        test_data_input.append({
            "target": item["target"],
            "start": item["start"],
            "freq": ds.freq,
        })

    train_tsdf, test_tsdf = predictor._preprocess_test_data(test_data_input)
    # If the index was reused, _moment_index_windows should log cache hits and
    # the collection count should remain stable.
    print({
        "train_rows": len(train_tsdf),
        "test_rows": len(test_tsdf),
        "num_items": len(train_tsdf.item_ids),
    })

# %%
ds = load_dataset_auto("M_DENSE/D", "short")
train_inputs = [{"target": e["target"], "start": e["start"], "freq": ds.freq} for e in ds.gluonts_dataset]
tsdf = TabPFNTSPredictor.convert_to_timeseries_dataframe(train_inputs)
P = ds.prediction_length  # 30 for D
sizes = tsdf.groupby(level="item_id").size().to_numpy()
expected = int(np.sum(np.maximum(0, sizes - P + 1)))  # step=1
print({"items": len(sizes), "min_len": int(sizes.min()), "max_len": int(sizes.max()), "expected_windows": expected})

# %%
def vectorize_dataset_db(
    dataset_name: str,
    term: str,
    *,
    context_length: Optional[int] = None,
    chroma_db_dir: Optional[Path] = None,
    moment_ctx_len: int = 0,  # 0 => adopt P
    moment_device: Optional[str] = None,
    moment_devices: Optional[Sequence[str]] = None,
    moment_batch_size: int = 512,
    moment_progress: bool = True,
    index_step: int = 1,
    index_tail_trim_multiple: int = 0,
    moment_index_sample_size: Optional[int] = None,
    moment_model_id: Optional[str] = None,
    source: str = "training",  # "training" or "full"
) -> Dict[str, Any]:
    """Vectorize (index) a dataset into a persistent Chroma DB with the given config.

    Returns a summary dict with collection name, counts before/after, size deltas, and timing.
    """
    section(f"Vectorize dataset DB: {dataset_name} [{term}]")
    ds = load_dataset_auto(dataset_name, term)

    # Prepare predictor with persistent collection naming
    predictor = TabPFNTSPredictor(
        ds_prediction_length=ds.prediction_length,
        ds_freq=ds.freq,
        tabpfn_mode=TabPFNMode.MOCK,
        context_length=context_length if context_length is not None else ds.prediction_length,
        batch_size=64,
        debug=True,
        few_shot_k=0,
        retrieval_mode="moment",
        dataset_id=ds.name,
        chroma_db_dir=(chroma_db_dir if chroma_db_dir is not None else CHROMA_DB_DIR),
        moment_ctx_len=moment_ctx_len,
        moment_device=moment_device,
        moment_devices=moment_devices,
        moment_batch_size=moment_batch_size,
        moment_progress=moment_progress,
        moment_index_sample_size=moment_index_sample_size,
        index_tail_trim_multiple=index_tail_trim_multiple,
        index_step=index_step,
        moment_model_id=(moment_model_id if moment_model_id is not None else "AutonLab/MOMENT-1-large"),
    )

    # Build raw TSDF from the chosen source and index
    if source == "full":
        entries = list(ds.gluonts_dataset)
        train_inputs: List[Dict[str, Any]] = [
            {"target": e["target"], "start": e["start"], "freq": ds.freq} for e in entries
        ]
    else:
        train_inputs = build_wrapper_inputs_from_training(ds)
        if len(train_inputs) == 0:
            # Fallback if training split is empty for this dataset/term
            entries = list(ds.gluonts_dataset)
            train_inputs = [
                {"target": e["target"], "start": e["start"], "freq": ds.freq} for e in entries
            ]
    train_tsdf_full = TabPFNTSPredictor.convert_to_timeseries_dataframe(train_inputs)

    coll = ensure_persistent_collection(predictor)
    cname = predictor._moment_collection_name()
    ccount_before = coll.count()  # type: ignore
    size_before = dir_size_mb(predictor.chroma_db_dir)

    t0 = time.time()
    predictor._moment_index_windows(train_tsdf_full)
    dt = time.time() - t0

    ccount_after = coll.count()  # type: ignore
    size_after = dir_size_mb(predictor.chroma_db_dir)
    summary = {
        "collection": cname,
        "count_before": int(ccount_before),
        "count_after": int(ccount_after),
        "added": int(ccount_after - ccount_before),
        "db_dir": str(predictor.chroma_db_dir),
        "size_mb_before": round(size_before, 2),
        "size_mb_after": round(size_after, 2),
        "delta_mb": round(size_after - size_before, 2),
        "index_time_s": round(dt, 2),
        "source": source,
    }
    print(summary)
    return summary

# %%
ds = load_dataset_auto("M_DENSE/D", "short")
train_inputs = [{ "target": e["target"], "start": e["start"], "freq": ds.freq } for e in ds.gluonts_dataset]
tsdf = TabPFNTSPredictor.convert_to_timeseries_dataframe(train_inputs)
P = ds.prediction_length
sizes = tsdf.groupby(level="item_id").size().to_numpy()
expected = int(np.sum(np.maximum(0, sizes - P + 1)))  # step=1
print({"items": len(sizes), "min_len": int(sizes.min()), "max_len": int(sizes.max()), "expected_windows": expected})

# %%
summary = vectorize_dataset_db(
  "bizitobs_l2c/H", "short",
  context_length=48,              # optional; used only for predictor’s ctx (not embedding)
  chroma_db_dir=CHROMA_DB_DIR,    # or custom Path
  moment_ctx_len=0,               # 0 => adopt P
  moment_devices=["cuda:0","cuda:1","cuda:2","cuda:3"],
  moment_batch_size=4096,
  index_step=1,
  index_tail_trim_multiple=0,  # example tail trim in steps
  moment_index_sample_size=None,
  moment_model_id="AutonLab/MOMENT-1-large",
  source="full",
)
print(summary)

# %% [markdown]
# Query existing index directly: demonstrate where filters by timestamps and item_id

# %%
def query_existing_index(dataset_name: str, term: str, n_results: int = 10) -> None:
    section(f"Query existing Chroma index (read-only): {dataset_name} [{term}]")
    ds = Dataset(name=dataset_name, term=term, to_univariate=True, storage_path=DATASET_STORAGE_PATH)

    # Build a reader predictor with identical collection naming (no re-indexing)
    predictor = TabPFNTSPredictor(
        ds_prediction_length=ds.prediction_length,
        ds_freq=ds.freq,
        tabpfn_mode=TabPFNMode.MOCK,
        context_length=CONTEXT_LENGTH,
        batch_size=64,
        debug=True,
        few_shot_k=0,
        retrieval_mode="moment",
        dataset_id=ds.name,
        chroma_db_dir=CHROMA_DB_DIR,
        moment_ctx_len=MOMENT_CTX_LEN,
        moment_device=MOMENT_DEVICE,
        moment_devices=MOMENT_DEVICES,
        moment_batch_size=MOMENT_BATCH_SIZE,
        moment_progress=False,
        index_tail_trim_multiple=INDEX_TAIL_TRIM_MULT,
        index_step=INDEX_STEP,
    )

    # Ensure we open the same persistent collection
    coll = ensure_persistent_collection(predictor)
    cname = predictor._moment_collection_name()
    print({"collection": cname, "count": int(coll.count())})  # type: ignore

    # Build a sample query embedding from baseline context
    # Use test_data to create the same context preprocessing but with no retrieval
    test_data_input: List[Dict[str, Any]] = []
    for entry in ds.test_data:
        item = entry[0] if isinstance(entry, tuple) else entry
        test_data_input.append({
            "target": item["target"],
            "start": item["start"],
            "freq": ds.freq,
        })
    base_pred = TabPFNTSPredictor(
        ds_prediction_length=ds.prediction_length,
        ds_freq=ds.freq,
        tabpfn_mode=TabPFNMode.MOCK,
        context_length=CONTEXT_LENGTH,
        batch_size=64,
        debug=False,
        few_shot_k=0,
    )
    train_base, _ = base_pred._preprocess_test_data(test_data_input)
    item_id = train_base.item_ids[0]
    q_df = train_base.loc[item_id].iloc[-min(ds.prediction_length, len(train_base.loc[item_id])) :]
    q_vec = predictor._moment_embed_series(q_df["target"].values)
    q_start_ts = q_df.index.get_level_values("timestamp")[0]
    q_start_ts_int = int(pd.Timestamp(q_start_ts).value)
    step_ns = pd.tseries.frequencies.to_offset(ds.freq).nanos

    # 1) Global: ensure full 2P support ends before context start (use horizon_end_ts_int)
    where_global = {"horizon_end_ts_int": {"$lt": q_start_ts_int}}
    res1 = coll.query(query_embeddings=[q_vec], n_results=n_results, where=where_global, include=["metadatas", "distances"])  # type: ignore
    metas1 = res1.get("metadatas", [[]])[0]
    dists1 = res1.get("distances", [[]])[0]
    print("Global < ctx_start results:", len(metas1))
    for j, md in enumerate(metas1[: min(5, len(metas1))]):
        print(f"  j={j} item_id={md.get('item_id')} start={md.get('start_ts')} end={md.get('end_ts')} dist={dists1[j] if j < len(dists1) else None}")

    # 2) Per-item: same item_id, full 2P ends before context start
    where_per_item = {"$and": [{"item_id": {"$eq": str(item_id)}}, {"horizon_end_ts_int": {"$lt": q_start_ts_int}}]}
    res2 = coll.query(query_embeddings=[q_vec], n_results=n_results, where=where_per_item, include=["metadatas", "distances"])  # type: ignore
    metas2 = res2.get("metadatas", [[]])[0]
    dists2 = res2.get("distances", [[]])[0]
    print("Per-item < ctx_start results:", len(metas2))
    for j, md in enumerate(metas2[: min(5, len(metas2))]):
        print(f"  j={j} item_id={md.get('item_id')} start={md.get('start_ts')} end={md.get('end_ts')} dist={dists2[j] if j < len(dists2) else None}")

    # 3) Time-bounded window: constrain the key window to a pre-context band
    lower = q_start_ts_int - 30 * step_ns
    upper = q_start_ts_int - 1 * step_ns
    where_band = {"$and": [{"start_ts_int": {"$gte": int(lower)}}, {"end_ts_int": {"$lte": int(upper)}}]}
    res3 = coll.query(query_embeddings=[q_vec], n_results=n_results, where=where_band, include=["metadatas", "distances"])  # type: ignore
    metas3 = res3.get("metadatas", [[]])[0]
    dists3 = res3.get("distances", [[]])[0]
    print("Within pre-context band results:", len(metas3))
    for j, md in enumerate(metas3[: min(5, len(metas3))]):
        print(f"  j={j} item_id={md.get('item_id')} start={md.get('start_ts')} end={md.get('end_ts')} dist={dists3[j] if j < len(dists3) else None}")

    # 4) Band on horizon end (ensure horizons also lie well before context)
    where_hor_band = {"$and": [{"horizon_end_ts_int": {"$lte": int(upper)}}]}
    res4 = coll.query(query_embeddings=[q_vec], n_results=n_results, where=where_hor_band, include=["metadatas", "distances"])  # type: ignore
    metas4 = res4.get("metadatas", [[]])[0]
    dists4 = res4.get("distances", [[]])[0]
    print("Horizon-end band results:", len(metas4))
    for j, md in enumerate(metas4[: min(5, len(metas4))]):
        print(f"  j={j} item_id={md.get('item_id')} start={md.get('start_ts')} end={md.get('end_ts')} horizon_end={md.get('horizon_end_ts')} dist={dists4[j] if j < len(dists4) else None}")

    # Additionally dump a random entry for inspection
    import random
    if len(metas1):
        md = random.choice(metas1)
        print("Random entry (global filter):", md)


# %% [markdown]
# Run indexing for both datasets (one-time), verify reuse, run quick retrieval

# %%
# for (ds_name, ds_term) in DATASETS:
#     index_dataset_once(ds_name, ds_term)

for (ds_name, ds_term) in DATASETS:
    verify_reuse(ds_name, ds_term)

# Optional: run a small retrieval to confirm reuse in action
quick_retrieval_test(DATASETS[0][0], DATASETS[0][1], few_shot_k=2)

# Demonstrate read-only querying of the existing index with where filters
query_existing_index(DATASETS[0][0], DATASETS[0][1], n_results=10)



# %%
query_existing_index("bizitobs_service", "short", n_results=11)
