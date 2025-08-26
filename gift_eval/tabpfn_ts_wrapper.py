from typing import Iterator, Tuple, Optional, List, Dict, Sequence
import logging

import numpy as np
import pandas as pd
from gluonts.model.forecast import QuantileForecast, Forecast
from gluonts.itertools import batcher
from pathlib import Path
import re
from math import ceil

from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series import (
    TabPFNTimeSeriesPredictor,
    FeatureTransformer,
    TabPFNMode,
    TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
    TimeSeriesDataFrame,
)
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)

logger = logging.getLogger(__name__)


class TabPFNTSPredictor:
    DEFAULT_FEATURES = [
        RunningIndexFeature(),
        CalendarFeature(),
        AutoSeasonalFeature(),
    ]

    def __init__(
        self,
        ds_prediction_length: int,
        ds_freq: str,
        tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL,
        context_length: int = 4096,
        batch_size: int = 1024,
        debug: bool = False,
        few_shot_k: int = 0,
        few_shot_len: int = 0,
        few_shot_seed: int = 42,
        # Retrieval strategy: "random" (default) or "moment"
        retrieval_mode: str = "random",
        # MOMENT + Chroma options (used when retrieval_mode=="moment")
        dataset_id: Optional[str] = None,
        chroma_db_dir: Optional[Path] = None,
        moment_model_id: str = "AutonLab/MOMENT-1-large",
        moment_ctx_len: int = 512,
        moment_device: Optional[str] = None,
        moment_devices: Optional[Sequence[str]] = None,
        moment_batch_size: int = 64,
        moment_index_sample_size: Optional[int] = None,
        moment_progress: bool = False,
        moment_local_files_only: bool = False,
        retrieval_scope: str = "per_item",  # or "global"
    ):
        self.ds_prediction_length = ds_prediction_length
        self.ds_freq = ds_freq
        self.tabpfn_predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=tabpfn_mode,
        )
        self.context_length = context_length
        self.debug = debug
        self.batch_size = batch_size
        self.few_shot_k = max(0, few_shot_k)
        # If not specified, default retrieval window to context_length + prediction_length
        self.few_shot_len = (
            few_shot_len if few_shot_len > 0 else (context_length + ds_prediction_length)
        )
        self.few_shot_seed = few_shot_seed

        # Retrieval configuration
        self.retrieval_mode = retrieval_mode.lower().strip()
        self.dataset_id = dataset_id or "unknown-dataset"
        # Default persistent DB under repo gift_eval/.moment_chroma if not provided
        self.chroma_db_dir = (
            Path(chroma_db_dir) if chroma_db_dir is not None else Path(__file__).parent / ".moment_chroma"
        )
        self.moment_model_id = moment_model_id
        self.moment_ctx_len = int(moment_ctx_len)
        self.moment_device = moment_device
        self.moment_devices = list(moment_devices) if moment_devices is not None else None
        self.moment_batch_size = int(moment_batch_size)
        self.moment_index_sample_size = moment_index_sample_size
        self.moment_progress = moment_progress
        self.moment_local_files_only = moment_local_files_only
        self.retrieval_scope = retrieval_scope

        # Lazy-loaded MOMENT pipeline and Chroma collection
        self._moment_pipeline = None  # type: ignore
        self._moment_pipelines: Dict[str, object] = {}
        self._chroma_client = None  # type: ignore
        self._chroma_collection = None  # type: ignore
        self._last_retrieval_info: Dict[str, List[Dict]] = {}

        self.feature_transformer = FeatureTransformer(self.DEFAULT_FEATURES)

    def predict(self, test_data_input) -> Iterator[Forecast]:
        logger.debug(f"len(test_data_input): {len(test_data_input)}, batch size: {self.batch_size}")

        forecasts = []
        for batch in batcher(test_data_input, batch_size=self.batch_size):
            forecasts.extend(self._predict_batch(batch))

        return forecasts

    def _predict_batch(self, test_data_input):
        logger.debug(f"Processing batch of size: {len(test_data_input)}")

        # Preprocess the input data
        train_tsdf, test_tsdf = self._preprocess_test_data(test_data_input)

        # TODO: for fixed context length retrieve chunks from 'larger train' here
        # TODO 1: check last and first timestamp 
        # overlap of 'train' vs 'local test' that gets split into context and test query

        # Generate predictions
        pred: TimeSeriesDataFrame = self.tabpfn_predictor.predict(train_tsdf, test_tsdf)
        pred = pred.drop(columns=["target"])

        # Pre-allocate forecasts list and get forecast quantile keys
        forecasts = [None] * len(pred.item_ids)
        forecast_keys = list(map(str, TABPFN_TS_DEFAULT_QUANTILE_CONFIG))

        # Generate QuantileForecast objects for each time series
        for i, (_, item_data) in enumerate(pred.groupby(level="item_id")):
            forecast_start_timestamp = item_data.index.get_level_values(1)[0]
            forecasts[i] = QuantileForecast(
                forecast_arrays=item_data.values.T,
                forecast_keys=forecast_keys,
                start_date=forecast_start_timestamp.to_period(self.ds_freq),
            )

        logger.debug(f"Generated {len(forecasts)} forecasts")
        return forecasts

    def _preprocess_test_data(
        self, test_data_input
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """
        Preprocess includes:
        - Turn the test_data_input into a TimeSeriesDataFrame
        - Handle NaN values in "target" column
        - If context_length is set, slice the train_tsdf to the last context_length timesteps
        - Generate test data and apply feature transformations
        """
        # Convert input to TimeSeriesDataFrame
        train_tsdf_full = self.convert_to_timeseries_dataframe(test_data_input)

        # Handle NaN values
        train_tsdf_full = self.handle_nan_values(train_tsdf_full)

        # Assert no more NaN in train_tsdf target
        assert not train_tsdf_full.target.isnull().any()

        # Slice if needed
        if self.context_length > 0:
            logger.info(
                f"Slicing train_tsdf to {self.context_length} timesteps for each time series"
            )
            train_tsdf = train_tsdf_full.slice_by_timestep(-self.context_length, None)
        else:
            train_tsdf = train_tsdf_full

        # Few-shot augmentation: either random (baseline) or MOMENT+Chroma retrieval
        if self.few_shot_k > 0 and self.few_shot_len > 0:
            if self.retrieval_mode == "moment":
                train_tsdf = self._augment_with_moment_retrieval(
                    train_tsdf=train_tsdf, train_tsdf_full=train_tsdf_full
                )
            else:
                train_tsdf = self._augment_with_random_retrieval(
                    train_tsdf=train_tsdf, train_tsdf_full=train_tsdf_full
                )

        # Generate test data and features
        test_tsdf = generate_test_X(
            train_tsdf, prediction_length=self.ds_prediction_length, freq=self.ds_freq
        )
        
        # Assign segment_id to baseline context+test (segment 0), and fill any missing
        train_tsdf = train_tsdf.copy()
        if "segment_id" in train_tsdf.columns:
            train_tsdf["segment_id"] = train_tsdf["segment_id"].fillna(0)
        else:
            train_tsdf["segment_id"] = 0
        test_tsdf = test_tsdf.copy()
        test_tsdf["segment_id"] = 0

        train_tsdf, test_tsdf = self.feature_transformer.transform(
            train_tsdf, test_tsdf
        )

        # Log baseline context and horizon boundaries (segment 0)
        if self.debug:
            for item_id, item_df in train_tsdf.groupby(level="item_id", sort=False):
                seg0 = item_df  # segment_id already dropped later; infer from time ordering
                # context = last context_length rows of pre-feature train_tsdf_full per item
                # approximate via last context_length of item_df where available
                ctx_len = min(self.context_length, len(item_df))
                ctx_slice = item_df.iloc[-ctx_len:]
                ctx_start = ctx_slice.index.get_level_values("timestamp")[0]
                ctx_end = ctx_slice.index.get_level_values("timestamp")[-1]
                hor_start = test_tsdf.loc[item_id].index.get_level_values("timestamp")[0]
                hor_end = test_tsdf.loc[item_id].index.get_level_values("timestamp")[-1]
                logger.info(
                    "baseline item_id=%s | ctx_start=%s ctx_end=%s | hor_start=%s hor_end=%s",
                    str(item_id), str(ctx_start), str(ctx_end), str(hor_start), str(hor_end)
                )

        # Drop segment_id from features to avoid leaking segment identity to the model
        if "segment_id" in train_tsdf.columns:
            train_tsdf = train_tsdf.drop(columns=["segment_id"])  # type: ignore
        if "segment_id" in test_tsdf.columns:
            test_tsdf = test_tsdf.drop(columns=["segment_id"])  # type: ignore

        return train_tsdf, test_tsdf

    # ----------------------------
    # Baseline random retrieval
    # ----------------------------
    def _augment_with_random_retrieval(
        self,
        train_tsdf: TimeSeriesDataFrame,
        train_tsdf_full: TimeSeriesDataFrame,
    ) -> TimeSeriesDataFrame:
        rng = np.random.default_rng(self.few_shot_seed)
        support_segments: List[pd.DataFrame] = []
        base_rows = len(train_tsdf)

        for item_id, full_item_df in train_tsdf_full.groupby(level="item_id", sort=False):
            # Determine range to sample from: exclude the most recent context region
            # and constrain to be within (few_shot_len + prediction_length) from the context cutoff.
            full_item_df = full_item_df.copy()
            full_len = len(full_item_df)
            exclude_recent = max(0, self.context_length)

            start_of_context = full_len - exclude_recent
            if start_of_context <= 0:
                continue

            # Allowed end index window (inclusive):
            #   earliest_end = start_of_context - (few_shot_len + prediction_length)
            #   latest_end   = start_of_context - 1
            earliest_end = max(0, start_of_context - (self.few_shot_len + self.ds_prediction_length))
            latest_end = start_of_context - 1

            # Convert to allowed start range (inclusive):
            # start ∈ [earliest_end - (L-1), latest_end - (L-1)]
            start_low = max(0, earliest_end - (self.few_shot_len - 1))
            start_high = latest_end - (self.few_shot_len - 1)

            if start_high < start_low:
                continue

            # Build candidate starts and sample
            # Keep overhead low by using range bounds directly in choice
            num_candidates = start_high - start_low + 1
            num_samples = min(self.few_shot_k, num_candidates)
            starts = start_low + rng.choice(num_candidates, size=num_samples, replace=False)
            seg_counter = 1
            for s in starts:
                window_df = full_item_df.iloc[s : s + self.few_shot_len].copy()
                window_df["segment_id"] = seg_counter
                if self.debug:
                    w_start_ts = window_df.index.get_level_values("timestamp")[0]
                    w_end_ts = window_df.index.get_level_values("timestamp")[-1]
                    logger.info(
                        "retrieval segment item_id=%s seg=%d | start=%s end=%s",
                        str(item_id), seg_counter, str(w_start_ts), str(w_end_ts)
                    )
                seg_counter += 1
                support_segments.append(window_df)

        if support_segments:
            support_tsdf = TimeSeriesDataFrame(pd.concat(support_segments))
            train_tsdf = TimeSeriesDataFrame(pd.concat([train_tsdf, support_tsdf]).sort_index())
            appended_rows = len(train_tsdf) - base_rows
            logger.info(
                "Retrieval appended rows: %d | mode=random | top_k=%d | few_shot_len=%d | context_length=%d | prediction_length=%d",
                appended_rows,
                self.few_shot_k,
                self.few_shot_len,
                self.context_length,
                self.ds_prediction_length,
            )
        else:
            logger.info(
                "Retrieval appended rows: %d | mode=random | top_k=%d | few_shot_len=%d | context_length=%d | prediction_length=%d",
                0,
                self.few_shot_k,
                self.few_shot_len,
                self.context_length,
                self.ds_prediction_length,
            )

        return train_tsdf

    # ----------------------------
    # MOMENT + Chroma retrieval
    # ----------------------------
    def _augment_with_moment_retrieval(
        self,
        train_tsdf: TimeSeriesDataFrame,
        train_tsdf_full: TimeSeriesDataFrame,
    ) -> TimeSeriesDataFrame:
        try:
            import chromadb
        except Exception as e:
            logger.error("chromadb is required for MOMENT retrieval. Please install it: pip install chromadb")
            raise

        # Ensure Chroma client and collection
        if self._chroma_client is None:
            self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
            # Use PersistentClient for local, on-disk processing
            self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_db_dir))

        collection_name = self._moment_collection_name()
        if self._chroma_collection is None:
            # Get or create collection with cosine distance
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        # Index any missing candidate windows for this batch
        self._moment_index_windows(train_tsdf_full)

        # Build per-item query embeddings from the current context
        query_segments: List[pd.DataFrame] = []
        self._last_retrieval_info = {}
        for item_id, item_df in train_tsdf.groupby(level="item_id", sort=False):
            # Query is the last moment_ctx_len of current context region
            ctx_slice = item_df.iloc[-min(self.moment_ctx_len, len(item_df)) :].copy()
            ctx_slice["segment_id"] = 0
            query_segments.append(ctx_slice)
            self._last_retrieval_info[str(item_id)] = []

        if not query_segments:
            return train_tsdf

        # Compute embeddings for queries
        queries_tsdf = TimeSeriesDataFrame(pd.concat(query_segments))
        q_ids: List[str] = []
        q_vecs: List[List[float]] = []
        q_meta: List[Dict] = []
        for item_id, q_df in queries_tsdf.groupby(level="item_id", sort=False):
            q_vec = self._moment_embed_series(q_df.target.values)
            q_ids.append(f"query::{self.dataset_id}::{str(item_id)}")
            q_vecs.append(q_vec)
            # cutoff for windows: strictly before the context start (safe bound)
            q_start_ts = q_df.index.get_level_values("timestamp")[0]
            q_meta.append({
                "item_id": str(item_id),
                "q_start_ts_int": int(q_start_ts.value),
            })

        # Perform per-item or global queries and collect top-k windows
        support_segments: List[pd.DataFrame] = []
        base_rows = len(train_tsdf)
        for i, q_id in enumerate(q_ids):
            # Build Chroma filter with a single top-level operator
            conds: List[Dict[str, object]] = []
            if self.retrieval_scope == "per_item":
                conds.append({"item_id": {"$eq": q_meta[i]["item_id"]}})
            conds.append({"end_ts_int": {"$lt": q_meta[i]["q_start_ts_int"]}})
            where: Dict[str, object]
            if len(conds) == 1:
                # single condition still needs an operator wrapper
                where = {"$and": conds}
            else:
                where = {"$and": conds}

            result = self._chroma_collection.query(
                query_embeddings=[q_vecs[i]],
                n_results=self.few_shot_k,
                where=where,
                include=["metadatas", "distances"],
            )

            ids = result.get("ids", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            dists = result.get("distances", [[]])[0]

            if self.debug:
                logger.info("MOMENT retrieval item=%s top_k=%d", q_meta[i]["item_id"], len(ids))
                for j, mid in enumerate(ids):
                    logger.info("  #%d id=%s dist=%.4f meta=%s", j + 1, str(mid), float(dists[j]), metas[j])

            # Reconstruct windows from metadata
            for j, md in enumerate(metas):
                iid = int(md["item_id"]) if str(md["item_id"]).isdigit() else md["item_id"]
                st = pd.Timestamp(md["start_ts"])
                en = pd.Timestamp(md["end_ts"])
                full_df = train_tsdf_full.loc[iid]
                window_df = full_df.loc[st:en, :].copy()
                # Restore MultiIndex [item_id, timestamp] expected by TimeSeriesDataFrame
                if not isinstance(window_df.index, pd.MultiIndex):
                    window_df.index = pd.MultiIndex.from_product(
                        [[iid], window_df.index], names=["item_id", "timestamp"]
                    )
                window_df["segment_id"] = j + 1
                support_segments.append(window_df)
                # Track retrieval info for debugging/plotting
                self._last_retrieval_info.setdefault(str(q_meta[i]["item_id"]), []).append(
                    {
                        "item_id": str(iid),
                        "start_ts": str(st),
                        "end_ts": str(en),
                        "distance": float(dists[j]) if j < len(dists) else None,
                    }
                )

        if support_segments:
            support_tsdf = TimeSeriesDataFrame(pd.concat(support_segments))
            train_tsdf = TimeSeriesDataFrame(pd.concat([train_tsdf, support_tsdf]).sort_index())
            appended_rows = len(train_tsdf) - base_rows
            logger.info(
                "Retrieval appended rows: %d | mode=moment | top_k=%d | few_shot_len=%d | context_length=%d | prediction_length=%d",
                appended_rows,
                self.few_shot_k,
                self.few_shot_len,
                self.context_length,
                self.ds_prediction_length,
            )
        else:
            logger.info(
                "Retrieval appended rows: %d | mode=moment | top_k=%d | few_shot_len=%d | context_length=%d | prediction_length=%d",
                0,
                self.few_shot_k,
                self.few_shot_len,
                self.context_length,
                self.ds_prediction_length,
            )

        return train_tsdf

    def _moment_collection_name(self) -> str:
        raw = f"{self.dataset_id}_freq_{self.ds_freq}_pred_{self.ds_prediction_length}_win_{self.few_shot_len}"
        # Sanitize to match Chroma's [a-zA-Z0-9._-], 3-512 chars, start/end alnum
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", raw)
        safe = safe.strip("._-")
        if not safe:
            safe = "ds"
        # Ensure starts/ends with alnum
        if not re.match(r"^[a-zA-Z0-9]", safe):
            safe = f"d{safe}"
        if not re.search(r"[a-zA-Z0-9]$", safe):
            safe = f"{safe}0"
        # Truncate to max length
        return safe[:200]

    def _moment_index_windows(self, train_tsdf_full: TimeSeriesDataFrame) -> None:
        """
        Build embeddings for all eligible windows in train_tsdf_full and upsert into Chroma.
        Eligible windows end before the current context start per item and have length few_shot_len.
        """
        to_upsert_ids: List[str] = []
        to_upsert_vecs: List[List[float]] = []
        to_upsert_metas: List[Dict] = []

        for item_id, full_item_df in train_tsdf_full.groupby(level="item_id", sort=False):
            full_item_df = full_item_df.copy()
            full_len = len(full_item_df)
            exclude_recent = max(0, self.context_length)
            start_of_context = full_len - exclude_recent
            if start_of_context <= 0:
                continue

            earliest_end = max(0, start_of_context - (self.few_shot_len + self.ds_prediction_length))
            latest_end = start_of_context - 1
            start_low = max(0, earliest_end - (self.few_shot_len - 1))
            start_high = latest_end - (self.few_shot_len - 1)
            if start_high < start_low:
                continue

            # Iterate possible starting indices; to limit compute, only take non-overlapping step of stride=1
            for s in range(start_low, start_high + 1):
                sub_df = full_item_df.iloc[s : s + self.few_shot_len]
                st_ts = sub_df.index.get_level_values("timestamp")[0]
                en_ts = sub_df.index.get_level_values("timestamp")[-1]
                uid = f"{self.dataset_id}::{str(item_id)}::{int(st_ts.value)}::{int(en_ts.value)}"
                to_upsert_ids.append(uid)
                to_upsert_metas.append({
                    "dataset_id": self.dataset_id,
                    "item_id": str(item_id),
                    "start_ts": str(st_ts),
                    "end_ts": str(en_ts),
                    "start_ts_int": int(st_ts.value),
                    "end_ts_int": int(en_ts.value),
                    "length": int(self.few_shot_len),
                })
                # Collect vectors later in a batch to avoid recomputing if present

        if not to_upsert_ids:
            return

        # Filter out already indexed ids
        existing = set()
        try:
            # Chroma does not support bulk existence check efficiently; chunk
            chunk = 256
            for i in range(0, len(to_upsert_ids), chunk):
                fetched = self._chroma_collection.get(ids=to_upsert_ids[i : i + chunk], include=[])  # type: ignore
                for fid in fetched.get("ids", []):
                    existing.add(fid)
        except Exception:
            # If .get not supported for non-existent ids, ignore and upsert all
            existing = set()

        # Select missing ids (optionally sample down to toy size)
        missing_pairs = [(uid, md) for uid, md in zip(to_upsert_ids, to_upsert_metas) if uid not in existing]
        if self.moment_index_sample_size is not None and len(missing_pairs) > self.moment_index_sample_size:
            # Deterministic sub-sample from start for reproducibility
            missing_pairs = missing_pairs[: self.moment_index_sample_size]

        # Compute embeddings for missing windows in batches (with optional tqdm)
        if missing_pairs:
            try:
                from tqdm.auto import tqdm as _tqdm  # type: ignore
                use_pbar = self.moment_progress
            except Exception:
                _tqdm = lambda x, **kwargs: x  # type: ignore
                use_pbar = False

            values_list: List[np.ndarray] = []
            indexed_item_ids: List[object] = []
            indexed_stamps: List[pd.Index] = []
            for uid, md in missing_pairs:
                iid = int(md["item_id"]) if str(md["item_id"]).isdigit() else md["item_id"]
                st = pd.Timestamp(md["start_ts"])
                en = pd.Timestamp(md["end_ts"])
                full_df = train_tsdf_full.loc[iid]
                sub_df = full_df.loc[st:en, :]
                values_list.append(sub_df.target.values.astype(np.float32))
                indexed_item_ids.append(iid)
                indexed_stamps.append(sub_df.index)

            # Batch forward
            total = len(values_list)
            bs = max(1, self.moment_batch_size)
            rng = range(0, total, bs)
            iterator = _tqdm(rng, total=ceil(total / bs), disable=not use_pbar, desc="Indexing MOMENT windows")

            batch_vecs: List[List[float]] = []
            for start in iterator:
                batch_vals = values_list[start : start + bs]
                batch_out = self._moment_embed_batch(batch_vals)
                batch_vecs.extend(batch_out)

            to_upsert_vecs.extend(batch_vecs)

        # Align ids, metas with computed vecs
        final_ids: List[str] = []
        final_metas: List[Dict] = []
        vi = 0
        for uid, md in missing_pairs:
            final_ids.append(uid)
            final_metas.append(md)
            vi += 1

        if final_ids:
            self._chroma_collection.upsert(ids=final_ids, embeddings=to_upsert_vecs, metadatas=final_metas)  # type: ignore
            if self.debug:
                # Log approximate on-disk size of Chroma directory
                try:
                    total_bytes = 0
                    for p in self.chroma_db_dir.rglob("*"):
                        if p.is_file():
                            total_bytes += p.stat().st_size
                    logger.info("ChromaDB dir size: %.2f MB at %s", total_bytes / (1024 * 1024), str(self.chroma_db_dir))
                except Exception:
                    pass

    def _moment_embed_series(self, values: np.ndarray) -> List[float]:
        """Compute MOMENT embedding for a 1D target series values."""
        # Use batch path for a single element
        out = self._moment_embed_batch([values])
        return out[0]

    def _ensure_moment_single_pipeline(self) -> None:
        if self._moment_pipeline is None:
            try:
                from momentfm import MOMENTPipeline
            except Exception:
                logger.error("momentfm is required for MOMENT retrieval. Please install it: pip install momentfm")
                raise
            self._moment_pipeline = MOMENTPipeline.from_pretrained(
                self.moment_model_id,
                model_kwargs={"task_name": "embedding"},
                local_files_only=self.moment_local_files_only,
            )
            self._moment_pipeline.init()
            # Prefer provided device; otherwise, auto-select CUDA if available
            try:
                import torch as _torch
                if self.moment_device:
                    try:
                        self._moment_pipeline.to(self.moment_device)
                    except Exception:
                        logger.warning("Failed to move MOMENT model to device %s; using default.", self.moment_device)
                else:
                    if _torch.cuda.is_available():
                        try:
                            self._moment_pipeline.to("cuda")
                            self.moment_device = "cuda"
                        except Exception:
                            pass
            except Exception:
                pass

            # Ensure eval mode to disable dropout/BN updates
            try:
                self._moment_pipeline.eval()  # type: ignore[attr-defined]
            except Exception:
                # Fallback if pipeline exposes underlying model
                try:
                    model_attr = getattr(self._moment_pipeline, "model", None)
                    if model_attr is not None and hasattr(model_attr, "eval"):
                        model_attr.eval()
                except Exception:
                    pass

    def _ensure_moment_multi_pipelines(self) -> List[str]:
        """Create per-device pipelines if multiple devices were requested."""
        devices: List[str] = []
        if self.moment_devices and len(self.moment_devices) > 1:
            devices = list(self.moment_devices)
            try:
                from momentfm import MOMENTPipeline
                for dev in devices:
                    if dev not in self._moment_pipelines:
                        pipe = MOMENTPipeline.from_pretrained(
                            self.moment_model_id,
                            model_kwargs={"task_name": "embedding"},
                            local_files_only=self.moment_local_files_only,
                        )
                        pipe.init()
                        try:
                            pipe.to(dev)
                        except Exception:
                            logger.warning("Failed to move MOMENT model to device %s; using default.", dev)
                        try:
                            pipe.eval()
                        except Exception:
                            pass
                        self._moment_pipelines[dev] = pipe
            except Exception:
                logger.warning("Falling back to single-device MOMENT; failed to init multi-device pipelines.")
                devices = []
        return devices

    def _moment_embed_batch(self, values_list: List[np.ndarray]) -> List[List[float]]:
        """Compute embeddings for a batch of 1D series using single or multi-device pipelines."""
        # Normalize and pad/truncate to fixed length first
        L = int(self.moment_ctx_len)
        proc: List[np.ndarray] = []
        for arr in values_list:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.std() > 0:
                arr = (arr - arr.mean()) / (arr.std() + 1e-8)
            if len(arr) >= L:
                arr_win = arr[-L:]
            else:
                pad = np.zeros(L, dtype=np.float32)
                pad[-len(arr) :] = arr
                arr_win = pad
            proc.append(arr_win)

        # If multiple devices requested, shard across them
        multi_devs = self._ensure_moment_multi_pipelines()
        import torch

        outputs: List[List[float]] = [None] * len(proc)  # type: ignore
        if multi_devs:
            # Round-robin assignment
            shards: Dict[str, List[Tuple[int, np.ndarray]]] = {dev: [] for dev in multi_devs}
            for idx, arr_win in enumerate(proc):
                dev = multi_devs[idx % len(multi_devs)]
                shards[dev].append((idx, arr_win))

            for dev, items in shards.items():
                if not items:
                    continue
                pipe = self._moment_pipelines.get(dev)
                batch = torch.from_numpy(np.stack([a for _, a in items], axis=0))[:, None, :]
                try:
                    batch = batch.to(dev)
                except Exception:
                    pass
                with torch.no_grad():
                    out = pipe(x_enc=batch)
                embs = out.embeddings.detach().cpu().numpy()
                for (idx, _), vec in zip(items, embs):
                    outputs[idx] = vec.tolist()
            return outputs

        # Single device path
        self._ensure_moment_single_pipeline()
        x = torch.from_numpy(np.stack(proc, axis=0))[:, None, :]
        if self.moment_device:
            try:
                x = x.to(self.moment_device)
            except Exception:
                pass
        with torch.no_grad():
            out = self._moment_pipeline(x_enc=x)
        embs = out.embeddings.detach().cpu().numpy().tolist()
        return embs

    @staticmethod
    def handle_nan_values(tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """
        Handle NaN values in the TimeSeriesDataFrame:
        - If time series has 0 or 1 valid value, fill with 0s
        - Else, drop the NaN values within the time series

        Args:
            tsdf: TimeSeriesDataFrame containing time series data

        Returns:
            TimeSeriesDataFrame: Processed data with NaN values handled
        """
        processed_series = []
        ts_with_0_or_1_valid_value = []
        ts_with_nan = []

        # Process each time series individually
        for item_id, item_data in tsdf.groupby(level="item_id"):
            target = item_data.target.values
            timestamps = item_data.index.get_level_values("timestamp")

            # If there are 0 or 1 valid values, fill NaNs with 0
            valid_value_count = np.count_nonzero(~np.isnan(target))
            if valid_value_count <= 1:
                ts_with_0_or_1_valid_value.append(item_id)
                target = np.where(np.isnan(target), 0, target)
                processed_df = pd.DataFrame(
                    {"target": target},
                    index=pd.MultiIndex.from_product(
                        [[item_id], timestamps], names=["item_id", "timestamp"]
                    ),
                )
                processed_series.append(processed_df)

            # Else drop NaN values
            elif np.isnan(target).any():
                ts_with_nan.append(item_id)
                valid_indices = ~np.isnan(target)
                processed_df = pd.DataFrame(
                    {"target": target[valid_indices]},
                    index=pd.MultiIndex.from_product(
                        [[item_id], timestamps[valid_indices]],
                        names=["item_id", "timestamp"],
                    ),
                )
                processed_series.append(processed_df)

            # No NaNs, keep as is
            else:
                processed_series.append(item_data)

        # Log warnings about NaN handling
        if ts_with_0_or_1_valid_value:
            logger.warning(
                f"Found time-series with 0 or 1 valid values, item_ids: {ts_with_0_or_1_valid_value}"
            )

        if ts_with_nan:
            logger.warning(
                f"Found time-series with NaN targets, item_ids: {ts_with_nan}"
            )

        # Combine processed series
        return TimeSeriesDataFrame(pd.concat(processed_series))

    @staticmethod
    def convert_to_timeseries_dataframe(test_data_input, use_covariates: bool = False):
        """
        Convert test_data_input to TimeSeriesDataFrame.

        Args:
            test_data_input: List of dictionaries containing time series data
            use_covariates: Whether to include covariates in the output

        Returns:
            TimeSeriesDataFrame: Converted data
        """
        # Pre-allocate list with known size
        time_series = [None] * len(test_data_input)

        for i, item in enumerate(test_data_input):
            target = item["target"]

            # Create timestamp index
            timestamp = pd.date_range(
                start=item["start"].to_timestamp(),
                periods=len(target),
                freq=item["freq"],
            )

            # Create DataFrame with target
            df = pd.DataFrame({"target": target}, index=timestamp)

            # Create MultiIndex DataFrame
            time_series[i] = df.set_index(
                pd.MultiIndex.from_product(
                    [[i], df.index], names=["item_id", "timestamp"]
                )
            )

        # Concat pre-allocated list
        return TimeSeriesDataFrame(pd.concat(time_series))
