# %%
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter


# %%
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGGREGATED_CSV_PATH = OUTPUT_DIR / "aggregated_results.csv"

## Base CTX_LEN per dataset/frequency (normalized to lowercase dataset names)
BASE_CTX_LEN_MAP: Dict[str, int] = {
    "loop_seattle/d": 30,
    "m_dense/d": 30,
    "m_dense/h": 48,
    "sz_taxi/15t": 48,
    "sz_taxi/h": 48,
    "bitbrains_fast_storage/h": 48,
    "bitbrains_rnd/h": 48,
    "bizitobs_application": 60,
    "bizitobs_l2c/h": 48,
    "bizitobs_service": 60,
    "hierarchical_sales/d": 30,
    "hierarchical_sales/w": 8,
}


# %%
def parse_dataset_field(dataset_field: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse dataset field formatted as one of:
    - "dataset_name/freq/term" (common)
    - "dataset_name/term" (rare; infer no freq)

    Returns (dataset_name, freq, term)
    """
    parts = dataset_field.split("/")
    if len(parts) >= 3:
        dataset_name, freq, term = parts[0], parts[1], parts[2]
        return dataset_name, freq, term
    if len(parts) == 2:
        dataset_name, term = parts[0], parts[1]
        return dataset_name, None, term
    # Fallback: only dataset name present; term/freq unknown
    return dataset_field, None, None


def parse_tail_context_dirname(dirname: str) -> Optional[int]:
    """Extract context length from directory name like 'tabpfn-ts-1000cntx'."""
    m = re.match(r"^tabpfn-ts-(\d+)cntx$", dirname)
    if not m:
        return None
    return int(m.group(1))


def parse_random_retrieval_dirname(dirname: str) -> Optional[Dict[str, Optional[str]]]:
    """
    Extract top_k, ctx, dataset hint, and term hint from random retrieval directory name,
    e.g., 'tabpfn-ts-random-retrieval-top-k10-ctx48-dataset-M_DENSE/H-term-short'

    Returns dict with keys: top_k (int), ctx (int), dataset_hint (str), freq_hint (str), term (str)
    Any value may be None if not parsable.
    """
    if not dirname.startswith("tabpfn-ts-random-retrieval-"):
        return None

    info: Dict[str, Optional[str]] = {
        "top_k": None,
        "ctx": None,
        "dataset_hint": None,
        "freq_hint": None,
        "term": None,
    }

    m_topk = re.search(r"top-k(\d+)", dirname)
    if m_topk:
        info["top_k"] = int(m_topk.group(1))
    else:
        # Fallback for alternative naming like 'top1'
        m_top_alt = re.search(r"top(\d+)", dirname)
        if m_top_alt:
            info["top_k"] = int(m_top_alt.group(1))

    m_ctx = re.search(r"ctx(\d+)", dirname)
    if m_ctx:
        info["ctx"] = int(m_ctx.group(1))

    # dataset-M_DENSE/H or dataset-LOOP_SEATTLE etc.
    m_dataset = re.search(r"dataset-([^/\-]+)(?:/([A-Za-z0-9]+))?", dirname)
    if m_dataset:
        info["dataset_hint"] = m_dataset.group(1)
        if m_dataset.lastindex and m_dataset.lastindex >= 2:
            info["freq_hint"] = m_dataset.group(2)

    m_term = re.search(r"term-(short|medium|long)", dirname)
    if m_term:
        info["term"] = m_term.group(1)

    return info


def parse_ctx_topk_from_model(model_value: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse context_len and top_k from a model string if present."""
    if not model_value or not isinstance(model_value, str):
        return None, None
    ctx_val: Optional[int] = None
    topk_val: Optional[int] = None
    m_ctx = re.search(r"ctx(\d+)", model_value)
    if m_ctx:
        try:
            ctx_val = int(m_ctx.group(1))
        except Exception:
            pass
    m_topk = re.search(r"top-k(\d+)", model_value) or re.search(r"top(\d+)", model_value)
    if m_topk:
        try:
            topk_val = int(m_topk.group(1))
        except Exception:
            pass
    return ctx_val, topk_val


def lookup_base_ctx_len(dataset_name: Optional[str], freq: Optional[str]) -> Optional[int]:
    if not dataset_name:
        return None
    ds = str(dataset_name).lower()
    fq = str(freq).lower() if freq is not None and str(freq) != "None" else None
    if fq is not None:
        key = f"{ds}/{fq}"
        if key in BASE_CTX_LEN_MAP:
            return BASE_CTX_LEN_MAP[key]
    # fallback to dataset-only key
    return BASE_CTX_LEN_MAP.get(ds)


def read_results_csv(csv_path: Path) -> pd.DataFrame:
    """Read a results.csv with robust dtype handling."""
    try:
        df = pd.read_csv(csv_path)
        df["__source_csv_path"] = str(csv_path)
        return df
    except Exception as exc:
        print(f"Failed to read {csv_path}: {exc}")
        return pd.DataFrame()


def standardize_dataframe(df: pd.DataFrame, setup: str, context_len: Optional[int], top_k: Optional[int]) -> pd.DataFrame:
    """
    Standardize a single results dataframe into a consistent schema.

    Adds columns: setup, context_len, top_k, dataset_name, freq, term, wql, model, domain, num_variates
    Keeps original metric columns when present.
    """
    if df.empty:
        return df

    # Ensure required columns exist even if missing
    for col in [
        "dataset",
        "model",
        "eval_metrics/mean_weighted_sum_quantile_loss",
        "domain",
        "num_variates",
    ]:
        if col not in df.columns:
            df[col] = None

    parsed = df["dataset"].apply(parse_dataset_field)
    df["dataset_name"] = parsed.apply(lambda t: t[0])
    df["freq"] = parsed.apply(lambda t: t[1])
    df["term"] = parsed.apply(lambda t: t[2])

    # Normalize case for dataset names
    df["dataset_name"] = df["dataset_name"].astype(str).str.strip()
    df["freq"] = df["freq"].astype(str).where(df["freq"].notna(), None)
    df["term"] = df["term"].astype(str).where(df["term"].notna(), None)

    # Metric aliases
    if "eval_metrics/mean_weighted_sum_quantile_loss" in df.columns:
        df["WQL"] = pd.to_numeric(df["eval_metrics/mean_weighted_sum_quantile_loss"], errors="coerce")
    else:
        df["WQL"] = pd.NA

    df["setup"] = setup
    df["context_len"] = context_len
    df["top_k"] = top_k

    # Fallbacks from model if missing
    if (df["context_len"].isna()).any() or (df["top_k"].isna()).any():
        parsed_ctx_topk = df["model"].apply(parse_ctx_topk_from_model)
        # fill where missing
        df["context_len"] = df.apply(
            lambda r: r["context_len"] if pd.notna(r["context_len"]) else parsed_ctx_topk[r.name][0], axis=1
        )
        df["top_k"] = df.apply(
            lambda r: r["top_k"] if pd.notna(r["top_k"]) else parsed_ctx_topk[r.name][1], axis=1
        )

    # Base CTX_LEN per dataset/freq
    df["base_context_len"] = df.apply(lambda r: lookup_base_ctx_len(r.get("dataset_name"), r.get("freq")), axis=1)

    # For random retrieval: derive retrieval_context_len = top_k*(base*2) + base
    # For tail context: comparable_context_len equals context_len
    df["retrieval_context_len"] = pd.NA
    if setup == "random_retrieval":
        # ensure numeric
        bk = pd.to_numeric(df["base_context_len"], errors="coerce")
        tk = pd.to_numeric(df["top_k"], errors="coerce")
        df["retrieval_context_len"] = (tk * (bk * 2) + bk)

    # Comparable context length across setups
    df["comparable_context_len"] = df["context_len"]
    if setup == "random_retrieval":
        df["comparable_context_len"] = df["retrieval_context_len"]

    # Move common columns to a friendly order
    preferred_cols = [
        "setup",
        "dataset_name",
        "freq",
        "term",
        "context_len",
        "base_context_len",
        "top_k",
        "retrieval_context_len",
        "comparable_context_len",
        "WQL",
        "model",
        "domain",
        "num_variates",
    ]
    other_cols = [c for c in df.columns if c not in preferred_cols]
    df = df[preferred_cols + other_cols]
    return df


def collect_tail_context_results(root: Path) -> pd.DataFrame:
    """
    Aggregate initial tail-context results from directories named 'tabpfn-ts-<cntx>cntx'.
    Skips any directory containing '-run'.
    """
    all_frames: List[pd.DataFrame] = []
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        if "-run" in item.name:
            continue
        context_len = parse_tail_context_dirname(item.name)
        if context_len is None:
            continue

        for csv_path in item.rglob("results.csv"):
            df = read_results_csv(csv_path)
            if df.empty:
                continue
            std = standardize_dataframe(df, setup="tail_context", context_len=context_len, top_k=None)
            all_frames.append(std)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def collect_random_retrieval_results(root: Path) -> pd.DataFrame:
    """
    Aggregate random-retrieval results from directories named like
    'tabpfn-ts-random-retrieval-top-k<k>-ctx<ctx>-dataset-...'.
    """
    all_frames: List[pd.DataFrame] = []
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        if not item.name.startswith("tabpfn-ts-random-retrieval-"):
            continue
        info = parse_random_retrieval_dirname(item.name)
        if info is None:
            continue
        top_k = info.get("top_k")
        ctx = info.get("ctx")

        for csv_path in item.rglob("results.csv"):
            df = read_results_csv(csv_path)
            if df.empty:
                continue
            std = standardize_dataframe(df, setup="random_retrieval", context_len=ctx, top_k=top_k)
            all_frames.append(std)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def save_aggregated_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Aggregated results saved to: {path}")


# %%
# Aggregate and save
tail_df = collect_tail_context_results(RESULTS_ROOT)
random_df = collect_random_retrieval_results(RESULTS_ROOT)

aggregated_df = pd.concat([d for d in [tail_df, random_df] if d is not None and not d.empty], ignore_index=True)

# Ensure consistent types
if not aggregated_df.empty:
    for col in ["context_len", "top_k"]:
        if col in aggregated_df.columns:
            aggregated_df[col] = pd.to_numeric(aggregated_df[col], errors="coerce")

save_aggregated_csv(aggregated_df, AGGREGATED_CSV_PATH)


# %%
def plot_tail_context_short_term(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot WQL vs context_len for short-term results in tail_context setup, one figure per dataset/freq."""
    short_tail = df[
        (df["setup"] == "tail_context") & (df["term"].str.lower() == "short")
    ].copy()
    if short_tail.empty:
        print("No tail_context short-term results to plot.")
        return

    output_dir = output_dir / "tail_context"
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = short_tail.groupby(["dataset_name", "freq"], dropna=False)
    for (dataset_name, freq), g in groups:
        g = g.dropna(subset=["context_len", "WQL"]).sort_values("context_len")
        if g.empty:
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(g["context_len"], g["WQL"], marker="o")
        title_freq = f"/{freq}" if isinstance(freq, str) and len(str(freq)) > 0 and str(freq) != "None" else ""
        plt.title(f"Tail Context - Short term WQL: {dataset_name}{title_freq}")
        plt.xlabel("context_len")
        plt.ylabel("WQL (mean weighted sum quantile loss)")
        plt.grid(True, alpha=0.3)
        safe_ds = str(dataset_name).replace("/", "-")
        safe_freq = str(freq) if freq is not None else "NA"
        out_path = output_dir / f"{safe_ds}_{safe_freq}_short_wql_vs_context_len.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


def plot_random_retrieval_short_term(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot WQL vs top_k for short-term results in random_retrieval setup, one figure per dataset/freq."""
    short_rr = df[
        (df["setup"] == "random_retrieval") & (df["term"].str.lower() == "short")
    ].copy()
    if short_rr.empty:
        print("No random_retrieval short-term results to plot.")
        return

    output_dir = output_dir / "random_retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = short_rr.groupby(["dataset_name", "freq"], dropna=False)
    for (dataset_name, freq), g in groups:
        g = g.dropna(subset=["top_k", "WQL"]).sort_values("top_k")
        if g.empty:
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(g["top_k"], g["WQL"], marker="o")
        title_freq = f"/{freq}" if isinstance(freq, str) and len(str(freq)) > 0 and str(freq) != "None" else ""
        plt.title(f"Random Retrieval - Short term WQL: {dataset_name}{title_freq}")
        plt.xlabel("top_k")
        plt.ylabel("WQL (mean weighted sum quantile loss)")
        plt.grid(True, alpha=0.3)
        safe_ds = str(dataset_name).replace("/", "-")
        safe_freq = str(freq) if freq is not None else "NA"
        out_path = output_dir / f"{safe_ds}_{safe_freq}_short_wql_vs_top_k.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


# %%
def plot_short_term_wql_vs_comparable_context(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot short-term WQL vs comparable_context_len, overlaying tail_context and random_retrieval
    for each dataset_name/freq.
    """
    short_df = df[df["term"].str.lower() == "short"].copy()
    if short_df.empty:
        print("No short-term results to plot for comparable context length.")
        return

    out_dir = output_dir / "context_matched"
    out_dir.mkdir(parents=True, exist_ok=True)

    for (dataset_name, freq), g in short_df.groupby(["dataset_name", "freq"], dropna=False):
        g = g.dropna(subset=["comparable_context_len", "WQL"]).sort_values("comparable_context_len")
        if g.empty:
            continue

        plt.figure(figsize=(6, 4))
        # Tail-context line
        g_tail = g[g["setup"] == "tail_context"]
        if not g_tail.empty:
            plt.plot(g_tail["comparable_context_len"], g_tail["WQL"], marker="o", label="tail_context")
        # Random-retrieval line
        g_rr = g[g["setup"] == "random_retrieval"]
        if not g_rr.empty:
            plt.plot(g_rr["comparable_context_len"], g_rr["WQL"], marker="o", label="random_retrieval")

        if g_tail.empty and g_rr.empty:
            plt.close()
            continue

        title_freq = f"/{freq}" if isinstance(freq, str) and len(str(freq)) > 0 and str(freq) != "None" else ""
        plt.title(f"Short term WQL vs Context Len: {dataset_name}{title_freq}")
        plt.xlabel("context_len (tail) / effective_context_len (retrieval)")
        plt.ylabel("WQL (mean weighted sum quantile loss)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add top x-axis showing top_k using mapping: ctx = base*(2*top_k + 1)
        base_vals = pd.to_numeric(g["base_context_len"], errors="coerce").dropna().unique()
        if base_vals.size > 0 and base_vals[0] > 0:
            base = float(base_vals[0])

            def ctx_to_topk(x):
                return (x / base - 1.0) / 2.0

            def topk_to_ctx(k):
                return base * (2.0 * k + 1.0)

            ax = plt.gca()
            ax_top = ax.secondary_xaxis("top", functions=(ctx_to_topk, topk_to_ctx))
            ax_top.set_xlabel("top_k")
            # Use sparse, integer ticks to avoid overlap
            ax_top.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune="both"))
            ax_top.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x))}"))
            ax_top.tick_params(axis="x", labelsize=8, pad=4)
        safe_ds = str(dataset_name).replace("/", "-")
        safe_freq = str(freq) if freq is not None else "NA"
        out_path = out_dir / f"{safe_ds}_{safe_freq}_short_wql_vs_context_len_compare.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


# %%
if not aggregated_df.empty:
    plot_tail_context_short_term(aggregated_df, OUTPUT_DIR)
    plot_random_retrieval_short_term(aggregated_df, OUTPUT_DIR)
    plot_short_term_wql_vs_comparable_context(aggregated_df, OUTPUT_DIR)
else:
    print("Aggregated dataframe is empty; skipping plots.")


