#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger("prepare_huffpost")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean HuffPost dataset and generate embeddings."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to HuffPost dataset file (.json, .jsonl, .csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where cleaned data and embeddings will be saved.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, for example 'cpu' or 'cuda'. Leave empty for auto.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic subset sampling.",
    )
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        default=[10000, 25000, 50000],
        help="Subset sizes to generate.",
    )
    parser.add_argument(
        "--save-full-cleaned",
        action="store_true",
        help="Also save the full cleaned dataset as CSV before subset creation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    LOGGER.info("Reading input file: %s", path)

    if suffix == ".csv":
        df = pd.read_csv(path)
        return df

    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    if suffix == ".json":
        try:
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            if rows and isinstance(rows[0], dict):
                return pd.DataFrame(rows)
        except json.JSONDecodeError:
            pass

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)
        raise ValueError(
            "Unsupported JSON format. Expected list[dict] or JSONL-like file."
        )

    raise ValueError(f"Unsupported input format: {suffix}")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    text = normalize_whitespace(text)
    return text


def clean_and_structure(df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Initial row count: %d", len(df))

    required_columns = ["category", "headline", "short_description", "date"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    working = df.copy()

    working.columns = [col.strip().lower() for col in working.columns]

    desired_columns = [
        "category",
        "headline",
        "short_description",
        "authors",
        "date",
        "link",
    ]
    existing_columns = [c for c in desired_columns if c in working.columns]
    working = working[existing_columns].copy()

    for col in ["category", "headline", "short_description", "authors", "link"]:
        if col in working.columns:
            working[col] = working[col].map(clean_text)

    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.date

    before = len(working)
    working = working[
        working["headline"].ne("")
        & working["short_description"].ne("")
        & working["category"].ne("")
        & working["date"].notna()
    ].copy()
    LOGGER.info("Removed rows with missing required values: %d", before - len(working))

    working["content"] = (
        working["headline"].str.strip()
        + ". "
        + working["short_description"].str.strip()
    ).map(normalize_whitespace)

    before = len(working)
    working = working[working["content"].str.len() >= 20].copy()
    LOGGER.info("Removed rows with too-short content: %d", before - len(working))

    working["category"] = working["category"].str.upper()

    before = len(working)
    working = working.drop_duplicates(
        subset=["headline", "short_description", "date"],
        keep="first",
    ).copy()
    LOGGER.info("Removed duplicate rows: %d", before - len(working))

    working = working.reset_index(drop=True)
    working.insert(0, "id", np.arange(1, len(working) + 1, dtype=np.int64))

    LOGGER.info("Final cleaned row count: %d", len(working))
    return working


def validate_subset_sizes(df: pd.DataFrame, subset_sizes: list[int]) -> list[int]:
    max_rows = len(df)
    valid_sizes = [size for size in subset_sizes if size <= max_rows]
    invalid_sizes = sorted(set(subset_sizes) - set(valid_sizes))

    if invalid_sizes:
        LOGGER.warning(
            "Skipping subset sizes larger than cleaned dataset (%d rows): %s",
            max_rows,
            invalid_sizes,
        )

    if not valid_sizes:
        raise ValueError(
            f"No valid subset sizes remain. Cleaned dataset contains only {max_rows} rows."
        )

    return valid_sizes


def create_subsets(
    df: pd.DataFrame,
    subset_sizes: list[int],
    seed: int,
) -> dict[int, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(df))
    shuffled_df = df.iloc[shuffled_indices].reset_index(drop=True)

    subsets: dict[int, pd.DataFrame] = {}
    for size in sorted(subset_sizes):
        subsets[size] = shuffled_df.iloc[:size].copy().reset_index(drop=True)

    return subsets


def generate_embeddings(
    texts: Iterable[str],
    model_name: str,
    batch_size: int,
    device: str | None = None,
) -> np.ndarray:
    LOGGER.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name, device=device)

    texts_list = list(texts)
    LOGGER.info("Generating embeddings for %d texts", len(texts_list))

    embeddings = model.encode(
        texts_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {embeddings.shape}")

    LOGGER.info("Embedding matrix shape: %s", embeddings.shape)
    return embeddings


def validate_embeddings(df: pd.DataFrame, embeddings: np.ndarray) -> int:
    if len(df) != len(embeddings):
        raise ValueError(
            f"Row count ({len(df)}) and embedding count ({len(embeddings)}) do not match."
        )

    dims = embeddings.shape[1]
    if dims <= 0:
        raise ValueError("Embedding dimension must be positive.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    LOGGER.info("Embedding validation passed. Dimension = %d", dims)
    return dims


def save_subset_outputs(
    subset_df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
    subset_size: int,
) -> None:
    subset_dir = output_dir / f"subset_{subset_size}"
    subset_dir.mkdir(parents=True, exist_ok=True)

    dims = embeddings.shape[1]

    clean_path_csv = subset_dir / "news_clean.csv"
    subset_df.to_csv(clean_path_csv, index=False)

    pgvector_df = subset_df.copy()
    pgvector_df["embedding"] = [
        "[" + ",".join(f"{float(x):.8f}" for x in row) + "]" for row in embeddings
    ]
    pgvector_csv = subset_dir / "news_with_embeddings_pgvector.csv"
    pgvector_df.to_csv(pgvector_csv, index=False)

    metadata = {
        "subset_size": subset_size,
        "embedding_dimension": dims,
        "row_count": len(subset_df),
        "columns": list(subset_df.columns) + ["embedding"],
        "output_files": [
            clean_path_csv.name,
            pgvector_csv.name,
            "metadata.json",
        ],
    }
    metadata_path = subset_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info("Saved subset %d CSV outputs to %s", subset_size, subset_dir)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_dataset(args.input)
    cleaned_df = clean_and_structure(raw_df)

    if args.save_full_cleaned:
        full_cleaned_path = args.output_dir / "full_cleaned.csv"
        cleaned_df.to_csv(full_cleaned_path, index=False)
        LOGGER.info("Saved full cleaned dataset to %s", full_cleaned_path)

    valid_subset_sizes = validate_subset_sizes(cleaned_df, args.subset_sizes)
    subsets = create_subsets(cleaned_df, valid_subset_sizes, seed=args.seed)

    for subset_size, subset_df in subsets.items():
        LOGGER.info("Processing subset size: %d", subset_size)

        embeddings = generate_embeddings(
            texts=subset_df["content"].tolist(),
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
        )

        validate_embeddings(subset_df, embeddings)

        save_subset_outputs(
            subset_df=subset_df,
            embeddings=embeddings,
            output_dir=args.output_dir,
            subset_size=subset_size,
        )

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
