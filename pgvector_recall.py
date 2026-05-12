#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import psycopg
except ImportError:
    raise SystemExit('psycopg is required. Install with: pip install "psycopg[binary]"')


def load_database_config() -> str:
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            "config.json was not found. Create config.json in the same directory as this script."
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    db = config["database"]
    connection_parts = [
        f"host={db['host']}",
        f"port={db['port']}",
        f"dbname={db['dbname']}",
        f"user={db['user']}",
    ]
    password = db.get("password")
    if password:
        connection_parts.append(f"password={password}")
    return " ".join(connection_parts)


CONFIG = {
    "connection_string": load_database_config(),
    "tables": [
        {"name": "public.news_articles_10k", "size": 10_000},
        {"name": "public.news_articles_25k", "size": 25_000},
        {"name": "public.news_articles_50k", "size": 50_000},
    ],
    "id_col": "id",
    "embedding_col": "embedding",
    "k": 20,
    "query_count": 5,
    "query_vector_source": "sample_rows",
    "hnsw_m": 16,
    "hnsw_ef_construction": 64,
    "hnsw_ef_search": 40,
    "hnsw_tuned_m": 16,
    "hnsw_tuned_ef_construction": 128,
    "hnsw_tuned_ef_search": 100,
    "ivfflat_lists_override": {},
    "ivfflat_probes": 10,
    "output_dir": "pgvector_recall_results",
}


@dataclass(frozen=True)
class IndexMode:
    name: str
    expected_index_family: str
    hnsw: bool = False
    hnsw_tuned: bool = False
    ivfflat: bool = False


INDEX_MODES = [
    IndexMode("hnsw", "hnsw", hnsw=True),
    IndexMode("hnsw_tuned", "hnsw_tuned", hnsw_tuned=True),
    IndexMode("ivfflat", "ivfflat", ivfflat=True),
]


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def qtable(table: str) -> str:
    return ".".join(qident(part) for part in table.split("."))


def table_short(table: str) -> str:
    return table.split(".")[-1]


def idx_name(table: str, suffix: str) -> str:
    return f"idx_{table_short(table)}_{suffix}"


def managed_vector_index_names(table: str) -> list[str]:
    return [
        idx_name(table, "embedding_hnsw"),
        idx_name(table, "embedding_hnsw_tuned"),
        idx_name(table, "embedding_ivfflat"),
    ]


def ivfflat_lists_for_size(size: int) -> int:
    override = CONFIG.get("ivfflat_lists_override", {})
    if isinstance(override, dict):
        if size in override:
            return int(override[size])
        if str(size) in override:
            return int(override[str(size)])
    return max(10, int(math.sqrt(size)))


def deterministic_probe_vector(dim: int, seed: int) -> str:
    values = []
    for i in range(dim):
        value = math.sin((i + seed * 17) * 0.173) * 0.05
        values.append(f"{value:.6f}")
    return "[" + ",".join(values) + "]"


def drop_vector_indexes(conn: psycopg.Connection, table: str) -> None:
    with conn.cursor() as cur:
        for idx in managed_vector_index_names(table):
            cur.execute(f"DROP INDEX IF EXISTS {qident(idx)};")


def create_index_for_mode(
    conn: psycopg.Connection, table: str, size: int, mode: IndexMode
) -> None:
    qt = qtable(table)
    emb = qident(CONFIG["embedding_col"])

    drop_vector_indexes(conn, table)

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        if mode.hnsw:
            cur.execute(
                f"CREATE INDEX {qident(idx_name(table, 'embedding_hnsw'))} "
                f"ON {qt} USING hnsw ({emb} vector_cosine_ops) "
                f"WITH (m = {int(CONFIG['hnsw_m'])}, "
                f"ef_construction = {int(CONFIG['hnsw_ef_construction'])});"
            )

        if mode.hnsw_tuned:
            cur.execute(
                f"CREATE INDEX {qident(idx_name(table, 'embedding_hnsw_tuned'))} "
                f"ON {qt} USING hnsw ({emb} vector_cosine_ops) "
                f"WITH (m = {int(CONFIG['hnsw_tuned_m'])}, "
                f"ef_construction = {int(CONFIG['hnsw_tuned_ef_construction'])});"
            )

        if mode.ivfflat:
            lists = ivfflat_lists_for_size(size)
            cur.execute(
                f"CREATE INDEX {qident(idx_name(table, 'embedding_ivfflat'))} "
                f"ON {qt} USING ivfflat ({emb} vector_cosine_ops) "
                f"WITH (lists = {lists});"
            )

        cur.execute(f"ANALYZE {qt};")


def apply_runtime_settings(conn: psycopg.Connection, mode: IndexMode | None) -> None:
    with conn.cursor() as cur:
        cur.execute("RESET ALL;")
        if mode is None:
            cur.execute("SET enable_indexscan = off;")
            cur.execute("SET enable_bitmapscan = off;")
            cur.execute("SET enable_indexonlyscan = off;")
        elif mode.name == "hnsw":
            cur.execute(f"SET hnsw.ef_search = {int(CONFIG['hnsw_ef_search'])};")
        elif mode.name == "hnsw_tuned":
            cur.execute(f"SET hnsw.ef_search = {int(CONFIG['hnsw_tuned_ef_search'])};")
        elif mode.name == "ivfflat":
            cur.execute(f"SET ivfflat.probes = {int(CONFIG['ivfflat_probes'])};")


def get_query_vectors(
    conn: psycopg.Connection, table: str, size: int
) -> list[dict[str, Any]]:
    source = str(CONFIG["query_vector_source"])
    count = int(CONFIG["query_count"])
    emb = qident(CONFIG["embedding_col"])
    idc = qident(CONFIG["id_col"])
    qt = qtable(table)

    vectors: list[dict[str, Any]] = []

    if source == "sample_rows":
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH ordered AS (
                    SELECT {idc}, {emb}::text AS embedding_text,
                           row_number() OVER (ORDER BY {idc}) AS rn,
                           count(*) OVER () AS total_rows
                    FROM {qt}
                )
                SELECT {idc}, embedding_text
                FROM ordered
                WHERE rn IN (
                    SELECT GREATEST(1, ROUND((total_rows::numeric - 1) * s / GREATEST(1, %s - 1))::int + 1)
                    FROM generate_series(0, %s - 1) AS s
                    LIMIT 1
                )
                LIMIT %s;
                """,
                (count, count, count),
            )
            rows = cur.fetchall()

        if len(rows) < count:
            rows = []
            with conn.cursor() as cur:
                for i in range(count):
                    offset = int((max(1, size) - 1) * i / max(1, count - 1))
                    cur.execute(
                        f"""
                        SELECT {idc}, {emb}::text
                        FROM {qt}
                        ORDER BY {idc}
                        OFFSET %s
                        LIMIT 1;
                        """,
                        (offset,),
                    )
                    row = cur.fetchone()
                    if row:
                        rows.append(row)

        for idx, row in enumerate(rows):
            vectors.append(
                {
                    "query_no": idx + 1,
                    "source_row_id": row[0],
                    "query_vector": row[1],
                }
            )

    elif source == "deterministic":
        dim = 384
        for i in range(count):
            vectors.append(
                {
                    "query_no": i + 1,
                    "source_row_id": None,
                    "query_vector": deterministic_probe_vector(dim, i + 1),
                }
            )
    else:
        raise ValueError(f"Unknown query_vector_source: {source}")

    if not vectors:
        raise RuntimeError(f"No query vectors found for {table}")

    return vectors


def walk_plan(node: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [node]
    for child in node.get("Plans", []) or []:
        nodes.extend(walk_plan(child))
    return nodes


def extract_plan_info(plan_json: Any) -> dict[str, Any]:
    root = plan_json[0]
    plan = root["Plan"]
    nodes = walk_plan(plan)

    index_names = sorted(
        {str(n.get("Index Name")) for n in nodes if n.get("Index Name")}
    )
    node_types = sorted({str(n.get("Node Type")) for n in nodes if n.get("Node Type")})

    idx_lower = " | ".join(index_names).lower()

    hnsw_tuned_used = "embedding_hnsw_tuned" in idx_lower
    hnsw_used = "embedding_hnsw" in idx_lower and not hnsw_tuned_used
    ivfflat_used = "embedding_ivfflat" in idx_lower

    shared_hit_blocks = sum(int(n.get("Shared Hit Blocks", 0) or 0) for n in nodes)
    shared_read_blocks = sum(int(n.get("Shared Read Blocks", 0) or 0) for n in nodes)

    return {
        "planning_time_ms": float(root.get("Planning Time", 0.0) or 0.0),
        "execution_time_ms": float(root.get("Execution Time", 0.0) or 0.0),
        "node_types": node_types,
        "index_names": index_names,
        "hnsw_used": hnsw_used,
        "hnsw_tuned_used": hnsw_tuned_used,
        "ivfflat_used": ivfflat_used,
        "shared_hit_blocks": shared_hit_blocks,
        "shared_read_blocks": shared_read_blocks,
    }


def run_topk_with_plan(
    conn: psycopg.Connection,
    table: str,
    query_vector: str,
    mode: IndexMode | None,
) -> tuple[list[int], dict[str, Any], float]:
    qt = qtable(table)
    idc = qident(CONFIG["id_col"])
    emb = qident(CONFIG["embedding_col"])
    k = int(CONFIG["k"])

    apply_runtime_settings(conn, mode)

    sql = f"""
        EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
        SELECT {idc}
        FROM {qt}
        ORDER BY {emb} <=> %s::vector
        LIMIT {k};
    """

    start = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, (query_vector,))
        plan_json = cur.fetchone()[0]
    wall_ms = (time.perf_counter() - start) * 1000.0
    info = extract_plan_info(plan_json)

    apply_runtime_settings(conn, mode)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {idc}
            FROM {qt}
            ORDER BY {emb} <=> %s::vector
            LIMIT {k};
            """,
            (query_vector,),
        )
        ids = [int(r[0]) for r in cur.fetchall()]

    return ids, info, wall_ms


def validate_index_usage(mode: IndexMode, info: dict[str, Any]) -> list[str]:
    warnings = []

    if mode.expected_index_family == "hnsw" and not info["hnsw_used"]:
        warnings.append("EXPECTED_HNSW_NOT_USED")
    if mode.expected_index_family == "hnsw_tuned" and not info["hnsw_tuned_used"]:
        warnings.append("EXPECTED_HNSW_TUNED_NOT_USED")
    if mode.expected_index_family == "ivfflat" and not info["ivfflat_used"]:
        warnings.append("EXPECTED_IVFFLAT_NOT_USED")

    if info["shared_read_blocks"] > 0:
        warnings.append("SHARED_READ_BLOCKS_PRESENT_CACHE_MAY_BE_COLD")

    return warnings


def recall_at_k(exact_ids: list[int], approx_ids: list[int], k: int) -> float:
    if not exact_ids:
        return 0.0
    exact_set = set(exact_ids[:k])
    approx_set = set(approx_ids[:k])
    return len(exact_set & approx_set) / min(k, len(exact_set))


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}

    for row in raw_rows:
        key = (row["table"], int(row["dataset_size"]), row["index_mode"])
        groups.setdefault(key, []).append(row)

    summary = []

    for (table, size, index_mode), rows in sorted(groups.items()):
        recalls = [float(r["recall_at_20"]) for r in rows]
        times = [float(r["execution_time_ms"]) for r in rows]
        planning = [float(r["planning_time_ms"]) for r in rows]
        overlap = [int(r["overlap_count"]) for r in rows]
        warnings = sorted({w for r in rows for w in r["warnings"]})
        index_names = sorted({idx for r in rows for idx in r["index_names"]})

        summary.append(
            {
                "table": table,
                "dataset_size": size,
                "index_mode": index_mode,
                "query_count": len(rows),
                "k": int(CONFIG["k"]),
                "avg_recall_at_20": round(statistics.mean(recalls), 3),
                "median_recall_at_20": round(statistics.median(recalls), 3),
                "min_recall_at_20": round(min(recalls), 3),
                "max_recall_at_20": round(max(recalls), 3),
                "avg_overlap_count": round(statistics.mean(overlap), 2),
                "median_execution_time_ms": round(statistics.median(times), 3),
                "avg_execution_time_ms": round(statistics.mean(times), 3),
                "avg_planning_time_ms": round(statistics.mean(planning), 3),
                "index_names": " | ".join(index_names),
                "warnings": " | ".join(warnings),
            }
        )

    return summary


def run() -> None:
    out_dir = Path(str(CONFIG["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "recall_raw_results.jsonl"
    summary_path = out_dir / "recall_summary.csv"
    warnings_path = out_dir / "recall_plan_warnings.csv"

    for p in [raw_path, summary_path, warnings_path]:
        if p.exists():
            p.unlink()

    raw_rows: list[dict[str, Any]] = []

    with psycopg.connect(str(CONFIG["connection_string"]), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        for table_cfg in CONFIG["tables"]:
            table = str(table_cfg["name"])
            size = int(table_cfg["size"])
            k = int(CONFIG["k"])

            query_vectors = get_query_vectors(conn, table, size)

            drop_vector_indexes(conn, table)

            exact_by_query: dict[int, list[int]] = {}

            for q in query_vectors:
                ids, info, wall_ms = run_topk_with_plan(
                    conn,
                    table,
                    q["query_vector"],
                    mode=None,
                )
                exact_by_query[int(q["query_no"])] = ids

            for mode in INDEX_MODES:
                create_index_for_mode(conn, table, size, mode)

                for q in query_vectors:
                    query_no = int(q["query_no"])
                    exact_ids = exact_by_query[query_no]

                    approx_ids, info, wall_ms = run_topk_with_plan(
                        conn,
                        table,
                        q["query_vector"],
                        mode=mode,
                    )

                    overlap_ids = sorted(set(exact_ids[:k]) & set(approx_ids[:k]))
                    recall = recall_at_k(exact_ids, approx_ids, k)
                    warnings = validate_index_usage(mode, info)

                    row = {
                        "table": table,
                        "dataset_size": size,
                        "index_mode": mode.name,
                        "query_no": query_no,
                        "source_row_id": q["source_row_id"],
                        "k": k,
                        "exact_ids": exact_ids,
                        "approx_ids": approx_ids,
                        "overlap_ids": overlap_ids,
                        "overlap_count": len(overlap_ids),
                        "recall_at_20": recall,
                        "planning_time_ms": info["planning_time_ms"],
                        "execution_time_ms": info["execution_time_ms"],
                        "wall_time_ms": wall_ms,
                        "node_types": info["node_types"],
                        "index_names": info["index_names"],
                        "hnsw_used": info["hnsw_used"],
                        "hnsw_tuned_used": info["hnsw_tuned_used"],
                        "ivfflat_used": info["ivfflat_used"],
                        "shared_hit_blocks": info["shared_hit_blocks"],
                        "shared_read_blocks": info["shared_read_blocks"],
                        "warnings": warnings,
                    }

                    raw_rows.append(row)
                    write_jsonl(raw_path, row)

    summary_rows = summarize(raw_rows)
    write_csv(summary_path, summary_rows)
    write_csv(warnings_path, [r for r in summary_rows if r.get("warnings")])
    print("\nDone.")


if __name__ == "__main__":
    run()
