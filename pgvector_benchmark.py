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
    "category_col": "category",
    "date_col": "published_date",
    "embedding_col": "embedding",
    "embedding_dim": 384,
    "category": "POLITICS",
    "date_from": "2016-01-01",
    "date_to": "2017-01-01",
    "limit": 20,
    "warmup_runs": 3,
    "measured_runs": 10,
    "hnsw_m": 16,
    "hnsw_ef_construction": 64,
    "hnsw_ef_search": 40,
    "hnsw_tuned_m": 16,
    "hnsw_tuned_ef_construction": 128,
    "hnsw_tuned_ef_search": 100,
    "ivfflat_lists_override": {},
    "ivfflat_probes": 10,
    "run_vector_first_experiment": True,
    "vector_first_candidate_pools_by_size": {
        10000: [500, 1000, 2000, 5000],
        25000: [1000, 2000, 5000, 10000],
        50000: [2000, 5000, 10000, 20000],
    },
    "vector_first_date_from": "2012-01-01",
    "vector_first_date_to": "2018-12-31",
    "output_dir": "pgvector_benchmark_results",
}


@dataclass(frozen=True)
class Mode:
    name: str
    description: str
    create_btree: bool = False
    create_hnsw: bool = False
    create_hnsw_tuned: bool = False
    create_ivfflat: bool = False


@dataclass(frozen=True)
class Case:
    case_id: str
    description: str
    sql: str
    expected_index_family: str | None
    expected_btree: bool | None
    required_full_limit: bool = True
    candidate_pool: int | None = None


MAIN_MODES = [
    Mode(
        name="no_indexes",
        description="No managed indexes. Used for exact vector and relational baselines.",
    ),
    Mode(
        name="btree_only",
        description="Only B-tree indexes. Used for relational filters and relational-first hybrid search.",
        create_btree=True,
    ),
    Mode(
        name="hnsw_only",
        description="Only default HNSW vector index. Used for pure HNSW vector search.",
        create_hnsw=True,
    ),
    Mode(
        name="hnsw_tuned_only",
        description="Only tuned HNSW vector index. Used for pure tuned HNSW vector search.",
        create_hnsw_tuned=True,
    ),
    Mode(
        name="ivfflat_only",
        description="Only IVFFlat vector index. Used for pure IVFFlat vector search.",
        create_ivfflat=True,
    ),
]

VECTOR_FIRST_MODES = [
    Mode(
        name="hnsw_only",
        description="Only default HNSW vector index. Used for vector-first hybrid experiment.",
        create_hnsw=True,
    ),
    Mode(
        name="hnsw_tuned_only",
        description="Only tuned HNSW vector index. Used for vector-first hybrid experiment.",
        create_hnsw_tuned=True,
    ),
    Mode(
        name="ivfflat_only",
        description="Only IVFFlat vector index. Used for vector-first hybrid experiment.",
        create_ivfflat=True,
    ),
]


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def qtable(table: str) -> str:
    return ".".join(qident(part) for part in table.split("."))


def table_short(table: str) -> str:
    return table.split(".")[-1]


def idx_name(table: str, suffix: str) -> str:
    return f"idx_{table_short(table)}_{suffix}"


def deterministic_probe_vector(dim: int) -> str:
    values = []
    for i in range(dim):
        value = math.sin(i * 0.173) * 0.05
        values.append(f"{value:.6f}")
    return "[" + ",".join(values) + "]"


def vector_distance_expr(alias: str | None = None) -> str:
    emb = qident(CONFIG["embedding_col"])
    if alias:
        emb = f"{alias}.{emb}"
    probe = deterministic_probe_vector(int(CONFIG["embedding_dim"]))
    return f"{emb} <=> '{probe}'::vector"


def category_literal() -> str:
    return str(CONFIG["category"]).replace("'", "''")


def base_sql_fragments(
    table: str, *, date_from: str | None = None, date_to: str | None = None
) -> dict[str, str]:
    qt = qtable(table)
    idc = qident(CONFIG["id_col"])
    cat = qident(CONFIG["category_col"])
    date = qident(CONFIG["date_col"])
    category = category_literal()
    limit = int(CONFIG["limit"])
    dist = vector_distance_expr()

    df = date_from or str(CONFIG["date_from"])
    dt = date_to or str(CONFIG["date_to"])

    return {
        "qt": qt,
        "idc": idc,
        "cat": cat,
        "date": date,
        "category": category,
        "limit": str(limit),
        "dist": dist,
        "date_from": df,
        "date_to": dt,
    }


def build_main_cases_for_mode(mode: Mode, table: str) -> list[Case]:
    f = base_sql_fragments(table)
    qt = f["qt"]
    idc = f["idc"]
    cat = f["cat"]
    date = f["date"]
    category = f["category"]
    limit = f["limit"]
    dist = f["dist"]
    date_from = f["date_from"]
    date_to = f["date_to"]

    rel_category_sql = f"""
        SELECT {idc}, {cat}, {date}
        FROM {qt}
        WHERE {cat} = '{category}'
        LIMIT {limit}
    """

    rel_date_sql = f"""
        SELECT {idc}, {cat}, {date}
        FROM {qt}
        WHERE {date} >= DATE '{date_from}'
          AND {date} < DATE '{date_to}'
        ORDER BY {date}
        LIMIT {limit}
    """

    rel_category_date_sql = f"""
        SELECT {idc}, {cat}, {date}
        FROM {qt}
        WHERE {cat} = '{category}'
          AND {date} >= DATE '{date_from}'
          AND {date} < DATE '{date_to}'
        ORDER BY {date}
        LIMIT {limit}
    """

    exact_vector_sql = f"""
        SELECT {idc}, {cat}, {date}, {dist} AS distance
        FROM {qt}
        ORDER BY {dist}
        LIMIT {limit}
    """

    rel_first_hybrid_sql = f"""
        SELECT {idc}, {cat}, {date}, {dist} AS distance
        FROM {qt}
        WHERE {cat} = '{category}'
          AND {date} >= DATE '{date_from}'
          AND {date} < DATE '{date_to}'
        ORDER BY {dist}
        LIMIT {limit}
    """

    if mode.name == "no_indexes":
        return [
            Case(
                "rel_category_baseline_no_index",
                "Relational category filter without managed indexes.",
                rel_category_sql,
                expected_index_family=None,
                expected_btree=False,
            ),
            Case(
                "rel_date_baseline_no_index",
                "Relational date filter without managed indexes.",
                rel_date_sql,
                expected_index_family=None,
                expected_btree=False,
            ),
            Case(
                "rel_category_date_baseline_no_index",
                "Relational category+date filter without managed indexes.",
                rel_category_date_sql,
                expected_index_family=None,
                expected_btree=False,
            ),
            Case(
                "vector_exact_no_index",
                "Exact vector search without vector index.",
                exact_vector_sql,
                expected_index_family=None,
                expected_btree=False,
            ),
        ]

    if mode.name == "btree_only":
        return [
            Case(
                "rel_category_btree",
                "Relational category filter with B-tree index available. Planner may still choose Seq Scan.",
                rel_category_sql,
                expected_index_family=None,
                expected_btree=None,
            ),
            Case(
                "rel_date_btree",
                "Relational date filter with B-tree index.",
                rel_date_sql,
                expected_index_family=None,
                expected_btree=True,
            ),
            Case(
                "rel_category_date_btree",
                "Relational category+date filter with composite B-tree index.",
                rel_category_date_sql,
                expected_index_family=None,
                expected_btree=True,
            ),
            Case(
                "hybrid_rel_first_category_date_btree",
                "Relational-first hybrid: B-tree pre-filter, then exact vector ranking.",
                rel_first_hybrid_sql,
                expected_index_family=None,
                expected_btree=True,
            ),
        ]

    if mode.name == "hnsw_only":
        return [
            Case(
                "vector_hnsw",
                "Pure vector search using isolated HNSW index.",
                exact_vector_sql,
                expected_index_family="hnsw",
                expected_btree=False,
            ),
        ]

    if mode.name == "hnsw_tuned_only":
        return [
            Case(
                "vector_hnsw_tuned",
                "Pure vector search using isolated tuned HNSW index.",
                exact_vector_sql,
                expected_index_family="hnsw_tuned",
                expected_btree=False,
            ),
        ]

    if mode.name == "ivfflat_only":
        return [
            Case(
                "vector_ivfflat",
                "Pure vector search using isolated IVFFlat index.",
                exact_vector_sql,
                expected_index_family="ivfflat",
                expected_btree=False,
            ),
        ]

    raise ValueError(f"Unknown mode: {mode.name}")


def build_vector_first_cases_for_mode(
    mode: Mode, table: str, candidate_pool: int
) -> list[Case]:
    f = base_sql_fragments(
        table,
        date_from=str(CONFIG["vector_first_date_from"]),
        date_to=str(CONFIG["vector_first_date_to"]),
    )

    qt = f["qt"]
    idc = f["idc"]
    cat = f["cat"]
    date = f["date"]
    category = f["category"]
    limit = f["limit"]
    dist = f["dist"]
    date_from = f["date_from"]
    date_to = f["date_to"]

    family = {
        "hnsw_only": "hnsw",
        "hnsw_tuned_only": "hnsw_tuned",
        "ivfflat_only": "ivfflat",
    }[mode.name]

    suffix = {
        "hnsw_only": "hnsw",
        "hnsw_tuned_only": "hnsw_tuned",
        "ivfflat_only": "ivfflat",
    }[mode.name]

    vector_first_category_sql = f"""
        WITH candidates AS MATERIALIZED (
            SELECT {idc}, {cat}, {date}, {dist} AS distance
            FROM {qt}
            ORDER BY {dist}
            LIMIT {candidate_pool}
        )
        SELECT *
        FROM candidates
        WHERE {cat} = '{category}'
        ORDER BY distance
        LIMIT {limit}
    """

    vector_first_category_date_sql = f"""
        WITH candidates AS MATERIALIZED (
            SELECT {idc}, {cat}, {date}, {dist} AS distance
            FROM {qt}
            ORDER BY {dist}
            LIMIT {candidate_pool}
        )
        SELECT *
        FROM candidates
        WHERE {cat} = '{category}'
          AND {date} >= DATE '{date_from}'
          AND {date} < DATE '{date_to}'
        ORDER BY distance
        LIMIT {limit}
    """

    return [
        Case(
            f"hybrid_vector_first_category_{suffix}_pool_{candidate_pool}",
            f"Vector-first hybrid using {suffix}; candidate_pool={candidate_pool}; category filter.",
            vector_first_category_sql,
            expected_index_family=family,
            expected_btree=False,
            required_full_limit=False,
            candidate_pool=candidate_pool,
        ),
        Case(
            f"hybrid_vector_first_category_date_{suffix}_pool_{candidate_pool}",
            f"Vector-first hybrid using {suffix}; candidate_pool={candidate_pool}; category+date filter.",
            vector_first_category_date_sql,
            expected_index_family=family,
            expected_btree=False,
            required_full_limit=False,
            candidate_pool=candidate_pool,
        ),
    ]


def managed_index_names(table: str) -> list[str]:
    return [
        idx_name(table, "category"),
        idx_name(table, "published_date"),
        idx_name(table, "category_date"),
        idx_name(table, "embedding_hnsw"),
        idx_name(table, "embedding_hnsw_tuned"),
        idx_name(table, "embedding_ivfflat"),
    ]


def drop_managed_indexes(conn: psycopg.Connection, table: str) -> None:
    with conn.cursor() as cur:
        for idx in managed_index_names(table):
            cur.execute(f"DROP INDEX IF EXISTS {qident(idx)};")


def ivfflat_lists_for_size(size: int) -> int:
    override = CONFIG.get("ivfflat_lists_override", {})
    if isinstance(override, dict):
        if size in override:
            return int(override[size])
        if str(size) in override:
            return int(override[str(size)])
    return max(10, int(math.sqrt(size)))


def prepare_mode_indexes(
    conn: psycopg.Connection, mode: Mode, table: str, size: int
) -> None:
    qt = qtable(table)
    cat = qident(CONFIG["category_col"])
    date = qident(CONFIG["date_col"])
    emb = qident(CONFIG["embedding_col"])

    drop_managed_indexes(conn, table)

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        if mode.create_btree:
            cur.execute(
                f"CREATE INDEX {qident(idx_name(table, 'category'))} ON {qt} ({cat});"
            )
            cur.execute(
                f"CREATE INDEX {qident(idx_name(table, 'published_date'))} ON {qt} ({date});"
            )
            cur.execute(
                f"CREATE INDEX {qident(idx_name(table, 'category_date'))} ON {qt} ({cat}, {date});"
            )

        if mode.create_hnsw:
            cur.execute(
                f"CREATE INDEX {qident(idx_name(table, 'embedding_hnsw'))} "
                f"ON {qt} USING hnsw ({emb} vector_cosine_ops) "
                f"WITH (m = {int(CONFIG['hnsw_m'])}, "
                f"ef_construction = {int(CONFIG['hnsw_ef_construction'])});"
            )

        if mode.create_hnsw_tuned:
            cur.execute(
                f"CREATE INDEX {qident(idx_name(table, 'embedding_hnsw_tuned'))} "
                f"ON {qt} USING hnsw ({emb} vector_cosine_ops) "
                f"WITH (m = {int(CONFIG['hnsw_tuned_m'])}, "
                f"ef_construction = {int(CONFIG['hnsw_tuned_ef_construction'])});"
            )

        if mode.create_ivfflat:
            lists = ivfflat_lists_for_size(size)
            cur.execute(
                f"CREATE INDEX {qident(idx_name(table, 'embedding_ivfflat'))} "
                f"ON {qt} USING ivfflat ({emb} vector_cosine_ops) "
                f"WITH (lists = {lists});"
            )

        cur.execute(f"ANALYZE {qt};")


def apply_runtime_settings(conn: psycopg.Connection, mode: Mode) -> None:
    with conn.cursor() as cur:
        cur.execute("RESET ALL;")
        if mode.name == "hnsw_only":
            cur.execute(f"SET hnsw.ef_search = {int(CONFIG['hnsw_ef_search'])};")
        elif mode.name == "hnsw_tuned_only":
            cur.execute(f"SET hnsw.ef_search = {int(CONFIG['hnsw_tuned_ef_search'])};")
        elif mode.name == "ivfflat_only":
            cur.execute(f"SET ivfflat.probes = {int(CONFIG['ivfflat_probes'])};")


def walk_plan(node: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [node]
    for child in node.get("Plans", []) or []:
        nodes.extend(walk_plan(child))
    return nodes


def extract_plan_info(plan_json: Any) -> dict[str, Any]:
    root = plan_json[0]
    plan = root["Plan"]
    nodes = walk_plan(plan)

    node_types = sorted({str(n.get("Node Type")) for n in nodes if n.get("Node Type")})
    index_names = sorted(
        {str(n.get("Index Name")) for n in nodes if n.get("Index Name")}
    )

    shared_hit_blocks = sum(int(n.get("Shared Hit Blocks", 0) or 0) for n in nodes)
    shared_read_blocks = sum(int(n.get("Shared Read Blocks", 0) or 0) for n in nodes)
    root_actual_rows = int(plan.get("Actual Rows", 0) or 0)

    idx_lower = " | ".join(index_names).lower()

    hnsw_tuned_used = "embedding_hnsw_tuned" in idx_lower
    hnsw_used = "embedding_hnsw" in idx_lower and not hnsw_tuned_used
    ivfflat_used = "embedding_ivfflat" in idx_lower
    vector_index_used = hnsw_used or hnsw_tuned_used or ivfflat_used

    btree_used = (
        "category" in idx_lower
        or "published_date" in idx_lower
        or "category_date" in idx_lower
    ) and not vector_index_used

    node_details = []
    for n in nodes:
        node_details.append(
            {
                "node_type": n.get("Node Type"),
                "index_name": n.get("Index Name"),
                "actual_rows": n.get("Actual Rows"),
                "plan_rows": n.get("Plan Rows"),
                "filter": n.get("Filter"),
                "index_cond": n.get("Index Cond"),
                "order_by": n.get("Order By"),
                "sort_key": n.get("Sort Key"),
            }
        )

    return {
        "planning_time_ms": float(root.get("Planning Time", 0.0) or 0.0),
        "execution_time_ms": float(root.get("Execution Time", 0.0) or 0.0),
        "root_actual_rows": root_actual_rows,
        "node_types": node_types,
        "index_names": index_names,
        "shared_hit_blocks": shared_hit_blocks,
        "shared_read_blocks": shared_read_blocks,
        "hnsw_used": hnsw_used,
        "hnsw_tuned_used": hnsw_tuned_used,
        "ivfflat_used": ivfflat_used,
        "vector_index_used": vector_index_used,
        "btree_used": btree_used,
        "node_details": node_details,
    }


def run_explain_analyze(
    conn: psycopg.Connection, sql: str
) -> tuple[dict[str, Any], float, Any]:
    full_sql = "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql
    start = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(full_sql)
        plan_json = cur.fetchone()[0]
    wall_time_ms = (time.perf_counter() - start) * 1000.0
    info = extract_plan_info(plan_json)
    return info, wall_time_ms, plan_json


def validate_plan(
    mode: Mode, case: Case, info: dict[str, Any], limit: int
) -> list[str]:
    warnings = []

    if case.expected_index_family == "hnsw" and not info["hnsw_used"]:
        warnings.append("EXPECTED_HNSW_NOT_USED")

    if case.expected_index_family == "hnsw_tuned" and not info["hnsw_tuned_used"]:
        warnings.append("EXPECTED_HNSW_TUNED_NOT_USED")

    if case.expected_index_family == "ivfflat" and not info["ivfflat_used"]:
        warnings.append("EXPECTED_IVFFLAT_NOT_USED")

    if case.expected_index_family is None and info["vector_index_used"]:
        warnings.append("UNEXPECTED_VECTOR_INDEX_USED")

    if case.expected_btree is True and not info["btree_used"]:
        warnings.append("EXPECTED_BTREE_NOT_USED")

    if case.expected_btree is False and info["btree_used"]:
        warnings.append("UNEXPECTED_BTREE_USED")

    if mode.name == "no_indexes" and info["index_names"]:
        warnings.append("INDEX_USED_IN_NO_INDEX_MODE")

    if mode.name == "btree_only" and info["vector_index_used"]:
        warnings.append("VECTOR_INDEX_USED_IN_BTREE_ONLY_MODE")

    if (
        mode.name in {"hnsw_only", "hnsw_tuned_only", "ivfflat_only"}
        and info["btree_used"]
    ):
        warnings.append("BTREE_INDEX_USED_IN_VECTOR_ONLY_MODE")

    if info["shared_read_blocks"] > 0:
        warnings.append("SHARED_READ_BLOCKS_PRESENT_CACHE_MAY_BE_COLD")

    if case.required_full_limit and info["root_actual_rows"] < limit:
        warnings.append("RETURNED_FEWER_THAN_LIMIT")

    if (not case.required_full_limit) and info["root_actual_rows"] < limit:
        warnings.append("VECTOR_FIRST_RETURNED_FEWER_THAN_LIMIT")

    if info["execution_time_ms"] < 0.1:
        warnings.append("VERY_SMALL_TIME_MEASUREMENT_NOISE_POSSIBLE")

    return warnings


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str, int | None], list[dict[str, Any]]] = {}

    for row in raw_rows:
        key = (
            row["table"],
            int(row["dataset_size"]),
            row["mode"],
            row["case_id"],
            row.get("candidate_pool"),
        )
        groups.setdefault(key, []).append(row)

    summary_rows = []

    for (table, size, mode, case_id, candidate_pool), rows in sorted(groups.items()):
        exec_times = [float(r["execution_time_ms"]) for r in rows]
        planning_times = [float(r["planning_time_ms"]) for r in rows]
        wall_times = [float(r["wall_time_ms"]) for r in rows]
        result_rows = [int(r["root_actual_rows"]) for r in rows]
        warnings = sorted({w for r in rows for w in r["warnings"]})
        node_types = sorted({x for r in rows for x in r["node_types"]})
        index_names = sorted({x for r in rows for x in r["index_names"]})
        first = rows[0]

        summary_rows.append(
            {
                "table": table,
                "dataset_size": size,
                "mode": mode,
                "case_id": case_id,
                "candidate_pool": "" if candidate_pool is None else candidate_pool,
                "description": first["description"],
                "runs": len(rows),
                "median_execution_time_ms": round(statistics.median(exec_times), 3),
                "avg_execution_time_ms": round(statistics.mean(exec_times), 3),
                "min_execution_time_ms": round(min(exec_times), 3),
                "max_execution_time_ms": round(max(exec_times), 3),
                "avg_planning_time_ms": round(statistics.mean(planning_times), 3),
                "avg_wall_time_ms": round(statistics.mean(wall_times), 3),
                "median_result_rows": statistics.median(result_rows),
                "min_result_rows": min(result_rows),
                "max_result_rows": max(result_rows),
                "avg_shared_hit_blocks": round(
                    statistics.mean(float(r["shared_hit_blocks"]) for r in rows), 1
                ),
                "avg_shared_read_blocks": round(
                    statistics.mean(float(r["shared_read_blocks"]) for r in rows), 1
                ),
                "hnsw_used": str(any(r["hnsw_used"] for r in rows)),
                "hnsw_tuned_used": str(any(r["hnsw_tuned_used"] for r in rows)),
                "ivfflat_used": str(any(r["ivfflat_used"] for r in rows)),
                "vector_index_used": str(any(r["vector_index_used"] for r in rows)),
                "btree_used": str(any(r["btree_used"] for r in rows)),
                "node_types": " | ".join(node_types),
                "index_names": " | ".join(index_names),
                "warnings": " | ".join(warnings),
            }
        )

    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_case_group(
    conn: psycopg.Connection,
    *,
    table: str,
    size: int,
    mode: Mode,
    cases: list[Case],
    raw_path: Path,
) -> list[dict[str, Any]]:
    rows = []
    limit = int(CONFIG["limit"])

    for case in cases:

        for _ in range(int(CONFIG["warmup_runs"])):
            apply_runtime_settings(conn, mode)
            run_explain_analyze(conn, case.sql)

        for run_no in range(1, int(CONFIG["measured_runs"]) + 1):
            apply_runtime_settings(conn, mode)
            info, wall_ms, _ = run_explain_analyze(conn, case.sql)
            warnings = validate_plan(mode, case, info, limit)

            row = {
                "table": table,
                "dataset_size": size,
                "mode": mode.name,
                "mode_description": mode.description,
                "case_id": case.case_id,
                "candidate_pool": case.candidate_pool,
                "description": case.description,
                "run_no": run_no,
                "planning_time_ms": info["planning_time_ms"],
                "execution_time_ms": info["execution_time_ms"],
                "wall_time_ms": wall_ms,
                "root_actual_rows": info["root_actual_rows"],
                "node_types": info["node_types"],
                "index_names": info["index_names"],
                "shared_hit_blocks": info["shared_hit_blocks"],
                "shared_read_blocks": info["shared_read_blocks"],
                "hnsw_used": info["hnsw_used"],
                "hnsw_tuned_used": info["hnsw_tuned_used"],
                "ivfflat_used": info["ivfflat_used"],
                "vector_index_used": info["vector_index_used"],
                "btree_used": info["btree_used"],
                "warnings": warnings,
                "node_details": info["node_details"],
            }

            rows.append(row)
            write_jsonl(raw_path, row)

    return rows


def candidate_pools_for_size(size: int) -> list[int]:
    mapping = CONFIG["vector_first_candidate_pools_by_size"]
    if size in mapping:
        return list(mapping[size])
    if str(size) in mapping:
        return list(mapping[str(size)])

    return sorted({max(100, int(size * ratio)) for ratio in (0.05, 0.10, 0.20)})


def run_main_benchmark(conn: psycopg.Connection, out_dir: Path) -> list[dict[str, Any]]:
    raw_path = out_dir / "main_raw_results.jsonl"
    all_rows = []

    for table_cfg in CONFIG["tables"]:
        table = str(table_cfg["name"])
        size = int(table_cfg["size"])

        for mode in MAIN_MODES:
            prepare_mode_indexes(conn, mode, table, size)
            cases = build_main_cases_for_mode(mode, table)
            all_rows.extend(
                run_case_group(
                    conn,
                    table=table,
                    size=size,
                    mode=mode,
                    cases=cases,
                    raw_path=raw_path,
                )
            )

    return all_rows


def run_vector_first_experiment(
    conn: psycopg.Connection, out_dir: Path
) -> list[dict[str, Any]]:
    raw_path = out_dir / "vector_first_raw_results.jsonl"
    all_rows = []

    for table_cfg in CONFIG["tables"]:
        table = str(table_cfg["name"])
        size = int(table_cfg["size"])
        pools = candidate_pools_for_size(size)

        for mode in VECTOR_FIRST_MODES:
            prepare_mode_indexes(conn, mode, table, size)

            for pool in pools:
                cases = build_vector_first_cases_for_mode(mode, table, pool)
                all_rows.extend(
                    run_case_group(
                        conn,
                        table=table,
                        size=size,
                        mode=mode,
                        cases=cases,
                        raw_path=raw_path,
                    )
                )

    return all_rows


def run() -> None:
    out_dir = Path(str(CONFIG["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        "main_raw_results.jsonl",
        "main_summary.csv",
        "main_plan_warnings.csv",
        "vector_first_raw_results.jsonl",
        "vector_first_summary.csv",
        "vector_first_plan_warnings.csv",
    ]

    for name in output_files:
        path = out_dir / name
        if path.exists():
            path.unlink()

    with psycopg.connect(str(CONFIG["connection_string"]), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        main_rows = run_main_benchmark(conn, out_dir)
        main_summary = summarize(main_rows)
        write_csv(out_dir / "main_summary.csv", main_summary)
        write_csv(
            out_dir / "main_plan_warnings.csv",
            [r for r in main_summary if r.get("warnings")],
        )

        if bool(CONFIG["run_vector_first_experiment"]):
            vector_first_rows = run_vector_first_experiment(conn, out_dir)
            vector_first_summary = summarize(vector_first_rows)
            write_csv(out_dir / "vector_first_summary.csv", vector_first_summary)
            write_csv(
                out_dir / "vector_first_plan_warnings.csv",
                [r for r in vector_first_summary if r.get("warnings")],
            )

    print("\nDone.")


if __name__ == "__main__":
    run()
