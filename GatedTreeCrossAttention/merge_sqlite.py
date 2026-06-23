#!/usr/bin/env python3
"""
Merge two SQLite parse-tree caches (produced by generate_tree_gtca.py) into one.

Schema (both source and destination):
    CREATE TABLE parsed_cache (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )

Collision policy: INSERT OR IGNORE — the first database's entry wins when two
examples happen to produce identical token sequences (identical hash). In
practice CLOTH and MMLU prompts are disjoint, so collisions are extremely rare.

Usage:
    python merge_sqlite.py \
        --db_a  cache/cloth_qwen.sqlite \
        --db_b  cache/mmlu_qwen.sqlite \
        --out   cache/cloth_mmlu_qwen.sqlite

The source databases are never modified.
"""

import argparse
import os
import sqlite3


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parsed_cache (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()


def count_rows(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM parsed_cache").fetchone()[0]


def merge_into(src_path: str, dst_conn: sqlite3.Connection) -> int:
    """Copy all rows from src_path into dst_conn; return number inserted."""
    dst_conn.execute(f"ATTACH DATABASE '{src_path}' AS src")
    dst_conn.execute(
        "INSERT OR IGNORE INTO main.parsed_cache (key, value) "
        "SELECT key, value FROM src.parsed_cache"
    )
    dst_conn.commit()
    dst_conn.execute("DETACH DATABASE src")
    return count_rows(dst_conn)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge two GTCA SQLite caches.")
    ap.add_argument("--db_a", required=True, help="First source SQLite file")
    ap.add_argument("--db_b", required=True, help="Second source SQLite file")
    ap.add_argument("--out",  required=True, help="Output (merged) SQLite file")
    args = ap.parse_args()

    if os.path.exists(args.out):
        ans = input(f"Output file {args.out!r} already exists. Overwrite? [y/N] ")
        if ans.strip().lower() != "y":
            print("Aborted.")
            return
        os.remove(args.out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"Creating merged cache: {args.out}")
    dst_conn = sqlite3.connect(args.out)
    # Enable WAL mode for faster bulk inserts
    dst_conn.execute("PRAGMA journal_mode=WAL")
    dst_conn.execute("PRAGMA synchronous=NORMAL")
    ensure_table(dst_conn)

    # Count source sizes for reporting
    conn_a = sqlite3.connect(args.db_a)
    n_a = count_rows(conn_a)
    conn_a.close()
    conn_b = sqlite3.connect(args.db_b)
    n_b = count_rows(conn_b)
    conn_b.close()
    print(f"  {args.db_a}: {n_a:,} entries")
    print(f"  {args.db_b}: {n_b:,} entries")

    print(f"Merging {args.db_a} …")
    after_a = merge_into(args.db_a, dst_conn)
    print(f"  → {after_a:,} rows in output so far")

    print(f"Merging {args.db_b} …")
    after_b = merge_into(args.db_b, dst_conn)
    collisions = (n_a + n_b) - after_b
    print(f"  → {after_b:,} rows in output ({collisions} collision(s) skipped)")

    dst_conn.execute("PRAGMA optimize")
    dst_conn.close()
    print(f"\nDone. Merged cache written to: {args.out}")


if __name__ == "__main__":
    main()