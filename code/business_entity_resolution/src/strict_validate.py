"""Strict streaming validation of both final TSVs; Python stdlib only.

Both outputs and all test catalogs are mandatory. Row order may differ. SQLite
stores catalog IDs, coverage flags and matching lists, never candidate lists.
RAM is O(largest line + SQLite cache); --target-index memory uses O(target IDs)
RAM for faster membership. Run on immutable files extracted from the final ZIP.

TSV validity does not prove equality to the model's scored-input manifest.
The release owner must separately reconcile that manifest and shard coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

EXPECTED_S1 = 1_732_544
HEADERS = {"matching": "source1_entity_id\tmatched_entity_ids",
           "candidate": "source1_entity_id\tcandidate_entity_ids"}
SOURCE_HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry"


class ValidationError(ValueError):
    """A required input or submission invariant failed."""


def lines(path, header, stats, max_line_bytes=0):
    """Strict UTF-8 physical lines; hash bytes without another file pass."""
    before = path.stat()
    stats.update(bytes=before.st_size, gib=before.st_size / 2**30,
                 sha256=None, completely_read=False)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        line_number = 0
        while True:
            raw = stream.readline(max_line_bytes + 1 if max_line_bytes else -1)
            if not raw:
                break
            line_number += 1
            if max_line_bytes and len(raw) > max_line_bytes:
                raise ValidationError(f"{path.name}:{line_number}: exceeds --max-line-bytes")
            digest.update(raw)
            body = raw[:-2] if raw.endswith(b"\r\n") else raw.removesuffix(b"\n")
            try:
                text = body.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ValidationError(f"{path.name}:{line_number}: invalid UTF-8") from exc
            if line_number == 1:
                if text != header:
                    raise ValidationError(f"{path.name}: header must be exactly {header!r}")
            else:
                yield line_number, text
        if not line_number:
            raise ValidationError(f"{path.name}: empty file (missing header)")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValidationError(f"{path.name}: file changed during validation")
    stats.update(sha256=digest.hexdigest(), completely_read=True)


def valid_token(value, prefixes):
    return (value.startswith(prefixes) and len(value) > 3
            and not any(c.isspace() or ord(c) < 32 or c in ',"\x7f' for c in value))


def parse_list(text, where):
    if not text:
        return set()
    ids = text.split(",")
    if any(not valid_token(item, ("S2-", "S3-")) for item in ids):
        raise ValidationError(f"{where}: invalid target ID/list syntax (no blanks or quoting)")
    unique = set(ids)
    if len(unique) != len(ids):
        raise ValidationError(f"{where}: duplicate target ID within list")
    return unique


def validate(matching, candidate, test_dir, *, expected_s1=EXPECTED_S1,
             scratch_dir=None, target_index="sqlite", cache_mib=64,
             max_line_bytes=0, progress=None):
    """Return a JSON report, failing closed on the first error.

    Override expected_s1 only for fixtures/subsets; the report marks that choice.
    Temporary SQLite indexes are disposable local scratch, never a reused cache.
    """
    started = time.monotonic()
    report = {"valid": False, "expected_s1": expected_s1,
              "competition_row_count": expected_s1 == EXPECTED_S1,
              "target_index": target_index, "files": {}, "errors": [],
              "scope": "TSV syntax, exact coverage, target membership, uniqueness and subset only"}
    paths = {"matching": Path(matching), "candidate": Path(candidate)}
    paths.update({f"source{s}": Path(test_dir) / f"test_source{s}.tsv" for s in (1, 2, 3)})
    try:
        if expected_s1 < 1 or cache_mib < 1 or max_line_bytes < 0:
            raise ValidationError("expected_s1/cache_mib must be positive; max_line_bytes nonnegative")
        if target_index not in ("sqlite", "memory"):
            raise ValidationError("target_index must be sqlite or memory")
        for role, path in paths.items():
            report["files"][role] = {"path": str(path), "bytes": path.stat().st_size,
                                     "gib": path.stat().st_size / 2**30}
        with tempfile.TemporaryDirectory(prefix="strict-validate-", dir=scratch_dir) as temporary:
            db = sqlite3.connect(str(Path(temporary) / "index.sqlite"))
            try:
                db.execute(f"PRAGMA cache_size=-{cache_mib * 1024}")
                db.execute("PRAGMA temp_store=FILE")
                db.execute("PRAGMA journal_mode=OFF")
                db.execute("PRAGMA synchronous=OFF")
                db.execute("CREATE TABLE queries(id TEXT PRIMARY KEY, matched TEXT, candidate_seen INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID")
                db.execute("CREATE TABLE targets(id TEXT PRIMARY KEY) WITHOUT ROWID")
                targets = set() if target_index == "memory" else None
                for source in (1, 2, 3):
                    role = f"source{source}"
                    stats = report["files"][role]
                    count = 0
                    for number, text in lines(paths[role], SOURCE_HEADER, stats, max_line_bytes):
                        entity, tab, _ = text.partition("\t")
                        if not tab or not valid_token(entity, (f"S{source}-",)):
                            raise ValidationError(f"{paths[role].name}:{number}: invalid catalog ID/row")
                        try:
                            if source == 1:
                                db.execute("INSERT INTO queries(id) VALUES (?)", (entity,))
                            elif targets is None:
                                db.execute("INSERT INTO targets VALUES (?)", (entity,))
                            else:
                                if entity in targets:
                                    raise ValidationError(f"{paths[role].name}:{number}: duplicate catalog ID")
                                targets.add(entity)
                        except sqlite3.IntegrityError as exc:
                            raise ValidationError(f"{paths[role].name}:{number}: duplicate catalog ID") from exc
                        count += 1
                        if count % 100_000 == 0:
                            db.commit()
                            if progress:
                                progress(f"{role}: {count:,} catalog IDs")
                    stats["rows"] = count
                    db.commit()
                    if source == 1 and count != expected_s1:
                        raise ValidationError(f"test_source1.tsv: {count:,} rows; expected {expected_s1:,}")

                def check_targets(ids, where):
                    if targets is not None:
                        unknown = not ids.issubset(targets)
                    else:
                        values = list(ids)
                        found = 0
                        for start in range(0, len(values), 900):
                            batch = values[start:start + 900]
                            placeholders = ",".join("?" for _ in batch)
                            found += db.execute(f"SELECT count(*) FROM targets WHERE id IN ({placeholders})", batch).fetchone()[0]
                        unknown = found != len(ids)
                    if unknown:
                        raise ValidationError(f"{where}: target ID does not exist in test S2/S3")

                for role in ("matching", "candidate"):
                    stats = report["files"][role]
                    stats.update(rows=0, pairs=0, empty_rows=0, max_ids_per_row=0)
                    for number, text in lines(paths[role], HEADERS[role], stats, max_line_bytes):
                        where = f"{paths[role].name}:{number}"
                        if text.count("\t") != 1:
                            raise ValidationError(f"{where}: expected exactly two tab-separated fields")
                        query, listed = text.split("\t")
                        if not valid_token(query, ("S1-",)):
                            raise ValidationError(f"{where}: invalid S1 ID")
                        row = db.execute("SELECT matched,candidate_seen FROM queries WHERE id=?", (query,)).fetchone()
                        if row is None:
                            raise ValidationError(f"{where}: S1 ID is not in test_source1.tsv")
                        if (role == "matching" and row[0] is not None) or (role == "candidate" and row[1]):
                            raise ValidationError(f"{where}: duplicate S1 row")
                        ids = parse_list(listed, where)
                        check_targets(ids, where)
                        if role == "matching":
                            db.execute("UPDATE queries SET matched=? WHERE id=?", (listed, query))
                        else:
                            matched = set(row[0].split(",")) if row[0] else set()
                            if not matched.issubset(ids):
                                raise ValidationError(f"{where}: matches are not a subset of candidates")
                            db.execute("UPDATE queries SET candidate_seen=1 WHERE id=?", (query,))
                        stats["rows"] += 1
                        stats["pairs"] += len(ids)
                        stats["empty_rows"] += not ids
                        stats["max_ids_per_row"] = max(stats["max_ids_per_row"], len(ids))
                        if stats["rows"] % 10_000 == 0:
                            db.commit()
                            if progress:
                                progress(f"{role}: {stats['rows']:,} rows; {stats['pairs']:,} IDs")
                    db.commit()
                    clause = "matched IS NULL" if role == "matching" else "candidate_seen=0"
                    missing = db.execute(f"SELECT count(*) FROM queries WHERE {clause}").fetchone()[0]
                    if missing or stats["rows"] != expected_s1:
                        raise ValidationError(f"{paths[role].name}: missing {missing:,} S1 rows; {stats['rows']:,} rows read; expected {expected_s1:,}")
                    stats["mean_ids_per_row"] = stats["pairs"] / expected_s1
                report["scratch_bytes"] = (Path(temporary) / "index.sqlite").stat().st_size
                report["valid"] = True
            finally:
                db.close()
    except (ValidationError, OSError, sqlite3.Error) as exc:
        report["errors"].append(str(exc))
    report["seconds"] = round(time.monotonic() - started, 3)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--expected-s1", type=int, default=EXPECTED_S1, help="Override ONLY for fixtures/subsets (default: %(default)s)")
    parser.add_argument("--scratch-dir", type=Path, help="Existing private directory with spare disk capacity")
    parser.add_argument("--target-index", choices=["sqlite", "memory"], default="sqlite")
    parser.add_argument("--cache-mib", type=int, default=64)
    parser.add_argument("--max-line-bytes", type=int, default=0, help="Optional resource guard; 0 means no artificial row-size cap")
    parser.add_argument("--report", type=Path, help="Write JSON to a new local report file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    # Exclusive creation prevents accidentally overwriting a validation input.
    report_stream = args.report.open("x", encoding="utf-8") if args.report else None
    try:
        report = validate(args.matching, args.candidate, args.test_dir, expected_s1=args.expected_s1,
                          scratch_dir=args.scratch_dir, target_index=args.target_index,
                          cache_mib=args.cache_mib, max_line_bytes=args.max_line_bytes,
                          progress=None if args.quiet else lambda text: print(text, file=sys.stderr, flush=True))
        serialized = json.dumps(report, indent=2) + "\n"
        if report_stream:
            report_stream.write(serialized)
        print(serialized, end="")
        print("PASS" if report["valid"] else "FAIL", file=sys.stderr)
        return 0 if report["valid"] else 1
    finally:
        if report_stream:
            report_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
