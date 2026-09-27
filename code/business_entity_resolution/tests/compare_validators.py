"""Run both CLIs on invented fixtures. Never run the supplied RAM-heavy tool on a large candidate file here."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from test_strict_validate import fixture, strict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplied", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    cases = [
        ("valid_reordered", None, None, True, 0),
        ("subset_violation", "candidate", ("S3-a,S2-a,S2-b", "S2-a"), False, 0),
        ("missing_candidate", "delete", None, False, 0),
        ("unknown_target", "candidate", ("S2-b", "S2-unknown"), False, 1),
        ("duplicate_target", "matching", ("S3-b", "S3-b,S3-b"), False, 1),
        ("duplicate_s1", "append", None, False, 1),
        ("missing_s1", "candidate", ("S1-b\t\n", ""), False, 1),
        ("non_exact_header", "matching", ("source1_entity_id", "SOURCE1_ENTITY_ID"), False, 0),
        ("blank_line", "append_blank", None, False, 0),
        ("missing_catalog", "delete_catalog", None, False, 0),
    ]
    for label, role, change, expected_strict, expected_supplied in cases:
        with tempfile.TemporaryDirectory(prefix="validator-comparison-") as directory:
            root = Path(directory)
            matching, candidate = fixture(root)
            if role in ("matching", "candidate"):
                path = matching if role == "matching" else candidate
                path.write_text(path.read_text().replace(*change), encoding="utf-8")
            elif role == "delete":
                candidate.unlink()
            elif role == "delete_catalog":
                (root / "test_source3.tsv").unlink()
            elif role in ("append", "append_blank"):
                with matching.open("a") as f:
                    f.write("S1-b\t\n" if role == "append" else "\n")
            command = ["--matching", str(matching), "--candidate", str(candidate), "--test-dir", str(root)]
            a = subprocess.run([sys.executable, str(Path(strict.__file__)), *command, "--expected-s1", "3", "--quiet"],
                               capture_output=True, encoding="utf-8", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
            b = subprocess.run([sys.executable, str(args.supplied.resolve()), *command, "--check-ids"],
                               capture_output=True, encoding="utf-8", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
            detail = json.loads(a.stdout)
            assert detail["valid"] == expected_strict, (label, a.stdout, a.stderr)
            assert a.returncode == (0 if expected_strict else 1), label
            assert b.returncode == expected_supplied, (label, b.stdout, b.stderr)
            rows.append({"case": label, "strict_exit": a.returncode, "supplied_exit": b.returncode,
                         "supplied_warnings": b.stdout.count("WARNING:")})
    print(json.dumps({"supplied_sha256": __import__("hashlib").sha256(args.supplied.read_bytes()).hexdigest(),
                      "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
