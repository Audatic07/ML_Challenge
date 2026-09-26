"""Invented fixtures only; run with pytest or unittest discovery."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("strict_validate", Path(__file__).parents[1] / "src" / "strict_validate.py")
strict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strict)


def fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    header = strict.SOURCE_HEADER + "\n"
    for source, rows in {
        1: ["S1-a\tAlpha\tOne\tUS", "S1-b\tBeta\tTwo\tIndia", "S1-c\tGamma\tThree\tFrance"],
        2: ["S2-a\tAlpha\tOne\tUS", "S2-b\tBeta\tTwo\tIndia"],
        3: ["S3-a\tAlpha\tOne\tUS", "S3-b\tGamma\tThree\tFrance"],
    }.items():
        (root / f"test_source{source}.tsv").write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    matching = root / "matching_results.tsv"
    candidate = root / "candidate_pairs.tsv"
    matching.write_text(strict.HEADERS["matching"] + "\nS1-c\tS3-b\nS1-a\tS2-a,S3-a\nS1-b\t\n", encoding="utf-8")
    candidate.write_text(strict.HEADERS["candidate"] + "\nS1-b\t\nS1-c\tS3-b\nS1-a\tS3-a,S2-a,S2-b\n", encoding="utf-8")
    return matching, candidate


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strict-tests-")
        self.root = Path(self.tmp.name)
        self.matching, self.candidate = fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def run_validation(self, **kwargs):
        return strict.validate(self.matching, self.candidate, self.root, expected_s1=3, **kwargs)

    def test_valid_unordered_zero_one_many_france_and_sizes_hashes(self):
        for mode in ("sqlite", "memory"):
            with self.subTest(mode=mode):
                report = self.run_validation(target_index=mode)
                self.assertTrue(report["valid"], report)
                self.assertFalse(report["competition_row_count"])
                self.assertEqual(report["files"]["matching"]["pairs"], 3)
                self.assertEqual(report["files"]["candidate"]["pairs"], 4)
                self.assertEqual(report["files"]["candidate"]["empty_rows"], 1)
                for role, path in (("matching", self.matching), ("candidate", self.candidate)):
                    self.assertEqual(report["files"][role]["bytes"], path.stat().st_size)
                    self.assertEqual(report["files"][role]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(list(self.root.glob("strict-validate-*")), [])

    def test_default_enforces_competition_count(self):
        result = strict.validate(self.matching, self.candidate, self.root)
        self.assertFalse(result["valid"])
        self.assertIn("1,732,544", result["errors"][0])

    def test_both_files_reject_bad_headers(self):
        for role in ("matching", "candidate"):
            for bad in ("uppercase", "padding", "extra_column", "bom", "csv"):
                with self.subTest(role=role, bad=bad):
                    self.matching, self.candidate = fixture(self.root)
                    p = self.matching if role == "matching" else self.candidate
                    text = p.read_text()
                    header, body = text.split("\n", 1)
                    replacement = {"uppercase": header.upper(), "padding": header + " ",
                                   "extra_column": header + "\textra", "bom": "\ufeff" + header,
                                   "csv": header.replace("\t", ",")}[bad]
                    p.write_text(replacement + "\n" + body, encoding="utf-8")
                    self.assertFalse(self.run_validation()["valid"])

    def test_both_files_reject_bad_rows(self):
        corruptions = {
            "missing": lambda x: x.replace("S1-c\tS3-b\n", ""),
            "duplicate_query": lambda x: x + "S1-b\t\n",
            "foreign_query": lambda x: x.replace("S1-c", "S1-foreign"),
            "duplicate_target": lambda x: x.replace("S3-b", "S3-b,S3-b"),
            "unknown_target": lambda x: x.replace("S3-b", "S3-not-in-test"),
            "self_match": lambda x: x.replace("S3-b", "S1-c"),
            "prefix": lambda x: x.replace("S3-b", "X3-b"),
            "trailing_comma": lambda x: x.replace("S3-b", "S3-b,"),
            "leading_comma": lambda x: x.replace("S3-b", ",S3-b"),
            "space": lambda x: x.replace("S3-b", " S3-b"),
            "quoted": lambda x: x.replace("S3-b", '"S3-b"'),
            "extra_column": lambda x: x.replace("S3-b", "S3-b\textra"),
            "no_tab": lambda x: x.replace("S1-c\tS3-b", "S1-c"),
            "blank_line": lambda x: x + "\n",
            "blank_empty_list": lambda x: x.replace("S1-b\t\n", "S1-b\t \n"),
            "nul": lambda x: x.replace("S3-b", "S3-b\x00"),
        }
        for role in ("matching", "candidate"):
            for label, mutate in corruptions.items():
                for mode in ("sqlite", "memory"):
                    with self.subTest(role=role, bad=label, mode=mode):
                        self.matching, self.candidate = fixture(self.root)
                        p = self.matching if role == "matching" else self.candidate
                        p.write_text(mutate(p.read_text()), encoding="utf-8")
                        self.assertFalse(self.run_validation(target_index=mode)["valid"])

    def test_subset_is_hard_failure(self):
        self.candidate.write_text(strict.HEADERS["candidate"] + "\nS1-a\tS2-a\nS1-b\t\nS1-c\tS3-b\n")
        result = self.run_validation()
        self.assertFalse(result["valid"])
        self.assertIn("subset", result["errors"][0])

    def test_missing_inputs_fail_closed(self):
        for name in ("matching_results.tsv", "candidate_pairs.tsv", "test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
            with self.subTest(name=name):
                self.matching, self.candidate = fixture(self.root)
                (self.root / name).unlink()
                self.assertFalse(self.run_validation()["valid"])

    def test_duplicate_catalog_ids_fail(self):
        for source in (1, 2, 3):
            for mode in ("sqlite", "memory"):
                with self.subTest(source=source, mode=mode):
                    self.matching, self.candidate = fixture(self.root)
                    p = self.root / f"test_source{source}.tsv"
                    text = p.read_text()
                    p.write_text(text + text.splitlines()[1] + "\n")
                    self.assertFalse(self.run_validation(target_index=mode)["valid"])

    def test_crlf_and_no_final_newline(self):
        for p in (self.matching, self.candidate):
            p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n").removesuffix(b"\r\n"))
        self.assertTrue(self.run_validation()["valid"])

    def test_invalid_utf8(self):
        with self.candidate.open("ab") as f:
            f.write(b"S1-b\tS2-\xff\n")
        result = self.run_validation()
        self.assertFalse(result["valid"])
        self.assertIn("UTF-8", result["errors"][0])

    def test_no_artificial_candidate_cap_and_sqlite_bind_batching(self):
        targets = [f"S2-wide{i:04}" for i in range(1200)]
        with (self.root / "test_source2.tsv").open("a") as f:
            f.writelines(f"{t}\tInvented\tStreet\tFrance\n" for t in targets)
        self.candidate.write_text(strict.HEADERS["candidate"] + "\nS1-a\tS2-a,S3-a\nS1-b\t"
                                  + ",".join(targets) + "\nS1-c\tS3-b\n")
        self.assertTrue(self.run_validation()["valid"])
        self.assertFalse(self.run_validation(max_line_bytes=1000)["valid"])

    def test_replaced_same_count_s1_is_rejected(self):
        self.candidate.write_text(self.candidate.read_text().replace("S1-b", "S1-a"))
        self.assertFalse(self.run_validation()["valid"])

    def test_empty_file_is_rejected(self):
        self.matching.write_bytes(b"")
        self.assertFalse(self.run_validation()["valid"])


if __name__ == "__main__":
    unittest.main()
