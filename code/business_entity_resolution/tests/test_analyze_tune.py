"""Invented tune partitions; no real labels or remote services."""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import polars as pl

spec = importlib.util.spec_from_file_location("analyze_tune", Path(__file__).parents[1] / "src" / "analyze_tune.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture():
    truth = pl.DataFrame({"s1": [0, 0, 0, 2, 3], "target_id": ["S2-a", "S2-b", "S2-c", "S2-f", "S2-g"]})
    scores = pl.DataFrame({"s1": [0, 0, 0, 1, 3, 4],
                           "t": [0, 1, 3, 4, 6, 6],
                           "target_id": ["S2-a", "S2-b", "S2-d", "S2-e", "S2-g", "S2-g"],
                           "p": [0.9, 0.3, 0.8, 0.85, 0.6, 0.9]})
    per = pl.DataFrame({"s1": list(range(6)), "country": ["India", "US", "India", "US", "US", "India"],
                        "g": [3, 0, 1, 1, 0, 0], "p": [2, 0, 0, 1, 0, 0], "tp": [2, 0, 0, 1, 0, 0],
                        "f": [5/11, 0., 0., 0., 0., 1.],
                        "oracle": [10/11, 1., 0., 1., 1., 1.]})
    return per, scores, truth


class AnalysisTests(unittest.TestCase):
    def test_exact_ordered_decomposition_exhaustive_counts(self):
        for g in range(8):
            for r in range(g + 1):
                for tp in range(r + 1):
                    for fp in range(5):
                        p = analysis.loss_parts(g, r, tp, fp)
                        parts = [p[c] for c in ("retrieval_loss", "false_positive_loss", "missed_retrieved_loss")]
                        self.assertTrue(all(x >= -1e-12 for x in parts))
                        self.assertAlmostEqual(sum(parts), 1 - p["f"])
        self.assertEqual(analysis.loss_parts(0, 0, 0, 1)["false_positive_loss"], 1)

    def test_assignment_missing_candidates_and_misleading_saved_counts(self):
        detail, tagged, missed = analysis.analyze(*fixture(), 0.5)
        row = detail.filter(pl.col("s1") == 0).to_dicts()[0]
        self.assertEqual((row["tp"], row["fp"], row["retrieval_misses"], row["missed_retrieved"]), (1, 1, 1, 1))
        self.assertAlmostEqual(row["false_positive_loss"], 5/7 - 5/11)
        self.assertAlmostEqual(row["missed_retrieved_loss"], 10/11 - 5/7)
        self.assertEqual(detail.filter(pl.col("s1") == 3)["owner_lost"].item(), 1)
        self.assertEqual(missed.height, 2)
        summary = analysis.summarize(detail)
        self.assertEqual(summary["overall"]["queries"], 6)
        self.assertEqual(summary["overall"]["zero_candidate_queries"], 2)
        self.assertEqual(summary["overall"]["singleton_false_merges"], 2)
        self.assertEqual(summary["overall"]["below_threshold"], 1)
        for c in ("loss", "retrieval_loss", "false_positive_loss", "missed_retrieved_loss"):
            self.assertAlmostEqual(sum(r[c + "_contribution"] for r in summary["by_country"]), summary["overall"][c])
        selected = analysis.choose_examples(detail, 20)
        self.assertEqual(len(selected), 5)
        self.assertEqual(len(set(selected)), 5)

    def test_owner_tie_uses_lower_s1(self):
        per, scores, truth = fixture()
        scores = scores.with_columns(pl.when(pl.col("t") == 6).then(0.9).otherwise(pl.col("p")).alias("p"))
        per = per.with_columns(pl.when(pl.col("s1").is_in([3, 4])).then(1.).otherwise(pl.col("f")).alias("f"))
        detail, _, _ = analysis.analyze(per, scores, truth, 0.5)
        self.assertEqual(detail.filter(pl.col("s1") == 3)["tp"].item(), 1)
        self.assertEqual(detail.filter(pl.col("s1") == 4)["fp"].item(), 0)

    def test_mismatched_artifacts_fail(self):
        per, scores, truth = fixture()
        with self.assertRaisesRegex(ValueError, "Recomputed f"):
            analysis.analyze(per, scores, truth, 0.95)
        with self.assertRaisesRegex(ValueError, "Duplicate score"):
            analysis.analyze(per, pl.concat([scores, scores.head(1)]), truth, 0.5)
        with self.assertRaisesRegex(ValueError, "probabilities"):
            analysis.analyze(per, scores.with_columns(pl.lit(float("nan")).alias("p")), truth, 0.5)
        with self.assertRaisesRegex(ValueError, "outside tune"):
            analysis.analyze(per, scores.with_columns((pl.col("s1") + 99).alias("s1")), truth, 0.5)

    def test_end_to_end_private_outputs(self):
        per, scores, _ = fixture()
        with tempfile.TemporaryDirectory(prefix="tune-test-") as tmp:
            root = Path(tmp)
            model = root / "model"
            train = root / "train"
            model.mkdir()
            train.mkdir()
            per.write_parquet(model / "tune_per_query.parquet")
            scores.write_parquet(model / "tune_scores.parquet")
            (model / "model_manifest.json").write_text(json.dumps({"threshold": .5, "plan_sha": "invented",
                "model_sha256": "invented", "metrics": {"tune_queries": 6, "macro_f05": (5/11+1)/6,
                "oracle_u": (10/11+4)/6}}))
            (root / "plan.json").write_text(json.dumps({"plan_sha": "invented", "tune_rows": list(range(6))}))
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            countries = per["country"].to_list()
            (train / "train_source1.tsv").write_text(header + "".join(f"S1-{i}\tInvented {i}\tRoad {i}\t{countries[i]}\n" for i in range(6)))
            (train / "train_source2.tsv").write_text(header + "".join(f"S2-{c}\tInvented {c}\tRoad {c}\tUS\n" for c in "abcdefg"))
            (train / "train_source3.tsv").write_text(header)
            (train / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n"
                "S1-0\tS2-a,S2-b,S2-c\nS1-1\t\nS1-2\tS2-f\nS1-3\tS2-g\nS1-4\t\nS1-5\t\n")
            analysis.main(["--model-dir", str(model), "--train-dir", str(train),
                           "--out", str(root / "out"), "--plan", str(root / "plan.json")])
            result = json.loads((root / "out" / "aggregate.json").read_text())
            self.assertFalse(result["audit_read"])
            self.assertEqual(result["example_count"], 5)
            examples = json.loads((root / "out" / "examples.private.json").read_text())
            self.assertEqual(len(examples), 5)
            self.assertEqual(len(result["source_files"]), 4)
            with self.assertRaisesRegex(ValueError, "Not ready"):
                analysis.main(["--model-dir", str(root / "missing"), "--train-dir", str(train), "--out", str(root / "missing-out")])


if __name__ == "__main__":
    unittest.main()
