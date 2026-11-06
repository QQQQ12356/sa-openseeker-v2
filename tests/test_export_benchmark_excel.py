"""Linux CPU export checks with synthetic files; no benchmark/model execution."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from search_comp.workflows import export_benchmark_excel as exporter


def write_json(filename, value):
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class ExportFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def score(self, name="score", *, complete=True, status="complete", benchmark="hotpotqa"):
        evaluation = self.root / "experiments/evaluation" / benchmark / "eval"
        directory = evaluation / "scores" / name
        summary = directory / "results/summary.metrics.json"
        write_json(summary, dict(benchmark=benchmark, complete=complete, count=2, em=0.0, f1=0.5))
        write_json(directory / "job.json", dict(operation="score", status=status,
                   settings=dict(method="plain", benchmark=benchmark,
                                 judge=dict(judge_model="/models/judge", judge_attempts=3))))
        write_json(evaluation / "evaluation.json", dict(method="plain", benchmark=benchmark,
                   dataset_name="research", training_run="/runs/research", generation=dict(limit=2)))
        write_json(evaluation / "results/manifest.json", dict(model="/models/trained"))
        return summary


class ExportSourceTests(ExportFixture, unittest.TestCase):
    def test_discovery_deduplicates_overlapping_inputs_and_rejects_empty_inputs(self):
        summary = self.score()
        self.assertEqual(exporter.discover_summaries([str(self.root), str(summary)]), [summary])
        for values in ([], [123], [""], [str(self.root / "missing")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                exporter.discover_summaries(values)

    def test_identity_and_zero_are_preserved_with_real_model_path(self):
        row, summary, eligible = exporter.load_record(self.score())
        self.assertTrue(eligible)
        self.assertEqual(row["dataset_name"], "research")
        self.assertEqual(row["model_path"], "/models/trained")
        self.assertEqual(row["em"], 0)
        self.assertIsNone(row["llm_judge_accuracy"])
        self.assertEqual(row["summary_sha256"], exporter.digest(row["summary_file"]))
        write_json(Path(row["evaluation_dir"]) / "evaluation.json", dict(method="beacon", benchmark="hotpotqa"))
        with self.assertRaisesRegex(ValueError, "method"):
            exporter.load_record(Path(row["summary_file"]))

    def test_incomplete_job_is_ineligible_even_with_complete_summary(self):
        _, _, eligible = exporter.load_record(self.score(status="failed"))
        self.assertFalse(eligible)

    def test_standalone_summary_has_explicit_untracked_status(self):
        summary = self.root / "copied/summary.metrics.json"
        write_json(summary, dict(benchmark="BrowseComp", complete=True, expected=3,
                   unique_predictions=2, judge_unknown_or_error=0, accuracy_all_questions=0.5))
        row, _, eligible = exporter.load_record(summary)
        self.assertTrue(eligible)
        self.assertEqual(row["job_status"], "untracked")
        self.assertIsNone(row["method"])
        self.assertEqual((row["count"], row["prediction_count"], row["judge_errors"]), (3, 2, 0))

    def test_bad_json_reports_its_source_and_nonfinite_numbers_are_rejected(self):
        summary = self.root / "summary.metrics.json"
        for content in ('{"f1":NaN}', '{"f1":Infinity}', '{"f1":1e999}', '{broken', '[]'):
            summary.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "summary.metrics.json"):
                exporter.read_object(summary)

    def test_online_details_filter_metadata_without_renumbering_items(self):
        filename = self.root / "result_for_judge_eval.jsonl"
        filename.write_text('\n{"type":"summary"}\n{"type":"item","item_index":8,"score":1}\n', encoding="utf-8")
        self.assertEqual(list(exporter.detail_rows(filename)), [(3, dict(type="item", item_index=8, score=1))])


@unittest.skipUnless(importlib.util.find_spec("openpyxl"), "Install requirements.tools-export.txt on Linux")
class ExcelExportTests(ExportFixture, unittest.TestCase):
    def export(self, summaries, **options):
        settings = dict(inputs=[str(value) for value in summaries], output_dir=self.root / "artifacts/reports",
                        run_name="test_excel", include_details=True, include_incomplete=False)
        settings.update(options)
        with patch.object(exporter, "ROOT", self.root), redirect_stdout(io.StringIO()):
            return exporter.export_benchmarks(**settings)

    def test_workbook_roundtrip_preserves_scales_literals_and_source_rows(self):
        from openpyxl import load_workbook

        summary = self.score()
        write_json(summary, dict(benchmark="hotpotqa", complete=True, count=2, em=0, f1=0.5,
                   **{"Accuracy (%)": 50, "avg_tool_stats": {"search": 1.5}}))
        write_json(summary.parent / "per_question.json", [dict(id="q1", prediction="=1+1", em=0, f1=0.5)])
        (summary.parent / "judge.jsonl").write_text('{"id":"q1","score":0,"candidate_answer":"=1+1"}\n', encoding="utf-8")
        destination = self.export([summary])
        workbook = load_workbook(destination)
        self.addCleanup(workbook.close)
        self.assertEqual(workbook.sheetnames, ["summary", "metrics", "details"])
        summary_rows = list(workbook["summary"].values)
        row = dict(zip(summary_rows[0], summary_rows[1]))
        self.assertEqual((row["em"], row["f1"]), (0, 0.5))
        self.assertIsNone(row["accuracy"])
        metrics = {r[2]: r[3] for r in list(workbook["metrics"].values)[1:]}
        self.assertEqual(metrics["Accuracy (%)"], 50)
        self.assertEqual(metrics["avg_tool_stats.search"], 1.5)
        details = workbook["details"]
        columns = {cell.value: cell.column for cell in details[1]}
        self.assertEqual(details.max_row, 3)  # Metric and judge rows remain distinct.
        cell = details.cell(2, columns["prediction"])
        self.assertEqual((cell.value, cell.data_type), ("=1+1", "s"))
        self.assertEqual(workbook["summary"].freeze_panes, "A2")
        manifest = exporter.read_object(destination.parent / "export_manifest.json")
        self.assertEqual(manifest["sheets"]["details"]["rows"], 2)
        self.assertEqual(len(manifest["detail_sources"]), 2)
        self.assertEqual(manifest["workbook_sha256"], exporter.digest(destination))
        self.assertEqual(exporter.read_object(destination.parent / "job.json")["status"], "complete")

    def test_skip_policy_override_and_fresh_output_directories(self):
        good = self.score("good")
        bad = self.score("bad", complete=False, status="failed")
        destination = self.export([good, bad], include_details=False)
        manifest = exporter.read_object(destination.parent / "export_manifest.json")
        self.assertEqual((len(manifest["accepted"]), len(manifest["skipped"])), (1, 1))
        self.assertNotIn("details", manifest["sheets"])
        other = self.export([good, bad], include_incomplete=True)
        self.assertNotEqual(other.parent, destination.parent)
        self.assertTrue(destination.is_file())
        self.assertEqual(len(exporter.read_object(other.parent / "export_manifest.json")["accepted"]), 2)
        with self.assertRaisesRegex(ValueError, "没有可导出"):
            self.export([bad])

    def test_unicode_limits_controls_and_long_ids_are_handled(self):
        from openpyxl import load_workbook

        summary = self.score()
        # Two UTF-16 units per emoji; Excel's capacity is not Python's len(text).
        write_json(summary.parent / "per_question.json", [dict(id=1234567890123456789,
                   prediction="😀" * 20000 + "\u0001", em=0)])
        destination = self.export([summary])
        workbook = load_workbook(destination)
        self.addCleanup(workbook.close)
        rows = list(workbook["details"].values)
        row = dict(zip(rows[0], rows[1]))
        self.assertEqual(row["id"], "1234567890123456789")
        self.assertLessEqual(len(row["prediction"].encode("utf-16-le")), 32767 * 2)
        self.assertIn("truncated", row["prediction"])
        self.assertNotIn("\u0001", row["prediction"])
        counters = exporter.read_object(destination.parent / "export_manifest.json")["sheets"]["details"]
        self.assertEqual((counters["clipped_cells"], counters["cleaned_cells"]), (1, 1))

    def test_output_cannot_escape_project_artifacts(self):
        with self.assertRaisesRegex(ValueError, "OUTPUT_DIR"):
            self.export([self.score()], output_dir=self.root / "outside")

    def test_browsecomp_plus_judge_fields_and_retrieval_only_missing_accuracy(self):
        from openpyxl import load_workbook

        summary = self.score(benchmark="browsecomp-plus")
        write_json(summary, dict(benchmark="browsecomp-plus", complete=True, count=2,
                   score_mode="retrieval-only", evidence_recall=0.25, gold_recall=0.5))
        (summary.parent / "judge.jsonl").write_text(json.dumps(dict(id="q1", completed=True,
            evidence_recall=0.25, judge_result=dict(correct=True, reasoning="Matches reference"))) + "\n", encoding="utf-8")
        destination = self.export([summary])
        workbook = load_workbook(destination)
        self.addCleanup(workbook.close)
        rows = list(workbook["summary"].values)
        row = dict(zip(rows[0], rows[1]))
        self.assertIsNone(row["accuracy"])
        self.assertEqual(row["evidence_recall"], 0.25)
        rows = list(workbook["details"].values)
        detail = dict(zip(rows[0], rows[1]))
        self.assertTrue(detail["correct"])
        self.assertEqual(detail["judge_reasoning"], "Matches reference")


if __name__ == "__main__":
    unittest.main()
