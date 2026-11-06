"""Public experiment contracts; standard library only, no model or network."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from search_comp.data.openseeker import VERSION
from search_comp.evaluation.tool_agent import Registry, ToolError
from search_comp.experiments.benchmarks import resolve_evaluation, resolve_target
from search_comp.experiments import benchmarks
from search_comp.experiments import data as dataset_jobs
from search_comp.experiments.shell_config import launcher_defaults


def parameters(script):
    namespace = runpy.run_path(str(script), run_name="inspect_parameters")
    launcher = script.parent.parent / "sh" / (script.stem + ".sh")
    return launcher_defaults(launcher, namespace["SCHEMA"])
from search_comp.experiments.runtime import ROOT, configure, digest, job, slug, stamp, write_json
from search_comp.experiments.training import resolve_training
from search_comp.trainer.launch import training_command


class ExperimentContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def dataset(self, source="polar"):
        directory = self.directory / "dataset"
        directory.mkdir()
        write_json(directory / "manifest.json", dict(protocol=VERSION,
                   arguments=dict(source=source, max_length=128), tokenizers={"local-model": {}}))
        for split in ("train", "validation", "test"):
            (directory / f"{split}.jsonl").write_text("{}\n" if split == "train" else "", encoding="utf-8")
        return directory

    def training_spec(self, method="plain", source="polar", dataset_name="openseeker"):
        return resolve_training(method, dataset_name, self.dataset(source), "local-model", "qwen35",
                                dict(max_length=128, per_device_batch_size=1), {})

    def evaluation(self):
        directory = self.directory / "evaluation"
        (directory / "results").mkdir(parents=True)
        predictions = directory / "results/predictions.jsonl"
        predictions.write_text('{"id":"q1","prediction":"Paris"}\n', encoding="utf-8")
        write_json(directory / "results/manifest.json", dict(benchmark="hotpotqa"))
        write_json(directory / "job.json", dict(operation="evaluate", status="complete"))
        write_json(directory / "evaluation.json", dict(method="plain", benchmark="hotpotqa",
                   predictions=str(predictions), predictions_sha256=digest(predictions),
                   engine_manifest_sha256=digest(directory / "results/manifest.json")))
        return directory

    def test_plain_retains_dataset_identity_and_empty_validation_is_disabled(self):
        cfg, manifest, fingerprints = self.training_spec()
        self.assertEqual(cfg["data_mode"], "openseeker")
        self.assertIsNone(cfg["val_data_path"])
        self.assertEqual(manifest["arguments"]["source"], "polar")
        self.assertEqual(fingerprints["train_data_path"], digest(cfg["train_data_path"]))

    def test_research_uses_same_protocol_without_becoming_openseeker_dataset(self):
        cfg, manifest, _ = self.training_spec(source="research", dataset_name="research")
        self.assertEqual(cfg["data_mode"], "openseeker")
        self.assertEqual(manifest["arguments"]["source"], "research")

    def test_baseline_training_is_rejected_before_any_data_is_read(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            resolve_training("baseline", "openseeker", "/missing", "model", "qwen35", {}, {})

    def test_wrong_dataset_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset_name"):
            self.training_spec(source="research")

    def test_wrong_tokenizer_identity_is_rejected(self):
        directory = self.dataset()
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            resolve_training("plain", "openseeker", directory, "different-model", "qwen35", dict(max_length=128), {})

    def test_training_cap_cannot_truncate_prepared_trajectories(self):
        with self.assertRaisesRegex(ValueError, "max_length"):
            resolve_training("plain", "openseeker", self.dataset(), "local-model", "qwen35", dict(max_length=64), {})

    def test_baseline_rejects_a_training_run(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            resolve_target("baseline", "some-run", "local-model")

    def test_training_method_rejects_an_explicit_model_override(self):
        with self.assertRaisesRegex(ValueError, "TRAINING_RUN"):
            resolve_target("plain", "some-run", "local-model")

    def test_planned_training_cannot_supply_a_checkpoint(self):
        write_json(self.directory / "job.json", dict(operation="plan", status="complete"))
        write_json(self.directory / "experiment.json", dict(method="plain", dataset_name="openseeker"))
        with self.assertRaisesRegex(ValueError, "plan/failed"):
            resolve_target("plain", self.directory, None)

    def test_completed_research_run_resolves_to_its_own_method(self):
        write_json(self.directory / "job.json", dict(operation="train", status="complete"))
        write_json(self.directory / "experiment.json", dict(method="beacon", dataset_name="research"))
        result = resolve_target("beacon", self.directory, None)
        self.assertEqual(result, ("research", self.directory.resolve(), None, "beacon", "checkpoint"))
        with self.assertRaisesRegex(ValueError, "方法"):
            resolve_target("plain", self.directory, None)

    def test_score_requires_matching_method_and_benchmark(self):
        directory = self.evaluation()
        resolve_evaluation("plain", "hotpotqa", directory)
        for method, benchmark in (("beacon", "hotpotqa"), ("plain", "browsecomp-plus")):
            with self.assertRaisesRegex(ValueError, "不一致"):
                resolve_evaluation(method, benchmark, directory)

    def test_modified_predictions_are_rejected(self):
        directory = self.evaluation()
        (directory / "results/predictions.jsonl").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "预测文件"):
            resolve_evaluation("plain", "hotpotqa", directory)

    def test_modified_evaluation_manifest_is_rejected(self):
        directory = self.evaluation()
        write_json(directory / "results/manifest.json", dict(benchmark="wrong"))
        with self.assertRaisesRegex(ValueError, "manifest"):
            resolve_evaluation("plain", "hotpotqa", directory)

    def test_job_records_complete_status_and_refuses_to_overwrite(self):
        directory = self.directory / "job"
        with job(directory, "test", dict(method="plain")):
            print("captured-output")
        self.assertEqual(json.loads((directory / "job.json").read_text())["status"], "complete")
        self.assertIn("captured-output", (directory / "console.log").read_text())
        with self.assertRaises(FileExistsError):
            with job(directory, "test", {}):
                pass

    def test_job_archives_the_actual_shell_configuration(self):
        launcher = self.directory / "entry.sh"
        launcher.write_text("export SA_PLAN_ONLY=true\n", encoding="utf-8")
        directory = self.directory / "archived"
        with patch.dict(os.environ, {"SA_LAUNCHER": str(launcher)}):
            with job(directory, "test", {}):
                pass
        self.assertEqual((directory / "source_launcher.sh").read_bytes(), launcher.read_bytes())

    def test_job_records_failure_and_reraises_the_original_error(self):
        directory = self.directory / "failed"
        with self.assertRaisesRegex(RuntimeError, "expected failure"):
            with job(directory, "test", {}):
                raise RuntimeError("expected failure")
        self.assertEqual(json.loads((directory / "job.json").read_text())["status"], "failed")

    def test_artifact_names_cannot_escape_their_method_directory(self):
        for label in ("../beacon", "a/b", "a\\b", "", "two words"):
            with self.assertRaises(ValueError):
                slug(label)
        self.assertNotEqual(stamp("repeat"), stamp("repeat"))

    def test_windows_guard_prevents_real_jobs(self):
        with patch("platform.system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "Linux"):
                configure({})

    def test_multi_gpu_training_uses_the_explicit_inline_zero_config(self):
        with patch.dict(os.environ, dict(CUDA_VISIBLE_DEVICES="2,3", NPROC_PER_NODE="2", DISTRIBUTED_BACKEND="zero2")):
            command = training_command("search_comp.trainer.plain_sft_trainer", ["--config", "run/train.yaml"],
                       deepspeed_configs={"zero2": {"zero_optimization": {"stage": 2}}})
        self.assertIn("--nproc_per_node=2", command)
        self.assertEqual(command[-2], "--set")
        self.assertEqual(json.loads(command[-1].split("=", 1)[1])["zero_optimization"]["stage"], 2)

    def test_beacon_zero3_is_rejected(self):
        with patch.dict(os.environ, dict(NPROC_PER_NODE="2", DISTRIBUTED_BACKEND="zero3")):
            with self.assertRaisesRegex(ValueError, "zero3"):
                training_command("search_comp.trainer.beacon_trainer", [])

    def test_online_tool_call_budget_blocks_the_whole_excess_batch(self):
        calls = []
        registry = Registry({"search": lambda **kwargs: calls.append(kwargs) or "ok"}, max_calls=1)
        action = dict(name="search", arguments=dict(query=["q"]))
        registry.execute([action])
        with self.assertRaisesRegex(ToolError, "budget"):
            registry.execute([action])
        self.assertEqual(calls, [dict(query=["q"])])

    def test_research_reselection_preserves_published_validation_membership(self):
        from search_comp.data.research import normalize_research
        raw = dict(messages=[dict(role="system", content='Research. <tools>{"name":"search"}</tools>'),
            dict(role="user", content="Where?"), dict(role="reasoning", content="Look up."),
            dict(role="tool_call", content='<tool_call>{"name":"search","arguments":{"query":["city"]}}</tool_call>'),
            dict(role="tool_output", content="Paris."), dict(role="answer", content="Paris")])
        original = self.directory / "validation.jsonl"
        original.write_text(json.dumps(raw) + "\n", encoding="utf-8")
        prepared = self.directory / "prepared"
        prepared.mkdir()
        row = normalize_research(raw)
        row["lengths"] = {"local-model": dict(total=100)}
        (prepared / "normalized.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        write_json(prepared / "manifest.json", dict(protocol=VERSION, tokenizers={"local-model": {}},
            arguments=dict(source="research", input=[str(original)], seed=42, val_fraction=0, test_fraction=0)))
        with patch.object(dataset_jobs, "configure"), patch.object(dataset_jobs, "ROOT", self.directory):
            dataset_jobs.select_dataset(dataset_name="research", prepared=prepared, profile="derived",
                tokenizer="local-model", min_length=1, max_length=128, exact_length=None,
                correctness="any", observations="any", environment={})
        selected = self.directory / "artifacts/datasets/research/prepared/derived"
        self.assertEqual((selected / "train.jsonl").read_text(), "")
        self.assertEqual(json.loads((selected / "validation.jsonl").read_text())["question"], "Where?")

    def test_every_method_entry_imports_without_running_or_loading_models(self):
        for method, count in (("plain", 8), ("beacon", 8), ("baseline", 6)):
            scripts = list((ROOT / "scripts" / method / "py").glob("*.py"))
            self.assertEqual(len(scripts), count)
            for script in scripts:
                namespace = parameters(script)
                self.assertIn("ENVIRONMENT", namespace)

    def test_all_fifteen_experiment_benchmark_pairs_archive_and_score_at_their_owner(self):
        """Exercise the public orchestration with a tiny in-memory engine adapter."""
        from search_comp.workflows import benchmark_openseeker, offline_benchmark
        methods = [("plain", "openseeker"), ("plain", "research"),
                   ("beacon", "openseeker"), ("beacon", "research"), ("baseline", "untrained")]
        for method, dataset in methods:
            for benchmark, number in (("browsecomp", 40), ("browsecomp-plus", 42), ("hotpotqa", 44)):
                with self.subTest(method=method, dataset=dataset, benchmark=benchmark):
                    filename = ROOT / "scripts" / method / "py" / f"{number}_test_{benchmark.replace('-', '_')}.py"
                    generation = parameters(filename)["GENERATION"]
                    assets = self.directory / "benchmarks" / benchmark
                    assets.mkdir(parents=True, exist_ok=True)
                    (assets / "questions.jsonl").write_text('{"query":"q","answer":"Paris"}\n', encoding="utf-8")
                    training_run = None
                    if method != "baseline":
                        training_run = self.directory / method / dataset
                        training_run.mkdir(parents=True, exist_ok=True)
                        write_json(training_run / "job.json", dict(operation="train", status="complete"))
                        write_json(training_run / "experiment.json", dict(method=method, dataset_name=dataset))
                    observed = []

                    def engine(args):
                        observed.append(args)
                        args.output_dir.mkdir(parents=True, exist_ok=False)
                        metadata = dict(benchmark=benchmark, data_sha256=digest(assets / "questions.jsonl"))
                        write_json(args.output_dir / "manifest.json", metadata)
                        if args.predictions is None:
                            name = f"result_tool{args.tool_count}.jsonl" if benchmark == "browsecomp" else "predictions.jsonl"
                            (args.output_dir / name).write_text('{"prediction":"Paris"}\n', encoding="utf-8")
                        else:
                            write_json(args.output_dir / "summary.metrics.json", dict(complete=True, em=1.0))

                    adapter = benchmark_openseeker if benchmark == "browsecomp" else offline_benchmark
                    with patch.object(benchmarks, "configure"), patch.object(benchmarks, "ROOT", self.directory), patch.object(adapter, "run", engine):
                        evaluation = benchmarks.evaluate(method=method, benchmark=benchmark, benchmark_dir=assets,
                            training_run=training_run, model_path="/local/base" if method == "baseline" else None,
                            prompt_data=None, run_name="smoke", generation=generation, environment={})
                        scoring = benchmarks.score(method=method, benchmark=benchmark, evaluation_dir=evaluation,
                            run_name="judge", judge=dict(judge_model=None, llm_judge=False), environment={})
                    owner = training_run or self.directory / "artifacts/experiments/baseline/untrained"
                    self.assertTrue(evaluation.is_relative_to(owner / "evaluation" / benchmark))
                    self.assertTrue(scoring.is_relative_to(evaluation / "scores"))
                    self.assertEqual(observed[0].kind, "beacon" if method == "beacon" else "plain")
                    self.assertEqual(observed[0].model_origin, "base-qwen35" if method == "baseline" else "checkpoint")
                    self.assertEqual(observed[1].limit, generation["limit"])
                    self.assertIsNone(observed[1].model)
                    self.assertEqual(json.loads((scoring / "job.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
