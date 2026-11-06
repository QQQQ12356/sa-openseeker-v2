"""Shell configuration contracts without invoking Bash, models, or network."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from search_comp.experiments.shell_config import assemble, decode, launcher_defaults, load_settings


class ShellConfigTests(unittest.TestCase):
    def test_nested_training_and_environment_names_are_preserved(self):
        schema = {
            "TRAINING__LOSS_SEGMENTS__ANSWER_CONTENT": {"env": "SA_WEIGHT", "type": "float"},
            "ENVIRONMENT__HF_ENDPOINT": {"env": "HF_ENDPOINT", "type": "str"},
        }
        result = assemble(schema, {"SA_WEIGHT": "0.5", "HF_ENDPOINT": "https://hf-mirror.com"})
        self.assertEqual(result, {"TRAINING": {"loss_segments": {"answer_content": 0.5}},
                                  "ENVIRONMENT": {"HF_ENDPOINT": "https://hf-mirror.com"}})

    def test_optional_paths_and_step_counts(self):
        self.assertIsNone(decode("", "optional", "MODEL_PATH"))
        self.assertEqual(decode("/models/with spaces", "optional", "MODEL_PATH"), "/models/with spaces")
        self.assertEqual(decode("100", "optional", "MAX_TRAIN_STEPS"), 100)

    def test_invalid_types_fail_before_business_operations(self):
        for value, kind in (("True", "int"), ("1.5", "int"), ("yes", "bool"), ("{}", "list")):
            with self.subTest(value=value, kind=kind), self.assertRaises(ValueError):
                decode(value, kind, "SETTING")
        self.assertEqual(decode('["/data/a b.jsonl"]', "list", "INPUTS"), ["/data/a b.jsonl"])
        self.assertFalse(decode("false", "bool", "PLAN_ONLY"))

    def test_missing_export_is_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "SA_MODEL_PATH"):
            assemble({"MODEL_PATH": {"env": "SA_MODEL_PATH", "type": "str"}}, {})

    def test_python_adapter_requires_its_own_launcher_identity(self):
        with patch.dict(os.environ, {"SA_ENTRY": "beacon/train"}):
            with self.assertRaisesRegex(RuntimeError, "plain/sh/train.sh"):
                load_settings(Path("scripts/plain/py/train.py"), {})
        with patch.dict(os.environ, {"SA_ENTRY": "plain/train", "SA_PLAN_ONLY": "true"}):
            result = load_settings(Path("scripts/plain/py/train.py"),
                                   {"PLAN_ONLY": {"env": "SA_PLAN_ONLY", "type": "bool"}})
        self.assertEqual(result, {"PLAN_ONLY": True})

    def test_static_inspection_does_not_expand_or_execute_shell_text(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "entry.sh"
            launcher.write_text("export SA_MODEL_PATH='/models/$(never_execute)'\n"
                                'export SCORER_API_KEY="${SCORER_API_KEY:-}"\n', encoding="utf-8")
            result = launcher_defaults(launcher, {
                "MODEL_PATH": {"env": "SA_MODEL_PATH", "type": "str"},
                "ENVIRONMENT__SCORER_API_KEY": {"env": "SCORER_API_KEY", "type": "str"}})
            self.assertEqual(result["MODEL_PATH"], "/models/$(never_execute)")
            self.assertEqual(result["ENVIRONMENT"]["SCORER_API_KEY"], "")


if __name__ == "__main__":
    unittest.main()
