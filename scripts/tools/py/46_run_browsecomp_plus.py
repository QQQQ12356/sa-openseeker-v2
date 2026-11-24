#!/usr/bin/env python3
"""Run one BrowseComp-Plus configuration using the same-name Linux launcher."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SCHEMA = {
    f"ENVIRONMENT__{name}": {"env": name, "type": "str"}
    for name in ("HF_ENDPOINT", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES")
}
for kind, names in {
    "str": ("METHOD BENCHMARK_DIR RUN_NAME OVERLAP GENERATION__DECODING GENERATION__RETRIEVER "
            "GENERATION__EMBEDDING_DEVICE JUDGE__JUDGE_BACKEND JUDGE__JUDGE_MODEL JUDGE__JUDGE_URL"),
    "optional": ("TRAINING_RUN CHECKPOINT MODEL_PATH PROMPT_DATA RESUME_FROM EVALUATION_DIR "
                 "GENERATION__VECTOR_INDEX_DIR GENERATION__EMBEDDING_MODEL_PATH GENERATION__SEARCH_SERVICE_URL"),
    "bool": "GENERATION__VERIFY_INDEX GENERATION__ALLOW_VISIT",
    "int": ("GENERATION__LIMIT GENERATION__NUM_WORKERS GENERATION__OBSERVATION_WINDOW "
            "GENERATION__MAX_CONTEXT GENERATION__MAX_NEW_TOKENS GENERATION__MAX_TURNS "
            "GENERATION__MAX_TOOL_TOKENS GENERATION__TOOL_COUNT GENERATION__TOOL_CALL_RETRIES "
            "GENERATION__TOPK GENERATION__SNIPPET_TOKENS GENERATION__READ_TOKENS "
            "GENERATION__GENERATION_SEED JUDGE__JUDGE_MAX_TOKENS JUDGE__JUDGE_ATTEMPTS"),
    "float": ("OVERLAP_THRESHOLD GENERATION__TITLE_WEIGHT GENERATION__SEARCH_SERVICE_TIMEOUT "
              "GENERATION__REPETITION_PENALTY GENERATION__TEMPERATURE GENERATION__TOP_P JUDGE__JUDGE_TIMEOUT"),
}.items():
    SCHEMA.update({name: {"env": f"SA_{name}", "type": kind} for name in names.split()})


if __name__ == "__main__":
    from search_comp.experiments.benchmarks import evaluate_and_score_browsecomp_plus
    from search_comp.experiments.shell_config import load_settings

    settings = load_settings(__file__, SCHEMA)
    evaluate_and_score_browsecomp_plus(**{key.lower(): value for key, value in settings.items()})
