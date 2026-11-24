"""Typed settings shared by the three method-owned BrowseComp-Plus launchers."""
from .shell_config import load_settings


def launcher_schema(method):
    schema = {
        f"ENVIRONMENT__{name}": {"env": name, "type": "str"}
        for name in ("HF_ENDPOINT", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES")
    }
    for kind, names in {
        "str": ("BENCHMARK_DIR RUN_NAME GENERATION__PROMPT_PROFILE GENERATION__DECODING "
                "GENERATION__RETRIEVER GENERATION__EMBEDDING_DEVICE JUDGE__JUDGE_BACKEND "
                "JUDGE__JUDGE_MODEL JUDGE__JUDGE_URL"),
        "optional": ("PROMPT_DATA RESUME_FROM EVALUATION_DIR GENERATION__VECTOR_INDEX_DIR "
                     "GENERATION__EMBEDDING_MODEL_PATH"),
        "bool": "SCORE_AFTER_GENERATION GENERATION__VERIFY_INDEX GENERATION__ALLOW_VISIT",
        "int": ("GENERATION__LIMIT GENERATION__NUM_WORKERS GENERATION__OBSERVATION_WINDOW "
                "GENERATION__MAX_CONTEXT GENERATION__MAX_NEW_TOKENS GENERATION__MAX_TURNS "
                "GENERATION__MAX_TOOL_TOKENS GENERATION__TOOL_COUNT GENERATION__TOOL_CALL_RETRIES "
                "GENERATION__TOPK GENERATION__SNIPPET_TOKENS GENERATION__READ_TOKENS "
                "GENERATION__GENERATION_SEED JUDGE__JUDGE_MAX_TOKENS JUDGE__JUDGE_ATTEMPTS"),
        "float": ("GENERATION__TITLE_WEIGHT GENERATION__REPETITION_PENALTY "
                  "GENERATION__TEMPERATURE GENERATION__TOP_P JUDGE__JUDGE_TIMEOUT"),
    }.items():
        schema.update({name: {"env": f"SA_{name}", "type": kind} for name in names.split()})
    method_fields = {"MODEL_PATH": "str"} if method == "baseline" else {
        "TRAINING_RUN": "str", "CHECKPOINT": "optional", "OVERLAP": "str", "OVERLAP_THRESHOLD": "float",
    }
    schema.update({name: {"env": f"SA_{name}", "type": kind} for name, kind in method_fields.items()})
    return schema


def launch(method, filename, schema):
    from .benchmarks import run_browsecomp_plus

    settings = load_settings(filename, schema)
    run_browsecomp_plus(method=method, **{key.lower(): value for key, value in settings.items()})
