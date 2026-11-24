"""BrowseComp-Plus prompt and final-answer contract for the local agent."""
import re

from search_comp.data.agent_prompt import build_system_prompt, resolve_system_prompt


ROLE = (
    "You are a deep research agent. Alternate reasoning with searches of the fixed corpus. "
    "Use complementary queries to gather evidence and verify the answer before replying."
)
ANSWER_INSTRUCTION = """When you finish searching, return these three fields:
Explanation: Briefly explain the evidence and cite supporting document IDs inline, for example [20].
Exact Answer: Give the short final answer here.
Confidence: Give your confidence as a percentage from 0% to 100%."""

_FIELD = re.compile(
    r"^[ \t]*(?:\*\*)?(explanation|exact answer|final answer|answer|confidence)"
    r"(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)


def system_prompt(profile, prompt_data, run_dir, allow_visit):
    if profile == "benchmark":
        system, source = resolve_system_prompt()
        system = ROLE + "\n\n" + system.split("\n\n", 1)[1]
        source = f"browsecomp-plus/benchmark-v1 ({source})"
    else:
        system, source = resolve_system_prompt(prompt_data, run_dir)
    return build_system_prompt(system=system, local=True, allow_visit=allow_visit,
                               answer_instruction=ANSWER_INSTRUCTION), source


def extract_exact_answer(response):
    """Extract the labeled answer from visible text already cleaned by the agent."""
    fields = list(_FIELD.finditer(response))
    if not fields:
        return response.strip()
    answers = {}
    for index, field in enumerate(fields):
        end = fields[index + 1].start() if index + 1 < len(fields) else len(response)
        name = field[1].lower()
        answers.setdefault(name, []).append(response[field.end():end].strip())
    for label in ("exact answer", "final answer", "answer"):
        if label in answers:
            values = set(answers[label])
            return values.pop() if len(values) == 1 else ""
    return ""
