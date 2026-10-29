"""Shared OpenSeeker agent instructions for SFT and native tool rollouts.

Backend capabilities and benchmark output contracts are explicit extensions;
model kind never changes the prompt. Search-R1 and upstream reproduction use
their own protocols.
"""
import hashlib
import json


PROMPT_VERSION = "openseeker-agent-v1"
BASE_INSTRUCTION = "You are a tool-augmented QA agent. Use the available tools to answer the user's question."
TOOL_SCHEMAS = [
    {"name": "search", "description": "Search the available sources with complementary queries; return ranked results.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, "required": ["query"]}},
    {"name": "visit", "description": "Read the supplied source URLs for the stated goal.",
     "parameters": {"type": "object", "properties": {
         "url": {"type": ["string", "array"], "items": {"type": "string"}, "minItems": 1},
         "goal": {"type": "string"}}, "required": ["url", "goal"]}},
]
TOOL_PROTOCOL = '''Call tools using this exact format:
<tool_calls_begin>
<tool_call>{"name": "search", "arguments": {"query": ["keywords"]}}</tool_call>
</tool_calls_end>
For multiple calls, put multiple <tool_call>JSON</tool_call> blocks inside one envelope.
Tool observations arrive inside <tool_response>...</tool_response>.
Use <think>...</think> for reasoning and <answer>...</answer> for the final answer.
Use tools to resolve uncertainty, then give a clear answer supported by the observations.'''


def build_system_prompt(*, allow_visit=True, local=False, answer_instruction=""):
    """Keep one base prompt; describe actual backend restrictions truthfully."""
    schemas = TOOL_SCHEMAS if allow_visit else TOOL_SCHEMAS[:1]
    prompt = (BASE_INSTRUCTION + "\n\n"
              "# Tools\n<tools>\n" + "\n".join(json.dumps(schema, ensure_ascii=False) for schema in schemas)
              + "\n</tools>\n\n" + TOOL_PROTOCOL)
    if local:
        prompt += ("\nUse only the fixed local document corpus. No external websites or network services are available."
                   " Results include document IDs; source URLs identify provenance."
                   " When reading a document, use its local://docid URL.")
    if not allow_visit:
        prompt += "\nOnly search is enabled for this run. Do not call visit."
    if answer_instruction:
        prompt += "\n" + answer_instruction.strip()
    return prompt


def initial_messages(question, **prompt_options):
    return [dict(role="system", content=build_system_prompt(**prompt_options)),
            dict(role="user", content=question)]


def prompt_metadata(system):
    return dict(prompt_version=PROMPT_VERSION if system.startswith(BASE_INSTRUCTION) else "custom",
                system_prompt_sha256=hashlib.sha256(system.encode()).hexdigest())
