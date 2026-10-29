"""Preserve training instructions; adapt only real evaluation capabilities."""
import hashlib
import json
from pathlib import Path
import re


PROMPT_VERSION = "openseeker-original-system-v1"
DEFAULT_SYSTEM_FILE = Path(__file__).with_name("openseeker_v1_system.txt")
TOOLS_RE = re.compile(r"<tools>(\s*\{.*?)</tools>", re.S)


def render_system(system, tools=None):
    """Render separately supplied schemas without rewriting the stored system."""
    if not tools or TOOLS_RE.search(system):
        return system
    from .openseeker import PROTOCOL
    schemas = [tool.get('function', tool) for tool in tools]
    return (system + '\n\n# Tools\n<tools>\n'
            + '\n'.join(json.dumps(schema, ensure_ascii=False, sort_keys=True) for schema in schemas)
            + '\n</tools>\n' + PROTOCOL)


def resolve_system_prompt(prompt_data=None, run_dir=None):
    """Explicit JSONL > run's training JSONL > bundled original Polar profile.

    Require a single effective system; never select an arbitrary row from mixed
    training instructions. Only system/tools are used, never questions or answers.
    """
    if prompt_data is None and run_dir is not None:
        spec = json.loads((Path(run_dir) / 'experiment.json').read_text())
        prompt_data = spec.get('training_config', {}).get('train_data_path')
        if not prompt_data:
            raise ValueError('RUN_DIR has no training prompt source; set --prompt-data explicitly')
    if prompt_data is None:
        return DEFAULT_SYSTEM_FILE.read_text(), str(DEFAULT_SYSTEM_FILE)
    path = Path(prompt_data).expanduser().resolve()
    system = None
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get('messages', row.get('trajectory'))
            if isinstance(messages, str):
                messages = json.loads(messages)
            if not messages or messages[0].get('role') != 'system':
                raise ValueError(f'prompt source requires an original system message: {path}')
            tools = row.get('tools') or []
            if isinstance(tools, str):
                tools = json.loads(tools)
            current = render_system(messages[0]['content'], tools)
            if not current.strip():
                raise ValueError(f'empty training system: {path}')
            if system is not None and current != system:
                raise ValueError('mixed training systems; provide --prompt-data with one matching prompt profile')
            system = current
    if system is None:
        raise ValueError(f'empty prompt source: {path}')
    return system, str(path)


def build_system_prompt(*, system=None, allow_visit=True, local=False, answer_instruction=""):
    """Online returns the original bytes. Local changes only tool capabilities."""
    prompt = DEFAULT_SYSTEM_FILE.read_text() if system is None else system
    if local or not allow_visit:
        match = TOOLS_RE.search(prompt)
        if match is None:
            raise ValueError('offline/capability adaptation requires a <tools> schema in the training system')
        decoder = json.JSONDecoder()
        body = match.group(1).strip()
        schemas = []
        while body:
            schema, end = decoder.raw_decode(body)
            name = schema.get('name')
            if name not in {'search', 'visit'}:
                raise ValueError(f'local backend does not implement training tool: {name}')
            if name == 'visit' and not allow_visit:
                body = body[end:].lstrip()
                continue
            if local:
                schema['description'] = (
                    'Search the fixed local corpus with batched queries; return ranked documents with docid and text.'
                    if name == 'search' else
                    'Read stored local://docid documents for the stated goal; return stored text without LLM summarization.')
            schemas.append(schema)
            body = body[end:].lstrip()
        if not schemas or not any(schema['name'] == 'search' for schema in schemas):
            raise ValueError('local backend requires the training search schema')
        prompt = (prompt[:match.start(1)] + '\n'
                  + '\n'.join(json.dumps(schema, ensure_ascii=False) for schema in schemas) + '\n'
                  + prompt[match.end(1):])
    if local:
        prompt += ('\nUse only the fixed local document corpus. No external websites or network services are available.'
                   ' Source URLs identify provenance; use local://docid URLs to read stored documents.')
    if not allow_visit:
        prompt += '\nOnly search is enabled for this run. Do not call visit.'
    if answer_instruction:
        prompt += '\n' + answer_instruction.strip()
    return prompt


def initial_messages(question, **prompt_options):
    return [dict(role='system', content=build_system_prompt(**prompt_options)),
            dict(role='user', content=question)]


def prompt_metadata(system):
    return dict(prompt_version=PROMPT_VERSION,
                system_prompt_sha256=hashlib.sha256(system.encode()).hexdigest())
