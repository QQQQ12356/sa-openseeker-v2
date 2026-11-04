"""Adapt the local RESEARCH export to the shared OpenSeeker training protocol."""
import copy
import json
import re

from .openseeker import CALL_BEGIN, CALL_END, CALL_RE, PROTOCOL, normalize, parse_calls

ADAPTER_VERSION = 'research-role-export-v1'


def normalize_research(row, source='research', index=0):
    raw = copy.deepcopy(row)
    messages = raw.get('messages', raw.get('trajectory'))
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list) or not messages:
        raise ValueError('RESEARCH requires nonempty messages')
    out = []
    final = None
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError('RESEARCH messages must be objects')
        if final is not None:
            raise ValueError('messages after RESEARCH final answer')
        role, content = message.get('role'), message.get('content')
        if not isinstance(content, str):
            raise ValueError('RESEARCH requires text content')
        if role in ('reasoning', 'tool_call', 'answer'):
            if role == 'reasoning':
                if not content.strip().startswith('<think>'):
                    content = '<think>' + content + '</think>'
                if not content.strip().endswith('</think>'):
                    raise ValueError('unclosed RESEARCH reasoning')
            elif role == 'tool_call':
                if CALL_BEGIN in content or CALL_END in content:
                    raise ValueError('unexpected RESEARCH call envelope')
                if not CALL_RE.findall(content):
                    raise ValueError('missing RESEARCH tool call')
                content = CALL_BEGIN + '\n' + content.strip() + '\n' + CALL_END
                for call in parse_calls(content):
                    name, args = call['name'], call['arguments']
                    if name == 'search':
                        queries = args.get('query')
                        if not isinstance(queries, list) or not queries or any(
                                not isinstance(q, str) or not q.strip() for q in queries):
                            raise ValueError('RESEARCH search requires nonempty query array')
                    elif name == 'visit':
                        urls = args.get('url')
                        urls = [urls] if isinstance(urls, str) else urls
                        if (not isinstance(urls, list) or not urls or any(
                                not isinstance(u, str) or not u.strip() for u in urls)
                                or not isinstance(args.get('goal'), str) or not args['goal'].strip()):
                            raise ValueError('RESEARCH visit requires url and goal')
                    else:
                        raise ValueError(f'unsupported RESEARCH tool: {name}')
            else:
                match = re.fullmatch(r'\s*<answer>(.*?)</answer>\s*', content, re.S)
                final = (match[1] if match else content).strip()
                if not final:
                    raise ValueError('empty RESEARCH final answer')
            # Reasoning and its action belong to one assistant turn. Splitting them
            # would train an end-of-turn token before the action at inference time.
            if out and out[-1]['role'] == 'assistant':
                if parse_calls(out[-1]['content']):
                    raise ValueError('RESEARCH tool call without observation')
                out[-1]['content'] += '\n' + content
            else:
                out.append(dict(role='assistant', content=content))
        elif role == 'tool_output':
            match = re.fullmatch(r'\s*<tool_response>(.*?)</tool_response>\s*', content, re.S)
            body = match[1] if match else content
            body = body.replace('<tool_response>', '&lt;tool_response&gt;').replace(
                '</tool_response>', '&lt;/tool_response&gt;')
            out.append(dict(role='tool', content=body))
        elif role in ('system', 'user'):
            if role == 'system':
                # Replace only the conflicting export protocol; preserve actual
                # research instructions and embedded search/visit schemas.
                marker = 'You are a tool-augmented QA agent.'
                if content.startswith('## Tool-Use Protocol'):
                    if marker not in content:
                        raise ValueError('unknown RESEARCH system protocol')
                    content = content[content.index(marker):]
                content += '\n\n' + PROTOCOL + '\nReturn the final answer inside <answer>...</answer>.'
            out.append(dict(role=role, content=content))
        else:
            raise ValueError(f'unsupported RESEARCH role: {role}')
    if messages[-1].get('role') != 'answer' or final is None:
        raise ValueError('RESEARCH export must end with answer')
    raw['messages'] = out
    raw.pop('trajectory', None)
    reference = raw.get('answer')
    raw['answer'] = reference or final
    result = normalize(raw, source, index)
    result.update(adapter_version=ADAPTER_VERSION, training_only=True,
                  answer_source='reference' if reference else 'demonstration_final_response')
    return result
