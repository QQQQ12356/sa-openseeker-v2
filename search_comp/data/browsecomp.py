"""Adapt reference-free BrowseComp SFT messages to the existing training protocol."""
from __future__ import annotations

import copy

from .openseeker import normalize


def normalize_browsecomp(row, source='browsecomp', index=0):
    """Use the demonstrated final response as a training-only target, never as gold."""
    raw = copy.deepcopy(row)
    messages = raw.get('messages')
    if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'assistant':
        raise ValueError('BrowseComp SFT requires messages ending in an assistant response')
    final = messages[-1].get('content')
    if not isinstance(final, str):
        raise ValueError('BrowseComp final response must be text')
    target = final.rsplit('</think>', 1)[-1].strip()
    if '<think>' in target or not target:
        raise ValueError('missing final response after reasoning')
    supplied_answer = raw.get('answer')
    raw['answer'] = supplied_answer or target
    escaped = 0
    for message in messages:
        if message.get('role') == 'tool' and isinstance(message.get('content'), str):
            # Literal protocol delimiters in a retrieved page must not close its wrapper.
            content = message['content']
            escaped += content.count('<tool_response>') + content.count('</tool_response>')
            message['content'] = content.replace('<tool_response>', '&lt;tool_response&gt;').replace(
                '</tool_response>', '&lt;/tool_response&gt;')
    result = normalize(raw, source, index)
    result['answer_source'] = 'reference' if supplied_answer else 'demonstration_final_response'
    result['training_only'] = True
    result['escaped_observation_delimiters'] = escaped
    return result
