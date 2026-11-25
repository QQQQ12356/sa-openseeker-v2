"""Qwen-native BrowseComp-Plus turns, with structured browser messages."""
from collections import deque
import json
import re
import time

from search_comp.data.agent_prompt import prompt_metadata
from search_comp.offline.browsecomp_plus import ANSWER_INSTRUCTION
from search_comp.offline.native_browser import TOOLS


_TOOL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_FUNCTION = re.compile(r"<function=([\w.]+)>(.*?)(?:</function>|$)", re.S)
_PARAMETER = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.S)
_FINAL = re.compile(r"(?im)^\s*(?:\*\*)?(?:exact answer|final answer|answer)(?:\*\*)?\s*:")
_ALIASES = {"search": "browser.search", "web_search": "browser.search",
            "visit": "browser.open", "web_browse": "browser.open"}


def system_prompt(day):
    return (
        "You are a deep research agent. Alternate reasoning with tool calls to search and browse for evidence. "
        "Repeat searches when needed. Pages are labeled by cursor [n]; cite supporting lines as 【n†Lx-Ly】. "
        "Quote at most ten words from a tool result. The browser source is web.\n\n"
        f"{ANSWER_INSTRUCTION}\n\nToday's date: {day}\n\n"
        "For independent requests in a single turn, output consecutive "
        "<tool_call>...</tool_call> blocks, one function per block."
    )


def _tool_calls(block):
    import json5

    functions = _FUNCTION.findall(block)
    if functions:
        calls = []
        for name, body in functions:
            arguments = {}
            for key, value in _PARAMETER.findall(body):
                try:
                    arguments[key] = json5.loads(value.strip())
                except ValueError:
                    arguments[key] = value.strip()
            calls.append((name, arguments))
        return calls
    value = json5.loads(block)
    items = value if isinstance(value, list) else value.get("tool_calls", [value])
    calls = []
    for item in items:
        function = item.get("function", item)
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json5.loads(arguments)
        if not isinstance(function.get("name"), str) or not isinstance(arguments, dict):
            raise ValueError("invalid tool name or arguments")
        calls.append((function["name"], arguments))
    return calls


def parse_assistant(text, turn):
    text = text.split("<tool_response>", 1)[0].strip()
    reasoning = ""
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
    elif text.startswith("<think>"):
        return dict(role="assistant", content="", reasoning_content=text.removeprefix("<think>").strip())
    blocks = _TOOL.findall(text)
    if not blocks and "</tool_call>" in text:
        blocks = [text.split("</tool_call>", 1)[0]]
    calls = []
    for block in blocks:
        for name, args in _tool_calls(block):
            name = _ALIASES.get(name, name)
            query = args.get("query")
            if name == "browser.search" and isinstance(query, list):
                calls.extend((name, dict(args, query=item)) for item in query)
            else:
                calls.append((name, args))
    message = dict(role="assistant", content="" if blocks else text.strip(), reasoning_content=reasoning)
    if calls:
        message["parallel_tool_calls"] = len(calls) > 1
        message["tool_calls"] = [
            dict(id=f"{turn}_{i}" if len(calls) > 1 else str(turn), type="function",
                 function=dict(name=name, arguments=args))
            for i, (name, args) in enumerate(calls, 1)
        ]
    return message


def run_agent(messages, tokenizer, generate, browser, *, max_turns, max_context, max_new_tokens,
              observation_window):
    full = list(messages)
    context = full.copy() if observation_window else full
    recent = deque()
    stats, events = [], []
    status, answer, masked = "turn_limit", "", 0
    generate.start_question(messages[1]["content"])

    def append(message):
        full.append(message)
        if context is not full:
            context.append(message)

    for turn in range(1, max_turns + 1):
        if observation_window and len(recent) > observation_window:
            for index, function in recent.popleft():
                context[index] = dict(context[index], content=(
                    f"[Auto-archived browser result] tool={function['name']} | args: "
                    + json.dumps(function["arguments"], ensure_ascii=False)))
                masked += 1
        prompt = tokenizer.apply_chat_template(
            context, tools=list(TOOLS), tokenize=False, add_generation_prompt=True,
            save_history_reasoning_content=True,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        budget = min(max_new_tokens, max_context - len(ids))
        if budget <= 0:
            status = "context_limit"
            break
        start = time.monotonic()
        text, _, tokens = generate({"input_ids": ids}, budget,
                                   stop_texts=("\n<tool_response>", "<tool_response>", "<|im_end|>"))
        stats.append(dict(round=turn, input_tokens=len(ids), output_tokens=tokens,
                          latency_s=time.monotonic() - start))
        try:
            message = parse_assistant(text, turn)
        except (ValueError, TypeError, AttributeError) as error:
            append(dict(role="assistant", content=""))
            append(dict(role="tool", content=f"Error: Invalid arguments for function: {error}"))
            events.append(dict(kind="invalid_call", error=str(error)))
            continue
        append(message)
        calls = message.get("tool_calls", [])
        if not calls:
            visible = message["content"]
            if _FINAL.search(visible) or ("<answer>" in visible and "</answer>" in visible):
                from .tool_agent import extract_response
                answer = extract_response(visible)
                status = "completed" if answer else "empty_answer"
                break
            continue
        if sum(browser.counts.values()) + len(calls) > browser.max_calls:
            status = "tool_budget"
            break
        current = []
        for call, (reply, succeeded) in zip(calls, browser.execute(calls)):
            result = dict(role="tool", content=reply, tool_call_id=call["id"])
            index = len(context)
            append(result)
            if succeeded:
                current.append((index, call["function"]))
                if len(reply) > 12000:
                    # Keep the full observation in the saved trace.
                    if context is full:
                        context = full.copy()
                    context[index] = dict(result, content=reply[:12000] + "\n...[truncated]")
        if observation_window and current:
            recent.append(current)
        events.append(dict(calls=calls))

    result = dict(prediction=answer, status=status, messages=full, events=events,
                  **prompt_metadata(messages[0]["content"]), generation_config=generate.metadata,
                  turn_stats=stats, total_input_tokens=sum(row["input_tokens"] for row in stats),
                  generated_tokens=sum(row["output_tokens"] for row in stats), turns=len(events))
    if context is not full:
        result.update(final_messages=context, masked_observations=masked)
    return result
