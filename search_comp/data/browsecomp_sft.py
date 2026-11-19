"""Normalize the two BrowseComp SFT exports to the Research training protocol."""
import json

from .agent_prompt import DEFAULT_SYSTEM_FILE, build_system_prompt
from .openseeker import CALL_RE
from .research import normalize_research


SOURCE = "browsecomp-sft"
REPO = "AmanPriyanshu/tool-reasoning-sft-RESEARCH-grill-lab-browsecomp-plus-runs-data-cleaned-rectified"
SHARDS = ("data.parquet", "syn_planner.parquet")
SYSTEM = DEFAULT_SYSTEM_FILE.read_text(encoding="utf-8")


def normalize_browsecomp_sft(row, source, index):
    messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    if [m["role"] for m in messages[:2]] != ["system", "user"]:
        raise ValueError("expected system and user messages")
    original = messages[1]["content"]
    question = row.get("canonical_question", original)
    planner = "data_source" in row

    def convert(match):
        call = json.loads(match[1])
        name = call["name"]
        args = call.get("arguments", call.get("parameters"))
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be an object")
        if name in {"search", "web_search"}:
            query = args.get("query", args.get("user_query"))
            name, args = "search", {"query": [query] if isinstance(query, str) else query}
        elif planner and name == "crawl_webpage":
            name, args = "visit", {"url": args["url"], "goal": question}
        else:
            raise ValueError(f"unsupported BrowseComp SFT tool: {name}")
        return "<tool_call>" + json.dumps(dict(name=name, arguments=args), ensure_ascii=False) + "</tool_call>"

    system = SYSTEM if planner else build_system_prompt(system=SYSTEM, allow_visit=False)
    converted = [dict(role="system", content=system), dict(role="user", content=question)]
    for message in messages[2:]:
        role, content = message["role"], message["content"]
        if role == "tool_call":
            content = CALL_RE.sub(convert, content)
        converted.append(dict(role=role, content=content))
    result = normalize_research(dict(row, messages=converted, question=question), source, index)
    result.update(upstream_source=row.get("data_source", row.get("category")),
                  upstream_qid=row.get("qid"), benchmark_question_id=row.get("benchmark_question_id"),
                  adapter_version="browsecomp-sft-v1")
    if original != question:
        result["original_question"] = original
    return result
