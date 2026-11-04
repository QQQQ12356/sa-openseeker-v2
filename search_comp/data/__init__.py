"""数据管线：语料构建、BM25 检索、Search-R1 SFT 数据、Dataset/collator。"""

from importlib import import_module


def __getattr__(name):
    """Keep protocol/data inspection usable without loading retrieval or torch."""
    if name not in __all__:
        raise AttributeError(name)
    if name in {'BM25Retriever', 'build_corpus_from_hotpotqa', 'format_docs_as_reference'}:
        module = 'retrieval'
    elif name in {'SearchR1Collator', 'SearchR1SFTDataset'}:
        module = 'searchr1_dataset'
    else:
        module = 'trajectory'
    value = getattr(import_module(f'.{module}', __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "BM25Retriever",
    "build_corpus_from_hotpotqa",
    "format_docs_as_reference",
    "format_document_blocks",
    "parse_document_blocks",
    "SearchR1SFTDataset",
    "SearchR1Collator",
    "SEARCH_INSTRUCTION",
    "BASE_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "build_search_chat_prompt",
    "extract_search_query",
    "format_information_block",
]
