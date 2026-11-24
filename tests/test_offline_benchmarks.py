"""Fixed-corpus regression: real index, upstream metrics, subprocesses and native tool loops."""
import base64
import bz2
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from search_comp.offline.assets import digest, install_official, verify_official
from search_comp.offline.data import document, load_official_functions, prepare, read_qrels, records
from search_comp.offline.index import LocalIndex, build_index
from search_comp.offline.scoring import load_predictions, score_browsecomp, score_hotpot
from search_comp.offline.tools import LocalTools
from search_comp.workflows import offline_benchmark as workflow

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return path


class CharacterTokenizer:
    def encode(self, value, **kwargs):
        return list(value.encode('utf-8'))

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode('utf-8', errors='ignore')


@pytest.mark.parametrize("window", [0, 1])
def test_observation_masking_changes_model_context_and_preserves_trace(window):
    from search_comp.evaluation.plain_generation import PlainGenerationState
    from search_comp.evaluation.tool_agent import ModelGenerator, Registry, run_agent

    tokenizer = CharacterTokenizer()

    def reply(*queries):
        return "<tool_calls_begin>" + "".join(
            '<tool_call>' + json.dumps(dict(name="search", arguments={"query": query})) + '</tool_call>'
            for query in queries) + "</tool_calls_end>"

    class Generator:
        def __init__(self):
            self.plain_state = PlainGenerationState(tokenizer)
            self.responses = iter([reply("early-one", "early-two"), reply("late"), "<answer>OK</answer>"])
            self.inputs = []

        encode_turn = ModelGenerator.encode_turn
        invalidate_history = ModelGenerator.invalidate_history

        def __call__(self, encoded, budget):
            self.inputs.append(tokenizer.decode(encoded["input_ids"]))
            return next(self.responses), True, 1

    generate = Generator()
    generate.tokenizer = tokenizer
    tools = Registry({"search": lambda query: "BODY:" + query})
    result = run_agent([dict(role="system", content="Test"), dict(role="user", content="question")],
                       tokenizer, generate, tools, observation_window=window, record_turn_stats=True)
    assert result["prediction"] == "OK"
    assert "BODY:early-one" in generate.inputs[1]
    assert "BODY:late" in generate.inputs[2]
    assert ("BODY:early-one" in generate.inputs[2]) == (window == 0)
    assert any("BODY:early-two" in message["content"] for message in result["messages"])
    assert len(result["turn_stats"]) == 3
    if window:
        assert generate.inputs[2].count("[Auto-archived search result]") == 2
        assert result["masked_observations"] == 2
    else:
        assert "final_messages" not in result


def test_bcp_openai_judge_uses_reasoning_model_request_format(monkeypatch):
    from search_comp.evaluation import judging

    def request_body(request, timeout):
        assert request.full_url == "https://judge.invalid/v1/chat/completions"
        assert timeout == 30
        assert json.loads(request.data) == dict(model="gpt-5-mini",
            messages=[dict(role="user", content="Prompt")], stream=False, max_completion_tokens=4096)
        reply = dict(choices=[dict(finish_reason="stop", message=dict(content="correct: yes"))])
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(judging.urllib.request, "urlopen", request_body)
    judge = judging.ChatJudge("https://judge.invalid/v1", "gpt-5-mini", "test-key",
                             max_tokens=4096, timeout=30, openai_api=True)
    assert judge([dict(role="user", content="Prompt")]) == "correct: yes"


@pytest.mark.parametrize(("response", "answer"), [
    ("Explanation: The article says so [20].\n**Exact Answer:** Paris\nConfidence: 90%", "Paris"),
    ("**Final Answer**: Paris\nConfidence: 90%", "Paris"),
    ("Paris", "Paris"),
    ("Explanation: The passage is ambiguous.\nConfidence: 20%", ""),
    ("Exact Answer:\nConfidence: 20%", ""),
    ("Exact Answer: Paris\nExact Answer: London", ""),
])
def test_bcp_answer_label_excludes_explanation_confidence_and_conflicts(response, answer):
    from search_comp.offline.browsecomp_plus import extract_exact_answer

    assert extract_exact_answer(response) == answer


def test_bcp_shared_prompt_is_independent_of_training_run(tmp_path):
    args = SimpleNamespace(benchmark="browsecomp-plus", prompt_profile="benchmark",
                           prompt_data=None, run_dir=tmp_path / "absent", allow_visit=True)
    system, source = workflow.agent_prompt(args)
    args.run_dir = None
    assert workflow.agent_prompt(args) == (system, source)
    assert 'browsecomp-plus/benchmark-v1' in source
    assert 'Exact Answer:' in system and 'Confidence:' in system and 'local://docid' in system


@pytest.mark.parametrize(("existing_run", "score_after_generation", "stages"), [
    (False, True, ["evaluate", "score"]),
    (False, False, ["evaluate"]),
    (True, False, ["score"]),
])
def test_bcp_launcher_routes_generation_and_scoring_independently(
        tmp_path, monkeypatch, existing_run, score_after_generation, stages):
    from search_comp.experiments import benchmarks

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps(dict(model_type="qwen3_5", text_config=dict(
        hidden_size=2560, intermediate_size=9216, num_hidden_layers=32))))
    evaluation_dir = tmp_path / "evaluation"
    calls = []

    def evaluate(**settings):
        calls.append("evaluate")
        assert settings["generation"] == {"retriever": "bm25"}
        return evaluation_dir

    def score(**settings):
        calls.append("score")
        assert settings["evaluation_dir"] == evaluation_dir
        return evaluation_dir / "scores" / "test"

    monkeypatch.setattr(benchmarks, "evaluate", evaluate)
    monkeypatch.setattr(benchmarks, "score", score)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = (dict(judge_backend="local", judge_model=str(base), judge_url="", judge_timeout=30,
                  judge_attempts=3, judge_max_tokens=4096) if "score" in stages else {})
    benchmarks.run_browsecomp_plus(
        method="baseline", benchmark_dir=tmp_path, training_run=None,
        model_path=None if existing_run else base, checkpoint=None, prompt_data=None,
        run_name="test", overlap="off", overlap_threshold=0.30, resume_from=None,
        evaluation_dir=evaluation_dir if existing_run else None,
        score_after_generation=score_after_generation, generation={"retriever": "bm25"},
        judge=judge, environment={})
    assert calls == stages


def hotpot_sources(tmp_path):
    data = tmp_path / 'dev.json'
    data.write_text(json.dumps([
        dict(_id='h1', question='Which bridge is red?', answer='red blue',
             supporting_facts=[['Bridge', 0]], context=[['REFERENCE_CONTEXT_DO_NOT_PROMPT', ['LEAK']]]),
        dict(_id='h2', question='Is the bridge open?', answer='yes', supporting_facts=[['Bridge', 1]])]))
    corpus = write_jsonl(tmp_path / 'wiki.jsonl', [
        dict(id='1', title='Bridge', url='https://wiki.invalid/bridge', text=[['Bridge'],
             ['A red blue bridge. ', 'It is open.']]),
        dict(id='2', title='Mountain', text=[['A snowy mountain.']])])
    return data, corpus


def prepared_hotpot(tmp_path):
    data, corpus = hotpot_sources(tmp_path)
    root = tmp_path / 'benchmark'
    install_official(root, 'hotpotqa')
    prepare('hotpotqa', root, data, corpus)
    build_index(root)
    return root


def prepared_bcp(tmp_path, count=2):
    root = tmp_path / 'benchmark'
    install_official(root, 'browsecomp-plus')
    evidence = read_qrels(root / 'official/topics-qrels/qrel_evidence.txt')
    gold = read_qrels(root / 'official/topics-qrels/qrel_golds.txt')
    ids = list(evidence)[:count]
    docs = sorted(set().union(*(evidence[q] | gold[q] for q in ids)))
    data = write_jsonl(tmp_path / 'queries.jsonl', [dict(query_id=q, query='synthetic question ' + q,
         answer='REFERENCE_DO_NOT_PROMPT', gold_docs=[], evidence_docs=[]) for q in ids])
    corpus = write_jsonl(tmp_path / 'corpus.jsonl', [dict(docid=d, text='A red bridge with evidence ' + d,
          url='https://invalid.example/' + d) for d in docs])
    prepare('browsecomp-plus', root, data, corpus)
    build_index(root)
    return root


def test_bundled_provenance_and_tamper_detection(tmp_path):
    metadata = install_official(tmp_path, 'hotpotqa')
    assert metadata['official_revision'] == '3635853403a8735609ee997664e1528f4480762a'
    verify_official(tmp_path, 'hotpotqa')
    path = tmp_path / 'official/hotpot_evaluate_v1.py'
    path.write_text(path.read_text() + '\n# changed\n')
    with pytest.raises(ValueError, match='changed'):
        verify_official(tmp_path, 'hotpotqa')


def test_hotpot_adapter_drops_reference_context_and_preserves_sentences(tmp_path):
    root = prepared_hotpot(tmp_path)
    rows = list(records(root / 'prepared/questions.jsonl'))
    assert 'REFERENCE_CONTEXT' not in json.dumps(rows)
    assert rows[0]['supporting_facts'] == [['Bridge', 0]]
    with LocalIndex(root / 'index.sqlite', verify=True) as index:
        doc = index.get('1')
        assert doc['sentences'] == ['A red blue bridge. ', 'It is open.']
        assert doc['text'] == 'A red blue bridge. It is open.'
    metadata = json.loads((root / 'prepared/manifest.json').read_text())
    assert not metadata['canonical_corpus']
    with pytest.raises(FileExistsError):
        prepare('hotpotqa', root, tmp_path / 'dev.json', tmp_path / 'wiki.jsonl')


def test_parquet_encrypted_bcp_uses_official_canary(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path / 'benchmark'
    install_official(root, 'browsecomp-plus')
    funcs = load_official_functions(root / 'official/scripts_build_index/decrypt_dataset.py', {'DEFAULT_CANARY'})
    key = hashlib.sha256(funcs['DEFAULT_CANARY'].encode()).digest()
    def enc(value):
        return base64.b64encode(bytes(c ^ key[i % 32] for i, c in enumerate(value.encode()))).decode()
    evidence = read_qrels(root / 'official/topics-qrels/qrel_evidence.txt')
    gold = read_qrels(root / 'official/topics-qrels/qrel_golds.txt')
    qid = next(iter(evidence))
    source = tmp_path / 'queries.parquet'
    pq.write_table(pa.Table.from_pylist([dict(query_id=qid, query=enc('中文 question'), answer=enc('answer'),
        evidence_docs=[dict(docid=enc(d), text=enc('do not prompt'), url=enc('url')) for d in evidence[qid]])]), source)
    corpus = write_jsonl(tmp_path / 'corpus.jsonl', [dict(docid=d, text='local text') for d in evidence[qid] | gold[qid]])
    prepare('browsecomp-plus', root, source, corpus, encrypted=True)
    rows = list(records(root / 'prepared/questions.jsonl'))
    assert rows[0]['query'] == '中文 question'
    assert rows[0]['answer'] == 'answer'
    assert set(rows[0]['evidence_docs']) == evidence[qid]
    assert 'do not prompt' not in json.dumps(rows)


def test_bcp_official_lucene_ranks_original_ids_and_checks_assets(tmp_path, monkeypatch):
    from search_comp.offline.assets import BCP_LUCENE_DIR
    from search_comp.offline.index import LuceneIndex, lucene_metadata

    root = prepared_bcp(tmp_path)
    directory = root / BCP_LUCENE_DIR
    directory.mkdir(parents=True)
    segments = directory / "segments_3"
    segments.write_bytes(b"synthetic segments")
    (directory / "write.lock").touch()
    assets = dict(index_revision="test", bm25_files={segments.name: digest(segments)})
    assert lucene_metadata(root, assets, verify=True)["revision"] == "test"
    segments.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum changed"):
        lucene_metadata(root, assets, verify=True)

    class Searcher:
        def __init__(self, path):
            assert path == str(directory)
            self.num_docs = db.manifest["count"]

        def search(self, query, *, k):
            assert (query, k) == ("find evidence", 15)
            return [SimpleNamespace(docid=docid)]

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "pyserini.search.lucene", SimpleNamespace(LuceneSearcher=Searcher))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with LocalIndex(root / "index.sqlite") as db:
        docid = db.search("bridge", 1)[0]["docid"]
        index = LuceneIndex(db, directory)
        assert index.search("find evidence", 15) == [db.get(docid)]
        index.close()
    import os
    assert "OPENAI_API_KEY" not in os.environ


def test_bcp_official_document_uses_frontmatter_title_and_body():
    raw = dict(docid=42, url="https://example.invalid/", title="separate field",
               text="---\ntitle: Article title\nauthor: An author\ndate: 2026-01-01\n---\n Body text. ")
    parsed = document(raw, "browsecomp-plus", aligned_parquet=True)
    assert (parsed["id"], parsed["title"], parsed["text"]) == ("42", "Article title", "Body text.")


def test_hotpot_prepare_skips_empty_bodies_and_preserves_valid_sentences(tmp_path, capsys):
    data, corpus = hotpot_sources(tmp_path)
    title = 'Albert Alcibiades, Margrave of Brandenburg-Kulmbach'
    empty_texts = [[title], [[title]], [[title], []], [[title], ['  ', '<br/>']]]
    rows = [dict(id=str(i), title=title, text=text) for i, text in enumerate(empty_texts)]
    rows.append(dict(id='valid', title='Bridge', text=[['Bridge'], ['First. ', '', 'Third.']]))
    write_jsonl(corpus, rows)
    root = tmp_path / 'benchmark'
    install_official(root, 'hotpotqa')
    manifest = prepare('hotpotqa', root, data, corpus, corpus_limit=1)
    assert manifest['documents'] == 1
    assert manifest['skipped_documents'] == 4
    skip_path = root / 'prepared/skipped_documents.jsonl'
    assert manifest['skipped_documents_sha256'] == digest(skip_path)
    skipped = list(records(skip_path))
    assert [row['id'] for row in skipped] == ['0', '1', '2', '3']
    assert [row['record_index'] for row in skipped] == [1, 2, 3, 4]
    assert all(row['title'] == title for row in skipped)
    documents = list(records(root / 'prepared/corpus.jsonl'))
    assert [row['id'] for row in documents] == ['valid']
    assert documents[0]['sentences'] == ['First. ', '', 'Third.']
    build_index(root)
    with LocalIndex(root / 'index.sqlite') as index:
        assert index.get('valid')['sentences'][2] == 'Third.'
    assert 'skipped_empty=4' in capsys.readouterr().out


def test_hotpot_prepare_does_not_skip_malformed_text(tmp_path):
    data, corpus = hotpot_sources(tmp_path)
    write_jsonl(corpus, [dict(id='bad', title='Bridge', text='unexpected string')])
    root = tmp_path / 'benchmark'
    install_official(root, 'hotpotqa')
    with pytest.raises(ValueError, match='sentence arrays'):
        prepare('hotpotqa', root, data, corpus)
    assert not (root / 'prepared/manifest.json').exists()


def test_hotpot_prepare_rejects_all_empty_corpus(tmp_path):
    data, corpus = hotpot_sources(tmp_path)
    write_jsonl(corpus, [dict(id='empty', title='Bridge', text=['Bridge'])])
    root = tmp_path / 'benchmark'
    install_official(root, 'hotpotqa')
    with pytest.raises(ValueError, match='questions/corpus must be nonempty'):
        prepare('hotpotqa', root, data, corpus)
    assert not (root / 'prepared/manifest.json').exists()


def test_tar_bz2_stream_without_extracting_paths(tmp_path):
    row = dict(id=1, title='Bridge', text=[['<a href="x">A bridge</a> exists.']])
    compressed = bz2.compress((json.dumps(row) + '\n').encode())
    archive = tmp_path / 'wiki.tar.bz2'
    with tarfile.open(archive, 'w:bz2') as target:
        info = tarfile.TarInfo('../../outside/wiki_00.bz2')
        info.size = len(compressed)
        target.addfile(info, io.BytesIO(compressed))
    actual = list(records(archive))
    assert actual == [row]
    assert document(actual[0], 'hotpotqa')['text'] == 'A bridge exists.'
    assert not (tmp_path.parent / 'outside').exists()


def test_index_persistent_rank_order_literal_query_and_read_only(tmp_path):
    root = prepared_hotpot(tmp_path)
    with LocalIndex(root / 'index.sqlite') as index:
        assert index.search('snowy mountain', 1)[0]['docid'] == '2'
        assert index.search('bridge', 1)[0]['docid'] == '1'
        assert index.search('" OR *** --', 5) == []
        assert index.search('!!!', 5) == []
        with pytest.raises(sqlite3.OperationalError):
            index.connection.execute('DELETE FROM documents')
    result = subprocess.run([sys.executable, '-c',
        'from search_comp.offline.index import LocalIndex; import sys; '
        'i=LocalIndex(sys.argv[1]); print(i.search("bridge",1)[0]["docid"]); i.close()', str(root / 'index.sqlite')],
        cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '1'
    with pytest.raises(FileExistsError):
        build_index(root)


def test_duplicate_corpus_ids_refuse_index_publication(tmp_path):
    root = prepared_hotpot(tmp_path)
    # Create a separate prepared corpus rather than modifying an existing index.
    duplicate = tmp_path / 'duplicates'
    duplicate.mkdir()
    import shutil
    shutil.copytree(root / 'prepared', duplicate / 'prepared')
    path = duplicate / 'prepared/corpus.jsonl'
    path.write_text(path.read_text() * 2)
    manifest_path = duplicate / 'prepared/manifest.json'
    metadata = json.loads(manifest_path.read_text())
    metadata.update(corpus_sha256=digest(path), documents=4)
    manifest_path.write_text(json.dumps(metadata))
    with pytest.raises(sqlite3.IntegrityError):
        build_index(duplicate)
    assert not (duplicate / 'index.sqlite').exists()
    assert (duplicate / 'index.building.sqlite').exists()


def test_local_tools_clip_trace_reset_and_no_external_visit(tmp_path):
    root = prepared_hotpot(tmp_path)
    with LocalIndex(root / 'index.sqlite') as index:
        tools = LocalTools(index, CharacterTokenizer(), topk=1, snippet_tokens=12, read_tokens=15)
        results = json.loads(tools.search(['bridge']))
        assert len(results[0]['results'][0]['text'].encode()) <= 12
        assert tools.retrieved == {'1'}
        assert tools.trace == [dict(query='bridge', ranked_docids=['1'])]
        assert len(json.loads(tools.visit('local://1', 'evidence'))[0]['text'].encode()) <= 15
        assert tools.visited == {'1'}
        with pytest.raises(ValueError, match='local://'):
            tools.visit('https://example.com', 'goal')
        tools.reset()
        assert not tools.retrieved and not tools.trace and not tools.visited
        tools.allow_visit = False
        with pytest.raises(ValueError, match='disabled'):
            tools.visit('local://1', 'goal')


def test_tool_call_budget_counts_batches(tmp_path):
    from search_comp.evaluation.tool_agent import ToolError
    root = prepared_hotpot(tmp_path)
    with LocalIndex(root / 'index.sqlite') as index:
        tools = LocalTools(index, CharacterTokenizer(), max_calls=1)
        tools.execute([dict(name='search', arguments=dict(query=['bridge']))])
        with pytest.raises(ToolError, match='budget'):
            tools.execute([dict(name='visit', arguments=dict(url='local://1', goal='goal'))])


@pytest.mark.parametrize('prediction,gold,expected', [('The RED!', 'red', 1.0), ('yes indeed', 'yes', 0.0),
    ('noanswer', 'no', 0.0), ('red', 'red blue', 2 / 3)])
def test_official_hotpot_normalization_and_yes_no_edges(tmp_path, prediction, gold, expected):
    install_official(tmp_path, 'hotpotqa')
    rows = [dict(id='q', query='question', answer=gold, supporting_facts=[['Bridge', 0]])]
    output = tmp_path / 'score'
    output.mkdir()
    metrics = score_hotpot(tmp_path, rows, {'q': dict(status='completed', final_response=prediction)}, output)
    assert metrics['f1'] == pytest.approx(expected)
    assert metrics['supporting_fact_predictions'] == 0


def test_hotpot_missing_failed_and_real_supporting_facts(tmp_path):
    root = prepared_hotpot(tmp_path)
    rows, _ = workflow.prepared_questions(root, 'hotpotqa')
    output = tmp_path / 'scores'
    output.mkdir()
    predictions = {'h1': dict(status='completed', final_response=json.dumps(dict(answer='red blue', sp=[['Bridge', 0]])))}
    metrics = score_hotpot(root, rows, predictions, output)
    assert metrics['em'] == metrics['f1'] == metrics['sp_f1'] == metrics['joint_f1'] == 0.5
    exported = json.loads((output / 'hotpot_predictions.json').read_text())
    assert 'h2' not in exported['answer']
    assert exported['sp']['h1'] == [['Bridge', 0]]
    output2 = tmp_path / 'scores2'
    output2.mkdir()
    metrics2 = score_hotpot(root, rows, {'h1': dict(status='tool_error', final_response='red blue')}, output2)
    assert metrics2['em'] == metrics2['f1'] == 0


def test_bcp_official_prompt_judge_recall_and_missing_denominator(tmp_path):
    root = prepared_bcp(tmp_path)
    rows, _ = workflow.prepared_questions(root, 'browsecomp-plus')
    response = ('Explanation: verified [' + rows[0]['evidence_docs'][0] + ']\n'
                'Exact Answer: REFERENCE_DO_NOT_PROMPT\nConfidence: 90%')
    predictions = {rows[0]['id']: dict(status='completed', final_response=response,
          exact_answer='REFERENCE_DO_NOT_PROMPT',
          retrieved_docids=rows[0]['evidence_docs'], tool_call_counts={'search': 1})}
    prompts = []
    def judge(prompt):
        prompts.append(prompt)
        return 'extracted_final_answer: answer\nreasoning: matches\ncorrect: yes\nconfidence: 90'
    output = tmp_path / 'scores'
    output.mkdir()
    metrics = score_browsecomp(root, rows, predictions, output, judge, judge_preflight=False)
    assert metrics['accuracy'] == metrics['evidence_recall'] == 0.5
    assert metrics['em'] == metrics['f1'] == 0.5
    assert metrics['judged_count'] == 1 and metrics['judged_accuracy'] == 1.0
    assert metrics['avg_tool_stats']['search'] == 0.5
    assert len(prompts) == 1 and response in prompts[0]
    assert metrics['complete']
    assert not metrics['calibration_available']
    assert metrics['citation_coverage'] == 0.5
    assert len(list((output / 'official_runs').glob('*.json'))) == 2


def test_bcp_unknown_judge_is_not_a_wrong_label_or_removed_question(tmp_path):
    root = prepared_bcp(tmp_path)
    rows, _ = workflow.prepared_questions(root, 'browsecomp-plus')
    predictions = {r['id']: dict(status='completed', final_response='prediction') for r in rows}
    output = tmp_path / 'score'
    output.mkdir()
    metrics = score_browsecomp(root, rows, predictions, output, lambda _: 'malformed judge output',
                               judge_preflight=False)
    assert metrics['accuracy'] == 0
    assert metrics['judge_errors'] == 2 and not metrics['complete']


def test_prediction_id_validation(tmp_path):
    rows = [dict(id='q', query='question')]
    for values in ([dict(id='foreign', status='completed', prediction='x')],
                   [dict(id='q', prediction='x')],
                   [dict(id='q', status='completed', prediction='x')] * 2):
        with pytest.raises(ValueError):
            load_predictions(write_jsonl(tmp_path / 'predictions.jsonl', values), rows)


def test_offline_guard_blocks_dns_and_ip_connections_in_child():
    source = '''from search_comp.offline.assets import enforce_offline
import socket
enforce_offline()
for operation in (lambda:socket.getaddrinfo('example.com',443), lambda:socket.socket().connect(('127.0.0.1',1))):
    try: operation()
    except RuntimeError: pass
    else: raise AssertionError('network allowed')
print('blocked')
'''
    result = subprocess.run([sys.executable, '-c', source], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'blocked'




@pytest.mark.parametrize('kind', ['plain', 'beacon'])
def test_native_real_agent_search_visit_answer_and_regions(tmp_path, monkeypatch, kind):
    root = prepared_hotpot(tmp_path)
    model_path = tmp_path / 'checkpoint'
    model_path.mkdir()
    output = tmp_path / 'run'
    output.mkdir()
    model = SimpleNamespace(eval=lambda: None)
    tokenizer = CharacterTokenizer()
    from search_comp.data.agent_prompt import build_system_prompt
    prompt_data = tmp_path / "training.jsonl"
    prompt_data.write_text(json.dumps(dict(messages=[dict(role="system",
        content=build_system_prompt() + "\nKeep original training instruction.")],
        answer="TRAINING_ANSWER_DO_NOT_COPY")) + "\n")
    monkeypatch.setitem(sys.modules, 'search_comp.models.model_loader', SimpleNamespace(
        load_tokenizer=lambda _: tokenizer, load_beacon_model=lambda _: model))
    monkeypatch.setitem(sys.modules, 'search_comp.models.plain_qwen3', SimpleNamespace(load_sft_model=lambda _: model))
    from search_comp.evaluation import tool_agent
    generator_kinds, saw_regions = [], []
    def generator(m, tok, beacon):
        generator_kinds.append(beacon)
        def generate(encoded, budget):
            assert 'REFERENCE_ANSWER_DO_NOT_PROMPT' not in tok.decode(encoded['input_ids'])
            assert 'REFERENCE_CONTEXT' not in tok.decode(encoded['input_ids'])
            assert 'TRAINING_ANSWER' not in tok.decode(encoded['input_ids'])
            assert 'Keep original training instruction.' in tok.decode(encoded['input_ids'])
            assert 'web searches' not in tok.decode(encoded['input_ids'])
            saw_regions.append(bool(encoded['regions']))
            turn = encoded['n_turns']
            if turn == 0:
                answer = '<tool_calls_begin><tool_call>{"name":"search","arguments":{"query":["bridge"]}}</tool_call></tool_calls_end>'
            elif turn == 1:
                answer = '<tool_calls_begin><tool_call>{"name":"visit","arguments":{"url":"local://1","goal":"read"}}</tool_call></tool_calls_end>'
            else:
                answer = '<answer>{"answer":"red blue","sp":[["Bridge",0]]}</answer>'
            return answer, True, len(answer)
        return generate
    monkeypatch.setattr(tool_agent, 'ModelGenerator', generator)
    args = workflow.make_parser().parse_args(['run', '--benchmark', 'hotpotqa', '--root', str(root),
          '--model', str(model_path), '--kind', kind, '--allow-visit', '--prompt-data', str(prompt_data)])
    rows, manifest = workflow.prepared_questions(root, 'hotpotqa', 1)
    rows[0]['answer'] = 'REFERENCE_ANSWER_DO_NOT_PROMPT'
    workflow.run_phase(args, output, rows, manifest)
    predictions = list(records(output / 'predictions.jsonl'))
    assert predictions[0]['status'] == 'completed'
    assert predictions[0]['prompt_source'] == str(prompt_data)
    assert predictions[0]['visited_docids'] == ['1']
    assert predictions[0]['tool_call_counts'] == dict(search=1, visit=1)
    assert generator_kinds == [kind == 'beacon']
    assert saw_regions == [False, True, True]
    assert predictions[0]['answer'] == rows[0]['answer']




def test_invalid_supporting_facts_do_not_change_official_answer_score(tmp_path):
    root = prepared_hotpot(tmp_path)
    rows, _ = workflow.prepared_questions(root, 'hotpotqa', 1)
    output = tmp_path / 'score'
    output.mkdir()
    predictions = {'h1': dict(status='completed', final_response=json.dumps(dict(answer='red blue', sp=[['Bridge', -1]])))}
    metrics = score_hotpot(root, rows, predictions, output)
    assert metrics['em'] == metrics['f1'] == 1.0
    assert metrics['supporting_fact_predictions'] == 0
    assert json.loads((output / 'per_question.json').read_text())[0]['supporting_parse_error']




def test_dataset_and_index_fingerprints_checked(tmp_path):
    root = prepared_hotpot(tmp_path)
    path = root / 'prepared/questions.jsonl'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='checksum'):
        workflow.prepared_questions(root, 'hotpotqa')
    index_path = root / 'index.sqlite'
    # A harmless trailing byte still changes the artifact's provenance hash.
    with index_path.open('ab') as target:
        target.write(b'x')
    with pytest.raises(ValueError, match='checksum'):
        LocalIndex(index_path, verify=True)


def test_local_judge_loads_and_generates_from_real_tiny_hf_checkpoint(tmp_path):
    """Exercise the actual offline HF judge loader/template/generate path, no downloaded weights."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from search_comp.offline.scoring import LocalJudge
    backend = Tokenizer(WordLevel({'<eos>': 0, '<pad>': 1, '<unk>': 2, 'question': 3}, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>',
                                       pad_token='<pad>', unk_token='<unk>')
    tokenizer.chat_template = '{{ messages[0]["content"] }}'
    config = Qwen3Config(vocab_size=4, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                        eos_token_id=0, pad_token_id=1, attention_dropout=0.0)
    model = Qwen3ForCausalLM(config)
    # A zero-logit model greedily emits token 0 (EOS), making the inference deterministic.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    judge = LocalJudge(tmp_path, max_tokens=4, max_context=32, temperature=0)
    with pytest.raises(ValueError, match='empty_judge_content'):
        judge('question')
    judge.max_context = 1
    with pytest.raises(ValueError, match='context budget'):
        judge('question')








@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'wrong_id', 'wrong_index', 'checksum', 'manifest'])
def test_parallel_merge_rejects_bad_shards(tmp_path, corruption):
    rows = [dict(id=str(i), query='q' + str(i)) for i in range(5)]
    manifest = {'questions_sha256': 'questions-fingerprint'}
    paths = []
    for rank in range(2):
        path = tmp_path / f'worker_{rank}' / 'predictions.jsonl'
        predictions = [dict(**row, sample_index=i, status='completed')
                       for i, row in enumerate(rows) if i % 2 == rank]
        if rank == 0:
            if corruption == 'missing':
                predictions.pop()
            elif corruption == 'duplicate':
                predictions.append(predictions[0])
            elif corruption == 'wrong_id':
                predictions[0]['id'] = 'other'
            elif corruption == 'wrong_index':
                predictions[0]['sample_index'] = 1
        write_jsonl(path, predictions)
        metadata = dict(status='complete', **manifest, predictions_sha256=digest(path),
                        selected_ids=[row['id'] for row in rows[rank::2]])
        if rank == 0 and corruption == 'manifest':
            metadata['selected_ids'] = []
        if rank == 0 and corruption == 'checksum':
            metadata['predictions_sha256'] = 'tampered'
        (path.parent / 'manifest.json').write_text(json.dumps(metadata))
        paths.append(path)
    with pytest.raises(ValueError, match='worker'):
        workflow.merge_worker_predictions(paths, tmp_path, rows, manifest)
    assert not (tmp_path / 'predictions.jsonl').exists()
    assert not (tmp_path / 'predictions.building.jsonl').exists()






def test_bcp_preflight_and_bounded_retry_preserve_attempts(tmp_path):
    root = prepared_bcp(tmp_path)
    rows, _ = workflow.prepared_questions(root, 'browsecomp-plus')
    predictions = {rows[0]['id']: dict(status='completed', final_response='prediction')}
    calls = []
    def judge(prompt):
        calls.append(prompt)
        if '[response]: Tokyo' in prompt:
            label = 'no'
        elif '[response]: prediction' in prompt and calls.count(prompt) == 1:
            return 'correct: yes\ncorrect: no'
        else:
            label = 'yes'
        return f'extracted_final_answer: value\nreasoning: compared\ncorrect: {label}\nconfidence: 90'
    output = tmp_path / 'scores'; output.mkdir()
    metrics = score_browsecomp(root, rows, predictions, output, judge, judge_attempts=2)
    assert metrics['complete'] and metrics['accuracy'] == 0.5
    report = json.loads((output / 'judge_preflight.json').read_text())
    assert report['passed']
    details = json.loads((output / 'per_question.json').read_text())
    assert len(details[0]['judge_attempts']) == 2
    assert not details[0]['judge_attempts'][0]['valid']
    assert details[1]['score_source'] == 'empty_or_failed_response_rule'


@pytest.mark.parametrize('raw', [
    'extracted_final_answer: a\nreasoning: compared\ncorrect: yes\nconfidence: 101',
    'extracted_final_answer: a\nreasoning: compared\ncorrect: yes\ncorrect: no\nconfidence: 90',
    '<think>correct: yes',
])
def test_bcp_rejects_conflicts_out_of_range_and_thinking(tmp_path, raw):
    root = prepared_bcp(tmp_path)
    rows, _ = workflow.prepared_questions(root, 'browsecomp-plus')
    predictions = {rows[0]['id']: dict(status='completed', final_response='prediction')}
    output = tmp_path / 'scores'; output.mkdir()
    metrics = score_browsecomp(root, rows, predictions, output, lambda _: raw,
                              judge_attempts=1, judge_preflight=False)
    assert not metrics['complete'] and metrics['judge_errors'] == 1
    assert metrics['accuracy'] == 0
