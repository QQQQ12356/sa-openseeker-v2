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
ENTRY = ROOT / 'scripts/py/52_offline_benchmark.py'


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return path


class CharacterTokenizer:
    def encode(self, value, **kwargs):
        return list(value.encode('utf-8'))

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode('utf-8', errors='ignore')


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
    predictions = {rows[0]['id']: dict(status='completed', final_response='answer [' + rows[0]['evidence_docs'][0] + ']',
          retrieved_docids=rows[0]['evidence_docs'], tool_call_counts={'search': 1})}
    prompts = []
    def judge(prompt):
        prompts.append(prompt)
        return 'extracted_final_answer: answer\nreasoning: matches\ncorrect: yes\nconfidence: 90'
    output = tmp_path / 'scores'
    output.mkdir()
    metrics = score_browsecomp(root, rows, predictions, output, judge)
    assert metrics['accuracy'] == metrics['evidence_recall'] == 0.5
    assert metrics['avg_tool_stats']['search'] == 0.5
    assert len(prompts) == 1 and 'REFERENCE_DO_NOT_PROMPT' in prompts[0]
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
    metrics = score_browsecomp(root, rows, predictions, output, lambda _: 'malformed judge output')
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


def test_hotpot_cli_prepare_index_and_official_score_without_model(tmp_path):
    data, corpus = hotpot_sources(tmp_path)
    root = tmp_path / 'cli'
    for action in ('prepare', 'index'):
        result = subprocess.run([sys.executable, str(ENTRY), action, '--benchmark', 'hotpotqa', '--root', str(root),
             '--data', str(data), '--corpus', str(corpus)], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    predictions = write_jsonl(tmp_path / 'predictions.jsonl', [dict(id='h1', status='completed', prediction='red blue')])
    output = tmp_path / 'scores'
    result = subprocess.run([sys.executable, str(ENTRY), 'score', '--benchmark', 'hotpotqa', '--root', str(root),
        '--predictions', str(predictions), '--output-dir', str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / 'summary.metrics.json').read_text())
    assert metrics['f1'] == 0.5 and metrics['missing_predictions'] == 1
    assert metrics['offline'] and metrics['subset']
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'


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
    workflow.run_phase(args, output, rows, manifest)
    predictions = list(records(output / 'predictions.jsonl'))
    assert predictions[0]['status'] == 'completed'
    assert predictions[0]['prompt_source'] == str(prompt_data)
    assert predictions[0]['visited_docids'] == ['1']
    assert predictions[0]['tool_call_counts'] == dict(search=1, visit=1)
    assert generator_kinds == [kind == 'beacon']
    assert saw_regions == [False, True, True]
    assert 'answer' not in predictions[0]


def test_shell_dry_run_has_no_writes_and_preserves_space_paths(tmp_path):
    for script in ('56_browsecomp_plus.sh', '57_hotpotqa_offline.sh'):
        command = ['bash', str(ROOT / 'scripts/sh' / script)]
        import os
        env = dict(os.environ, ACTION='run', DRY_RUN='1', ROOT_DIR=str(tmp_path / 'root with spaces'),
                   MODEL_PATH=str(tmp_path / 'model with spaces'), PYTHON=sys.executable,
                   NUM_WORKERS='4', CUDA_VISIBLE_DEVICES='0,1,2,3')
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert parsed['model'] == str(tmp_path / 'model with spaces')
        assert parsed['num_workers'] == 4
        assert not (tmp_path / 'root with spaces').exists()


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


def test_bcp_retrieval_only_cli_needs_no_judge_or_model(tmp_path):
    root = prepared_bcp(tmp_path)
    rows, _ = workflow.prepared_questions(root, 'browsecomp-plus')
    predictions = write_jsonl(tmp_path / 'predictions.jsonl', [dict(id=r['id'], status='completed', prediction='x',
                        retrieved_docids=r['evidence_docs']) for r in rows])
    output = tmp_path / 'scores'
    result = subprocess.run([sys.executable, str(ENTRY), 'score', '--benchmark', 'browsecomp-plus', '--root', str(root),
        '--predictions', str(predictions), '--retrieval-only', '--output-dir', str(output)], cwd=ROOT,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / 'summary.metrics.json').read_text())
    assert metrics['evidence_recall'] == 1
    assert 'accuracy' not in metrics and metrics['score_mode'] == 'retrieval-only'


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
    assert judge('question') == ''
    judge.max_context = 1
    with pytest.raises(ValueError, match='context budget'):
        judge('question')


def parallel_harness(tmp_path, monkeypatch):
    """Real child processes and agent loop; replace only expensive model inference."""
    harness = tmp_path / 'worker_harness.py'
    harness.write_text('''
import os
import runpy
import sys
import time
from types import SimpleNamespace
sys.path.insert(0, ROOT_PATH)
if os.environ.get('OFFLINE_TEST_FAIL'):
    if os.environ['CUDA_VISIBLE_DEVICES'] == '3':
        sys.exit(7)
    time.sleep(30)
class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())
    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode(errors='ignore')
model = SimpleNamespace(eval=lambda: None)
sys.modules['search_comp.models.model_loader'] = SimpleNamespace(
    load_tokenizer=lambda _: Tokenizer(), load_beacon_model=lambda _: model)
sys.modules['search_comp.models.plain_qwen3'] = SimpleNamespace(load_sft_model=lambda _: model)
from search_comp.evaluation import tool_agent
def generator(model, tokenizer, beacon):
    assert beacon
    def generate(encoded, budget):
        assert 'REFERENCE_DO_NOT_PROMPT' not in tokenizer.decode(encoded['input_ids'])
        text = '<answer>gpu=' + os.environ['CUDA_VISIBLE_DEVICES'] + '</answer>'
        return text, True, len(text)
    return generate
tool_agent.ModelGenerator = generator
runpy.run_path(ENTRY_PATH, run_name='__main__')
'''.replace('ROOT_PATH', repr(str(ROOT))).replace('ENTRY_PATH', repr(str(ENTRY))))
    original = workflow.worker_command
    def command(*args):
        result = original(*args)
        return [result[0], str(harness), *result[2:]]
    monkeypatch.setattr(workflow, 'worker_command', command)


@pytest.mark.parametrize('workers,limit,action', [(2, -1, 'run'), (4, 1, 'run'), (2, 3, 'all')])
def test_parallel_real_processes_merge_and_score(tmp_path, monkeypatch, workers, limit, action):
    root = prepared_bcp(tmp_path, count=5)
    model = tmp_path / 'model with spaces'
    model.mkdir()
    output = tmp_path / 'parallel run'
    parallel_harness(tmp_path, monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,3,5,7')
    monkeypatch.setattr(sys, 'argv', [str(ENTRY), action, '--benchmark', 'browsecomp-plus',
        '--root', str(root), '--model', str(model), '--kind', 'beacon', '--limit', str(limit),
        '--decrypted', '--retrieval-only',
        '--num-workers', str(workers), '--output-dir', str(output)])
    workflow.main()
    rows, _ = workflow.prepared_questions(root, 'browsecomp-plus', limit)
    predictions = list(records(output / 'predictions.jsonl'))
    assert [row['id'] for row in predictions] == [row['id'] for row in rows]
    assert [row['sample_index'] for row in predictions] == list(range(len(rows)))
    devices = ['0', '3'][:min(workers, len(rows))]
    assert [row['final_response'] for row in predictions] == [
        'gpu=' + devices[i % len(devices)] for i in range(len(rows))]
    metrics = json.loads((output / 'generation.metrics.json').read_text())
    assert metrics['count'] == len(rows)
    assert metrics['statuses'] == {'completed': len(rows)}
    assert metrics['devices'] == devices
    metadata = json.loads((output / 'manifest.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['selected_ids'] == [row['id'] for row in rows]
    assert metadata['predictions_sha256'] == digest(output / 'predictions.jsonl')
    assert len(list((output / 'workers').glob('worker_*/manifest.json'))) == len(devices)
    score = tmp_path / 'scored'
    monkeypatch.setattr(sys, 'argv', [str(ENTRY), 'score', '--benchmark', 'browsecomp-plus',
        '--root', str(root), '--predictions', str(output / 'predictions.jsonl'),
        '--limit', str(limit), '--retrieval-only', '--output-dir', str(score)])
    workflow.main()
    summary = json.loads((score / 'summary.metrics.json').read_text())
    assert summary['missing_predictions'] == 0
    assert summary['complete']


def test_parallel_failure_stops_other_workers_and_marks_manifest(tmp_path, monkeypatch):
    root = prepared_bcp(tmp_path)
    model = tmp_path / 'model'
    model.mkdir()
    output = tmp_path / 'failed run'
    parallel_harness(tmp_path, monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,3')
    monkeypatch.setenv('OFFLINE_TEST_FAIL', '1')
    processes = []
    original = subprocess.Popen
    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(workflow.subprocess, 'Popen', popen)
    monkeypatch.setattr(sys, 'argv', [str(ENTRY), 'run', '--benchmark', 'browsecomp-plus',
        '--root', str(root), '--model', str(model), '--kind', 'beacon',
        '--num-workers', '2', '--output-dir', str(output)])
    with pytest.raises(RuntimeError, match='exited with code 7'):
        workflow.main()
    assert len(processes) == 2 and all(p.poll() is not None for p in processes)
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'failed'
    assert not (output / 'predictions.jsonl').exists()
    assert not (output / 'summary.metrics.json').exists()
    assert (output / 'workers/worker_1.log').is_file()


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


@pytest.mark.parametrize('visible,workers', [('0', '2'), ('0,0', '2'), ('', '2'), ('0,1', '0')])
def test_parallel_cli_rejects_bad_gpu_config_before_writes(tmp_path, visible, workers):
    import os
    result = subprocess.run([sys.executable, str(ENTRY), 'run', '--benchmark', 'browsecomp-plus',
        '--num-workers', workers, '--root', str(tmp_path / 'absent'), '--dry-run'],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible), capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / 'absent').exists()
