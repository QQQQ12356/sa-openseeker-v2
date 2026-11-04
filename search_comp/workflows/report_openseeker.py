#!/usr/bin/env python
"""Compare quality/language/length strata from a full measured dataset; emit JSON."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from itertools import zip_longest
from search_comp.workflows.prepare_openseeker import stats


def population(rows, model):
    """Describe one explicit population using cached training-protocol lengths."""
    from search_comp.data.openseeker import parse_calls
    values = defaultdict(list)
    tools = Counter()
    correctness = Counter()
    count = 0
    for row in rows:
        count += 1
        lengths = row['lengths'][model]
        for key in ('total', 'observations', 'assistant'):
            values[key].append(lengths[key])
        calls = [call for message in row['messages'] if message['role'] == 'assistant'
                 for call in parse_calls(message['content'])]
        tools.update(call['name'] for call in calls)
        correctness[str(row.get('correctness'))] += 1
        values['calls'].append(len(calls))
        values['messages'].append(len(row['messages']))
    return dict(count=count, tools=dict(tools), correctness=dict(correctness),
                distributions={key: stats(values[key]) for key in
                               ('total', 'observations', 'assistant', 'calls', 'messages')})


def jsonl(path):
    with path.open(encoding='utf-8') as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model', help='Exact tokenizer key; auto-select only when manifest has one tokenizer')
    args=p.parse_args()
    manifest=json.loads((args.prepared/'manifest.json').read_text(encoding='utf-8'))
    models=list(manifest['tokenizers'])
    if args.model is None:
        if len(models) != 1:
            p.error('multiple tokenizers: set --model explicitly; lengths are never mixed')
        args.model=models[0]
    if args.model not in models:
        p.error('model must match a tokenizer key in manifest')
    args.output.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(lambda:dict(count=0,cjk=0,tools=Counter(),lengths=[]))
    declared=Counter();totals=[]
    with (args.prepared/'normalized.jsonl').open(encoding='utf-8') as f,(args.prepared/'measurements.jsonl').open(encoding='utf-8') as g:
        for a,b in zip_longest(f,g):
            row,measure=json.loads(a),json.loads(b)
            assert row['id']==measure['id']
            total=row['lengths'][args.model]['total'];totals.append(total)
            cjk=any('\u4e00'<=c<='\u9fff' for c in row['question'])
            turns=sum(m['role']=='assistant' for m in row['messages'])
            declared['equals_assistant_turns']+=row['declared_tool_calls']==turns
            declared['equals_actual_calls']+=row['declared_tool_calls']==measure['calls']
            declared['equals_calls_plus_one']+=row['declared_tool_calls']==measure['calls']+1
            keys=['all',str(row['correctness'])]
            for threshold in (8192,16384,32768,65536):
                if total<=threshold:
                    keys.append(f'all_le_{threshold}')
                    if row['correctness']=='Correct':keys.append(f'correct_le_{threshold}')
            for key in keys:
                entry=groups[key];entry['count']+=1;entry['cjk']+=cjk
                entry['tools'].update(measure['tools']);entry['lengths'].append(total)
    splits={name: population(jsonl(args.prepared/f'{name}.jsonl'), args.model)
            for name in ('train','validation','test')}
    retained=population((row for name in ('train','validation','test')
                         for row in jsonl(args.prepared/f'{name}.jsonl')), args.model)
    rejections=Counter(str(row.get('reason','unspecified')) for row in jsonl(args.prepared/'rejected.jsonl'))
    result=dict(model=args.model, source=manifest.get('arguments',{}).get('source'),
                input_counts=manifest.get('counts',{}), rejections=dict(rejections),
                populations=dict(before_filter=population(jsonl(args.prepared/'normalized.jsonl'),args.model),
                                 retained=retained, **splits),
                declared_field=declared,groups={k:{**v,'lengths':stats(v['lengths'])} for k,v in groups.items()})
    (args.output/'quality_length.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved',args.output)


if __name__=='__main__':main()
