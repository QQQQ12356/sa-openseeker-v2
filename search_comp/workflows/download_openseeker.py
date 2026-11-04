#!/usr/bin/env python
"""Download pinned HF snapshots, retain originals and export every split as JSONL."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def export_parquet(path, target):
    """Export without changing any downloaded source file."""
    import pyarrow.parquet as pq
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.jsonl.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as out:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['polar','zephyr','research','all'], default='polar')
    p.add_argument('--endpoint', default=os.getenv('HF_ENDPOINT','https://huggingface.co'))
    p.add_argument('--revision', default='main')
    p.add_argument('--output', type=Path, default=Path('outputs/data/raw'))
    args = p.parse_args()
    os.environ['HF_ENDPOINT'] = args.endpoint
    from huggingface_hub import HfApi, snapshot_download
    repos = dict(polar='PolarSeeker/OpenSeeker-v1-Data', zephyr='Zephyr271828/openseeker_v1_sft',
                 research='AmanPriyanshu/tool-reasoning-sft-RESEARCH-OpenSeeker-v1-Data')
    for source, repo in repos.items():
        if args.source not in ('all', source):
            continue
        info = HfApi(endpoint=args.endpoint).dataset_info(repo, revision=args.revision)
        directory = args.output / source / info.sha
        snapshot_download(repo, repo_type='dataset', revision=info.sha, endpoint=args.endpoint,
                          local_dir=str(directory), allow_patterns=['*.jsonl','*.parquet','README.md'])
        files = []
        jsonl_inputs = []
        for path in sorted(directory.rglob('*')):
            if path.relative_to(directory).parts[0] in ('exports', '.cache'):
                continue
            if path.suffix not in ('.jsonl','.parquet'):
                continue
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
            files.append(dict(path=str(path),sha256=h.hexdigest(),bytes=path.stat().st_size))
            if path.suffix == '.parquet':
                # Prefer an existing native JSONL of the same stem to avoid
                # feeding the same representation twice into preprocessing.
                if path.with_suffix('.jsonl').is_file():
                    continue
                target = directory / 'exports' / path.relative_to(directory).with_suffix('.jsonl')
                export_parquet(path, target)
                jsonl_inputs.append(str(target.resolve()))
            else:
                jsonl_inputs.append(str(path.resolve()))
        if not jsonl_inputs:
            raise ValueError('snapshot contains no supported JSONL/Parquet data files')
        (directory/'download_manifest.json').write_text(json.dumps(dict(repo=repo,revision=info.sha,
            endpoint=args.endpoint,files=files,jsonl_inputs=jsonl_inputs),indent=2)+'\n', encoding='utf-8')
        print(directory)


if __name__=='__main__': main()
