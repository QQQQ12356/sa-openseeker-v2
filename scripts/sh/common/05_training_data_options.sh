#!/usr/bin/env bash
# 修改临时实验配置；只覆盖明确设置的变量，不改变原入口默认行为。
# 被 20_plain_train.sh / 30_beacon_train.sh source，需已定义 PY、CONFIG_FILE。
"$PY" - "$CONFIG_FILE" <<'PY'
import os
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
spec = yaml.safe_load(path.read_text(encoding='utf-8'))
if os.environ.get('DATASET_DIR'):
    spec['dataset'] = os.environ['DATASET_DIR']
if os.environ.get('MODEL_PATH'):
    spec['models'][spec['model_family']] = os.environ['MODEL_PATH']
if os.environ.get('MAX_LENGTH'):
    length = int(os.environ['MAX_LENGTH'])
    if length < 1:
        raise SystemExit('MAX_LENGTH 必须为正整数')
    spec['training']['max_length'] = length
path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding='utf-8')
PY
