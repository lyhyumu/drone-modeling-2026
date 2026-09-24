# -*- coding: utf-8 -*-
"""公共工具：数据读取（全量核对 DATA_PROFILE）、结果保存、节点索引。"""
import os, sys, json, random
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from pathlib import Path
import numpy as np
import pandas as pd
import params as P

ROOT = P.ROOT
FIGDIR = ROOT / 'figures'
OUTDIR = ROOT / 'output'
FIGDIR.mkdir(exist_ok=True)
OUTDIR.mkdir(exist_ok=True)

_PROFILE = json.loads((ROOT / 'DATA_PROFILE.json').read_text(encoding='utf-8'))['files']


def set_all_seeds(seed=P.SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_boxes():
    """读取逐箱货箱清单(80行)，核对 DATA_PROFILE 行数。"""
    fname = '物资需求与配送时限.xlsx'
    _all = pd.read_excel(ROOT / 'user_data' / fname, sheet_name=None)  # 全表读取
    df = _all['逐箱货箱清单']
    prof_rows = _PROFILE[fname]['sheets']['逐箱货箱清单']['rows']
    assert len(df) == prof_rows, f"[摄入不全] 逐箱清单实读{len(df)}行≠建档{prof_rows}行"
    print(f"[data_ingest] {fname}::逐箱货箱清单 全量 {len(df)} 行已核对通过")
    boxes = []
    for _, r in df.iterrows():
        boxes.append({
            'box_id': r['货箱编号'],
            'area': r['服务区编号'],
            'type': r['物资类型'],
            'mass': float(r['单箱质量（kg）']),
            'vol': float(r['单箱体积（m³）']),
            'is_first': (r['是否首批保障'] == '是'),
            't_first': (float(r['首批截止时间（s）']) if pd.notna(r['首批截止时间（s）']) else None),
            't_exp': float(r['期望送达时间（s）']),
            'w': float(r['应急优先系数']),
            'is_med': (r['物资类型'] == '医疗物资'),
        })
    return boxes


def nodes_by_id():
    """返回 {id: node dict(lon,lat,alt_m)}，含 O01 与 15 服务区。"""
    d = {P.O01['id']: dict(P.O01)}
    for s in P.SERVICE_AREAS:
        d[s['id']] = dict(s)
    return d


def area_of(box):
    return box['area']


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[save] {path} ({Path(path).stat().st_size} bytes)")


def np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


if __name__ == '__main__':
    boxes = load_boxes()
    print(f"总箱数 {len(boxes)}; 医疗 {sum(b['is_med'] for b in boxes)}; 首批 {sum(b['is_first'] for b in boxes)}")
    from collections import Counter
    print("各区箱数:", dict(sorted(Counter(b['area'] for b in boxes).items())))
    nd = nodes_by_id()
    print(f"节点数 {len(nd)} (含O01+15区)")
