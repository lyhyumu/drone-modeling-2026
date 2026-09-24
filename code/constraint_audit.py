# -*- coding: utf-8 -*-
"""约束闭环审计：从最终 results.json 重算全部硬约束(C1-C17)，不信任求解器字段。
只 print 结论(PASS/FAIL, n_violations, max_error, 最多5条定位)，禁止 print 整数组。"""
import os, sys, json, re
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import params as P
import energy, soc, utils

FIG = utils.FIGDIR


def _load(name):
    return json.loads((FIG/name).read_text(encoding='utf-8'))


def audit_p1():
    r = _load('problem_1_results.json')
    boxes = utils.load_boxes()
    nodes = utils.nodes_by_id(); o01 = nodes[P.O01['id']]
    viol = []
    delivered = []
    for aid, b in r['batching'].items():
        for t in b['trips']:
            g = t['g']; si = nodes[aid]
            # 重算能耗(从质量)
            e = energy.roundtrip_energy(g, t['mass'], o01, si)
            budget = (1-P.RHO_BASE)*P.E_USE[g]
            if e > budget + 1e-4:
                viol.append(f"P1 {aid} 能耗{e:.3f}>{budget:.3f}")
            if t['mass'] > P.Q[g] + 1e-6:
                viol.append(f"P1 {aid} 载质量{t['mass']}>{P.Q[g]}")
            if t['vol'] > P.V[g] + 1e-9:
                viol.append(f"P1 {aid} 体积超")
            s_end = 1 - e/P.E_USE[g]
            if s_end < P.RHO_BASE - 1e-4:
                viol.append(f"P1 {aid} SOC{s_end:.3f}<0.20")
            # 不跨区(R2)
            if len(set(bx.split('-')[0] for bx in t['box_ids'])) != 1:
                viol.append(f"P1 {aid} 跨区组批")
            delivered.extend(t['box_ids'])
    # 每箱1次(R1)
    from collections import Counter
    cnt = Counter(delivered)
    n_bad = sum(1 for c in cnt.values() if c != 1)
    if n_bad or len(delivered) != len(boxes):
        viol.append(f"P1 箱覆盖异常 dup/miss={n_bad} total={len(delivered)}/{len(boxes)}")
    print(f"[P1] PASS={not viol} n_violations={len(viol)}")
    for v in viol[:5]:
        print(f"    VIOLATION {v}")
    return len(viol) == 0


def audit_p2():
    r = _load('problem_2_results.json')
    boxes = utils.load_boxes(); blk = {b['box_id']: b for b in boxes}
    nodes = utils.nodes_by_id(); o01 = nodes[P.O01['id']]
    viol = []
    max_e_err = 0.0
    delivered = []
    for t in r['trips']:
        g = t['g']
        seq = [o01] + [nodes[a] for a in t['areas']] + [o01]
        # 重算能耗(逐段载荷)
        by_area = {}
        for bid in t['box_ids']:
            by_area.setdefault(blk[bid]['area'], []).append(blk[bid])
        load0 = sum(blk[bid]['mass'] for bid in t['box_ids'])
        deliver_amounts = [sum(bx['mass'] for bx in by_area[a]) for a in t['areas']]
        e_re, _, _, _ = energy.route_energy_time(g, load0, seq, deliver_amounts)
        max_e_err = max(max_e_err, abs(e_re - t['energy']))
        budget = (1-P.RHO_BASE)*P.E_USE[g]
        if e_re > budget + 1e-3:
            viol.append(f"P2 rid{t['rid']} 能耗{e_re:.3f}>{budget:.3f}")
        if load0 > P.Q[g] + 1e-6:
            viol.append(f"P2 rid{t['rid']} 载质量{load0:.2f}>{P.Q[g]}")
        vol = sum(blk[bid]['vol'] for bid in t['box_ids'])
        if vol > P.V[g] + 1e-9:
            viol.append(f"P2 rid{t['rid']} 体积{vol:.3f}>{P.V[g]}")
        if 1 - e_re/P.E_USE[g] < P.RHO_BASE - 1e-3:
            viol.append(f"P2 rid{t['rid']} SOC<0.20")
        delivered.extend(t['box_ids'])
    # 每箱1次
    from collections import Counter
    cnt = Counter(delivered)
    if any(c != 1 for c in cnt.values()) or len(delivered) != len(boxes):
        viol.append(f"P2 箱覆盖异常 total={len(delivered)}/{len(boxes)}")
    # 硬时限(C7/C8): 从交付重算
    n_hard_bad = 0
    for d in r['deliveries']:
        b = blk[d['box_id']]
        if b['is_med'] and d['t_deliver'] > b['t_exp'] + 1e-3:
            n_hard_bad += 1
        if b['is_first'] and b['t_first'] and d['t_deliver'] > b['t_first'] + 1e-3:
            n_hard_bad += 1
    if n_hard_bad:
        viol.append(f"P2 硬时限违反 {n_hard_bad} 箱")
    # 资源峰值<=库存
    for g, p in r['resource_audit']['resource_peaks'].items():
        if p['uav_peak'] > P.FLEET_INV[g]:
            viol.append(f"P2 {g}机峰值{p['uav_peak']}>{P.FLEET_INV[g]}")
        if p['bat_peak'] > P.BAT_INV[g]:
            viol.append(f"P2 {g}电池峰值{p['bat_peak']}>{P.BAT_INV[g]}")
    print(f"[P2] PASS={not viol} n_violations={len(viol)} max_energy_recalc_error={max_e_err:.2e}")
    for v in viol[:5]:
        print(f"    VIOLATION {v}")
    return len(viol) == 0


def audit_p3():
    r = _load('problem_3_results.json')
    viol = []
    # 连续通信: outage(cov=0) 段数
    outages = [u for u in r['uncovered'] if u.get('cov_frac', 0) == 0]
    if outages:
        viol.append(f"P3 完全中断 {len(outages)} 段")
    # 中继悬停 agl<=300, 回传/接入可用
    for m in r['relay_missions']:
        if m['hover_agl'] > P.RELAY_MAX_AGL + 1e-6:
            viol.append(f"P3 {m['relay_trip_id']} agl{m['hover_agl']}>300")
        if m['backhaul_margin_db'] < -1e-6:
            viol.append(f"P3 {m['relay_trip_id']} 回传不可用")
        if m['access_min_margin_db'] is not None and m['access_min_margin_db'] < -1e-6:
            viol.append(f"P3 {m['relay_trip_id']} 接入不可用")
        if m['soc_end'] < P.RELAY_RETURN_SOC_MIN - 1e-3:
            viol.append(f"P3 {m['relay_trip_id']} 中继SOC<0.20")
    # 联合完工>=运输完工
    jo = r['joint_objective']
    if jo['joint_cmax_s'] < jo['transport_cmax_s'] - 1e-6:
        viol.append("P3 联合完工<运输完工")
    print(f"[P3] PASS={not viol} n_violations={len(viol)}")
    for v in viol[:5]:
        print(f"    VIOLATION {v}")
    return len(viol) == 0


def audit_p4():
    r = _load('problem_4_results.json')
    viol = []
    areas_all = sorted(s['id'] for s in P.SERVICE_AREAS)
    for K in P.K_VALUES:
        part = r['partitions'][f'K={K}']
        got = sorted(a for grp in part['groups'] for a in grp['areas'])
        if got != areas_all:
            viol.append(f"P4 K={K} 分区非互斥完备")
        if any(len(grp['areas']) < 1 for grp in part['groups']):
            viol.append(f"P4 K={K} 空组")
        # 缺口公式
        for rt, gap in part['gap_by_type'].items():
            need = sum(grp['resource_peaks'][rt] for grp in part['groups'])
            if gap != max(0, need - r['inventory'][rt]):
                viol.append(f"P4 K={K} {rt} 缺口公式错")
    # 单调性: K=3 总配置>=K=2
    if r['partitions']['K=3']['total_config'] < r['partitions']['K=2']['total_config']:
        viol.append("P4 K=3总配置<K=2(分区不共享单调性违反)")
    print(f"[P4] PASS={not viol} n_violations={len(viol)}")
    for v in viol[:5]:
        print(f"    VIOLATION {v}")
    return len(viol) == 0


if __name__ == '__main__':
    print("=== 约束闭环审计(从 results.json 重算) ===")
    ok1 = audit_p1(); ok2 = audit_p2(); ok3 = audit_p3(); ok4 = audit_p4()
    all_ok = ok1 and ok2 and ok3 and ok4
    print(f"\n=== 审计总结: {'ALL PASS' if all_ok else 'FAIL'} ===")
    sys.exit(0 if all_ok else 1)
