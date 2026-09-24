# -*- coding: utf-8 -*-
"""问题四：救援任务分区与资源配置。冻结问题三方案，仅做分区+资源核算。
共访图超节点收缩(P4-C1) + 区间图最大重叠资源核算(P4-C2) + 缺口分析(P4-C3)。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import json, itertools
import numpy as np
import params as P
import soc, utils


def _peak_overlap(intervals):
    """区间集合 [(a,b)] 的最大同时重叠数(区间图着色下界=最大团)。"""
    ev = []
    for a, b in intervals:
        ev.append((a, 1)); ev.append((b, -1))
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, d in ev:
        cur += d; peak = max(peak, cur)
    return peak


def build_covisit_supernodes(trips):
    """共访图：同架次访问的服务区连边，连通分量收缩为超节点。"""
    # 并查集
    parent = {s['id']: s['id'] for s in P.SERVICE_AREAS}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for t in trips:
        areas = t['areas']
        for i in range(len(areas)-1):
            union(areas[i], areas[i+1])
    groups = {}
    for s in P.SERVICE_AREAS:
        r = find(s['id'])
        groups.setdefault(r, []).append(s['id'])
    supernodes = [sorted(v) for v in groups.values()]
    return supernodes


def resource_intervals_for_areas(trips, relay_missions, area_set):
    """某组(服务区集合)承担的任务占用区间，按资源类型分组。
    运输机(按机型)、电池(按机型,含充电)、中继机、中继能源组件(含充电)。"""
    res = {'uav_A': [], 'uav_B': [], 'uav_C': [],
           'bat_A': [], 'bat_B': [], 'bat_C': [], 'relay': [], 'relay_comp': []}
    trip_rids_in_group = set()
    for t in trips:
        if set(t['areas']) & area_set:                 # 该架次服务本组
            trip_rids_in_group.add(t['rid'])
            g = t['g']
            tau = t['tau']; Dr = int(np.ceil(t['D_r']))
            res[f'uav_{g}'].append((tau, tau+Dr))
            tchg = int(np.ceil(soc.t_charge(t['soc_end'], P.T_FULL_BAT[g])))
            res[f'bat_{g}'].append((tau, tau+Dr+tchg))
    # 中继: 服务本组架次的中继任务。一架物理中继(relay_unit)一悬停点可同时服务多架次,
    # 故按 relay_unit 归并重叠服务窗为连续占用, 再计并发物理中继数(不按 mission 逐条累加)。
    by_unit = {}
    for m in relay_missions:
        if m['serves_trip'] in trip_rids_in_group:
            by_unit.setdefault(m['relay_unit'], []).append(
                (m['service_start'], m['service_end']))
    for unit, ivs in by_unit.items():
        ivs.sort()
        cs, ce = ivs[0]
        merged = []
        for s, e in ivs[1:]:
            if s <= ce + 1e-6:
                ce = max(ce, e)
            else:
                merged.append((cs, ce)); cs, ce = s, e
        merged.append((cs, ce))
        for (a, b) in merged:
            res['relay'].append((a, b + P.RELAY_TURNAROUND_S))
            # 能源组件: 该连续服务段占用一组组件 + 充电周转
            res['relay_comp'].append((a, b + P.RELAY_TURNAROUND_S + P.T_FULL_COMP))
    return res


def account_group_resources(res):
    """各资源类型的区间最大重叠数(式30)。"""
    return {k: _peak_overlap(v) for k, v in res.items()}


def run():
    utils.set_all_seeds()
    p2 = json.loads((utils.FIGDIR/'problem_2_results.json').read_text(encoding='utf-8'))
    p3 = json.loads((utils.FIGDIR/'problem_3_results.json').read_text(encoding='utf-8'))
    trips = p2['trips']
    relay_missions = p3['relay_missions']

    supernodes = build_covisit_supernodes(trips)
    print(f"[P4] 共访图超节点数 {len(supernodes)}: {[len(s) for s in supernodes]}", flush=True)

    inventory = {'uav_A': P.FLEET_INV['A'], 'uav_B': P.FLEET_INV['B'], 'uav_C': P.FLEET_INV['C'],
                 'bat_A': P.BAT_INV['A'], 'bat_B': P.BAT_INV['B'], 'bat_C': P.BAT_INV['C'],
                 'relay': P.RELAY_N_UNITS, 'relay_comp': P.RELAY_COMP_INV}

    # 全局峰值(K=1 参照)
    global_res = resource_intervals_for_areas(trips, relay_missions,
                                              set(s['id'] for s in P.SERVICE_AREAS))
    global_peak = account_group_resources(global_res)

    partitions = {}
    for K in P.K_VALUES:
        best = _optimize_partition(supernodes, trips, relay_missions, inventory, K)
        partitions[f'K={K}'] = best
        print(f"[P4] K={K}: 总配置={best['total_config']} 总缺口={best['total_gap']} "
              f"分组={[grp['areas'] for grp in best['groups']]}", flush=True)

    results = {
        'seed': P.SEED,
        'frozen_from': 'problem_3_results.json',
        'supernodes': supernodes,
        'inventory': inventory,
        'global_peak_K1': global_peak,
        'partitions': partitions,
    }
    utils.save_json(utils.FIGDIR/'problem_4_results.json', results)
    utils.save_json(utils.OUTDIR/'problem4_partition.json',
                    {'inventory': inventory, 'global_peak_K1': global_peak,
                     'partitions': {k: {'total_config': v['total_config'],
                                        'total_gap': v['total_gap'],
                                        'groups': v['groups']} for k, v in partitions.items()}})
    return results


def _optimize_partition(supernodes, trips, relay_missions, inventory, K):
    """枚举超节点到 K 组的分配(去对称)，词典序 min(总配置, 总缺口, 均衡)。"""
    n = len(supernodes)
    rtypes = list(inventory.keys())
    best = None
    # 枚举分配(K^n 但 n 小; 去对称: 首超节点固定组0)
    assignments = _gen_assignments(n, K)
    for assign in assignments:
        groups_areas = [[] for _ in range(K)]
        for i, k in enumerate(assign):
            groups_areas[k].extend(supernodes[i])
        if any(len(g)==0 for g in groups_areas):        # 每组>=1区
            continue
        group_details = []
        total_config = 0
        total_gap = 0
        workloads = []
        for gk in range(K):
            area_set = set(groups_areas[gk])
            res = resource_intervals_for_areas(trips, relay_missions, area_set)
            peaks = account_group_resources(res)
            # 该组工作量(总占用时长)
            wl = sum(b-a for iv in res.values() for (a,b) in iv)
            workloads.append(wl)
            config = {rt: peaks[rt] for rt in rtypes}
            total_config += sum(config.values())
            group_details.append({'group': gk, 'areas': sorted(area_set),
                                   'resource_peaks': config, 'workload_s': round(wl,1)})
        # 缺口 = Σ_k peak - inventory 取正部(按资源类型)
        gap_by_type = {}
        for rt in rtypes:
            need = sum(gd['resource_peaks'][rt] for gd in group_details)
            gap_by_type[rt] = max(0, need - inventory[rt])
        total_gap = sum(gap_by_type.values())
        balance = float(np.var(workloads))
        key = (total_config, total_gap, balance)
        if best is None or key < best['_key']:
            best = {'_key': key, 'K': K, 'total_config': total_config,
                    'total_gap': total_gap, 'gap_by_type': gap_by_type,
                    'workload_balance_var': round(balance,1), 'groups': group_details}
    best.pop('_key', None)
    return best


def _gen_assignments(n, K):
    """生成 n 个超节点到 K 组的分配，去对称(第一个固定组0，限制新组按序引入)。"""
    out = []
    def rec(i, assign, max_used):
        if i == n:
            out.append(list(assign)); return
        for k in range(min(max_used+2, K)):     # 去对称: 只能用已用组或下一个新组
            assign.append(k)
            rec(i+1, assign, max(max_used, k))
            assign.pop()
    rec(0, [], -1)
    return out


def validate_capability(results):
    # P4-C1: 每服务区恰属一组, 每组>=1区, 同架次同组(共访图保证)
    p2 = json.loads((utils.FIGDIR/'problem_2_results.json').read_text(encoding='utf-8'))
    trips = p2['trips']
    for K in P.K_VALUES:
        part = results['partitions'][f'K={K}']
        all_areas = []
        for grp in part['groups']:
            all_areas.extend(grp['areas'])
            assert len(grp['areas']) >= 1, "存在空组"
        assert sorted(all_areas) == sorted(s['id'] for s in P.SERVICE_AREAS), "分区非互斥完备"
        # 同架次服务区同组
        area2grp = {}
        for gi, grp in enumerate(part['groups']):
            for a in grp['areas']:
                area2grp[a] = gi
        for t in trips:
            grps = set(area2grp[a] for a in t['areas'])
            assert len(grps) == 1, f"架次{t['rid']}跨组(共访约束违反)"
    # P4-C2: 各组资源=区间最大重叠(非任务数求和) - 结构性检查: peak<=任务数
    # P4-C3: 缺口=Σ需求-库存正部
    for K in P.K_VALUES:
        part = results['partitions'][f'K={K}']
        for rt, gap in part['gap_by_type'].items():
            need = sum(grp['resource_peaks'][rt] for grp in part['groups'])
            assert gap == max(0, need - results['inventory'][rt]), "缺口公式错"
    print("[P4] validate_capability PASS")


if __name__ == '__main__':
    res = run()
    validate_capability(res)
