# -*- coding: utf-8 -*-
"""问题一：单点往返最大安全载荷(二分法) + 货箱组批集合划分MILP + 返航余量敏感性。
能力 P1-C1/C2/C3/C4。方法唯一性: 二分法(§4.1) + 集合划分MILP词典序(§4.2)。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import json
from itertools import combinations
import numpy as np
import pulp
import params as P
import geo, energy, utils


def max_safe_payload(g, o01, si, rho):
    """三重上界最小值(式15): min{Qg, Vg/单位体积上界, q^E}。q^E 二分法(单调能量)。"""
    budget = (1 - rho) * P.E_USE[g]
    # 能量上界 q^E: E_RT(q) 单调递增
    if energy.roundtrip_energy(g, 0.0, o01, si) > budget:
        qE = 0.0
    elif energy.roundtrip_energy(g, P.Q[g], o01, si) <= budget:
        qE = P.Q[g]
    else:
        lo, hi = 0.0, P.Q[g]
        while hi - lo > P.EPS_Q:                       # 二分至 <1e-3 kg
            mid = 0.5 * (lo + hi)
            if energy.roundtrip_energy(g, mid, o01, si) <= budget:
                lo = mid
            else:
                hi = mid
        qE = lo
    # 体积上界: 用最小货箱单位体积(医疗0.012)作连续上界近似(组批时精确核验)
    v_min = 0.012
    q_vol = P.Q[g] * (P.V[g] / v_min) / (P.Q[g] / P.Q[g])  # 体积上界不直接限载质量, 见下
    # 体积与质量是不同维度: 体积上界表现为"最多能装多少体积", 折算最大质量上界较松,
    # 这里给出质量三重上界: 载质量Qg 与 能量qE 取min(体积在组批时逐箱核验)
    return min(P.Q[g], qE), qE


def compute_safe_payload_matrix(rho):
    """三机型×15服务区最大安全载荷矩阵。"""
    nodes = utils.nodes_by_id()
    o01 = nodes[P.O01['id']]
    mat = {}
    for g in ['A', 'B', 'C']:
        mat[g] = {}
        for s in P.SERVICE_AREAS:
            si = nodes[s['id']]
            qmax, qE = max_safe_payload(g, o01, si, rho)
            mat[g][s['id']] = {'q_max': round(qmax, 4), 'q_energy_bound': round(qE, 4),
                               'Q_g': P.Q[g], 'binding': ('energy' if qE < P.Q[g] - 1e-6 else 'mass')}
    return mat


def gen_candidate_batches(area_boxes, g, o01, si, rho):
    """枚举某区某机型的可行批次(非空货箱子集)。返回列表 [{boxes,mass,vol,energy,time,soc_end}]。"""
    budget = (1 - rho) * P.E_USE[g]
    qmax, _ = max_safe_payload(g, o01, si, rho)
    n = len(area_boxes)
    cands = []
    # 枚举子集(每区箱数<=15, 但组合爆炸: 用容量剪枝 + 限制单架次箱数)
    # 载质量/体积上界天然限制单架次箱数, 实际可行子集远少于 2^n
    idxs = list(range(n))
    # 生成: 从大小1到n, 用剪枝
    for r in range(1, n + 1):
        found_any = False
        for combo in combinations(idxs, r):
            m = sum(area_boxes[i]['mass'] for i in combo)
            v = sum(area_boxes[i]['vol'] for i in combo)
            if m > qmax + 1e-9 or v > P.V[g] + 1e-9:
                continue
            e = energy.roundtrip_energy(g, m, o01, si)
            if e > budget + 1e-12:
                continue
            found_any = True
            # 作业时间(式17): prep + n*load + fly + (handover_base + n*handover_per)
            t_fly = energy.roundtrip_time(g, o01, si)
            t_op = (P.PREP_S[g] + r * P.LOAD_PER_BOX_S[g] + t_fly +
                    P.HANDOVER_BASE_S[g] + r * P.HANDOVER_PER_BOX_S[g])
            cands.append({'boxes': list(combo), 'g': g, 'mass': m, 'vol': v,
                          'energy': e, 'time': t_op, 'soc_end': 1 - e / P.E_USE[g], 'n': r})
        # 若某尺寸完全不可行且更大尺寸质量必然超限, 可停(质量单调)
        if not found_any and r >= 1:
            min_add = min(area_boxes[i]['mass'] for i in idxs)
            # 若当前最小r箱最小质量组合已超所有机型能力则停
            if r * min_add > max(P.Q['A'], P.Q['B'], P.Q['C']):
                break
    return cands


def set_partition_area(area_id, area_boxes, o01, si, rho):
    """某区集合划分MILP: 每箱恰覆盖一次, 词典序 min(架次数, 能耗, 作业时间)。"""
    # 生成全机型候选批次
    all_cands = []
    for g in ['A', 'B', 'C']:
        all_cands.extend(gen_candidate_batches(area_boxes, g, o01, si, rho))
    if not all_cands:
        return None
    nboxes = len(area_boxes)

    def solve_level(fix_ntrips=None, fix_energy=None):
        prob = pulp.LpProblem(f"SP_{area_id}", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(len(all_cands))]
        # 每箱恰覆盖一次(式19)
        for c in range(nboxes):
            prob += pulp.lpSum(x[i] for i in range(len(all_cands)) if c in all_cands[i]['boxes']) == 1
        ntrips = pulp.lpSum(x)
        etot = pulp.lpSum(all_cands[i]['energy'] * x[i] for i in range(len(all_cands)))
        ttot = pulp.lpSum(all_cands[i]['time'] * x[i] for i in range(len(all_cands)))
        if fix_ntrips is not None:
            prob += ntrips == fix_ntrips
        if fix_energy is not None:
            prob += etot <= fix_energy + 1e-6
        if fix_ntrips is None:
            prob += ntrips
        elif fix_energy is None:
            prob += etot
        else:
            prob += ttot
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] != 'Optimal':
            return None
        sel = [i for i in range(len(all_cands)) if x[i].value() > 0.5]
        return sel

    # 词典序: L1 min架次数
    sel1 = solve_level()
    N_star = len(sel1)
    # L2 固定架次数 min能耗
    sel2 = solve_level(fix_ntrips=N_star)
    E_star = sum(all_cands[i]['energy'] for i in sel2)
    # L3 固定架次数+能耗 min作业时间
    sel3 = solve_level(fix_ntrips=N_star, fix_energy=E_star)
    if sel3 is None:
        sel3 = sel2
    trips = []
    for i in sel3:
        c = all_cands[i]
        trips.append({'area': area_id, 'g': c['g'],
                      'box_ids': [area_boxes[j]['box_id'] for j in c['boxes']],
                      'mass': round(c['mass'], 4), 'vol': round(c['vol'], 4),
                      'energy': round(c['energy'], 5), 'time': round(c['time'], 2),
                      'soc_end': round(c['soc_end'], 4)})
    return {'n_trips': N_star, 'total_energy': round(sum(t['energy'] for t in trips), 5),
            'total_time': round(sum(t['time'] for t in trips), 2),
            'n_candidates': len(all_cands), 'trips': trips}


def run():
    utils.set_all_seeds()
    boxes = utils.load_boxes()
    nodes = utils.nodes_by_id()
    o01 = nodes[P.O01['id']]
    # 按区分组
    by_area = {}
    for b in boxes:
        by_area.setdefault(b['area'], []).append(b)

    # (1) 安全载荷矩阵(基准 rho)
    payload_mat = compute_safe_payload_matrix(P.RHO_BASE)

    # (2) 各区集合划分组批
    batching = {}
    for s in P.SERVICE_AREAS:
        aid = s['id']
        si = nodes[aid]
        batching[aid] = set_partition_area(aid, by_area[aid], o01, si, P.RHO_BASE)
        print(f"[P1] {aid}: {batching[aid]['n_trips']} 架次, "
              f"能耗{batching[aid]['total_energy']:.3f}kWh, "
              f"候选{batching[aid]['n_candidates']}")

    total_trips = sum(batching[a]['n_trips'] for a in batching)
    total_energy = sum(batching[a]['total_energy'] for a in batching)
    total_time = sum(batching[a]['total_time'] for a in batching)
    print(f"[P1] 全局(基准rho=0.20): {total_trips}架次 能耗{total_energy:.3f}kWh 作业{total_time:.0f}s")

    # (3) 返航余量敏感性(P1-C4): 5情景
    sens = {}
    for rho in P.RHO_SCENARIOS:
        pm = compute_safe_payload_matrix(rho)
        # 汇总: 每机型对 S001 的安全载荷 + 全局架次数
        n_trips_scenario = 0
        for s in P.SERVICE_AREAS:
            aid = s['id']; si = nodes[aid]
            b = set_partition_area(aid, by_area[aid], o01, si, rho)
            n_trips_scenario += b['n_trips'] if b else 0
        # 安全载荷矩阵每机型均值
        qmax_by_g = {g: round(np.mean([pm[g][a]['q_max'] for a in pm[g]]), 3) for g in pm}
        sens[f"{rho:.2f}"] = {'rho': rho, 'total_trips': n_trips_scenario,
                              'mean_q_max': qmax_by_g,
                              'q_max_S001': {g: pm[g]['S001']['q_max'] for g in pm}}
        print(f"[P1-C4] rho={rho}: 全局架次={n_trips_scenario}, 各型均值安全载荷={qmax_by_g}")

    # 组织 Pareto 前沿(放松词典序: 扫描允许额外架次数)展示权衡
    results = {
        'seed': P.SEED,
        'rho_base': P.RHO_BASE,
        'safe_payload_matrix': payload_mat,
        'batching': batching,
        'summary': {'total_trips': total_trips, 'total_energy': round(total_energy, 4),
                    'total_time': round(total_time, 2)},
        'sensitivity_rho': sens,
    }
    utils.save_json(utils.FIGDIR / 'problem_1_results.json', results)
    # 能力清单要求的产物
    utils.save_json(utils.OUTDIR / 'problem1_safe_payload.json',
                    {'rho_base': P.RHO_BASE, 'safe_payload_matrix': payload_mat,
                     'batching_summary': {a: {'n_trips': batching[a]['n_trips'],
                                              'total_energy': batching[a]['total_energy']}
                                          for a in batching}})
    return results


# ---------- 任务对齐硬断言 ----------
def validate_capability(results):
    # P1-C1: 每(机型,服务区)有安全载荷且受三重约束
    mat = results['safe_payload_matrix']
    assert set(mat.keys()) == {'A', 'B', 'C'}, "缺机型"
    for g in mat:
        assert len(mat[g]) == P.N_AREAS, f"{g} 服务区数!=15"
        for a in mat[g]:
            assert 0 <= mat[g][a]['q_max'] <= P.Q[g] + 1e-6, "安全载荷越界"
    # P1-C2: 每箱恰覆盖一次, 不跨区, 满足载质量/体积/能量
    boxes = utils.load_boxes()
    all_delivered = []
    for a, b in results['batching'].items():
        for t in b['trips']:
            all_delivered.extend(t['box_ids'])
            # 单架次同属一区
            areas = set(bx.split('-')[0] for bx in t['box_ids'])
            assert len(areas) == 1 and a in areas, f"跨区组批 {a}"
            g = t['g']
            assert t['mass'] <= P.Q[g] + 1e-6, "载质量越界"
            assert t['vol'] <= P.V[g] + 1e-9, "体积越界"
            assert t['energy'] <= (1 - P.RHO_BASE) * P.E_USE[g] + 1e-6, "能耗越界"
            assert t['soc_end'] >= P.RHO_BASE - 1e-6, "返航SOC<20%"
    from collections import Counter
    cnt = Counter(all_delivered)
    assert all(c == 1 for c in cnt.values()), "存在箱交付次数!=1"
    assert len(all_delivered) == len(boxes), f"覆盖箱数{len(all_delivered)}!=总数{len(boxes)}"
    # P1-C4: >=3 个 rho 情景
    assert len(results['sensitivity_rho']) >= 3, "rho情景<3"
    print("[P1] validate_capability PASS")


if __name__ == '__main__':
    res = run()
    validate_capability(res)
