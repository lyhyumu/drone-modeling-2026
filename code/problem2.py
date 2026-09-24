# -*- coding: utf-8 -*-
"""问题二：异构多点多架次调度 CP-SAT 主模型。
决策: 路线选择 z_r / 起飞时刻 τ_r / 机型-实体机指派 / 电池池(累积资源)。
硬约束: 每箱覆盖1次, 医疗/首批时限, 同型机与电池同时占用<=库存。
词典序目标: 加权延误 ≻ 完工 ≻ 能耗 ≻ 架次数。能力 P2-C1..C4。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import json
import numpy as np
from ortools.sat.python import cp_model
import params as P
import geo, energy, soc, utils, routes

HORIZON = 40000            # 时间上界 s（首批<=10800, 期望<=18000, 多架次串行留余量）


def _box_lookup(boxes):
    return {b['box_id']: b for b in boxes}


def build_and_solve(pool, boxes, time_limit=300, w_makespan_phase=True,
                    initial_trips=None):
    """构建 CP-SAT 模型并按词典序求解。返回选中路线方案。"""
    blk = _box_lookup(boxes)
    R = len(pool)
    m = cp_model.CpModel()

    # 决策: z_r 选用路线
    z = [m.NewBoolVar(f"z_{r}") for r in range(R)]
    # 起飞时刻 τ_r (整数秒)
    tau = [m.NewIntVar(0, HORIZON, f"tau_{r}") for r in range(R)]

    # (C1) 每箱恰覆盖一次
    box_routes = {bid: [] for bid in blk}
    for r, rt in enumerate(pool):
        for bid in rt['box_ids']:
            box_routes[bid].append(r)
    for bid, rs in box_routes.items():
        m.Add(sum(z[r] for r in rs) == 1)

    # 交付时刻 t_deliver(c) = τ_r + δ_{c,r}（仅对选中路线有效）
    # 硬时限: 对含硬时限箱的路线, 收紧 τ_r 上界
    for r, rt in enumerate(pool):
        g = rt['g']
        latest_tau = HORIZON
        for bid in rt['box_ids']:
            b = blk[bid]
            deadline = None
            if b['is_med']:
                deadline = b['t_exp']            # 医疗: 期望时间为硬
            if b['is_first'] and b['t_first'] is not None:
                deadline = b['t_first'] if deadline is None else min(deadline, b['t_first'])
            if deadline is not None:
                dl = deadline - rt['delta'][bid]
                latest_tau = min(latest_tau, dl)
        if latest_tau < 0:
            # 该路线无法满足硬时限 -> 禁用
            m.Add(z[r] == 0)
        else:
            # τ_r <= latest_tau 当 z_r=1
            m.Add(tau[r] <= int(latest_tau)).OnlyEnforceIf(z[r])

    # 资源: 同机型实体机 与 电池池, 用累积约束(容量=库存)
    # 运输机(按机型): 占用区间 [τ_r, τ_r+D_r]
    # 电池(按机型): 占用区间 [τ_r, τ_r+D_r+t_charge(soc_end)]
    intervals_uav = {g: [] for g in ['A', 'B', 'C']}
    intervals_bat = {g: [] for g in ['A', 'B', 'C']}
    for r, rt in enumerate(pool):
        g = rt['g']
        Dr = int(np.ceil(rt['D_r']))
        end_uav = m.NewIntVar(0, HORIZON + Dr, f"eu_{r}")
        m.Add(end_uav == tau[r] + Dr)
        iv_u = m.NewOptionalIntervalVar(tau[r], Dr, end_uav, z[r], f"ivu_{r}")
        intervals_uav[g].append(iv_u)
        # 电池占用 = 任务 + 充电至100%
        tchg = int(np.ceil(soc.t_charge(rt['soc_end'], P.T_FULL_BAT[g])))
        Dbat = Dr + tchg
        end_bat = m.NewIntVar(0, HORIZON + Dbat, f"eb_{r}")
        m.Add(end_bat == tau[r] + Dbat)
        iv_b = m.NewOptionalIntervalVar(tau[r], Dbat, end_bat, z[r], f"ivb_{r}")
        intervals_bat[g].append(iv_b)
    for g in ['A', 'B', 'C']:
        m.AddCumulative(intervals_uav[g], [1]*len(intervals_uav[g]), P.FLEET_INV[g])
        m.AddCumulative(intervals_bat[g], [1]*len(intervals_bat[g]), P.BAT_INV[g])

    # 目标构造
    # Φ1 加权延误 = Σ w_c * max(0, t_deliver - T_exp) 对非硬箱(软)
    delay_terms = []
    for r, rt in enumerate(pool):
        g = rt['g']
        for bid in rt['box_ids']:
            b = blk[bid]
            if b['is_med'] or b['is_first']:
                continue                          # 硬箱不计延误(已作硬约束)
            # t_deliver = τ_r + δ; 延误 = max(0, t_deliver - t_exp)
            td = m.NewIntVar(0, HORIZON, f"td_{r}_{bid}")
            m.Add(td == tau[r] + int(round(rt['delta'][bid]))).OnlyEnforceIf(z[r])
            m.Add(td == 0).OnlyEnforceIf(z[r].Not())
            over = m.NewIntVar(0, HORIZON, f"ov_{r}_{bid}")
            m.Add(over >= td - int(b['t_exp'])).OnlyEnforceIf(z[r])
            m.Add(over == 0).OnlyEnforceIf(z[r].Not())
            delay_terms.append((int(b['w']), over))
    phi1 = m.NewIntVar(0, HORIZON * 10000, "phi1")
    m.Add(phi1 == sum(w * ov for w, ov in delay_terms))

    # Φ2 完工时间 = max over selected routes (τ_r + D_r)  [所有机返O01最晚]
    cmax = m.NewIntVar(0, HORIZON + 50000, "cmax")
    for r, rt in enumerate(pool):
        Dr = int(np.ceil(rt['D_r']))
        m.Add(cmax >= tau[r] + Dr).OnlyEnforceIf(z[r])

    # Φ4 架次数
    ntrips = m.NewIntVar(0, R, "ntrips")
    m.Add(ntrips == sum(z))

    etot = m.NewIntVar(0, 10**7, "etot")
    m.Add(etot == sum(int(round(pool[r]['energy']*1000)) * z[r] for r in range(R)))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = P.SEED
    status_records = []

    if initial_trips is not None:
        old = {(t['g'], tuple(t['areas']), tuple(sorted(t['box_ids']))): t
               for t in initial_trips}
        matched = set()
        for r, route in enumerate(pool):
            key = (route['g'], tuple(route['areas']),
                   tuple(sorted(route['box_ids'])))
            prior = old.get(key)
            m.AddHint(z[r], int(prior is not None))
            m.AddHint(tau[r], int(prior['tau']) if prior is not None else 0)
            if prior is not None:
                matched.add(key)
        if matched != set(old):
            raise RuntimeError(f"initial solution missing {len(set(old)-matched)} routes")

    def _record_status(phase, status, target, limit_s):
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"{phase} has no feasible solution: {solver.StatusName(status)}")
        status_records.append({
            'phase': phase, 'status': solver.StatusName(status),
            'incumbent': solver.Value(target),
            'best_bound': solver.BestObjectiveBound(),
            'wall_time_s': solver.WallTime(), 'time_limit_s': limit_s,
        })

    def _hint_from(sol_vals):
        """用上一级解做 warm start，加速下一级收敛。"""
        m.ClearHints()
        for r in range(R):
            m.AddHint(z[r], sol_vals['z'][r])
            m.AddHint(tau[r], sol_vals['tau'][r])

    def _snap():
        return {'z': [solver.Value(z[r]) for r in range(R)],
                'tau': [solver.Value(tau[r]) for r in range(R)]}

    # 词典序: L1 min Φ1
    solver.parameters.max_time_in_seconds = float(time_limit)
    m.Minimize(phi1)
    st = solver.Solve(m)
    _record_status('weighted_delay', st, phi1, float(time_limit))
    print(f"[P2-L1] status={solver.StatusName(st)} Φ1(加权延误)={solver.Value(phi1)}", flush=True)
    phi1_star = solver.Value(phi1)
    snap = _snap()
    m.Add(phi1 <= phi1_star)

    # L2 min Cmax
    solver.parameters.max_time_in_seconds = float(max(30, time_limit // 2))
    _hint_from(snap)
    m.Minimize(cmax)
    st = solver.Solve(m)
    _record_status('makespan', st, cmax, float(max(30, time_limit // 2)))
    print(f"[P2-L2] status={solver.StatusName(st)} Cmax(完工)={solver.Value(cmax)}s", flush=True)
    cmax_star = solver.Value(cmax)
    snap = _snap()
    m.Add(cmax <= cmax_star)

    # L3 min 能耗(整数化 mWh)
    _hint_from(snap)
    m.Minimize(etot)
    st = solver.Solve(m)
    _record_status('energy_mWh', st, etot, float(max(30, time_limit // 2)))
    print(f"[P2-L3] status={solver.StatusName(st)} 能耗={solver.Value(etot)/1000:.3f}kWh", flush=True)
    etot_star = solver.Value(etot)
    snap = _snap()
    m.Add(etot <= etot_star)

    # L4 min 架次数
    _hint_from(snap)
    m.Minimize(ntrips)
    st = solver.Solve(m)
    _record_status('n_trips', st, ntrips, float(max(30, time_limit // 2)))
    print(f"[P2-L4] status={solver.StatusName(st)} 架次数={solver.Value(ntrips)}", flush=True)

    # 提取解
    chosen = []
    for r in range(R):
        if solver.Value(z[r]) > 0.5:
            rt = dict(pool[r])
            rt['tau'] = solver.Value(tau[r])
            rt['return_time'] = solver.Value(tau[r]) + int(np.ceil(rt['D_r']))
            chosen.append(rt)
    chosen.sort(key=lambda x: x['tau'])
    return chosen, {
        'phi1_weighted_delay': solver.Value(phi1),
        'cmax_makespan': solver.Value(cmax),
        'total_energy': round(solver.Value(etot)/1000, 4),
        'n_trips': solver.Value(ntrips),
        'solver_diagnostics': status_records,
    }


def assign_units_and_batteries(chosen):
    """区间图着色：将同机型架次指派到具体实体机 + 电池组，满足占用不重叠。
    贪心：按开始时刻排序，分配最早空闲的机/电池。"""
    # 实体机 by type
    units = {'A': [u['id'] for u in P.FLEET if u['type']=='A'],
             'B': [u['id'] for u in P.FLEET if u['type']=='B'],
             'C': [u['id'] for u in P.FLEET if u['type']=='C']}
    bats = {g: [f"{g}BAT{i+1}" for i in range(P.BAT_INV[g])] for g in ['A','B','C']}
    unit_free = {u: 0 for g in units for u in units[g]}          # 机空闲时刻
    bat_free = {b: 0 for g in bats for b in bats[g]}             # 电池空闲时刻
    for rt in sorted(chosen, key=lambda x: x['tau']):
        g = rt['g']
        tau = rt['tau']; Dr = int(np.ceil(rt['D_r']))
        # 分配机: 选最早空闲且<=tau 的机(贪心取空闲最早)
        cand_u = sorted(units[g], key=lambda u: unit_free[u])
        u = cand_u[0]
        rt['unit'] = u
        unit_free[u] = tau + Dr
        # 分配电池: 占用 = 任务 + 充电
        tchg = int(np.ceil(soc.t_charge(rt['soc_end'], P.T_FULL_BAT[g])))
        cand_b = sorted(bats[g], key=lambda b: bat_free[b])
        bb = cand_b[0]
        rt['battery'] = bb
        bat_free[bb] = tau + Dr + tchg
        rt['charge_time'] = tchg
    return chosen


def resource_audit(chosen):
    """P2-C4: 逐架次重算质量/体积/能耗/SOC; 逐资源占用区间重叠数<=库存。"""
    audit = {'trip_violations': [], 'resource_peaks': {}}
    for rt in chosen:
        g = rt['g']
        if rt['mass'] > P.Q[g] + 1e-6:
            audit['trip_violations'].append(f"{rt['rid']} mass {rt['mass']}>{P.Q[g]}")
        if rt['vol'] > P.V[g] + 1e-9:
            audit['trip_violations'].append(f"{rt['rid']} vol超")
        if rt['energy'] > (1-P.RHO_BASE)*P.E_USE[g] + 1e-6:
            audit['trip_violations'].append(f"{rt['rid']} energy超")
        if rt['soc_end'] < P.RHO_BASE - 1e-6:
            audit['trip_violations'].append(f"{rt['rid']} soc<20%")
    # 逐机型峰值(区间最大重叠) - 机与电池
    for g in ['A','B','C']:
        # UAV
        ev = []
        for rt in chosen:
            if rt['g']!=g: continue
            Dr = int(np.ceil(rt['D_r']))
            ev.append((rt['tau'], 1)); ev.append((rt['tau']+Dr, -1))
        peak_u = _peak(ev)
        # battery
        evb = []
        for rt in chosen:
            if rt['g']!=g: continue
            Dr = int(np.ceil(rt['D_r']))
            tchg = int(np.ceil(soc.t_charge(rt['soc_end'], P.T_FULL_BAT[g])))
            evb.append((rt['tau'], 1)); evb.append((rt['tau']+Dr+tchg, -1))
        peak_b = _peak(evb)
        audit['resource_peaks'][g] = {'uav_peak': peak_u, 'uav_inv': P.FLEET_INV[g],
                                      'bat_peak': peak_b, 'bat_inv': P.BAT_INV[g],
                                      'uav_ok': peak_u<=P.FLEET_INV[g], 'bat_ok': peak_b<=P.BAT_INV[g]}
    return audit


def _peak(events):
    """区间事件最大同时重叠数。events=[(t,+1/-1)]。"""
    events.sort(key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


def run(time_limit=300):
    utils.set_all_seeds()
    boxes = utils.load_boxes()
    nodes = utils.nodes_by_id()
    pool, by_area = routes.build_route_pool(boxes, nodes)
    print(f"[P2] 候选路线池 {len(pool)} 条")
    chosen, obj = build_and_solve(pool, boxes, time_limit=time_limit)
    chosen = assign_units_and_batteries(chosen)
    audit = resource_audit(chosen)

    # 逐箱交付
    blk = _box_lookup(boxes)
    deliveries = []
    for rt in chosen:
        for bid in rt['box_ids']:
            td = rt['tau'] + rt['delta'][bid]
            b = blk[bid]
            deadline = b['t_first'] if (b['is_first'] and b['t_first']) else None
            hard = b['t_exp'] if b['is_med'] else deadline
            deliveries.append({'box_id': bid, 'area': b['area'], 'rid': rt['rid'],
                               'unit': rt['unit'], 't_deliver': round(td, 1),
                               't_exp': b['t_exp'], 't_first': b['t_first'],
                               'is_med': b['is_med'], 'is_first': b['is_first'],
                               'hard_deadline': hard,
                               'meets_hard': (hard is None or td <= hard + 1e-6)})

    trips_out = [{'rid': rt['rid'], 'g': rt['g'], 'unit': rt['unit'], 'battery': rt['battery'],
                  'tau': rt['tau'], 'return_time': rt['return_time'], 'areas': rt['areas'],
                  'box_ids': rt['box_ids'], 'mass': rt['mass'], 'vol': rt['vol'],
                  'energy': rt['energy'], 'soc_end': rt['soc_end'], 'D_r': rt['D_r'],
                  'charge_time': rt['charge_time']} for rt in chosen]

    results = {'seed': P.SEED, 'objective': obj, 'n_routes_pool': len(pool),
               'trips': trips_out, 'deliveries': deliveries, 'resource_audit': audit}
    utils.save_json(utils.FIGDIR / 'problem_2_results.json', results)
    utils.save_json(utils.OUTDIR / 'problem2_delivery.json',
                    {'objective': obj, 'deliveries': deliveries,
                     'n_med_ontime': sum(1 for d in deliveries if d['is_med'] and d['meets_hard']),
                     'n_first_ontime': sum(1 for d in deliveries if d['is_first'] and d['meets_hard'])})
    return results


def validate_capability(results):
    boxes = utils.load_boxes()
    # P2-C1: 6类决策存在 + 分段载荷(能耗非全程起飞载荷 - routes 已逐段)
    for t in results['trips']:
        assert 'unit' in t and 'battery' in t and 'tau' in t and 'areas' in t, "缺决策维度"
    # 每箱覆盖1次
    allb = [b for t in results['trips'] for b in t['box_ids']]
    from collections import Counter
    cnt = Counter(allb)
    assert all(c==1 for c in cnt.values()) and len(allb)==len(boxes), "箱覆盖!=1次"
    # P2-C2: 所有医疗/首批硬时限满足
    bad = [d for d in results['deliveries'] if (d['is_med'] or d['is_first']) and not d['meets_hard']]
    assert not bad, f"[硬时限违反] {len(bad)} 箱超期: {[d['box_id'] for d in bad[:5]]}"
    # P2-C4: 资源峰值<=库存
    for g, p in results['resource_audit']['resource_peaks'].items():
        assert p['uav_ok'] and p['bat_ok'], f"{g} 资源超库存"
    assert not results['resource_audit']['trip_violations'], "架次约束违反"
    print("[P2] validate_capability PASS")


if __name__ == '__main__':
    tl = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    res = run(time_limit=tl)
    validate_capability(res)
