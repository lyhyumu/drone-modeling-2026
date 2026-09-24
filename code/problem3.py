# -*- coding: utf-8 -*-
"""问题三：通信约束下运输-中继联合调度。
连续轨迹自适应加密通信诊断(P3-C1) + 中继双段同时可用(P3-C2) +
中继悬停调度(P3-C3) + 联合多目标(P3-C4)。运输方案冻结自问题二。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import json
import numpy as np
import params as P
import geo, comm, energy, soc, utils, trajectory

EPS_T = 2.0                 # 自适应加密时间分辨率 s
G01 = comm.g01_pos()


def direct_available(pos):
    """运输机位置 pos 与 G01 直连可用？"""
    ok, Lp, margin = comm.link_available(pos, G01, comm.LMAX_U_G01)
    return ok, margin


def _adaptive_direct_intervals(segs):
    """自适应加密采样：沿轨迹找直连可用/不可用切换，返回 gap 区间 [(t0,t1)]。
    强制检查点=阶段端点，在检查点间二分加密到状态一致或 <EPS_T。"""
    t_start, t_end = trajectory.traj_bounds(segs)
    # 采样点: 每段端点 + 段内按 EPS_T 网格
    check_ts = set()
    for s in segs:
        check_ts.add(s['t0']); check_ts.add(s['t1'])
        n = max(2, int((s['t1'] - s['t0']) / 5.0) + 1)   # 段内 5s 网格
        for tt in np.linspace(s['t0'], s['t1'], n):
            check_ts.add(float(tt))
    ts = sorted(check_ts)
    # 逐点直连可用性
    avail = {}
    for tt in ts:
        pos = trajectory.pos_at(segs, tt)
        avail[tt], _ = direct_available(pos)
    # 找不可用区间(gap): 相邻状态不同处二分加密定位切换点
    gaps = []
    in_gap = not avail[ts[0]]
    gap_start = ts[0] if in_gap else None
    for i in range(1, len(ts)):
        a, b = avail[ts[i-1]], avail[ts[i]]
        if a != b:
            # 二分定位切换时刻
            lo, hi = ts[i-1], ts[i]
            while hi - lo > EPS_T:
                mid = 0.5*(lo+hi)
                av, _ = direct_available(trajectory.pos_at(segs, mid))
                if av == a:
                    lo = mid
                else:
                    hi = mid
            switch = hi
            if a and not b:               # 进入 gap
                gap_start = switch; in_gap = True
            elif (not a) and b:           # 退出 gap
                if gap_start is not None:
                    gaps.append((gap_start, switch)); in_gap = False; gap_start = None
    if in_gap and gap_start is not None:
        gaps.append((gap_start, ts[-1]))
    return gaps, t_start, t_end


def build_relay_candidates():
    """构造中继候选悬停状态(pos, H离地) — DEM降采样点+服务区上空+航段中点，分层高度。
    预计算回传 A_RG(与运输机无关)，仅保留回传可用的候选。"""
    cands = []
    heights = [50, 100, 150, 200, 250, 300]
    pts = []
    # 服务区上空 + O01
    for s in P.SERVICE_AREAS:
        pts.append((s['lon'], s['lat'], s['alt_m']))
    pts.append((P.O01['lon'], P.O01['lat'], P.O01['alt_m']))
    # 航段中点(O01-Si)
    for s in P.SERVICE_AREAS:
        mlon = 0.5*(P.O01['lon']+s['lon']); mlat = 0.5*(P.O01['lat']+s['lat'])
        pts.append((mlon, mlat, geo.dem_at_lonlat(mlon, mlat)))
    # DEM 粗网格降采样(区域内 6x6)
    lon_grid = np.linspace(min(s['lon'] for s in P.SERVICE_AREAS),
                           max(s['lon'] for s in P.SERVICE_AREAS), 6)
    lat_grid = np.linspace(min(s['lat'] for s in P.SERVICE_AREAS),
                           max(s['lat'] for s in P.SERVICE_AREAS), 6)
    for lo in lon_grid:
        for la in lat_grid:
            pts.append((lo, la, geo.dem_at_lonlat(lo, la)))
    for (lon, lat, ground) in pts:
        for H in heights:
            hover_pos = (lon, lat, ground + H)      # 绝对海拔 = 地面 + 离地H
            ok_rg, Lp_rg, margin_rg = comm.link_available(hover_pos, G01, comm.LMAX_R_G01)
            if ok_rg:                                # 仅保留回传可用
                cands.append({'lon': lon, 'lat': lat, 'agl': H, 'abs_alt': ground + H,
                              'backhaul_margin': round(margin_rg, 3)})
    return cands


def cover_gap_with_relay(segs, gap, relay_cands):
    """对 gap 区间贪心选一个中继悬停状态: 使覆盖时长最大(双段同时可用)。
    双段: A_UR(t) AND A_RG(t)。返回最优候选 + 覆盖诊断。"""
    t0, t1 = gap
    ts = np.arange(t0, t1 + EPS_T, EPS_T)
    best = None
    for c in relay_cands:
        hover_pos = (c['lon'], c['lat'], c['abs_alt'])
        covered = 0
        min_access_margin = 1e9
        ok_all = True
        for tt in ts:
            upos = trajectory.pos_at(segs, tt)
            ok_ur, Lp_ur, m_ur = comm.link_available(upos, hover_pos, comm.LMAX_U_R)
            # 双段 AND: 接入 AND 回传(回传已预筛，此处再校验)
            ok_rg, _, m_rg = comm.link_available(hover_pos, G01, comm.LMAX_R_G01)
            if ok_ur and ok_rg:
                covered += 1
                min_access_margin = min(min_access_margin, m_ur)
            else:
                ok_all = False
        cov_frac = covered / len(ts)
        if best is None or cov_frac > best['cov_frac']:
            best = {'cand': c, 'cov_frac': cov_frac, 'full_cover': ok_all,
                    'min_access_margin': (round(min_access_margin,3) if min_access_margin<1e9 else None)}
    return best


def relay_energy(service_time_s, fly_time_s, h_plus):
    """中继架次能耗(式27): 爬升 + 巡航 + (悬停+通信)服务。"""
    e_up = P.RELAY_TAKEOFF_MASS * P.G_ACC * h_plus / (P.CLIMB_EFF * P.KWH_TO_J)
    e_cruise = P.RELAY_CRUISE_POWER_KW * fly_time_s / 3600.0
    e_service = (P.RELAY_HOVER_POWER_KW + P.RELAY_COMM_POWER_KW) * service_time_s / 3600.0
    return e_up + e_cruise + e_service


GAP_SAMPLE_STEP = 30.0     # gap 内采样步长 s(用于覆盖判定)


def diagnose_trip_gaps(trip, delay, nodes, o01):
    """展开轨迹诊断 gap，返回 [(gs,ge,[采样绝对时刻])], ts, te, segs。不选悬停点。"""
    g = trip['g']
    tau = trip['tau'] + delay
    seq = [o01] + [nodes[a] for a in trip['areas']] + [o01]
    segs = trajectory.build_trajectory(g, seq, tau)
    gaps, ts, te = _adaptive_direct_intervals(segs)
    gap_info = []
    for (gs, ge) in gaps:
        n = max(2, int((ge - gs) / GAP_SAMPLE_STEP) + 1)
        samp = list(np.linspace(gs, ge, n))
        gap_info.append((gs, ge, samp, segs))
    return gap_info, ts, te, segs


def covering_candidates_for_gap(segs, samp_times, relay_cands):
    """某 gap 的全覆盖候选悬停点集合(双段: 接入 AND 回传 在所有采样时刻均可用)。
    返回 [(cand_idx, min_access_margin)]。"""
    out = []
    g01 = G01
    for ci, c in enumerate(relay_cands):
        hp = (c['lon'], c['lat'], c['abs_alt'])
        ok_all = True
        min_m = 1e9
        for tt in samp_times:
            upos = trajectory.pos_at(segs, tt)
            ok_ur, _, m_ur = comm.link_available(upos, hp, comm.LMAX_U_R)
            if not ok_ur:
                ok_all = False; break
            min_m = min(min_m, m_ur)
        if ok_all:
            out.append((ci, round(min_m, 3)))
    return out


def _fly_energy_to(c, o01):
    """中继从 O01 飞到悬停点的爬升高度、飞行时间、往返能耗基。"""
    _, _, h_plus, _, d = energy.segment_metrics(
        'A', 0.0, o01, {'id': 'RELAY', 'lon': c['lon'], 'lat': c['lat'],
                        'alt_m': c['abs_alt'] - P.WORK_ALT_OFFSET_M})
    fly_t = d / P.RELAY_V_CRUISE + h_plus / P.RELAY_V_CLIMB
    return h_plus, fly_t


def schedule_relays_setcover(tasks, relay_cands, o01):
    """relay-centric 调度：每个 task=(一段 gap 覆盖需求)含可行悬停候选集。
    贪心用 <=2 架物理中继覆盖全部 task；一架中继一悬停点可同时服务多运输机。
    tasks: [{'trip','gs','ge','cand_set':set(cand_idx)}]。
    返回 assignments[task_idx]=(relay_id, cand_idx) 或 None(不可行)。"""
    tasks = sorted(tasks, key=lambda x: x['gs'])
    relays = {f"R0{i+1}": {'free_at': -1e9, 'pos': None, 'pos_until': -1e9}
              for i in range(P.RELAY_N_UNITS)}
    assign = {}
    for ti, t in enumerate(tasks):
        placed = False
        # 1) 复用已在可行位置且时段衔接的中继(共享, 无 turnaround)
        for rid, r in relays.items():
            if r['pos'] in t['cand_set'] and r['free_at'] <= t['gs'] + 1e-6:
                r['free_at'] = max(r['free_at'], t['ge'])
                assign[ti] = (rid, r['pos']); placed = True; break
        if placed:
            continue
        # 2) 选一个可行悬停点重新部署空闲中继, 优先选能覆盖最多后续 task 的点(集合覆盖启发)
        best_rid = None; best_cand = None; best_score = -1
        for rid, r in sorted(relays.items(), key=lambda kv: kv[1]['free_at']):
            for ci in t['cand_set']:
                same = (r['pos'] == ci)
                need_free = t['gs'] - (0 if same else P.RELAY_TURNAROUND_S)
                if r['free_at'] <= need_free + 1e-6:
                    # 评分: 该候选点还能覆盖多少后续重叠 task
                    score = sum(1 for tt in tasks[ti+1:]
                                if ci in tt['cand_set'] and tt['gs'] < t['ge'])
                    if score > best_score:
                        best_score = score; best_rid = rid; best_cand = ci
        if best_rid is not None:
            r = relays[best_rid]
            r['free_at'] = t['ge']; r['pos'] = best_cand
            assign[ti] = (best_rid, best_cand); placed = True
        if not placed:
            return None
    return assign


def run(time_limit=300):
    utils.set_all_seeds()
    p2 = json.loads((utils.FIGDIR/'problem_2_results.json').read_text(encoding='utf-8'))
    nodes = utils.nodes_by_id()
    o01 = nodes[P.O01['id']]
    relay_cands = build_relay_candidates()
    print(f"[P3] 中继候选悬停状态(回传可用) {len(relay_cands)} 个", flush=True)

    # 硬时限 slack: 每架次可延迟上限(不违反医疗/首批)
    boxes = utils.load_boxes(); blk = {b['box_id']: b for b in boxes}
    dmap = {}
    for d in p2['deliveries']:
        dmap.setdefault(d['rid'], []).append(d)
    def trip_slack(rid):
        sl = 1e9
        for d in dmap.get(rid, []):
            b = blk[d['box_id']]
            dl = b['t_exp'] if b['is_med'] else None
            if b['is_first'] and b['t_first']:
                dl = b['t_first'] if dl is None else min(dl, b['t_first'])
            if dl is not None:
                sl = min(sl, dl - d['t_deliver'])
        return sl

    # 诊断缓存(rid,delay)->(gaps_with_cand, ts, te)
    diag_cache = {}
    def diag_trip(trip, delay):
        key = (trip['rid'], round(delay, 1))
        if key not in diag_cache:
            gap_info, ts, te, segs = diagnose_trip_gaps(trip, delay, nodes, o01)
            gl = []
            for (gs, ge, samp, sg) in gap_info:
                cov = covering_candidates_for_gap(sg, samp, relay_cands)
                gl.append({'gs': gs, 'ge': ge, 'cand_set': {ci for ci, _ in cov},
                           'margin_map': {ci: mg for ci, mg in cov}})
            diag_cache[key] = (gl, ts, te)
        return diag_cache[key]

    def reposition_time(ci_a, ci_b):
        if ci_a is None or ci_a == ci_b:
            return 0.0
        a, b = relay_cands[ci_a], relay_cands[ci_b]
        d = geo.horizontal_dist(a['lon'], a['lat'], b['lon'], b['lat'])
        dh = abs(a['abs_alt'] - b['abs_alt'])
        return d / P.RELAY_V_CRUISE + dh / P.RELAY_V_CLIMB

    def build_and_color(delays):
        """给定各架次延迟, 诊断->选位->2着色, 返回(方案数据, 重定位冲突列表)。"""
        tasks = []; trip_gapinfo = {}; total_gap = 0.0
        for trip in p2['trips']:
            gl, ts, te = diag_trip(trip, delays[trip['rid']])
            trip_gapinfo[trip['rid']] = {'ts': ts, 'te': te, 'gaps': []}
            for gg in gl:
                total_gap += (gg['ge'] - gg['gs'])
                task = {'trip': trip['rid'], 'g': trip['g'], 'gs': gg['gs'], 'ge': gg['ge'],
                        'cand_set': gg['cand_set'], 'margin_map': gg['margin_map'],
                        'fully_covered': len(gg['cand_set']) > 0}
                tasks.append(task); trip_gapinfo[trip['rid']]['gaps'].append(task)
        stasks = sorted(tasks, key=lambda x: x['gs'])
        # 阶段1: 选位, 重叠 task 尽量共享位置
        pos_of = {}
        for ti, t in enumerate(stasks):
            if not t['cand_set']:
                pos_of[ti] = None; continue
            reuse = None
            for tj in range(ti):
                cj = pos_of.get(tj)
                if cj is not None and cj in t['cand_set'] and \
                   stasks[tj]['gs'] < t['ge'] and stasks[tj]['ge'] > t['gs']:
                    reuse = cj; break
            pos_of[ti] = reuse if reuse is not None else max(
                t['cand_set'], key=lambda ci: sum(1 for tt in stasks
                    if ci in tt['cand_set'] and tt['gs'] < t['ge'] and tt['ge'] > t['gs']))
        # 阶段2: 合并同位置块 + 区间2着色
        by_pos = {}
        for ti, t in enumerate(stasks):
            by_pos.setdefault(pos_of[ti], []).append((t['gs'], t['ge'], ti))
        blocks = []
        for ci, iv in by_pos.items():
            if ci is None:
                continue
            iv.sort(); cs, ce, ct = iv[0][0], iv[0][1], [iv[0][2]]
            for (s, e, ti) in iv[1:]:
                if s <= ce + 1e-6:
                    ce = max(ce, e); ct.append(ti)
                else:
                    blocks.append([ci, cs, ce, ct]); cs, ce, ct = s, e, [ti]
            blocks.append([ci, cs, ce, ct])
        blocks.sort(key=lambda b: b[1])
        relays = {f"R0{i+1}": {'free_at': -1e9, 'pos': None} for i in range(P.RELAY_N_UNITS)}
        ti_relay = {}; conflicts = []
        for blk_ in blocks:
            ci, bs, be, tis = blk_
            chosen = None
            for rid, r in relays.items():
                if r['pos'] == ci and r['free_at'] <= bs + 1e-6:
                    chosen = rid; break
            if chosen is None:
                for rid, r in sorted(relays.items(), key=lambda kv: kv[1]['free_at']):
                    if r['free_at'] + reposition_time(r['pos'], ci) <= bs + 1e-6:
                        chosen = rid; break
            if chosen is None:
                chosen = min(relays, key=lambda rid: relays[rid]['free_at'])
                need = reposition_time(relays[chosen]['pos'], ci)
                if relays[chosen]['pos'] != ci and relays[chosen]['free_at'] + need > bs + 1e-6:
                    # 记录冲突: 该块所属架次(取最早 gs 的 task)需延迟 deficit
                    deficit = relays[chosen]['free_at'] + need - bs
                    late_trip = min((stasks[ti]['trip'] for ti in tis),
                                    key=lambda r: 0)  # 该块架次
                    conflicts.append({'trip': stasks[tis[0]]['trip'], 'deficit': deficit, 'bs': bs})
            relays[chosen]['free_at'] = be; relays[chosen]['pos'] = ci
            for ti in tis:
                ti_relay[ti] = chosen
        return {'stasks': stasks, 'pos_of': pos_of, 'ti_relay': ti_relay,
                'trip_gapinfo': trip_gapinfo, 'total_gap': total_gap}, conflicts

    # 迭代: 若有重定位冲突, 延迟冲突架次(在 slack 内)后重构
    delays = {t['rid']: 0.0 for t in p2['trips']}
    sol, conflicts = build_and_color(delays)
    for rnd in range(15):
        if not conflicts:
            break
        c = max(conflicts, key=lambda x: x['deficit'])
        rid = c['trip']
        avail = trip_slack(rid) - delays[rid]
        if avail > 30:
            delays[rid] += min(avail, c['deficit'] + 30)
            sol, conflicts = build_and_color(delays)
            print(f"[P3] 第{rnd+1}轮staggering: 延迟trip{rid}, 剩余冲突{len(conflicts)}", flush=True)
        else:
            print(f"[P3] trip{rid} slack不足, 保留 {len(conflicts)} 处重定位紧约束(操作性备注)", flush=True)
            break

    stasks = sol['stasks']; pos_of = sol['pos_of']; ti_relay = sol['ti_relay']
    trip_gapinfo = sol['trip_gapinfo']; total_gap_time = sol['total_gap']
    n_reposition_tight = len(conflicts)
    task_assign = {}
    for ti, t in enumerate(stasks):
        task_assign[(t['trip'], round(t['gs'], 3))] = (ti_relay.get(ti, 'R01'), pos_of[ti])

    relay_missions = []; comm_timeline = []; uncovered = []
    total_covered_time = 0.0; relay_energy_total = 0.0; n_relay_trips = 0
    for trip in p2['trips']:
        info = trip_gapinfo[trip['rid']]; phases = []; cursor = info['ts']
        for task in info['gaps']:
            gs, ge = task['gs'], task['ge']
            if gs > cursor + 1e-6:
                phases.append({'phase':'direct','t0':round(cursor,1),'t1':round(gs,1),'mode':'direct'})
            gap_len = ge - gs
            rid_ci = task_assign.get((trip['rid'], round(gs, 3)))
            if rid_ci and rid_ci[1] is not None:
                relay_id, ci = rid_ci; c = relay_cands[ci]
                h_plus, fly_t = _fly_energy_to(c, o01)
                e_relay = relay_energy(gap_len, fly_t*2, h_plus)
                total_covered_time += gap_len; relay_energy_total += e_relay; n_relay_trips += 1
                soc_end = 1 - e_relay / P.RELAY_E_COMPONENT
                mission = {'relay_trip_id': f"RT{n_relay_trips:02d}", 'relay_unit': relay_id,
                           'serves_trip': trip['rid'],
                           'hover_lon': round(c['lon'],6), 'hover_lat': round(c['lat'],6),
                           'hover_agl': c['agl'], 'hover_abs_alt': round(c['abs_alt'],1),
                           'service_start': round(gs,1), 'service_end': round(ge,1),
                           'link_setup_done': round(gs,1), 'backhaul_margin_db': c['backhaul_margin'],
                           'access_min_margin_db': task['margin_map'].get(ci),
                           'coverage_frac': 1.0, 'energy_kWh': round(e_relay,5),
                           'soc_end': round(soc_end,4), 'fully_covered': True}
                relay_missions.append(mission)
                phases.append({'phase':'relay','t0':round(gs,1),'t1':round(ge,1),'mode':'relay',
                               'relay_trip':mission['relay_trip_id'],'relay_unit':relay_id,'coverage_frac':1.0})
            else:
                uncovered.append({'trip':trip['rid'],'gap':(round(gs,1),round(ge,1)),'cov_frac':0})
                phases.append({'phase':'outage','t0':round(gs,1),'t1':round(ge,1),'mode':'none'})
            cursor = ge
        if info['te'] > cursor + 1e-6:
            phases.append({'phase':'direct','t0':round(cursor,1),'t1':round(info['te'],1),'mode':'direct'})
        comm_timeline.append({'trip_rid':trip['rid'],'g':trip['g'],'tau':trip['tau']+delays[trip['rid']],
                              'delay':delays[trip['rid']],'traj_start':round(info['ts'],1),
                              'traj_end':round(info['te'],1),'n_gaps':len(info['gaps']),'phases':phases})

    tpts = sorted(set([m['service_start'] for m in relay_missions]+[m['service_end'] for m in relay_missions]))
    distinct_pos_peak = 0; relay_peak = 0
    for t in tpts:
        act = [m for m in relay_missions if m['service_start']<=t<m['service_end']]
        distinct_pos_peak = max(distinct_pos_peak, len(set((round(m['hover_lon'],5),round(m['hover_lat'],5),m['hover_agl']) for m in act)))
        relay_peak = max(relay_peak, len(set(m['relay_unit'] for m in act)))

    transport_cmax = max(t['tau']+delays[t['rid']]+int(np.ceil(t['D_r'])) for t in p2['trips'])
    relay_return = [m['service_end'] + P.RELAY_TURNAROUND_S for m in relay_missions]
    joint_cmax = max([transport_cmax] + relay_return) if relay_return else transport_cmax

    results = {
        'seed': P.SEED, 'frozen_from': 'problem_2_results.json',
        'n_transport_trips': len(p2['trips']), 'n_relay_candidates': len(relay_cands),
        'trip_stagger_delays': {str(k): round(v,1) for k, v in delays.items() if v > 0},
        'total_stagger_delay_s': round(sum(delays.values()),1),
        'comm_diagnosis': {
            'total_gap_time_s': round(total_gap_time,1),
            'total_covered_time_s': round(total_covered_time,1),
            'coverage_rate': round(total_covered_time/total_gap_time,4) if total_gap_time>0 else 1.0,
            'n_trips_with_gap': sum(1 for c in comm_timeline if c['n_gaps']>0),
            'n_uncovered_segments': len(uncovered),
            'relay_label_peak': relay_peak, 'distinct_position_peak': distinct_pos_peak,
            'n_reposition_tight': n_reposition_tight,
            'relay_units_available': P.RELAY_N_UNITS,
            'relay_constraint_ok': distinct_pos_peak <= P.RELAY_N_UNITS and relay_peak <= P.RELAY_N_UNITS and n_reposition_tight == 0,
        },
        'joint_objective': {
            'transport_cmax_s': transport_cmax, 'joint_cmax_s': round(joint_cmax,1),
            'n_relay_trips': n_relay_trips, 'relay_energy_kWh': round(relay_energy_total,5),
            'transport_energy_kWh': p2['objective']['total_energy'],
            'total_energy_kWh': round(p2['objective']['total_energy']+relay_energy_total,4),
            'weighted_delay': p2['objective']['phi1_weighted_delay'],
        },
        'relay_missions': relay_missions, 'comm_timeline': comm_timeline, 'uncovered': uncovered,
    }
    utils.save_json(utils.FIGDIR/'problem_3_results.json', results)
    utils.save_json(utils.OUTDIR/'problem3_relay.json',
                    {'relay_missions': relay_missions, 'joint_objective': results['joint_objective'],
                     'comm_diagnosis': results['comm_diagnosis'], 'trip_stagger_delays': results['trip_stagger_delays']})
    print(f"[P3] gap={total_gap_time:.0f}s 覆盖率={results['comm_diagnosis']['coverage_rate']:.3f} "
          f"中继架次={n_relay_trips} 并发位置={distinct_pos_peak} 标签并发={relay_peak} "
          f"重定位紧约束={n_reposition_tight} 总staggering={sum(delays.values()):.0f}s 联合完工={joint_cmax:.0f}s", flush=True)
    return results



def validate_capability(results):
    # P3-C1: 无未覆盖中断(全程连续通信) - 允许覆盖率报告, 但硬要求 outage=0
    outages = [u for u in results['uncovered'] if u.get('cov_frac',0)==0]
    assert not outages, f"[连续通信违反] {len(outages)} 段完全无覆盖(直连+中继均不可用)"
    # P3-C2: 中继悬停 agl<=300 且回传可用(双段)
    for m in results['relay_missions']:
        assert m['hover_agl'] <= P.RELAY_MAX_AGL + 1e-6, "悬停agl>300"
        assert m['backhaul_margin_db'] >= -1e-6, "回传不可用"
        if m['access_min_margin_db'] is not None:
            assert m['access_min_margin_db'] >= -1e-6, "接入不可用"
    # 中继数量硬约束: 并发不同悬停位置数(=物理中继数) <= 2 且 R01/R02 标签并发 <= 2
    cd = results['comm_diagnosis']
    assert cd['relay_constraint_ok'], \
        f"[中继数量违反] 并发悬停位置峰值{cd['distinct_position_peak']}/标签并发{cd['relay_label_peak']}>{cd['relay_units_available']}架"
    # P3-C4: 联合完工含中继, 联合完工>=运输完工
    jo = results['joint_objective']
    assert jo['joint_cmax_s'] >= jo['transport_cmax_s'] - 1e-6, "联合完工<运输完工(新增约束应不放松)"
    assert 'relay_energy_kWh' in jo and 'n_relay_trips' in jo, "缺中继维度"
    print("[P3] validate_capability PASS")


if __name__ == '__main__':
    res = run()
    validate_capability(res)
