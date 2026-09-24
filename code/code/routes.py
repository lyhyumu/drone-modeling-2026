# -*- coding: utf-8 -*-
"""候选路线生成(问题二/三外层)：单区批次 + 多区最近邻/2-opt，逐段更新剩余载荷(式9)。
每条路线记录箱覆盖、能耗、占用时长、逐箱相对交付时刻 δ。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import itertools
import numpy as np
import params as P
import geo, energy, utils


def _route_metrics(g, node_seq, area_boxes_map):
    """给定机型 g、节点序列 [O01, S.., .., O01]、各中间区投送箱列表，
    逐段更新剩余载荷计算：总能耗、占用时长 D_r、逐箱相对交付时刻 δ。
    area_boxes_map: {area_id: [box dicts]} 该路线在各区投送的箱。"""
    mid_areas = [n['id'] for n in node_seq[1:-1]]
    # 起飞总载荷 = 所有箱质量之和
    all_boxes = []
    for aid in mid_areas:
        all_boxes.extend(area_boxes_map[aid])
    load0 = sum(b['mass'] for b in all_boxes)
    n_total = len(all_boxes)
    deliver_amounts = [sum(b['mass'] for b in area_boxes_map[aid]) for aid in mid_areas]
    tot_e, tot_fly, segs, arrival_fly = energy.route_energy_time(g, load0, node_seq, deliver_amounts)
    # 时间轴: prep + load_all 在 O01, 然后逐段飞行+交接
    t0 = P.PREP_S[g] + n_total * P.LOAD_PER_BOX_S[g]     # 起飞前(装载完成)
    delta = {}                                            # box_id -> 相对交付时刻(从架次开始)
    t_cursor = t0
    for k, aid in enumerate(mid_areas):
        # 飞到该区(段 k 的飞行时间)
        t_cursor += segs[k][1]
        # 交接(base + n_area*per)
        n_area = len(area_boxes_map[aid])
        t_cursor += P.HANDOVER_BASE_S[g] + n_area * P.HANDOVER_PER_BOX_S[g]
        for b in area_boxes_map[aid]:
            delta[b['box_id']] = t_cursor                # 该区所有箱在交接完成时交付
    # 返程段(最后一段, 无交接)
    t_cursor += segs[-1][1]
    D_r = t_cursor                                        # 占用总时长
    soc_end = 1 - tot_e / P.E_USE[g]
    total_vol = sum(b['vol'] for b in all_boxes)
    return {
        'g': g, 'areas': mid_areas, 'box_ids': [b['box_id'] for b in all_boxes],
        'mass': round(load0, 4), 'vol': round(total_vol, 5), 'energy': round(tot_e, 6),
        'fly_time': round(tot_fly, 2), 'D_r': round(D_r, 2), 'soc_end': round(soc_end, 5),
        'delta': {k: round(v, 2) for k, v in delta.items()}, 'n_boxes': n_total,
    }


def _feasible(m, g):
    """载质量/体积/能量/返航SOC 可行性过滤。"""
    if m['mass'] > P.Q[g] + 1e-9:
        return False
    if m['vol'] > P.V[g] + 1e-9:
        return False
    if m['energy'] > (1 - P.RHO_BASE) * P.E_USE[g] + 1e-9:
        return False
    if m['soc_end'] < P.RHO_BASE - 1e-9:
        return False
    return True


def gen_single_area_batches(area_id, boxes, nodes, cap_per_area=50,
                            allowed_types=('A', 'B', 'C')):
    """某区单区候选批次：全 singleton(保证可分) + 有限枚举的高质量多箱批次(按装载效率择优)。
    为控制主模型规模，多箱批次每区截断到 cap_per_area 条(按覆盖箱数降序、能耗升序)。"""
    o01 = nodes[P.O01['id']]
    si = nodes[area_id]
    n = len(boxes)
    idxs = list(range(n))
    seen = set()
    singles = []
    multis_by_type = {g: [] for g in allowed_types}
    for g in allowed_types:
        for r in range(1, n + 1):
            stop = True
            for combo in itertools.combinations(idxs, r):
                sub = [boxes[i] for i in combo]
                m = sum(b['mass'] for b in sub)
                v = sum(b['vol'] for b in sub)
                if m > P.Q[g] + 1e-9 or v > P.V[g] + 1e-9:
                    continue
                stop = False
                met = _route_metrics(g, [o01, si, o01], {area_id: sub})
                if not _feasible(met, g):
                    continue
                key = (g, area_id, tuple(sorted(b['box_id'] for b in sub)))
                if key in seen:
                    continue
                seen.add(key)
                (singles if r == 1 else multis_by_type[g]).append(met)
            if stop:
                break
    # 多箱批次择优截断：覆盖箱数多、能耗低优先(减架次)
    # Equal per-type cap prevents one aircraft class from consuming the
    # candidate budget of another before the optimization model sees it.
    selected_multis = []
    for g in allowed_types:
        multis_by_type[g].sort(key=lambda x: (-x['n_boxes'], x['energy']))
        selected_multis.extend(multis_by_type[g][:cap_per_area])
    return singles + selected_multis


def _dedup_dominated(routes):
    """删被支配路线：同覆盖箱集，(能耗,时长)均不劣者保留最优。"""
    best = {}
    for r in routes:
        # Never prune across aircraft classes: different fleet/battery
        # availability and delivery times make cross-type dominance invalid.
        key = (r['g'], tuple(sorted(r['box_ids'])))
        if key not in best:
            best[key] = r
        else:
            b = best[key]
            if (r['energy'], r['D_r']) < (b['energy'], b['D_r']):
                best[key] = r
    return list(best.values())


def gen_multi_area_routes(by_area, nodes, max_areas=3,
                          allowed_types=('C', 'B', 'A')):
    """多区最近邻路线：按机型 C/B 合并邻近区(最近邻+2opt思想)，投送各区全部箱。
    仅生成"整区打包"的多区路线作为省架次候选(箱粒度拆分由单区批次+主模型组合)。"""
    o01 = nodes[P.O01['id']]
    area_ids = [s['id'] for s in P.SERVICE_AREAS]
    # 区间米制坐标
    xy = {aid: geo.to_xy(nodes[aid]['lon'], nodes[aid]['lat']) for aid in area_ids}
    xyO = geo.to_xy(o01['lon'], o01['lat'])
    out = []
    for g in allowed_types:
        for base in area_ids:
            # 最近邻扩展
            chosen = [base]
            remaining = [a for a in area_ids if a != base]
            while len(chosen) < max_areas and remaining:
                last = xy[chosen[-1]]
                nxt = min(remaining, key=lambda a: np.hypot(xy[a][0]-last[0], xy[a][1]-last[1]))
                # 试加入
                trial = chosen + [nxt]
                boxmap = {aid: by_area[aid] for aid in trial}
                seq = [o01] + [nodes[a] for a in trial] + [o01]
                met = _route_metrics(g, seq, boxmap)
                if _feasible(met, g):
                    chosen = trial
                    remaining.remove(nxt)
                else:
                    break
            if len(chosen) >= 2:
                # 2-opt: 尝试反转顺序取更省能耗
                boxmap = {aid: by_area[aid] for aid in chosen}
                best_seq = [o01] + [nodes[a] for a in chosen] + [o01]
                best_met = _route_metrics(g, best_seq, boxmap)
                for perm in itertools.permutations(chosen):
                    seq = [o01] + [nodes[a] for a in perm] + [o01]
                    met = _route_metrics(g, seq, boxmap)
                    if _feasible(met, g) and met['energy'] < best_met['energy']:
                        best_met = met
                if _feasible(best_met, g):
                    out.append(best_met)
    return _dedup_dominated(out)


def build_route_pool(boxes, nodes, allowed_types=('A', 'B', 'C')):
    """构建候选路线池：单区批次(含singleton保证可分) + 多区打包。"""
    by_area = {}
    for b in boxes:
        by_area.setdefault(b['area'], []).append(b)
    pool = []
    for aid, bx in by_area.items():
        pool.extend(gen_single_area_batches(aid, bx, nodes,
                                            allowed_types=allowed_types))
    pool = _dedup_dominated(pool)
    multi_types = tuple(g for g in ('C', 'B', 'A') if g in allowed_types)
    pool.extend(gen_multi_area_routes(by_area, nodes,
                                      allowed_types=multi_types))
    # 赋 route id
    for i, r in enumerate(pool):
        r['rid'] = i
    return pool, by_area


if __name__ == '__main__':
    utils.set_all_seeds()
    boxes = utils.load_boxes()
    nodes = utils.nodes_by_id()
    pool, by_area = build_route_pool(boxes, nodes)
    print(f"[routes] 候选路线池 {len(pool)} 条")
    multi = [r for r in pool if len(r['areas']) > 1]
    print(f"  多区路线 {len(multi)} 条; 单区 {len(pool)-len(multi)} 条")
    print(f"  示例多区: {multi[0]['areas'] if multi else 'none'}")
