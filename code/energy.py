# -*- coding: utf-8 -*-
"""时间/能耗计算器（式4-10）。等效航程范围口径(假设3) + 势能爬升(假设4)。
分段载荷动态更新(式9)，防"全程起飞载荷"。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import params as P
import geo


def L_of_q(g, q):
    """载荷 q 下等效航程 L_g(q) = L0 - (L0-LF)*(q/Qg)^1.5（式4）。"""
    q = max(0.0, min(q, P.Q[g]))
    return P.L0[g] - (P.L0[g] - P.LF[g]) * (q / P.Q[g]) ** 1.5


def seg_time(g, h_plus, d, h_minus):
    """航段飞行时间（式5，三阶段求和）。"""
    return h_plus / P.V_CLIMB[g] + d / P.V_CRUISE[g] + h_minus / P.V_DESCENT[g]


def seg_energy(g, q, h_plus, d):
    """航段总运输能耗 kWh（式6-8）：水平巡航能耗 + 爬升附加能耗，下降不计。
    q = 该航段剩余载荷(分段动态)。"""
    # 水平巡航能耗: 比能耗 = E_use/L(q)
    Lq = L_of_q(g, q)
    e_hor = P.E_USE[g] * d / Lq
    # 爬升附加能耗: 势能增量/爬升效率
    m_tot = P.EMPTY[g] + q                       # 全机质量(空载总质量+载荷)
    e_up = m_tot * P.G_ACC * h_plus / (P.CLIMB_EFF * P.KWH_TO_J)
    return e_hor + e_up


def segment_metrics(g, q, node_i, node_j):
    """给定机型/剩余载荷/两端节点，返回该航段 (时间s, 能耗kWh, h_plus, h_minus, d)。"""
    d = geo.horizontal_dist(node_i['lon'], node_i['lat'], node_j['lon'], node_j['lat'])
    Hc = geo.cruise_altitude(node_i['lon'], node_i['lat'], node_j['lon'], node_j['lat'])
    zi = geo.work_alt(node_i)
    zj = geo.work_alt(node_j)
    h_plus = max(0.0, Hc - zi)
    h_minus = max(0.0, Hc - zj)
    t = seg_time(g, h_plus, d, h_minus)
    e = seg_energy(g, q, h_plus, d)
    return t, e, h_plus, h_minus, d


def roundtrip_energy(g, q, o01, si):
    """单点往返 O01->Si->O01 总能耗（去程载 q，回程载 0）（式16）。"""
    _, e_go, _, _, _ = segment_metrics(g, q, o01, si)
    _, e_back, _, _, _ = segment_metrics(g, 0.0, si, o01)
    return e_go + e_back


def roundtrip_time(g, o01, si):
    """单点往返飞行时间（去+回，与载荷无关）。"""
    t_go, _, _, _, _ = segment_metrics(g, 0.0, o01, si)
    t_back, _, _, _, _ = segment_metrics(g, 0.0, si, o01)
    return t_go + t_back


def route_energy_time(g, load0, seq_nodes, deliver_amounts):
    """多点路线逐段更新剩余载荷（式9）计算总能耗/飞行时间。
    seq_nodes = [O01, S_a, S_b, ..., O01]（首尾为 O01）。
    deliver_amounts = 每个中间服务区投送质量列表（与 seq_nodes[1:-1] 对齐）。
    返回 (总能耗kWh, 总飞行时间s, 各段(能耗,时间), 各服务区到达飞行时刻列表)。"""
    q = load0
    tot_e = 0.0
    tot_t = 0.0
    segs = []
    arrival_fly_time = []          # 到达每个中间节点的累计飞行时间
    k = 0                          # 投送计数
    for idx in range(len(seq_nodes) - 1):
        ni, nj = seq_nodes[idx], seq_nodes[idx + 1]
        t, e, _, _, _ = segment_metrics(g, q, ni, nj)
        tot_e += e
        tot_t += t
        segs.append((e, t))
        # 到达 nj（若为服务区，非最终 O01）
        if idx + 1 < len(seq_nodes) - 1:
            arrival_fly_time.append(tot_t)
            q = max(0.0, q - deliver_amounts[k])   # 投送后减载
            k += 1
    return tot_e, tot_t, segs, arrival_fly_time


if __name__ == '__main__':
    # 自检: 零载/满载等效航程
    for g in ['A', 'B', 'C']:
        assert abs(L_of_q(g, 0) - P.L0[g]) < 1e-6, f"{g} 零载航程错"
        assert abs(L_of_q(g, P.Q[g]) - P.LF[g]) < 1e-6, f"{g} 满载航程错"
    print("[energy] 零/满载等效航程自检通过")
    # 单调性
    qs = np.linspace(0, P.Q['A'], 50)
    Ls = [L_of_q('A', q) for q in qs]
    assert all(Ls[i] >= Ls[i+1] for i in range(len(Ls)-1)), "L(q) 非单调递减"
    print("[energy] L(q) 单调递减自检通过")
    # 往返能耗示例
    o = P.O01; s1 = P.SERVICE_AREAS[0]
    for g in ['A', 'B', 'C']:
        e = roundtrip_energy(g, P.Q[g], o, s1)
        budget = (1 - P.RHO_BASE) * P.E_USE[g]
        print(f"[energy] {g}型 O01-S001-O01 满载往返能耗={e:.4f} kWh, 预算={budget:.4f} kWh")
