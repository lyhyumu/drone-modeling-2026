# -*- coding: utf-8 -*-
"""运输架次连续三维轨迹展开(爬升/巡航/下降/投送四阶段)，供问题三通信诊断。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import params as P
import geo


def build_trajectory(g, node_seq, tau0, delta_map=None):
    """展开架次轨迹为分阶段时间段。返回段列表，每段 dict:
    {phase, t0, t1, P0=(lon,lat,alt), P1=(lon,lat,alt)}。绝对时刻从 tau0 起。
    node_seq = [O01, S.., .., O01]。含各航段爬升-巡航-下降 + 服务区投送悬停。"""
    segs = []
    t = tau0
    for idx in range(len(node_seq) - 1):
        ni, nj = node_seq[idx], node_seq[idx + 1]
        d = geo.horizontal_dist(ni['lon'], ni['lat'], nj['lon'], nj['lat'])
        Hc = geo.cruise_altitude(ni['lon'], ni['lat'], nj['lon'], nj['lat'])
        zi = geo.work_alt(ni); zj = geo.work_alt(nj)
        h_plus = max(0.0, Hc - zi); h_minus = max(0.0, Hc - zj)
        t_cl = h_plus / P.V_CLIMB[g]
        t_cr = d / P.V_CRUISE[g]
        t_de = h_minus / P.V_DESCENT[g]
        # 爬升: 在 ni 上方垂直爬升(经纬不变, 海拔 zi->Hc)
        segs.append({'phase': 'climb', 't0': t, 't1': t + t_cl,
                     'P0': (ni['lon'], ni['lat'], zi), 'P1': (ni['lon'], ni['lat'], Hc)})
        t += t_cl
        # 巡航: 水平 ni->nj 于 Hc
        segs.append({'phase': 'cruise', 't0': t, 't1': t + t_cr,
                     'P0': (ni['lon'], ni['lat'], Hc), 'P1': (nj['lon'], nj['lat'], Hc)})
        t += t_cr
        # 下降: 在 nj 上方 Hc->zj
        segs.append({'phase': 'descent', 't0': t, 't1': t + t_de,
                     'P0': (nj['lon'], nj['lat'], Hc), 'P1': (nj['lon'], nj['lat'], zj)})
        t += t_de
        # 投送悬停(若 nj 为服务区, 非最终 O01)
        if idx + 1 < len(node_seq) - 1:
            n_area = 1
            t_hover = P.HANDOVER_BASE_S[g] + n_area * P.HANDOVER_PER_BOX_S[g]
            segs.append({'phase': 'deliver', 't0': t, 't1': t + t_hover,
                         'P0': (nj['lon'], nj['lat'], zj), 'P1': (nj['lon'], nj['lat'], zj)})
            t += t_hover
    return segs


def pos_at(segs, t):
    """给定轨迹段与绝对时刻 t，线性插值三维位置 (lon,lat,alt)。"""
    for s in segs:
        if s['t0'] <= t <= s['t1'] + 1e-9:
            span = s['t1'] - s['t0']
            frac = 0.0 if span <= 0 else (t - s['t0']) / span
            return tuple(s['P0'][k] + frac * (s['P1'][k] - s['P0'][k]) for k in range(3))
    # 越界返回端点
    if t < segs[0]['t0']:
        return segs[0]['P0']
    return segs[-1]['P1']


def traj_bounds(segs):
    return segs[0]['t0'], segs[-1]['t1']
