# -*- coding: utf-8 -*-
"""通信计算器（式12-14）。FSPL + 地形遮挡 + 双向 min 门限 + 链路可用性。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import params as P
import geo


def fspl_db(D_m):
    """自由空间损耗 dB（式12），D 以 km 计入。"""
    D_km = max(D_m, 1.0) / 1000.0
    return 32.45 + 20 * np.log10(P.FREQ_MHZ) + 20 * np.log10(D_km)


def path_loss_db(P_a, P_b):
    """总传播损耗 dB（式13）= FSPL + L_obs*遮挡指示。P=(lon,lat,abs_alt_m)。"""
    D = geo.dist_3d(P_a, P_b)
    blocked = geo.line_of_sight_blocked(P_a, P_b)
    return fspl_db(D) + (P.L_OBS_DB if blocked else 0.0), blocked, D


def L_max_directional(Pt_a, G_a, G_b):
    """方向 a->b 最大允许损耗（式14）：Pt_a + G_a + G_b - L_sys - P_th_b。"""
    return Pt_a + G_a + G_b - P.L_SYS_DB - P.P_TH_DBM


def L_max_bidir(ep_a, ep_b):
    """双向门限 = min(L_max^{a->b}, L_max^{b->a})（式14）。ep=(Pt,G) dict。"""
    ab = L_max_directional(ep_a['Pt_dBm'], ep_a['G_dBi'], ep_b['G_dBi'])
    ba = L_max_directional(ep_b['Pt_dBm'], ep_b['G_dBi'], ep_a['G_dBi'])
    return min(ab, ba)


# 预计算三类链路的双向门限
LMAX_U_G01 = L_max_bidir(P.COMM_UAV, P.COMM_G01)                       # 运输机<->G01 直连
LMAX_U_R = L_max_bidir(P.COMM_UAV, P.COMM_RELAY_ACCESS)                # 运输机<->中继接入
LMAX_R_G01 = L_max_bidir(P.COMM_RELAY_BACKHAUL, P.COMM_G01)            # 中继回传<->G01


def link_available(P_a, P_b, Lmax):
    """链路可用性 A=1 iff L_path<=Lmax。返回 (可用bool, L_path, 裕量dB)。"""
    Lp, blocked, D = path_loss_db(P_a, P_b)
    margin = Lmax - Lp
    return (Lp <= Lmax), Lp, margin


def g01_pos():
    """固定网关 G01 位置：与 O01 同址，天线离地 20 m。"""
    return (P.O01['lon'], P.O01['lat'], P.O01['alt_m'] + P.G01_ANTENNA_AGL)


if __name__ == '__main__':
    print(f"[comm] 双向门限 dB: U-G01={LMAX_U_G01:.2f}  U-R={LMAX_U_R:.2f}  R-G01={LMAX_R_G01:.2f}")
    # 手算校验 U-G01: min(20+3+12, 27+12+3) - 3 - (-90) = min(35,42)-3+90 = 35-3+90=122
    assert abs(LMAX_U_G01 - 122.0) < 1e-6, "U-G01 门限手算不符"
    # 示例: G01 到 S001 上空 30m 作业高度的运输机
    g01 = g01_pos()
    s1 = P.SERVICE_AREAS[0]
    Pu = (s1['lon'], s1['lat'], s1['alt_m'] + P.WORK_ALT_OFFSET_M)
    avail, Lp, margin = link_available(Pu, g01, LMAX_U_G01)
    print(f"[comm] S001上空运输机<->G01: 可用={avail} L_path={Lp:.2f}dB 裕量={margin:.2f}dB")
    print("[comm] 门限手算校验通过")
