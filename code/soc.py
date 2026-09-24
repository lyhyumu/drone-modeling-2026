# -*- coding: utf-8 -*-
"""SOC / 两阶段充电状态机（式11）。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import params as P


def soc_end(g_energy_used, e_use):
    """任务结束 SOC = 1 - E_trip/E_use。"""
    return 1.0 - g_energy_used / e_use


def t_charge(s, t_full):
    """从 SOC=s 充至 100% 耗时（式11，两阶段）。
    0<=s<0.90: T_full*[0.65*(0.90-s)/0.90 + 0.35]
    0.90<=s<=1: T_full*0.35*(1-s)/0.10"""
    s = max(0.0, min(s, 1.0))
    if s < 0.90:
        return t_full * (0.65 * (0.90 - s) / 0.90 + 0.35)
    return t_full * 0.35 * (1.0 - s) / 0.10


if __name__ == '__main__':
    # 自检: s=1 -> t_chg=0; s=0 -> t_chg=T_full
    for tf in [1800, 2400, 3000]:
        assert abs(t_charge(1.0, tf) - 0.0) < 1e-9, "s=1 应 0 充电"
        assert abs(t_charge(0.0, tf) - tf) < 1e-6, "s=0 应完全充电"
        # 连续性 @ 0.90
        left = tf * (0.65 * (0.90 - 0.90) / 0.90 + 0.35)
        right = tf * 0.35 * (1 - 0.90) / 0.10
        assert abs(left - right) < 1e-6, "s=0.9 分段不连续"
    print("[soc] 两阶段充电自检通过 (s=1->0, s=0->T_full, 0.9连续)")
    print(f"  示例 A电池 s_end=0.20 充满耗时 = {t_charge(0.20, 1800):.1f} s")
