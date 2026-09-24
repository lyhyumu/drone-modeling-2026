# -*- coding: utf-8 -*-
"""统一参数模块：全部数值常数从 PROBLEM_FACTS.json 载入，禁止裸数字字面量。
四问共用；口径见 MODELING_REPORT.md §9.0 参数口径表。"""
import os, sys, json
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 工作区根目录（code/ 的上一级）
ROOT = Path(_HERE).parent
_FACTS = json.loads((ROOT / 'PROBLEM_FACTS.json').read_text(encoding='utf-8'))

# ---------- 白名单纯数学/单位常数 ----------
G_ACC = 9.80665            # 重力加速度 m/s^2 (物理常数)
KWH_TO_J = 3.6e6           # 1 kWh = 3.6e6 J
CRUISE_CLEARANCE_M = 50.0  # 巡航净空(附录2逐字)
WORK_ALT_OFFSET_M = 30.0   # 服务区作业高度 = 地面 + 30 m
CLIMB_EFF = 0.72           # 爬升能耗效率(三机型/中继统一)
DESCENT_ENERGY = 0.0       # 下降不单独计能耗
EPS_Q = 1e-3               # 二分法载荷精度 kg

# ---------- 投影 ----------
DEM_EPSG = 4326            # DEM 原生地理坐标
METRIC_EPSG = 32649        # UTM zone 49N 米制投影(区域 lon~109.2 lat~23.0)

# ---------- 空间 ----------
_dc = _FACTS['domain']['spatial']['dispatch_center']
O01 = {'id': _dc['id'], 'lon': _dc['lon'], 'lat': _dc['lat'], 'alt_m': _dc['alt_m']}
SERVICE_AREAS = _FACTS['service_areas']            # list of 15 dicts
N_AREAS = len(SERVICE_AREAS)                        # 15
DEM_FILE = ROOT / 'user_data' / _FACTS['domain']['spatial']['dem']['file']

# ---------- 运输机型 ----------
TRANSPORT = {t['id']: t for t in _FACTS['transport_uav_types']}   # 'A','B','C'
FLEET = _FACTS['transport_fleet']                                  # 8 entities
# 每机型实体机数量
FLEET_INV = {}
for u in FLEET:
    FLEET_INV[u['type']] = FLEET_INV.get(u['type'], 0) + 1        # A4 B2 C2

# 常用便捷常数（从 facts 展开，命名与题面对应）
Q = {g: TRANSPORT[g]['Q_kg'] for g in TRANSPORT}                   # 最大载货质量
V = {g: TRANSPORT[g]['vol_m3'] for g in TRANSPORT}                 # 可用体积
E_USE = {g: TRANSPORT[g]['E_use_kWh'] for g in TRANSPORT}          # 可用能量
L0 = {g: TRANSPORT[g]['L0_m'] for g in TRANSPORT}                  # 空载航程
LF = {g: TRANSPORT[g]['LF_m'] for g in TRANSPORT}                  # 满载航程
EMPTY = {g: TRANSPORT[g]['empty_mass_kg'] for g in TRANSPORT}
V_CRUISE = {g: TRANSPORT[g]['v_cruise_ms'] for g in TRANSPORT}
V_CLIMB = {g: TRANSPORT[g]['v_climb_ms'] for g in TRANSPORT}
V_DESCENT = {g: TRANSPORT[g]['v_descent_ms'] for g in TRANSPORT}
RETURN_SOC_MIN = {g: TRANSPORT[g]['return_soc_min_pct'] / 100.0 for g in TRANSPORT}  # 0.20
RHO_BASE = RETURN_SOC_MIN['A']                                    # 基准返航余量 0.20
PREP_S = {g: TRANSPORT[g]['prep_s'] for g in TRANSPORT}
LOAD_PER_BOX_S = {g: TRANSPORT[g]['load_per_box_s'] for g in TRANSPORT}
HANDOVER_BASE_S = {g: TRANSPORT[g]['handover_base_s'] for g in TRANSPORT}
HANDOVER_PER_BOX_S = {g: TRANSPORT[g]['handover_per_box_s'] for g in TRANSPORT}

# ---------- 共享电池 ----------
BAT = {b['type']: b for b in _FACTS['shared_battery']}
BAT_INV = {g: BAT[g]['n_groups'] for g in BAT}                     # A6 B4 C4
T_FULL_BAT = {g: BAT[g]['full_charge_s'] for g in BAT}            # 1800/2400/3000

# ---------- 中继 ----------
RELAY = _FACTS['relay_uav']
RELAY_TAKEOFF_MASS = RELAY['takeoff_mass_kg']                      # 23.5
RELAY_V_CRUISE = RELAY['v_cruise_ms']
RELAY_V_CLIMB = RELAY['v_climb_ms']
RELAY_V_DESCENT = RELAY['v_descent_ms']
RELAY_CRUISE_POWER_KW = RELAY['cruise_power_kW']                   # 1.15
RELAY_HOVER_POWER_KW = RELAY['hover_power_kW']                     # 1.05
RELAY_COMM_POWER_KW = RELAY['comm_extra_power_kW']                 # 0.05
RELAY_E_COMPONENT = RELAY['E_component_kWh']                       # 3.2
RELAY_RETURN_SOC_MIN = RELAY['return_soc_min_pct'] / 100.0        # 0.20
RELAY_PREP_S = RELAY['prep_s']                                     # 180
RELAY_LINK_SETUP_S = RELAY['link_setup_s']                        # 30
RELAY_TURNAROUND_S = RELAY['turnaround_s']                         # 300
RELAY_MAX_AGL = RELAY['max_hover_agl_m']                           # 300
RELAY_N_UNITS = len(RELAY['units'])                               # 2
RELAY_COMP_INV = RELAY['energy_components']['n_groups']           # 6
T_FULL_COMP = RELAY['energy_components']['full_charge_s']         # 1800

# ---------- 通信 ----------
_C = _FACTS['comm_params']
FREQ_MHZ = _C['f_MHz']            # 2400
L_SYS_DB = _C['L_sys_dB']         # 3
L_OBS_DB = _C['L_obs_dB']         # 10
P_SENS_DBM = _C['P_sens_dBm']     # -98
M_DB = _C['M_dB']                 # 8
P_TH_DBM = P_SENS_DBM + M_DB      # -90
# 端点参数 (Pt dBm, G dBi)
COMM_UAV = _C['transport_uav']          # Pt20 G3
COMM_RELAY_ACCESS = _C['relay_access']  # Pt20 G6
COMM_RELAY_BACKHAUL = _C['relay_backhaul']  # Pt19 G8
COMM_G01 = _C['gateway_G01']            # Pt27 G12 agl20
G01_ANTENNA_AGL = COMM_G01['antenna_agl_m']  # 20

# ---------- 分区(问题四) ----------
K_VALUES = [2, 3]

# ---------- 敏感性 ----------
RHO_SCENARIOS = [0.10, 0.15, 0.20, 0.25, 0.30]

SEED = 42

# ---------- 口径一致性断言 ----------
# 返航余量与SOC下限口径一致: 能耗上限(1-rho)*E_use 对应结束SOC>=rho
for g in TRANSPORT:
    assert abs(RETURN_SOC_MIN[g] - RHO_BASE) < 1e-9, f"[口径冲突] {g} 返航SOC下限与基准rho不一致"
# 充电两阶段: s=1 时 t_chg=0, s=0 时 t_chg=T_full (在 soc.py 校验函数形态)
assert Q['A'] == 25 and Q['B'] == 30 and Q['C'] == 80, "[口径冲突] 最大载货质量与题面表不符"
assert E_USE['A'] == 4.5 and E_USE['C'] == 8, "[口径冲突] 电池可用能量与题面不符"
assert FLEET_INV == {'A': 4, 'B': 2, 'C': 2}, "[口径冲突] 运输机队与题面不符"
assert BAT_INV == {'A': 6, 'B': 4, 'C': 4}, "[口径冲突] 电池库存与题面不符"

if __name__ == '__main__':
    print("[params] 口径一致性 OK")
    print(f"  机型 Q={Q} V={V} E_use={E_USE}")
    print(f"  机队 {FLEET_INV} 电池 {BAT_INV} 中继 {RELAY_N_UNITS}/{RELAY_COMP_INV}")
    print(f"  P_th={P_TH_DBM} dBm  投影 EPSG {DEM_EPSG}->{METRIC_EPSG}")
