# -*- coding: utf-8 -*-
"""地理/航段计算器：经纬度->UTM米制投影、DEM取样、Bresenham沿线全像元巡航海拔。
方法唯一性(MODELING §9.4 步骤1-2)：DEM epsg/transform 投影，Bresenham 全像元取 max+50。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from functools import lru_cache
from scipy.io import loadmat
from pyproj import Transformer
import params as P

# ---------- DEM 载入(全量, 单例) ----------
_MAT = loadmat(str(P.DEM_FILE))
DEM = _MAT['dem'].astype(np.float64)                 # (1309,1486) 行=lat 列=lon
_NODATA = float(_MAT['nodata'].ravel()[0])
DEM = np.where(DEM == _NODATA, np.nan, DEM)
LAT = _MAT['latitude'].ravel().astype(np.float64)    # 1309, 递减(北->南)
LON = _MAT['longitude'].ravel().astype(np.float64)   # 1486, 递增(西->东)
_TR = _MAT['transform'].ravel()                       # [px, 0, x0, 0, -py, y0]
DEM_NROWS, DEM_NCOLS = DEM.shape
_LON0, _dLON = LON[0], LON[1] - LON[0]                # 起始经度, 每列经度增量(+)
_LAT0, _dLAT = LAT[0], LAT[1] - LAT[0]                # 起始纬度, 每行纬度增量(-)

# 投影器(经纬度 -> 米制)
_XY = Transformer.from_crs(P.DEM_EPSG, P.METRIC_EPSG, always_xy=True)


def to_xy(lon, lat):
    """经纬度 -> 米制平面坐标 (X,Y)。"""
    x, y = _XY.transform(lon, lat)
    return float(x), float(y)


def lonlat_to_rc(lon, lat):
    """经纬度 -> DEM 行列索引(最近像元，裁剪到边界内)。"""
    col = int(round((lon - _LON0) / _dLON))
    row = int(round((lat - _LAT0) / _dLAT))
    col = min(max(col, 0), DEM_NCOLS - 1)
    row = min(max(row, 0), DEM_NROWS - 1)
    return row, col


def dem_at_lonlat(lon, lat):
    """经纬度处地面高程 m（最近像元，NaN 用邻域均值兜底）。"""
    r, c = lonlat_to_rc(lon, lat)
    v = DEM[r, c]
    if np.isnan(v):
        sub = DEM[max(0, r-2):r+3, max(0, c-2):c+3]
        v = np.nanmean(sub) if np.isfinite(np.nanmean(sub)) else 0.0
    return float(v)


def _bresenham(r0, c0, r1, c1):
    """整数栅格 Bresenham 直线，返回沿线全部像元 (row,col) 列表。"""
    pts = []
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        pts.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr; c += sc
        if e2 < dc:
            err += dc; r += sr
    return pts


def cruise_altitude(lon_i, lat_i, lon_j, lat_j):
    """计划巡航海拔 = 航段沿水平直线穿越 DEM 像元最高地面高程 + 50 m（式1）。
    Bresenham 沿线全像元取 max，非端点两点最大值。"""
    r0, c0 = lonlat_to_rc(lon_i, lat_i)
    r1, c1 = lonlat_to_rc(lon_j, lat_j)
    # 规范化端点顺序，保证正反向遍历同一像元集（Bresenham 方向无关对称）
    if (r0, c0) > (r1, c1):
        r0, c0, r1, c1 = r1, c1, r0, c0
    pts = _bresenham(r0, c0, r1, c1)
    vals = [DEM[r, c] for (r, c) in pts]
    vals = [v for v in vals if np.isfinite(v)]
    hmax = max(vals) if vals else 0.0
    return float(hmax + P.CRUISE_CLEARANCE_M)


def horizontal_dist(lon_i, lat_i, lon_j, lat_j):
    """两节点米制平面欧氏水平距离 m（式2）。"""
    xi, yi = to_xy(lon_i, lat_i)
    xj, yj = to_xy(lon_j, lat_j)
    return float(np.hypot(xi - xj, yi - yj))


def line_of_sight_blocked(P_a, P_b, n_samp=64):
    """三维视线遮挡判定：视线任一采样点低于地面高程即遮挡。
    P_a,P_b = (lon,lat,alt_m 绝对海拔)。返回 True=遮挡。"""
    lon_a, lat_a, z_a = P_a
    lon_b, lat_b, z_b = P_b
    ts = np.linspace(0, 1, n_samp)
    for t in ts[1:-1]:                       # 端点为发射/接收机，不判自身
        lon = lon_a + t * (lon_b - lon_a)
        lat = lat_a + t * (lat_b - lat_a)
        z_line = z_a + t * (z_b - z_a)
        z_ground = dem_at_lonlat(lon, lat)
        if z_line < z_ground:                # 视线埋入地形
            return True
    return False


def dist_3d(P_a, P_b):
    """三维直线距离 m。P=(lon,lat,alt_m)。"""
    xa, ya = to_xy(P_a[0], P_a[1])
    xb, yb = to_xy(P_b[0], P_b[1])
    return float(np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (P_a[2] - P_b[2]) ** 2))


# ---------- 节点海拔便捷 ----------
def work_alt(node):
    """作业绝对海拔：O01=地面海拔；服务区=地面+30 m。"""
    if node['id'] == P.O01['id']:
        return node['alt_m']
    return node['alt_m'] + P.WORK_ALT_OFFSET_M


if __name__ == '__main__':
    o = P.O01
    s1 = P.SERVICE_AREAS[0]
    d = horizontal_dist(o['lon'], o['lat'], s1['lon'], s1['lat'])
    print(f"[geo] O01-S001 dist = {d:.1f} m = {d/1000:.3f} km (期望≈3.07km)")
    h = cruise_altitude(o['lon'], o['lat'], s1['lon'], s1['lat'])
    print(f"[geo] O01-S001 巡航海拔 = {h:.1f} m")
    # 自检: 同点距离0, 对称
    assert abs(horizontal_dist(o['lon'], o['lat'], o['lon'], o['lat'])) < 1e-6
    d_rev = horizontal_dist(s1['lon'], s1['lat'], o['lon'], o['lat'])
    assert abs(d - d_rev) < 1e-6, "距离不对称"
    h_rev = cruise_altitude(s1['lon'], s1['lat'], o['lon'], o['lat'])
    assert abs(h - h_rev) < 1e-6, "巡航海拔不对称"
    print("[geo] 自检通过: 同点距离0/正反向对称")
