"""
通用地理空间工具函数 (v5.3 fixed)

v5.3 改进:
  - crossing_angle 返回值语义明确化
  - 新增 batch_rasterize 批量栅格化辅助
  - 新增 covers_or_touches 替代 contains 用于边界判定
  - 优化 segment_line 性能
"""
import os
import json
import logging
import math
import numpy as np
from pathlib import Path

logger = logging.getLogger("transmission_planning")


def polygon_area_m2(geom) -> float:
    """★P2 (v0.6)★ 计算面状几何在 working_crs (投影米) 下的面积 (m²), 保留 1 位小数。

    用途: 给 forbidden/no_tower/cost 三类面 GPKG 每行补 area_m2 字段, 供算法端粗规划阶段
    按阈值保留大面、忽略细碎面 (R4)。

    设计要点:
      - 几何重算 (geom.area), 不依赖源 Shape_Area —— 因 M2 _fix_protection_geometry 用
        make_valid / 河流 intersection 改过几何, 源字段对不上算法端真正栅格化的那块几何。
      - 对 Polygon 返回其面积; 对 MultiPolygon 返回各子面面积之和 (整面口径)。
        注: 算法端 feature_loader 读盘时会把 MultiPolygon 拍扁为多个独立要素;
        若对接阶段需逐子面面积, 算法端可对拍扁后的单面 geom.area 重算 (零成本)。
      - None / 空 / 非面状 (点/线) → 0.0 (占位行 _placeholder 即走此路)。
    """
    try:
        if geom is None or geom.is_empty:
            return 0.0
        gtype = geom.geom_type
        if gtype in ("Polygon", "MultiPolygon"):
            return round(float(geom.area), 1)
        # 点/线等非面状几何无面积概念 (占位 Point(0,0) 等)
        return 0.0
    except Exception:
        # 几何异常不应阻断写盘; area_m2 退化为 0.0 并交由上层日志/校验发现
        return 0.0


# ──────────────────────────────────────────────────────────────────
# ★P4 (v0.6)★ bbox 运算 (纯算术, 不依赖 GIS, 便于单测)
# 用于把"起终点+缓冲"外扩出的工作区 bbox 裁回输入数据实际范围 (矢量∪栅格),
# 避免小地图被额外膨胀 (R3 / Q4-A)。
# bbox 约定: (xmin, ymin, xmax, ymax)。
# ──────────────────────────────────────────────────────────────────
def bbox_intersect(a, b):
    """两个 bbox 的交集; 任一为 None 或无重叠返回 None。"""
    if a is None or b is None:
        return None
    xmin = max(a[0], b[0])
    ymin = max(a[1], b[1])
    xmax = min(a[2], b[2])
    ymax = min(a[3], b[3])
    if xmax <= xmin or ymax <= ymin:
        return None  # 不重叠 / 退化
    return (xmin, ymin, xmax, ymax)


def bbox_union(bboxes):
    """多个 bbox 的并集 (忽略 None); 全为 None 返回 None。"""
    bs = [b for b in bboxes if b is not None]
    if not bs:
        return None
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))


def _bbox_close(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def clip_bbox_to_extent(expanded, data_extent, must_include=None):
    """★P4★ 把 expanded 裁到 data_extent (交集), 但结果至少包含 must_include。

    - expanded:     起终点(或必经)窗口外扩后的 bbox。
    - data_extent:  输入数据实际范围 (矢量∪栅格); None 表示无可裁范围。
    - must_include: 起终点窗口 (未外扩), 保证裁剪不丢端点 (端点贴数据边角时, 交集
                    可能切掉端点附近, 用并集兜回)。None 则不做端点保护。

    返回 (result: tuple, clipped: bool):
      - data_extent 为 None        -> (expanded, False)        不裁
      - expanded 与 data_extent 无重叠 (异常) -> (expanded, False)  不裁(避免空 bbox)
      - 正常 -> (union(intersect(expanded, data_extent), must_include), result != expanded)
    """
    exp = tuple(expanded)
    if data_extent is None:
        return exp, False
    inter = bbox_intersect(exp, tuple(data_extent))
    if inter is None:
        return exp, False
    result = inter if must_include is None else bbox_union([inter, tuple(must_include)])
    return result, (not _bbox_close(result, exp))


def setup_logging(log_dir: str = None, level=logging.INFO):
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    root = logging.getLogger("transmission_planning")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "preprocessing.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return root


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)


def azimuth_deg(x1, y1, x2, y2) -> float:
    """计算两点之间的方位角（0-360°，正北为0°顺时针）"""
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dx, dy)) % 360
    return angle


def crossing_angle(az1: float, az2: float) -> float:
    """
    计算两条线段的交叉角（锐角，0-90°）。
    az1, az2 为方位角（0-360°）。
    返回值：两线夹角的锐角值，范围 [0, 90]。
    """
    diff = abs(az1 - az2) % 180
    return min(diff, 180 - diff)


def angle_diff_unsigned(az1: float, az2: float) -> float:
    """
    计算两方位角之间的无符号差（0-180°）。
    用于平行判定等需要区分小角度差和大角度差的场景。
    """
    diff = abs(az1 - az2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


def segment_line(coords, seg_length_m: float = 50.0):
    """
    将坐标序列按指定长度分段，返回 [(LineString, azimuth_deg), ...]
    v5.3: 优化性能，避免重复创建 LineString 对象
    """
    import shapely.geometry as sg
    line = sg.LineString(coords)
    total_len = line.length
    if total_len <= 0:
        return []
    segments = []
    dist = 0.0
    while dist < total_len:
        end_dist = min(dist + seg_length_m, total_len)
        p1 = line.interpolate(dist)
        p2 = line.interpolate(end_dist)
        if p1.distance(p2) < 0.01:
            dist = end_dist
            continue
        az = azimuth_deg(p1.x, p1.y, p2.x, p2.y)
        seg_line = sg.LineString([p1, p2])
        segments.append((seg_line, az))
        dist = end_dist
    return segments


def segment_line_fast(coords, seg_length_m: float = 50.0):
    """
    快速版分段：直接在坐标序列上按距离切分，避免Shapely interpolate开销。
    返回 [(LineString, azimuth_deg), ...]
    """
    from shapely.geometry import LineString
    if len(coords) < 2:
        return []

    # 先算累积距离
    cum_dist = [0.0]
    for i in range(1, len(coords)):
        d = math.hypot(coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
        cum_dist.append(cum_dist[-1] + d)
    total_len = cum_dist[-1]
    if total_len <= 0:
        return []

    # 按seg_length_m切分
    segments = []
    seg_start = 0.0
    coord_idx = 0
    while seg_start < total_len:
        seg_end = min(seg_start + seg_length_m, total_len)

        # 找到 seg_start 和 seg_end 所在的线段并插值
        p1 = _interpolate_at(coords, cum_dist, seg_start)
        p2 = _interpolate_at(coords, cum_dist, seg_end)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx * dx + dy * dy < 0.0001:
            seg_start = seg_end
            continue

        az = math.degrees(math.atan2(dx, dy)) % 360
        seg_line = LineString([p1, p2])
        segments.append((seg_line, az))
        seg_start = seg_end

    return segments


def _interpolate_at(coords, cum_dist, target_dist):
    """在累积距离数组中插值获取点坐标"""
    if target_dist <= 0:
        return coords[0]
    if target_dist >= cum_dist[-1]:
        return coords[-1]
    # 二分查找
    lo, hi = 0, len(cum_dist) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if cum_dist[mid] <= target_dist:
            lo = mid
        else:
            hi = mid
    # 在 lo-hi 段上插值
    seg_len = cum_dist[hi] - cum_dist[lo]
    if seg_len < 1e-10:
        return coords[lo]
    ratio = (target_dist - cum_dist[lo]) / seg_len
    x = coords[lo][0] + ratio * (coords[hi][0] - coords[lo][0])
    y = coords[lo][1] + ratio * (coords[hi][1] - coords[lo][1])
    return (x, y)


def river_wide_barrier_polys(disks, river_geom):
    """
    ★P7 (v0.6)★ 宽河段 → 真实多边形禁区 (替换 v0.5 的 pt.buffer(w/2) 圆盘)。

    把宽段各采样点的圆盘 (pt.buffer(width/2)) 先 unary_union, 再与河流面 intersection,
    得到**裁回河岸以内**、覆盖整段宽河的多边形 (连续宽段自然合成 1 个, 分叉/断开则多个)。
    这样不再像单个圆盘那样向河岸两侧/首尾鼓出 ~w/2 把陆地误判为禁区。

    参数:
      disks:      [Polygon] 各宽采样点的圆盘 (调用方按 width≥threshold 收集)。
      river_geom: 河流面 (Polygon / MultiPolygon); None/空 → 不裁, 直接返回 union 后的多边形。
    返回:
      [Polygon] 已 explode、去空 (area>0) 的多边形列表; 无输入 → []。
    """
    from shapely.ops import unary_union
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    if not disks:
        return []
    merged = unary_union(disks)
    if river_geom is not None and not getattr(river_geom, "is_empty", True):
        merged = merged.intersection(river_geom)
    if merged is None or merged.is_empty:
        return []
    if isinstance(merged, Polygon):
        cand = [merged]
    elif isinstance(merged, (MultiPolygon, GeometryCollection)):
        cand = list(merged.geoms)
    else:
        cand = []
    return [g for g in cand
            if isinstance(g, Polygon) and (not g.is_empty) and g.area > 0]


def river_narrow_cross_segments(center_coords, widths, width_points,
                                threshold_m, seg_length_m: float = 50.0,
                                width_filter: bool = True):
    """
    ★P7 (v0.6)★ 窄河段 → linear_cross 线段 (带逐段方位角), 供算法端按 min_cross_angle 约束。

    沿河流中心线 (center_coords) 用 segment_line_fast 切成 seg_length_m 的小段 (自带方位角),
    再按**每段中点最近的宽度采样**分类: width<threshold 的段才返回 (宽段已由
    river_wide_barrier_polys 的多边形禁区覆盖, 这里跳过, 避免宽河被重复表达为可跨线)。

    参数:
      center_coords: 中心线坐标序列 [(x,y),...] (调用方传 _compute_river_widths 返回的 center_line.coords)。
      widths:        [float] 各采样点宽度 (与 width_points 等长)。
      width_points:  [shapely Point] 各采样点 (与 widths 等长)。
      threshold_m:   宽窄阈值 (m), 默认上游 major_river_threshold_m=900。
    返回:
      [(LineString, azimuth_deg), ...] 仅窄段; 无可切/无窄段 → []。

    注 (子风险 R10): center_coords 来自 _compute_river_widths 的最小外接矩形直长轴,
    对强弯曲/分叉河流方位角近似恒定; 本函数本身能逐段跟随给定坐标 (弯折坐标 → 方位角随之变化),
    精度上限在于上游中心线是直轴, 彻底解需真实中轴线骨架化 (不在本专项)。
    """
    if not center_coords or len(center_coords) < 2:
        return []
    # ★A.7 (v0.6.2)★ width_filter=False (形态学 A2 路径): 传进来的中心线本就取自
    #   narrow = river − wide, 按 A2 定义**每一处都已是窄段**, 再按逐站宽度二次
    #   过滤属于两套宽窄口径并存 —— 一旦某站宽度被局部形态(汇流口/展宽/发卡)抬高,
    #   该段被误丢 → 中心线断开。故形态学路径关掉此过滤, 口径统一到开运算。
    #   width_filter=True 保留给 v0.5 圆盘回退路径(那条路上宽窄确实靠逐站宽度分)。
    if width_filter and not width_points:
        return []
    out = []
    for seg_geom, az in segment_line_fast(center_coords, seg_length_m=seg_length_m):
        if not width_filter:
            out.append((seg_geom, az))
            continue
        mid = seg_geom.interpolate(0.5, normalized=True)
        best_w, best_d = None, float("inf")
        for w, wp in zip(widths, width_points):
            d = mid.distance(wp)
            if d < best_d:
                best_d, best_w = d, w
        if best_w is not None and best_w < threshold_m:
            out.append((seg_geom, az))
    return out


def _pip_raycast(pts, ext, holes):
    """点是否在多边形(外环 ext + 洞 holes)内 (numpy 射线法, 无 shapely)。pts[N,2]→bool[N]。

    ★A.8 (v0.7.0) 性能★ 原实现按环上的边**逐条 Python 循环**, 3201 顶点的河流面
    单次调用耗时约 28ms。中心线改为"沿河道行走"后调用量增加一个数量级, 必须向量化。
    现改为一次性对全部边做数组运算, 同量级输入下快约两个数量级。
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    # ★A.8★ 分块: (N点 × M边) 矩阵在"几千点 × 上万顶点"时可达数百 MB, 会被 OOM 杀掉。
    #   按点分批, 单批矩阵控制在千万元素以内。
    n_edges = max(len(np.asarray(ext)), 1)
    batch = max(1, min(len(pts), int(4_000_000 // max(n_edges, 1)) or 1))
    if len(pts) > batch:
        out = np.empty(len(pts), dtype=bool)
        for i in range(0, len(pts), batch):
            out[i:i + batch] = _pip_raycast(pts[i:i + batch], ext, holes)
        return out
    x = pts[:, 0]
    y = pts[:, 1]

    def ring_test(ring):
        R = np.asarray(ring, dtype=float)
        if len(R) < 3:
            return np.zeros(len(pts), dtype=bool)
        xi = R[:, 0]
        yi = R[:, 1]
        xj = np.roll(xi, 1)
        yj = np.roll(yi, 1)
        # (N, M): N 个点 × M 条边
        cond_y = (yi[None, :] > y[:, None]) != (yj[None, :] > y[:, None])
        denom = (yj - yi)[None, :]
        denom = np.where(np.abs(denom) < 1e-18, 1e-18, denom)
        xint = (xj - xi)[None, :] * (y[:, None] - yi[None, :]) / denom + xi[None, :]
        return (cond_y & (x[:, None] < xint)).sum(axis=1) % 2 == 1

    inside = ring_test(ext)
    for h in (holes or []):
        inside &= ~ring_test(h)
    return inside


def _cs_chords(px, py, dx, dy, ext, holes):
    """过点(px,py)沿方向(dx,dy)的直线与多边形的交, 配成'河内弦'。返回 [(ta,tb),...]。

    ★A.8 (v0.7.0) 性能★ 同 _pip_raycast: 原按边逐条 Python 循环, 3201 顶点时
    单次约 100ms; 现全部向量化, 并把"弦中点是否在河内"的判断合并成一次批量 PIP。
    """
    p = np.array([px, py], dtype=float)
    d = np.array([dx, dy], dtype=float)
    ts_all = []
    for ring in [ext] + list(holes or []):
        R = np.asarray(ring, dtype=float)
        if len(R) < 2:
            continue
        A = R
        B = np.roll(R, -1, axis=0)
        E = B - A                                   # (M,2) 边向量
        det = d[0] * (-E[:, 1]) - (-E[:, 0]) * d[1]
        ok = np.abs(det) >= 1e-12
        if not ok.any():
            continue
        rhs = A - p                                 # (M,2)
        det_ok = np.where(ok, det, 1.0)
        t = (rhs[:, 0] * (-E[:, 1]) - (-E[:, 0]) * rhs[:, 1]) / det_ok
        sv = (d[0] * rhs[:, 1] - rhs[:, 0] * d[1]) / det_ok
        sel = ok & (sv >= -1e-9) & (sv <= 1 + 1e-9)
        if sel.any():
            ts_all.append(t[sel])
    if not ts_all:
        return []
    ts = np.sort(np.concatenate(ts_all))
    if len(ts) < 2:
        return []
    tm = 0.5 * (ts[:-1] + ts[1:])                   # 相邻交点的中点, 一次性批量判内外
    mids = p[None, :] + tm[:, None] * d[None, :]
    inside = _pip_raycast(mids, ext, holes)
    idx = np.where(inside)[0]
    return [(float(ts[k]), float(ts[k + 1])) for k in idx]


def _cs_smooth(pts, k=2):
    """★A.7 (v0.6.2)★ 端点保持型滑动平均。

    原实现在两端用**非对称**窗口 (lo=max(0,i-k), hi=min(n,i+k+1)) 取均值 →
    首点被拉向内部约 k/2 个站距、末点同理 → guide 折线每轮精修**两端各缩短
    ~k*interval**。refine_iters=3 时累计缩 ~400m/端 (interval=100, k=2),
    与河段长度无关。实测纵向覆盖: 1000m 河段仅 20%、2000m 仅 60%。
    后果: 每个河流连通块的中心线两端各缺 ~400m → linear_cross 段成片缺失,
    汇流口(多条 channel 的端头汇聚处)尤其明显 —— 各条线都在离口子 400m 处停住。

    修复: 首末点原样保留; 内部点用**对称**窗口 kk=min(k, i, n-1-i) →
    平滑仍有效, 但折线长度不再被侵蚀 (实测覆盖回到 88~98%)。
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n < 3:
        return pts
    out = pts.copy()
    for i in range(1, n - 1):
        kk = min(k, i, n - 1 - i)
        out[i] = pts[i - kk:i + kk + 1].mean(axis=0)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def _cs_tangents(pts):
    pts = np.asarray(pts, dtype=float)
    d = np.gradient(pts, axis=0)
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)


def _cs_axis(ext):
    """边长加权 PCA 主轴 —— 只用于确定行走的**起点与初始方向**。

    注意: 主轴是整个**连通块**的整体走向, 不是某一小段的方向。对蜿蜒河道,
    它只能给出"这段河大体朝哪儿", 不能用来布站(见 _centerline_walk 说明)。
    """
    ext = np.asarray(ext, dtype=float)
    seg_mid = 0.5 * (ext[:-1] + ext[1:])
    seg_len = np.hypot(*(ext[1:] - ext[:-1]).T)
    if seg_len.sum() > 1e-9 and len(seg_mid) >= 2:
        wgt = seg_len / seg_len.sum()
        c = (seg_mid * wgt[:, None]).sum(axis=0)
        Xw = seg_mid - c
        cov = (Xw * wgt[:, None]).T @ Xw
    else:
        c = ext.mean(axis=0)
        Xw = ext - c
        cov = Xw.T @ Xw
    evals, V = np.linalg.eigh(cov)
    ax = V[:, int(np.argmax(evals))]
    proj = (ext - c) @ ax
    return c, ax, float(proj.min()), float(proj.max())


def _cs_boundary_distance(Q, ext, holes):
    """点到多边形**边界**(外环+所有洞)的最短距离, 向量化。"""
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    best = np.full(len(Q), np.inf)
    for ring in [ext] + list(holes or []):
        R = np.asarray(ring, dtype=float)
        if len(R) < 2:
            continue
        A = R[:-1]
        B = R[1:]
        AB = B - A
        denom = np.einsum('ij,ij->i', AB, AB)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        AP_x = Q[:, 0][:, None] - A[:, 0][None, :]
        AP_y = Q[:, 1][:, None] - A[:, 1][None, :]
        t = np.clip((AP_x * AB[:, 0][None, :] + AP_y * AB[:, 1][None, :]) / denom[None, :], 0.0, 1.0)
        dx = AP_x - t * AB[:, 0][None, :]
        dy = AP_y - t * AB[:, 1][None, :]
        best = np.minimum(best, np.sqrt(dx * dx + dy * dy).min(axis=1))
    return best


def _cs_fit_arc(P):
    """对末端若干点做最小二乘圆拟合(Kasa)。返回 (center, R, ok)。

    用于让端部外推跟随河道弯势(仅全局轴扫描路径使用; 行走路径不做端部外推)。
    """
    P = np.asarray(P, dtype=float)
    if len(P) < 3:
        return None, np.inf, False
    x = P[:, 0]
    y = P[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones(len(P))], axis=1)
    rhs = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    except Exception:
        return None, np.inf, False
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = c + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0:
        return None, np.inf, False
    R = float(np.sqrt(r2))
    if not np.isfinite(R) or R <= 0:
        return None, np.inf, False
    return np.array([cx, cy], dtype=float), R, True


def _cs_effective_length(pts, ext, holes, k: int = 9):
    """折线的**有效长度** = 各段长度 × 该段落在河面内的比例。

    用于在"行走"与"轴扫描"两条路径间择优: 轴扫描常常靠横切河湾把线拉长,
    单纯比长度会选错; 扣掉出河部分后, 横切的收益被抵消。
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    total = 0.0
    ts = np.linspace(0.0, 1.0, k + 2)[1:-1]
    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        L = float(np.hypot(*seg))
        if L <= 0:
            continue
        samp = pts[i][None, :] + ts[:, None] * seg[None, :]
        total += L * float(_pip_raycast(samp, ext, holes).mean())
    return total


def _cs_reach_length_estimate(ext, holes, width_hint=0.0):
    """估计河段应有的长度 = **面积 / 平均河宽**。

    ★A.8.4★ 一定要用面积法, 不能用周长法。真实河岸是锯齿的, 周长被锯齿严重放大 ——
    实测同一条河: 真实弧长 2578m, 周长法估出 5764m(失真 2.2 倍), 面积法 2572m(准确)。
    面积对锯齿不敏感(正负噪声相互抵消), 这是它稳健的原因。
    width_hint 传入实测河宽中位数; 缺省时退回按外接矩形短边估。
    """
    try:
        ext = np.asarray(ext, dtype=float)

        def _area(R):
            R = np.asarray(R, dtype=float)
            x, y = R[:, 0], R[:, 1]
            return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

        A = _area(ext) - sum(_area(h) for h in (holes or []))
        if A <= 0:
            return 0.0
        w = float(width_hint)
        if not np.isfinite(w) or w <= 0:
            c, ax, pmin, pmax = _cs_axis(ext)
            span = max(pmax - pmin, 1e-6)
            w = A / span
        return A / max(w, 1e-6)
    except Exception:
        return 0.0


def _centerline_walk(ext, holes, interval_m, max_steps: int = 6000,
                     relaxed: bool = False):
    """★A.8 (v0.7.0)★ **沿河道行走**求中心线, 取代原来的"全局轴扫描"。

    为什么必须换掉全局轴扫描
    ------------------------
    原 pass0: 取连通块主轴, 沿**轴**每 interval 作垂线截河面取弦中点。
    对顺直/缓弯河道没问题; 但对**细窄蜿蜒**河道:
      · 垂直于全局轴的截线会一次横穿好几个河湾, "取最长弦"可能取到另一个湾;
      · 沿轴等距 = 沿河道**不等距**, 相邻站在河道上可能隔了两三百米,
        两点连线直接横切河湾。
    实测(几何合法的蛇形夹具, 宽 25~60m): 密采样出河率 **11~29%**,
    最大站距 144~187m(名义 100m) —— 与真实数据截图完全吻合。
    3 轮局部精修救不回来, 因为初始线本身就横跨河湾。

    本函数改为**局部行走**, 全程不依赖全局轴:
      起点  : 主轴一端向内缩一点, 取垂直于主轴的弦中点(主轴只用来定起点与初始朝向)
      每一步: 沿当前切向前进 step → 取垂直于切向、**含该点**的弦 → 中点即新的中心点
              → 用最近两点更新切向
      步长  : **自适应** step = clip(0.5×局部河宽, 5m, 0.5×interval)
              细河步子小(跟得住弯), 宽河步子大(不浪费算力);
              若前进后落到面外, 步长减半重试(最多 4 次)—— 弯道顶点自动收步。
      守卫  : 重新定心的横移 > 0.6×局部宽 → 停(跳到别的湾);
              前进量 <= 0.2×step → 停(折返)。
      方向  : 从起点**正反两个方向**各走一遍再拼接, 因此起点落在河段中部也不丢覆盖。

    返回 (pts[M,2], widths[M]); 走不动 → (空, 空), 由调用方回退老路径。
    """
    ext = np.asarray(ext, dtype=float)
    holes = [np.asarray(h, dtype=float) for h in (holes or [])]

    def chord_at(pt, tdir, pick_containing=True):
        px, py = float(pt[0]), float(pt[1])
        dx, dy = float(-tdir[1]), float(tdir[0])
        ch = _cs_chords(px, py, dx, dy, ext, holes)
        if not ch:
            return None
        if pick_containing:
            a, b = min(ch, key=lambda z: abs(0.5 * (z[0] + z[1])))
        else:
            a, b = max(ch, key=lambda z: abs(z[1] - z[0]))
        w = abs(b - a) * float(np.hypot(dx, dy))
        m = np.array([px, py], dtype=float) + 0.5 * (a + b) * np.array([dx, dy])
        return m, w

    c, ax, pmin, pmax = _cs_axis(ext)
    axis_len = pmax - pmin
    if axis_len < 1e-6:
        return np.empty((0, 2)), np.empty((0,))

    # ── 起点: 从河段**中部**起步, 向两侧退让寻找可用位置 ──
    #   ★不能取主轴极值端★: 那里是河段端帽, 垂直截面被端边剪短, 行走会在端帽内打转
    #   (实测: 种子放 2% 处, 走 34 步全在端帽区兜圈, 宽度估计由 32m 退化到 12m)。
    seed = None
    for frac in (0.50, 0.42, 0.58, 0.34, 0.66, 0.25, 0.75, 0.15, 0.85):
        cand = c + ax * (pmin + frac * axis_len)
        r = chord_at(cand, ax, pick_containing=False)
        if r is not None and _pip_raycast(r[0][None, :], ext, holes)[0]:
            seed = r
            break
    if seed is None:
        return np.empty((0, 2)), np.empty((0,))

    # ── 起步方向: 取**局部**河道走向, 而不是全局主轴 ──
    #   蜿蜒河道中部的局部走向可能与全局主轴接近垂直。做法: 扫描 180° 内的方向,
    #   取"垂直于该方向的截面弦最短"者 —— 河宽是河道的最窄截面, 该方向即局部流向。
    seed_pt = np.asarray(seed[0], dtype=float)
    best_dir, best_w = ax, float("inf")
    for a_deg in range(0, 180, 10):
        a = np.radians(a_deg)
        dv = np.array([np.cos(a), np.sin(a)])
        rr = chord_at(seed_pt, dv, pick_containing=True)
        if rr is not None and rr[1] < best_w:
            best_w, best_dir = float(rr[1]), dv
    if np.isfinite(best_w) and best_w > 0:
        seed = (seed_pt, best_w)
        ax_start = best_dir
    else:
        ax_start = ax

    # 扇形候选方向: 优先直行, 走不通再逐级转向(急弯处必需)
    _FAN = [0.0]
    for _a in (4, 8, 13, 19, 26, 34, 43, 53, 64, 75):
        _FAN.extend([np.radians(_a), -np.radians(_a)])

    def walk(start_pt, start_w, t0):
        """从起点沿一个方向走到底。

        ★关键★ 每一步在 t 的 ±75° 扇形内搜索可行方向, 优先小转角。
        只沿固定 t 前进(哪怕带平滑)在急弯处必然走出河面 —— 实测蛇形夹具
        走 5 步(73m)就因 t 跟不上转弯而停; 扇形搜索后可走完全程。
        """
        pts = []
        widths = []
        P = np.asarray(start_pt, dtype=float)
        t = np.asarray(t0, dtype=float)
        nt = float(np.hypot(*t))
        if nt < 1e-9:
            return pts, widths
        t = t / nt
        w_cur = float(start_w) if start_w > 0 else float(interval_m)
        shrink = 1.0        # 上一步转角大 → 本步收步(急弯自适应)
        relax_run = 0       # 连续使用放宽守卫的步数
        for _ in range(int(max_steps)):
            # 步长必须显著小于局部曲率半径, 否则急弯处一步就跨到河外。
            #   0.35×河宽 是实测折中: 再大在 R≈1.6×半宽 的急弯走不过去,
            #   再小则步数(与耗时)线性上升。
            step0 = float(np.clip(0.35 * w_cur * shrink, 4.0, 0.5 * float(interval_m)))
            best = None
            step = step0
            n_relax = 0
            for _shrink in range(6):
                # ★A.8.3★ 每一步先按严格守卫找方向; 找不到再用放宽守卫**就地重试**,
                #   而不是直接停。真实岸线在局部锯齿/窄颈处常让严格守卫全扇形失败,
                #   一停就是整段少走几百米 —— 诊断实测 gap 中位数 222m, 而端部剔除
                #   总共只削 0~240m, 说明缺口主要来自行走提前终止。
                for _pass in (0, 1):
                    lat_k = 0.6 if _pass == 0 else 1.2
                    for ang in _FAN:
                        ca, sa = float(np.cos(ang)), float(np.sin(ang))
                        dirv = np.array([t[0] * ca - t[1] * sa, t[0] * sa + t[1] * ca])
                        cand = P + dirv * step
                        if not _pip_raycast(cand[None, :], ext, holes)[0]:
                            continue
                        r = chord_at(cand, dirv, pick_containing=True)
                        if r is None:
                            continue
                        mm, ww = r
                        if w_cur > 0 and float(np.hypot(*(mm - cand))) > lat_k * w_cur:
                            continue
                        if _pass == 0 and w_cur > 0 and ww > 3.0 * w_cur:
                            continue
                        if float(np.dot(mm - P, dirv)) <= 0.2 * step:
                            continue
                        if not _pip_raycast((0.5 * (P + mm))[None, :], ext, holes)[0]:
                            continue
                        best = (mm, ww, dirv)
                        n_relax = _pass
                        break
                    if best is not None:
                        break
                if best is not None:
                    break
                step *= 0.5
                if step < 2.0:
                    break
            if best is None:
                break
            m, w_new, dirv = best
            # 防打转: 新点若贴近 6 步以前访问过的位置, 说明在端帽/环形区兜圈, 停。
            if len(pts) > 6:
                old_pts = np.asarray(pts[:-6])
                if float(np.hypot(*(old_pts - m).T).min()) < 0.5 * max(w_cur, 5.0):
                    break
            pts.append(m)
            widths.append(float(w_new))
            adv = m - P
            n2 = float(np.hypot(*adv))
            if n2 > 1e-9:
                newt = adv / n2
                turn = abs(float(np.arctan2(t[0] * newt[1] - t[1] * newt[0],
                                            t[0] * newt[0] + t[1] * newt[1])))
                shrink = 0.5 if turn > np.radians(22) else min(1.0, shrink * 1.5)
                t = newt                              # 直接采用实际前进方向
            P = m
            if w_new > 0:
                w_cur = 0.5 * w_cur + 0.5 * w_new
        return pts, widths

    fwd_p, fwd_w = walk(seed[0], seed[1], ax_start)
    bwd_p, bwd_w = walk(seed[0], seed[1], -ax_start)
    pts = list(reversed(bwd_p)) + [np.asarray(seed[0], dtype=float)] + fwd_p
    widths = list(reversed(bwd_w)) + [float(seed[1])] + fwd_w
    if len(pts) < 2:
        return np.empty((0, 2)), np.empty((0,))
    return np.asarray(pts, dtype=float), np.asarray(widths, dtype=float)


def _centerline_cross_section_core(ext, holes, interval_m, refine_iters=3):
    """★A.6-alt★ 截面中点法中心线 (纯 numpy; 绿线思路 + 局部切向精修修 S 弯'空洞')。

    - pass0: 主轴(PCA≈最小外接矩形长轴) + 沿轴每 interval 投**全局垂线**截河流面, 取(最长弦)中点
      → 初始 guide。这就是 pre-A.6 绿线的做法。
    - pass1..n: 沿(平滑后)guide 弧长布站, 每站取**垂直于局部切向**的截面, 选**含 guide 点的弦**取中点。
      局部切向使截面在 S 弯/宽弯处与河道对齐 → 修绿线 R10 的'一条全局垂线截到两段、取错段'导致的
      中心线跳变/出河('空洞'); 且**始终是单条线**, 对岸线锯齿不敏感(弦中点抵消两岸锯齿) → 不会像
      中轴线那样在锯齿/宽域碎裂出乱线与闭合环。

    返回 (pts[M,2], widths[M]); 退化 → (空, 空), 由包装层回退 MABR 长轴。
    """
    ext = np.asarray(ext, dtype=float)
    holes = [np.asarray(h, dtype=float) for h in (holes or [])]

    # ★A.8 (v0.7.0)★ 首选"沿河道行走"(见 _centerline_walk):
    #   可用环境变量 PREPROCESS_RIVER_WALK=0 关闭, 回退到旧的全局轴扫描路径,
    #   便于在真实数据上 A/B 对比。
    #   全局轴扫描在细窄蜿蜒河道上会横切河湾(实测出河率 11~29%), 行走法不依赖全局轴。
    #   行走失败(极短块/异常形状)才回退到下面的全局轴扫描 + 局部精修。
    _use_walk = os.environ.get('PREPROCESS_RIVER_WALK', '1') not in ('0', 'false', 'False')
    walk_pts, walk_w = (_centerline_walk(ext, holes, interval_m)
                        if _use_walk else (np.empty((0, 2)), np.empty((0,))))
    if _use_walk and len(walk_pts) < 2:
        # ★A.8.2★ 严格守卫在少数真实形态上会让行走在第一步就失败 → 整块退回全局轴扫描。
        #   实测: A.8.1 收紧守卫后, 真实数据退化回退由 0 升到 10 个连通块(8.4%),
        #   窄段 linear 由 2756 降到 2649, 且退回的旧路径带端部外推, 产生了
        #   2.4% 的"端点落在河面外"。故先用放宽守卫再走一次, 尽量不回退。
        walk_pts, walk_w = _centerline_walk(ext, holes, interval_m, relaxed=True)
    # ── 覆盖率闸门: 行走走短了就退回全局轴扫描 ──
    #   ★A.8.4★ 两种方法的失效模式是**互补**的:
    #     · 全局轴扫描: 站位铺满整条主轴, **不会提前停**, 但在蜿蜒河道上横切河湾;
    #     · 沿河道行走: 忠实跟随河道(细窄蜿蜒河的唯一正解), 但遇到局部锯齿/窄颈
    #       可能提前停下, 整段少走几百米 —— 这正是"宽河 S 弯不如 A.7.1"的原因。
    #   故按覆盖率择优: 用面积与周长按矩形模型估出河段应有长度 L_est,
    #   行走长度不足 0.75×L_est 时, 再跑一次轴扫描, 取更长者。
    #   这样宽缓河拿回轴扫描的连贯性, 细窄蜿蜒河保留行走的正确性。
    def _axis_scan():
        # ★A.7.4★ 主轴用**边长加权** PCA, 不用顶点计数 PCA。
        #   顶点计数 PCA 对岸线顶点疏密敏感: 长边采样密、短边采样疏时, 方差被短边的
        #   坐标值主导 → 近方形河块(如被两条道路夹出的 150×120m 块)会把主轴选到**横向**,
        #   pass0 沿横向布站 → 站数不足 → 整块退化(实测 150m 块直接产 0 点)。
        #   边长加权后主轴只取决于形状, 与顶点密度无关。
        seg_mid = 0.5 * (ext[:-1] + ext[1:])
        seg_len = np.hypot(*(ext[1:] - ext[:-1]).T)
        if seg_len.sum() > 1e-9 and len(seg_mid) >= 2:
            wgt = seg_len / seg_len.sum()
            c = (seg_mid * wgt[:, None]).sum(axis=0)
            Xw = seg_mid - c
            cov = (Xw * wgt[:, None]).T @ Xw
        else:
            c = ext.mean(axis=0)
            Xw = ext - c
            cov = Xw.T @ Xw
        evals, V = np.linalg.eigh(cov)
        ax = V[:, int(np.argmax(evals))]
        X = ext - c
        proj = X @ ax
        p1 = c + proj.min() * ax
        p2 = c + proj.max() * ax
        d = p2 - p1
        L = float(np.hypot(*d))
        if L < 1e-6:
            return np.empty((0, 2)), np.empty((0,))
        u = d / L
        perp = np.array([-u[1], u[0]])

        def mid_of(px, py, dirx, diry, pick=False):
            ch = _cs_chords(px, py, dirx, diry, ext, holes)
            if not ch:
                return None
            if pick:
                a, b = min(ch, key=lambda c: abs(0.5 * (c[0] + c[1])))   # pick 点在截面 t≈0
            else:
                a, b = max(ch, key=lambda c: abs(c[1] - c[0]))
            # ★宽度口径 = **所选弦**的长度(=中心线所在这条河道的局部宽度), 不是"各弦总和"。
            #   否则在 S 弯/回折处一条截线会穿河道多段, 总和被夸大(可达真河宽数倍) → 该处宽度被误判
            #   ≥ 阈值 → river_narrow_cross_segments 把这些段当宽段丢弃 → S 弯出现"线段缺失"。
            w = abs(b - a) * float(np.hypot(dirx, diry))
            m = np.array([px, py], dtype=float) + 0.5 * (a + b) * np.array([dirx, diry])
            return m, w

        mids = []
        # ★A.7.1 (v0.6.3)★ 站位从主轴两端各**内缩半个 interval**。
        #   原来 dist 从 0 起、到 L 止, 首末站正好落在多边形沿主轴的极值处 = **边界上**。
        #   站点贴边时, 垂直截面的射线会先从端边穿出、再到不了对岸 → 弦被端边截断,
        #   弦中点因此失去意义并大幅偏移。实测直河末端: 站点 x=0.16(边界), 截得弦长 287m
        #   (真值 500m), 中点被甩到 y=106.5; 该坏点再经 3 轮精修被逐步放大, 末端整段甩向岸边。
        #   斜切端"拐向角落"与直河末端"甩出去"是同一个病。
        #   内缩后端部覆盖由 _cs_extend_ends 用"切向直线 + 二分求边界"补回 —— 那条路径
        #   不依赖截面, 对端边形状不敏感。
        for inset in (min(0.5 * interval_m, 0.25 * L), 0.0):
            # 短块(纵向 < 2×interval)内缩后可能凑不满 2 站 → 退回 inset=0 再试一次,
            # 此时端部偏差由 _cs_trim_degenerate_ends 兜住。
            mids = []
            dist = inset
            stop = L - inset
            while dist <= stop + 1e-9:
                p = p1 + u * dist
                r = mid_of(p[0], p[1], perp[0], perp[1])
                if r is not None:
                    mids.append(r[0])
                dist += interval_m
            if len(mids) >= 2:
                break
        if len(mids) < 2:
            return np.empty((0, 2)), np.empty((0,))
        mids = np.asarray(mids, dtype=float)

        for _ in range(int(refine_iters)):
            guide = _cs_smooth(mids)
            seg = np.hypot(*np.diff(guide, axis=0).T)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            if s[-1] < 1e-6:
                break
            tang = _cs_tangents(guide)
            new = []
            # ★A.7★ 补末站: np.arange 不含终点, 末尾不足一个 interval 的一截会被丢,
            #   叠加在每轮上又是一处系统性缩短。显式把弧长终点补进站位表。
            stations = list(np.arange(0.0, s[-1], interval_m))
            if (not stations) or (s[-1] - stations[-1] > 1e-6):
                stations.append(float(s[-1]))
            for st in stations:
                i = min(max(int(np.searchsorted(s, st)), 1), len(guide) - 1)
                f = (st - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
                p = guide[i - 1] * (1 - f) + guide[i] * f
                tg = tang[min(i, len(tang) - 1)]
                pdir = np.array([-tg[1], tg[0]])
                r = mid_of(p[0], p[1], pdir[0], pdir[1], pick=True)
                if r is not None:
                    new.append(r[0])
            if len(new) < 2:
                break
            mids = np.asarray(new, dtype=float)

        guide = mids
        tang = _cs_tangents(guide)
        pts = []
        widths = []
        for k in range(len(guide)):
            p = guide[k]
            tg = tang[k]
            pdir = np.array([-tg[1], tg[0]])
            r = mid_of(p[0], p[1], pdir[0], pdir[1], pick=True)
            if r is not None:
                pts.append(r[0])
                widths.append(r[1])
        if len(pts) < 2:
            return mids, np.zeros(len(mids))
        pts = np.asarray(pts, dtype=float)
        widths = np.asarray(widths, dtype=float)

        # ★A.7.1 (v0.6.3)★ 端头修复 —— 必须在滚动中位数之前做(否则退化宽度已被抹平)。
        #   现象(真实数据): 河流面被道路/桥切成斜直边时, 中心线末端不指向端边中点,
        #   而是**拐向端边的一个角**, 偏离真中心线可达半个河宽。
        #   根因: pass0 的首/末站取自主轴极值 p1 = c + proj.min()*ax, 它落在多边形沿主轴
        #   最远的**顶点**(斜切边的角)。该处垂线弦近乎退化(实测弦长仅全线中位的 0~23%),
        #   弦中点就是那个角。A.7 之前非对称平滑把两端各削 ~400m, 恰好削掉了这个坏端点,
        #   缺陷被掩盖; A.7 保端点后暴露。
        #   修法: ①按**原始**弦宽剔除两端连续的退化站; ②再沿局部切向逐步外推、每步重新
        #   定心, 直到出面 → 既去掉尖角伪点, 又把纵向覆盖补回真实端边。
        pts, widths = _cs_trim_degenerate_ends(pts, widths)
        pts, widths = _cs_extend_ends(pts, widths, ext, holes, interval_m, mid_of)
        if len(pts) < 2:
            return np.empty((0, 2)), np.empty((0,))

        # 发卡/回折顶点处, 垂直于切向的截面会近乎"沿河道"→ 该点弦被拉长成孤立尖峰。
        #   滚动中位数抹掉这类孤立尖峰(真正的宽段跨连续多点, 中位数会保留), 避免这些点被误判宽而丢段。
        if len(widths) >= 3:
            sm = widths.copy()
            for i in range(len(widths)):
                lo = max(0, i - 2)
                hi = min(len(widths), i + 3)
                sm[i] = float(np.median(widths[lo:hi]))
            widths = sm
        # ★A.7★ 折角补站: 保端点后, 发卡/回折顶点处相邻两站的**连线**可能抄近路切出河岸
        #   (站点本身仍在河内)。算法端是拿这条折线做"跨越相交判定 + 交叉角", 连线出河
        #   = 该处判定错位, 故在出河的相邻站之间插一站 (至多 2 轮, 单调收敛)。
        pts, widths = _cs_densify_out_of_river(pts, widths, ext, holes)
        return pts, widths
    _wh = float(np.median(walk_w)) if len(walk_w) else 0.0
    est_len = _cs_reach_length_estimate(ext, holes, _wh)
    walk_res = None
    if len(walk_pts) >= 2:
        if len(walk_pts) >= 2:
            pts, widths = walk_pts, walk_w
            if len(widths) >= 3:
                sm = widths.copy()
                for i in range(len(widths)):
                    sm[i] = float(np.median(widths[max(0, i - 2):min(len(widths), i + 3)]))
                widths = sm
            pts, widths = _cs_trim_degenerate_ends(pts, widths)
            pts, widths = _cs_trim_end_hooks(pts, widths, interval_m)
            pts, widths = _cs_drop_outside_ends(pts, widths, ext, holes)
            # ★A.8★ 行走法**不做端部外推**: walk 本身就一路走到河道尽头(实测纵向覆盖
            #   100%), 再沿切向外推 2×interval 只会冲出河心 —— 实测直河末端偏离
            #   由 0 升到 212m(85% 半宽)。外推只属于旧的"全局轴扫描"路径(那条路的
            #   端部会被剔除锥缩站削短, 需要补回)。
            pts, widths = _cs_densify_out_of_river(pts, widths, ext, holes)
            walk_res = (pts, widths)
    if walk_res is not None:
        w_len = float(np.hypot(*np.diff(walk_res[0], axis=0).T).sum()) if len(walk_res[0]) > 1 else 0.0
        if est_len <= 0 or w_len >= 0.75 * est_len:
            return walk_res
        alt = _axis_scan()
        if alt is not None and len(alt[0]) >= 2:
            # ★A.8.4★ 比的是**有效长度**(扣掉落在河面外的部分), 不能只比长度 ——
            #   轴扫描常靠横切河湾"变长", 单纯比长度会把蜿蜒河judged错。
            a_len = _cs_effective_length(alt[0], ext, holes)
            w_len = _cs_effective_length(walk_res[0], ext, holes)
            if a_len > w_len:
                logger.info(
                    f"  河流中心线: 行走覆盖不足 ({w_len:.0f}m < 0.75×{est_len:.0f}m), "
                    f"改用全局轴扫描 ({a_len:.0f}m)")
                return alt
        return walk_res
    res = _axis_scan()
    return res if res is not None else (np.empty((0, 2)), np.empty((0,)))

def _cs_trim_end_hooks(pts, widths, interval_m, max_turn_deg: float = 100.0,
                       max_trim_frac: float = 0.25):
    """剪掉两端"往回弯折"的钩子。

    ★A.8.2★ 真实数据现象(河道宽面端点细节.png): 线走到道路切口附近后,
    最后一小段拐回来形成一个钩。成因是行走进入端帽区后, 扇形搜索(±75°)找到了
    一个沿端帽横向的可行方向, 于是继续走 —— 走出来的就是钩。

    判据: 以端部往内约 1 个 interval 的走向为基准, 若端点段与基准夹角 > max_turn_deg,
    就把端点删掉, 逐点向内重复。只削端部, 不动中段。
    """
    pts = np.asarray(pts, dtype=float)
    widths = np.asarray(widths, dtype=float)
    n = len(pts)
    if n < 5:
        return pts, widths
    cap = max(1, int(n * max_trim_frac))
    cos_lim = float(np.cos(np.radians(max_turn_deg)))

    def _ref_dir(idx_list):
        """idx_list: 由外向内的下标; 取跨约 1 个 interval 的内侧走向"""
        acc = 0.0
        j = 1
        while j < len(idx_list) - 1 and acc < float(interval_m):
            acc += float(np.hypot(*(pts[idx_list[j]] - pts[idx_list[j + 1]])))
            j += 1
        v = pts[idx_list[1]] - pts[idx_list[min(j, len(idx_list) - 1)]]
        nv = float(np.hypot(*v))
        return (v / nv) if nv > 1e-9 else None

    lo, hi = 0, n - 1
    for _ in range(cap):
        idx = list(range(lo, hi + 1))
        if len(idx) < 5:
            break
        ref = _ref_dir(idx)
        if ref is None:
            break
        e = pts[idx[0]] - pts[idx[1]]
        ne = float(np.hypot(*e))
        if ne < 1e-9 or float(np.dot(e / ne, ref)) >= cos_lim:
            break
        lo += 1
    for _ in range(cap):
        idx = list(range(hi, lo - 1, -1))
        if len(idx) < 5:
            break
        ref = _ref_dir(idx)
        if ref is None:
            break
        e = pts[idx[0]] - pts[idx[1]]
        ne = float(np.hypot(*e))
        if ne < 1e-9 or float(np.dot(e / ne, ref)) >= cos_lim:
            break
        hi -= 1
    if hi - lo + 1 < 2:
        return pts, widths
    return pts[lo:hi + 1], widths[lo:hi + 1]


def _cs_drop_outside_ends(pts, widths, ext, holes):
    """安全网: 删掉两端落在河面**外**的点。

    ★A.8.2★ 诊断脚本实测有 2.4% 的端点 clearance < 0(在面外)。中心线的端点跑到
    河面外没有任何合理解释, 一律删掉。中间点不动(极少数情况下窄颈处的数值抖动
    不应导致整条线被拆断)。
    """
    pts = np.asarray(pts, dtype=float)
    widths = np.asarray(widths, dtype=float)
    n = len(pts)
    if n < 2:
        return pts, widths
    inside = _pip_raycast(pts, ext, holes)
    lo = 0
    while lo < n - 1 and not inside[lo]:
        lo += 1
    hi = n - 1
    while hi > lo and not inside[hi]:
        hi -= 1
    if hi - lo + 1 < 2:
        return pts, widths
    return pts[lo:hi + 1], widths[lo:hi + 1]


def _cs_trim_degenerate_ends(pts, widths, taper_ratio: float = 0.70,
                             max_trim_frac: float = 0.3):
    """剔除两端"截面被端边剪断"的站。

    ★A.8★ 判据改为**按弧长比较**, 而不是比相邻两站:
      行走法的相邻点只隔 10~20m, 宽度比恒接近 1, 相邻比较完全失效 ——
      端帽区(道路斜切出的横断边)里所有点的宽度都被剪短, 却一站也剔不掉,
      结果线沿端帽横向走、端点被甩到一侧(实测直河末端偏离 213m = 85% 半宽)。
      现改为: 把端点的宽度与"往内 1 个 interval 弧长处的宽度中位数"比,
      低于 taper_ratio 即剔除。这样密集点与稀疏点两种输入都成立。
    """
    pts = np.asarray(pts, dtype=float)
    widths = np.asarray(widths, dtype=float)
    n = len(pts)
    if n < 5:
        return pts, widths
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    span = min(max(total * 0.25, 1.0), 200.0)      # 参考段弧长(不超过 1/4 全长)
    cap = max(1, int(n * max_trim_frac))

    def _inner_ref(i, forward):
        if forward:
            sel = (cum > cum[i]) & (cum <= cum[i] + span)
        else:
            sel = (cum < cum[i]) & (cum >= cum[i] - span)
        vals = widths[sel]
        vals = vals[vals > 0]
        return float(np.median(vals)) if len(vals) else 0.0

    lo = 0
    while lo < cap and lo < n - 4:
        ref = _inner_ref(lo, True)
        if ref <= 0 or widths[lo] >= taper_ratio * ref:
            break
        lo += 1
    hi = n - 1
    cut = 0
    while cut < cap and hi > lo + 3:
        ref = _inner_ref(hi, False)
        if ref <= 0 or widths[hi] >= taper_ratio * ref:
            break
        hi -= 1
        cut += 1
    if hi - lo + 1 < 2:
        return pts, widths
    return pts[lo:hi + 1], widths[lo:hi + 1]


def _cs_split_zigzag(pts, widths, interval_m,
                     jump_factor: float = 2.5, turn_deg: float = 135.0):
    """把中心线在**跳变 / 大折返**处切开, 返回 [(pts_i, widths_i), ...]。

    ★A.7.2★ 汇流口/分叉处一个河流面里有多条河道, 弦选取可能在支流间来回切换,
    产生斜穿折返的假线。这些假线会被切成 50m 段带着**错误方位角**进
    linear_cross_indexed, 而它们仍在河面内, 出河检测抓不到 —— 必须显式切开。

    ★A.8.1 修正★ 判据必须**适配点间距**:
      行走法输出的点间距只有 10~50m(全局轴扫描时是 100m)。沿用固定阈值后, 真实数据
      分岔口切分数由 7 暴涨到 26, 表现为宽河 S 弯出现肉眼可见的断线。
      现在:
        · 跳变阈值 = max(2.5×interval, 3×**实际中位间距**);
        · 大折角处先判断"折返幅度": 该点偏离前后两点连线的距离 < 0.6×局部河宽 时,
          视为单点抖动 → **删掉该点**而不是切开(切开会白白制造一段缺口);
          只有偏离显著(真的横穿到别的河道)才切。
    """
    pts = np.asarray(pts, dtype=float)
    widths = np.asarray(widths, dtype=float)
    n = len(pts)
    if n < 2:
        return []
    seg = np.hypot(*np.diff(pts, axis=0).T)
    med = float(np.median(seg)) if len(seg) else float(interval_m)
    max_jump = max(float(jump_factor) * float(interval_m), 3.0 * med)

    # ── 1) 先剔除"单点抖动": 大折角但偏离幅度小 ──
    keep = np.ones(n, dtype=bool)
    if n >= 3:
        for i in range(1, n - 1):
            a_, b_, c_ = pts[i - 1], pts[i], pts[i + 1]
            v = c_ - a_
            nv = float(np.hypot(*v))
            if nv < 1e-9:
                continue
            u = v / nv
            off = abs(float((b_ - a_)[0] * u[1] - (b_ - a_)[1] * u[0]))
            d1 = b_ - a_
            d2 = c_ - b_
            n1, n2 = float(np.hypot(*d1)), float(np.hypot(*d2))
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cosang = float(np.clip(np.dot(d1 / n1, d2 / n2), -1.0, 1.0))
            if np.degrees(np.arccos(cosang)) > float(turn_deg):
                w_loc = widths[i] if widths[i] > 0 else med
                if off < 0.6 * w_loc and max(n1, n2) <= max_jump:
                    keep[i] = False          # 小幅抖动 → 删点, 不切
    pts = pts[keep]
    widths = widths[keep]
    n = len(pts)
    if n < 2:
        return []

    # ── 2) 剩下的大跳变 / 大折返才切开 ──
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cut = set(int(i) for i in np.where(seg > max_jump)[0])
    if n >= 3:
        d = np.diff(pts, axis=0)
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        turn = np.degrees(np.arccos(np.clip(np.sum(d[:-1] * d[1:], axis=1), -1.0, 1.0)))
        for i in np.where(turn > float(turn_deg))[0]:
            cut.add(int(i))
            cut.add(int(i) + 1)
    parts = []
    start = 0
    for i in range(n - 1):
        if i in cut:
            if i + 1 - start >= 2:
                parts.append((pts[start:i + 1], widths[start:i + 1]))
            start = i + 1
    if n - start >= 2:
        parts.append((pts[start:], widths[start:]))
    return parts


def _cs_extend_ends(pts, widths, ext, holes, interval_m, mid_of=None,
                    max_extend_factor: float = 2.0,
                    margin_m: float = 5.0,
                    bank_floor_ratio: float = 0.15):
    """两端补回纵向覆盖: **沿拟合圆弧推进到河岸边界**(二分求交)。

    ★A.7.5 (v0.6.6)★ 本函数改了四版, 结论如下, 记录在案免得再走弯路:

      · A.7.1「切向直线 + 二分到边界」
          覆盖最好(真实数据窄段 linear 2359 条), 但直线在弯道处离开河心 ——
          端点偏离达 70% 半宽, 表现为线头从岸边起再横折入河道(仅 1 处)。
      · A.7.3「逐步再定心 + **截面**守卫」
          偏移压到 16~20% 半宽, 但截面在端部本就被端边剪断, 守卫频繁早停 →
          2359 → **2202**, 端头出现肉眼可见的断线。
      · A.7.4「圆弧外推 + **岸距**守卫(按走向排除端边)」
          合成算例覆盖大幅回升, 但真实数据 **2199**, 与 A.7.3 持平 ——
          真实岸线锯齿使大量岸段方向散乱, 走向过滤失效, 守卫照样早停。
          **合成夹具无法代表真实岸线, 这是这三轮反复的根因。**
      · A.7.5(本版) 回到 A.7.1 的推进方式(二分到边界, 保住覆盖),
          **只把"直线"换成"拟合圆弧"** —— 针对性修掉 A.7.1 唯一的失败模式(弯道漂移),
          不再引入任何会提前终止的守卫; 仅保留一道极松的兜底(贴岸 < 15% 河宽才停),
          正常情况下不会触发。
    """
    P = [np.asarray(q, dtype=float) for q in pts]
    W = [float(w) for w in widths]
    if len(P) < 2:
        return np.asarray(P, dtype=float), np.asarray(W, dtype=float)
    max_ext = float(interval_m) * float(max_extend_factor)

    # ★A.8★ 参考段按**弧长**取(约 1 个 interval), 不能再按固定点数取:
    #   行走法的点间距只有 10~20m, "最后 5 个点"仅跨 50~80m, 拟合圆弧的曲率噪声极大,
    #   外推 200m 会直接飞出河道(实测端部偏离升至 90% 半宽)。
    arr = np.asarray(P, dtype=float)
    seglen = np.hypot(*np.diff(arr, axis=0).T) if len(arr) > 1 else np.array([0.0])
    for side in (0, 1):
        want = float(interval_m)
        k = 2
        acc = 0.0
        while k < len(P) and acc < want:
            acc += float(seglen[k - 2] if side == 0 else seglen[len(seglen) - (k - 1)])
            k += 1
        k = int(np.clip(k, 3, len(P)))
        ref = [v for v in (W[:k] if side == 0 else W[-k:]) if v > 0]
        w_ref = float(np.median(ref)) if ref else 0.0
        tail = np.array(P[:k][::-1]) if side == 0 else np.array(P[-k:])
        end_pt = tail[-1]
        # ★A.8★ 方向取**整个参考段**的跨度, 不是最后两点之差 ——
        #   行走法相邻点仅隔 10~20m, 单段差分的方向噪声在 200m 外推后被放大约 20 倍
        #   (实测直河末端偏离 224m = 90% 半宽)。
        tg = tail[-1] - tail[0]
        nrm = float(np.hypot(*tg))
        if nrm < 1e-9 and len(tail) >= 2:
            tg = tail[-1] - tail[-2]
            nrm = float(np.hypot(*tg))
        if nrm < 1e-9:
            continue
        tg = tg / nrm

        center, R, ok = _cs_fit_arc(tail)
        use_arc = bool(ok and np.isfinite(R) and R > max(2.0 * w_ref, 1.0) and R < 1e7)
        if use_arc:
            rad = end_pt - center
            rr = float(np.hypot(*rad))
            if rr < 1e-9:
                use_arc = False
            else:
                sgn = 1.0 if float(rad[0] * tg[1] - rad[1] * tg[0]) > 0 else -1.0
                th0 = float(np.arctan2(rad[1], rad[0]))

        def _at(dist):
            if use_arc:
                dth = sgn * dist / R
                return center + rr * np.array([np.cos(th0 + dth), np.sin(th0 + dth)])
            return end_pt + tg * dist

        def _acceptable(q):
            if not _pip_raycast(q[None, :], ext, holes)[0]:
                return False
            # 极松兜底: 只在明显贴岸时否决(正常外推不会触发)。
            #   绝对上限 25m —— 否则宽河(如 500m)的比例阈值会让外推在离切口 75m 处
            #   就停下, 又变成肉眼可见的断线。
            if w_ref > 0:
                floor = min(bank_floor_ratio * w_ref, 25.0)
                if float(_cs_boundary_distance(q[None, :], ext, holes)[0]) < floor:
                    return False
            return True

        # 二分求出沿弧线仍可接受的最远距离(与 A.7.1 同一策略, 保住覆盖)
        lo_d, hi_d = 0.0, max_ext
        if _acceptable(_at(hi_d)):
            lo_d = hi_d
        else:
            for _ in range(24):
                mid_d = 0.5 * (lo_d + hi_d)
                if _acceptable(_at(mid_d)):
                    lo_d = mid_d
                else:
                    hi_d = mid_d
        d_use = lo_d - float(margin_m)
        if d_use <= float(interval_m) * 0.2:
            continue
        q_end = _at(d_use)
        if not _pip_raycast(q_end[None, :], ext, holes)[0]:
            continue
        # 弧长较长时补一个中间点, 保证折线贴合弧线
        cands = [_at(0.5 * d_use), q_end] if d_use > float(interval_m) * 0.8 else [q_end]
        prev = end_pt
        for q in cands:
            if not _pip_raycast((0.5 * (prev + q))[None, :], ext, holes)[0]:
                break
            if side == 0:
                P.insert(0, q)
                W.insert(0, w_ref if w_ref > 0 else W[0])
            else:
                P.append(q)
                W.append(w_ref if w_ref > 0 else W[-1])
            prev = q
    return np.asarray(P, dtype=float), np.asarray(W, dtype=float)


def _cs_densify_out_of_river(pts, widths, ext, holes, max_rounds: int = 2):
    """相邻中心线点的连线中点若落在河**外**, 在两点间补一站; 至多 max_rounds 轮。

    返回 (pts[M,2], widths[M])。纯 numpy, 与核心同口径 (截面取含该点的弦)。
    """
    pts = np.asarray(pts, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if len(pts) < 2:
        return pts, widths

    def _mid_at(px, py, dx, dy):
        ch = _cs_chords(px, py, dx, dy, ext, holes)
        if not ch:
            return None
        a, b = min(ch, key=lambda c: abs(0.5 * (c[0] + c[1])))
        w = abs(b - a) * float(np.hypot(dx, dy))
        m = np.array([px, py], dtype=float) + 0.5 * (a + b) * np.array([dx, dy])
        return m, w

    P = [p for p in pts]
    W = [float(w) for w in widths]
    for _ in range(int(max_rounds)):
        mids = 0.5 * (np.asarray(P[:-1]) + np.asarray(P[1:]))
        bad = np.where(~_pip_raycast(mids, ext, holes))[0]
        if len(bad) == 0:
            break
        bad = set(int(i) for i in bad)
        newP, newW = [], []
        for i in range(len(P)):
            newP.append(P[i])
            newW.append(W[i])
            if i in bad:
                m = 0.5 * (P[i] + P[i + 1])
                tg = P[i + 1] - P[i]
                nrm = float(np.hypot(*tg))
                if nrm < 1e-9:
                    continue
                tg = tg / nrm
                r = _mid_at(m[0], m[1], -tg[1], tg[0])
                if r is not None:
                    newP.append(r[0])
                    newW.append(float(r[1]))
        if len(newP) == len(P):
            break
        P, W = newP, newW
    return np.asarray(P, dtype=float), np.asarray(W, dtype=float)


def river_polygon_centerline(polygon, interval_m: float = 100.0, refine_iters: int = 3,
                             return_parts: bool = False):
    """
    ★P7 (v0.6) + A.6-alt★ 从河流**面**提取中心线 + 逐站宽度。

    做法 = 截面中点法 + 局部切向精修 (见 _centerline_cross_section_core):
      - 沿主轴投垂线截河流面, 取(主河道)中点连成中心线 → 跟随河道横向摆动;
      - **局部切向精修**修正强 S 弯/宽弯处"一条全局垂线截错段"的跳变/出河(pre-A.6 的 R10 空洞);
      - 始终**单条线**、对岸线锯齿不敏感 → 不产生中轴线那样的乱线/闭合环。

    返回:
      - return_parts=False (默认, 向后兼容): (widths, width_points, center_line) 单条,
        多段时取**最长**的一段;
      - return_parts=True: [(widths, width_points, line), ...] 全部分段。
      面为空 → ([], [], None) / []; 退化(核心产不出≥2点) → 回退最小外接矩形长轴。

    refine_iters: 局部切向精修迭代次数(默认 3; 0 = 仅初始全局垂线中点, 等价旧绿线)。

    ★A.7.2★ 汇流口/分叉处中心线可能在支流间斜穿折返, 由 _cs_split_zigzag 切开成多段
    (见该函数说明)。分段共用同一 parent_feature_id, 算法端按 parent 去重, 不重复计费。
    """
    from shapely.geometry import LineString as LS, Point as Pt
    if polygon is None or getattr(polygon, "is_empty", True):
        return [] if return_parts else ([], [], None)
    # ── 优先: numpy 截面中点核心(含局部切向精修) ──
    ext = None
    holes = []
    try:
        ext = np.asarray(polygon.exterior.coords, dtype=float)
        holes = [np.asarray(r.coords, dtype=float) for r in polygon.interiors]
    except Exception:
        ext = None
    if ext is not None and len(ext) >= 4:
        try:
            pts, widths = _centerline_cross_section_core(ext, holes, interval_m, refine_iters)
        except Exception:
            pts, widths = np.empty((0, 2)), np.empty((0,))
        if len(pts) >= 2:
            parts = _cs_split_zigzag(pts, widths, interval_m) or [(pts, widths)]
            packed = []
            for ppts, pw in parts:
                packed.append((
                    [float(w) for w in pw],
                    [Pt(float(x), float(y)) for x, y in ppts],
                    LS([(float(x), float(y)) for x, y in ppts]),
                ))
            if return_parts:
                return packed
            # 兼容口径: 取最长的一段
            return max(packed, key=lambda t: t[2].length)
    # ── 退化回退: 最小外接矩形长轴(保证非空) ──
    # ★A.7 (v0.6.2)★ 原实现回退时返回 widths=[] / width_points=[] →
    #   下游 river_narrow_cross_segments 的守卫 `if ... or not width_points: return []`
    #   会让**整个连通块产 0 条 linear_cross 段** (QGIS 上表现为该段河完全没有中心线)。
    #   短河段/小水面(纵向 < ~1.5×interval)必然走这条路 → 成片丢段。
    #   修复: 回退轴上照样按 interval 布站, 用与核心同口径的截面取弦量宽度,
    #   保证 (widths, width_points) 与 center_line 同长且非空。
    try:
        rect = polygon.minimum_rotated_rectangle
        rc = list(rect.exterior.coords)
        edges = [(Pt(rc[i]), Pt(rc[i + 1]), Pt(rc[i]).distance(Pt(rc[i + 1])))
                 for i in range(4)]
        edges.sort(key=lambda e: e[2], reverse=True)
        p1, p2, axis_len = edges[0]
        axis = LS([p1, p2])
        widths, width_points = _axis_station_widths(
            polygon, p1, p2, axis_len, interval_m)
        if return_parts:
            return [(widths, width_points, axis)]
        return widths, width_points, axis
    except Exception:
        return [] if return_parts else ([], [], None)


def _axis_station_widths(polygon, p1, p2, axis_len, interval_m):
    """沿直轴 p1→p2 每 interval_m 布一站, 用垂直截线量该站河宽 (退化回退专用)。

    返回 (widths[float], width_points[shapely Point]); 任何异常 → 轴两端两站、
    宽度取 0 (宁可让段全部保留, 也不要因为量不出宽度而丢段)。
    """
    from shapely.geometry import Point as Pt
    try:
        ext = np.asarray(polygon.exterior.coords, dtype=float)
        holes = [np.asarray(r.coords, dtype=float) for r in polygon.interiors]
    except Exception:
        ext, holes = None, []
    dx, dy = p2.x - p1.x, p2.y - p1.y
    L = float(axis_len) or float(np.hypot(dx, dy))
    if L < 1e-6:
        return [], []
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    widths, width_points = [], []
    n = max(1, int(L // max(float(interval_m), 1e-6)))
    for i in range(n + 1):
        t = min(L, i * float(interval_m))
        sx, sy = p1.x + ux * t, p1.y + uy * t
        w = 0.0
        if ext is not None:
            try:
                ch = _cs_chords(sx, sy, px, py, ext, holes)
                if ch:
                    a, b = min(ch, key=lambda c: abs(0.5 * (c[0] + c[1])))
                    w = abs(b - a)
            except Exception:
                w = 0.0
        widths.append(float(w))
        width_points.append(Pt(sx, sy))
    return widths, width_points


def river_morphological_wide(river_geom, width_threshold_m):
    """★P7 v0.6.1 (A2)★ 形态学开运算提取河流宽段 (≥ width_threshold_m)。

    wide = open(river, r=T/2) ∩ river
         = river.buffer(-T/2).buffer(+T/2).intersection(river)   (T=width_threshold_m)

    腐蚀 buffer(-T/2) 抹去所有 < T 宽的河段 (含窄段/过渡段); 膨胀 buffer(+T/2) 把
    留下的"宽芯"长回来; 再 ∩river 裁回河岸 (⊆river)。完全沿河道、弯河/分叉天然正确,
    且**不依赖中心线**(摆脱圆盘法对中心线+圆盘半径 w/2 的过覆盖, 以及直长轴 R10 近似)。

    返回: 已 explode、去空 (area>0) 的 [Polygon]; river 为空 / 腐蚀后为空 → []。

    ⚠ 已知限制 (诚实标注, 区别于圆盘法的过覆盖):
      形态学开运算**各向同性** —— 一段河被判"宽", 需在二维上能容下半径 T/2 的圆盘,
      即沿流向与横向都 ≳ T。因此**很短但很宽的水域 (沿流向长度 < T)** 会被判为窄段。
      典型大江宽河段 (沿流向远长于 T) 分类正确; 若某工程含此类"短而宽"水域且需禁跨,
      应在调用侧另加"横向宽度 ≥ T"的 backstop (用 river_polygon_centerline 的逐站宽度补判)。
    """
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    if river_geom is None or getattr(river_geom, "is_empty", True):
        return []
    r = float(width_threshold_m) / 2.0
    if r <= 0:
        return []
    try:
        opened = river_geom.buffer(-r).buffer(r)
    except Exception:
        return []
    if opened is None or opened.is_empty:
        return []
    try:
        wide = opened.intersection(river_geom)
    except Exception:
        return []
    if wide is None or wide.is_empty:
        return []
    if isinstance(wide, Polygon):
        cand = [wide]
    elif isinstance(wide, (MultiPolygon, GeometryCollection)):
        cand = list(getattr(wide, "geoms", []))
    else:
        cand = []
    return [g for g in cand
            if isinstance(g, Polygon) and (not g.is_empty) and g.area > 0]


def river_split_by_width(river_geom, width_threshold_m):
    """★P7 v0.6.1 (A2)★ 形态学分宽窄, 返回 (wide_polys, narrow_polys)。

      wide   = river_morphological_wide(river, T)    (≥T → 进 forbidden 禁区, 禁跨+禁立)
      narrow = river − ∪wide                          (<T → 进 linear_cross, 可跨)

    二者构成 river 的划分 (wide ∪ narrow = river, 且均 ⊆ river, 无圆盘式过覆盖)。
    narrow_polys 已按连通块 explode, 供调用侧**逐窄段取局部中心线** (每块短、近直,
    最小外接矩形长轴近似很准 → R10 大幅削弱)。
    """
    from shapely.ops import unary_union
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    if river_geom is None or getattr(river_geom, "is_empty", True):
        return [], []
    wide = river_morphological_wide(river_geom, width_threshold_m)
    if wide:
        try:
            narrow_geom = river_geom.difference(unary_union(wide))
        except Exception:
            narrow_geom = river_geom
    else:
        narrow_geom = river_geom
    if narrow_geom is None or narrow_geom.is_empty:
        narrow = []
    elif isinstance(narrow_geom, Polygon):
        narrow = [narrow_geom]
    elif isinstance(narrow_geom, (MultiPolygon, GeometryCollection)):
        narrow = [g for g in getattr(narrow_geom, "geoms", [])
                  if isinstance(g, Polygon) and (not g.is_empty) and g.area > 0]
    else:
        narrow = []
    return wide, narrow


def evaluate_cross_cost(rule: dict, crossing_angle_deg: float = 90.0,
                        crossing_length_km: float = 0.0) -> float:
    """
    根据规则表计算跨越代价。
    - fixed: 固定值一次性代价
    - angle_formula: base + angle_coeff * cos(α)
    - length_formula: base + length_coeff * L
    - forbidden: 999999
    """
    cost_type = rule.get("cost_type", "fixed")
    cross_cost = rule.get("cross_cost", 0)
    formula = rule.get("cross_cost_formula")

    if cost_type == "forbidden":
        return 999999.0

    if cost_type == "angle_formula" and formula:
        base = formula.get("base", 0)
        angle_coeff = formula.get("angle_coeff", 0)
        alpha_rad = math.radians(max(0, min(90, crossing_angle_deg)))
        return base + angle_coeff * math.cos(alpha_rad)

    if cost_type == "length_formula" and formula:
        base = formula.get("base", 0)
        length_coeff = formula.get("length_coeff", 0)
        return base + length_coeff * crossing_length_km

    if isinstance(cross_cost, (int, float)):
        return float(cross_cost)

    if isinstance(cross_cost, str) and "禁止" in cross_cost:
        return 999999.0

    return 0.0


def evaluate_land_cost(rule: dict) -> float:
    """立塔代价（一次性固定值）。禁止立塔返回999999。"""
    land_cost = rule.get("land_cost", 0)
    if isinstance(land_cost, (int, float)):
        return float(land_cost)
    if isinstance(land_cost, str):
        if "禁止" in land_cost:
            return 999999.0
    return 0.0


def get_base_cross_cost(rule: dict) -> float:
    """
    获取跨越代价基础值（仅用于启发/预分类/粗近似，不用于最终代价累计）。
    """
    cost_type = rule.get("cost_type", "fixed")
    formula = rule.get("cross_cost_formula")
    if cost_type == "forbidden":
        return 999999.0
    if cost_type in ("angle_formula", "length_formula") and formula:
        return formula.get("base", 0)
    cross_cost = rule.get("cross_cost", 0)
    if isinstance(cross_cost, (int, float)):
        return float(cross_cost)
    if isinstance(cross_cost, str) and "禁止" in cross_cost:
        return 999999.0
    return 0.0


def is_one_time_cost(rule: dict) -> bool:
    """
    判断代价是否为一次性事件代价（不应铺进连续LPCF）。
    规则：未涉及L/α的固定值为一次性；公式型也是事件型（base可粗近似但不累加）。
    """
    cost_type = rule.get("cost_type", "fixed")
    if cost_type == "forbidden":
        return False  # 禁止区域在mask中体现
    if cost_type in ("angle_formula", "length_formula"):
        return True  # 公式型 = 事件型
    # fixed非零 = 一次性
    cross_cost = rule.get("cross_cost", 0)
    land_cost = rule.get("land_cost", 0)
    if isinstance(cross_cost, (int, float)) and cross_cost > 0:
        return True
    if isinstance(land_cost, (int, float)) and land_cost > 0:
        return True
    return False


# v0.2: 把 parallel_range_m 的默认值语义收到一处。
# 规则: 规则字段显式给了就用给的; 没给但有 parallel_reward(意味着可贴近奖励) 则默认 200m;
# 其他情况一律 0 (未配置时不做贴近检测)。
# 原来 M2._process_preferred_corridors 和 M3._compile_rule_config 两处有不同默认值,
# 会给下游算法端带来一致性隐患, 这里统一。
DEFAULT_PARALLEL_RANGE_M = 200


def parallel_range_m_for(rule: dict) -> int:
    """
    统一地为一个规则条目返回 parallel_range_m (米)。

    - 优先取规则文件里的显式值
    - 否则, 若配置了 parallel_reward 才落到 DEFAULT_PARALLEL_RANGE_M
    - 再否则 0 (= 不参与贴近检测)
    """
    v = rule.get("parallel_range_m")
    if isinstance(v, (int, float)) and v > 0:
        return int(v)
    if rule.get("parallel_reward"):
        return DEFAULT_PARALLEL_RANGE_M
    return 0


def _cost_num(v) -> float:
    """代价数值化: 数值原样转 float; 非数值 (如 "禁止"/"保护范围禁止"/None) → 0.0。
    ★P1★ 注意: 这里"禁止"故意映射为 0 (而非 evaluate_land_cost 的 999999 哨兵),
    因为"禁止"的语义由行为位 (cross_allow=false / is_landable=false) 表达,
    数值字段只承载"可通行/可立塔时的代价"。两者分工, 避免 999999 漏进代价累加。
    """
    return float(v) if isinstance(v, (int, float)) else 0.0


def decompose_cost_fields(rule: dict) -> dict:
    """★P1 (v0.6)★ 把一条规则的跨越/立塔代价拆成与算法端 FeatureType 同名的**数值**字段,
    供 M3 写进 rule_config.compiled_rules, 免得算法端解析 "30+30*cosα" 之类字符串。

    输出 4 个数值字段 (单位=万元, 已与算法端 features.py FeatureType 核对一致):
      - cross_cost:             跨越基础代价(万元)。fixed 型 = cross_cost(数值);
                                angle/length 型 = cross_cost_formula.base;
                                forbidden 型 cross_cost="禁止" → 0 (由 cross_allow=false 表达)。
                                ★命名与 FeatureType.cross_cost 完全一致 (对接零改名读取)。
      - cross_cost_angle_coeff: 角度项系数 (万元·×cosα)。仅 angle_formula 有, 否则 0。
      - cross_cost_per_km:      长度项系数 (万元/km, ×L)。仅 length_formula 有, 否则 0。
      - tower_cost:             立塔基础代价(万元) = land_cost(数值); "保护范围禁止" → 0
                                (由 is_landable=false 表达)。

    注: 这些字段名与算法端 FeatureType 同名 (cross_cost / cross_cost_angle_coeff /
        cross_cost_per_km / tower_cost), 故 rule_config 里规则表原始的人读 cross_cost 字符串
        ("30+30*cosα") 由 _compile_rule_config 改存到 cross_cost_expr (审计, 预处理内部用)。
    设计: 与 get_base_cross_cost/evaluate_land_cost 的 999999 哨兵口径**不同** —— 见 _cost_num。
    复用 (不重造): parallel_range_m_for / is_one_time_cost 仍由 _compile_rule_config 单独处理。
    """
    ct = rule.get("cost_type", "fixed")
    f = rule.get("cross_cost_formula") or {}
    if ct in ("angle_formula", "length_formula"):
        base = f.get("base", 0)
    else:
        # fixed 取数值 cross_cost; forbidden 的 cross_cost="禁止" 经 _cost_num → 0
        base = rule.get("cross_cost", 0)
    return {
        "cross_cost": _cost_num(base),
        "cross_cost_angle_coeff": _cost_num(f.get("angle_coeff", 0)),
        "cross_cost_per_km": _cost_num(f.get("length_coeff", 0)),
        "tower_cost": _cost_num(rule.get("land_cost", 0)),
    }


def compute_crossing_length_km(path_line, feature_geometry) -> float:
    """计算路径穿过地物区域的实际长度（km）。"""
    try:
        if path_line is None or feature_geometry is None:
            return 0.0
        if not path_line.intersects(feature_geometry):
            return 0.0
        intersection = path_line.intersection(feature_geometry)
        if intersection.is_empty:
            return 0.0
        return intersection.length / 1000.0
    except Exception:
        return 0.0


def covers_or_touches(geometry_union, point, buffer_tolerance_m=1.0):
    """
    v5.3: 替代 contains 的边界安全判定。
    使用 covers 或轻微 buffer 后判定，避免边界点被误判为在外部。
    """
    try:
        if geometry_union.covers(point):
            return True
        # 轻微buffer容差
        if buffer_tolerance_m > 0:
            return geometry_union.buffer(buffer_tolerance_m).covers(point)
        return False
    except Exception:
        return False


def batch_rasterize_max(geom_value_pairs, out_shape, transform, fill=0, dtype=np.float32):
    """
    v5.3: 批量栅格化（取最大值模式）。
    将多个 (geometry, value) 对一次性栅格化，避免逐个rasterize+np.maximum的循环。
    
    Args:
        geom_value_pairs: [(shapely_geometry, value), ...]
        out_shape: (height, width)
        transform: rasterio transform
        fill: 填充值
        dtype: 数据类型
    
    Returns:
        numpy array
    """
    from rasterio.features import rasterize
    from shapely.geometry import mapping

    if not geom_value_pairs:
        return np.full(out_shape, fill, dtype=dtype)

    # 按value分组，相同value的geometry一次rasterize
    from collections import defaultdict
    value_groups = defaultdict(list)
    for geom, val in geom_value_pairs:
        if geom and not geom.is_empty:
            value_groups[val].append(geom)

    result = np.full(out_shape, fill, dtype=dtype)
    for val, geoms in value_groups.items():
        shapes = [(mapping(g), val) for g in geoms]
        try:
            layer = rasterize(shapes, out_shape=out_shape,
                              transform=transform, fill=fill, dtype=dtype)
            result = np.maximum(result, layer)
        except Exception as e:
            # v0.3: 原先 silent pass, 改为 warning 便于排查"整批代价都没落下来"的问题
            logger.warning(f"batch_rasterize_max: 一批 {len(geoms)} 个几何 val={val} "
                           f"栅格化失败, 这批将被跳过: {e}")

    return result


def batch_rasterize_add(geom_value_pairs, out_shape, transform, fill=0, dtype=np.float32):
    """
    v5.3: 批量栅格化(累加模式). 用于贴近奖励等需要累加的场景.
    """
    from rasterio.features import rasterize
    from shapely.geometry import mapping

    if not geom_value_pairs:
        return np.full(out_shape, fill, dtype=dtype)

    result = np.full(out_shape, fill, dtype=dtype)
    # 逐个累加(无法避免,因为同一像素可能被多个geometry覆盖)
    fail_count = 0
    for geom, val in geom_value_pairs:
        if geom and not geom.is_empty:
            try:
                layer = rasterize([(mapping(geom), val)],
                                  out_shape=out_shape, transform=transform,
                                  fill=0, dtype=dtype)
                result += layer
            except Exception:
                fail_count += 1
    if fail_count:
        # v0.3: 汇总一次而非每次都打日志, 避免日志污染
        logger.warning(f"batch_rasterize_add: 共 {fail_count}/{len(geom_value_pairs)} "
                       f"个几何栅格化失败, 已跳过")
    return result


def resample_to_workspace(src_path: str,
                          dst_height: int, dst_width: int,
                          dst_transform, dst_crs,
                          method: str = "max",
                          fill=0, dtype=np.float32,
                          src_band: int = 1,
                          src_nodata=None):
    """
    v0.3: 把任意源栅格对齐重采样到工作区网格 (一次性解决两个bug)。

    这个函数同时修复:
      [bug A] Resampling.max/min/med/q1/q3/sum/rms 用在 DatasetReader.read() 上会抛
              "can be used for warp operations but not for reads and writes" ——
              因为 GDAL RasterIO 不支持这些聚合型重采样, 只能走 Warp 通道。
      [bug B] `ds.read(1, out_shape=(h,w))` 不做窗口、不做重投影, 只把整张源栅格
              按比例拉伸到目标形状。当源 bbox ≠ 工作区 bbox (真实工程 DEM 通常
              比线路走廊大得多), 所有像素的物理位置都会错位。

    用 rasterio.warp.reproject 同时处理:
      - 空间对齐: src_transform+src_crs → dst_transform+dst_crs (自动窗口+重投影)
      - 语义对齐: method 支持 max/min/average/nearest/mode/median

    Args:
        src_path: 源栅格路径
        dst_height, dst_width: 工作区网格尺寸
        dst_transform: 工作区 affine transform (from_bounds 算出)
        dst_crs: 工作区 CRS (rasterio.CRS 或 "EPSG:xxxx")
        method: "max"(聚合最大值,如窄谷/最坏坡度) /
                "min"(最严格限制,如风冰 max_turn) /
                "average" / "nearest" / "mode" / "median"
        fill: 目标区域超出源范围时的填充值
        dtype: 返回 numpy 数组 dtype
        src_band: 要读的源波段 (1-based)
        src_nodata: 可选覆盖源 nodata (默认用源文件声明的)

    Returns:
        np.ndarray, shape=(dst_height, dst_width), dtype=dtype

    Raises:
        抛出底层异常 (由调用方决定降级策略, 本函数不吞异常)。
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    method_map = {
        "max": Resampling.max,
        "min": Resampling.min,
        "average": Resampling.average,
        "nearest": Resampling.nearest,
        "mode": Resampling.mode,
        "median": Resampling.med,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    if method not in method_map:
        raise ValueError(f"resample_to_workspace 不支持 method={method}, "
                         f"可选: {list(method_map.keys())}")
    resampling = method_map[method]

    dst = np.full((dst_height, dst_width), fill, dtype=dtype)
    with rasterio.open(src_path) as src:
        nodata = src_nodata if src_nodata is not None else src.nodata
        reproject(
            source=rasterio.band(src, src_band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=fill,
            resampling=resampling,
        )
    return dst


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def get_config_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def write_gdf_to_gpkg_safe(gdf, path: str, layer_label: str = ""):
    """
    安全写 GeoDataFrame 到 GPKG, 自动处理 'FieldError: Error adding field X to layer' 类异常.
    
    ★修复 #17 (v0.5+)★ 真实甲方 GDB 数据中, 不同 layer 同名字段 (如 'name') 类型不一致
    (字符串/整数/nullable string 混合) → pd.concat 合并后 dtype=object →
    pyogrio 写 GPKG 时无法决定 OGR 字段类型 → FieldError.
    
    fallback 策略:
      1. 先按原方式尝试写 (对正常数据保持原性能, 不损失字段类型)
      2. FieldError 触发 → 把所有非几何/非元数据字段强制转字符串后重试
      3. 仍失败 → log error 跳过, 返回 False (调用方决定是否中断)
    
    Args:
        gdf: GeoDataFrame
        path: 输出 GPKG 路径
        layer_label: 用于日志的层名标识 (如 'unified_vectors', 'standardized_features')
    
    Returns:
        bool: True 成功 (含 fallback 成功), False 完全失败
    """
    label = layer_label or os.path.basename(path)
    try:
        gdf.to_file(path, driver="GPKG")
        return True
    except Exception as e:
        err_msg = str(e)[:120]
        logger.warning(
            f"写 {label}.gpkg 直接失败 ({type(e).__name__}: {err_msg}); "
            f"启用字段类型清洗 fallback (混合 schema 的常见原因)"
        )
    
    # Fallback: 所有非几何/非保留元数据字段强制转 str + 加 attr_ 前缀
    # ★Bug 3 修复 (v0.5+, Round 4 升级)★ 演进史:
    #   v0.5 初版: apply(lambda v: str(v)) — pandas dtype 仍是 object, pyogrio 仍报 FieldError
    #   Round 2: 加 astype('string') 让 StringDtype 明确 — 在大埔仍失败,
    #            因为 GDB 不同 layer 同名字段 (如 'name') concat 后 OGR 字段名记忆冲突
    #   Round 4: 重命名所有原始字段加 'attr_' 前缀 — 让 pyogrio 必须独立判定类型,
    #            不再因为字段名 'name' (OGR 保留字风险) 而拒绝.
    # 这样既保留了 GDB 原始信息 (name/编号/类型代码 等业务字段) 给 QGIS 审计用,
    # 又彻底回避了 FieldError, 不需要走"最小 schema 兜底"丢字段的悲催路径.
    try:
        cleaned = gdf.copy()
        # 1) 先收集需要重命名的字段 (排除 geometry / _xxx / std_xxx)
        rename_map = {}
        for col in list(cleaned.columns):
            if col == "geometry":
                continue
            if col.startswith("_") or col.startswith("std_") or col.startswith("attr_"):
                continue  # 元数据 / 已加前缀的不动
            rename_map[col] = f"attr_{col}"
        if rename_map:
            cleaned = cleaned.rename(columns=rename_map)
        # 2) 把所有 attr_ 字段强转 string (NaN/None → "")
        for col in list(cleaned.columns):
            if col == "geometry" or col.startswith("_") or col.startswith("std_"):
                continue
            try:
                cleaned[col] = (
                    cleaned[col]
                    .apply(lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v))
                    .astype("string")
                )
            except Exception as col_err:
                logger.warning(f"  字段 {col!r} 清洗失败, 整列丢弃: {col_err}")
                cleaned = cleaned.drop(columns=[col])
        # ★修复 (真实数据)★ GeoPackage 字段名大小写不敏感: 多源 SHP 合并后常出现
        #   仅大小写不同的同名列 (SHAPE_Area / Shape_Area / shape_Area), GPKG 视为重复 →
        #   FieldError → 掉到最小 schema 丢属性。此处给冲突列加后缀去重, 保住源属性供审计。
        seen_lower = {"geometry": 0}  # 预占 geometry, 防源属性名 'Geometry' 与几何列撞
        dedup_cols, n_dedup = [], 0
        for col in cleaned.columns:
            if col == "geometry":
                dedup_cols.append(col)
                continue
            low = col.lower()
            if low in seen_lower:
                seen_lower[low] += 1
                n_dedup += 1
                dedup_cols.append(f"{col}__{seen_lower[low]}")
            else:
                seen_lower[low] = 0
                dedup_cols.append(col)
        if n_dedup:
            cleaned.columns = dedup_cols
            logger.info(f"  {label}: 解决 {n_dedup} 处字段名大小写冲突 (加后缀去重, 源属性保留)")
        try:
            cleaned.to_file(path, driver="GPKG")
            logger.info(
                f"输出 {label}.gpkg (字段已加 attr_ 前缀并清洗为 str, "
                f"共 {len(rename_map)} 个原始字段保留): {path}"
            )
            return True
        except Exception as e2:
            # 第二次兜底: drop 所有非元数据非几何列, 保留最小可写 schema
            # (Round 4 后这条路径在大埔实际工程数据上应该极少触发, 留作防御)
            logger.warning(
                f"  attr_ 前缀+清洗后仍失败 ({type(e2).__name__}: {str(e2)[:120]}); "
                f"启用最小 schema 兜底 (仅保留几何 + 元数据列)"
            )
            try:
                keep_cols = [c for c in cleaned.columns
                             if c == "geometry" or c.startswith("_") or c.startswith("std_")]
                minimal = cleaned[keep_cols].copy()
                minimal.to_file(path, driver="GPKG")
                logger.info(f"输出 {label}.gpkg (最小 schema, 仅几何 + 元数据列): {path}")
                return True
            except Exception as e3:
                logger.error(
                    f"写 {label}.gpkg 仍失败, 跳过此文件 (审计层产物, 不阻塞下游): "
                    f"{type(e3).__name__}: {str(e3)[:200]}"
                )
                return False
    except Exception as outer:
        logger.error(f"写 {label}.gpkg fallback 阶段崩溃: {outer}")
        return False

