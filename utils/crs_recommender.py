"""
v0.4 新增: working_crs 推荐与校验工具

目的: 解决 preprocess_v0.3.1 问题 1 —— working_crs 用户手填, 填错时程序静默运行
      (例: 工程在东经 117° 却填 EPSG:4547 (CM 114°), 产生 >4/10000 尺度畸变)。

设计原则:
  - 不替换用户的显式选择 (保持 "配置显式" 哲学)
  - 只给建议 + 警告, 让用户自己判断
  - 支持多种数据源推断地理范围: start_end / must_path / 矢量 bbox / 栅格 bbox

国内工程常用投影决策树:
  - 经度跨度 > 3° → 3° 带 CGCS2000 可能畸变过大, 建议 6° 带或 UTM
  - 经度跨度 < 1.5° → CGCS2000 3° 带 (最精准, 推荐)
  - 中心经度靠近 3° 带中央子午线 (偏差 < 1.5°) → 对应 3° 带
  - 否则 → 就近 3° 带 + WARN

3° 带 CGCS2000 EPSG 编码(大陆常用):
  CM 75°  → 4534
  CM 78°  → 4535
  CM 81°  → 4536
  CM 84°  → 4537
  CM 87°  → 4538
  CM 90°  → 4539
  CM 93°  → 4540
  CM 96°  → 4541
  CM 99°  → 4542
  CM 102° → 4543
  CM 105° → 4544
  CM 108° → 4545
  CM 111° → 4546
  CM 114° → 4547  ★ 粤西/湘东/鄂东
  CM 117° → 4548  ★ 粤东/闽西
  CM 120° → 4549  ★ 闽东/浙东
  CM 123° → 4550
  CM 126° → 4551
  CM 129° → 4552
  CM 132° → 4553
  CM 135° → 4554
"""
import logging
import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger("transmission_planning.crs")


# CGCS2000 3° 带中央子午线 → EPSG
CGCS2000_3DEG_MAP = {
    75: 4534, 78: 4535, 81: 4536, 84: 4537, 87: 4538, 90: 4539,
    93: 4540, 96: 4541, 99: 4542, 102: 4543, 105: 4544, 108: 4545,
    111: 4546, 114: 4547, 117: 4548, 120: 4549, 123: 4550, 126: 4551,
    129: 4552, 132: 4553, 135: 4554,
}

# CGCS2000 6° 带中央子午线 → EPSG (常用 18-23 带, 对应 CM 105-135)
CGCS2000_6DEG_MAP = {
    75: 4491, 81: 4492, 87: 4493, 93: 4494, 99: 4495, 105: 4496,
    111: 4497, 117: 4498, 123: 4499, 129: 4500, 135: 4501,
}


def _nearest_cm_3deg(lon_center: float) -> int:
    """给定中心经度, 返回最近的 3° 带中央子午线 (75/78/81/.../135)"""
    # 3° 带中央子午线: 75, 78, 81, ..., 每 3° 一带
    idx = round((lon_center - 75) / 3)
    cm = 75 + idx * 3
    # 限制到大陆常用范围
    cm = max(75, min(135, cm))
    return cm


def _nearest_cm_6deg(lon_center: float) -> int:
    """给定中心经度, 返回最近的 6° 带中央子午线 (75/81/.../135)"""
    idx = round((lon_center - 75) / 6)
    cm = 75 + idx * 6
    cm = max(75, min(135, cm))
    return cm


def _crs_to_cm(crs_epsg: int) -> Optional[int]:
    """EPSG:45xx → 中央子午线。返回 None 表示不是 CGCS2000 3/6° 带。"""
    for cm, epsg in CGCS2000_3DEG_MAP.items():
        if epsg == crs_epsg:
            return cm
    for cm, epsg in CGCS2000_6DEG_MAP.items():
        if epsg == crs_epsg:
            return cm
    return None


def _parse_epsg(crs_str: str) -> Optional[int]:
    """从 'EPSG:4547' / '4547' / CRS 对象的字符串形式 提取 EPSG 代码"""
    if not crs_str:
        return None
    s = str(crs_str).strip()
    if s.upper().startswith("EPSG:"):
        s = s[5:]
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _estimate_scale_distortion(cm_declared: int, lon_center_actual: float) -> float:
    """
    粗估横轴墨卡托在偏离中央子午线 dlon 度处的尺度畸变。
    公式: k ≈ 1 + (dlon * cos_lat)^2 / 2, lon/lat 换算为弧度
    返回相对畸变值 (如 0.0004 表示 4/10000)
    """
    import math
    dlon_rad = math.radians(abs(lon_center_actual - cm_declared))
    # 取北纬 30° (中国中部) 作为 cos_lat 的近似
    cos_lat = math.cos(math.radians(30))
    k = (dlon_rad * cos_lat) ** 2 / 2.0
    return k


def recommend_working_crs(
    bbox_lonlat: Optional[Tuple[float, float, float, float]] = None,
    lon_center: Optional[float] = None,
) -> Dict[str, Any]:
    """
    基于项目地理范围推荐 working_crs。

    Args:
        bbox_lonlat: [lon_min, lat_min, lon_max, lat_max] 经纬度单位
        lon_center: 可选, 若已知中心经度可直接传入 (bbox_lonlat 优先)

    Returns:
        dict: {
          "best": {"epsg": 4547, "reason": "3° 带 CM 114°, 工程在带内",
                   "scale_distortion": 0.00012, "lon_span_deg": 0.8},
          "alternatives": [{"epsg": 4548, "reason": "...", ...}, ...],
          "warnings": [],  # 工程跨带等
          "lon_center": 114.2,
          "lon_span_deg": 0.8,
        }
    """
    result = {
        "best": None,
        "alternatives": [],
        "warnings": [],
        "lon_center": None,
        "lon_span_deg": None,
    }

    if bbox_lonlat is not None:
        lon_min, lat_min, lon_max, lat_max = bbox_lonlat
        lon_center_val = (lon_min + lon_max) / 2.0
        lon_span = lon_max - lon_min
        result["lon_center"] = lon_center_val
        result["lon_span_deg"] = lon_span
    elif lon_center is not None:
        lon_center_val = lon_center
        lon_span = 0.0
        result["lon_center"] = lon_center_val
        result["lon_span_deg"] = 0.0
    else:
        result["warnings"].append("无任何地理范围信息, 无法推荐 working_crs")
        return result

    # 1) 最近的 3° 带
    cm3 = _nearest_cm_3deg(lon_center_val)
    epsg3 = CGCS2000_3DEG_MAP.get(cm3)
    dist3 = _estimate_scale_distortion(cm3, lon_center_val)

    # 2) 最近的 6° 带
    cm6 = _nearest_cm_6deg(lon_center_val)
    epsg6 = CGCS2000_6DEG_MAP.get(cm6)
    dist6 = _estimate_scale_distortion(cm6, lon_center_val)

    # 3) 决策
    # 工程经度跨度 > 3° -> 3° 带会把两端其中一端拉出带外, 建议 6° 带
    if lon_span > 3.0:
        result["warnings"].append(
            f"工程经度跨度 {lon_span:.2f}° > 3°, 3° 带会在两端产生高畸变, "
            f"建议使用 6° 带"
        )
        if epsg6:
            result["best"] = {
                "epsg": epsg6,
                "crs": f"EPSG:{epsg6}",
                "reason": f"CGCS2000 6° 带 (CM {cm6}°), 工程跨度 {lon_span:.2f}° 适合 6° 带",
                "scale_distortion": round(dist6, 6),
                "lon_span_deg": lon_span,
            }
            if epsg3:
                result["alternatives"].append({
                    "epsg": epsg3,
                    "crs": f"EPSG:{epsg3}",
                    "reason": f"CGCS2000 3° 带 (CM {cm3}°, 备选, 但跨带畸变 {dist3*10000:.1f}/10000)",
                    "scale_distortion": round(dist3, 6),
                })
    else:
        # 工程经度跨度 <= 3° -> 3° 带为最佳
        if epsg3:
            result["best"] = {
                "epsg": epsg3,
                "crs": f"EPSG:{epsg3}",
                "reason": f"CGCS2000 3° 带 (CM {cm3}°), 工程跨度 {lon_span:.2f}°, 最大畸变 ≈ {dist3*10000:.2f}/10000",
                "scale_distortion": round(dist3, 6),
                "lon_span_deg": lon_span,
            }
        if epsg6:
            result["alternatives"].append({
                "epsg": epsg6,
                "crs": f"EPSG:{epsg6}",
                "reason": f"CGCS2000 6° 带 (CM {cm6}°, 精度略低但适合大跨度工程)",
                "scale_distortion": round(dist6, 6),
            })

    return result


def validate_working_crs(
    declared_crs: str,
    bbox_lonlat: Optional[Tuple[float, float, float, float]] = None,
    max_acceptable_distortion: float = 0.0005,
) -> Dict[str, Any]:
    """
    校验用户声明的 working_crs 是否适合工程范围。

    Args:
        declared_crs: 用户在 project.json 填的 working_crs (如 "EPSG:4547")
        bbox_lonlat: 工程范围 (经纬度)
        max_acceptable_distortion: 可接受的最大尺度畸变 (默认 5/10000)

    Returns:
        dict: {
          "ok": True/False,
          "severity": "info" / "warning" / "error",
          "message": "...",
          "declared_epsg": 4547,
          "declared_cm": 114,
          "actual_lon_center": 117.2,
          "estimated_distortion": 0.0008,
          "recommendation": {...}  # 来自 recommend_working_crs
        }
    """
    epsg = _parse_epsg(declared_crs)
    if epsg is None:
        return {
            "ok": False,
            "severity": "error",
            "message": f"无法解析 working_crs: '{declared_crs}'",
            "declared_epsg": None,
        }

    if bbox_lonlat is None:
        return {
            "ok": True,
            "severity": "info",
            "message": "无地理范围信息, 跳过 working_crs 合理性校验",
            "declared_epsg": epsg,
        }

    declared_cm = _crs_to_cm(epsg)
    recommendation = recommend_working_crs(bbox_lonlat)
    lon_center = recommendation.get("lon_center")

    result = {
        "ok": True,
        "severity": "info",
        "declared_epsg": epsg,
        "declared_crs": declared_crs,
        "declared_cm": declared_cm,
        "actual_lon_center": round(lon_center, 4) if lon_center is not None else None,
        "recommendation": recommendation,
    }

    if declared_cm is None:
        # 不是 CGCS2000 3/6° 带 (可能用了 WGS84/UTM 或其他), 不做强校验
        result["severity"] = "info"
        result["message"] = (
            f"声明的 CRS EPSG:{epsg} 不在 CGCS2000 3/6° 带白名单中, "
            f"跳过专项校验"
        )
        return result

    if lon_center is None:
        result["severity"] = "info"
        result["message"] = "未能确定工程中心经度, 跳过校验"
        return result

    distortion = _estimate_scale_distortion(declared_cm, lon_center)
    result["estimated_distortion"] = round(distortion, 6)

    if distortion > max_acceptable_distortion:
        # 畸变过大, 建议更换
        result["ok"] = False
        result["severity"] = "warning"
        best = recommendation.get("best") or {}
        result["message"] = (
            f"working_crs=EPSG:{epsg} 中央子午线 {declared_cm}° 与工程中心经度 "
            f"{lon_center:.2f}° 偏差 {abs(lon_center - declared_cm):.2f}°, "
            f"估计尺度畸变 {distortion*10000:.2f}/10000 超阈值 "
            f"({max_acceptable_distortion*10000:.2f}/10000)。"
            f"建议改用 {best.get('crs', 'N/A')}"
        )
    else:
        result["message"] = (
            f"working_crs=EPSG:{epsg} 与工程中心经度 {lon_center:.2f}° 匹配, "
            f"估计畸变 {distortion*10000:.2f}/10000 (可接受)"
        )

    return result


def infer_bbox_lonlat_from_project(
    control_objects: Optional[Dict] = None,
    raster_inventory: Optional[List[Dict]] = None,
    source_crs: str = "EPSG:4490",
    project_bbox: Optional[List[float]] = None,
    project_bbox_crs: Optional[str] = None,
    vector_bbox: Optional[List[float]] = None,
    vector_bbox_crs: Optional[str] = None,
    max_acceptable_lon_span_deg: float = 5.0,
) -> Optional[Tuple[float, float, float, float]]:
    """
    从可用数据源按优先级推断工程经纬度 bbox。

    优先级 (v0.4.2 调整):
      1) 起终点 (最可靠, 工程端点真实表达)
      2) project.json.bbox (用户显式)
      3) 矢量 (gdb/shp) 并集 bbox  ← v0.4.2 新增, 挤掉 DEM 的次席位
         附加"飞地夹逼": lon 跨度 > max_acceptable_lon_span_deg 时跳过
         (防止甲方数据含远离工程的飞地污染 CRS 推荐)
      4) DEM bounds (最后备选, 因为 DEM 常为整省/整市未裁剪, 易误导)

    为什么这个顺序和 `M3._determine_bbox` 的 "start_end > must_path > 矢量 > DEM"
    语义一致: 工程数据永远比 DEM 更代表工程实际地理位置。

    Args:
        control_objects: M0 读到的 control_objects (含 'start_end' 等)
        raster_inventory: M0 读到的 raster_inventory 列表
        source_crs: 无 CRS 时的兜底假设
        project_bbox: project.json 的 bbox 字段 (若有)
        project_bbox_crs: project.json bbox 使用的 CRS
        vector_bbox: 矢量 (gdb/shp) 并集 bbox, 已在 vector_bbox_crs 下
        vector_bbox_crs: 上述 bbox 的 CRS
        max_acceptable_lon_span_deg: 矢量 bbox 经度跨度夹逼 (飞地过滤)

    Returns:
        (lon_min, lat_min, lon_max, lat_max) 或 None
    """
    # ── 通用小工具: 把任意 CRS 下的 bbox 转成 EPSG:4490 经纬度 ──
    def _to_lonlat_bbox(bbox, crs):
        if not bbox or len(bbox) != 4:
            return None
        crs = crs or source_crs
        try:
            from pyproj import Transformer, CRS as PyCRS
            epsg = _parse_epsg(crs)
            if epsg in (4490, 4326):
                return tuple(float(v) for v in bbox)
            try:
                crs_obj = PyCRS.from_user_input(crs)
                if crs_obj.is_geographic:
                    return tuple(float(v) for v in bbox)
            except Exception:
                pass
            tf = Transformer.from_crs(crs, "EPSG:4490", always_xy=True)
            lon_min, lat_min = tf.transform(bbox[0], bbox[1])
            lon_max, lat_max = tf.transform(bbox[2], bbox[3])
            # 防御性: 经度若环绕到负值 (跨 180° 反子午线), 返回 None
            if not all(map(np.isfinite, [lon_min, lat_min, lon_max, lat_max])):
                return None
            if lon_min > lon_max or lat_min > lat_max:
                return None
            return (lon_min, lat_min, lon_max, lat_max)
        except Exception:
            return None

    # 1) 起终点 (最可靠)
    if control_objects and "start_end" in control_objects:
        try:
            gdf = control_objects["start_end"]
            if gdf is not None and len(gdf) > 0:
                # 转到 WGS84 / CGCS2000 地理坐标
                import geopandas as gpd
                g = gdf.copy()
                if g.crs is None:
                    g = g.set_crs(source_crs)
                # CGCS2000 地理坐标 = EPSG:4490, 等效于 WGS84 在中国地区
                g = g.to_crs("EPSG:4490")
                b = g.total_bounds  # [minx, miny, maxx, maxy] = [lon_min, lat_min, lon_max, lat_max]
                # 起终点往往只是两个点, 给一个小扩张避免 zero-area
                lon_span = max(0.01, b[2] - b[0])
                lat_span = max(0.01, b[3] - b[1])
                return (b[0] - lon_span * 0.1, b[1] - lat_span * 0.1,
                        b[2] + lon_span * 0.1, b[3] + lat_span * 0.1)
        except Exception as e:
            logger.debug(f"从起终点推断 bbox 失败: {e}")

    # 2) project.json 的 bbox
    if project_bbox and len(project_bbox) == 4:
        got = _to_lonlat_bbox(project_bbox, project_bbox_crs or source_crs)
        if got is not None:
            return got

    # 3) v0.4.2 新增: 矢量 (gdb/shp) 并集 bbox, 带飞地夹逼
    if vector_bbox and len(vector_bbox) == 4:
        got = _to_lonlat_bbox(vector_bbox, vector_bbox_crs or source_crs)
        if got is not None:
            lon_span = got[2] - got[0]
            if lon_span > max_acceptable_lon_span_deg:
                logger.warning(
                    f"矢量 bbox 经度跨度 {lon_span:.2f}° > "
                    f"{max_acceptable_lon_span_deg}° (疑似含飞地或数据超出工程范围), "
                    f"跳过该步骤, 降级到 DEM"
                )
            else:
                logger.info(
                    f"从矢量并集推断 bbox (lon 跨度 {lon_span:.2f}°)"
                )
                return got

    # 4) DEM bounds (最后备选)
    if raster_inventory:
        for r in raster_inventory:
            if r.get("inferred_type") == "DEM":
                bounds = r.get("bounds")
                crs = r.get("crs", source_crs)
                got = _to_lonlat_bbox(bounds, crs)
                if got is not None:
                    logger.info(
                        "从 DEM bounds 推断 bbox (兜底: 起终点/矢量/project.bbox 均缺失; "
                        "若 DEM 未裁剪到工程范围, 推荐的 working_crs 可能偏离工程真实位置)"
                    )
                    return got

    return None
