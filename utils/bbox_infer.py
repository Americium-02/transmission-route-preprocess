"""
工作区 bbox 推断共享工具 (v0.4.3 新增)

目的: 让 M2 的 _process_terrain 和 M3 的 _determine_bbox 共用同一套优先级,
避免两处各算一次导致行为漂移。

优先级 (与 M3._determine_bbox 保持一致):
  1) project.json.bbox          (用户显式最高)
  2) start_end + buffer          (端点真实表达)
  3) must_path / must_pass + buffer
  4) 矢量并集 (飞地过滤) + 5% 缓冲
  5) DEM bounds_in_working_crs + 5% 缓冲 (最后备选)
  6) None (彻底无信息)

与 M3 的差异:
  - 返回 None 时调用方自行决定兜底 (M3 会用 [0,0,50000,50000], M2 会回落到全量)
  - 无 "source" 字段, 改用 result.source 字符串透明说明来源
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("transmission_planning.bbox_infer")


def infer_work_bbox(
    project_config: dict,
    control_objects: Optional[Dict] = None,
    m2_geoms: Optional[dict] = None,
    raster_inventory: Optional[List[dict]] = None,
    default_start_end_buffer_km: float = 5.0,
    default_enclave_km: float = 150.0,
) -> Dict:
    """
    按统一优先级推断工程工作区 bbox (working_crs 下)。

    Args:
        project_config: project.json 解析后的 dict, 读取:
            - bbox, bbox_start_end_buffer_km, bbox_enclave_filter_km
        control_objects: dict[str, GeoDataFrame], 由 M0 读取
            - start_end / must_path / must_pass / ...
        m2_geoms: 可选, 若 M2 已有 forbidden_polygons/cost_polygons 等, 可参与矢量推断
            {
              "forbidden_polygons": [{"geometry": ..., ...}, ...],
              "no_tower_polygons": [...],
              "cost_polygons": [...],
              "linear_cross_segments": [...],
            }
        raster_inventory: 由 M0 读取的栅格清单

    Returns:
        {
          "bbox": (xmin, ymin, xmax, ymax) 或 None,
          "source": "project_config" | "start_end" | "must_path" | "vectors" | "dem" | None,
          "buffer_applied": "20km" | "5%" | ...,
          "warnings": [...],
        }
    """
    solver_p = project_config.get("solver_params", {}) or {}
    buf_km = float(solver_p.get("bbox_start_end_buffer_km", default_start_end_buffer_km))
    enclave_km = float(solver_p.get("bbox_enclave_filter_km", default_enclave_km))
    buf_m = buf_km * 1000.0

    warnings: List[str] = []

    # 1) project.json.bbox
    if "bbox" in project_config:
        b = project_config["bbox"]
        if b and len(b) == 4:
            return {
                "bbox": tuple(float(v) for v in b),
                "source": "project_config",
                "buffer_applied": "none",
                "warnings": warnings,
            }

    # 2/3) 控制对象
    control_objects = control_objects or {}
    se_bbox = _bbox_from_gdf(control_objects.get("start_end"))
    mp_bbox = _bbox_from_gdf(control_objects.get("must_path"))
    mpp_bbox = _bbox_from_gdf(control_objects.get("must_pass"))

    if se_bbox:
        merged = _merge([b for b in [se_bbox, mp_bbox, mpp_bbox] if b])
        return {
            "bbox": _expand(merged, buf_m, 0.0),
            "source": "start_end",
            "buffer_applied": f"{buf_km}km",
            "warnings": warnings,
        }
    if mp_bbox or mpp_bbox:
        merged = _merge([b for b in [mp_bbox, mpp_bbox] if b])
        return {
            "bbox": _expand(merged, buf_m, 0.0),
            "source": "must_path",
            "buffer_applied": f"{buf_km}km",
            "warnings": warnings,
        }

    # 4) 矢量并集 (飞地过滤 — 此时已无 start_end, 飞地过滤传 None, 不过滤)
    vec_bbox = _bbox_from_m2_geoms(m2_geoms, enclave_ref=None, enclave_km=enclave_km)
    if vec_bbox:
        return {
            "bbox": _expand(vec_bbox, 0.0, 0.05),
            "source": "vectors",
            "buffer_applied": "5%",
            "warnings": warnings,
        }

    # 5) DEM bounds_in_working_crs
    if raster_inventory:
        for r in raster_inventory:
            if r.get("inferred_type") != "DEM":
                continue
            b = r.get("bounds_in_working_crs") or r.get("bounds")
            if not b or len(b) != 4:
                continue
            src_note = "bounds_in_working_crs" if r.get("bounds_in_working_crs") else "bounds(原生CRS)"
            warnings.append(
                f"回退到 DEM {src_note}; 若 DEM 未裁剪到工程范围, 会拉大工作区"
            )
            return {
                "bbox": _expand(tuple(float(v) for v in b), 0.0, 0.05),
                "source": "dem",
                "buffer_applied": "5%",
                "warnings": warnings,
            }

    # 6) 无信息
    warnings.append("无任何可用数据推断工作区 bbox")
    return {"bbox": None, "source": None, "buffer_applied": None, "warnings": warnings}


# ─── 内部工具 ────────────────────────────────────────────────

def _bbox_from_gdf(gdf) -> Optional[Tuple[float, float, float, float]]:
    """从 GeoDataFrame 取 total_bounds"""
    if gdf is None:
        return None
    try:
        if len(gdf) == 0 or gdf.geometry.is_empty.all():
            return None
        b = gdf.total_bounds
        if any(not np.isfinite(v) for v in b):
            return None
        return tuple(float(v) for v in b)
    except Exception:
        return None


def _merge(bboxes: list) -> Tuple[float, float, float, float]:
    xmin = min(b[0] for b in bboxes)
    ymin = min(b[1] for b in bboxes)
    xmax = max(b[2] for b in bboxes)
    ymax = max(b[3] for b in bboxes)
    return (xmin, ymin, xmax, ymax)


def _expand(bbox, fixed_m: float, ratio: float) -> Tuple[float, float, float, float]:
    dx = (bbox[2] - bbox[0]) * ratio + fixed_m
    dy = (bbox[3] - bbox[1]) * ratio + fixed_m
    if dx < 100:
        dx = max(dx, 100.0)
    if dy < 100:
        dy = max(dy, 100.0)
    return (bbox[0] - dx, bbox[1] - dy, bbox[2] + dx, bbox[3] + dy)


def _bbox_from_m2_geoms(
    m2_geoms: Optional[dict],
    enclave_ref: Optional[Tuple[float, float, float, float]],
    enclave_km: float,
) -> Optional[Tuple[float, float, float, float]]:
    """从 M2 已产出的几何集合取总 bbox, 可选飞地过滤"""
    if not m2_geoms:
        return None
    geoms = []
    for key in ("forbidden_polygons", "no_tower_polygons", "cost_polygons"):
        for item in m2_geoms.get(key, []) or []:
            g = item.get("geometry") if isinstance(item, dict) else item
            if g is not None and not g.is_empty:
                geoms.append(g)
    for seg in m2_geoms.get("linear_cross_segments", []) or []:
        g = seg.get("geometry") if isinstance(seg, dict) else seg
        if g is not None and not g.is_empty:
            geoms.append(g)
    if not geoms:
        return None

    if enclave_ref is not None:
        cx = (enclave_ref[0] + enclave_ref[2]) / 2
        cy = (enclave_ref[1] + enclave_ref[3]) / 2
        r_m = enclave_km * 1000.0
        filtered = []
        for g in geoms:
            try:
                b = g.bounds
                gcx = (b[0] + b[2]) / 2
                gcy = (b[1] + b[3]) / 2
                if (gcx - cx) ** 2 + (gcy - cy) ** 2 <= r_m ** 2:
                    filtered.append(g)
            except Exception:
                filtered.append(g)
        if filtered:
            geoms = filtered

    try:
        from shapely.ops import unary_union
        combined = unary_union(geoms)
        return tuple(float(v) for v in combined.bounds)
    except Exception as e:
        logger.debug(f"矢量合并 bbox 失败: {e}")
        return None
