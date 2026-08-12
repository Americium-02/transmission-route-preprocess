"""
预处理输出加载器 (下游算法端使用)

M4/M5/M6 等算法模块以文件为输入来读取预处理产出。
但 M4PathPlanner.run() 历史上接受一个 `control_objects: dict[str, GeoDataFrame]`
作为函数参数, 而不是从磁盘读取。

本模块提供桥接辅助函数, 把 <output_dir>/m2/control_*.gpkg 文件还原成
算法端历史接口所需的 `control_objects` 字典, 避免算法端改动。

用法 (在算法端 run_algorithm.py 中):

    from utils.control_io import load_control_objects, validate_controls
    from utils.manifest import verify_manifest  # v0.2 新增
    from modules.m4_path_planning import M4PathPlanner

    # 算法端第一件事: 校验 manifest 再说
    vr = verify_manifest("./output")
    if not vr["ok"]:
        raise RuntimeError(f"预处理包不完整: {vr['mismatches']} / 缺: {vr['required_missing']}")

    control_objects = load_control_objects("./output")
    planner = M4PathPlanner("./output", project_config)
    result = planner.run(control_objects)

控制对象键约定 (与 M0/M2 一致):
    - start_end        : 起终点 (Point; type 字段区分 start/end)
    - must_pass        : 必经点 (Point)
    - must_path        : 必经路径 (LineString)
    - dense_corridor   : 密集通道 (Polygon)
    - accessible_area  : 可入区域 (Polygon)
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd

logger = logging.getLogger("transmission_planning.control_io")

# 预处理阶段支持的控制对象类型 (与 m0/m2 一致)
CONTROL_OBJECT_KEYS = [
    "start_end",
    "must_pass",
    "must_path",
    "dense_corridor",
    "accessible_area",
]

# 算法端一般要求至少这些存在
DEFAULT_REQUIRED_KEYS = ["start_end"]


def load_control_objects(output_or_m2_dir: str,
                         required_keys: Optional[list] = None
                         ) -> Dict[str, gpd.GeoDataFrame]:
    """
    从预处理输出目录中加载控制对象, 返回算法端 M4PathPlanner.run() 所需的 dict 结构。

    Args:
        output_or_m2_dir: 预处理输出根目录 (例如 "./output"), 或者直接传 m2 子目录
                         ("./output/m2"). 两种均可, 自动识别。
        required_keys: 可选, 只加载指定的控制对象键; 默认加载全部存在的。

    Returns:
        dict[str, GeoDataFrame]: 键为 CONTROL_OBJECT_KEYS 中的名字, 值为 GeoDataFrame。
                                 若某个键的文件不存在, 跳过 (不包含在返回字典中)。
    """
    root = Path(output_or_m2_dir)

    # 自动识别: 如果传的是 output_dir 而非 m2 子目录, 拼接 m2/
    if (root / "m2").exists() and not (root / "control_start_end.gpkg").exists():
        m2_dir = root / "m2"
    else:
        m2_dir = root

    if not m2_dir.exists():
        logger.warning(f"控制对象目录不存在: {m2_dir}")
        return {}

    keys = required_keys or CONTROL_OBJECT_KEYS
    control_objects: Dict[str, gpd.GeoDataFrame] = {}

    for key in keys:
        fpath = m2_dir / f"control_{key}.gpkg"
        if not fpath.exists():
            # M0 也会产出一份 control_{key}.gpkg (在 m0/ 目录下), 兜底查一次
            alt = m2_dir.parent / "m0" / f"control_{key}.gpkg"
            if alt.exists():
                fpath = alt
            else:
                logger.debug(f"控制对象 {key} 未提供 (无 {fpath})")
                continue
        try:
            gdf = gpd.read_file(str(fpath))
            control_objects[key] = gdf
            logger.info(f"  加载控制对象 {key}: {len(gdf)} 要素")
        except Exception as e:
            logger.warning(f"控制对象读取失败 {key}: {e}")

    return control_objects


def validate_controls(output_or_m2_dir: str,
                      required_keys: Optional[List[str]] = None) -> Dict:
    """
    *算法端调用点*: 在 load_control_objects 之前/之后都可调, 提前暴露错误。

    - 检查 required_keys 对应的控制对象文件是否存在且非空
    - 返回一个 {ok, missing, empty, details} 字典, 下游据此决定是否中止

    v0.2 新增, 目的: 避免算法端死在 control_objects["start_end"] 的 KeyError
    """
    root = Path(output_or_m2_dir)
    if (root / "m2").exists():
        m2_dir = root / "m2"
    else:
        m2_dir = root

    req = list(required_keys or DEFAULT_REQUIRED_KEYS)
    missing: List[str] = []
    empty: List[str] = []
    details: Dict[str, Dict] = {}

    for key in req:
        fpath = m2_dir / f"control_{key}.gpkg"
        alt = m2_dir.parent / "m0" / f"control_{key}.gpkg"
        real = fpath if fpath.exists() else (alt if alt.exists() else None)
        if real is None:
            missing.append(key)
            details[key] = {"exists": False}
            continue
        try:
            gdf = gpd.read_file(str(real))
            n = len(gdf)
            details[key] = {"exists": True, "path": str(real), "feature_count": n}
            if n == 0:
                empty.append(key)
        except Exception as e:
            missing.append(key)
            details[key] = {"exists": True, "path": str(real),
                            "read_error": str(e)}

    return {
        "ok": (not missing) and (not empty),
        "missing": missing,
        "empty": empty,
        "details": details,
    }


def list_available_control_objects(output_or_m2_dir: str) -> list:
    """仅返回已经存在的控制对象键列表, 不实际加载 GeoDataFrame。"""
    root = Path(output_or_m2_dir)
    if (root / "m2").exists():
        m2_dir = root / "m2"
    else:
        m2_dir = root
    available = []
    for key in CONTROL_OBJECT_KEYS:
        if (m2_dir / f"control_{key}.gpkg").exists():
            available.append(key)
    return available
