"""
下游读取辅助函数 (v0.4.4 新增)

目的: v0.4.3 把空图层 (forbidden_polygons/no_tower_polygons/cost_polygons) 改为
写占位 gpkg (带 _placeholder=True 字段和 Point(0,0) 几何), 避免 manifest 误报
required_missing。但这带来一个新的契约负担: 下游读取时必须过滤占位行, 否则
会把 Point(0,0) 当成真实地物。

本模块提供统一的读取函数, 让算法端和第三方工具有一致的 API。

约定:
  - 输入: gpkg 路径
  - 输出: (gdf, is_placeholder_only)
    - gdf:  已过滤掉 _placeholder=True 行的 GeoDataFrame (可能是 0 行)
    - is_placeholder_only: 若文件仅含占位行, 返回 True, 否则 False
                           下游可据此区分 "原本就空" vs "读取出问题"
"""
import logging
from typing import Tuple

logger = logging.getLogger("transmission_planning.io")


# v0.4.3 引入, 下游读取时必须过滤的占位标志字段
PLACEHOLDER_FIELD = "_placeholder"


def read_required_layer(path: str, crs_hint: str = None) -> Tuple["gpd.GeoDataFrame", bool]:
    """
    读取 M2 的 required 面图层, 自动过滤 v0.4.3 引入的占位行。

    典型用法 (算法端 M4+):
        from utils.io_helpers import read_required_layer
        forbidden_gdf, is_empty = read_required_layer("m2/forbidden_polygons.gpkg")
        if is_empty:
            logger.info("工程无禁区")
        # 直接用 forbidden_gdf, 已过滤好

    Args:
        path: gpkg 路径
        crs_hint: 若 gpkg 自身无 CRS, 用此值兜底 (可选)

    Returns:
        (filtered_gdf, is_placeholder_only)
          - filtered_gdf: 不含 _placeholder=True 的行; 若原文件仅含占位, 返回 0 行
          - is_placeholder_only: True 表示文件只有占位 (工程本来就无该类地物),
                                  False 表示含真实数据 (或完全为空 gpkg, 罕见)
    """
    import geopandas as gpd  # 延迟导入, 避免 import 时就要求 gpd

    gdf = gpd.read_file(path)

    if crs_hint and gdf.crs is None:
        gdf = gdf.set_crs(crs_hint)

    if PLACEHOLDER_FIELD in gdf.columns:
        placeholder_mask = gdf[PLACEHOLDER_FIELD].astype(bool) == True  # noqa: E712
        real_gdf = gdf[~placeholder_mask].copy()
        if PLACEHOLDER_FIELD in real_gdf.columns:
            real_gdf = real_gdf.drop(columns=[PLACEHOLDER_FIELD])
        is_placeholder_only = (len(real_gdf) == 0 and placeholder_mask.any())
        return real_gdf, is_placeholder_only

    # 无占位字段 = 正常满数据图层 (或全空, 但这种状态 manifest 不会接受)
    return gdf, False


def is_placeholder_layer(path: str) -> bool:
    """
    快速判断某个 gpkg 是否是占位图层 (不做数据过滤, 仅读少量数据)。

    场景: 某些性能敏感路径只需知道"要不要继续处理", 不需要真实 gdf。
    """
    try:
        import geopandas as gpd
        # 只读 schema, geopandas 无直接接口, 退而读全量 + head
        gdf = gpd.read_file(path, rows=5)
        if PLACEHOLDER_FIELD not in gdf.columns:
            return False
        # 占位文件仅 1 行, 全是 _placeholder=True
        if len(gdf) == 0:
            return False
        return bool(gdf[PLACEHOLDER_FIELD].astype(bool).all())
    except Exception as e:
        logger.debug(f"is_placeholder_layer({path}) 失败: {e}")
        return False
