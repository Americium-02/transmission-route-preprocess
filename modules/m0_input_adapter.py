"""
M0: 原始输入适配器
- 递归扫描项目目录，自动发现 .gdb / .shp / .tif
- 统一 CRS 到 working_crs
- 输出 gdb_inventory.json 和统一内部矢量集合
- 读取控制对象（起终点、必经点等）
- 读取 DEM 栅格元信息
"""
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import geopandas as gpd
import fiona
import rasterio
from pyproj import CRS

from utils.geo_utils import load_json, save_json, ensure_dir, get_config_dir

logger = logging.getLogger("transmission_planning.m0")

# ─── v0.5 新增: LEVEL2 别名映射 (E1 + Q4 决策) ─────────────────
# 用途: 把 GDB layer 名中的非规则表标准名 (词序倒置/全角括号/笔误/类别归并)
#       映射到 default_feature_rules.json 中的规范名,供规则查询使用。
# Option A 行为: 这里只做"规则查询用的别名规范化",原始 layer 基名仍保留在
#                _path_level2 字段;canonical 名只写入 _path_level2_canonical 字段。
LEVEL2_ALIAS = {
    # A 组: 道路类别归并 (2 条) ─────────────────────
    "乡道": "一般公路",
    "县道": "一般公路",
    # B 组: 输电线路词序倒置 (5 条) ─────────────
    "输电线路110kV": "110kV输电线路",
    "输电线路220kV": "220kV输电线路",
    "输电线路35kV":  "35kV输电线路",
    "输电线路500kV": "500kV输电线路",   # ★ 大埔工程实际 layer 名 (v0.5+ 补)
    "输电线路550kV": "500kV输电线路",   # ★ 笔误修正: 早期甲方数据曾写成 550kV, 实际是 500kV (防御性保留)
    # C 组: 字面/全角括号差异 (3 条) ─────────────
    "一般耕地":              "一般农田",
    "生态保护红线一般区域":  "生态保护红线（一般区域）",
    "弱电线路":              "弱电线路（I、II、III级）",
    # D 组: 类别归并 (2 条, 2026-05-11 用户确认) ─
    "其他型电站":     "变电站",            # ★ 归并到变电站规则参数
    "未定级输电线路": "110kV输电线路",     # ★ 按最低级别保守估计
    # E 组: 模糊类别归并 (2026-05-13 用户确认) ─
    # 大埔甲方 GDB 直接写 "铁路" (没有区分标准/窄轨/高铁), 规则表无此 catch-all 名,
    # 之前 503+4 个铁路要素走 M0 unresolved + M1 fuzzy 兜底, 不严谨.
    # 归并到 "普通铁路（标准轨距）" (rule 33: cross=30+30cosα, buffer=150m, 角度=45),
    # 这是中国铁路网默认形态, 最保守 (与窄轨参数相同; 高铁参数更严, 不主动猜).
    "铁路": "普通铁路（标准轨距）",
}

# ─── v0.5 新增: feature_dataset 名 → level1 映射 (E3 决策) ────
# 用途: 当 GDB 的 feature_dataset 探针成功时,把 dataset 名映射到规则表 level1 字段。
#       关键: GDB 里 dataset 名是"河流敏感点",规则表 level1 字段值是"河流",
#             这里做一次重命名对齐。
LEVEL1_DATASET_MAP = {
    "交通敏感点":                "交通敏感点",
    "河流敏感点":                "河流",                  # ★ 关键重命名
    "生态敏感点":                "生态敏感点",
    "电力设施敏感点":            "电力设施敏感点",
    "重要设施与政府规划敏感点":  "重要设施与政府规划敏感点",
    "管廊敏感点":                "管廊敏感点",
    "密集通道":                  "密集通道",
}

# ─── 路径关键词 → (level1, level2) 映射 ──────────────────
# 优先级：精确匹配 > 包含匹配。匹配的是 .gdb 所在的父级文件夹链中的每一级文件夹名。
# 例如路径  "生态敏感点/森林公园/xxx.gdb"  →  匹配到  ("生态敏感点","森林公园")
#
# 三区三线专用映射（文件夹名固定）
PATH_KEYWORD_TO_CATEGORY = {
    # ── 三区三线标准数据 ──────────────────────────────
    "永久基本农田":           ("生态敏感点", "基本农田"),
    "基本农田":               ("生态敏感点", "基本农田"),
    "城镇开发边界":           ("重要设施与政府规划敏感点", "城镇规划区"),
    "城镇开发":               ("重要设施与政府规划敏感点", "城镇规划区"),
    "生态保护红线陆域":       ("生态敏感点", "生态保护红线（一般区域）"),
    "生态保护红线":           ("生态敏感点", "生态保护红线（一般区域）"),
    "耕地保护目标":           ("生态敏感点", "一般农田"),
    # ── 生态敏感点 ────────────────────────────────────
    "森林公园":               ("生态敏感点", "森林公园"),
    "自然保护区核心区":       ("生态敏感点", "自然保护区核心区"),
    "自然保护区缓冲区":       ("生态敏感点", "自然保护区缓冲区"),
    "自然保护区试验区":       ("生态敏感点", "自然保护区试验区"),
    "一级水源保护区":         ("生态敏感点", "一级水源保护区"),
    "二级水源保护区":         ("生态敏感点", "二级水源保护区"),
    "水源准保护区":           ("生态敏感点", "水源准保护区"),
    "国有林场":               ("生态敏感点", "国有林场"),
    "一级林地":               ("生态敏感点", "一级林地"),
    "二级林地":               ("生态敏感点", "二级林地"),
    "一般农田":               ("生态敏感点", "一般农田"),
    "湿地公园":               ("生态敏感点", "湿地公园"),
    "地质公园":               ("生态敏感点", "地质公园"),
    "风景名胜区":             ("生态敏感点", "风景名胜区"),
    "海洋保护区":             ("生态敏感点", "海洋保护区"),
    # ── 重要设施与政府规划敏感点 ──────────────────────
    "城镇规划区":             ("重要设施与政府规划敏感点", "城镇规划区"),
    "禁止建设区":             ("重要设施与政府规划敏感点", "禁止建设区"),
    "建筑物":                 ("重要设施与政府规划敏感点", "建筑物"),
    "其他规划用地":           ("重要设施与政府规划敏感点", "其他规划用地"),
    "矿产资源":               ("重要设施与政府规划敏感点", "矿产资源"),
    "采石场":                 ("重要设施与政府规划敏感点", "采石场"),
    "军事敏感点":             ("重要设施与政府规划敏感点", "军事敏感点"),
    "军事设施":               ("重要设施与政府规划敏感点", "军事敏感点"),
    "无线电设施":             ("重要设施与政府规划敏感点", "无线电设施"),
    "导航台":                 ("重要设施与政府规划敏感点", "导航台"),
    "炸药库":                 ("重要设施与政府规划敏感点", "炸药库"),
    "油气存储站":             ("重要设施与政府规划敏感点", "油气存储站"),
    "地震地磁台":             ("重要设施与政府规划敏感点", "地震地磁台"),
    "气象站":                 ("重要设施与政府规划敏感点", "气象站"),
    # ── 交通敏感点 ────────────────────────────────────
    "机场":                   ("交通敏感点", "机场"),
    "普通铁路":               ("交通敏感点", "普通铁路（标准轨距）"),
    "铁路":                   ("交通敏感点", "普通铁路（标准轨距）"),
    "高铁":                   ("交通敏感点", "高铁"),
    "高速铁路":               ("交通敏感点", "高铁"),
    "高速公路":               ("交通敏感点", "高速公路"),
    "国道":                   ("交通敏感点", "国道"),
    "省道":                   ("交通敏感点", "省道"),
    "一般公路":               ("交通敏感点", "一般公路"),
    "电车道":                 ("交通敏感点", "电车道（有轨及无轨）"),
    # ── 电力设施敏感点 ────────────────────────────────
    "1000kV输电线路":         ("电力设施敏感点", "1000kV输电线路"),
    "±800kV输电线路":         ("电力设施敏感点", "±800kV输电线路"),
    "800kV输电线路":          ("电力设施敏感点", "±800kV输电线路"),
    "750kV输电线路":          ("电力设施敏感点", "750kV输电线路"),
    "500kV输电线路":          ("电力设施敏感点", "500kV输电线路"),
    "±500kV输电线路":         ("电力设施敏感点", "±500kV输电线路"),
    "400kV输电线路":          ("电力设施敏感点", "400kV输电线路"),
    "330kV输电线路":          ("电力设施敏感点", "330kV输电线路"),
    "220kV输电线路":          ("电力设施敏感点", "220kV输电线路"),
    "110kV输电线路":          ("电力设施敏感点", "110kV输电线路"),
    "66kV输电线路":           ("电力设施敏感点", "66kV输电线路"),
    "35kV输电线路":           ("电力设施敏感点", "35kV输电线路"),
    "接地极线路":             ("电力设施敏感点", "接地极线路"),
    "中低压输电线路":         ("电力设施敏感点", "中低压输电线路"),
    "变电站":                 ("电力设施敏感点", "变电站"),
    "换流站":                 ("电力设施敏感点", "换流站"),
    "接地极":                 ("电力设施敏感点", "接地极"),
    "弱电线路":               ("电力设施敏感点", "弱电线路（I、II、III级）"),
    "既有输电线路":           ("电力设施敏感点", "500kV输电线路"),  # 需后续字段确认电压
    "既有线路":               ("电力设施敏感点", "500kV输电线路"),
    # ── 管廊敏感点 ────────────────────────────────────
    "输油管道":               ("管廊敏感点", "输油管道"),
    "输气管道":               ("管廊敏感点", "输气管道"),
    "光纤管道":               ("管廊敏感点", "光纤管道"),
    # ── 河流 ──────────────────────────────────────────
    "通航河流":               ("河流", "通航河流"),
    "非通航河流":             ("河流", "非通航河流"),
    "河流":                   ("河流", "非通航河流"),
    "河流面":                 ("河流", "非通航河流"),
    # ── 一级类别名称（当文件夹只有一级没有二级时做兜底匹配）
    "生态敏感点":             ("生态敏感点", ""),
    "重要设施与政府规划敏感点": ("重要设施与政府规划敏感点", ""),
    "交通敏感点":             ("交通敏感点", ""),
    "电力设施敏感点":         ("电力设施敏感点", ""),
    "管廊敏感点":             ("管廊敏感点", ""),
    "地形":                   ("地形", ""),
    "密集通道":               ("密集通道", "密集通道"),
}

# 行政区编码正则
DISTRICT_CODE_RE = re.compile(r"(\d{6})")
DISTRICT_NAME_RE = re.compile(r"\d{6}([\u4e00-\u9fa5]+[区县市])")


class M0InputAdapter:
    """M0 原始输入适配器"""

    def __init__(self, project_dir: str, project_config: dict, output_dir: str):
        """
        Args:
            project_dir: 项目数据根目录
            project_config: project.json 内容
            output_dir: 预处理输出目录
        """
        self.project_dir = Path(project_dir)
        self.config = project_config
        self.output_dir = Path(output_dir)
        # ★v0.6 新增★ working_crs 支持 "auto": 缺省/空/显式 "auto" → 据数据地理范围自动选 3°/6° 带
        #   (在 _validate_working_crs_against_data 内解析后回写 project_config, 供 M1/M2/M3/manifest)
        _wc = project_config.get("working_crs")
        self.working_crs = _wc if (_wc and str(_wc).strip().lower() != "auto") else "auto"
        self.source_crs = project_config.get("source_crs", "EPSG:4490")

        # 输出集合
        self.gdb_inventory: List[dict] = []
        self.all_vector_layers: List[gpd.GeoDataFrame] = []
        self.raster_inventory: List[dict] = []
        self.control_objects: Dict[str, Any] = {}
        self.read_log: List[dict] = []
        # v0.5 新增: 规则表 level2 → level1 索引 (按需懒加载)
        self._rule_l2_to_l1_cache: Optional[Dict[str, str]] = None
        # v0.5 新增: layer 解析结果 (variant_inventory.json 来源)
        # 结构: {base_level2: {primary: int, protection: int, reward: int, alias_from: [orig_names]}}
        self.variant_inventory: Dict[str, Dict[str, Any]] = {}
        # v0.5 新增: 解析路径分布统计
        self.resolution_path_distribution: Dict[str, int] = {
            "rule_table_lookup": 0,
            "feature_dataset_probe": 0,
            "path_inference": 0,
            "unresolved": 0,
        }
        # v0.4: 诊断信息收集器, 写入 read_log 和 m0 产出的 crs_diagnostic.json
        self.crs_diagnostic: Dict[str, Any] = {
            "working_crs_validation": None,   # 来自 validate_working_crs
            "working_crs_axis_info": None,    # CRS 轴顺序提示
            "start_end_coord_check": None,    # 起终点坐标合理性
            "vector_bbox_in_source_crs": None,  # v0.4.2: 矢量并集 bbox (为 CRS 推荐服务)
        }

    def run(self) -> dict:
        """执行M0全部流程"""
        logger.info("===== M0: 原始输入适配器启动 =====")

        # v0.4: 启动期打印 working_crs 轴序信息 (问题 9 防护)
        self._log_working_crs_axis_info()

        # 1. 扫描 GDB
        self._scan_gdb_files()

        # 2. 扫描 SHP
        self._scan_shp_files()

        # 3. 扫描栅格 (DEM/DSM/风区覆冰等)
        self._scan_raster_files()

        # 4. 读取控制对象
        self._read_control_objects()

        # v0.4: 起终点坐标合理性校验 + working_crs 合理性校验
        # (此时 control_objects / raster_inventory 均已就绪, 可推断工程范围)
        self._validate_working_crs_against_data()

        # 5. 统一CRS并合并
        unified_gdf = self._unify_and_merge()

        # 6. 输出
        result = self._export(unified_gdf)

        logger.info(f"M0完成: {len(self.all_vector_layers)}个图层, "
                    f"{len(self.gdb_inventory)}个GDB, "
                    f"{len(self.raster_inventory)}个栅格")
        return result

    # ─── v0.4: CRS 诊断 ────────────────────────────────────
    def _log_working_crs_axis_info(self):
        """
        问题 9 防护: 启动时打印 working_crs 的轴序信息,
        让用户自检 start/end 点坐标是否按预期顺序填写。
        """
        if str(self.working_crs).strip().lower() == "auto":
            logger.info("working_crs=auto: 将在读入数据后据工程地理范围自动选择 (3°/6° 带)")
            self.crs_diagnostic["working_crs_axis_info"] = {"pending_auto_select": True}
            return
        try:
            crs_obj = CRS(self.working_crs)
            axis_info = crs_obj.axis_info
            is_geographic = crs_obj.is_geographic
            axis_order_desc = " → ".join(
                f"{a.abbrev}({a.direction})" for a in axis_info
            ) if axis_info else "未知"
            axis_dict = {
                "working_crs": str(self.working_crs),
                "is_geographic": is_geographic,
                "is_projected": crs_obj.is_projected,
                "axis_order": axis_order_desc,
                "unit": axis_info[0].unit_name if axis_info else None,
                "name": crs_obj.name,
            }
            self.crs_diagnostic["working_crs_axis_info"] = axis_dict
            logger.info(f"working_crs={self.working_crs} ({crs_obj.name}), "
                        f"轴序={axis_order_desc}, 单位={axis_dict['unit']}")
            if is_geographic:
                logger.warning(
                    f"working_crs={self.working_crs} 是地理坐标系 (度单位)。"
                    f"输电工程推荐用投影坐标系 (米单位), 否则分辨率/面积/距离计算都以度为单位。"
                )
        except Exception as e:
            logger.warning(f"无法解析 working_crs 轴序信息: {e}")
            self.crs_diagnostic["working_crs_axis_info"] = {"error": str(e)}

    def _validate_working_crs_against_data(self):
        """
        问题 1 落地: 基于已读入的数据 (起终点/DEM bounds), 反推工程经纬度范围,
        与声明的 working_crs 比较, 不匹配时 WARN (不强制覆盖)。

        v0.4.2 (Q3): 把 gdb/shp 矢量并集 bbox 也纳入推荐链, 挤掉 DEM 的次席
        (DEM 常为省级未裁剪, 容易把 CRS 推到错误的中央子午线)。

        同时校验起终点坐标合理性 (问题 9): 若投影 CRS 下坐标看起来像 lat/lon,
        或地理 CRS 下坐标超出 [-180,180]×[-90,90], 都发出警告。
        """
        from utils.crs_recommender import (
            validate_working_crs,
            infer_bbox_lonlat_from_project,
        )

        # 1) 起终点坐标合理性 (问题 9)
        start_end_check = self._check_start_end_coords()
        self.crs_diagnostic["start_end_coord_check"] = start_end_check
        if start_end_check and start_end_check.get("severity") == "warning":
            logger.warning(f"起终点坐标校验: {start_end_check['message']}")

        # 2) v0.4.2: 把 gdb/shp 矢量并集 bbox 算好 (在 source_crs 下), 传给推荐器
        vector_bbox_src = self._compute_vector_bbox_in_source_crs()
        self.crs_diagnostic["vector_bbox_in_source_crs"] = (
            list(vector_bbox_src) if vector_bbox_src else None
        )

        # 3) 推断工程经纬度 bbox (走 start_end → project.bbox → 矢量 → DEM 链)
        bbox_lonlat = infer_bbox_lonlat_from_project(
            control_objects=self.control_objects,
            raster_inventory=self.raster_inventory,
            source_crs=self.source_crs,
            project_bbox=self.config.get("bbox"),
            project_bbox_crs=self.config.get("bbox_crs"),
            vector_bbox=vector_bbox_src,
            vector_bbox_crs=self.source_crs,
        )

        # ★v0.6 新增★ working_crs=auto: 据工程地理范围自动选最优带, 回写 config 供下游
        if str(self.working_crs).strip().lower() == "auto":
            self._auto_select_working_crs(bbox_lonlat)

        # 4) 调 validator (auto 选完后校验, 理应通过/低畸变)
        validation = validate_working_crs(self.working_crs, bbox_lonlat=bbox_lonlat)
        self.crs_diagnostic["working_crs_validation"] = validation

        msg = validation.get("message", "")
        sev = validation.get("severity", "info")
        if sev == "warning":
            logger.warning(f"working_crs 合理性校验: {msg}")
        elif sev == "error":
            logger.error(f"working_crs 合理性校验: {msg}")
        else:
            logger.info(f"working_crs 合理性校验: {msg}")

    def _auto_select_working_crs(self, bbox_lonlat):
        """★v0.6★ working_crs=auto: 用 recommend_working_crs 据工程经纬度范围自动选
        最优 CGCS2000 3°/6° 带, 设 self.working_crs 并回写 self.config["working_crs"]
        (供 M1/M2/M3/manifest 一致使用)。无法推断时回退 EPSG:4547 并 WARN。
        """
        from utils.crs_recommender import recommend_working_crs
        fallback = "EPSG:4547"
        if not bbox_lonlat:
            self.working_crs = fallback
            self.config["working_crs"] = fallback
            logger.warning(
                f"working_crs=auto 但无法从数据推断工程地理范围 (缺起终点/矢量/DEM), "
                f"回退 {fallback}; 建议在 project.json 显式指定 working_crs")
            return
        rec = recommend_working_crs(bbox_lonlat=bbox_lonlat)
        best = rec.get("best") or {}
        chosen = best.get("crs")
        if not chosen:
            self.working_crs = fallback
            self.config["working_crs"] = fallback
            logger.warning(
                f"working_crs=auto 推断失败 (lon_center={rec.get('lon_center')}), 回退 {fallback}")
            return
        self.working_crs = chosen
        self.config["working_crs"] = chosen  # ★回写★ 同一 project_config dict → M1/M2/M3/manifest 生效
        self.crs_diagnostic["working_crs_auto_selected"] = {
            "chosen": chosen, "lon_center": rec.get("lon_center"),
            "lon_span_deg": rec.get("lon_span_deg"), "reason": best.get("reason"),
            "scale_distortion": best.get("scale_distortion"),
        }
        logger.info(
            f"working_crs=auto → 自动选择 {chosen} "
            f"(工程中心经度≈{rec.get('lon_center'):.2f}°): {best.get('reason')}")
        for w in rec.get("warnings", []):
            logger.warning(f"working_crs 自动选择: {w}")
        # 重新打印轴序信息 (此时是真实 CRS, 之前 auto 时跳过了)
        self._log_working_crs_axis_info()

    def _compute_vector_bbox_in_source_crs(self) -> Optional[Tuple[float, float, float, float]]:
        """
        v0.4.2 Q3: 从 all_vector_layers 算矢量并集 bbox, 统一到 source_crs。
        用于 infer_bbox_lonlat_from_project 把矢量作为 CRS 推荐的二优先源
        (次于起终点/project.bbox, 但优于常被整省污染的 DEM)。

        此时 self.all_vector_layers 还没走 _unify_and_merge, 每层可能在
        各自原生 CRS 下, 需要逐层转到 source_crs 再合并 bbox。

        Returns:
            (xmin, ymin, xmax, ymax) in source_crs, 或 None 若无矢量/全部失败
        """
        if not self.all_vector_layers:
            return None
        try:
            source = CRS(self.source_crs)
        except Exception:
            return None

        import numpy as np
        minx = miny = maxx = maxy = None
        success_layer_count = 0
        for gdf in self.all_vector_layers:
            try:
                if gdf is None or len(gdf) == 0:
                    continue
                g = gdf
                # 缺 CRS 的按 source_crs 假设 (与 _unify_and_merge 语义一致)
                if g.crs is None:
                    g = g.set_crs(self.source_crs)
                # 转到 source_crs (常是 EPSG:4490 经纬度) 以便合并 bbox
                try:
                    if g.crs != source:
                        g = g.to_crs(source)
                except Exception:
                    continue
                b = g.total_bounds
                if any(not np.isfinite(v) for v in b):
                    continue
                if minx is None:
                    minx, miny, maxx, maxy = b
                else:
                    minx = min(minx, b[0])
                    miny = min(miny, b[1])
                    maxx = max(maxx, b[2])
                    maxy = max(maxy, b[3])
                success_layer_count += 1
            except Exception:
                continue

        if minx is None or success_layer_count == 0:
            return None
        logger.debug(
            f"矢量并集 bbox (source_crs={self.source_crs}, 参与层数 {success_layer_count}): "
            f"({minx}, {miny}, {maxx}, {maxy})"
        )
        return (float(minx), float(miny), float(maxx), float(maxy))

    def _check_start_end_coords(self) -> Optional[Dict[str, Any]]:
        """
        问题 9: 校验起终点坐标是否与声明 CRS 匹配。
        规则:
          - 声明 CRS 是地理坐标 -> 坐标应在 [-180,180]×[-90,90]
          - 声明 CRS 是投影坐标 -> 坐标通常在百万级 (米单位, 东偏+500000 base)
          - 若投影 CRS 下坐标 abs < 200, 很可能是填了经纬度但声明了投影, 轴序搞错
        """
        se_source = None
        coords = []
        # v0.4.3 审核问题 5: 记录坐标实际来源的 CRS, 用于后续校验
        # 如果起终点来自 project.json, 按 source_crs 校验 (保持旧行为);
        # 如果起终点来自 control 文件, 优先用 gdf.crs, 兜底 source_crs
        effective_crs = self.source_crs

        # 优先从 project.json 读 (最可能手填, 最容易错)
        sp = self.config.get("start_point")
        ep = self.config.get("end_point")
        if sp and ep and len(sp) >= 2 and len(ep) >= 2:
            coords = [(sp[0], sp[1]), (ep[0], ep[1])]
            se_source = "project.json.start_point/end_point"
            effective_crs = self.source_crs  # project.json 的坐标按 source_crs 解释

        # 否则从 control_objects 读
        if not coords and "start_end" in self.control_objects:
            try:
                gdf = self.control_objects["start_end"]
                if len(gdf) > 0:
                    for geom in gdf.geometry:
                        if geom is not None and not geom.is_empty:
                            coords.append((geom.x, geom.y))
                    se_source = "control/start_end.geojson"
                    # 优先用 gdf 自身声明的 CRS (用户可能存成 EPSG:4548 米制),
                    # 兜底才用 source_crs
                    if gdf.crs is not None:
                        effective_crs = str(gdf.crs)
                    else:
                        effective_crs = self.source_crs
            except Exception:
                pass

        if not coords:
            return None

        try:
            crs_obj = CRS(effective_crs)
            is_geographic = crs_obj.is_geographic
        except Exception:
            return None

        issues = []
        for i, (x, y) in enumerate(coords):
            if is_geographic:
                # 地理坐标, 应 [-180,180]×[-90,90]
                if not (-180 <= x <= 180 and -90 <= y <= 90):
                    issues.append(
                        f"点 #{i}=[{x:.4f},{y:.4f}] 超出地理坐标合理范围 "
                        f"([-180,180]×[-90,90], CRS={effective_crs})"
                    )
                # 疑似 [lat, lon] 颠倒: y 超过 90 但 x 在 ±90 之内
                if abs(y) > 90 and abs(x) <= 90:
                    issues.append(
                        f"点 #{i}=[{x:.4f},{y:.4f}] 疑似 [lat,lon] 顺序, "
                        f"geojson/shapely 约定 [lon,lat]"
                    )
            else:
                # 投影坐标, 中国境内 CGCS2000 3° 带大致 400000-900000 (E) / 2000000-5000000 (N)
                if abs(x) < 200 and abs(y) < 200:
                    issues.append(
                        f"点 #{i}=[{x:.4f},{y:.4f}] 在投影 CRS {effective_crs} 下"
                        f"坐标值过小, 疑似填了经纬度"
                    )
                # 过大也可疑
                if abs(x) > 1e8 or abs(y) > 1e8:
                    issues.append(
                        f"点 #{i}=[{x:.4f},{y:.4f}] 坐标值过大"
                    )

        return {
            "source": se_source,
            "coords": coords,
            "declared_source_crs": self.source_crs,
            "effective_crs": effective_crs,  # v0.4.3: 实际用于校验的 CRS
            "is_geographic": is_geographic,
            "severity": "warning" if issues else "info",
            "issues": issues,
            "message": (
                "; ".join(issues) if issues
                else f"起终点坐标范围合理 ({se_source}, CRS={effective_crs})"
            ),
        }

    # ─── GDB 扫描 ───────────────────────────────────────────
    def _scan_gdb_files(self):
        """递归扫描所有 .gdb 目录"""
        logger.info("扫描GDB文件...")
        gdb_dirs = []
        for root, dirs, files in os.walk(self.project_dir):
            for d in dirs:
                if d.endswith(".gdb"):
                    gdb_dirs.append(os.path.join(root, d))

        for gdb_path in gdb_dirs:
            self._read_single_gdb(gdb_path)

        logger.info(f"发现 {len(self.gdb_inventory)} 个GDB文件")

    # ─── v0.5 新增: GDB layer 三级解析链 ─────────────────────
    def _get_rule_l2_to_l1_index(self) -> Dict[str, str]:
        """懒加载 default_feature_rules.json 并构建 {level2: level1} 索引.
        
        与 M1 / M2 / M3 共享同一份 config 文件, 但 M0 此处只需要 (level2, level1)
        反查, 不需要完整规则参数, 故只构建轻量索引.
        """
        if self._rule_l2_to_l1_cache is not None:
            return self._rule_l2_to_l1_cache
        try:
            rules_path = os.path.join(get_config_dir(), "default_feature_rules.json")
            rules_data = load_json(rules_path)
            # 兼容两种结构: {features: [...]} 或直接 [...]
            features = rules_data.get("features", rules_data) if isinstance(rules_data, dict) else rules_data
            index = {}
            for feat in features:
                l2 = feat.get("level2")
                l1 = feat.get("level1")
                if l2 and l1:
                    index[l2] = l1
            self._rule_l2_to_l1_cache = index
            logger.debug(f"规则表 l2→l1 索引构建完成: {len(index)} 条")
        except Exception as e:
            logger.warning(f"加载 default_feature_rules.json 失败: {e}; l2→l1 索引为空")
            self._rule_l2_to_l1_cache = {}
        return self._rule_l2_to_l1_cache

    def _probe_feature_dataset(self, gdb_path: str, layer_name: str) -> Optional[str]:
        """L2 探针: 尝试获取 layer 所属的 feature_dataset 名.
        
        GDAL 3.6+ 支持; 失败时静默返回 None, 不影响主流程 (由 L3 兜底).
        实现策略: 优先尝试 osgeo.ogr (更稳定), 失败 fallback fiona.
        """
        # 策略 1: osgeo.ogr (GDAL 3.6+ 有 GetParentDataset / metadata 查询)
        try:
            from osgeo import ogr  # type: ignore
            ds = ogr.Open(gdb_path)
            if ds is None:
                return None
            try:
                layer = ds.GetLayerByName(layer_name)
                if layer is None:
                    return None
                # 部分 GDAL 版本通过 metadata 暴露 FEATURE_DATASET
                md = layer.GetMetadata_Dict() if hasattr(layer, "GetMetadata_Dict") else {}
                fds = md.get("FEATURE_DATASET") or md.get("feature_dataset")
                if fds:
                    return str(fds)
            finally:
                ds = None
        except Exception:
            pass
        # 策略 2: fiona (部分版本 schema 含 feature_dataset)
        try:
            with fiona.open(gdb_path, layer=layer_name) as src:
                meta = getattr(src, "meta", {}) or {}
                fds = meta.get("feature_dataset") or (meta.get("schema", {}) or {}).get("feature_dataset")
                if fds:
                    return str(fds)
        except Exception:
            pass
        return None

    def _update_variant_inventory(self, base_l2: str, canonical_l2: str, variant: str):
        """更新 variant_inventory 计数和别名反向链.
        
        GDB 与 SHP 两个通道共用此方法, 保证行为一致.
        
        ★Bug 4 修复 (v0.5+)★ alias_from 统一为 [] (空数组).
        之前老版本对"有别名归并的 base 条目"用 alias_from=None 标记,
        但 JSON schema 校验和前端展示困扰. 现在统一: alias_from 永远是 list,
        语义是"哪些其他名字会归并到本条目";base 条目自然为 [] (因为 base 自己不收别名).
        canonical 字段已经标了规范名, alias_from=[] 不会丢失语义.
        
        Args:
            base_l2: 原始基名 (如 "其他型电站")
            canonical_l2: 规范化后名 (如 "变电站"; 若无别名归并则等于 base_l2)
            variant: "primary" / "protection" / "reward"
        """
        # base_l2 条目: 累加 variant 计数
        if base_l2 not in self.variant_inventory:
            self.variant_inventory[base_l2] = {
                "primary": 0, "protection": 0, "reward": 0,
                "canonical": canonical_l2,
                "alias_from": [],   # ★Bug 4: 统一为 [] (任何条目都是 list, 不再用 None)
            }
        self.variant_inventory[base_l2][variant] += 1
        
        # 别名归并情况: 同步在 canonical 名下记一份反向链
        if canonical_l2 != base_l2:
            if canonical_l2 not in self.variant_inventory:
                self.variant_inventory[canonical_l2] = {
                    "primary": 0, "protection": 0, "reward": 0,
                    "canonical": canonical_l2,
                    "alias_from": [],
                }
            # ★Bug 4★ canonical 条目的 alias_from 可能在创建时为 [] (无问题), 直接 append
            if base_l2 not in self.variant_inventory[canonical_l2]["alias_from"]:
                self.variant_inventory[canonical_l2]["alias_from"].append(base_l2)

    def _parse_gdb_layer_structure(self, gdb_path: str) -> List[Dict[str, Any]]:
        """v0.5 三级解析链: 把每个 layer 解析为 (level1, base_level2, canonical_level2, variant, resolution_path).
        
        优先级:
          L1: layer 后缀正则 + 规则表反查 (首选, GDAL 版本无关, 最鲁棒)
          L2: feature_dataset 元数据探针 (GDAL 3.6+ 可用, 失败回退)
          L3: 父级文件夹路径推断 (v0.4.6 旧逻辑兜底)
        
        Option A 行为: base_level2 保留 layer 原始基名 (剥变体后缀);
                       canonical_level2 经 LEVEL2_ALIAS 规范化, 供后续查规则.
        
        Returns:
            List[Dict] 每个 layer 一条, 含 layer/level1/base_level2/canonical_level2/variant/resolution_path
        """
        try:
            layer_names = fiona.listlayers(gdb_path)
        except Exception as e:
            logger.warning(f"无法列出 GDB layers: {gdb_path}, 错误: {e}")
            return []
        
        rule_l2_to_l1 = self._get_rule_l2_to_l1_index()
        # 变体后缀 → variant 类型映射
        variant_suffix_map = {
            None:        "primary",
            "_保护范围": "protection",
            "_缓冲范围": "protection",   # 同义归并 (防御性保留)
            "_河道范围": "protection",   # ★大埔工程实际数据★ 通航河流的保护范围 layer 用此后缀
            "_奖励范围": "reward",
        }
        # 变体后缀正则: 用非贪婪匹配, 后缀可选
        variant_re = re.compile(r"^(.+?)(_保护范围|_奖励范围|_缓冲范围|_河道范围)?$")
        
        results = []
        for layer_name in layer_names:
            m = variant_re.match(layer_name)
            base_l2 = m.group(1)
            variant_suffix = m.group(2)
            variant = variant_suffix_map.get(variant_suffix, "primary")
            
            # Option A: base_l2 保留原始名 (不替换); canonical 仅用于规则查询
            canonical_l2 = LEVEL2_ALIAS.get(base_l2, base_l2)
            
            # === L1: 规则表反查 ===
            level1 = rule_l2_to_l1.get(canonical_l2)
            if level1 is not None:
                results.append({
                    "layer": layer_name,
                    "level1": level1,
                    "base_level2": base_l2,             # ★Q4★ 原始基名 (如 "其他型电站")
                    "canonical_level2": canonical_l2,   # ★Q4★ 规范名 (如 "变电站")
                    "variant": variant,
                    "resolution_path": "rule_table_lookup",
                })
                continue
            
            # === L2: feature_dataset 元数据探针 ===
            try:
                fds_name = self._probe_feature_dataset(gdb_path, layer_name)
                level1 = LEVEL1_DATASET_MAP.get(fds_name) if fds_name else None
                if level1 is not None:
                    results.append({
                        "layer": layer_name,
                        "level1": level1,
                        "base_level2": base_l2,
                        "canonical_level2": canonical_l2,
                        "variant": variant,
                        "resolution_path": "feature_dataset_probe",
                    })
                    continue
            except Exception:
                pass
            
            # === L3: 父级文件夹路径推断 (v0.4.6 兜底) ===
            path_l1, _path_l2_ignored = self._infer_category_from_path(gdb_path)
            if path_l1:
                results.append({
                    "layer": layer_name,
                    "level1": path_l1,
                    "base_level2": base_l2,
                    "canonical_level2": canonical_l2,
                    "variant": variant,
                    "resolution_path": "path_inference",
                })
                continue
            
            # 全部失败 -> unresolved, 让 M1 模糊匹配兜底
            results.append({
                "layer": layer_name,
                "level1": "",
                "base_level2": base_l2,
                "canonical_level2": canonical_l2,
                "variant": variant,
                "resolution_path": "unresolved",
            })
        
        return results

    def _read_single_gdb(self, gdb_path: str):
        """读取单个GDB，列出所有空间图层。
        v0.5: 改用三级解析链 _parse_gdb_layer_structure 解析每个 layer 的
              (level1, base_level2, canonical_level2, variant);
              不再依赖单一的文件夹路径推断。
        """
        rel_path = os.path.relpath(gdb_path, self.project_dir)
        
        # 推断行政区 (与 v0.4.6 相同)
        district_code = self._extract_district_code(gdb_path)
        district_name = self._extract_district_name(gdb_path)
        
        # ★ v0.5 核心: 三级解析链 ★
        layer_specs = self._parse_gdb_layer_structure(gdb_path)
        if not layer_specs:
            # 解析失败 (GDB 打不开等) 已在 _parse_gdb_layer_structure 内打过日志
            self.read_log.append({"path": rel_path, "status": "error",
                                  "message": "_parse_gdb_layer_structure 返回空"})
            return
        
        # 兼容 v0.4.6: 同时保留文件夹推断结果作为 _theme 字段
        # (M1 仍用 _theme 走"第一级 确定性映射"分支)
        folder_l1, folder_l2 = self._infer_category_from_path(gdb_path)
        theme = folder_l2 or folder_l1 or ""
        
        spatial_layers = []
        for spec in layer_specs:
            layer_name = spec["layer"]
            try:
                gdf = gpd.read_file(gdb_path, layer=layer_name)
                if gdf.geometry.isna().all() or len(gdf) == 0:
                    logger.debug(f"跳过非空间/空表: {layer_name} in {rel_path}")
                    continue
                
                # 记录CRS
                layer_crs = str(gdf.crs) if gdf.crs else "UNKNOWN"
                
                # ★ v0.5 字段写入: Option A 双字段策略 ★
                gdf["_source_gdb"] = rel_path
                gdf["_source_layer"] = layer_name
                gdf["_district_code"] = district_code
                gdf["_district_name"] = district_name
                gdf["_theme"] = theme
                gdf["_path_level1"] = spec["level1"]
                gdf["_path_level2"] = spec["base_level2"]                  # ★Q4★ 原始基名
                gdf["_path_level2_canonical"] = spec["canonical_level2"]   # ★Q4★ 规则查询规范名
                gdf["_variant"] = spec["variant"]
                gdf["_resolution_path"] = spec["resolution_path"]
                
                self.all_vector_layers.append(gdf)
                spatial_layers.append(layer_name)
                
                # 更新 variant_inventory (使用统一辅助方法, 与 SHP 通道一致)
                self._update_variant_inventory(
                    spec["base_level2"], spec["canonical_level2"], spec["variant"]
                )
                
                # 更新解析路径分布统计
                self.resolution_path_distribution[spec["resolution_path"]] += 1
                
                self.read_log.append({
                    "path": rel_path, "layer": layer_name,
                    "status": "ok", "features": len(gdf), "crs": layer_crs,
                    "path_level1": spec["level1"],
                    "path_level2": spec["base_level2"],
                    "path_level2_canonical": spec["canonical_level2"],
                    "variant": spec["variant"],
                    "resolution_path": spec["resolution_path"],
                })
            
            except Exception as e:
                logger.warning(f"读取图层失败: {layer_name} in {gdb_path}: {e}")
                self.read_log.append({
                    "path": rel_path, "layer": layer_name,
                    "status": "error", "message": str(e)
                })

        if spatial_layers:
            self.gdb_inventory.append({
                "path": rel_path,
                "district_code": district_code,
                "district_name": district_name,
                "theme": theme,
                "path_level1": folder_l1,
                "path_level2": folder_l2,
                "layers": spatial_layers,
                "crs": str(self.all_vector_layers[-1].crs) if self.all_vector_layers else "UNKNOWN",
                "feature_count": sum(
                    len(g) for g in self.all_vector_layers
                    if g["_source_gdb"].iloc[0] == rel_path
                )
            })

    # ─── SHP 扫描 ───────────────────────────────────────────
    def _scan_shp_files(self):
        """递归扫描所有 .shp 文件"""
        logger.info("扫描SHP文件...")
        shp_count = 0
        for root, dirs, files in os.walk(self.project_dir):
            # 跳过 gdb 内部
            if ".gdb" in root:
                continue
            for f in files:
                if f.lower().endswith(".shp"):
                    shp_path = os.path.join(root, f)
                    self._read_single_shp(shp_path)
                    shp_count += 1
        logger.info(f"发现 {shp_count} 个SHP文件")

    def _read_single_shp(self, shp_path: str):
        """读取单个SHP.
        v0.5 (E3): 与 GDB 通道同款做变体后缀识别 + Option A 双字段写入.
                   SHP 没有 feature_dataset 概念, 故跳过 L2 探针, 直接走 L1 (规则反查) 或 L3 (路径推断).
        """
        rel_path = os.path.relpath(shp_path, self.project_dir)
        try:
            gdf = gpd.read_file(shp_path)
            if gdf.geometry.isna().all() or len(gdf) == 0:
                logger.debug(f"跳过空SHP: {rel_path}")
                return
            
            # ★ v0.5 (E3): SHP filename stem 套变体后缀正则 ★
            filename_stem = os.path.splitext(os.path.basename(shp_path))[0]
            variant_re = re.compile(r"^(.+?)(_保护范围|_奖励范围|_缓冲范围|_河道范围)?$")
            m = variant_re.match(filename_stem)
            base_l2 = m.group(1)
            variant_suffix = m.group(2)
            variant = {
                None: "primary", "_保护范围": "protection",
                "_缓冲范围": "protection",
                "_河道范围": "protection",  # ★大埔工程实际数据★
                "_奖励范围": "reward",
            }.get(variant_suffix, "primary")
            
            # Option A 双字段
            canonical_l2 = LEVEL2_ALIAS.get(base_l2, base_l2)
            
            # ★ 提前算 folder_l1/folder_l2: 后面 _theme 字段需要 (与 GDB 通道一致),
            #   L3 路径推断兜底也会用到 folder_l1.
            folder_l1, folder_l2 = self._infer_category_from_path(shp_path)
            
            # === L1: 规则表反查 ===
            rule_l2_to_l1 = self._get_rule_l2_to_l1_index()
            level1 = rule_l2_to_l1.get(canonical_l2)
            resolution_path = "rule_table_lookup" if level1 is not None else None
            
            # === L3: 路径推断兜底 (SHP 跳过 L2, 因为没有 feature_dataset) ===
            if level1 is None:
                if folder_l1:
                    level1 = folder_l1
                    # 注: 若文件夹推断出了 level2, 但 filename 已剥变体后缀给出了 base_l2,
                    #     我们仍以 filename 的 base_l2 为准 (变体识别优先)
                    resolution_path = "path_inference"
                else:
                    level1 = ""
                    resolution_path = "unresolved"
            
            # 写入字段 (与 GDB 通道一致)
            gdf["_source_gdb"] = ""
            gdf["_source_layer"] = rel_path
            gdf["_district_code"] = self._extract_district_code(shp_path)
            gdf["_district_name"] = self._extract_district_name(shp_path)
            # ★ 修复 #2: _theme 来自文件夹推断 (与 GDB 通道一致), 不是 base_l2.
            #   M1 第一级"确定性映射"用 _theme 查 DETERMINISTIC_THEME_MAP (含三区三线关键词
            #   如"永久基本农田"、"城镇开发边界"), 来源必须是文件夹路径而非 filename.
            gdf["_theme"] = folder_l2 or folder_l1 or ""
            gdf["_path_level1"] = level1
            gdf["_path_level2"] = base_l2                  # ★Q4★ 原始基名
            gdf["_path_level2_canonical"] = canonical_l2   # ★Q4★ 规则查询规范名
            gdf["_variant"] = variant
            gdf["_resolution_path"] = resolution_path
            
            self.all_vector_layers.append(gdf)
            
            # 更新 variant_inventory (使用统一辅助方法, 与 GDB 通道一致)
            self._update_variant_inventory(base_l2, canonical_l2, variant)
            self.resolution_path_distribution[resolution_path] += 1
            
            self.read_log.append({
                "path": rel_path, "status": "ok", "features": len(gdf),
                "crs": str(gdf.crs) if gdf.crs else "UNKNOWN",
                "path_level1": level1,
                "path_level2": base_l2,
                "path_level2_canonical": canonical_l2,
                "variant": variant,
                "resolution_path": resolution_path,
            })
        except Exception as e:
            logger.warning(f"读取SHP失败: {shp_path}: {e}")
            self.read_log.append({"path": rel_path, "status": "error", "message": str(e)})

    # ─── 栅格扫描 ───────────────────────────────────────────
    def _scan_raster_files(self):
        """扫描 DEM/DSM/DOM 及其他栅格"""
        logger.info("扫描栅格文件...")
        for root, dirs, files in os.walk(self.project_dir):
            if ".gdb" in root:
                continue
            for f in files:
                if f.lower().endswith((".tif", ".tiff")):
                    tif_path = os.path.join(root, f)
                    self._read_raster_meta(tif_path)
        logger.info(f"发现 {len(self.raster_inventory)} 个栅格文件")

    def _read_raster_meta(self, tif_path: str):
        """读取栅格元数据。

        v0.4 修复:
          - 问题 7: ds.crs is None 时用 source_crs 兜底 (与矢量对齐)
          - 问题 2: 计算 bounds_in_working_crs 供下游 M2/M3 对齐使用
        """
        rel_path = os.path.relpath(tif_path, self.project_dir)
        try:
            with rasterio.open(tif_path) as ds:
                # v0.4: CRS 兜底 (问题 7)
                src_crs = ds.crs
                crs_fallback_used = False
                if src_crs is None:
                    logger.warning(
                        f"栅格缺少CRS, 假设为 source_crs={self.source_crs}: {rel_path}"
                    )
                    try:
                        src_crs = CRS(self.source_crs)
                    except Exception:
                        src_crs = None
                    crs_fallback_used = True

                meta = {
                    "path": rel_path,
                    "abs_path": tif_path,
                    "crs": str(src_crs) if src_crs is not None else "None",
                    "crs_fallback_used": crs_fallback_used,
                    "width": ds.width,
                    "height": ds.height,
                    "bounds": list(ds.bounds),
                    "res": list(ds.res),
                    "dtype": str(ds.dtypes[0]),
                    "band_count": ds.count,
                    "nodata": ds.nodata,
                    "inferred_type": self._infer_raster_type(rel_path, ds),
                }

                # v0.4: 计算 bounds_in_working_crs (问题 2)
                # 让下游 M3._determine_bbox 和 M2._process_terrain 能用米制坐标
                bounds_wcrs = self._compute_bounds_in_working_crs(
                    ds.bounds, src_crs, self.working_crs
                )
                if bounds_wcrs is not None:
                    meta["bounds_in_working_crs"] = list(bounds_wcrs)
                    # 同时提供 working_crs 下的像素分辨率估计 (用于问题 4 的精度判定)
                    width_wcrs = bounds_wcrs[2] - bounds_wcrs[0]
                    height_wcrs = bounds_wcrs[3] - bounds_wcrs[1]
                    if ds.width > 0 and ds.height > 0:
                        meta["res_in_working_crs"] = [
                            width_wcrs / ds.width,
                            height_wcrs / ds.height,
                        ]

                self.raster_inventory.append(meta)
                self.read_log.append({"path": rel_path, "status": "ok", "type": "raster"})
        except Exception as e:
            logger.warning(f"读取栅格失败: {tif_path}: {e}")
            self.read_log.append({"path": rel_path, "status": "error", "message": str(e)})

    @staticmethod
    def _compute_bounds_in_working_crs(bounds, src_crs, working_crs):
        """
        v0.4 新增 (问题 2): 把 bounds 从 src_crs 转到 working_crs。
        用 rasterio.warp.transform_bounds 避免单点投影导致的角度-投影边界偏差。

        Returns:
            (xmin, ymin, xmax, ymax) 或 None (失败时)
        """
        if src_crs is None or bounds is None:
            return None
        try:
            from rasterio.warp import transform_bounds
            src = CRS(src_crs) if not isinstance(src_crs, CRS) else src_crs
            dst = CRS(working_crs) if not isinstance(working_crs, CRS) else working_crs
            if src == dst:
                return tuple(bounds)
            xmin, ymin, xmax, ymax = transform_bounds(
                src, dst, bounds[0], bounds[1], bounds[2], bounds[3], densify_pts=21
            )
            return (xmin, ymin, xmax, ymax)
        except Exception as e:
            logger.debug(f"bounds 转 working_crs 失败: {e}")
            return None

    def _infer_raster_type(self, rel_path: str, ds) -> str:
        """推断栅格类型 (v0.4.3: 补中文关键词)

        真实甲方数据常按中文命名目录/文件名, 原来只识别英文 dem/dsm 等,
        漏识别会让关键栅格被标 UNKNOWN_RASTER, 下游完全取不到数据。
        """
        path_lower = rel_path.lower()
        # 英文关键词
        if "dem" in path_lower or "dsm" in path_lower:
            return "DEM"
        if "dom" in path_lower:
            return "DOM"
        if "wind" in path_lower or "ice" in path_lower:
            return "WIND_ICE_ZONE"
        if "landuse" in path_lower or "land_use" in path_lower:
            return "LANDUSE"
        # v0.4.3: 中文关键词 (不做 lower, 中文不受大小写影响; 直接在原始 rel_path 上匹配)
        # DEM: 数字高程/高程/地形 均为工程常用文件夹名
        if any(k in rel_path for k in ("数字高程", "高程", "地形", "DEM")):
            return "DEM"
        if any(k in rel_path for k in ("数字表面", "地表模型")):
            return "DSM"
        # 风冰: "风冰"/"风区覆冰"/"风区"/"覆冰"
        if any(k in rel_path for k in ("风冰", "风区覆冰", "风区", "覆冰")):
            return "WIND_ICE_ZONE"
        # 土地利用
        if any(k in rel_path for k in ("土地利用", "用地分类", "地类")):
            return "LANDUSE"
        return "UNKNOWN_RASTER"

    # ─── 控制对象读取 ───────────────────────────────────────
    def _read_control_objects(self):
        """读取起终点、必经点、必经路径、密集通道、可入区域"""
        logger.info("读取控制对象...")
        control_dir = self.project_dir / "control"
        control_map = {
            "start_end": ["start_end.geojson", "start_end.shp"],
            "must_pass": ["must_pass.geojson", "must_pass.shp"],
            "must_path": ["must_path.geojson", "must_path.shp"],
            "dense_corridor": ["dense_corridor.geojson", "dense_corridor.shp"],
            "accessible_area": ["accessible_area.geojson", "accessible_area.shp"],
        }

        for key, candidates in control_map.items():
            loaded = False
            for fname in candidates:
                fpath = control_dir / fname
                if fpath.exists():
                    try:
                        gdf = gpd.read_file(str(fpath))
                        self.control_objects[key] = gdf
                        logger.info(f"读取控制对象 {key}: {len(gdf)} 要素")
                        loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"读取控制对象失败 {key}: {e}")

            if not loaded:
                logger.info(f"控制对象 {key} 未提供")

        # 从 project.json 直接读起终点坐标 (备选)
        if "start_end" not in self.control_objects:
            start = self.config.get("start_point")
            end = self.config.get("end_point")
            if start and end:
                from shapely.geometry import Point
                import pandas as pd
                gdf = gpd.GeoDataFrame(
                    {"type": ["start", "end"]},
                    geometry=[Point(start), Point(end)],
                    crs=self.source_crs,
                )
                self.control_objects["start_end"] = gdf
                logger.info("从project.json读取起终点")
        
        # ★v0.5 C3★ 起终点完全缺失时记录降级事件 (不抛错, 日志降级)
        # 设计: m0 不应在起终点缺失时崩, 让 m3 的 _determine_bbox 兜底链路兜
        #        (DEM bounds → [0,0,50000,50000] 默认值);
        #        此处只是在 crs_diagnostic 中留下事件痕迹, 供 manifest 审计.
        if "start_end" not in self.control_objects:
            self.crs_diagnostic.setdefault("missing_control_objects", []).append({
                "name": "start_end",
                "severity": "info",  # 降级而非 warning/error: m3 有兜底
                "message": "起终点 layer 未提供 (control/start_end.* 不存在, "
                           "project.json 也未填 start_point/end_point); "
                           "M3 将走 bbox 兜底链路 (DEM bounds → 默认 50km 矩形)",
                "suggested_action": "建议补齐 control/start_end.geojson 或 "
                                    "project.json 中的 start_point / end_point",
            })
            logger.info("[C3] 起终点未提供, 已在 crs_diagnostic 中记录降级事件")

    # ─── CRS统一与合并 ─────────────────────────────────────
    def _unify_and_merge(self) -> gpd.GeoDataFrame:
        """将所有矢量图层统一到 working_crs"""
        logger.info(f"统一CRS到 {self.working_crs}...")
        unified = []
        target_crs = CRS(self.working_crs)

        for gdf in self.all_vector_layers:
            try:
                if gdf.crs is None:
                    logger.warning(f"图层缺少CRS，假设为 {self.source_crs}: "
                                   f"{gdf['_source_layer'].iloc[0]}")
                    gdf = gdf.set_crs(self.source_crs)
                if gdf.crs != target_crs:
                    gdf = gdf.to_crs(target_crs)
                unified.append(gdf)
            except Exception as e:
                logger.error(f"CRS转换失败: {gdf['_source_layer'].iloc[0]}: {e}")

        # 统一控制对象CRS
        for key, gdf in self.control_objects.items():
            try:
                if gdf.crs is None:
                    gdf = gdf.set_crs(self.source_crs)
                if gdf.crs != target_crs:
                    gdf = gdf.to_crs(target_crs)
                self.control_objects[key] = gdf
            except Exception as e:
                logger.error(f"控制对象CRS转换失败 {key}: {e}")

        if not unified:
            logger.warning("没有有效的矢量图层！")
            import pandas as pd
            return gpd.GeoDataFrame(columns=["geometry"], crs=self.working_crs)

        # 合并所有图层（字段取并集）
        import pandas as pd
        result = gpd.GeoDataFrame(pd.concat(unified, ignore_index=True), crs=self.working_crs)
        logger.info(f"合并完成: {len(result)} 个要素")
        return result

    # ─── 输出 ──────────────────────────────────────────────
    def _export(self, unified_gdf: gpd.GeoDataFrame) -> dict:
        """输出 M0 产物.
        v0.5 新增: variant_inventory.json + protection_coverage_report.json 两个审计文件.
        """
        m0_dir = ensure_dir(str(self.output_dir / "m0"))

        # gdb_inventory.json
        save_json({"gdb_files": self.gdb_inventory},
                  os.path.join(m0_dir, "gdb_inventory.json"))

        # raster_inventory.json
        save_json({"rasters": self.raster_inventory},
                  os.path.join(m0_dir, "raster_inventory.json"))

        # read_log.json
        save_json({"log": self.read_log},
                  os.path.join(m0_dir, "read_log.json"))

        # v0.4: CRS 诊断报告 (working_crs 合理性 / 轴序 / 起终点坐标校验)
        save_json(self.crs_diagnostic,
                  os.path.join(m0_dir, "crs_diagnostic.json"))

        # ★ v0.5 新增 1: variant_inventory.json (Phase A · A6) ★
        # 列出每个 (base_level2) 各变体存在情况, 含 canonical 名与 alias_from 反向链
        variant_inv_out = {
            "by_level2": self.variant_inventory,
            "resolution_path_distribution": self.resolution_path_distribution,
        }
        save_json(variant_inv_out, os.path.join(m0_dir, "variant_inventory.json"))
        logger.info(f"输出 variant_inventory.json: {len(self.variant_inventory)} 个二级类别")

        # ★ v0.5 新增 2: protection_coverage_report.json (Phase A · A6) ★
        # 列出规则表中 buffer_m > 0 但 GDB 缺失 _保护范围 变体的清单
        coverage_report = self._build_protection_coverage_report()
        save_json(coverage_report, os.path.join(m0_dir, "protection_coverage_report.json"))
        if coverage_report["missing_protection_for_required"]:
            logger.warning(
                f"protection_coverage_report: 有 {len(coverage_report['missing_protection_for_required'])} "
                f"条规则要求保护范围但 GDB 无对应 _保护范围 变体, 详情见 m0/protection_coverage_report.json"
            )

        # 统一矢量存储
        # ★修复 #17 (v0.5+)★ 用公共安全写 GPKG 工具 (含 FieldError fallback)
        if len(unified_gdf) > 0:
            unified_path = os.path.join(m0_dir, "unified_vectors.gpkg")
            from utils.geo_utils import write_gdf_to_gpkg_safe
            write_gdf_to_gpkg_safe(unified_gdf, unified_path, "unified_vectors")

        # 控制对象 (control_objects 字段简单, 一般不会触发 FieldError, 但兜底用安全写)
        from utils.geo_utils import write_gdf_to_gpkg_safe
        for key, gdf in self.control_objects.items():
            ctrl_path = os.path.join(m0_dir, f"control_{key}.gpkg")
            write_gdf_to_gpkg_safe(gdf, ctrl_path, f"control_{key}")

        return {
            "unified_gdf": unified_gdf,
            "gdb_inventory": self.gdb_inventory,
            "raster_inventory": self.raster_inventory,
            "control_objects": self.control_objects,
            "read_log": self.read_log,
            "crs_diagnostic": self.crs_diagnostic,  # v0.4
            # v0.5 新增 (供下游审计/诊断, 不参与算法决策)
            "variant_inventory": self.variant_inventory,
            "resolution_path_distribution": self.resolution_path_distribution,
        }

    def _build_protection_coverage_report(self) -> Dict[str, Any]:
        """v0.5 新增: 检查规则表中要求保护范围 (buffer_m > 0) 的 level2,
        在 variant_inventory 中是否有对应的 _保护范围 变体.
        
        Returns:
            {
              "missing_protection_for_required": [{"rule_id":...,"level2":...,"buffer_m":...}],
              "summary": {"required_count":N,"covered_count":M,"missing_count":N-M},
            }
        """
        try:
            rules_path = os.path.join(get_config_dir(), "default_feature_rules.json")
            rules_data = load_json(rules_path)
            features = rules_data.get("features", rules_data) if isinstance(rules_data, dict) else rules_data
        except Exception as e:
            logger.warning(f"protection_coverage_report 无法加载规则表: {e}")
            return {"missing_protection_for_required": [], "summary": {}}
        
        missing = []
        required = 0
        covered = 0
        # 注: 检查时按 canonical 名查 inventory; canonical 不同于 base 的 alias 也算覆盖
        # variant_inventory 同时记录了 base 名和 canonical 名两个 key (canonical 那个 key 的 alias_from 非空)
        for feat in features:
            buffer_m = feat.get("buffer_m", 0) or 0
            l2 = feat.get("level2", "")
            if buffer_m <= 0 or not l2:
                continue
            required += 1
            # 检查 inventory 中 canonical=l2 或 base=l2 的条目有 protection 变体
            has_protection = False
            for base_name, info in self.variant_inventory.items():
                if (base_name == l2 or info.get("canonical") == l2) and info.get("protection", 0) > 0:
                    has_protection = True
                    break
            if has_protection:
                covered += 1
            else:
                missing.append({
                    "rule_id": feat.get("id"),
                    "level1": feat.get("level1"),
                    "level2": l2,
                    "buffer_m": buffer_m,
                    "note": "规则要求 buffer_m > 0 但 GDB 中未找到 _保护范围 变体",
                })
        
        return {
            "missing_protection_for_required": missing,
            "summary": {
                "required_count": required,
                "covered_count": covered,
                "missing_count": required - covered,
            },
        }

    # ─── 辅助：从路径推断信息 ──────────────────────────────
    def _extract_district_code(self, path: str) -> str:
        match = DISTRICT_CODE_RE.search(os.path.basename(os.path.dirname(path)))
        if not match:
            match = DISTRICT_CODE_RE.search(path)
        return match.group(1) if match else ""

    def _extract_district_name(self, path: str) -> str:
        match = DISTRICT_NAME_RE.search(path)
        return match.group(1) if match else ""

    def _infer_category_from_path(self, file_path: str):
        """
        ★ 核心方法：从文件路径的父级文件夹链中推断地物 (level1, level2)。

        遍历从 .gdb/.shp 所在目录往上的每一级文件夹名，
        尝试匹配 PATH_KEYWORD_TO_CATEGORY。

        匹配策略：
          1. 优先匹配二级类别（精确）
          2. 其次匹配一级类别
          3. 如果路径中同时命中了一级和二级，组合返回

        示例：
          "生态敏感点/森林公园/xxx.gdb"  → ("生态敏感点", "森林公园")
          "441402梅江区/城镇开发边界/xxx.gdb"  → ("重要设施与政府规划敏感点", "城镇规划区")
          "vectors/all_features.shp"  → ("", "")  # 无法从路径推断
        """
        # 取相对于项目根目录的路径，拆分为文件夹列表
        try:
            rel = os.path.relpath(file_path, self.project_dir)
        except ValueError:
            rel = file_path
        parts = Path(rel).parts  # e.g. ("441402梅江区", "城镇开发边界", "xxx.gdb")

        # 过滤掉 .gdb 目录本身和纯数字/编码目录
        folder_names = []
        for p in parts:
            if p.endswith(".gdb") or p.endswith(".shp"):
                continue
            # 跳过纯数字目录名（如 441402）
            if re.match(r"^\d+$", p):
                continue
            folder_names.append(p)

        # 从最内层（最靠近.gdb）往外扫描，收集所有命中
        matched_l1 = ""
        matched_l2 = ""
        for folder in reversed(folder_names):
            # 精确匹配
            if folder in PATH_KEYWORD_TO_CATEGORY:
                l1, l2 = PATH_KEYWORD_TO_CATEGORY[folder]
                if l2 and not matched_l2:
                    matched_l2 = l2
                    if not matched_l1:
                        matched_l1 = l1
                elif l1 and not matched_l1:
                    matched_l1 = l1
                continue

            # 包含匹配：文件夹名中包含关键词
            for keyword, (l1, l2) in PATH_KEYWORD_TO_CATEGORY.items():
                if keyword in folder:
                    if l2 and not matched_l2:
                        matched_l2 = l2
                        if not matched_l1:
                            matched_l1 = l1
                    elif l1 and not matched_l1:
                        matched_l1 = l1
                    break

        return matched_l1, matched_l2

    def _infer_theme(self, path: str) -> str:
        """兼容旧接口：返回最细粒度的主题名"""
        l1, l2 = self._infer_category_from_path(path)
        return l2 or l1 or ""
