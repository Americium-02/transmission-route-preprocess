"""
M2: 几何预处理与专题对象构建 (v0.5)

v0.5 主要改造:
  - B1: _process_points 不再自动 buffer (Q2 决策, 甲方提供 _保护范围 变体)
  - B2: _process_polygons 按 _variant 字段分流 (primary/protection/reward)
       + _classify_protection_polygon 12 行分发表
       + _fix_protection_geometry 几何修复 (make_valid 处理甲方保护范围)
       + _safe_level2 静态辅助方法 (处理 std_level2 NaN/pd.NA 鲁棒性)
  - B5: _process_preferred_corridors 从 _奖励范围 变体读走廊
       (★Q1 决策★ 删除 parallel_valid_zone 字段, 不再实例化 ParallelAnalyzer)
  - B7: _process_linear_cross 不再 buffer 线对象进 no_tower
  - B9: _process_rivers 加 _variant=='primary' 过滤
  - feature-flag 守卫 (B3 机场 + B4 建筑物聚类) 默认禁用 legacy 处理

v5.3 旧版要点 (保留): preferred_corridors 携带完整元数据, 河流/风区/覆冰区处理.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import (
    Point, LineString, Polygon, MultiPolygon, MultiLineString, mapping
)
from shapely.ops import unary_union
from shapely.validation import make_valid  # v0.5 新增: B2 _fix_protection_geometry 用
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import label as ndimage_label

from utils.geo_utils import (
    load_json, save_json, ensure_dir, get_config_dir,
    azimuth_deg, segment_line_fast, evaluate_land_cost, get_base_cross_cost,
    parallel_range_m_for,
    river_wide_barrier_polys, river_narrow_cross_segments,  # ★P7★ 河流宽窄分治(v0.6 圆盘法, 留作回退/工具)
    river_polygon_centerline,                               # ★P7★ 中点中心线
    river_split_by_width,                                   # ★P7 v0.6.1 (A2)★ 形态学开运算分宽窄
)

logger = logging.getLogger("transmission_planning.m2")


class M2GeometryPreprocessor:
    """M2 几何预处理"""

    def __init__(self, project_config: dict, output_dir: str,
                 raster_inventory: list):
        self.config = project_config
        self.output_dir = Path(output_dir)
        self.raster_inventory = raster_inventory
        self.working_crs = project_config.get("working_crs", "EPSG:4547")
        self.data_avail = project_config.get("data_availability", {})

        # v0.4: 问题 6 修复
        # 老 v0.3 语义: data_availability 里的 river_polygon / wind_ice_zone_raster
        # 一旦扫描到对应数据就会被强制 True, 用户填什么都不管用。现在改为:
        #   - 默认: 扫到即用 (保留旧行为兼容)
        #   - 明确 force_disable: 即使扫到数据也走降级 (便于回归测试/质量不达标时降级)
        self.force_disable_river_polygon = bool(
            project_config.get("force_disable_river_polygon", False)
            or self.data_avail.get("force_disable_river_polygon", False)
        )
        self.force_disable_wind_ice_raster = bool(
            project_config.get("force_disable_wind_ice_raster", False)
            or self.data_avail.get("force_disable_wind_ice_raster", False)
        )

        rules_path = os.path.join(get_config_dir(), "default_feature_rules.json")
        rules_data = load_json(rules_path)
        self.rule_by_id = {f["id"]: f for f in rules_data["features"]}
        self.rule_by_l2 = {f["level2"]: f for f in rules_data["features"]}

        self.building_cluster_params = project_config.get("building_cluster", {
            "buffer_m": 100, "merge_gap_m": 50,
            "dense_zone_penalty": 3000, "min_cluster_area_m2": 50000,
        })
        self.river_params = project_config.get("river_rule", {
            "major_river_threshold_m": 900,
            # v5.6: 数据降级时的保守宽河名单 (按河名关键词匹配)
            # 默认空列表 = 行为与 v5.5 一致; 项目可在 project.json 的 river_rule 里配:
            # "conservative_wide_river_names": ["长江", "黄河", "珠江", "松花江"]
            "conservative_wide_river_names": [],
        })
        # 保证 river_params 里始终有保守名单字段, 方便 _river_degraded 读取
        self.river_params.setdefault("conservative_wide_river_names", [])
        self.river_params.setdefault("major_river_threshold_m", 900)

        # ★P7 (v0.6)★ 河流宽窄分治开关 (默认 True = 新行为: 宽河真实多边形禁区 + 窄河角度线段;
        #   False = 回退 v0.5: 宽段 pt.buffer(w/2) 圆盘禁区 + 窄段仅 river_crossing_windows)。
        #   来源与 M3 的 enable_fine_resolution 一致, 取 project_config.solver_params。
        self.enable_river_real_barrier = bool(
            (project_config.get("solver_params", {}) or {}).get(
                "enable_river_real_barrier", True))

        # ★P5 (v0.6)★ 不消费产物总开关 (默认 False): 关闭算法端不消费的输出栅格。
        #   M2 侧覆盖 wind_ice / terrain(slope/tpi/valley/peak) 栅格的**写盘**;
        #   分析结果(slope_max / valley_pixels / wind_ice_nodata_ratio 等)仍入
        #   preprocessing_report(供交付级别判定), 不受影响。M3 侧门控 mask/lpcf/tscf/
        #   tower_difficulty/nav_graph。来源同 M3, 取 project_config.solver_params。
        self.emit_unconsumed_outputs = bool(
            (project_config.get("solver_params", {}) or {}).get(
                "emit_unconsumed_outputs", False))

        # ★P6 (v0.6)★ 地形处理分辨率下限 (米): _process_terrain 不按 DEM 原生分辨率处理,
        #   而是降采样到 max(原生, 下限)。默认 10m (对齐细分辨率产物; 1m DEM→10m 后内存可控)。
        #   slope/TPI 对粗规划走廊在 10–100m 尺度足够; 真实高分辨率 DEM(亚米)按原生处理会内存爆。
        #   将来若需更细的地形(如亚米塔位), 调小此值 + 确保内存。0/None → 不限制(按原生, 慎用)。
        _tfloor = (project_config.get("solver_params", {}) or {}).get(
            "terrain_proc_resolution_floor_m",
            project_config.get("terrain_proc_resolution_floor_m", 10.0))
        try:
            self.terrain_proc_res_floor_m = float(_tfloor) if _tfloor else 0.0
        except (TypeError, ValueError):
            self.terrain_proc_res_floor_m = 10.0

        # 输出集合
        self.forbidden_polygons = []
        self.no_tower_polygons = []
        self.cost_polygons = []
        self.linear_cross_segments = []
        self.preferred_corridors = []
        # ★Q2 决策 (v0.5)★ buffered_points 保留空列表向后兼容;
        # 不再被填充, _process_points 不再自动 buffer (甲方已提供 _保护范围 变体)
        self.buffered_points = []
        self.building_clusters_gdf = None
        self.river_crossing_windows = None
        self.wide_river_barriers = None
        self.processed_polygons = []
        self.preprocessing_report = {}
        # v5.3: 收集生态敏感区面（用于平行有效带扣除）
        self.eco_sensitive_polygons = []
        # ★v0.5 新增 (B2)★ 几何修复日志: _fix_protection_geometry 写入此列表,
        # 在 _export 末尾输出为 m2/geometry_fixes_log.json 供审计
        self.geometry_fixes_log: List[Dict] = []

    def run(self, standardized_gdf: gpd.GeoDataFrame,
            control_objects: dict) -> dict:
        logger.info("===== M2: 几何预处理启动 =====")

        # v0.4.3: 让 _process_terrain (以及未来的其它步骤) 能读到控制对象,
        # 用于推断工作区 bbox 以裁剪 DEM, 避免全量处理
        self._m2_control_objects = control_objects or {}

        if len(standardized_gdf) > 0:
            points_gdf = standardized_gdf[
                standardized_gdf.geometry.geom_type.isin(["Point", "MultiPoint"])
            ].copy()
            lines_gdf = standardized_gdf[
                standardized_gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])
            ].copy()
            polygons_gdf = standardized_gdf[
                standardized_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            ].copy()

            logger.info(f"要素统计: 点={len(points_gdf)}, 线={len(lines_gdf)}, 面={len(polygons_gdf)}")

            self._process_points(points_gdf)
            self._process_polygons(polygons_gdf)
            self._process_linear_cross(lines_gdf)
            self._process_airports(lines_gdf, polygons_gdf)
            self._process_rivers(lines_gdf, polygons_gdf)
            self._process_building_clusters(polygons_gdf, points_gdf)
            self._process_preferred_corridors(polygons_gdf, lines_gdf)  # ★v0.5 B5★ 增加 polygons_gdf 入参

        self._process_wind_ice()
        self._process_terrain()
        self._process_control_objects(control_objects)

        result = self._export(control_objects)
        logger.info("M2完成")
        return result

    # ─── 8.1 点状对象处理 ──────────────────────────────────
    def _process_points(self, points_gdf: gpd.GeoDataFrame):
        """v0.5 (B1): 不再自动 buffer 点对象, 不再消费 extra_protection_tiers.
        
        ★Q2 决策落地★ 甲方已将变电站/换流站/接地极等点状对象的保护范围
        以 _保护范围 变体 layer 形式直接提供 (M0 通过变体识别归入 protection 流);
        M2 本函数仅做关键基础设施的计数统计登记.
        
        - self.buffered_points 保留为空列表 (向后兼容; _export 守卫 if 自然不写 gpkg)
        - extra_protection_tiers 字段在配置层 (Phase D · D4) 与代码层 (本函数) 同步下线
        """
        logger.info("处理点状对象 (v0.5: 仅做统计登记, 不再自动 buffer)...")
        critical_infra_summary = {"变电站": 0, "换流站": 0, "接地极": 0, "其他": 0}
        
        for _, row in points_gdf.iterrows():
            rule_id = row.get("std_rule_id", -1)
            l2 = row.get("std_level2", "")
            if rule_id < 0 and l2 == "UNKNOWN":
                continue
            rule = self.rule_by_id.get(rule_id, self.rule_by_l2.get(l2))
            if not rule:
                continue
            
            # 关键基础设施统计 (沿用 v5.7 R4-5 行为, 仅去掉 buffer)
            l2_name = rule.get("level2", "")
            if l2_name in critical_infra_summary:
                critical_infra_summary[l2_name] += 1
            elif rule.get("cost_type") == "forbidden":
                critical_infra_summary["其他"] += 1
        
        # v5.7 R4-5: 关键基础设施统计
        critical_sum = sum(critical_infra_summary.values())
        if critical_sum > 0:
            logger.info(f"  关键基础设施点登记: {critical_infra_summary}")
            self.preprocessing_report["critical_infrastructure_buffers"] = critical_infra_summary

    # ─── 8.2 面状对象处理 ──────────────────────────────────
    def _process_polygons(self, polygons_gdf: gpd.GeoDataFrame):
        """v0.5 (B2): 按 _variant 字段分流处理面对象.
        
        ★Q2 决策★ primary 不再自动 buffer (甲方已提供 _保护范围 变体);
        ★Q4 决策★ entry["level2"] 优先取 std_level2 (原始名), 保留别名归并下的原始类别.
        
        变体分流逻辑:
        - primary:     原始地物本体, 原样进 _classify_polygon_entry; 顺手收集 eco
        - protection:  禁立塔/跨越代价环, 经 _fix_protection_geometry 后由
                       _classify_protection_polygon 按 12 行分发表分发到三库
        - reward:      贴近奖励走廊, 由 _process_preferred_corridors (B5) 处理, 此处跳过
        """
        logger.info("处理面状对象 (v0.5: 按 _variant 分流)...")
        # 统计三类变体处理计数
        variant_counts = {"primary": 0, "protection": 0, "reward": 0, "unknown": 0}
        
        for idx, row in polygons_gdf.iterrows():
            rule_id = row.get("std_rule_id", -1)
            l2 = row.get("std_level2", "")
            if rule_id < 0 and l2 == "UNKNOWN":
                continue
            rule = self.rule_by_id.get(rule_id, self.rule_by_l2.get(l2))
            if not rule:
                continue
            
            # 读 variant 字段 (M0 v0.5 写出); 向后兼容: 旧数据无字段时默认 primary
            # ★修复 #8 + #11★ 用 pd.isna() 同时处理 None / np.nan / pd.NA / pd.NaT 四种缺失值类型:
            #   - row.get("_variant", "primary") 在 _variant=NaN 时返回 NaN, 不是 default
            #   - pandas 在 nullable StringDtype 列下用 pd.NA, 不是 np.nan
            #   - pd.isna() 是 pandas 处理所有缺失值类型的标准接口, 最鲁棒
            _variant_raw = row.get("_variant", None)
            if pd.isna(_variant_raw):
                variant = "primary"
            else:
                variant = str(_variant_raw).strip() or "primary"
            
            if variant == "reward":
                # reward 变体走 B5 _process_preferred_corridors 单独处理, 此处不收集
                variant_counts["reward"] += 1
                continue
            
            if variant == "protection":
                # 几何修复 (make_valid 处理甲方提供的保护范围环)
                geom_fixed = self._fix_protection_geometry(
                    row.geometry, rule_id=rule["id"],
                    level2=self._safe_level2(row, rule),  # ★修复 #14★ pd.NA/NaN 安全
                    variant=variant,
                )
                if geom_fixed is None or geom_fixed.is_empty:
                    continue
                # 浅复制 row 后替换几何; 不污染原 gdf
                row_fixed = row.copy()
                row_fixed.geometry = geom_fixed
                self._classify_protection_polygon(row_fixed, rule)
                variant_counts["protection"] += 1
                # ★修复 #7★ protection 变体也写入 processed_polygons (供 processed_polygons.gpkg 审计)
                # v0.4.6 时 processed_polygons 含所有面对象; v0.5 protection 经分发后仍应进入审计视图
                self.processed_polygons.append({
                    "geometry": geom_fixed,
                    "level1": rule["level1"],
                    "level2": self._safe_level2(row, rule),  # ★修复 #14★ Q4 原名 + pd.NA 安全
                    "rule_id": rule["id"],
                    "_variant": "protection",  # ★v0.5★ 标记变体来源, 便于审计
                })
                # 顺手收集生态敏感面 (即便是 protection 变体也算 eco 区, 用于 B5 走廊扣减)
                if rule["level1"] == "生态敏感点":
                    self.eco_sensitive_polygons.append(geom_fixed)
                continue
            
            # primary 分支: 原样送下游, ★不再自动 buffer★
            # ★修复 #8★ NaN/未知 variant 不再误判: variant 已经过滤为合法值或 "primary" fallback
            if variant not in ("primary", "protection", "reward"):
                # 上面 protection/reward 已 continue, 这里只剩异常或 primary;
                # 任何不在三类合法变体中的值都视为异常
                variant_counts["unknown"] += 1
                logger.warning(f"未知 _variant={variant!r} 落到 primary 处理 (rule_id={rule_id})")
            else:
                variant_counts["primary"] += 1
            
            entry = {
                "geometry": row.geometry,
                "level1": rule["level1"],
                "level2": self._safe_level2(row, rule),  # ★修复 #14★ Q4 原名 + pd.NA 安全
                "rule_id": rule["id"],
                "_variant": "primary",  # ★修复 #7★ 标记变体来源, 与 protection 分支一致
                # 注: v0.5 故意不写 entry["buffered_geometry"] —— 不再自动 buffer
                #     _classify_polygon_entry 中 entry.get("buffered_geometry", entry["geometry"])
                #     fallback 写法天然正确, 无需改 _classify_polygon_entry
            }
            self._classify_polygon_entry(entry, rule)
            self.processed_polygons.append(entry)
            
            # v5.3: 收集生态敏感区面
            if rule["level1"] == "生态敏感点" and row.geometry and not row.geometry.is_empty:
                self.eco_sensitive_polygons.append(row.geometry)
        
        if sum(variant_counts.values()) > 0:
            logger.info(f"  面对象 variant 分布: {variant_counts}")
            self.preprocessing_report["polygon_variant_counts"] = variant_counts

    @staticmethod
    def _safe_level2(row_or_value, rule: dict) -> str:
        """v0.5 (修复 #14): 安全获取 level2 字段, 处理 pandas truthy NaN 陷阱.
        
        问题背景: `row.get("std_level2") or rule["level2"]` 在 std_level2 是 np.nan 时
                   会取 NaN 而非 fallback (NaN 在 bool 上下文是 truthy).
                   pd.NA 在 bool 上下文会抛 TypeError.
        本函数统一处理 None / np.nan / pd.NA / pd.NaT / 空字符串 / 空白字符串 6 种缺失:
        全部 fallback 到 rule["level2"] (规则表 canonical 名).
        
        Args:
            row_or_value: 要么是 pd.Series (调用 .get("std_level2")), 要么直接是 level2 值
            rule: 规则字典 (含 level2 字段作 fallback)
        Returns:
            优先返回原始 std_level2; 任何缺失/空值时 fallback 到 rule["level2"] (规范名)
        """
        if hasattr(row_or_value, "get"):
            raw = row_or_value.get("std_level2")
        else:
            raw = row_or_value
        # pd.isna 覆盖 None / np.nan / pd.NA / pd.NaT
        if pd.isna(raw):
            return rule.get("level2", "")
        s = str(raw).strip()
        if not s:
            return rule.get("level2", "")
        return s

    def _fix_protection_geometry(self, geom, rule_id: int, level2: str, variant: str):
        """v0.5 (B2): 修复甲方提供的保护/奖励/缓冲范围环型面的无效拓扑.
        
        只修复 invalid 几何 (make_valid 是 shapely 2.x 的拓扑修复标准接口,
        保证修复后与原几何拓扑等价, 不改变边界). 修复事件写入 geometry_fixes_log
        供 _export 输出 m2/geometry_fixes_log.json 审计.
        
        ★不用 buffer(0)★ 老写法 buffer(0) 在部分实现上会引入微小膨胀; 不用 simplify
        (会丢精度).
        
        Returns:
            修复后的有效几何, 或 None (修复后变空 / 输入空)
        """
        if geom is None or geom.is_empty:
            return None
        if geom.is_valid:
            return geom
        try:
            fixed = make_valid(geom)
        except Exception as e:
            self.geometry_fixes_log.append({
                "rule_id": rule_id, "level2": level2, "variant": variant,
                "fix_type": "make_valid_exception", "error": str(e)[:200],
                "discarded": True,
            })
            return None
        # GeometryCollection 处理: 只保留有面积的 Polygon/MultiPolygon 分量
        if fixed.geom_type == "GeometryCollection":
            polys = [g for g in fixed.geoms
                     if g.geom_type in ("Polygon", "MultiPolygon") and g.area > 0]
            if not polys:
                self.geometry_fixes_log.append({
                    "rule_id": rule_id, "level2": level2, "variant": variant,
                    "fix_type": "make_valid_to_empty", "discarded": True,
                })
                return None
            fixed = unary_union(polys)
        # 仍可能是 LineString/Point 等非面类型 (尖刺修复后退化), 丢弃
        if fixed.geom_type not in ("Polygon", "MultiPolygon"):
            self.geometry_fixes_log.append({
                "rule_id": rule_id, "level2": level2, "variant": variant,
                "fix_type": "make_valid_non_polygon", "discarded": True,
                "result_type": fixed.geom_type,
            })
            return None
        self.geometry_fixes_log.append({
            "rule_id": rule_id, "level2": level2, "variant": variant,
            "fix_type": "make_valid", "discarded": False,
        })
        return fixed

    def _classify_protection_polygon(self, row, rule: dict):
        """v0.5 (B2): 把 protection 变体面按规则两轴 (立塔原则 × 跨越规则)
        分发到 forbidden / no_tower / cost 三库.
        
        规则表两个属性独立正交:
          立塔原则: 禁止立塔 / 数值代价(常数) / 0
          跨越规则: 禁止跨越 / 数值代价(常数) / 公式代价 / 0
        共 3 × 4 = 12 组合, 每组合分发模式不同.
        分发表参见 implementation_guide §4.2.
        
        Args:
            row: 已经过 _fix_protection_geometry 的几何 + std_level2 / std_rule_id 等字段
            rule: 来自 default_feature_rules.json 的规则字典 (按 std_rule_id 或 std_level2 查得)
        """
        geom = row.geometry
        l2 = self._safe_level2(row, rule)  # ★修复 #14★ Q4 原名 + pd.NA 安全
        rule_id = rule["id"]
        
        land_cost_val = evaluate_land_cost(rule)
        is_landable = rule.get("is_landable", True)
        cross_cost_val = get_base_cross_cost(rule)
        cross_allow = rule.get("cross_allow", True)
        cross_formula = rule.get("cross_cost_formula")
        cost_type = rule.get("cost_type", "fixed")
        
        # 分发轴 1: 跨越禁 (禁止跨越 / cross_cost ≥ 999999) → forbidden
        if not cross_allow or cross_cost_val >= 999999:
            self.forbidden_polygons.append({
                "geometry": geom, "level2": l2, "rule_id": rule_id,
                "source": "client_protection_range",
            })
        
        # 分发轴 2: 立塔禁 (禁止立塔 / land_cost ≥ 999999) → no_tower
        if not is_landable or land_cost_val >= 999999:
            self.no_tower_polygons.append({
                "geometry": geom, "level2": l2, "rule_id": rule_id,
                "source": "client_protection_range",
            })
        
        # 分发轴 3: 任何"非禁"的代价 → cost
        # 注: 禁 (999999) 不进 cost, 因为已经在 forbidden/no_tower 表达
        lc = land_cost_val if land_cost_val < 999999 else 0
        cc = cross_cost_val if cross_cost_val < 999999 else 0
        if lc > 0 or cc > 0 or cross_formula:
            self.cost_polygons.append({
                "geometry": geom, "level2": l2, "rule_id": rule_id,
                "land_cost": lc, "cross_cost": cc,
                "cost_type": cost_type,
                "cross_cost_formula": cross_formula,
                "source": "client_protection_range",
            })

    def _classify_polygon_entry(self, entry: dict, rule: dict):
        is_enterable = rule.get("is_enterable", True)
        is_landable = rule.get("is_landable", True)
        cross_allow = rule.get("cross_allow", True)
        cost_type = rule.get("cost_type", "fixed")

        if not is_enterable and not cross_allow:
            geom = entry.get("buffered_geometry", entry["geometry"])
            self.forbidden_polygons.append({
                "geometry": geom, "level2": entry["level2"],
                "rule_id": entry["rule_id"],
            })
            return

        land_cost_val = evaluate_land_cost(rule)
        if not is_landable or land_cost_val >= 999999:
            geom = entry.get("buffered_geometry", entry["geometry"])
            self.no_tower_polygons.append({
                "geometry": geom, "level2": entry["level2"],
                "rule_id": entry["rule_id"],
            })

        lc = land_cost_val if land_cost_val < 999999 else 0
        cc = get_base_cross_cost(rule)
        cc = cc if cc < 999999 else 0

        if lc > 0 or cc > 0:
            self.cost_polygons.append({
                "geometry": entry["geometry"],
                "level2": entry["level2"], "rule_id": entry["rule_id"],
                "land_cost": lc, "cross_cost": cc,
                "cost_type": cost_type,
                "cross_cost_formula": rule.get("cross_cost_formula"),
            })

    # ─── 8.3 线状交叉对象方向索引 ─────────────────────────
    def _process_linear_cross(self, lines_gdf: gpd.GeoDataFrame):
        logger.info("构建线状交叉对象方向索引...")
        seg_count = 0
        for idx, row in lines_gdf.iterrows():
            rule_id = row.get("std_rule_id", -1)
            l2 = row.get("std_level2", "")
            rule = self.rule_by_id.get(rule_id, self.rule_by_l2.get(l2))
            if not rule:
                continue
            min_cross_angle = rule.get("min_cross_angle")
            cost_type = rule.get("cost_type", "fixed")
            cross_cost_formula = rule.get("cross_cost_formula")
            buffer_m = rule.get("buffer_m", 0)
            base_cc = get_base_cross_cost(rule)
            if min_cross_angle is None and base_cc <= 0:
                continue

            geom = row.geometry
            coords_list = self._extract_coords(geom)
            parent_id = f"feat_{idx}"

            for coords in coords_list:
                # v5.3: 使用快速分段
                segments = segment_line_fast(coords, seg_length_m=50.0)
                for i, (seg_geom, az) in enumerate(segments):
                    self.linear_cross_segments.append({
                        "segment_id": f"seg_{idx}_{i}",
                        "parent_feature_id": parent_id,
                        "level1": rule["level1"], "level2": rule["level2"],
                        "rule_id": rule["id"],
                        "azimuth_deg": round(az, 2),
                        "min_cross_angle_deg": min_cross_angle,
                        "cross_cost": base_cc, "cost_type": cost_type,
                        "cross_cost_formula": cross_cost_formula,
                        "buffer_dist_m": buffer_m,
                        "geometry": seg_geom,
                    })
                    seg_count += 1
            
            # ★Q2 决策 (v0.5 B7)★ 删除 v0.4.6 在此处对线状对象做 buffer 进 no_tower 的块.
            # 线对象的禁立塔环 (65m/80m 等) 已由甲方以 输电线路N kV_保护范围 等 _保护范围
            # 变体面提供, M2 不再自行 buffer 线对象.
        logger.info(f"线状交叉分段: {seg_count} 段")

    @staticmethod
    def _extract_coords(geom) -> list:
        if geom.geom_type == "LineString":
            return [list(geom.coords)]
        elif geom.geom_type == "MultiLineString":
            return [list(part.coords) for part in geom.geoms]
        return []

    # ─── 8.4 机场处理 ──────────────────────────────────────
    def _process_airports(self, lines_gdf, polygons_gdf):
        """v0.5 (B3): 默认禁用 (feature-flag 守卫). 
        
        机场轮廓 4000m 缓冲带与机场线转面后的缓冲已由甲方以 机场_保护范围 变体面提供,
        M2 不再自行处理. 保留旧逻辑在 enable_airport_legacy_processing=True 时可回退.
        """
        if not self.config.get("enable_airport_legacy_processing", False):
            logger.info("跳过机场 legacy 处理 (v0.5 默认: 走甲方 _保护范围 变体)")
            return
        logger.info("处理机场 (legacy 模式)...")
        airport_rule = self.rule_by_l2.get("机场")
        if not airport_rule:
            return
        airport_polys = polygons_gdf[polygons_gdf["std_level2"] == "机场"]
        for _, row in airport_polys.iterrows():
            buffered = row.geometry.buffer(airport_rule["buffer_m"])
            self.forbidden_polygons.append({
                "geometry": buffered, "level2": "机场",
                "rule_id": airport_rule["id"],
            })
        airport_lines = lines_gdf[lines_gdf["std_level2"] == "机场"]
        if len(airport_lines) > 0:
            from shapely.ops import polygonize
            for _, row in airport_lines.iterrows():
                try:
                    polys = list(polygonize([row.geometry]))
                    for poly in polys:
                        buffered = poly.buffer(airport_rule["buffer_m"])
                        self.forbidden_polygons.append({
                            "geometry": buffered, "level2": "机场",
                            "rule_id": airport_rule["id"],
                        })
                except Exception as e:
                    logger.warning(f"机场线转面失败: {e}")

    # ─── 8.5 河流处理 ──────────────────────────────────────
    def _process_rivers(self, lines_gdf, polygons_gdf):
        logger.info("处理河流...")
        # v0.4 问题 6 修复: 明确三种状态的含义
        #   - force_disable_river_polygon=True -> 用户显式要求走降级, 即使有数据也忽略
        #   - 否则, 扫到河流面域 -> 完整分析
        #   - 否则, 走降级
        if self.force_disable_river_polygon:
            logger.warning("force_disable_river_polygon=True, 强制走降级 (用户显式要求)")
            self._river_degraded(lines_gdf)
            self.preprocessing_report["river_polygon_available"] = False
            self.preprocessing_report["river_width_analysis"] = "force_disabled"
            self.preprocessing_report["river_impact"] = (
                "用户显式 force_disable, 宽河屏障未生成"
            )
            return

        river_polygon_available = self.data_avail.get("river_polygon", False)
        # ★v0.5 (B9)★ 只在 primary 变体面上做宽度分析;
        #   _保护范围 变体面由 _process_polygons 走 B2 protection 分支处理 (禁立塔/跨越代价环),
        #   不应进入宽度分析 (那是地物本体的属性, 不是保护带的属性).
        if len(polygons_gdf) > 0 and "_variant" in polygons_gdf.columns:
            river_polys = polygons_gdf[
                (polygons_gdf["std_level1"] == "河流") &
                (polygons_gdf["_variant"] == "primary")
            ]
        elif len(polygons_gdf) > 0:
            # 向后兼容: 旧数据无 _variant 字段, 按 v0.4.6 行为走
            river_polys = polygons_gdf[polygons_gdf["std_level1"] == "河流"]
        else:
            river_polys = gpd.GeoDataFrame()
        if not river_polygon_available and len(river_polys) > 0:
            river_polygon_available = True
            logger.info("从主矢量数据中发现河流面域数据 (primary 变体)")
        threshold_m = self.river_params.get("major_river_threshold_m", 900)
        if river_polygon_available and len(river_polys) > 0:
            logger.info("河流面域数据可用，执行完整宽度分析...")
            self._river_full_analysis(river_polys, lines_gdf, threshold_m)
            self.preprocessing_report["river_polygon_available"] = True
        else:
            # ★v0.5 (B9)★ 降级路径调整:
            #   若 primary 缺失但 _保护范围 变体存在 (即 protection variant 已被 _process_polygons
            #   处理并已进 no_tower/cost), 不需要执行旧的 _river_degraded (那是 v0.4.6 没有 protection
            #   变体时基于线推断宽河区域的兜底).
            has_river_protection = False
            if len(polygons_gdf) > 0 and "_variant" in polygons_gdf.columns:
                has_river_protection = len(polygons_gdf[
                    (polygons_gdf["std_level1"] == "河流") &
                    (polygons_gdf["_variant"] == "protection")
                ]) > 0
            if has_river_protection:
                logger.info("河流 primary 缺失但 _保护范围 变体存在, 跳过 _river_degraded "
                            "(禁立塔/跨越代价已由 B2 protection 分支处理)")
                self.preprocessing_report["river_polygon_available"] = False
                self.preprocessing_report["river_width_analysis"] = "skipped_no_primary"
                self.preprocessing_report["river_impact"] = (
                    "primary 缺失, 仅保护范围环生效; 宽河窗口/屏障未生成"
                )
            else:
                logger.info("河流面域数据暂缺，执行降级处理...")
                self._river_degraded(lines_gdf)
                self.preprocessing_report["river_polygon_available"] = False
                self.preprocessing_report["river_width_analysis"] = "skipped"
                self.preprocessing_report["river_impact"] = (
                    "宽河屏障未生成，路径可能跨越实际不可跨越的宽河段"
                )

    def _river_full_analysis(self, river_polys, lines_gdf, threshold_m):
        # ★P7 v0.6.1 (A2)★ 宽窄分治:
        #   enable_river_real_barrier=True  → 宽段: 形态学开运算 open(river,r=T/2)∩river = 真实多边形禁区
        #                                     (完全沿河道, 无圆盘过覆盖, 不依赖中心线);
        #                                     窄段: river−宽段, 逐连通块取局部中心线切 50m 进 linear_cross
        #                                     (带方位角+min_cross_angle, R10 大幅削弱)。
        #   enable_river_real_barrier=False → 回退 v0.5: 宽段 pt.buffer(w/2) 圆盘直接进 forbidden; 窄段仅 windows。
        #   river_crossing_windows.gpkg / wide_river_barriers.gpkg 两分支都产 (审计/排查, 算法端不消费)。
        real_barrier = getattr(self, "enable_river_real_barrier", True)
        crossing_windows = []   # 审计: 窄段窗口点
        wide_barriers = []      # 审计: 宽段圆盘
        n_real_poly = 0         # 真实多边形禁区数 (real_barrier 路径)
        n_narrow_seg = 0        # 窄河 linear 段数 (real_barrier 路径)
        n_axis_fallback = 0     # ★A.7★ 中心线退化回退 MABR 直轴的连通块数 (审计)
        n_narrow_comp = 0       # ★A.7★ 窄段连通块总数 (审计, 配合上者看回退率)
        n_centerline_split = 0  # ★A.7.2★ 中心线因跳变/折返被切成多段的连通块数 (≈分岔口数)
        for idx, row in river_polys.iterrows():
            geom = row.geometry
            l2 = row.get("std_level2", "非通航河流")
            rule = self.rule_by_l2.get(l2, self.rule_by_l2.get("非通航河流"))
            is_navigable = (l2 == "通航河流")
            rid = rule.get("id") if rule else (61 if is_navigable else 62)
            min_angle = rule.get("min_cross_angle") if rule else None
            base_cc = get_base_cross_cost(rule) if rule else 0
            cost_type = rule.get("cost_type", "fixed") if rule else "fixed"
            cross_formula = rule.get("cross_cost_formula") if rule else None
            rlevel1 = rule.get("level1", "河流") if rule else "河流"
            if real_barrier:
                # ── ★P7 v0.6.1 (A2)★ 形态学开运算分宽窄 (替换圆盘法) ──
                #   wide = open(river, r=T/2) ∩ river  (完全沿河道, 无圆盘过覆盖, 不依赖中心线)
                #   narrow = river − wide              (逐连通块取局部中心线 → R10 大幅削弱)
                try:
                    wide_polys, narrow_polys = river_split_by_width(geom, threshold_m)
                except Exception as e:
                    logger.warning(f"河流形态学分治失败, 该河整体按窄段处理: {e}")
                    wide_polys, narrow_polys = [], [geom]

                # 宽段 → 真实多边形禁区 (rule_id=-1); 审计 wide_river_barriers 也存形态学多边形 (非圆盘)
                for poly in wide_polys:
                    self.forbidden_polygons.append({
                        "geometry": poly,
                        "level2": f"宽河屏障(≥{threshold_m}m)",
                        "rule_id": -1,
                    })
                    n_real_poly += 1
                    wide_barriers.append({
                        "geometry": poly, "river_name": row.get("name", ""),
                        "level2": l2, "source": "morph_open",
                    })

                # 窄段 → linear_cross 段 (逐窄段连通块取局部中心线, 带方位角 + 角度约束)。
                # ★P7 修正★ 仅对"有角度约束或有跨越代价"的河产窄段 (与通用 _process_linear_cross
                #   同口径): 通航(min_cross_angle=45)产; 非通航(无角度+cross_cost=0)纯噪声, 跳过。
                if min_angle is not None or base_cc > 0:
                    for ci, comp in enumerate(narrow_polys):
                        n_narrow_comp += 1
                        try:
                            # ★A.7.2 (v0.6.3)★ return_parts=True: 汇流口/分叉处中心线
                            #   在支流间斜穿折返的部分已被切开, 逐段处理。斜穿段若不切开,
                            #   会带着错误方位角进 linear_cross → 算法端跨越判定/交叉角错位;
                            #   且它仍在河面内, 出河检测抓不到 (见 geo_utils._cs_split_zigzag)。
                            parts = self._compute_river_widths(
                                comp, interval_m=100, return_parts=True)
                        except Exception as e:
                            logger.warning(f"窄段中心线计算失败: {e}")
                            continue
                        if not parts:
                            continue
                        if len(parts) > 1:
                            n_centerline_split += 1
                        for pi, (w_c, wp_c, cl_c) in enumerate(parts):
                            if cl_c is None:
                                continue
                            # ★A.7 (v0.6.2)★ 退化回退计数 (核心产不出中心线 → 走 MABR 直轴):
                            #   回退本身不再丢段(geo_utils 已补逐站宽度), 但回退率高说明
                            #   河流面被切得过碎, 值得在日志里可见。
                            if pi == 0 and len(parts) == 1 and len(list(cl_c.coords)) <= 2:
                                n_axis_fallback += 1
                            # ★A.7★ width_filter=False: 宽窄已由 A2 形态学开运算判定
                            #   (narrow = river − wide), 此处不再按逐站宽度二次过滤,
                            #   否则汇流口/展宽处宽度被抬高 → 段被误丢 → 中心线断开。
                            segs = river_narrow_cross_segments(
                                list(cl_c.coords), w_c, wp_c,
                                threshold_m, seg_length_m=50.0,
                                width_filter=False)
                            for si, (seg_geom, az) in enumerate(segs):
                                self.linear_cross_segments.append({
                                    "segment_id": f"river_{idx}_{ci}_{pi}_{si}",
                                    "parent_feature_id": f"river_feat_{idx}",
                                    "level1": rlevel1, "level2": l2,
                                    "rule_id": rid,
                                    "azimuth_deg": round(az, 2),
                                    "min_cross_angle_deg": min_angle,  # 通航 45 / 非通航 None
                                    "cross_cost": base_cc, "cost_type": cost_type,
                                    "cross_cost_formula": cross_formula,
                                    "buffer_dist_m": 0,  # 河本体面已表达范围, 线段不再 buffer
                                    "geometry": seg_geom,
                                })
                                n_narrow_seg += 1
                                crossing_windows.append({   # 审计窗口 = 窄段中点
                                    "window_id": f"rw_{idx}_{ci}_{pi}_{si}",
                                    "river_name": row.get("name", ""),
                                    "is_navigable": is_navigable,
                                    "width_m": None,
                                    "azimuth_deg": round(az, 1),
                                    "min_cross_angle_deg": min_angle,
                                    "cross_cost": base_cc,
                                    "geometry": seg_geom.interpolate(0.5, normalized=True),
                                })
            else:
                # ── v0.5 回退 (enable_river_real_barrier=False): 圆盘法, 圆盘直接进 forbidden ──
                try:
                    widths, width_points, center_line = self._compute_river_widths(
                        geom, interval_m=100)
                except Exception as e:
                    logger.warning(f"河流宽度计算失败: {e}")
                    widths, width_points, center_line = [], [], None
                for i, (w, pt) in enumerate(zip(widths, width_points)):
                    if w >= threshold_m:
                        disk = pt.buffer(w / 2)
                        self.forbidden_polygons.append({
                            "geometry": disk,
                            "level2": f"宽河屏障(≥{threshold_m}m)",
                            "rule_id": -1,
                        })
                        n_real_poly += 1
                        wide_barriers.append({
                            "geometry": disk, "river_name": row.get("name", ""),
                            "width_m": round(w, 1), "level2": l2,
                        })
                    else:
                        az = self._estimate_river_azimuth(geom, pt)
                        crossing_windows.append({
                            "window_id": f"rw_{idx}_{i}",
                            "river_name": row.get("name", ""),
                            "is_navigable": is_navigable,
                            "width_m": round(w, 1),
                            "azimuth_deg": round(az, 1),
                            "min_cross_angle_deg": min_angle,
                            "cross_cost": base_cc,
                            "geometry": pt,
                        })

        # ---- 审计产物 (两分支都产; 算法端不消费, 仅供 QGIS 排查) ----
        if crossing_windows:
            self.river_crossing_windows = gpd.GeoDataFrame(
                crossing_windows, crs=self.working_crs)
        if wide_barriers:
            self.wide_river_barriers = gpd.GeoDataFrame(
                wide_barriers, crs=self.working_crs)
        self.preprocessing_report["river_polygon_available"] = True
        self.preprocessing_report["river_real_barrier"] = bool(real_barrier)
        self.preprocessing_report["river_method"] = (
            "morph_open" if real_barrier else "disk_v05")
        self.preprocessing_report["river_crossing_windows_count"] = len(crossing_windows)
        self.preprocessing_report["wide_river_barriers_count"] = len(wide_barriers)
        self.preprocessing_report["river_real_barrier_poly_count"] = n_real_poly
        self.preprocessing_report["river_narrow_segment_count"] = n_narrow_seg
        # ★A.7★ 中心线质量审计: 回退率高 = 河流面被切得碎 / 参数需调
        self.preprocessing_report["river_narrow_component_count"] = n_narrow_comp
        self.preprocessing_report["river_centerline_axis_fallback_count"] = n_axis_fallback
        self.preprocessing_report["river_centerline_split_component_count"] = n_centerline_split
        if n_narrow_comp:
            logger.info(
                f"  河流中心线: 窄段连通块 {n_narrow_comp} 个, "
                f"其中退化回退直轴 {n_axis_fallback} 个 "
                f"({100.0 * n_axis_fallback / n_narrow_comp:.1f}%), "
                f"分岔口切分 {n_centerline_split} 个 "
                f"({100.0 * n_centerline_split / n_narrow_comp:.1f}%); "
                f"窄段 linear 段 {n_narrow_seg} 条")

    def _compute_river_widths(self, polygon, interval_m=100, return_parts=False):
        # ★P7 (v0.6)★ 委托 geo_utils.river_polygon_centerline:
        #   中心线点改用"垂直交线中点"(跟随河道横向摆动, 缓解 R10), 宽度口径不变。
        # ★A.7.2★ return_parts=True → 返回 [(widths, width_points, line), ...],
        #   汇流口斜穿折返处已切开; False → 单条(取最长), 兼容 v0.5 回退路径。
        return river_polygon_centerline(polygon, interval_m=interval_m,
                                        return_parts=return_parts)

    def _estimate_river_azimuth(self, river_poly, point) -> float:
        rect = river_poly.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
        edges = []
        for i in range(4):
            p1, p2 = coords[i], coords[i + 1]
            d = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
            edges.append((p1, p2, d))
        edges.sort(key=lambda e: e[2], reverse=True)
        p1, p2 = edges[0][0], edges[0][1]
        return azimuth_deg(p1[0], p1[1], p2[0], p2[1])

    def _river_degraded(self, lines_gdf):
        """v5.6: 降级模式保留 + 可选的"保守宽河名单"支持

        配置项 river_rule.conservative_wide_river_names (在 project.json 中):
          字符串列表, 如 ["长江", "黄河", "珠江", "松花江"]
          命中的河流线按"宽河屏障"处理: 沿线 450m 缓冲生成 forbidden 面
        这样在数据未到位时至少能保证主要大江不会被当成可跨越河流。
        未配置或为空列表时行为与 v5.5 完全一致。
        """
        conservative_names = self.river_params.get("conservative_wide_river_names", []) or []
        # 标准化: 去空格, 保留原字符串比对
        name_set = set(str(n).strip() for n in conservative_names if n)
        wide_count = 0

        river_lines = lines_gdf[lines_gdf["std_level1"] == "河流"]
        for idx, row in river_lines.iterrows():
            l2 = row.get("std_level2", "非通航河流")
            rule = self.rule_by_l2.get(l2)
            if not rule: continue

            # v5.6: 保守宽河检查 — 按名称匹配
            river_name = str(row.get("name", "") or row.get("名称", "") or "")
            is_conservative_wide = False
            if name_set:
                for key in name_set:
                    if key and key in river_name:
                        is_conservative_wide = True
                        break

            if is_conservative_wide:
                # 整条河线按保守宽度(默认 900m/2=450m)做缓冲屏障
                try:
                    geom = row.geometry
                    if geom is not None and not geom.is_empty:
                        buf = geom.buffer(self.river_params.get("major_river_threshold_m", 900) / 2)
                        self.forbidden_polygons.append({
                            "geometry": buf,
                            "level2": f"保守宽河屏障({river_name})",
                            "rule_id": -1,
                        })
                        wide_count += 1
                        continue  # 不再走线状跨越逻辑
                except Exception as e:
                    logger.warning(f"  保守宽河屏障生成失败 ({river_name}): {e}")

            coords_list = self._extract_coords(row.geometry)
            for coords in coords_list:
                segments = segment_line_fast(coords, seg_length_m=50.0)
                for i, (seg_geom, az) in enumerate(segments):
                    base_cc = get_base_cross_cost(rule) if rule else 0
                    self.linear_cross_segments.append({
                        "segment_id": f"river_seg_{idx}_{i}",
                        "parent_feature_id": f"river_{idx}",
                        "level1": "河流", "level2": l2,
                        "rule_id": rule["id"], "azimuth_deg": round(az, 2),
                        "min_cross_angle_deg": rule.get("min_cross_angle"),
                        "cross_cost": base_cc,
                        "cost_type": rule.get("cost_type", "fixed"),
                        "cross_cost_formula": rule.get("cross_cost_formula"),
                        "buffer_dist_m": rule.get("buffer_m", 0),
                        "geometry": seg_geom,
                    })

        if wide_count > 0:
            logger.info(f"  保守宽河屏障: {wide_count} 条 (按名称匹配)")
            self.preprocessing_report["conservative_wide_rivers_applied"] = wide_count

    # ─── 8.6 建筑物群聚类 ─────────────────────────────────
    def _process_building_clusters(self, polygons_gdf, points_gdf):
        """v0.5 (B4): 默认禁用 (feature-flag 守卫).
        
        建筑物聚类 (高密度居民区) 已由甲方以 建筑物_保护范围 变体面 (或专门的密集通道 layer) 提供,
        M2 不再自行做聚类计算. 保留旧逻辑在 enable_building_cluster_legacy=True 时可回退.
        """
        if not self.config.get("enable_building_cluster_legacy", False):
            logger.info("跳过建筑物聚类 legacy 处理 (v0.5 默认: 走甲方提供的密集通道/建筑物保护范围)")
            return
        logger.info("处理建筑物聚类 (legacy 模式)...")
        params = self.building_cluster_params
        buf_m = params.get("buffer_m", 100)
        merge_gap = params.get("merge_gap_m", 50)
        min_area = params.get("min_cluster_area_m2", 50000)
        penalty = params.get("dense_zone_penalty", 3000)

        building_geoms = []
        if len(polygons_gdf) > 0:
            bld_poly = polygons_gdf[polygons_gdf["std_level2"] == "建筑物"]
            building_geoms.extend(bld_poly.geometry.tolist())
        if len(points_gdf) > 0:
            bld_pts = points_gdf[points_gdf["std_level2"] == "建筑物"]
            building_geoms.extend(bld_pts.geometry.tolist())
        if not building_geoms:
            logger.info("无建筑物数据"); return

        buffered = [g.buffer(buf_m) for g in building_geoms]
        merged = unary_union(buffered)
        if merged.geom_type == "Polygon":
            cluster_polys = [merged]
        elif merged.geom_type == "MultiPolygon":
            cluster_polys = list(merged.geoms)
        else:
            cluster_polys = []

        if merge_gap > 0:
            extra_buffered = [p.buffer(merge_gap) for p in cluster_polys]
            merged2 = unary_union(extra_buffered)
            if merged2.geom_type == "Polygon":
                cluster_polys = [merged2.buffer(-merge_gap)]
            elif merged2.geom_type == "MultiPolygon":
                cluster_polys = [p.buffer(-merge_gap) for p in merged2.geoms]

        clusters = []
        for poly in cluster_polys:
            if poly.is_valid and not poly.is_empty and poly.area >= min_area:
                clusters.append({
                    "geometry": poly, "area_m2": round(poly.area, 1),
                    "penalty": penalty,
                })
                self.no_tower_polygons.append({
                    "geometry": poly, "level2": "建筑物密集区", "rule_id": -1,
                })
                self.cost_polygons.append({
                    "geometry": poly, "level2": "建筑物密集区", "rule_id": -1,
                    "land_cost": penalty, "cross_cost": penalty,
                })
        if clusters:
            self.building_clusters_gdf = gpd.GeoDataFrame(clusters, crs=self.working_crs)
        logger.info(f"建筑物聚类: {len(clusters)} 个密集区")

    # ─── 贴近奖励走廊（v5.3增强：携带完整元数据） ─────────
    def _process_preferred_corridors(self, polygons_gdf: gpd.GeoDataFrame,
                                     lines_gdf: gpd.GeoDataFrame):
        """v0.5 (B5): 从 _奖励范围 变体面读走廊, 完整保留 M3 消费的字段集.
        
        M3 (m3_rule_compile_and_output.py:516-553) 强消费这些字段:
          - corridor_geometry (走廊几何, 用于栅格化奖励)
          - parallel_reward (奖励值, W/1000m)
          - voltage_kv (800kV+ 排他判定)
          - exclusion_geometry (800kV+ 600m 禁并行几何; 设计路径规则 #17 派生)
        缺一个 M3 都会静默跳过对应规则.
        
        ★Q1 决策★ 本函数不再实例化 ParallelAnalyzer, 不再生成 parallel_valid_zone 字段
                  (M3 / 算法端零引用; 且其内部含违背 v0.5 设计的自 buffer).
        ★Q4 决策★ level2 字段优先取 std_level2 (原始名), 保留 "其他型电站" 等归并下的原名.
        """
        import re
        # ★Q1★ 不再 import ParallelAnalyzer (utils/parallel_geometry.py 模块保留不删, 仅 M2 不再引用)
        logger.info("处理贴近奖励走廊 (v0.5: 从 _奖励范围 变体读)...")
        
        # 主路径: 从 polygons_gdf 中筛选 _variant=="reward" 的面 (含空数据守卫)
        if len(polygons_gdf) == 0 or "_variant" not in polygons_gdf.columns:
            reward_polys = gpd.GeoDataFrame(geometry=[], crs=self.working_crs)
        else:
            reward_polys = polygons_gdf[polygons_gdf["_variant"] == "reward"]
        
        # 建立同 level2 的 primary 线索引 (推 line_azimuth / exclusion_geometry 用)
        primary_lines_by_l2 = {}
        if len(lines_gdf) > 0 and "_variant" in lines_gdf.columns:
            primary_lines = lines_gdf[lines_gdf["_variant"] == "primary"]
            for l2_key, grp in primary_lines.groupby("std_level2"):
                geoms = [g for g in grp.geometry.tolist() if g is not None and not g.is_empty]
                if geoms:
                    try:
                        primary_lines_by_l2[l2_key] = unary_union(geoms)
                    except Exception as e:
                        logger.warning(f"primary 线 unary_union 失败 (level2={l2_key}): {e}")
        elif len(lines_gdf) > 0:
            # 向后兼容: 旧数据无 _variant 字段, 把所有 lines 作为 primary
            for l2_key, grp in lines_gdf.groupby("std_level2"):
                geoms = [g for g in grp.geometry.tolist() if g is not None and not g.is_empty]
                if geoms:
                    try:
                        primary_lines_by_l2[l2_key] = unary_union(geoms)
                    except Exception as e:
                        logger.warning(f"primary 线 unary_union 失败 (level2={l2_key}): {e}")
        
        # ─── eco 处理策略 (B2 设计, 2026-05-13 用户确认) ───
        # v0.5 之前: 硬扣 — 把 reward 走廊中与 eco 重叠的部分用 difference 切掉
        # v0.5+: 不扣 — 走廊几何完整保留, 仅记录与 eco 的重叠比例作审计字段
        #
        # 改架构理由 (3 个):
        # 1) 语义正确: eco 自身在 cost_polygons 已经栅格化进 lpcf (代价 100~5000),
        #    reward 奖励(0.001 × parallel_reward)在数值上被 eco 高代价完全淹没.
        #    算法走最短代价路径时, 不会被几毫的奖励诱导钻进 eco. 硬扣是在做算法该做的事.
        # 2) 信息保留原则: 预处理不应替算法做不可逆决策.
        #    硬扣等于让算法永远看不到"这条走廊在 eco 里的部分".
        # 3) 可缓存性: 用户改 eco 属性时, M2 reward 走廊几何不变, 只需重跑 M3.
        #    硬扣架构下, 改 eco 属性会让 reward 几何被裁剪结果变化, 必须重跑 M2 (~20min).
        #
        # ─── 3.1 STRtree 加速 ──────────────────────────
        # 对每条走廊算 intersection 跟原 difference 一样慢, 必须用空间索引加速.
        # 用 STRtree 查询"和当前走廊 bbox 相交的 eco 子集", 只对小子集做 intersection.
        # shapely 2.0+ STRtree.query(geom) 返回 ndarray of int, 索引到原列表.
        eco_tree = None
        eco_polys_list = None
        if self.eco_sensitive_polygons:
            try:
                from shapely.strtree import STRtree   # shapely 2.0+
                # 过滤无效几何后建索引
                eco_polys_list = [g for g in self.eco_sensitive_polygons
                                  if g is not None and not g.is_empty]
                if eco_polys_list:
                    eco_tree = STRtree(eco_polys_list)
                    logger.info(
                        f"  eco 空间索引就绪: STRtree 含 {len(eco_polys_list)} 个 eco 面"
                    )
            except Exception as e:
                logger.warning(
                    f"  eco STRtree 索引构建失败 ({type(e).__name__}: {e}), "
                    f"reward 走廊将无 eco_overlap_ratio 信息"
                )
                eco_tree = None
                eco_polys_list = None
        
        # 运行时统计 (供跑后审计)
        _no_eco_overlap_count = 0      # bbox 完全不相交, overlap_ratio=0
        _has_eco_overlap_count = 0     # 真的有相交, overlap_ratio > 0
        _strtree_query_failed = 0      # STRtree 查询/intersection 异常退回 ratio=0
        _eco_overlap_ratios = []       # 用于统计分布
        
        for _, row in reward_polys.iterrows():
            l2 = row.get("std_level2", "")
            rule_id = row.get("std_rule_id", -1)
            rule = self.rule_by_id.get(rule_id, self.rule_by_l2.get(l2))
            if not rule:
                continue
            reward = rule.get("parallel_reward")
            if not reward or reward <= 0:
                continue
            # ★修复 #14★ 把 l2 规范化为安全字符串 (处理 NaN/pd.NA/空), 后续使用
            l2 = self._safe_level2(row, rule)
            
            # 几何修复 (甲方提供的奖励范围环可能有 invalid 拓扑)
            corridor_geom = self._fix_protection_geometry(
                row.geometry, rule_id=rule["id"],
                level2=l2,  # ★修复 #14★ 已规范化的 l2
                variant="reward",
            )
            if corridor_geom is None or corridor_geom.is_empty:
                continue
            
            # ─── B2 策略: 不裁剪, 只计算 eco_overlap 审计字段 ───
            # corridor_geom 永远保持原始几何形状, 不做 difference
            eco_overlap_area = 0.0
            eco_overlap_ratio = 0.0
            corridor_area = corridor_geom.area
            
            if eco_tree is not None and corridor_area > 0:
                try:
                    # STRtree.query 返回 candidate 索引数组 (基于 bbox 相交).
                    # 这是真实相交集的超集 (宁可多选, 不可漏选), 后续 intersection
                    # 严格筛掉假阳性. 对低候选数场景, 这一步通常 <1ms.
                    cand_idx = eco_tree.query(corridor_geom)
                    if len(cand_idx) > 0:
                        # 局部 eco 子集 — 只 union 走廊 bbox 范围内的 eco 面
                        local_eco_polys = [eco_polys_list[int(i)] for i in cand_idx]
                        if len(local_eco_polys) == 1:
                            local_eco = local_eco_polys[0]
                        else:
                            local_eco = unary_union(local_eco_polys)
                        # 用 intersection 精确算重叠面积
                        inter = corridor_geom.intersection(local_eco)
                        if inter is not None and not inter.is_empty:
                            # intersection 结果可能是 GeometryCollection, 只取面积
                            # 对所有 sub-geom 都计入 area (.area 自带处理)
                            eco_overlap_area = float(inter.area)
                            eco_overlap_ratio = eco_overlap_area / corridor_area
                            # 浮点防御
                            if eco_overlap_ratio > 1.0:
                                eco_overlap_ratio = 1.0
                            _has_eco_overlap_count += 1
                            _eco_overlap_ratios.append(eco_overlap_ratio)
                        else:
                            _no_eco_overlap_count += 1
                    else:
                        _no_eco_overlap_count += 1
                except Exception as e:
                    # intersection / STRtree.query 异常: 不阻塞走廊建立, ratio 留 0
                    _strtree_query_failed += 1
                    if _strtree_query_failed <= 3:  # 只警告前 3 个, 避免日志爆炸
                        logger.warning(
                            f"  reward 面 eco overlap 计算失败 ({l2}, rule_id={rule['id']}): "
                            f"{type(e).__name__}: {e}"
                        )
            
            # 电压解析 (从 level2 名字提取 kV 数字)
            voltage_kv = 0
            m = re.search(r'(\d+)\s*kV', l2, re.IGNORECASE)
            if m:
                voltage_kv = int(m.group(1))
            
            # 800kV+ 禁并行带 (设计路径规则 #17 派生几何):
            # ★Q4 决策注释★ 这里的 600m buffer 不是"保护范围扩充":
            #   - 它是 "设计路径规则 #17" 的几何派生 (800kV+ 600m 禁止并行排列)
            #   - 甲方不会提供 _禁止并行范围 这种变体层 (那是路径规则, 不是地物属性)
            #   - 因此 M2 必须从既有线路 primary 几何自行派生该禁并行带
            #   - 与 v0.5 "甲方提供所有保护范围 buffer" 的设计原则不矛盾
            exclusion_geometry = None
            primary_geom = primary_lines_by_l2.get(l2)
            if voltage_kv >= 800 and primary_geom is not None:
                exclusion_geometry = primary_geom.buffer(600)  # 设计路径规则 #17 派生
            
            # line_azimuth 推算 (从同 level2 的 primary 线)
            line_azimuth = 0
            if primary_geom is not None:
                try:
                    if primary_geom.geom_type == "MultiLineString":
                        longest = max(primary_geom.geoms, key=lambda g: g.length)
                        coords = list(longest.coords)
                    elif primary_geom.geom_type == "LineString":
                        coords = list(primary_geom.coords)
                    else:
                        coords = []
                    if len(coords) >= 2:
                        line_azimuth = azimuth_deg(
                            coords[0][0], coords[0][1],
                            coords[-1][0], coords[-1][1]
                        )
                except Exception as e:
                    logger.warning(f"line_azimuth 推算失败 ({l2}): {e}")
            
            self.preferred_corridors.append({
                "geometry": row.geometry,                    # 原始 reward 面 (保留备用)
                "corridor_geometry": corridor_geom,          # ★ M3 line 517 读这个 (B2 后: 完整原始几何, 不再扣 eco)
                # ★Q1 决策★ parallel_valid_zone 字段在 v0.5 删除 (M3 不消费, 且其内部含违背 v0.5 设计的自 buffer)
                "level2": l2,                                # ★修复 #14★ l2 已规范化 (Q4 原名或 rule fallback)
                "rule_id": rule["id"],
                "parallel_reward": reward,                   # ★ 保留旧字段名: M3 line 520 读这个
                "parallel_range_m": rule.get("parallel_range_m", 200),
                "buffer_m": rule.get("buffer_m", 0),
                "voltage_kv": voltage_kv,                    # ★ M3 line 538 读这个
                "exclusion_geometry": exclusion_geometry,    # ★ M3 line 540 读这个; 设计路径规则 #17 派生 (非保护范围)
                "line_azimuth": line_azimuth,                # ★ 留给算法端做平行性判定
                "source": "client_reward_range",
                # ★B2 审计字段★ 与 eco 的重叠比例 / 面积. 仅供算法端参考与运维审计,
                # corridor_geometry 几何本身不受 eco 影响.
                # eco_overlap_ratio ∈ [0, 1]: 0=无重叠, 1=完全在 eco 内
                "eco_overlap_ratio": eco_overlap_ratio,
                "eco_overlap_area": eco_overlap_area,
                # eco_clipped 保留但语义变了: True = M2 在 eco_sensitive_polygons 非空时跑过相交计算
                # (不再代表"几何被扣过"). 让审计向后兼容, 算法端如果在用应该转用 eco_overlap_ratio.
                "eco_clipped": False,                        # B2: 永远不再裁剪
            })
        
        # legacy fallback: 若 feature-flag 打开, 走 v0.4.6 旧逻辑做补充
        # (用 lines_gdf 自己 buffer 出走廊, 仅作回退用)
        if self.config.get("enable_corridor_legacy", False):
            logger.info("启用走廊 legacy 模式 (从 lines_gdf buffer 推走廊)")
            self._preferred_corridors_legacy(lines_gdf)
        
        logger.info(f"贴近奖励走廊: {len(self.preferred_corridors)} 条")
        # ★B2 审计统计★ 输出 eco overlap 分布, 便于跑后看 eco 与 reward 走廊的关系
        if _has_eco_overlap_count or _no_eco_overlap_count:
            total = _has_eco_overlap_count + _no_eco_overlap_count
            overlap_pct = _has_eco_overlap_count / total if total > 0 else 0
            logger.info(
                f"  eco overlap 分布: 无重叠 {_no_eco_overlap_count}/{total} "
                f"({(1-overlap_pct):.1%}), 有重叠 {_has_eco_overlap_count}/{total} "
                f"({overlap_pct:.1%})"
            )
            if _eco_overlap_ratios:
                ratios_arr = np.array(_eco_overlap_ratios)
                full_in_eco = int((ratios_arr >= 0.99).sum())
                near_full = int(((ratios_arr >= 0.8) & (ratios_arr < 0.99)).sum())
                logger.info(
                    f"  有重叠走廊 ratio 分布: mean={ratios_arr.mean():.3f}, "
                    f"median={np.median(ratios_arr):.3f}, "
                    f"完全在 eco 内(>=0.99): {full_in_eco} 条, "
                    f"主要在 eco 内([0.8, 0.99)): {near_full} 条"
                )
        if _strtree_query_failed:
            logger.warning(
                f"  ⚠ {_strtree_query_failed} 条走廊 eco overlap 计算异常 "
                f"(eco_overlap_ratio 默认 0; 走廊几何未受影响)"
            )
    
    def _preferred_corridors_legacy(self, lines_gdf: gpd.GeoDataFrame):
        """v0.4.6 旧逻辑保留 (feature-flag enable_corridor_legacy=True 时启用).
        
        用 lines_gdf 自己 buffer 出走廊, 不依赖 _奖励范围 变体. 仅作回退用.
        """
        import re
        for _, row in lines_gdf.iterrows():
            rule_id = row.get("std_rule_id", -1)
            l2 = row.get("std_level2", "")
            rule = self.rule_by_id.get(rule_id, self.rule_by_l2.get(l2))
            if not rule or not rule.get("parallel_reward"):
                continue
            reward = rule["parallel_reward"]
            range_m = parallel_range_m_for(rule)
            buffer_m = rule.get("buffer_m", 0)
            # ★修复 #14★ 规范化 l2 (避免 re.search 在 NaN 上抛 TypeError)
            l2 = self._safe_level2(row, rule)
            corridor_geom = row.geometry.buffer(range_m)
            voltage_kv = 0
            m = re.search(r'(\d+)\s*kV', l2, re.IGNORECASE)
            if m:
                voltage_kv = int(m.group(1))
            exclusion_geometry = None
            if voltage_kv >= 800:
                exclusion_geometry = row.geometry.buffer(600)
            line_azimuth = 0
            if hasattr(row.geometry, 'coords') and len(row.geometry.coords) >= 2:
                c = list(row.geometry.coords)
                line_azimuth = azimuth_deg(c[0][0], c[0][1], c[-1][0], c[-1][1])
            self.preferred_corridors.append({
                "geometry": row.geometry,
                "corridor_geometry": corridor_geom,
                "level2": l2,  # ★修复 #14★ 已规范化
                "rule_id": rule["id"],
                "parallel_reward": reward,
                "parallel_range_m": range_m,
                "buffer_m": buffer_m,
                "voltage_kv": voltage_kv,
                "exclusion_geometry": exclusion_geometry,
                "line_azimuth": line_azimuth,
                "source": "legacy_line_buffer",
                # B2 schema 一致性: legacy 分支不计算 eco overlap (legacy 走 lines buffer,
                # 与 eco 关系由 lpcf 网格自然处理), 字段留 0
                "eco_overlap_ratio": 0.0,
                "eco_overlap_area": 0.0,
                "eco_clipped": False,
            })

    # ─── 8.7 风区/覆冰区处理 ──────────────────────────────
    def _process_wind_ice(self):
        logger.info("处理风区/覆冰区...")
        # v0.4 问题 6 修复
        if self.force_disable_wind_ice_raster:
            logger.warning(
                "force_disable_wind_ice_raster=True, 强制走降级 (用户显式要求)"
            )
            cost_table_path = os.path.join(get_config_dir(), "wind_ice_cost_table.json")
            cost_table = load_json(cost_table_path) if os.path.exists(cost_table_path) else None
            self._wind_ice_degraded(cost_table)
            self.preprocessing_report["wind_ice_available"] = False
            self.preprocessing_report["wind_ice_status"] = "force_disabled"
            return

        wind_ice_available = self.data_avail.get("wind_ice_zone_raster", False)
        wind_ice_raster_path = None
        for r in self.raster_inventory:
            if r.get("inferred_type") == "WIND_ICE_ZONE":
                wind_ice_raster_path = r.get("abs_path")
                wind_ice_available = True; break
        cost_table_path = os.path.join(get_config_dir(), "wind_ice_cost_table.json")
        cost_table = load_json(cost_table_path) if os.path.exists(cost_table_path) else None
        if wind_ice_available and wind_ice_raster_path:
            logger.info("风区/覆冰区栅格可用，执行完整处理...")
            self._wind_ice_full(wind_ice_raster_path, cost_table)
            self.preprocessing_report["wind_ice_available"] = True
            return
        # ★v0.6 新增★ 无 .tif 栅格 → 若有 GDB/矢量风冰, 标准化成 GPKG(重投影+原始属性,
        #   不套代价/不分档/不栅格化) 作为空间交付物, 供算法端将来消费时自行栅格化+套代价。
        #   当前算法端用 manifest 全域 wind_zone/ice_zone, 故仍走 degraded 设全域参数。
        wi_vec_cfg = self.config.get("wind_ice_vector", {}) or {}
        if wi_vec_cfg.get("enabled", True):
            vec_path, is_gdb = self._find_wind_ice_vector()
            if vec_path:
                self._standardize_wind_ice_vector(vec_path, is_gdb)
        logger.info("风区/覆冰区: 当前用全域参数 (空间数据若存在已标准化为 GPKG 待算法端消费)...")
        self._wind_ice_degraded(cost_table)
        self.preprocessing_report["wind_ice_available"] = False

    def _find_wind_ice_vector(self):
        """★v0.6★ 扫 project_dir 找风冰矢量源 (含 .gdbtable 的 GDB 目录, 或 .shp),
        路径含 风区/覆冰/风冰/wind/ice 关键字。返回 (path, is_gdb) 或 (None, False)。"""
        pdir = self.config.get("_project_dir")
        if not pdir or not os.path.isdir(pdir):
            return None, False

        def _hit(name):
            low = name.lower()
            return (any(k in low for k in ("wind", "ice")) or
                    any(k in name for k in ("风区", "覆冰", "风冰")))

        # 1) GDB: 含 .gdbtable 的目录 (其相对路径含关键字)
        for root, _dirs, files in os.walk(pdir):
            if any(f.endswith(".gdbtable") for f in files):
                if _hit(os.path.relpath(root, pdir)):
                    return root, True
        # 2) .shp
        for root, _dirs, files in os.walk(pdir):
            for f in files:
                if f.lower().endswith(".shp") and _hit(
                        os.path.relpath(os.path.join(root, f), pdir)):
                    return os.path.join(root, f), False
        return None, False

    def _standardize_wind_ice_vector(self, vec_path, is_gdb):
        """★v0.6★ 风冰矢量(GDB/shp, 可多图层) → 重投影到 working_crs → 标准化为 GPKG
        (m2/wind_ice_zones.gpkg, 带 wi_kind=wind/ice/unknown + wi_layer + 原始属性)。
        **不套代价、不分档、不栅格化** —— 代价/分档/栅格化交由算法端在消费时按代价模型
        处理, 从而预处理现在即可完工、未来定了风冰代价也无需回改预处理 (最小化复工)。
        配置 project.json.wind_ice_vector: {enabled, layers?}。返回 gpkg 路径 或 None。
        """
        cfg = self.config.get("wind_ice_vector", {}) or {}
        _gdb_cleanup = None
        try:
            import geopandas as gpd
            import pandas as pd
            from utils.geo_utils import write_gdf_to_gpkg_safe

            # GDB 目录缺 .gdb 后缀 → 复制到临时 *.gdb 供 GDAL 识别
            open_path = vec_path
            if is_gdb and not vec_path.lower().endswith(".gdb"):
                import tempfile, shutil
                _gdb_cleanup = tempfile.mkdtemp(prefix="wi_gdb_")
                open_path = os.path.join(_gdb_cleanup, "wind_ice_zone.gdb")
                shutil.copytree(vec_path, open_path)
                logger.info(f"GDB 目录缺 .gdb 后缀, 复制到临时 {open_path} 供 GDAL 识别")

            # 确定图层 (配置 或 全部)
            layers = cfg.get("layers")
            if not layers:
                if is_gdb:
                    try:
                        import fiona
                        layers = fiona.listlayers(open_path)
                    except Exception:
                        layers = [None]
                else:
                    layers = [None]

            parts = []
            for L in layers:
                try:
                    g = gpd.read_file(open_path, layer=L) if L else gpd.read_file(open_path)
                except Exception as e:
                    logger.warning(f"风冰图层 {L} 读取失败: {e}")
                    continue
                if g is None or len(g) == 0:
                    continue
                if g.crs is not None and str(g.crs) != str(self.working_crs):
                    g = g.to_crs(self.working_crs)
                lname = (str(L) if L else "").lower()
                kind = ("wind" if ("wind" in lname or "风" in str(L or ""))
                        else "ice" if ("ice" in lname or "冰" in str(L or ""))
                        else "unknown")
                g = g.copy()
                g["wi_kind"] = kind
                g["wi_layer"] = str(L) if L else os.path.basename(vec_path)
                parts.append(g)
            if not parts:
                logger.warning("风冰矢量无有效图层, 跳过标准化")
                return None

            merged = gpd.GeoDataFrame(
                pd.concat(parts, ignore_index=True), crs=self.working_crs)
            m2_dir = ensure_dir(str(self.output_dir / "m2"))
            out = os.path.join(m2_dir, "wind_ice_zones.gpkg")
            write_gdf_to_gpkg_safe(merged, out, "wind_ice_zones")

            kinds = sorted(set(str(k) for k in merged["wi_kind"]))
            logger.info(
                f"风冰矢量已标准化 → {out} (图层={layers}, kind={kinds}, "
                f"{len(merged)} 要素, working_crs; 未套代价/未栅格化)")
            self.preprocessing_report["wind_ice_spatial_source"] = "vector_gpkg"
            self.preprocessing_report["wind_ice_zones_gpkg"] = "wind_ice_zones.gpkg"
            self.preprocessing_report["wind_ice_vector_layers"] = [str(x) for x in layers]
            self.preprocessing_report["wind_ice_vector_kinds"] = kinds
            return out
        except ImportError as e:
            logger.warning(f"风冰矢量标准化需 geopandas/fiona: {e}, 跳过")
            return None
        except Exception as e:
            logger.error(f"风冰矢量标准化失败: {e}, 跳过")
            return None
        finally:
            if _gdb_cleanup:
                try:
                    import shutil
                    shutil.rmtree(_gdb_cleanup, ignore_errors=True)
                except Exception:
                    pass

    def _wind_ice_full(self, raster_path, cost_table):
        zone_code_map = {}
        if cost_table:
            for c in cost_table.get("combinations", []):
                zone_code_map[c["zone_code"]] = c
        try:
            with rasterio.open(raster_path) as ds:
                data = ds.read(1); nodata = ds.nodata
                transform = ds.transform; profile = ds.profile.copy()
                src_crs = ds.crs
            if nodata is not None:
                nodata_mask = (data == nodata) | (data == 0)
            else:
                nodata_mask = (data == 0)
            nodata_ratio = nodata_mask.sum() / data.size
            if nodata_ratio > 0.2:
                logger.warning(f"风区/覆冰区栅格NoData占比 {nodata_ratio:.1%} > 20%")
            default_ice = self.config.get("ice_zone", 10)
            default_wind = self.config.get("wind_zone", "B")
            default_combo = self._find_combo(zone_code_map, default_wind, default_ice)
            h, w = data.shape
            path_adder = np.zeros((h, w), dtype=np.float32)
            max_turn = np.full((h, w), 90.0, dtype=np.float32)
            tower_mult = np.ones((h, w), dtype=np.float32)
            # v0.3: 历史上 line_cost_multiplier 在 cost_table / _wind_ice_degraded 都已声明,
            # M3 也有对应的读写代码路径, 但 _wind_ice_full 忘了产出, 导致 manifest 里
            # wind_ice_line_multiplier_*.tif 永远标 optional_missing。补上。
            line_mult = np.ones((h, w), dtype=np.float32)
            for code, combo in zone_code_map.items():
                mask = (data == code)
                path_adder[mask] = combo.get("path_cost_adder", 0)
                max_turn[mask] = combo.get("max_turn_angle_deg", 90)
                tower_mult[mask] = combo.get("tower_cost_multiplier", 1.0)
                line_mult[mask] = combo.get("line_cost_multiplier", 1.0)

            # v0.4.3 审核问题 2 修复: unknown zone_code 应和 NoData 一样回退默认组合,
            # 而不是保持初始化的 0/90/1/1 (会把重覆冰区错当普通区, 放宽转角限制)
            if zone_code_map:
                known_codes = list(zone_code_map.keys())
                known_mask = np.isin(data, known_codes)
                unknown_mask = (~nodata_mask) & (~known_mask)
                unknown_count = int(unknown_mask.sum())
                if unknown_count > 0:
                    logger.warning(
                        f"风区/覆冰区栅格含 {unknown_count} 个未定义 zone_code 像素 "
                        f"(占比 {unknown_count/data.size:.2%}), 回退到默认组合"
                    )
                fallback_mask = nodata_mask | unknown_mask
            else:
                fallback_mask = nodata_mask

            if default_combo:
                path_adder[fallback_mask] = default_combo.get("path_cost_adder", 0)
                max_turn[fallback_mask] = default_combo.get("max_turn_angle_deg", 90)
                tower_mult[fallback_mask] = default_combo.get("tower_cost_multiplier", 1.0)
                line_mult[fallback_mask] = default_combo.get("line_cost_multiplier", 1.0)

            m2_dir = ensure_dir(str(self.output_dir / "m2"))
            out_profile = profile.copy()
            out_profile.update(dtype="float32", count=1)

            # v0.4.3 审核问题 3 修复: 若源栅格缺 CRS, 用 source_crs 兜底,
            # 与 M0 _read_raster_meta 的兜底策略一致; 否则下游 resample_to_workspace
            # 会因 src.crs=None 失败, 走"假性降级"
            if out_profile.get("crs") is None:
                fallback_crs = self.config.get("source_crs", "EPSG:4490")
                out_profile["crs"] = fallback_crs
                logger.warning(
                    f"风冰栅格缺 CRS, 兜底写入 source_crs={fallback_crs}"
                )

            # ★P5 (v0.6)★ emit_unconsumed_outputs 总开关: 关闭时不写风冰栅格
            #   (算法端不消费; 下方 report 字段仍照常写, 供交付级别判定)。
            if self.emit_unconsumed_outputs:
                for arr, name in [(path_adder, "wind_ice_path_adder.tif"),
                                  (max_turn, "wind_ice_max_turn.tif"),
                                  (tower_mult, "wind_ice_tower_multiplier.tif"),
                                  (line_mult, "wind_ice_line_multiplier.tif")]:  # v0.3
                    with rasterio.open(os.path.join(m2_dir, name), "w", **out_profile) as dst:
                        dst.write(arr, 1)
            else:
                logger.info("emit_unconsumed_outputs=false, 跳过风冰栅格写盘 (分析结果仍入 report)")
            self.preprocessing_report["wind_ice_available"] = True
            self.preprocessing_report["wind_ice_nodata_ratio"] = round(nodata_ratio, 3)
            self.preprocessing_report["wind_ice_unknown_zone_pixels"] = int(
                (~nodata_mask & ~np.isin(data, list(zone_code_map.keys()))).sum()
                if zone_code_map else 0
            )
        except Exception as e:
            logger.error(f"风区/覆冰区栅格处理失败: {e}")
            self._wind_ice_degraded(cost_table)

    def _wind_ice_degraded(self, cost_table):
        ice_zone = self.config.get("ice_zone", 10)
        wind_zone = self.config.get("wind_zone", "B")
        unified_params = {
            "tower_cost_multiplier": 1.0, "line_cost_multiplier": 1.0,
            "max_turn_angle_deg": 45 if ice_zone >= 20 else 90,
            "path_cost_adder": 0,
        }
        if cost_table:
            for c in cost_table.get("combinations", []):
                if c.get("wind_zone") == wind_zone and c.get("ice_zone") == ice_zone:
                    unified_params.update({
                        "tower_cost_multiplier": c.get("tower_cost_multiplier", 1.0),
                        "line_cost_multiplier": c.get("line_cost_multiplier", 1.0),
                        "max_turn_angle_deg": c.get("max_turn_angle_deg", 90),
                        "path_cost_adder": c.get("path_cost_adder", 0),
                    })
                    break
        self.preprocessing_report["wind_ice_available"] = False
        self.preprocessing_report["wind_ice_unified_params"] = unified_params

    @staticmethod
    def _find_combo(zone_code_map, wind, ice):
        for code, combo in zone_code_map.items():
            if combo.get("wind_zone") == wind and combo.get("ice_zone") == ice:
                return combo
        return None

    # ─── DEM地形分析 ───────────────────────────────────────
    def _write_polygon_gpkg_or_placeholder(self, rows: list, filename: str, m2_dir: str):
        """
        v0.4.3: 面图层写 gpkg, 若空则写带 _placeholder=True 的占位文件。
        v0.5 修复 #17: 用 write_gdf_to_gpkg_safe 工具处理混合 schema 字段 (FieldError fallback)

        空工程(完全无禁区/高代价面)是合法状态, 但 manifest 把这些图层列为
        required, 不写会触发假性 required_missing。占位行带 Point(0,0) 几何
        仅为让 gpkg 文件合法; 算法端据 _placeholder 字段过滤即可。
        """
        from utils.geo_utils import write_gdf_to_gpkg_safe, polygon_area_m2
        path = os.path.join(m2_dir, filename)
        if rows:
            # ★P2 (v0.6)★ 统一补 area_m2 (几何重算 = 权威值)。在写盘单一出口注入,
            # 自动覆盖所有来源行 (classify / 河流 / building_clusters 等),
            # 对未来新增的 append 点免疫; 比在各 append 处分别加更不易漏。
            for r in rows:
                r["area_m2"] = polygon_area_m2(r.get("geometry"))
            gdf = gpd.GeoDataFrame(rows, crs=self.working_crs)
            write_gdf_to_gpkg_safe(gdf, path, filename.replace(".gpkg", ""))
            return
        # 占位: 1 行, 带标志字段, 零面积几何
        try:
            placeholder = gpd.GeoDataFrame(
                [{"_placeholder": True, "level2": "", "area_m2": 0.0,
                  "geometry": Point(0, 0)}],  # ★P2★ 占位行 area_m2=0.0
                crs=self.working_crs,
            )
            write_gdf_to_gpkg_safe(placeholder, path, filename.replace(".gpkg", "") + "_placeholder")
            logger.info(
                f"{filename}: 空图层, 写占位 gpkg (含 _placeholder=True 标志)"
            )
        except Exception as e:
            logger.warning(f"写占位 {filename} 失败: {e}")

    def _decimate_full_read(self, ds, src_transform, src_bounds,
                            src_width, src_height):
        """全量读 DEM 但按地形处理下限抽稀 (out_shape), 防高分辨率 DEM 全量读内存爆。
        返回 (dem_float32, out_transform)。抽稀因子 dec = 下限/原生分辨率 (≥1)。"""
        from rasterio.transform import from_bounds as _trans_fb
        nat = min(abs(src_transform[0]), abs(src_transform[4]))
        floor = getattr(self, "terrain_proc_res_floor_m", 10.0) or 0.0
        dec = max(1, int(round(floor / nat))) if (nat > 0 and floor > 0) else 1
        out_w = max(1, src_width // dec)
        out_h = max(1, src_height // dec)
        dem = ds.read(1, out_shape=(out_h, out_w)).astype(np.float32)
        if out_w != src_width or out_h != src_height:
            out_transform = _trans_fb(*src_bounds, out_w, out_h)
        else:
            out_transform = src_transform
        return dem, out_transform

    def _process_terrain(self):
        """DEM 地形分析 (v0.4.3 重写)

        修复:
          - 问题 2 (v0.4): DEM 进入流程第一步就重投影到 working_crs
          - 问题 4 (v0.4): dem_quality.resolution_m 以 working_crs 单位 (米) 为准
          - 问题 5 (v0.4): nodata 识别扩展
          - v0.4.3 审核问题 1: DEM 按工作区 bbox + 500m 缓冲裁剪, 不再全量重投影
            (真实省级 DEM 下避免内存/耗时爆炸)
          - v0.4.3 审核小问题 3: slope 用 binary_dilation(invalid_mask,1) 扩张一像素,
            屏蔽 np.gradient 中心差分对 nodata 边界的污染
          - 隐藏 bug: TPI 计算前不再用 nanmean 填充, 改为 nan-aware 滑动平均
        """
        logger.info("处理DEM地形...")
        dem_entry = None
        for r in self.raster_inventory:
            if r.get("inferred_type") == "DEM":
                dem_entry = r
                break
        if not dem_entry:
            logger.info("无DEM数据，跳过地形分析")
            self.preprocessing_report["terrain_analysis"] = "skipped_no_dem"
            # ★Round 5 Bug C 修复★ 新增 dem_coverage 字段, 即便无 DEM 也写,
            # 让下游 manifest / 算法端能无条件读取覆盖率信息.
            self.preprocessing_report["dem_coverage"] = {
                "status": "no_dem",                # no_dem / not_cover / partial / full / failed
                "ratio_to_workspace": 0.0,         # DEM 与工作区相交面积比 (0-1)
                "covered_bbox_wcrs": None,         # DEM 在 working_crs 下的实际覆盖 bbox
                "workspace_bbox_wcrs": None,
                "recommended_mode": "2D_PLANAR",   # 算法端建议运行模式
            }
            return

        dem_path = dem_entry.get("abs_path")

        # ── v0.4.3: 推断 working bbox (如能), 避免整张省级 DEM 全量处理 ──
        from utils.bbox_infer import infer_work_bbox
        # 此时 M2 正在 run() 中间, control_objects 可从 self._m2_control_objects 取
        # (run() 在调用 _process_terrain 前已 set); m2_geoms 可以从 self 已产出的
        # forbidden/cost_polygons 取
        ctrl_objs = getattr(self, "_m2_control_objects", None) or {}
        m2_geoms = {
            "forbidden_polygons": self.forbidden_polygons,
            "no_tower_polygons": self.no_tower_polygons,
            "cost_polygons": self.cost_polygons,
            "linear_cross_segments": self.linear_cross_segments,
        }
        bbox_result = infer_work_bbox(
            project_config=self.config,
            control_objects=ctrl_objs,
            m2_geoms=m2_geoms,
            raster_inventory=self.raster_inventory,
        )
        work_bbox = bbox_result.get("bbox")
        bbox_src = bbox_result.get("source")
        if work_bbox is None:
            # 没任何工作区信息, 退回全量处理 + 警告 (真实数据下不应发生)
            logger.warning(
                "未能推断工作区 bbox, DEM 将被全量处理 (可能内存爆炸); "
                "建议补齐 project.json 的 bbox 或 control/start_end"
            )
        elif bbox_src == "dem":
            # 回退到 DEM 自身 bounds → 没有收益, 也提示用户
            logger.warning(
                "bbox 推断回退到 DEM 自身, DEM 将被全量处理; "
                "建议补齐 project.json 的 bbox 或 control/start_end 以启用裁剪"
            )
            work_bbox = None  # 语义统一: 没有来自工程数据的 bbox, 就全量
        else:
            logger.info(
                f"DEM 处理将按工作区 bbox 裁剪 (来源={bbox_src}, "
                f"缓冲={bbox_result.get('buffer_applied')}): {work_bbox}"
            )

        try:
            # ── 第一步: 读原生 DEM, 必要时重投影到 working_crs (带工作区裁剪) ──
            from pyproj import CRS as PyCRS
            from rasterio.warp import calculate_default_transform, reproject, Resampling

            with rasterio.open(dem_path) as ds:
                src_crs = ds.crs
                src_transform = ds.transform
                src_nodata = ds.nodata
                src_bounds = ds.bounds
                src_width = ds.width
                src_height = ds.height
                src_profile = ds.profile.copy()

                # 判断是否需要重投影: CRS 缺失 / 与 working_crs 不同 / 是地理坐标
                need_reproject = False
                if src_crs is None:
                    # M0 已经标记过 crs_fallback_used; 这里还是按 self.config.source_crs 兜底
                    fallback_crs = self.config.get("source_crs", "EPSG:4490")
                    logger.warning(
                        f"DEM 缺少 CRS, 兜底假设 {fallback_crs}"
                    )
                    src_crs = PyCRS.from_user_input(fallback_crs)
                    need_reproject = True
                else:
                    try:
                        if str(PyCRS.from_user_input(src_crs)) != \
                                str(PyCRS.from_user_input(self.working_crs)):
                            need_reproject = True
                    except Exception:
                        need_reproject = True

                if need_reproject:
                    # v0.4.3 审核问题 1 + v0.4.4 审核问题 3:
                    # 若有工作区 bbox, 只处理工作区 + 500m 缓冲
                    # 缓冲目的: 为 np.gradient 的中心差分 (±1 像素) 和 TPI 的
                    # uniform_filter (window 大小可变) 留出安全边界, 避免边界假值
                    crop_bbox_wcrs = None
                    if work_bbox is not None:
                        CROP_BUFFER_M = 500.0
                        crop_bbox_wcrs = (
                            work_bbox[0] - CROP_BUFFER_M,
                            work_bbox[1] - CROP_BUFFER_M,
                            work_bbox[2] + CROP_BUFFER_M,
                            work_bbox[3] + CROP_BUFFER_M,
                        )

                        # v0.4.4 审核问题 3: 检查 DEM 在 working_crs 下的范围是否
                        # 与工作区相交; 不相交则直接 severe, 不 fallback 全量
                        # (否则白读一张全 NaN 的 DEM, 浪费时间且给出错误结果)
                        dem_bounds_wcrs = self._transform_src_bounds_to_wcrs(
                            src_bounds, src_crs, self.working_crs
                        )
                        cov = self._bbox_coverage_ratio(
                            crop_bbox_wcrs, dem_bounds_wcrs
                        )
                        if cov["area_ratio"] < 0.05:
                            msg = (
                                f"DEM 范围与工作区相交面积比 {cov['area_ratio']:.1%} < 5%, "
                                f"DEM 不覆盖工程区 (DEM bounds in working_crs="
                                f"{dem_bounds_wcrs}, work_bbox={work_bbox})"
                            )
                            logger.error(msg)
                            self.preprocessing_report["terrain_analysis"] = (
                                "skipped_dem_not_cover_workspace"
                            )
                            self.preprocessing_report["dem_quality"] = {
                                "resolution_m": 0.0,
                                "nodata_ratio": 1.0,
                                "nodata_warning": True,
                                "resolution_warning": True,
                                "reprojected_to_working_crs": False,
                                "invalid_pixels_extended": 0,
                                "coverage_error": msg,
                                "severe": True,
                            }
                            # ★Round 5 Bug C 修复★ 同步写 dem_coverage 字段
                            self.preprocessing_report["dem_coverage"] = {
                                "status": "not_cover",
                                "ratio_to_workspace": round(float(cov["area_ratio"]), 4),
                                "covered_bbox_wcrs": list(dem_bounds_wcrs) if dem_bounds_wcrs else None,
                                "workspace_bbox_wcrs": list(crop_bbox_wcrs) if crop_bbox_wcrs else None,
                                "recommended_mode": "2D_PLANAR",
                            }
                            return
                        if cov["area_ratio"] < 0.5:
                            logger.warning(
                                f"DEM 只覆盖工作区 {cov['area_ratio']:.1%}, "
                                f"边缘区域将填 NaN 并标为 nodata"
                            )

                    logger.info(
                        f"DEM CRS={src_crs} → working_crs={self.working_crs}, "
                        f"执行重投影 (裁剪到工作区={'是' if crop_bbox_wcrs else '否/全量'})"
                    )
                    # 计算目标变换: 裁剪到工作区或全量
                    if crop_bbox_wcrs is not None:
                        # v0.4.4: 先把 crop_bbox 夹到 DEM 实际范围内, 避免分配
                        # 超出 DEM 的"无效像素" (工作区比 DEM 大时, 以前会多分配内存)
                        cb = (
                            max(crop_bbox_wcrs[0], dem_bounds_wcrs[0]),
                            max(crop_bbox_wcrs[1], dem_bounds_wcrs[1]),
                            min(crop_bbox_wcrs[2], dem_bounds_wcrs[2]),
                            min(crop_bbox_wcrs[3], dem_bounds_wcrs[3]),
                        )

                        # calculate_default_transform 的 dst_bounds 必须通过
                        # resolution 参数或 GDAL 风格 dstSRS/cutline 才能生效;
                        # 最稳方式: 先算全量默认分辨率, 然后用 bbox 手工裁窗口
                        dst_transform_full, _, _ = calculate_default_transform(
                            src_crs, self.working_crs,
                            src_width, src_height,
                            *src_bounds,
                        )
                        from rasterio.transform import from_bounds as trans_from_bounds
                        # 取 full 变换的分辨率 (米/像素)
                        full_res_x = abs(dst_transform_full[0])
                        full_res_y = abs(dst_transform_full[4])
                        # ★P6 (v0.6) 修复★ 不按 DEM 原生分辨率处理: 高分辨率 DEM(亚米)在工作区
                        #   范围内会产出数千万像素的数组, 光 bool 掩膜就内存爆。降采样到地形处理下限。
                        proc_res_x = max(full_res_x, self.terrain_proc_res_floor_m)
                        proc_res_y = max(full_res_y, self.terrain_proc_res_floor_m)
                        if proc_res_x > full_res_x or proc_res_y > full_res_y:
                            logger.info(
                                f"DEM 原生分辨率 ≈({full_res_x:.3f},{full_res_y:.3f})m 高于地形处理"
                                f"下限 {self.terrain_proc_res_floor_m}m, 降采样处理 (slope/TPI 对粗规划足够)"
                            )
                        dst_width = max(1, int(round((cb[2] - cb[0]) / proc_res_x)))
                        dst_height = max(1, int(round((cb[3] - cb[1]) / proc_res_y)))
                        dst_transform = trans_from_bounds(
                            cb[0], cb[1], cb[2], cb[3], dst_width, dst_height
                        )
                        logger.info(
                            f"DEM 裁剪到 {dst_width}×{dst_height} 像素 "
                            f"(原 DEM {src_width}×{src_height}, 节省 "
                            f"{100 * (1 - dst_width * dst_height / max(src_width * src_height, 1)):.1f}%)"
                        )
                    else:
                        # 全量 (回退路径)
                        dst_transform, dst_width, dst_height = calculate_default_transform(
                            src_crs, self.working_crs,
                            src_width, src_height,
                            *src_bounds,
                        )
                    dem = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
                    reproject(
                        source=rasterio.band(ds, 1),
                        destination=dem,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        src_nodata=src_nodata,
                        dst_transform=dst_transform,
                        dst_crs=self.working_crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
                    out_transform = dst_transform
                    res_x = abs(dst_transform[0])
                    res_y = abs(dst_transform[4])
                    out_profile = src_profile
                    out_profile.update({
                        "crs": self.working_crs,
                        "transform": dst_transform,
                        "width": dst_width,
                        "height": dst_height,
                        "dtype": "float32",
                        "count": 1,
                        "nodata": np.nan,
                    })
                else:
                    # DEM 已在 working_crs 下, 尽量用窗口读只载入工作区
                    if work_bbox is not None:
                        CROP_BUFFER_M = 500.0
                        cb = (
                            work_bbox[0] - CROP_BUFFER_M, work_bbox[1] - CROP_BUFFER_M,
                            work_bbox[2] + CROP_BUFFER_M, work_bbox[3] + CROP_BUFFER_M,
                        )
                        # v0.4.4 审核问题 3: 相交检查 (已同 CRS 路径也做)
                        dem_bounds_wcrs = tuple(src_bounds)
                        cov = self._bbox_coverage_ratio(cb, dem_bounds_wcrs)
                        if cov["area_ratio"] < 0.05:
                            msg = (
                                f"DEM 范围与工作区相交面积比 {cov['area_ratio']:.1%} < 5%, "
                                f"DEM 不覆盖工程区 (DEM bounds={dem_bounds_wcrs}, "
                                f"work_bbox={work_bbox})"
                            )
                            logger.error(msg)
                            self.preprocessing_report["terrain_analysis"] = (
                                "skipped_dem_not_cover_workspace"
                            )
                            self.preprocessing_report["dem_quality"] = {
                                "resolution_m": 0.0,
                                "nodata_ratio": 1.0,
                                "nodata_warning": True,
                                "resolution_warning": True,
                                "reprojected_to_working_crs": False,
                                "invalid_pixels_extended": 0,
                                "coverage_error": msg,
                                "severe": True,
                            }
                            # ★Round 5 Bug C 修复★ 同步写 dem_coverage 字段
                            self.preprocessing_report["dem_coverage"] = {
                                "status": "not_cover",
                                "ratio_to_workspace": round(float(cov["area_ratio"]), 4),
                                "covered_bbox_wcrs": list(dem_bounds_wcrs),
                                "workspace_bbox_wcrs": list(cb),
                                "recommended_mode": "2D_PLANAR",
                            }
                            return
                        if cov["area_ratio"] < 0.5:
                            logger.warning(
                                f"DEM 只覆盖工作区 {cov['area_ratio']:.1%}, "
                                f"边缘区域将填 NaN 并标为 nodata"
                            )
                        try:
                            from rasterio.windows import from_bounds as win_from_bounds
                            window = win_from_bounds(*cb, transform=src_transform)
                            window = window.round_lengths().round_offsets()
                            # 夹到栅格范围内
                            win_col_off = max(0, int(window.col_off))
                            win_row_off = max(0, int(window.row_off))
                            win_w = max(1, min(int(window.width),
                                               src_width - win_col_off))
                            win_h = max(1, min(int(window.height),
                                               src_height - win_row_off))
                            from rasterio.windows import Window
                            clipped = Window(win_col_off, win_row_off, win_w, win_h)
                            # ★P6 (v0.6) 修复★ 同 CRS 路径也降采样: DEM 已在 working_crs 但若为
                            #   高分辨率(亚米), 全分辨率窗口读会内存爆。按地形处理下限抽稀读(out_shape)。
                            nat_x = abs(src_transform[0]); nat_y = abs(src_transform[4])
                            dec_x = max(1, int(round(self.terrain_proc_res_floor_m / nat_x))) if nat_x > 0 else 1
                            dec_y = max(1, int(round(self.terrain_proc_res_floor_m / nat_y))) if nat_y > 0 else 1
                            out_w = max(1, win_w // dec_x)
                            out_h = max(1, win_h // dec_y)
                            dem = ds.read(1, window=clipped,
                                          out_shape=(out_h, out_w)).astype(np.float32)
                            win_bounds = rasterio.windows.bounds(clipped, src_transform)
                            if out_w != win_w or out_h != win_h:
                                from rasterio.transform import from_bounds as _trans_fb
                                out_transform = _trans_fb(*win_bounds, out_w, out_h)
                            else:
                                out_transform = rasterio.windows.transform(
                                    clipped, src_transform)
                            logger.info(
                                f"DEM 已在 working_crs 下, 按工作区窗口读取 "
                                f"{out_w}×{out_h} 像素 (窗口 {win_w}×{win_h}, 原 {src_width}×{src_height}, "
                                f"抽稀 {dec_x}×{dec_y})"
                            )
                        except Exception as e:
                            # v0.4.4: 窗口读取失败是技术性失败 (罕见), 这里保留
                            # fallback 全量; 但相交检查已在上面把"根本不相交"的
                            # 情形提前拒绝, 所以 fallback 全量不再是"用户填错 bbox"
                            # 的遮羞布
                            logger.warning(
                                f"工作区窗口裁剪失败, 回退全量读取 (相交检查已通过): {e}"
                            )
                            dem, out_transform = self._decimate_full_read(
                                ds, src_transform, src_bounds, src_width, src_height)
                    else:
                        dem, out_transform = self._decimate_full_read(
                            ds, src_transform, src_bounds, src_width, src_height)
                    # ★P6★ res 取自实际(可能已抽稀)的 out_transform, 不能用原生 ds.res
                    res_x = abs(out_transform[0])
                    res_y = abs(out_transform[4])
                    out_profile = src_profile
                    out_profile.update({
                        "dtype": "float32", "count": 1,
                        "transform": out_transform,
                        "width": dem.shape[1],
                        "height": dem.shape[0],
                        "nodata": float("nan"),
                    })

            # ── 第二步 (问题 5): 扩展 nodata 识别 ──
            invalid_mask = self._extended_nodata_mask(dem, src_nodata)
            dem_masked = dem.copy()
            dem_masked[invalid_mask] = np.nan

            dem_nodata_ratio = float(np.isnan(dem_masked).sum() / dem_masked.size)

            # ── 第三步: 坡度 (在米制 working_crs 下) ──
            # 对 NaN 做不污染的梯度: 先用 0 填充算 gradient
            dem_for_grad = np.nan_to_num(dem_masked, nan=0.0)
            dy, dx = np.gradient(dem_for_grad, res_y, res_x)
            slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
            # v0.4.3 审核小问题 3 修复: np.gradient 中心差分的模板宽度是 ±1 像素,
            # 所以 invalid 边界"外"的有效像素会因邻居是 0 被拉出假陡坡。
            # 把 invalid_mask 膨胀 1 像素正好匹配 gradient 的影响范围。
            from scipy.ndimage import binary_dilation
            slope_invalid = binary_dilation(invalid_mask, iterations=1)
            slope = np.where(slope_invalid, 0.0, slope).astype(np.float32)

            # ── 第四步: TPI (用带 nan 的 nanmean 滑动平均, 避免海岸线伪影) ──
            from scipy.ndimage import uniform_filter
            window_size = max(3, int(500 / min(res_x, res_y)))
            if window_size % 2 == 0:
                window_size += 1

            # 正确的 TPI: 有效区和无效区分别算邻域均值
            # 用 box-filter 技巧算 nan-aware 均值:
            # sum_valid = uniform_filter(dem_valid_filled, window) * window^2
            # count_valid = uniform_filter(~invalid_mask, window) * window^2
            # mean = sum_valid / count_valid (只在 count_valid>0 时)
            valid = (~invalid_mask).astype(np.float32)
            dem_valid_only = np.where(invalid_mask, 0.0, dem_masked).astype(np.float32)
            sum_local = uniform_filter(dem_valid_only, size=window_size, mode='constant', cval=0.0)
            count_local = uniform_filter(valid, size=window_size, mode='constant', cval=0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_elev = np.where(
                    count_local > 0.1,  # 窗口内至少 10% 的有效像素才算
                    sum_local / np.maximum(count_local, 1e-9),
                    np.nan,
                )
            tpi = dem_masked - mean_elev
            # 计算 valley/peak 前先用 0 填 nan 做阈值比较 (nan 与 <>= 的比较总是 False)
            # 单独保留一份用于写 TIF 的 nan 版本, 保证 profile.nodata=nan 与数据自洽。
            tpi_for_threshold = np.where(np.isnan(tpi), 0.0, tpi).astype(np.float32)

            tpi_threshold = 30.0
            valley_mask = (tpi_for_threshold < -tpi_threshold) & (~invalid_mask)
            peak_mask = (tpi_for_threshold > tpi_threshold) & (~invalid_mask)

            # v0.4.3 Bug B + 审核小问题 3: 写入磁盘时 invalid 像素(含 gradient 1px
            # 污染边界)必须用 nan (配合 profile.nodata=nan), 保证下游
            # resample_to_workspace 能把它们当 nodata 正确屏蔽。
            # slope 用膨胀 1px 的 mask, tpi 只需原始 invalid_mask (已用 nan-aware 滤波)。
            slope_out = np.where(slope_invalid, np.nan, slope).astype(np.float32)
            tpi_out = np.where(invalid_mask, np.nan, tpi_for_threshold).astype(np.float32)

            # ── 第五步: 输出栅格 (全部在 working_crs 下) ──
            m2_dir = ensure_dir(str(self.output_dir / "m2"))

            # mask 用 0/1 值 (uint8 不能放 nan), invalid 区 valley/peak 都是 0 (不是 valley),
            # 这在语义上正确 (不该立塔的地方本来就不算山谷)
            out_profile_mask = out_profile.copy()
            out_profile_mask.update(dtype="uint8", count=1, nodata=255)

            # float 输出: nodata=nan 与数据自洽
            out_profile_f = out_profile.copy()
            out_profile_f.update(dtype="float32", count=1, nodata=float("nan"))

            def _write_tif(arr, name, profile):
                with rasterio.open(os.path.join(m2_dir, name), "w", **profile) as dst:
                    dst.write(arr, 1)

            # ★P5 (v0.6)★ emit_unconsumed_outputs 总开关: 关闭时不写地形栅格 (算法端不消费;
            #   下方 slope_max/valley_pixels/dem_quality 等仍入 report, 供交付级别判定)。
            if self.emit_unconsumed_outputs:
                _write_tif(slope_out, "terrain_slope.tif", out_profile_f)
                _write_tif(tpi_out, "terrain_tpi.tif", out_profile_f)
                _write_tif(valley_mask.astype(np.uint8), "valley_mask.tif", out_profile_mask)
                _write_tif(peak_mask.astype(np.uint8), "peak_mask.tif", out_profile_mask)
            else:
                logger.info("emit_unconsumed_outputs=false, 跳过地形栅格写盘 (分析结果仍入 report)")

            # ── 第六步 (问题 4): 分辨率以米为单位 (此时已在 working_crs) ──
            dem_resolution_m = min(res_x, res_y)
            # 若 working_crs 是地理坐标 (不推荐但兼容处理), 换算成米。
            # v0.4 Bug C 修复: 中心纬度必须从输出栅格 (working_crs) 算,
            # 不能用 src_bounds — 如果源是投影 (米) 而 working 是地理, src_bounds[1]
            # 是百万级米值, 丢进 cos(radians) 会得到完全错误的 meters_per_deg。
            try:
                if PyCRS.from_user_input(self.working_crs).is_geographic:
                    import math
                    from rasterio.transform import array_bounds
                    out_h, out_w = dem.shape
                    out_bounds = array_bounds(out_h, out_w, out_transform)
                    # 输出是 working_crs=地理, out_bounds[1]/[3] 就是南/北纬
                    center_lat = (out_bounds[1] + out_bounds[3]) / 2
                    # 合理性夹逼: 中国大陆 ~15-55°N, 超过这个范围说明上游出问题, 不做换算
                    if not (-60 < center_lat < 75):
                        logger.warning(
                            f"working_crs 为地理坐标但输出中心纬度 {center_lat:.2f}° 异常, "
                            f"跳过米制换算, 保持 {dem_resolution_m:.6f}° 作为 resolution_m "
                            f"(下游阈值判定将失效, 请手动检查 DEM CRS)"
                        )
                    else:
                        meters_per_deg = 111320 * max(0.1, math.cos(math.radians(center_lat)))
                        dem_resolution_m = dem_resolution_m * meters_per_deg
                        logger.info(
                            f"working_crs 为地理坐标, 分辨率换算: "
                            f"{min(res_x, res_y):.6f}° @ lat {center_lat:.2f}° × "
                            f"{meters_per_deg:.1f} m/° ≈ {dem_resolution_m:.1f}m"
                        )
            except Exception as e:
                logger.debug(f"度→米换算失败, 保留原值: {e}")

            self.preprocessing_report["terrain_analysis"] = "completed"
            # nanmax 会自动跳过 nan 像素, 不受新的 nan 哨兵影响
            self.preprocessing_report["slope_max"] = (
                float(np.nanmax(slope_out)) if slope_out.size and np.any(~invalid_mask) else 0.0
            )
            self.preprocessing_report["valley_pixels"] = int(valley_mask.sum())
            self.preprocessing_report["peak_pixels"] = int(peak_mask.sum())
            self.preprocessing_report["dem_quality"] = {
                "nodata_ratio": round(dem_nodata_ratio, 4),
                "resolution_m": round(dem_resolution_m, 2),
                "nodata_warning": dem_nodata_ratio > 0.05,
                "resolution_warning": dem_resolution_m > 30,
                "reprojected_to_working_crs": need_reproject,
                "invalid_pixels_extended": int(invalid_mask.sum()),
            }
            # ★Round 5 Bug C 修复★ 完成分支也写 dem_coverage
            # 覆盖率从 nodata_ratio 反推: 工作区 + 500m 缓冲范围内,
            # 非 nan 像素占比 ≈ DEM 真实覆盖工作区比例
            effective_coverage = max(0.0, min(1.0, 1.0 - dem_nodata_ratio))
            if effective_coverage >= 0.95:
                coverage_status = "full"
                recommended_mode = "3D"
            elif effective_coverage >= 0.5:
                coverage_status = "partial"
                # 部分覆盖: 算法端要么走"混合 3D/2D" (需读 dem_coverage 字段),
                # 要么按 allow_planar_2d_mode 整体降级. 这里给出"混合"建议.
                recommended_mode = "3D_PARTIAL"
            else:
                coverage_status = "partial_low"
                # 覆盖率太低, 3D 模式信息量不足, 建议直接降 2D
                recommended_mode = "2D_PLANAR"
            self.preprocessing_report["dem_coverage"] = {
                "status": coverage_status,
                "ratio_to_workspace": round(effective_coverage, 4),
                # 计算 DEM 实际覆盖的有效 bbox (排除边缘 nan)
                "covered_bbox_wcrs": self._compute_dem_effective_bbox(
                    invalid_mask, out_transform
                ) if not invalid_mask.all() else None,
                "workspace_bbox_wcrs": list(work_bbox) if work_bbox else None,
                "recommended_mode": recommended_mode,
            }
        except Exception as e:
            logger.error(f"DEM地形分析失败: {e}", exc_info=True)
            self.preprocessing_report["terrain_analysis"] = f"error: {e}"
            # ★Round 5 Bug C 修复★ error 分支也写 dem_coverage, 保 schema 一致
            self.preprocessing_report["dem_coverage"] = {
                "status": "failed",
                "ratio_to_workspace": 0.0,
                "covered_bbox_wcrs": None,
                "workspace_bbox_wcrs": None,
                "recommended_mode": "2D_PLANAR",
                "error": str(e)[:300],
            }

    @staticmethod
    def _extended_nodata_mask(dem: np.ndarray, declared_nodata) -> np.ndarray:
        """
        v0.4 问题 5: 扩展 nodata 识别, 命中任一即为 nodata/invalid。

        规则:
          - 已声明 nodata
          - 已是 nan
          - dem < -1000 (海洋常填 <0, -1000 是保守阈值)
          - dem > 9000 (珠峰 8849m 以上基本都是垃圾值)
          - abs(dem) > 1e6 (3.4e38 类浮点极值)
          - inf / -inf
        """
        mask = np.isnan(dem) | np.isinf(dem)
        if declared_nodata is not None:
            try:
                nd = float(declared_nodata)
                mask |= (dem == nd)
            except (TypeError, ValueError):
                pass
        mask |= (dem < -1000)
        mask |= (dem > 9000)
        mask |= (np.abs(dem) > 1e6)
        return mask

    @staticmethod
    def _transform_src_bounds_to_wcrs(src_bounds, src_crs, working_crs):
        """v0.4.4: 把 DEM 源 CRS 下的 bounds 转到 working_crs (米制)"""
        try:
            from rasterio.warp import transform_bounds
            from pyproj import CRS as _CRS
            src = _CRS.from_user_input(src_crs) if not isinstance(src_crs, _CRS) else src_crs
            dst = _CRS.from_user_input(working_crs) if not isinstance(working_crs, _CRS) else working_crs
            if src == dst:
                return tuple(src_bounds)
            return transform_bounds(
                src, dst,
                src_bounds[0], src_bounds[1], src_bounds[2], src_bounds[3],
                densify_pts=21,
            )
        except Exception:
            return tuple(src_bounds)

    @staticmethod
    def _bbox_coverage_ratio(work_bbox, dem_bbox) -> dict:
        """
        v0.4.4 审核问题 3: 计算 DEM bbox 与工作区 bbox 的覆盖关系。

        Returns:
            {
              "intersect": bool,               # 是否有交集
              "intersect_area": float,         # 交集面积 (米²)
              "work_area": float,              # 工作区面积
              "area_ratio": float,             # 交集 / 工作区 (0 = 完全不覆盖, 1 = 完全覆盖)
            }
        """
        ix_min = max(work_bbox[0], dem_bbox[0])
        iy_min = max(work_bbox[1], dem_bbox[1])
        ix_max = min(work_bbox[2], dem_bbox[2])
        iy_max = min(work_bbox[3], dem_bbox[3])
        if ix_max <= ix_min or iy_max <= iy_min:
            return {"intersect": False, "intersect_area": 0.0,
                    "work_area": max(0.0, (work_bbox[2] - work_bbox[0])
                                          * (work_bbox[3] - work_bbox[1])),
                    "area_ratio": 0.0}
        intersect_area = (ix_max - ix_min) * (iy_max - iy_min)
        work_area = max(1.0, (work_bbox[2] - work_bbox[0])
                              * (work_bbox[3] - work_bbox[1]))
        return {
            "intersect": True,
            "intersect_area": float(intersect_area),
            "work_area": float(work_area),
            "area_ratio": float(intersect_area / work_area),
        }

    @staticmethod
    def _compute_dem_effective_bbox(invalid_mask, transform):
        """
        ★Round 5 Bug C★ 计算 DEM 实际覆盖的有效 bbox (排除边缘 nan 区).
        
        invalid_mask: bool 数组, True 表示 nodata 像素
        transform: rasterio Affine, 把像素坐标映射到 working_crs
        Returns: [xmin, ymin, xmax, ymax] in working_crs, 或 None (全是 nan)
        """
        import numpy as np
        valid = ~invalid_mask
        if not valid.any():
            return None
        # 找有效区的行/列范围
        rows = np.where(valid.any(axis=1))[0]
        cols = np.where(valid.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return None
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        # 把像素坐标 (含 row_max+1, col_max+1 表示像素边界外侧) 转 working_crs 坐标
        # transform * (col, row) -> (x, y), 注意行号增加 = y 减少 (北上南下)
        x_min, y_top = transform * (col_min, row_min)
        x_max, y_bot = transform * (col_max + 1, row_max + 1)
        # 排序保证 ymin < ymax
        return [
            float(min(x_min, x_max)),
            float(min(y_top, y_bot)),
            float(max(x_min, x_max)),
            float(max(y_top, y_bot)),
        ]

    # ─── 8.8 控制对象处理 ──────────────────────────────────
    def _process_control_objects(self, control_objects: dict):
        logger.info("处理控制对象...")
        self.preprocessing_report["control_objects"] = {
            k: len(v) for k, v in control_objects.items()
        }

    # ─── 输出 ──────────────────────────────────────────────
    def _export(self, control_objects: dict) -> dict:
        m2_dir = ensure_dir(str(self.output_dir / "m2"))

        if self.buffered_points:
            gpd.GeoDataFrame(self.buffered_points, crs=self.working_crs).to_file(
                os.path.join(m2_dir, "buffered_points.gpkg"), driver="GPKG")
        if self.linear_cross_segments:
            gpd.GeoDataFrame(self.linear_cross_segments, crs=self.working_crs).to_file(
                os.path.join(m2_dir, "linear_cross_indexed.gpkg"), driver="GPKG")
        if self.building_clusters_gdf is not None:
            self.building_clusters_gdf.to_file(
                os.path.join(m2_dir, "building_clusters.gpkg"), driver="GPKG")
        if self.river_crossing_windows is not None:
            self.river_crossing_windows.to_file(
                os.path.join(m2_dir, "river_crossing_windows.gpkg"), driver="GPKG")
        if self.wide_river_barriers is not None:
            self.wide_river_barriers.to_file(
                os.path.join(m2_dir, "wide_river_barriers.gpkg"), driver="GPKG")
        # v0.4.3 审核问题 4 修复: manifest 要求这 3 个图层必须存在
        # 即便某工程完全没有禁区/禁立塔/高代价面 (空工程), 也要写占位 gpkg,
        # 否则 manifest 会把它记为 required_missing 误触发 SEVERE_DEGRADED。
        # 占位文件在 schema 里加 _placeholder=True, 算法端能分辨真数据 vs 占位。
        self._write_polygon_gpkg_or_placeholder(
            self.forbidden_polygons, "forbidden_polygons.gpkg", m2_dir
        )
        self._write_polygon_gpkg_or_placeholder(
            self.no_tower_polygons, "no_tower_polygons.gpkg", m2_dir
        )
        self._write_polygon_gpkg_or_placeholder(
            self.cost_polygons, "cost_polygons.gpkg", m2_dir
        )
        if self.preferred_corridors:
            pc_data = []
            for c in self.preferred_corridors:
                pc_data.append({
                    "geometry": c["corridor_geometry"],
                    "level2": c["level2"],
                    "parallel_reward": c["parallel_reward"],
                    "voltage_kv": c.get("voltage_kv", 0),
                    "parallel_range_m": c.get("parallel_range_m", 200),
                    "buffer_m": c.get("buffer_m", 0),
                    # ★B2 审计字段★ 算法端可读, 用于"软避让 eco"决策
                    "eco_overlap_ratio": c.get("eco_overlap_ratio", 0.0),
                    "eco_overlap_area": c.get("eco_overlap_area", 0.0),
                })
            gdf_pc = gpd.GeoDataFrame(pc_data, crs=self.working_crs)
            from utils.geo_utils import write_gdf_to_gpkg_safe
            write_gdf_to_gpkg_safe(gdf_pc, os.path.join(m2_dir, "preferred_corridors.gpkg"),
                                     "preferred_corridors")
        from utils.geo_utils import write_gdf_to_gpkg_safe
        for key, gdf in control_objects.items():
            write_gdf_to_gpkg_safe(gdf, os.path.join(m2_dir, f"control_{key}.gpkg"),
                                     f"control_{key}")
        if self.processed_polygons:
            gdf_pp = gpd.GeoDataFrame(self.processed_polygons, crs=self.working_crs)
            write_gdf_to_gpkg_safe(gdf_pp, os.path.join(m2_dir, "processed_polygons.gpkg"),
                                     "processed_polygons")
        
        # ★v0.5 (B2)★ 几何修复日志: 用于审计甲方 _保护范围/_奖励范围 变体面的拓扑修复事件
        if self.geometry_fixes_log:
            try:
                from utils.geo_utils import save_json
                save_json({"fixes": self.geometry_fixes_log,
                           "summary": {
                               "total": len(self.geometry_fixes_log),
                               "discarded": sum(1 for f in self.geometry_fixes_log if f.get("discarded")),
                           }},
                          os.path.join(m2_dir, "geometry_fixes_log.json"))
                logger.info(f"几何修复事件: {len(self.geometry_fixes_log)} 条 (审计在 m2/geometry_fixes_log.json)")
            except Exception as e:
                logger.warning(f"写出 geometry_fixes_log.json 失败: {e}")

        logger.info(f"M2输出: 禁区={len(self.forbidden_polygons)}, "
                    f"禁立塔={len(self.no_tower_polygons)}, "
                    f"高代价={len(self.cost_polygons)}, "
                    f"线状分段={len(self.linear_cross_segments)}")

        return {
            "forbidden_polygons": self.forbidden_polygons,
            "no_tower_polygons": self.no_tower_polygons,
            "cost_polygons": self.cost_polygons,
            "linear_cross_segments": self.linear_cross_segments,
            "preferred_corridors": self.preferred_corridors,
            "building_clusters_gdf": self.building_clusters_gdf,
            "river_crossing_windows": self.river_crossing_windows,
            "wide_river_barriers": self.wide_river_barriers,
            "control_objects": control_objects,
            "preprocessing_report": self.preprocessing_report,
            "raster_inventory": self.raster_inventory,
            # v0.2: eco_sensitive_polygons 只在 M2 内部给
            # _process_preferred_corridors/parallel_valid_zone 用, 未被 M3 或下游消费,
            # 不对外返回, 避免给下游造成"有这个契约字段"的误解。
        }
