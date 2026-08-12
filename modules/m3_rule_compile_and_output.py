"""
M3: 规则编译 + 标准化输出包 + 混合导航图骨架构建 (v5.3 fixed)

v5.3 关键修复:
  P0-1: 一次性跨越/立塔代价从LPCF中剥离
  P0-6: 平行贴近规则统一200m + 800kV禁止并行区
  v5.3-1: 高代价区LPCF引导值按代价大小分级（解决问题清单2.6）
  v5.3-2: 骨架边cutoff距离自适应（解决50km范围下连通性不足）
  v5.3-3: batch_rasterize_max 批量栅格化提升性能
"""
import os
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, geometry_mask
from rasterio.enums import MergeAlg  # ★Tier1-1.1★ 用于批量 add 模式栅格化,与逐个减法完全等价
from rasterio.transform import from_bounds
from shapely.geometry import Point, LineString, box, mapping
from shapely.ops import unary_union  # ★P4 修复★ _compute_vector_extent 矢量并集 (真实数据才触发)
from shapely.strtree import STRtree

from utils.geo_utils import (load_json, save_json, ensure_dir, get_config_dir,
                              azimuth_deg, get_base_cross_cost, evaluate_land_cost,
                              is_one_time_cost, batch_rasterize_max,
                              parallel_range_m_for, resample_to_workspace,
                              decompose_cost_fields)

logger = logging.getLogger("transmission_planning.m3")

# 默认求解器参数
DEFAULT_SOLVER_PARAMS = {
    "corridor_top_k": 5,
    "corridor_overlap_max": 0.5,
    "max_turn_angle_deg": 90,
    "weighted_astar_epsilon": 1.5,
    "ara_initial_epsilon": 3.0,
    "max_local_repair_attempts": 5,
    "tower_feedback_iterations": 3,
    "major_river_threshold_m": 900,
    "must_path_coverage_ratio_min": 0.8,
    "must_path_max_deviation_m": 500,
    "building_cluster_buffer_m": 100,
    "building_cluster_merge_gap_m": 50,
    "tpi_valley_threshold": 30.0,
    "parallel_proximity_m": 200,
    "parallel_min_angle_deg": 15,
    "parallel_min_length_m": 500,
    "corridor_half_width_normal_m": 225,
    "corridor_half_width_high_cost_m": 300,
    "compensation_cost_per_crossing_wan": 5.0,
    "dem_nodata_ratio_threshold": 0.05,
    "dem_resolution_threshold_m": 30,
    # ─── v0.5 新增设计规则 ─────────────────────────────
    "min_tower_spacing_m": 100,
    "max_no_tower_span_m": 1000,
    "avg_span_normal_m": 450,
    "avg_span_costly_m": 600,
    "dense_corridor_max_angle_deg": 15,
    "dense_corridor_min_length_m": 500,
    "high_voltage_parallel_exclusion_m": 600,
    "high_voltage_parallel_threshold_kv": 800,
    # 方案数量: >=30km输出3-5个, 5-30km输出2个
    "scheme_count_long_route_min": 3,
    "scheme_count_long_route_max": 5,
    "scheme_count_short_route": 2,
    "long_route_threshold_km": 30,
    # ─── v5.4 新增性能和反馈参数 ─────────────────────────
    # M5->M4 反馈: patch 注入 LPCF 时使用的代价值
    "feedback_patch_cost_value": 8.0,
    # M5->M4 反馈: patch 的几何缓冲(米), 避免仅点级别影响
    "feedback_patch_buffer_m": 150,
    # M4 合规修复的接受准则: 修复后总代价最多允许的相对增加比例
    "repair_accept_relative_worsen_max": 0.20,
    # M4 修复失败后是否触发走廊内重规划
    "repair_corridor_replan_enabled": True,
    # M4 走廊内重规划最多轮数
    "repair_corridor_replan_max_rounds": 1,
    # M4 ARA*: 早停目标 epsilon (达到后不再继续收紧, 显著提速)
    "ara_star_target_epsilon": 1.2,
    # M4 ARA*: 单次加权搜索最大展开数 (安全阀)
    "ara_star_max_expand_per_epsilon": 200000,
    # M4 使用预计算 tower_difficulty_50m (v5.4), 否则回退到在线采样
    "tower_difficulty_use_precomputed": True,
    # ─── v5.5 新增参数 ─────────────────────────────────
    # 每走廊保留 Top-K 差异化路径
    "corridor_topk_paths": 2,
    # Top-K 路径差异度阈值(Hausdorff 距离, 米)
    "corridor_topk_min_hausdorff_m": 300,
    # 局部精化窗口半径(米)
    "refine_radius_cross_m": 1000,       # 跨越段
    "refine_radius_big_turn_m": 500,     # 大转角(>60°)
    "refine_radius_turn_m": 300,         # 普通转角(>30°)
    # ARA* Anytime 标准实现: 使用 OPEN/INCONS/CLOSED 三集合
    "ara_anytime_enabled": True,
    # ─── v5.7 新增参数 (R4-1) ─────────────────────────
    # 多段 Top-K beam search 的 beam 宽度 (每段保留的中间候选数)
    "corridor_topk_beam_width": 4,
    # ─── v0.2 骨架性能重构参数 ─────────────────────────
    # 骨架节点总数预算 (超过则自动收紧 dense_step 再来一轮)
    "skeleton_budget_nodes": 12000,
    # 骨架边候选对数预算 (超过则自动收紧 cutoff)
    "skeleton_edge_candidate_budget": 600000,
    # cutoff 密度因子 (cutoff = median_nn_dist × factor, 夹在 [min, max] 区间)
    "skeleton_cutoff_nn_factor": 4.0,
    "skeleton_cutoff_min_m": 500.0,
    "skeleton_cutoff_max_m": 1500.0,
    # 骨架构建模式: "immediate" | "deferred" | "skip"
    #   immediate: M3 同步构建骨架 (默认, 与 v5.7 行为一致)
    #   deferred:  M3 跳过, 但在 manifest.nav_graph.status 标记 "deferred",
    #              算法端首次使用时自行构建 (需算法端实现懒加载)
    #   skip:      彻底跳过, 不构建也不标记可构建 (仅用于栅格级算法)
    # ★P0 (v0.6)★ 默认改 "skip": 算法端不消费 nav_graph (它自己重栅格化 GPKG),
    #   默认不建骨架可省 M3 数分钟。需要时显式设 "immediate"/"deferred"。
    #   ★P5★ emit_unconsumed_outputs 对骨架的统一门控已接入 _build_nav_graph_skeleton。
    "skeleton_build_mode": "skip",
    # ─── v0.4 新增 bbox 决定相关参数 (问题 3/8/12) ─────
    # 基于起终点推断 bbox 时的默认缓冲 (km)
    # ★P0 (v0.6)★ 20→5: 起终点窗口远小于数据范围时, 5km 已够给算法寻路余量,
    #   避免小地图被额外膨胀。裁到数据范围的逻辑在 P4 接入。
    "bbox_start_end_buffer_km": 5.0,
    # 矢量 bbox 推断时的飞地过滤半径 (km);
    # 距起终点中心超过此距离的要素视为飞地, 不参与 bbox 计算
    "bbox_enclave_filter_km": 150.0,
}


class M3RuleCompiler:
    """M3 规则编译器 + 标准化输出包构建"""

    def __init__(self, project_config: dict, output_dir: str):
        self.config = project_config
        self.output_dir = Path(output_dir)
        self.working_crs = project_config.get("working_crs", "EPSG:4547")
        self.data_avail = project_config.get("data_availability", {})
        self.voltage_kv = project_config.get("voltage_kv", 500)

        rules_path = os.path.join(get_config_dir(), "default_feature_rules.json")
        rules_data = load_json(rules_path)
        self.rule_by_id = {f["id"]: f for f in rules_data["features"]}

        self.bbox = None
        # ★P4 (v0.6)★ bbox 裁剪遥测 (写入 workspace.json 便于排查)
        self._bbox_clipped = False
        self._bbox_raw = None
        # ★Tier1 优化 2.2★ fine_res/coarse_res 改为配置项,
        # 默认仍是 50m/10m (向后兼容). project.json 可在 solver_params 块下覆盖:
        #   "solver_params": { "fine_resolution_m": 15, "coarse_resolution_m": 50 }
        # 当前阶段算法端走平面 2D 选线 (无 DEM, 无塔位精细优化),
        # 把 fine_res 调到 15m 可让 M3 10m 栅格化阶段 (2.5 亿格)
        # 像素数减少 ~56% (1.1 亿格), M3 时间预计省 8-15 分钟.
        # ⚠️ 注意: 等算法端推进到塔位精细优化阶段, 必须切回 10m, 否则塔位精度受损.
        solver_params_cfg = project_config.get("solver_params", {}) or {}
        # ★P0 (v0.6)★ 粗分辨率默认 50→100, 与算法端 cell_size=100 对齐 (Q3-A)。
        self.coarse_res = float(solver_params_cfg.get("coarse_resolution_m", 100))
        self.fine_res = float(solver_params_cfg.get("fine_resolution_m", 10))
        # ★P0 (v0.6)★ 两个输出开关 (缺省安全, 向后兼容)。
        #   enable_fine_resolution: 是否产 *_fine.tif (默认 false; 消费在 P3)
        #   emit_unconsumed_outputs: 是否产算法端不消费的栅格/骨架 (默认 false; ★P5★ 门控已接入)
        # 本阶段(P0)仅读取并存储, 供 manifest 动态必需清单使用;
        # ★P5★ 栅格/骨架产出门控已接入 (_build_raster_layers / _build_nav_graph_skeleton);
        self.enable_fine_resolution = bool(
            solver_params_cfg.get("enable_fine_resolution", False))
        self.emit_unconsumed_outputs = bool(
            solver_params_cfg.get("emit_unconsumed_outputs", False))
        if self.fine_res != 10:
            logger.warning(
                f"★Tier1 优化 2.2★ fine_resolution_m 已从 project.json 覆盖为 "
                f"{self.fine_res}m (默认 10m). 塔位精度受影响, 仅适用于当前平面 2D 阶段."
            )

        # v0.3: 运行时降级原因收集器
        # 各子模块通过 self._record_degrade(code, severity, detail) 追加,
        # _determine_delivery_level 会把它透传到 report.delivery_level 里,
        # manifest 层再据此综合出 FORMAL/PRELIMINARY/SEVERE 三档。
        # 这是替代旧"except: pass 静默吞异常"的统一通道。
        # severity: "severe" => 触发 SEVERE_DEGRADED
        #           "degraded" => 触发 PRELIMINARY_ROUTE_ONLY
        #           其它     => 仅记录, 不影响级别
        self._runtime_degrades: list = []

        # v0.4 问题 11: 骨架元数据供 _determine_delivery_level 使用
        # 由 run() 在 _build_nav_graph_skeleton 后 set
        self._nav_graph_meta: Optional[dict] = None

    def _record_degrade(self, code: str, severity: str, detail: str = "",
                        where: str = "") -> None:
        """记录一条运行时降级原因。

        Args:
            code: 机读编码 (如 VALLEY_RESAMPLE_FAILED / SLOPE_RESAMPLE_FAILED /
                  WIND_ICE_RESAMPLE_FAILED / MULTI_TIER_BUFFER_FAILED)
            severity: "severe" | "degraded" | "info"
            detail: 人读描述 (通常是异常消息)
            where: 代码定位 (可选, 如 "_build_rasters_at_resolution/valley")
        """
        entry = {"code": code, "severity": severity,
                 "detail": str(detail)[:500], "where": where}
        self._runtime_degrades.append(entry)
        # 同时打日志, 防止被遗漏
        log_msg = (f"[M3 降级] code={code} severity={severity} "
                   f"where={where or '-'} detail={detail}")
        if severity == "severe":
            logger.error(log_msg)
        elif severity == "degraded":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def run(self, m2_output: dict) -> dict:
        logger.info("===== M3: 规则编译+标准化输出包构建 =====")
        self._determine_bbox(m2_output)
        self._build_raster_layers(m2_output)

        rule_config = self._compile_rule_config(m2_output)
        solver_params = self._compile_solver_params()
        workspace = self._compile_workspace(m2_output)
        # v0.4 问题 11 修复: 骨架先于 report 构建,
        # 这样 _determine_delivery_level 能看到骨架节点/边数, 判定是否严重降级
        nav_graph_meta = self._build_nav_graph_skeleton(m2_output)
        self._nav_graph_meta = nav_graph_meta  # 给 _determine_delivery_level 使用
        preprocessing_report = self._compile_preprocessing_report(m2_output)

        result = self._export(
            rule_config, solver_params, workspace,
            preprocessing_report, nav_graph_meta
        )
        logger.info("M3完成")
        return result

    # ─── 工作区范围 ────────────────────────────────────────
    def _determine_bbox(self, m2_output: dict):
        """工作区 bbox 决定 (v0.4 重写)

        新的优先级 (解决问题 3 + 8 + 12):
          1) project.json 的 bbox (用户显式, 最高优先级)
          2) 起终点 (start_end) + 缓冲 (最可靠, 工程端点真实表达)
          3) must_path / must_pass + 缓冲
          4) 矢量并集 (带飞地过滤 + 5% 外缓冲)
          5) DEM bounds_in_working_crs (最后备选, 仅在其他全无时使用)

        相比 v0.3 的改进:
          - 起终点不再仅是"备选", 而是工程范围最权威的来源
          - DEM 分支不再裸用原生 CRS bounds, 而是使用 M0 预先计算的 bounds_in_working_crs
          - 矢量并集前先过滤"飞地" (距起终点中心 > bbox_enclave_filter_km 的要素)
          - DEM 分支也加 5% 缓冲 (与矢量分支对齐)
        """
        # 1) 用户显式 bbox
        if "bbox" in self.config:
            self.bbox = self.config["bbox"]
            logger.info(f"工作区范围(project.json): {self.bbox}")
            return

        # 读取用户可调参数
        solver_p = self.config.get("solver_params", {}) or {}
        buf_km = float(solver_p.get("bbox_start_end_buffer_km", 5.0))
        enclave_km = float(solver_p.get("bbox_enclave_filter_km", 150.0))
        buf_m = buf_km * 1000.0

        control_objs = m2_output.get("control_objects", {}) or {}

        # 2) 起终点 + 缓冲
        start_end_bbox = self._bbox_from_control(control_objs.get("start_end"))
        must_path_bbox = self._bbox_from_control(control_objs.get("must_path"))
        must_pass_bbox = self._bbox_from_control(control_objs.get("must_pass"))

        if start_end_bbox:
            merged = self._merge_bboxes(
                [b for b in [start_end_bbox, must_path_bbox, must_pass_bbox] if b]
            )
            expanded = self._expand_bbox(merged, buf_m, 0.0)
            # ★P4★ 裁到数据范围 (端点保护: 至少含起终点窗口 merged)
            self.bbox = self._maybe_clip_bbox(
                expanded, merged, m2_output, solver_p,
                src_desc=f"起终点+缓冲{buf_km}km")
            return

        # 3) 只有 must_path/must_pass 没有 start_end 的情况
        if must_path_bbox or must_pass_bbox:
            merged = self._merge_bboxes(
                [b for b in [must_path_bbox, must_pass_bbox] if b]
            )
            expanded = self._expand_bbox(merged, buf_m, 0.0)
            # ★P4★ 同样裁到数据范围
            self.bbox = self._maybe_clip_bbox(
                expanded, merged, m2_output, solver_p,
                src_desc=f"必经点/必经路径+缓冲{buf_km}km")
            return

        # 4) 矢量并集 (飞地过滤)
        vec_bbox, filtered_count = self._bbox_from_vectors(
            m2_output, enclave_ref_bbox=start_end_bbox, enclave_km=enclave_km
        )
        if vec_bbox:
            self.bbox = self._expand_bbox(vec_bbox, 0.0, 0.05)
            if filtered_count > 0:
                logger.info(
                    f"工作区范围(矢量推断, 过滤 {filtered_count} 个飞地要素, +5%缓冲): {self.bbox}"
                )
            else:
                logger.info(f"工作区范围(矢量推断, +5%缓冲): {self.bbox}")
            return

        # 5) DEM bounds_in_working_crs (最后备选)
        for r in m2_output.get("raster_inventory", []):
            if r.get("inferred_type") == "DEM":
                # v0.4 问题 2 + 8 修复: 优先用 bounds_in_working_crs, 加缓冲
                b = r.get("bounds_in_working_crs") or r.get("bounds")
                if b is None or len(b) != 4:
                    continue
                src = "bounds_in_working_crs" if r.get("bounds_in_working_crs") else "bounds(原生CRS, 可能单位不匹配!)"
                self.bbox = self._expand_bbox(tuple(b), 0.0, 0.05)
                logger.warning(
                    f"工作区范围(DEM {src}, +5%缓冲): {self.bbox} "
                    f"[未提供起终点, 建议补齐 project.json 的 start_point/end_point 或 control/start_end.geojson]"
                )
                return

        # 6) 彻底兜底
        self.bbox = [0, 0, 50000, 50000]
        logger.warning("无法确定工作区范围，使用默认值 [0,0,50000,50000]")

    @staticmethod
    def _bbox_from_control(gdf) -> Optional[Tuple[float, float, float, float]]:
        """从单个控制对象 GeoDataFrame 提取 total_bounds (工程 CRS), 空或无效返回 None"""
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

    @staticmethod
    def _merge_bboxes(bboxes: list) -> Tuple[float, float, float, float]:
        """合并多个 bbox 取并集"""
        xmin = min(b[0] for b in bboxes)
        ymin = min(b[1] for b in bboxes)
        xmax = max(b[2] for b in bboxes)
        ymax = max(b[3] for b in bboxes)
        return (xmin, ymin, xmax, ymax)

    @staticmethod
    def _expand_bbox(bbox, fixed_m: float, ratio: float) -> list:
        """扩展 bbox: fixed_m 是绝对外扩 (米), ratio 是相对外扩 (如 0.05 = 5%)"""
        dx = (bbox[2] - bbox[0]) * ratio + fixed_m
        dy = (bbox[3] - bbox[1]) * ratio + fixed_m
        # 防止 degenerate 情况 (例如只有一个点), 至少给 100m 半径
        if dx < 100:
            dx = max(dx, 100.0)
        if dy < 100:
            dy = max(dy, 100.0)
        return [bbox[0] - dx, bbox[1] - dy, bbox[2] + dx, bbox[3] + dy]

    def _bbox_from_vectors(self, m2_output: dict,
                           enclave_ref_bbox: Optional[Tuple[float, float, float, float]],
                           enclave_km: float) -> Tuple[Optional[Tuple], int]:
        """
        从 M2 产出的几何集合推断 bbox, 带飞地过滤 (问题 12)。

        过滤规则: 若提供了起终点参考 bbox, 计算其中心点,
        过滤掉距中心 > enclave_km 的所有几何。
        无参考时不做过滤 (退化为旧行为)。
        """
        all_geoms = []
        for key in ["forbidden_polygons", "no_tower_polygons", "cost_polygons"]:
            for item in m2_output.get(key, []):
                if item.get("geometry") and not item["geometry"].is_empty:
                    all_geoms.append(item["geometry"])
        for seg in m2_output.get("linear_cross_segments", []):
            if seg.get("geometry") and not seg["geometry"].is_empty:
                all_geoms.append(seg["geometry"])
        for key, gdf in (m2_output.get("control_objects", {}) or {}).items():
            try:
                if len(gdf) > 0:
                    all_geoms.extend(g for g in gdf.geometry.tolist()
                                     if g is not None and not g.is_empty)
            except Exception:
                continue

        if not all_geoms:
            return None, 0

        filtered_count = 0
        if enclave_ref_bbox is not None:
            # 以起终点 bbox 中心为参考
            cx = (enclave_ref_bbox[0] + enclave_ref_bbox[2]) / 2
            cy = (enclave_ref_bbox[1] + enclave_ref_bbox[3]) / 2
            radius_m = enclave_km * 1000.0
            filtered = []
            for g in all_geoms:
                try:
                    b = g.bounds
                    # 若几何 bbox 距中心 > radius, 认为是飞地
                    gcx = (b[0] + b[2]) / 2
                    gcy = (b[1] + b[3]) / 2
                    if (gcx - cx) ** 2 + (gcy - cy) ** 2 > radius_m ** 2:
                        filtered_count += 1
                        continue
                    filtered.append(g)
                except Exception:
                    filtered.append(g)  # 出错了就保留, 不激进
            if not filtered and all_geoms:
                # 全部被过滤掉说明起终点 bbox 离工程数据很远, 不做过滤回退到原逻辑
                logger.warning(
                    f"飞地过滤把全部 {len(all_geoms)} 个几何都剔除了, "
                    f"起终点 bbox {enclave_ref_bbox} 可能与工程数据不匹配, 回退不过滤"
                )
                filtered = all_geoms
                filtered_count = 0
            all_geoms = filtered

        try:
            combined = unary_union(all_geoms)
            return tuple(float(v) for v in combined.bounds), filtered_count
        except Exception as e:
            logger.warning(f"矢量 bbox 合并失败: {e}")
            return None, filtered_count

    def _compute_data_extent(self, m2_output: dict):
        """★P4 (v0.6)★ 输入数据实际范围 = 矢量并集 bounds ∪ 各栅格 bounds_in_working_crs。
        用于把"起终点+缓冲"外扩出的 bbox 裁回数据范围 (R3 / Q4-A)。无矢量/栅格时返回 None。
        仅采用栅格的 bounds_in_working_crs (米制可靠); 缺该字段的栅格跳过, 避免混入原生 CRS 单位。
        """
        parts = []
        vec_bbox, _ = self._bbox_from_vectors(
            m2_output, enclave_ref_bbox=None, enclave_km=0.0)
        if vec_bbox:
            parts.append(tuple(vec_bbox))
        for r in m2_output.get("raster_inventory", []):
            b = r.get("bounds_in_working_crs")
            if b and len(b) == 4 and all(np.isfinite(v) for v in b):
                parts.append(tuple(float(v) for v in b))
        if not parts:
            return None
        return self._merge_bboxes(parts)

    def _maybe_clip_bbox(self, expanded, must_include, m2_output, solver_p, src_desc):
        """★P4★ 视 bbox_clip_to_data_extent 把 expanded 裁到数据范围 (端点保护),
        记录遥测 (_bbox_clipped / _bbox_raw), 返回最终 bbox (list)。"""
        from utils.geo_utils import clip_bbox_to_extent
        self._bbox_raw = list(expanded)
        if not bool(solver_p.get("bbox_clip_to_data_extent", True)):
            self._bbox_clipped = False
            logger.info(f"工作区范围({src_desc}, 开关关闭未裁剪): {list(expanded)}")
            return list(expanded)
        data_extent = self._compute_data_extent(m2_output)
        result, clipped = clip_bbox_to_extent(
            expanded, data_extent, must_include=must_include)
        self._bbox_clipped = bool(clipped)
        if clipped:
            logger.info(
                f"工作区范围({src_desc}, 裁到数据范围): {list(result)} "
                f"(裁剪前 {list(expanded)})")
        elif data_extent is None:
            logger.info(
                f"工作区范围({src_desc}, 无数据范围可裁, 保持外扩): {list(result)}")
        else:
            logger.info(
                f"工作区范围({src_desc}, 未超出数据范围, 无需裁剪): {list(result)}")
        return list(result)

    # ─── 9.1 栅格层构建 ────────────────────────────────────
    def _build_raster_layers(self, m2_output: dict):
        # ★P5 (v0.6)★ emit_unconsumed_outputs 总开关: 关闭时一张栅格都不产。
        #   算法端不消费预处理栅格(自己从 GPKG 用 rasterio 栅格化), 故 forbidden_mask/
        #   tower_mask/lpcf/tscf/tower_difficulty 全部跳过。这也落实了 P1 推迟的
        #   "分离式代价场 lpcf/tscf 停产"(分级引导值/is_one_time_cost 逻辑保留在
        #   _build_rasters_at_resolution 里, flag 开时复用, 不删除)。
        if not self.emit_unconsumed_outputs:
            logger.info("emit_unconsumed_outputs=false, 跳过全部栅格层构建 "
                        "(算法端自行从 GPKG 栅格化; lpcf/tscf 等停产)")
            return
        logger.info("构建栅格层...")
        m3_dir = ensure_dir(str(self.output_dir / "m3"))
        # ★Round 4★ 文件名 suffix 改为"分辨率角色"(fine/coarse), 不再硬编码具体数值.
        # 这样 fine_res 在 project.json 配成 10/12.5/15/20 m 等不同值时,
        # 文件名 forbidden_mask_fine.tif 都符合实际语义.
        # 具体数值在 manifest.fine_resolution_m / coarse_resolution_m 字段查.
        # ★P3 (v0.6)★ 分辨率门控: fine 档仅在 enable_fine_resolution=true 时构建。
        # 算法端 Stage-1 用单一粗分辨率 (coarse=100m), 默认不产 *_fine.tif 可省一遍全分辨率栅格化。
        # ★P5★ emit_unconsumed_outputs 总门控已在 _build_raster_layers 入口接入 (关则不产)。
        res_list = [(self.coarse_res, "coarse")]
        if self.enable_fine_resolution:
            res_list.append((self.fine_res, "fine"))
        for res, suffix in res_list:
            self._build_rasters_at_resolution(m2_output, res, suffix, m3_dir)

    def _build_rasters_at_resolution(self, m2_output: dict, res: float,
                                     suffix: str, out_dir: str):
        xmin, ymin, xmax, ymax = self.bbox
        width = max(1, int(math.ceil((xmax - xmin) / res)))
        height = max(1, int(math.ceil((ymax - ymin) / res)))
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

        logger.info(f"栅格 {suffix}: {width}x{height}, 分辨率={res}m")

        profile = {
            "driver": "GTiff", "dtype": "float32",
            "width": width, "height": height, "count": 1,
            "crs": self.working_crs, "transform": transform,
            "nodata": -9999, "compress": "lzw",
        }

        # ─── forbidden_mask ───
        forbidden_mask = np.zeros((height, width), dtype=np.uint8)
        forbidden_geoms = [
            (mapping(item["geometry"]), 1)
            for item in m2_output.get("forbidden_polygons", [])
            if item.get("geometry") and not item["geometry"].is_empty
        ]
        if forbidden_geoms:
            forbidden_mask = rasterize(
                forbidden_geoms, out_shape=(height, width),
                transform=transform, fill=0, dtype=np.uint8,
            )
        mask_profile = profile.copy()
        mask_profile.update(dtype="uint8", nodata=255)
        self._write_raster(
            os.path.join(out_dir, f"forbidden_mask_{suffix}.tif"),
            forbidden_mask, mask_profile
        )

        # ─── tower_mask ───
        tower_mask = np.ones((height, width), dtype=np.uint8)
        tower_mask[forbidden_mask == 1] = 0
        no_tower_geoms = [
            (mapping(item["geometry"]), 1)
            for item in m2_output.get("no_tower_polygons", [])
            if item.get("geometry") and not item["geometry"].is_empty
        ]
        if no_tower_geoms:
            nt_raster = rasterize(
                no_tower_geoms, out_shape=(height, width),
                transform=transform, fill=0, dtype=np.uint8,
            )
            tower_mask[nt_raster == 1] = 0

        valley_path = os.path.join(str(self.output_dir / "m2"), "valley_mask.tif")
        if os.path.exists(valley_path):
            # v0.3 修复: 原先用 ds.read(1, out_shape=, resampling=Resampling.max) 有两个 bug
            #   (1) Resampling.max 不被 ds.read 支持, 抛 "can be used for warp operations but
            #       not for reads and writes", 被外层 except: pass 静默吞 -> 山谷从未扣过
            #   (2) ds.read(out_shape=) 不做空间对齐, 源栅格 bbox != 工作区 bbox 时像素错位
            # 修复: 走 warp.reproject 统一解决 CRS/bbox/分辨率/聚合方法 四件事
            # 山谷是硬规则"禁立塔"(规则表第64条), 聚合用 max (块内只要有 1 像素山谷就视作山谷)
            try:
                valley = resample_to_workspace(
                    valley_path, height, width, transform, self.working_crs,
                    method="max", fill=0, dtype=np.uint8,
                )
                tower_mask[valley == 1] = 0
            except Exception as e:
                self._record_degrade(
                    "VALLEY_RESAMPLE_FAILED", "severe", e,
                    where=f"_build_rasters_at_resolution({suffix})/valley",
                )

        self._write_raster(
            os.path.join(out_dir, f"tower_mask_{suffix}.tif"),
            tower_mask, mask_profile
        )

        # ─── lpcf: 路径代价场 ───
        # P0-1修复: 只放真正连续型通过代价，不放一次性事件代价
        # v5.3: 一次性代价使用分级引导值（解决问题清单2.6）
        lpcf = np.ones((height, width), dtype=np.float32)

        # v5.3: 收集所有一次性代价面，按代价等级分配引导值
        guide_pairs = []  # (geom, guide_value)
        for item in m2_output.get("cost_polygons", []):
            geom = item.get("geometry")
            if not geom or geom.is_empty:
                continue
            cost_type = item.get("cost_type", "fixed")
            cross_cost = item.get("cross_cost", 0)
            land_cost = item.get("land_cost", 0)
            tmp_rule = {
                "cost_type": cost_type, "cross_cost": cross_cost,
                "land_cost": land_cost,
                "cross_cost_formula": item.get("cross_cost_formula"),
            }
            if is_one_time_cost(tmp_rule):
                # v5.3: 引导值按代价大小分级（解决问题清单2.6）
                # 原来统一1.5x太弱，5000级别和100级别无法区分
                max_cost = max(
                    cross_cost if isinstance(cross_cost, (int, float)) else 0,
                    land_cost if isinstance(land_cost, (int, float)) else 0
                )
                if max_cost >= 5000:
                    guide_val = 8.0   # 强引导
                elif max_cost >= 1000:
                    guide_val = 4.0   # 中等引导
                elif max_cost >= 100:
                    guide_val = 2.0   # 轻引导
                else:
                    guide_val = 1.5   # 微弱引导
                guide_pairs.append((geom, guide_val))

        # v5.3: 批量栅格化
        if guide_pairs:
            guide_layer = batch_rasterize_max(
                guide_pairs, (height, width), transform, fill=0)
            lpcf = np.maximum(lpcf, guide_layer)

        # 禁区设为极大值
        lpcf[forbidden_mask == 1] = 999999

        # 贴近奖励 — ★Tier1 优化 1.1★ 批量栅格化, 与原逐个减法**像素级完全等价**
        # 关键: MergeAlg.add 让 rasterize 把所有 shapes 的值"按像素累加"到同一 raster,
        # 然后只做一次 lpcf -= reward_raster * 0.001, 与原 for 循环逐个减完全等价
        # (a-b-c == a-(b+c)). 复杂度从 O(N × W × H) 降到 O(N + W × H),
        # 大埔工程 2418 走廊 × 2.5 亿格的循环 → 一次 rasterize, 估计省 ~20 分钟。
        reward_shapes = []
        for corridor in m2_output.get("preferred_corridors", []):
            cg = corridor.get("corridor_geometry")
            if not cg or cg.is_empty:
                continue
            reward = corridor.get("parallel_reward", 0)
            if reward and reward > 0:
                reward_shapes.append((mapping(cg), float(reward)))
        if reward_shapes:
            try:
                reward_raster = rasterize(
                    reward_shapes, out_shape=(height, width),
                    transform=transform, fill=0, dtype=np.float32,
                    merge_alg=MergeAlg.add,
                )
                lpcf = lpcf - reward_raster * 0.001
            except Exception as e:
                # 批量失败时退回逐条 (保持产出正确性优先于速度)
                logger.warning(
                    f"贴近奖励批量栅格化失败 ({type(e).__name__}: {e}), "
                    f"退回逐条模式"
                )
                for corridor in m2_output.get("preferred_corridors", []):
                    cg = corridor.get("corridor_geometry")
                    if not cg or cg.is_empty:
                        continue
                    reward = corridor.get("parallel_reward", 0)
                    if reward > 0:
                        try:
                            reward_raster = rasterize(
                                [(mapping(cg), reward)],
                                out_shape=(height, width),
                                transform=transform, fill=0, dtype=np.float32,
                            )
                            lpcf = lpcf - reward_raster * 0.001
                        except Exception as e2:
                            logger.warning(
                                f"贴近奖励栅格化失败 (corridor rule_id={corridor.get('rule_id')}, "
                                f"voltage_kv={corridor.get('voltage_kv')}): {e2}"
                            )

        # P0-6: 800kV及以上线路600m禁止并行区 → 高代价（不是禁止通过，但禁止并行）
        # ★Tier1 优化 1.2★ 与 1.1 同样, 用批量栅格化替代循环. 但这里语义是"取最大",
        # 因为多条 800kV+ 走廊禁并行带重叠时, 不应累加 (会变 1000), 应该保持 500.
        # 用现有的 batch_rasterize_max (utils/geo_utils.py) — 它正是干这个的.
        excl_pairs = []
        for corridor in m2_output.get("preferred_corridors", []):
            voltage = corridor.get("voltage_kv", 0)
            if voltage >= DEFAULT_SOLVER_PARAMS["high_voltage_parallel_threshold_kv"]:
                excl_geom = corridor.get("exclusion_geometry")
                if excl_geom and not excl_geom.is_empty:
                    excl_pairs.append((excl_geom, 500.0))
        if excl_pairs:
            try:
                excl_raster = batch_rasterize_max(
                    excl_pairs, (height, width), transform, fill=0, dtype=np.float32
                )
                lpcf = np.maximum(lpcf, excl_raster)
            except Exception as e:
                logger.warning(
                    f"800kV+禁并行区批量栅格化失败 ({type(e).__name__}: {e}), "
                    f"该层将被跳过"
                )

        # ─── 山峰 (peak) reward (Round 5 实现, 修预存遗漏 Bug A) ───
        # 规则表 rule_id=63 "山峰": parallel_reward=30 (W/1000m), parallel_range_m=200,
        # is_landable=True (可立塔), 含义是"线路靠近山峰走有奖励".
        # M2 已写出 peak_mask.tif (基于 TPI), M3 之前只消费了 valley_mask (禁立塔),
        # peak_mask 一直没消费 → 山峰奖励永远没生效. 本次补上.
        #
        # 实现策略与"贴近奖励走廊"一致 (lpcf -= peak_raster * 0.001 × 30):
        #   - reward=30 → peak 像素 lpcf 降 0.03 (远小于 eco/agriculture 等的高代价,
        #     不会诱导线路钻进禁区, 行为安全)
        #   - 同 valley 消费一样, 只在 peak_mask.tif 存在时启用 → 无 DEM 时零行为变化
        peak_path = os.path.join(str(self.output_dir / "m2"), "peak_mask.tif")
        if os.path.exists(peak_path):
            try:
                # 与 valley 处理对称: max 聚合 (块内只要有 1 像素山峰就视作山峰像素)
                peak = resample_to_workspace(
                    peak_path, height, width, transform, self.working_crs,
                    method="max", fill=0, dtype=np.uint8,
                )
                # rule_id=63 parallel_reward=30, 与 lpcf reward 走廊语义一致
                PEAK_REWARD = 30.0
                lpcf = lpcf - peak.astype(np.float32) * (PEAK_REWARD * 0.001)
            except Exception as e:
                self._record_degrade(
                    "PEAK_RESAMPLE_FAILED", "degraded", e,
                    where=f"_build_rasters_at_resolution({suffix})/peak",
                )

        # 风区/覆冰区附加代价
        wind_adder_path = os.path.join(str(self.output_dir / "m2"), "wind_ice_path_adder.tif")
        if os.path.exists(wind_adder_path):
            # v0.3 修复: ds.read(out_shape=) 不做空间对齐, 源栅格(省级分区)bbox
            # 远大于工作区时像素会错位; 改走 warp.reproject。
            # 附加代价是逐像素加到 lpcf 的连续量, 下采样用 average 保持总量近似。
            try:
                wind_adder = resample_to_workspace(
                    wind_adder_path, height, width, transform, self.working_crs,
                    method="average", fill=0.0, dtype=np.float32,
                )
                lpcf += wind_adder
            except Exception as e:
                self._record_degrade(
                    "WIND_ICE_ADDER_RESAMPLE_FAILED", "degraded", e,
                    where=f"_build_rasters_at_resolution({suffix})/wind_ice_adder",
                )
                # 降级: 用 preprocessing_report 里的统一参数兜底
                report = m2_output.get("preprocessing_report", {})
                adder = report.get("wind_ice_unified_params", {}).get("path_cost_adder", 0)
                if adder > 0:
                    lpcf += adder
        else:
            report = m2_output.get("preprocessing_report", {})
            adder = report.get("wind_ice_unified_params", {}).get("path_cost_adder", 0)
            if adder > 0:
                lpcf += adder

        self._write_raster(
            os.path.join(out_dir, f"lpcf_{suffix}.tif"), lpcf, profile
        )

        # ─── tscf: 立塔代价场 ───
        # P0-1: tscf同样只保留连续型立塔代价
        # 一次性立塔代价由M5事件模型在实际立塔时收取
        tscf = np.ones((height, width), dtype=np.float32)

        # v0.4.5: 收集所有 (geom, guide_val), 稍后一次性批量栅格化
        tscf_guide_pairs = []

        for item in m2_output.get("cost_polygons", []):
            geom = item.get("geometry")
            if not geom or geom.is_empty:
                continue
            land_cost = item.get("land_cost", 0)
            cost_type = item.get("cost_type", "fixed")
            if isinstance(land_cost, (int, float)) and land_cost > 0:
                # 一次性立塔代价仅作为微小引导（用于塔位前瞻的粗近似）
                guide_val = min(land_cost * 0.01, 10.0)
                tscf_guide_pairs.append((geom, guide_val))  # v0.4.5 批量栅格化收集

        # v0.4.5 性能修复: 原代码逐 polygon rasterize + np.maximum, 在 17 万
        # cost_polygons × 1000 万像素的规模下单轮需要 1-3 小时 (分配 40MB 内存 +
        # 整图 maximum × 17 万次); 改为批量栅格化 (按 guide_val 分组一次性生成),
        # 总耗时从小时级降到秒级。语义与原代码等价。
        if tscf_guide_pairs:
            try:
                tscf_guide_layer = batch_rasterize_max(
                    tscf_guide_pairs, (height, width), transform,
                    fill=0, dtype=np.float32,
                )
                tscf = np.maximum(tscf, tscf_guide_layer)
            except Exception as e:
                logger.warning(
                    f"tscf 引导值批量栅格化失败 ({len(tscf_guide_pairs)} 个多边形): {e}"
                )

        # v0.2 BUGFIX: 禁塔区赋值原本在这里, 被后面的 tscf *= tower_mult 改掉了
        # (tower_mult=0.5 时 999999×0.5=499999.5, 算法端会当成高代价区而不是禁区)
        # 现在挪到所有乘法之后, 保证禁塔区的"硬 999999"不被破坏

        # 风区/覆冰区塔代价乘数
        tower_mult_path = os.path.join(str(self.output_dir / "m2"), "wind_ice_tower_multiplier.tif")
        if os.path.exists(tower_mult_path):
            # v0.3 修复: ds.read(out_shape=) 不做空间对齐; 塔乘数是 multiplier
            # (通常 0.5~2.0), 用 average 做邻域平均下采样合理。
            try:
                tower_mult = resample_to_workspace(
                    tower_mult_path, height, width, transform, self.working_crs,
                    method="average", fill=1.0, dtype=np.float32,
                )
                tscf *= tower_mult
                # v5.6: 把重采样后的 multiplier 也保存为独立栅格, 供 M5 每塔查表
                self._write_raster(
                    os.path.join(out_dir, f"wind_ice_tower_multiplier_{suffix}.tif"),
                    tower_mult, profile
                )
            except Exception as e:
                self._record_degrade(
                    "WIND_ICE_TOWER_MULT_RESAMPLE_FAILED", "degraded", e,
                    where=f"_build_rasters_at_resolution({suffix})/wind_ice_tower_mult",
                )

        # v5.6: 同样保存 line multiplier (如果 M2 有生成)
        line_mult_path = os.path.join(str(self.output_dir / "m2"), "wind_ice_line_multiplier.tif")
        if os.path.exists(line_mult_path):
            # v0.3 修复: 同上, 改走 warp.reproject 确保空间对齐
            try:
                line_mult = resample_to_workspace(
                    line_mult_path, height, width, transform, self.working_crs,
                    method="average", fill=1.0, dtype=np.float32,
                )
                self._write_raster(
                    os.path.join(out_dir, f"wind_ice_line_multiplier_{suffix}.tif"),
                    line_mult, profile
                )
            except Exception as e:
                self._record_degrade(
                    "WIND_ICE_LINE_MULT_RESAMPLE_FAILED", "degraded", e,
                    where=f"_build_rasters_at_resolution({suffix})/wind_ice_line_mult",
                )

        # v0.2 BUGFIX (接上): 禁塔区硬 999999 放在所有乘法/加法之后, 确保不被改写
        tscf[tower_mask == 0] = 999999

        self._write_raster(
            os.path.join(out_dir, f"tscf_{suffix}.tif"), tscf, profile
        )

        # ─── terrain_slope (仅精分辨率) ───
        if suffix == "fine":  # ★Round 4★ 原 "10m" → "fine" (分辨率角色, 与具体数值解耦)
            slope_src = os.path.join(str(self.output_dir / "m2"), "terrain_slope.tif")
            if os.path.exists(slope_src):
                # v0.3 修复: 同样的两个 bug (Resampling.max 不支持 read + bbox 错位),
                # 改走 warp.reproject。slope 用 max 聚合, 保留 50m/10m 目标单元内的
                # 最坏坡度, 让"塔腿方向坡度 < 40°"规则可靠。
                try:
                    slope = resample_to_workspace(
                        slope_src, height, width, transform, self.working_crs,
                        method="max", fill=0.0, dtype=np.float32,
                    )
                    self._write_raster(
                        os.path.join(out_dir, f"terrain_slope_{suffix}.tif"),
                        slope, profile
                    )
                except Exception as e:
                    self._record_degrade(
                        "SLOPE_10M_RESAMPLE_FAILED", "degraded", e,
                        where=f"_build_rasters_at_resolution({suffix})/slope_10m",
                    )

            max_turn_src = os.path.join(str(self.output_dir / "m2"), "wind_ice_max_turn.tif")
            if os.path.exists(max_turn_src):
                # v0.3 修复: 同上, max_turn 是"允许的最大转角", 值越小越严格。
                # 下采样用 min 保留最严格限制, 避免重覆冰区的 45° 被弱化为 90°。
                try:
                    max_turn = resample_to_workspace(
                        max_turn_src, height, width, transform, self.working_crs,
                        method="min", fill=90.0, dtype=np.float32,
                    )
                    self._write_raster(
                        os.path.join(out_dir, f"wind_ice_max_turn_{suffix}.tif"),
                        max_turn, profile
                    )
                except Exception as e:
                    self._record_degrade(
                        "WIND_ICE_MAX_TURN_RESAMPLE_FAILED", "degraded", e,
                        where=f"_build_rasters_at_resolution({suffix})/max_turn",
                    )

        # v5.4: 预计算塔位难度粗栅格(coarse), 给M4搜索时做 O(1) 查询替代逐边像素采样
        # 思路: difficulty = w_sparse*(1-valid_ratio) + w_tscf*norm_tscf + w_slope*max(slope-25,0)
        #       valid_ratio / norm_tscf / slope 都在 coarse 分辨率上做一次局部窗口聚合即可
        if suffix == "coarse":  # ★Round 4★ 原 "50m" → "coarse"
            try:
                self._build_tower_difficulty_raster_50m(
                    out_dir, tower_mask, tscf, height, width, transform, profile
                )
            except Exception as e:
                logger.warning(f"  塔位难度预计算失败(非致命): {e}")

    def _build_tower_difficulty_raster_50m(self, out_dir, tower_mask, tscf,
                                            height, width, transform, profile):
        """v5.4 / v0.3: 预先计算塔位难度栅格(50m), 避免M4逐边重复栅格采样

        数值量级与旧tower_difficulty_proxy保持一致:
          difficulty ≈ w_sparse*(1-valid_ratio)*1000 + w_tscf*norm_tscf*1000 + w_slope*max(slope-25,0)

        v0.3 关键修复:
          原 v0.2 试图用 ds.read(..., resampling=Resampling.max) 做"最坏坡度"聚合,
          但 rasterio 的 ds.read 底层走 GDAL RasterIO, 不支持 max/min/med/q1/q3/sum/rms,
          实际运行时会抛 "Resampling.max can be used for warp operations but not for
          reads and writes", slope_50m 被 except 降级成 0, w_slope 项恒 0,
          塔位前瞻失真 (GPT 原问题)。

          同时 ds.read(out_shape=...) 不做空间对齐: 真实工程 DEM 覆盖范围
          (通常是市/县级) 远大于工作区 bbox, 即便 ds.read 接受了 max, 也会把
          整张 DEM 按比例拉伸到 dst_shape, 每个像素物理位置全错。

          改用 utils.resample_to_workspace -> rasterio.warp.reproject 一次性解决:
            - max 聚合保留最坏坡度 (对应设计规则第 9 条"塔腿方向坡度 < 40°")
            - 通过 src_transform/src_crs → dst_transform/dst_crs 正确空间对齐
            - 失败走 _record_degrade 走严重降级, 不用 0 伪装平地
        """
        # 1) valid_ratio: 用 5x5 均值滤波(50m*5=250m 窗口, 覆盖走廊半宽附近)
        try:
            from scipy.ndimage import uniform_filter
            tm_float = tower_mask.astype(np.float32)
            valid_ratio = uniform_filter(tm_float, size=5, mode="constant", cval=0.0)
        except Exception:
            valid_ratio = tower_mask.astype(np.float32)

        # 2) 归一化 tscf (避免 999999 主导)
        tscf_clip = np.where(tscf >= 999999, 0.0, tscf).astype(np.float32)
        norm_tscf = np.clip(tscf_clip / 1000.0, 0.0, 1.0)

        # 3) 坡度: 读 m2 的 terrain_slope, 对齐到 50m 工作区网格
        # v0.3 修复:
        #   (a) 原先 Resampling.max 传给 ds.read 会抛 "can be used for warp only" ->
        #       被 except 降级为填 0, w_slope 项恒为 0, 塔位前瞻失真 (GPT 原问题)
        #   (b) ds.read(out_shape=) 不做空间对齐, M2 的 terrain_slope.tif 是在 DEM
        #       原 bbox 上算的, 真实工程 DEM 通常比工作区大得多 -> 像素错位
        #   改用 resample_to_workspace 走 warp.reproject, 一次性解决两件事
        # 另: slope 失败升为 SEVERE —— 坡度前瞻完全不可用会让算法端在山区选出
        #     不合规的塔位, 比 value 填 0 伪装成"地势平坦"更糟
        slope_50m = None
        slope_src = os.path.join(str(self.output_dir / "m2"), "terrain_slope.tif")
        if os.path.exists(slope_src):
            try:
                slope_50m = resample_to_workspace(
                    slope_src, height, width, transform, self.working_crs,
                    method="max", fill=0.0, dtype=np.float32,
                )
                # 源分辨率诊断信息 (仅日志)
                with rasterio.open(slope_src) as sds:
                    src_res = max(abs(sds.res[0]), abs(sds.res[1]))
                # v0.3 修复: 原条件 src_res < 40 永远命中无意义;
                # 改成只在"源分辨率粗于 50m"时警告 (真正的精度不足场景)
                if src_res > 50:
                    logger.warning(
                        f"  塔位难度 slope 源分辨率 {src_res:.1f}m > 50m, "
                        f"上采样到 50m 精度可能不足, 请考虑换更高精度 DEM"
                    )
                else:
                    logger.info(
                        f"  塔位难度 slope: 源 {src_res:.1f}m → 50m (max 聚合)"
                    )
            except Exception as e:
                # 坡度读失败是塔位前瞻失真的主要原因, 升为 severe
                self._record_degrade(
                    "TOWER_DIFFICULTY_SLOPE_FAILED", "severe", e,
                    where="_build_tower_difficulty_raster_50m",
                )
                slope_50m = None
        else:
            # M2 没产出 slope (比如 DEM 缺失), 这里也记一个降级,
            # 严重度让 DEM 缺失本身触发 (_check_dem_quality 已经会报 severe)
            self._record_degrade(
                "TOWER_DIFFICULTY_SLOPE_MISSING", "degraded",
                "terrain_slope.tif 不存在 (DEM 缺失或 M2 处理失败)",
                where="_build_tower_difficulty_raster_50m",
            )
        if slope_50m is None:
            slope_50m = np.zeros((height, width), dtype=np.float32)

        slope_penalty = np.maximum(slope_50m - 25.0, 0.0)

        w_sparse, w_tscf, w_slope = 0.7, 0.3, 200.0
        difficulty = (w_sparse * (1.0 - valid_ratio) * 1000.0
                      + w_tscf * norm_tscf * 1000.0
                      + w_slope * slope_penalty)

        # 禁立塔像素直接拉到很高(但不做为硬禁区, 硬禁区由forbidden_mask保证)
        difficulty = np.where(tower_mask == 0, 50000.0, difficulty).astype(np.float32)

        # ★Round 4★ 文件名跟 coarse_resolution_m 走, 不再写死 50m
        self._write_raster(
            os.path.join(out_dir, "tower_difficulty_coarse.tif"),
            difficulty, profile
        )
        logger.info(f"  塔位难度栅格已预计算: tower_difficulty_coarse.tif")

    @staticmethod
    def _write_raster(path, data, profile):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)

    # ─── 9.3 配置文件编译 ──────────────────────────────────
    def _compile_rule_config(self, m2_output: dict) -> dict:
        rules_path = os.path.join(get_config_dir(), "default_feature_rules.json")
        rules_data = load_json(rules_path)

        compiled_rules = []
        for feat in rules_data["features"]:
            rule = {
                "rule_id": feat["id"],
                "level1": feat["level1"],
                "level2": feat["level2"],
                "is_enterable": feat.get("is_enterable", True),
                "is_landable": feat.get("is_landable", True),
                "cross_allow": feat.get("cross_allow", True),
                "land_cost": feat.get("land_cost", 0),
                # ★P1 (v0.6)★ 规则表原始人读 cross_cost ("30+30*cosα"/数值) 存为 cross_cost_expr
                # (审计, 预处理内部用); 算法端要的数值基础项叫 cross_cost, 由下方 decompose 提供,
                # 与 FeatureType.cross_cost 同名, 避免覆盖、对接零改名。
                "cross_cost_expr": feat.get("cross_cost", 0),
                "cost_type": feat.get("cost_type", "fixed"),
                "cross_cost_formula": feat.get("cross_cost_formula"),
                "buffer_dist_m": feat.get("buffer_m", 0),
                "min_cross_angle_deg": feat.get("min_cross_angle"),
                "parallel_reward": feat.get("parallel_reward"),
                # v0.2: parallel_range_m 默认值通过 utils.geo_utils.parallel_range_m_for
                # 统一, 与 M2 行为保持一致
                "parallel_range_m": parallel_range_m_for(feat),
                # ★P1 (v0.6)★ 并入与算法端 FeatureType 同名的数值代价字段:
                # cross_cost / cross_cost_angle_coeff / cross_cost_per_km / tower_cost
                # (land_cost 与 cross_cost_expr/cost_type/cross_cost_formula 保留作人读审计)。
                **decompose_cost_fields(feat),
            }

            geom_family = self._infer_geometry_family(feat)
            rule["geometry_family"] = geom_family
            behavior = self._determine_behavior(feat)
            rule["compiled_behavior"] = behavior
            compiled_rules.append(rule)

        report = m2_output.get("preprocessing_report", {})
        river_rule = self.config.get("river_rule", {
            "major_river_threshold_m": 900,
            "major_river_cross_mode": "forbidden",
            "major_river_landable": False,
            "minor_river_cross_mode": "allowed_with_constraints",
        })

        building_cluster = self.config.get("building_cluster", {
            "buffer_m": 100, "merge_gap_m": 50,
            "dense_zone_penalty": 3000, "min_cluster_area_m2": 50000,
        })

        wind_ice_params = report.get("wind_ice_unified_params", {
            "tower_cost_multiplier": 1.0,
            "line_cost_multiplier": 1.0,
            "max_turn_angle_deg": 45 if self.config.get("ice_zone", 10) >= 20 else 90,
            "path_cost_adder": 0,
        })

        return {
            "voltage_kv": self.voltage_kv,
            "compiled_rules": compiled_rules,
            "river_rule": river_rule,
            "building_cluster": building_cluster,
            "wind_ice_params": wind_ice_params,
            "ice_zone": self.config.get("ice_zone", 10),
            "wind_zone": self.config.get("wind_zone", "B"),
        }

    def _compile_solver_params(self) -> dict:
        params = DEFAULT_SOLVER_PARAMS.copy()
        ice_zone = self.config.get("ice_zone", 10)
        if ice_zone >= 20:
            params["max_turn_angle_deg"] = 45
        user_params = self.config.get("solver_params", {})
        params.update(user_params)
        return params

    def _compile_workspace(self, m2_output: dict) -> dict:
        # v0.2 BUGFIX: start_end 解析原本用 "for _, row in iterrows()" 的索引变量 _
        # 与 se_gdf.index[0]/[-1] 比较, 对非默认 RangeIndex 或只有一行的情况会把同一行
        # 同时赋给 start 和 end。重写为两遍扫描: 先按 type 精确匹配, 再按顺序兜底。
        start_point = None
        end_point = None
        ctrl = m2_output.get("control_objects", {})
        if "start_end" in ctrl:
            se_gdf = ctrl["start_end"]
            rows = list(se_gdf.iterrows())
            # Pass 1: 精确按 type 字段匹配
            for _, row in rows:
                if row.geometry is None or row.geometry.is_empty:
                    continue
                t = str(row.get("type", "")).lower()
                if t in ("start", "起点") and start_point is None:
                    start_point = [row.geometry.x, row.geometry.y]
                elif t in ("end", "终点") and end_point is None:
                    end_point = [row.geometry.x, row.geometry.y]
            # Pass 2: 兜底 — 有多行但没 type 字段时, 按出现顺序取首尾两个不同的点
            if start_point is None or end_point is None:
                valid_pts = [
                    [r.geometry.x, r.geometry.y]
                    for _, r in rows
                    if r.geometry is not None and not r.geometry.is_empty
                ]
                if len(valid_pts) >= 2:
                    if start_point is None:
                        start_point = valid_pts[0]
                    if end_point is None and valid_pts[-1] != start_point:
                        end_point = valid_pts[-1]
                elif len(valid_pts) == 1 and start_point is None:
                    # 只有一行时不猜测终点, 让调用者用 project.json 的 end_point 兜底
                    start_point = valid_pts[0]

        return {
            "project_name": self.config.get("project_name", ""),
            "project_crs": self.config.get("source_crs", "EPSG:4490"),
            "working_crs": self.working_crs,
            "bbox": self.bbox,
            "bbox_clipped": bool(self._bbox_clipped),   # ★P4★ 是否被裁到数据范围
            "bbox_raw": self._bbox_raw,                  # ★P4★ 裁剪前 (外扩后) 的 bbox; 未裁则与 bbox 同
            "coarse_resolution_m": self.coarse_res,
            "fine_resolution_m": self.fine_res,
            "start_point": start_point or self.config.get("start_point"),
            "end_point": end_point or self.config.get("end_point"),
            "must_pass_ordered": True,
            "corridor_count": 5,
        }

    def _compile_preprocessing_report(self, m2_output: dict) -> dict:
        report = m2_output.get("preprocessing_report", {}).copy()
        report.update({
            "voltage_kv": self.voltage_kv,
            "working_crs": self.working_crs,
            "bbox": self.bbox,
            "forbidden_polygon_count": len(m2_output.get("forbidden_polygons", [])),
            "no_tower_polygon_count": len(m2_output.get("no_tower_polygons", [])),
            "cost_polygon_count": len(m2_output.get("cost_polygons", [])),
            "linear_cross_segment_count": len(m2_output.get("linear_cross_segments", [])),
            "preferred_corridor_count": len(m2_output.get("preferred_corridors", [])),
        })
        # v0.2: 带上 raster_inventory 快照 (只记元信息, 不存数据), 便于
        # manifest 层判断 DEM 是否存在
        # v0.4: 同时带上 bounds_in_working_crs / res_in_working_crs / crs_fallback_used
        # 方便下游审核时直接看到"M0 是否成功对齐到 working_crs"
        report["_raster_inventory_snapshot"] = [
            {k: v for k, v in r.items()
             if k in ("inferred_type", "abs_path", "res", "bounds",
                      "bounds_in_working_crs", "res_in_working_crs",
                      "crs", "crs_fallback_used")}
            for r in m2_output.get("raster_inventory", [])
        ]
        report["delivery_level"] = self._determine_delivery_level(report, m2_output)
        return report

    def _determine_delivery_level(self, report: dict, m2_output: dict) -> dict:
        # v0.2: 升级为三档 FORMAL_DELIVERY / PRELIMINARY_ROUTE_ONLY / SEVERE_DEGRADED
        # SEVERE 触发条件 (连初步选线都不靠谱):
        #   - DEM 完全缺失 (DEM 里的 severe 标记)
        #   - bbox 未定或面积为 0
        #   - forbidden_polygons 与 cost_polygons 全空 (意味着 M0/M1 基本啥都没扫到)
        #
        # 注: manifest 层还会再叠一轮文件级 severe 判断 (核心栅格缺失等),
        # 详见 utils/manifest.determine_severity_from_manifest。这里只给出 M3 能看到的那部分。
        reasons = []
        severe_reasons = []
        upgrade_actions = []

        # 1) 数据可用性降级
        if not report.get("river_polygon_available", True):
            reasons.append("河流面域数据未提供，宽河屏障未生成")
            upgrade_actions.append("补齐河流面域数据")

        if not report.get("wind_ice_available", True):
            reasons.append("风区/覆冰区组合栅格未提供，使用统一参数")
            upgrade_actions.append("补齐风区/覆冰区组合栅格")

        # 2) DEM 质量(_check_dem_quality 现在会返回 severe=True/False)
        dem_info = self._check_dem_quality(m2_output)
        if dem_info.get("degraded"):
            if dem_info.get("severe"):
                severe_reasons.append(dem_info["reason"])
            else:
                reasons.append(dem_info["reason"])
            upgrade_actions.append(dem_info["action"])

        # 3) bbox 合法性
        b = self.bbox
        if not b or len(b) != 4:
            severe_reasons.append("工作区 bbox 未定")
            upgrade_actions.append("在 project.json 设置 bbox 或提供 DEM/矢量以推断")
        else:
            try:
                if (b[2] - b[0]) <= 0 or (b[3] - b[1]) <= 0:
                    severe_reasons.append(f"工作区 bbox 面积为 0 或退化: {b}")
                    upgrade_actions.append("检查 project.json 的 bbox / 起终点坐标")
            except Exception:
                severe_reasons.append(f"工作区 bbox 无法解析: {b}")

        # 4) 关键矢量层全空
        if (not m2_output.get("forbidden_polygons")
                and not m2_output.get("cost_polygons")
                and not m2_output.get("no_tower_polygons")):
            severe_reasons.append("M2 未产出任何禁区/禁立塔/高代价面 "
                                 "(可能是 M0 扫描或 M1 映射失败)")
            upgrade_actions.append("检查 M0 扫描的 GDB/SHP 清单与 M1 映射统计")

        # 5) v0.3: 运行时降级原因 (替代旧 except: pass 静默吞异常)
        # 由 _record_degrade() 收集, 例如:
        #   VALLEY_RESAMPLE_FAILED -> severe (山谷禁立塔是硬规则, 失败意味着 tower_mask 没扣谷)
        #   SLOPE_RESAMPLE_FAILED  -> severe (影响塔位难度前瞻和坡度约束判定)
        #   WIND_ICE_RESAMPLE_FAILED -> degraded (有统一参数兜底)
        #   MULTI_TIER_BUFFER_FAILED -> info (非关键, 最多丢一层保护环)
        runtime_summary: Dict[str, int] = {}
        for d in self._runtime_degrades:
            key = f"{d['code']}({d['severity']})"
            runtime_summary[key] = runtime_summary.get(key, 0) + 1
            if d["severity"] == "severe":
                msg = f"{d['code']}: {d['detail']}"
                if msg not in severe_reasons:
                    severe_reasons.append(msg)
            elif d["severity"] == "degraded":
                msg = f"{d['code']}: {d['detail']}"
                if msg not in reasons:
                    reasons.append(msg)

        # 5b) v0.4 问题 11 + v0.4.3 审核问题 6: 骨架退化检查
        # 即使栅格/矢量一切正常, 若骨架节点数极少或 0 边,
        # 下游 M4 绝对找不到路径, 不能标 PRELIMINARY 而必须 SEVERE。
        # v0.4.3: 新增 "节点边数都正常但起终点不在同一连通分量" 这种真实常见失败
        nav_meta = self._nav_graph_meta or {}
        # 只对 immediate 模式做检查; deferred/skip 是用户主动放弃, 不报警
        if nav_meta.get("build_mode") == "immediate":
            node_count = int(nav_meta.get("node_count") or 0)
            edge_count = int(nav_meta.get("edge_count") or 0)
            status = nav_meta.get("status", "")
            if status in ("skipped", "failed"):
                msg = f"骨架构建 status={status}: {nav_meta.get('skip_reason', '')}"
                severe_reasons.append(msg)
                upgrade_actions.append("检查 lpcf_coarse.tif / forbidden_mask_coarse.tif 是否就绪")
            elif node_count < 10:
                severe_reasons.append(
                    f"骨架节点数过少 ({node_count} < 10), M4 无法搜索路径"
                )
                upgrade_actions.append(
                    "检查 bbox 是否过小或禁区覆盖率过高; 必要时放宽 skeleton "
                    "阈值或重新确定 bbox"
                )
            elif edge_count == 0:
                severe_reasons.append(
                    f"骨架边数为 0 ({node_count} 节点), 拓扑不连通, M4 无法搜索"
                )
                upgrade_actions.append(
                    "检查骨架 cutoff 距离; 可能 bbox 太大导致节点太稀疏"
                )
            else:
                # v0.4.3: 起终点连通性检查
                conn = nav_meta.get("start_end_connectivity", {}) or {}
                if conn.get("checked") and not conn.get("all_connected"):
                    severe_reasons.append(
                        f"起终点连通性失败: {conn.get('reason', '')} "
                        f"(起终点 {conn.get('start_end_count')} 个分散在 "
                        f"{conn.get('components_involved')} 个分量)"
                    )
                    upgrade_actions.append(
                        "检查禁区是否形成完整屏障隔断起终点; "
                        "可尝试放宽骨架 cutoff 或调整起终点位置"
                    )
                elif edge_count < node_count * 0.5:
                    # 边数少于节点数一半, 连通性堪忧
                    reasons.append(
                        f"骨架拓扑稀疏 (node={node_count}, edge={edge_count}), "
                        f"搜索可能退化"
                    )

        # 6) 最终级别
        if severe_reasons:
            level = "SEVERE_DEGRADED"
        elif reasons:
            level = "PRELIMINARY_ROUTE_ONLY"
        else:
            level = "FORMAL_DELIVERY"

        return {
            "level": level,
            "reasons": reasons,                # 保持向后兼容 (老消费者只认这个字段)
            "severe_reasons": severe_reasons,  # 新字段
            "upgrade_actions": upgrade_actions,
            "runtime_degrades": self._runtime_degrades,  # v0.3: 完整明细
            "runtime_degrades_summary": runtime_summary,  # v0.3: 按 code 汇总
            "nav_graph_health": {  # v0.4 新增
                "node_count": int(nav_meta.get("node_count") or 0),
                "edge_count": int(nav_meta.get("edge_count") or 0),
                "build_mode": nav_meta.get("build_mode"),
                "status": nav_meta.get("status"),
            },
            "dem_quality": {
                "degraded": dem_info.get("degraded", False),
                "severe": dem_info.get("severe", False),
                "worst_res_m": dem_info.get("worst_res_m"),
                "worst_nodata_ratio": dem_info.get("worst_nodata_ratio"),
            },
            # ★Round 5 Bug C★ 透传 M2 算出的 dem_coverage 给 manifest
            # M2 在 _process_terrain 已经在 preprocessing_report 写了 dem_coverage 字段,
            # 这里转嵌到 delivery_level 下, 与 dem_quality 平级, 方便 manifest 一并取
            "dem_coverage": (
                m2_output.get("preprocessing_report", {}).get("dem_coverage")
                or {
                    "status": "no_dem",
                    "ratio_to_workspace": 0.0,
                    "covered_bbox_wcrs": None,
                    "workspace_bbox_wcrs": None,
                    "recommended_mode": "2D_PLANAR",
                }
            ),
        }

    def _check_dem_quality(self, m2_output: dict) -> dict:
        # v0.2 BUGFIX:
        #   1. 原代码只检查第一个 DEM (return {"degraded": False} 在循环内)
        #   2. DEM 完全缺失时返回 degraded=False, 让假阳性"正式交付"通过
        # 修复: 聚合所有 DEM 的最坏指标; DEM 缺失时明确标降级 + severe 标记
        #
        # v0.4 BUGFIX: 分辨率和空洞比例都按米制单位 + 扩展 nodata 计算,
        # 而不是盲信原生 ds.res / ds.nodata。
        # 读取优先级:
        #   1) m2_output["preprocessing_report"]["dem_quality"]  (M2 已经按扩展
        #      nodata 识别 + 米制分辨率计算过, 最权威)
        #   2) raster_inventory[*]["res_in_working_crs"]        (M0 计算, 米制)
        #   3) raster_inventory[*]["res"]                         (原生, 可能是度)
        nodata_threshold = DEFAULT_SOLVER_PARAMS.get("dem_nodata_ratio_threshold", 0.05)
        res_threshold = DEFAULT_SOLVER_PARAMS.get("dem_resolution_threshold_m", 30)

        dem_entries = [
            r for r in m2_output.get("raster_inventory", [])
            if r.get("inferred_type") == "DEM"
        ]

        if not dem_entries:
            return {
                "degraded": True,
                "severe": True,
                "reason": "未发现 DEM 数据 (坡度/山谷/山峰全部无法分析)",
                "action": "补齐 DEM 数据 (推荐分辨率 ≤ 12.5 m)",
            }

        # v0.4: 优先读 M2 report (米制 + 扩展 nodata)
        m2_report = m2_output.get("preprocessing_report", {}) or {}
        m2_dem_q = m2_report.get("dem_quality") or {}

        # v0.4.4 审核问题 3: 如果 M2 已经标 severe (比如 DEM 不覆盖工作区),
        # 直接返回, 不要再走分辨率/空洞阈值判定 (那些数字在这种场景下没意义)
        if m2_dem_q.get("severe"):
            reason = (
                m2_dem_q.get("coverage_error")
                or m2_report.get("terrain_analysis")
                or "M2 标记 DEM 质量 severe"
            )
            return {
                "degraded": True,
                "severe": True,
                "reason": reason,
                "action": "检查工作区 bbox 是否在 DEM 覆盖范围内, 或补齐工程区 DEM",
                "worst_res_m": m2_dem_q.get("resolution_m"),
                "worst_nodata_ratio": m2_dem_q.get("nodata_ratio"),
            }

        worst_res = 0.0
        worst_nodata = 0.0

        if "resolution_m" in m2_dem_q:
            # M2 已经算好米制分辨率 (必要时从度换算), 直接用
            worst_res = float(m2_dem_q["resolution_m"])
        if "nodata_ratio" in m2_dem_q:
            # M2 用扩展 nodata mask 算的空洞比例, 更准
            worst_nodata = float(m2_dem_q["nodata_ratio"])

        # 只要 M2 没给, 才退回到 inventory 或现场读盘
        if worst_res == 0.0:
            for r in dem_entries:
                # 优先 res_in_working_crs (M0 计算, 保证米制)
                rw = r.get("res_in_working_crs")
                if rw and len(rw) >= 2:
                    actual_res = max(abs(rw[0]), abs(rw[1]))
                else:
                    # 最后退回到原生 res (可能是度单位!)
                    res = r.get("res", [10, 10])
                    actual_res = max(abs(res[0]), abs(res[1]))
                worst_res = max(worst_res, actual_res)

        if worst_nodata == 0.0:
            for r in dem_entries:
                dem_path = r.get("abs_path")
                if dem_path and os.path.exists(dem_path):
                    try:
                        with rasterio.open(dem_path) as ds:
                            data = ds.read(1)
                            nodata = ds.nodata
                        if nodata is not None:
                            ratio = float((data == nodata).sum()) / max(data.size, 1)
                        else:
                            ratio = float(np.isnan(data).sum()) / max(data.size, 1)
                        worst_nodata = max(worst_nodata, ratio)
                    except Exception:
                        pass

        # 两种降级触发 (都非 severe): 精度不够 / 空洞过多
        if worst_res > res_threshold:
            return {
                "degraded": True,
                "severe": False,
                "reason": f"DEM 标称精度最差 {worst_res:.0f} m > {res_threshold} m",
                "action": f"替换为精度 ≤ {res_threshold} m 的 DEM",
                "worst_res_m": round(worst_res, 2),
                "worst_nodata_ratio": round(worst_nodata, 4),
            }
        if worst_nodata > nodata_threshold:
            return {
                "degraded": True,
                "severe": False,
                "reason": f"DEM 空洞最大比例 {worst_nodata:.1%} > {nodata_threshold:.0%}",
                "action": "修复 DEM 空洞或更换数据源",
                "worst_res_m": round(worst_res, 2),
                "worst_nodata_ratio": round(worst_nodata, 4),
            }

        return {
            "degraded": False,
            "severe": False,
            "worst_res_m": round(worst_res, 2),
            "worst_nodata_ratio": round(worst_nodata, 4),
        }

    # ─── 9.4 混合导航图骨架 ────────────────────────────────
    def _build_nav_graph_skeleton(self, m2_output: dict) -> dict:
        """v0.2: 支持三种构建模式 (由 solver_params.skeleton_build_mode 控制)

        - "immediate" (默认): 立即构建骨架并写盘, 与 v5.7 行为一致
        - "deferred":  M3 跳过骨架, 但输出的 manifest 里会标 status="deferred",
                       告诉算法端"你自己按需构建", 一次性节省 M3 阶段几分钟。
                       这符合 v5_2 改进清单的问题 6 (M3 耗时占比过大, 骨架延迟构建)。
        - "skip":      不构建, 也不标记可构建 (给纯栅格级算法的场景, 比如初步可达性分析)。
        """
        solver_p = DEFAULT_SOLVER_PARAMS.copy()
        solver_p.update(self.config.get("solver_params", {}) or {})
        mode = solver_p.get("skeleton_build_mode", "skip")

        # ★P5 (v0.6)★ emit_unconsumed_outputs 总开关优先于 skeleton_build_mode:
        #   关闭时, 无论 skeleton_build_mode 为何, 都不构建骨架/跨越窗口, 仅写 metadata 占位
        #   (走下方 skip 分支)。打开时, skeleton_build_mode 生效(immediate/deferred/skip)。
        if not getattr(self, "emit_unconsumed_outputs", False):
            if mode not in ("deferred", "skip"):
                logger.info(
                    f"emit_unconsumed_outputs=false, 骨架强制跳过 "
                    f"(原 skeleton_build_mode={mode}, 仅写 metadata 占位)")
            mode = "skip"

        if mode in ("deferred", "skip"):
            logger.info(f"骨架构建模式={mode}, 跳过 M3 内同步构建")
            m3_dir = ensure_dir(str(self.output_dir / "m3"))
            # v0.4.3 审核问题 2-A 修复: 通过统一的 _save_nav_graph 写 metadata,
            # 保证 immediate / deferred / skip 三条路径的 metadata schema 一致
            deferred_hint = (
                "算法端可按需调用预处理包提供的骨架构建函数 (见 README 第七节)"
                if mode == "deferred" else None
            )
            self._save_nav_graph(
                nodes=[], control_nodes=[], edges=[], windows=[],
                out_dir=m3_dir,
                status=mode, build_mode=mode,
                connectivity=None,
                extra_meta={"deferred_hint": deferred_hint} if deferred_hint else {},
            )
            return {"status": mode, "node_count": 0, "edge_count": 0,
                    "crossing_window_count": 0, "build_mode": mode,
                    "deferred_hint": deferred_hint}

        logger.info("构建混合导航图骨架 (mode=immediate)...")

        m3_dir = ensure_dir(str(self.output_dir / "m3"))
        # ★Round 4★ 文件名跟 coarse 走, 不再硬编码 50m
        lpcf_path = os.path.join(m3_dir, "lpcf_coarse.tif")
        forb_path = os.path.join(m3_dir, "forbidden_mask_coarse.tif")

        if not os.path.exists(lpcf_path) or not os.path.exists(forb_path):
            skip_reason = "lpcf_coarse or forbidden_mask_coarse missing"
            logger.warning(f"栅格层未就绪，跳过导航图构建: {skip_reason}")
            # v0.4.4: 这条路径以前不写 metadata, 下游读 metadata 会找不到文件
            self._save_nav_graph(
                nodes=[], control_nodes=[], edges=[], windows=[],
                out_dir=m3_dir,
                status="skipped", build_mode="immediate",
                connectivity=None,
                extra_meta={"skip_reason": skip_reason},
            )
            return {"status": "skipped", "build_mode": "immediate",
                    "node_count": 0, "edge_count": 0, "crossing_window_count": 0,
                    "skip_reason": skip_reason}

        with rasterio.open(lpcf_path) as ds:
            lpcf = ds.read(1)
            transform = ds.transform
        with rasterio.open(forb_path) as ds:
            forbidden = ds.read(1)

        h, w = lpcf.shape
        res = abs(transform[0])

        # 生成均匀骨架节点
        nodes = self._generate_skeleton_nodes(lpcf, forbidden, transform, res, h, w)

        # 控制节点
        ctrl = m2_output.get("control_objects", {})
        control_nodes = self._generate_control_nodes(ctrl)

        # 跨越窗口
        windows = self._build_crossing_windows(m2_output)

        # 骨架边
        edges = self._build_skeleton_edges(nodes, control_nodes, lpcf, forbidden, transform)

        # v0.4.3 审核问题 2-B 修复: 先算连通性, 再 _save_nav_graph, 让
        # hybrid_nav_graph_metadata.json 也含 start_end_connectivity
        connectivity = self._check_start_end_connectivity(
            nodes, control_nodes, edges
        )

        # 保存 (metadata 包含完整健康信息)
        self._save_nav_graph(
            nodes, control_nodes, edges, windows, m3_dir,
            status="ok", build_mode="immediate",
            connectivity=connectivity,
        )

        meta = {
            "status": "ok",
            "node_count": len(nodes) + len(control_nodes),
            "edge_count": len(edges),
            "crossing_window_count": len(windows),
            "build_mode": "immediate",
            # 连通性诊断, 供 _determine_delivery_level 判定 severe/degraded
            "start_end_connectivity": connectivity,
        }
        logger.info(
            f"导航图: {meta['node_count']}N, {meta['edge_count']}E, "
            f"{meta['crossing_window_count']}W, "
            f"start_end_connected={connectivity.get('all_connected')}"
        )
        return meta

    def _check_start_end_connectivity(
        self, nodes: list, control_nodes: list, edges: list
    ) -> dict:
        """
        v0.4.3: 对骨架跑 union-find, 检查 start_end_* 节点是否都在同一连通分量。

        Returns:
            {
              "checked": bool,
              "start_end_count": int,
              "components_involved": int,    # 起终点分散在几个连通分量
              "all_connected": bool,          # 所有起终点在同一 component
              "largest_component_size": int,
              "total_components": int,
              "reason": str,                  # 失败原因 (如 "no start_end nodes")
            }
        """
        # 全部节点 id 列表
        all_ids = [n["node_id"] for n in nodes] + [n["node_id"] for n in control_nodes]
        start_end_ids = [
            n["node_id"] for n in control_nodes
            if str(n.get("node_type", "")).startswith("start_end_")
        ]

        result = {
            "checked": False,
            "start_end_count": len(start_end_ids),
            "components_involved": 0,
            "all_connected": False,
            "largest_component_size": 0,
            "total_components": 0,
            "reason": "",
        }

        if len(start_end_ids) < 2:
            result["reason"] = (
                f"起终点节点 < 2 ({len(start_end_ids)}), 跳过连通性检查"
            )
            return result
        if not edges:
            result["reason"] = "无边可用, 所有起终点不可达"
            result["components_involved"] = len(start_end_ids)
            return result
        if not all_ids:
            result["reason"] = "无节点"
            return result

        # Union-find
        parent = {nid: nid for nid in all_ids}

        def find(x):
            # path compression (iterative)
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        skipped_edges = 0
        for e in edges:
            u, v = e.get("from_id"), e.get("to_id")
            if u in parent and v in parent:
                union(u, v)
            else:
                skipped_edges += 1

        # 统计分量
        roots_of_se = {find(nid) for nid in start_end_ids}
        component_sizes: dict = {}
        for nid in all_ids:
            r = find(nid)
            component_sizes[r] = component_sizes.get(r, 0) + 1

        result["checked"] = True
        result["components_involved"] = len(roots_of_se)
        result["all_connected"] = len(roots_of_se) == 1
        result["largest_component_size"] = max(component_sizes.values())
        result["total_components"] = len(component_sizes)
        if skipped_edges:
            result["skipped_edges"] = skipped_edges
        if not result["all_connected"]:
            result["reason"] = (
                f"起终点分散在 {len(roots_of_se)} 个连通分量, M4 找不到跨越路径"
            )
        return result

    def _generate_skeleton_nodes(self, lpcf, forbidden, transform, res, h, w):
        """v0.2: 三级自适应骨架节点 — 带阈值校准 + 每块节点硬上限

        v5.5 原版在实际数据上会把 ~68% 块判为"dense" + dense_step=100m → 单个 500m 块
        内产 25 个节点, 20×20 km 区域爆出 22k 节点, 后续边构建成为 O(N²) 瓶颈。

        v0.2 改进:
          - 阈值校准: dense 要求 (forbid_ratio>15% AND min_dist<100m) 或 hc_ratio>50%
                        (原来 forbid_ratio>5% AND min_dist<150m 或 hc_ratio>40% 过松)
          - 高代价掩膜阈值从 >3.0 升到 >=4.0 (只有"中等引导值"以上才算 high_cost;
            原来 2.0 级的基本农田类数据被错判为 high_cost)
          - dense_step 100m → 150m (节点密度降 55% 而仍满足 100m 最小塔间距的搜索粒度)
          - 每块节点硬上限: dense≤12 / complex≤9 / open≤1 (防极端数据再次爆表)
          - 全局节点预算: skeleton_budget_nodes (solver_params 里配, 默认 12000), 超过则
            自动上调 dense_step 再来一轮
        """
        from scipy.ndimage import distance_transform_edt

        # 参数 (阈值 + 步长 + 每块上限)
        BLOCK_SIZE_M = 500
        THRESH = {
            "dense":   {"forbid_ratio": 0.15, "min_dist_m": 100.0, "hc_ratio": 0.50},
            "complex": {"min_dist_m": 300.0, "hc_ratio": 0.20},
        }
        STEP_M = {"dense": 150, "complex": 200, "open": 400}
        CAP = {"dense": 12, "complex": 9, "open": 2}
        HIGH_COST_LPCF = 4.0  # > 4.0 才算高代价区

        # 预算
        solver_p = DEFAULT_SOLVER_PARAMS.copy()
        solver_p.update(self.config.get("solver_params", {}) or {})
        budget = int(solver_p.get("skeleton_budget_nodes", 12000))

        block_size_px = max(1, int(BLOCK_SIZE_M / res))

        # 距禁区的距离(像素)
        try:
            dist_to_forbid = distance_transform_edt(forbidden == 0)
        except Exception:
            dist_to_forbid = np.full(forbidden.shape, 1e6, dtype=np.float32)

        high_cost = (lpcf > HIGH_COST_LPCF) & (lpcf < 999999)

        def _sample_block(r_lo, r_hi, c_lo, c_hi, step_px, cap):
            """在块内均匀采样, 返回 [(r, c), ...], 满足 |positions| <= cap"""
            bh, bw = r_hi - r_lo, c_hi - c_lo
            if bh <= 0 or bw <= 0:
                return []
            rs = list(range(r_lo + step_px // 2, r_hi, step_px))
            cs = list(range(c_lo + step_px // 2, c_hi, step_px))
            if not rs or not cs:
                return []
            positions = [(r, c) for r in rs for c in cs]
            if cap and len(positions) > cap:
                stride = int(math.ceil(len(positions) / cap))
                positions = positions[::stride][:cap]
            return positions

        def _classify_block(fr, md_m, hcr):
            """返回块复杂度 'dense' / 'complex' / 'open'"""
            t_dense = THRESH["dense"]
            if (fr > t_dense["forbid_ratio"] and md_m < t_dense["min_dist_m"]) \
                    or hcr > t_dense["hc_ratio"]:
                return "dense"
            t_cx = THRESH["complex"]
            if md_m < t_cx["min_dist_m"] or hcr > t_cx["hc_ratio"]:
                return "complex"
            return "open"

        def _generate_once(dense_step_m_override=None):
            nodes_local = []
            nid_counter = 0
            cnt = {"dense": 0, "complex": 0, "open": 0}
            dense_step_m = dense_step_m_override or STEP_M["dense"]
            dense_step_px = max(1, int(dense_step_m / res))
            complex_step_px = max(1, int(STEP_M["complex"] / res))
            open_step_px = max(1, int(STEP_M["open"] / res))
            step_tab = {"dense": dense_step_px, "complex": complex_step_px, "open": open_step_px}

            for br in range(0, h, block_size_px):
                for bc in range(0, w, block_size_px):
                    r_lo, r_hi = br, min(br + block_size_px, h)
                    c_lo, c_hi = bc, min(bc + block_size_px, w)
                    if r_hi <= r_lo or c_hi <= c_lo:
                        continue

                    block_forbid = forbidden[r_lo:r_hi, c_lo:c_hi]
                    block_hc = high_cost[r_lo:r_hi, c_lo:c_hi]
                    forbid_ratio = float(block_forbid.sum()) / max(block_forbid.size, 1)
                    hc_ratio = float(block_hc.sum()) / max(block_hc.size, 1)
                    min_dist_px = float(dist_to_forbid[r_lo:r_hi, c_lo:c_hi].min())
                    min_dist_m = min_dist_px * res

                    level = _classify_block(forbid_ratio, min_dist_m, hc_ratio)
                    cnt[level] += 1

                    positions = _sample_block(
                        r_lo, r_hi, c_lo, c_hi,
                        step_tab[level], CAP[level],
                    )
                    for (rr, cc) in positions:
                        if rr < 0 or rr >= h or cc < 0 or cc >= w:
                            continue
                        if forbidden[rr, cc] == 1:
                            continue
                        if lpcf[rr, cc] >= 999999:
                            continue
                        x = transform[2] + cc * abs(transform[0]) + abs(transform[0]) / 2
                        y = transform[5] - rr * abs(transform[4]) - abs(transform[4]) / 2
                        nodes_local.append({
                            "node_id": f"sk_{nid_counter}",
                            "x": x, "y": y,
                            "node_type": "skeleton",
                            "lpcf_val": round(float(lpcf[rr, cc]), 2),
                        })
                        nid_counter += 1
            return nodes_local, cnt

        # 第一轮
        nodes, cnt = _generate_once()
        # 超预算 → 把 dense_step 再放大一档 (150→225→300), 最多再试 2 次
        tried = 0
        while len(nodes) > budget and tried < 2:
            tried += 1
            new_step_m = STEP_M["dense"] + 75 * tried  # 150 → 225 → 300
            logger.warning(
                f"  骨架节点超预算 ({len(nodes)} > {budget}), "
                f"dense_step 临时放大到 {new_step_m}m 重采样"
            )
            nodes, cnt = _generate_once(dense_step_m_override=new_step_m)

        logger.info(
            f"  骨架节点三级采样: 开阔={cnt['open']}块 复杂={cnt['complex']}块 "
            f"密集={cnt['dense']}块, 总节点={len(nodes)}"
            + (f" [已触发预算回退×{tried}]" if tried else "")
        )
        return nodes

    def _build_crossing_windows(self, m2_output: dict) -> list:
        """构建跨越窗口索引"""
        windows = []
        lc_path = os.path.join(str(self.output_dir / "m2"), "linear_cross_indexed.gpkg")
        if os.path.exists(lc_path):
            try:
                gdf = gpd.read_file(lc_path)
                for _, row in gdf.iterrows():
                    geom = row.geometry
                    if geom and not geom.is_empty:
                        windows.append({
                            "geometry": geom.centroid,
                            "window_id": row.get("segment_id", ""),
                            "rule_id": row.get("rule_id", -1),
                            "level2": row.get("level2", ""),
                            "cross_cost": row.get("cross_cost", 0),
                            "min_cross_angle_deg": row.get("min_cross_angle_deg"),
                            "buffer_m": row.get("buffer_dist_m", 0),
                            "azimuth_deg": row.get("azimuth_deg", 0),
                        })
            except Exception as e:
                logger.warning(f"读取跨越窗口索引失败: {e}")

        return windows

    def _generate_control_nodes(self, ctrl: dict) -> list:
        control_nodes = []
        nid = 0

        # 必经点
        if "must_pass" in ctrl:
            for _, row in ctrl["must_pass"].iterrows():
                control_nodes.append({
                    "node_id": f"ctrl_mp_{nid}",
                    "x": row.geometry.x, "y": row.geometry.y,
                    "node_type": "must_pass",
                })
                nid += 1

        # 必经路径采样
        if "must_path" in ctrl:
            for _, row in ctrl["must_path"].iterrows():
                line = row.geometry
                if line and not line.is_empty:
                    total = line.length
                    dist = 0
                    while dist <= total:
                        pt = line.interpolate(dist)
                        control_nodes.append({
                            "node_id": f"ctrl_mpath_{nid}",
                            "x": pt.x, "y": pt.y,
                            "node_type": "must_path_sample",
                        })
                        nid += 1
                        dist += 200

        # 密集通道入口
        if "dense_corridor" in ctrl:
            for _, row in ctrl["dense_corridor"].iterrows():
                geom = row.geometry
                if hasattr(geom, "exterior"):
                    centroid = geom.centroid
                    control_nodes.append({
                        "node_id": f"ctrl_dc_{nid}",
                        "x": centroid.x, "y": centroid.y,
                        "node_type": "dense_corridor",
                    })
                    nid += 1

        # 起终点
        if "start_end" in ctrl:
            for _, row in ctrl["start_end"].iterrows():
                t = row.get("type", "waypoint")
                control_nodes.append({
                    "node_id": f"ctrl_se_{nid}",
                    "x": row.geometry.x, "y": row.geometry.y,
                    "node_type": f"start_end_{t}",
                })
                nid += 1

        return control_nodes

    def _build_skeleton_edges(self, nodes, control_nodes, lpcf, forbidden, transform):
        """v0.2: 用 scipy.spatial.cKDTree.query_pairs 替代 STRtree 单点查询循环。

        v5.3 原版对每个节点调 tree.query(Point.buffer(cutoff)), 22K 节点下累计 22K 次
        Point.buffer 构造 + 22K 次 STRtree 查询, 即使后续 _line_crosses_forbidden 已经
        向量化, 总开销仍在几分钟量级, GPT 复现时就卡在这一步。

        v0.2 关键改动:
          1) cKDTree.query_pairs(r=cutoff) 一次性拿到所有候选对, C 级实现
          2) cutoff_dist 改为密度驱动: median_nn × 4, 而不是按面积硬拍
          3) 候选对数超预算 (edge_candidate_budget) 时自动收紧 cutoff
          4) 每条边仍然用原 numpy 向量化的 _line_crosses_forbidden / _sample_avg_cost
        """
        from scipy.spatial import cKDTree

        all_nodes = nodes + control_nodes
        if len(all_nodes) < 2:
            return []

        coords = np.array([[n["x"], n["y"]] for n in all_nodes], dtype=np.float64)
        res_x = abs(transform[0])

        # 1) 密度驱动 cutoff
        # 用最近邻距离中位数 (k=2: 第一个是自己 dist=0, 第二个才是最近邻)
        try:
            tree = cKDTree(coords)
            if len(coords) >= 2:
                nn_dist, _ = tree.query(coords, k=2)
                median_nn = float(np.median(nn_dist[:, 1]))
            else:
                median_nn = 500.0
        except Exception as e:
            logger.warning(f"  cKDTree 建树失败, 回退到按面积算 cutoff: {e}")
            tree = None
            median_nn = 500.0

        solver_p = DEFAULT_SOLVER_PARAMS.copy()
        solver_p.update(self.config.get("solver_params", {}) or {})
        cutoff_factor = float(solver_p.get("skeleton_cutoff_nn_factor", 4.0))
        cutoff_min = float(solver_p.get("skeleton_cutoff_min_m", 500.0))
        cutoff_max = float(solver_p.get("skeleton_cutoff_max_m", 1500.0))
        cand_budget = int(solver_p.get("skeleton_edge_candidate_budget", 600_000))

        cutoff_dist = max(cutoff_min, min(cutoff_max, median_nn * cutoff_factor))

        if tree is None:
            # 回退: 还是按 v5.3 行为跑单点查询, 但用 cKDTree 的 query_ball_point 批量接口
            from shapely.strtree import STRtree
            node_points = [Point(c[0], c[1]) for c in coords]
            tree_fb = STRtree(node_points)
            pair_idx = []
            for i, (x, y) in enumerate(coords):
                cand = tree_fb.query(Point(x, y).buffer(cutoff_dist))
                for j in cand:
                    if j > i:
                        pair_idx.append((i, int(j)))
        else:
            # 2) 一次性拿所有候选对
            # 第一轮: 用 median_nn × cutoff_factor 估算
            candidate_pairs = tree.query_pairs(r=cutoff_dist, output_type="ndarray")
            n_pairs = len(candidate_pairs)

            # 3) 预算收敛: 候选对数爆表时收紧 cutoff, 最多再试 2 次
            tried = 0
            while n_pairs > cand_budget and cutoff_dist > cutoff_min and tried < 2:
                tried += 1
                cutoff_dist = max(cutoff_min, cutoff_dist * 0.75)
                candidate_pairs = tree.query_pairs(r=cutoff_dist, output_type="ndarray")
                n_pairs = len(candidate_pairs)
                logger.warning(
                    f"  骨架候选边超预算, cutoff 收紧到 {cutoff_dist:.0f}m, 候选对={n_pairs}"
                )

            logger.info(
                f"  骨架边构建 (cKDTree.query_pairs): N={len(all_nodes)}, "
                f"median_nn={median_nn:.1f}m, cutoff={cutoff_dist:.0f}m, "
                f"候选对={n_pairs}" + (f" [已收紧×{tried}]" if tried else "")
            )
            pair_idx = [(int(i), int(j)) for i, j in candidate_pairs]

        # 4) 对每条候选边做禁区穿越检查 + 代价采样
        edges = []
        for (i, j) in pair_idx:
            ni, nj = all_nodes[i], all_nodes[j]
            dx = nj["x"] - ni["x"]
            dy = nj["y"] - ni["y"]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 1.0:
                continue

            if self._line_crosses_forbidden(
                ni["x"], ni["y"], nj["x"], nj["y"],
                forbidden, transform, res_x
            ):
                continue

            avg_cost = self._sample_avg_cost(
                ni["x"], ni["y"], nj["x"], nj["y"],
                lpcf, transform, res_x
            )
            edge_cost = dist * max(avg_cost, 0.1)

            edges.append({
                "from_id": ni["node_id"],
                "to_id": nj["node_id"],
                "distance_m": round(dist, 1),
                "avg_lpcf": round(avg_cost, 2),
                "edge_cost": round(edge_cost, 1),
            })

        logger.info(f"  骨架边有效数: {len(edges)}")
        return edges

    def _line_crosses_forbidden(self, x1, y1, x2, y2, forbidden, transform, res):
        # v5.4-perf: numpy向量化，单次调用 ~10-30x 提速
        h, w = forbidden.shape
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        steps = max(2, int(dist / res))
        ts = np.linspace(0.0, 1.0, steps + 1)
        xs = x1 + ts * (x2 - x1)
        ys = y1 + ts * (y2 - y1)
        cs = ((xs - transform[2]) / abs(transform[0])).astype(np.int32)
        rs = ((transform[5] - ys) / abs(transform[4])).astype(np.int32)
        valid = (rs >= 0) & (rs < h) & (cs >= 0) & (cs < w)
        if not valid.any():
            return False
        return bool(np.any(forbidden[rs[valid], cs[valid]] == 1))

    def _sample_avg_cost(self, x1, y1, x2, y2, lpcf, transform, res):
        # v5.4-perf: numpy向量化
        h, w = lpcf.shape
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        steps = max(2, int(dist / res))
        ts = np.linspace(0.0, 1.0, steps + 1)
        xs = x1 + ts * (x2 - x1)
        ys = y1 + ts * (y2 - y1)
        cs = ((xs - transform[2]) / abs(transform[0])).astype(np.int32)
        rs = ((transform[5] - ys) / abs(transform[4])).astype(np.int32)
        valid = (rs >= 0) & (rs < h) & (cs >= 0) & (cs < w)
        if not valid.any():
            return 1.0
        vals = lpcf[rs[valid], cs[valid]]
        good = vals[vals < 999999]
        if good.size == 0:
            return 1.0
        return float(good.mean())

    def _save_nav_graph(self, nodes, control_nodes, edges, windows, out_dir,
                         status: str = "ok", build_mode: str = "immediate",
                         connectivity: Optional[dict] = None,
                         extra_meta: Optional[dict] = None):
        """
        v0.4.4: 统一 immediate / deferred / skip 三条路径的 metadata 写出。

        metadata 统一包含:
          status, build_mode, node_count, edge_count, crossing_window_count,
          start_end_connectivity (若 connectivity 非 None)
          + extra_meta 里的任何额外字段 (如 deferred_hint, skip_reason)
        """
        all_nodes = nodes + control_nodes
        if all_nodes:
            node_gdf = gpd.GeoDataFrame(
                all_nodes,
                geometry=[Point(n["x"], n["y"]) for n in all_nodes],
                crs=self.working_crs,
            )
            from utils.geo_utils import write_gdf_to_gpkg_safe
            write_gdf_to_gpkg_safe(node_gdf, os.path.join(out_dir, "nav_graph_nodes.gpkg"),
                                     "nav_graph_nodes")

        if edges:
            save_json({"edges": edges}, os.path.join(out_dir, "nav_graph_edges.json"))

        if windows:
            wdf = gpd.GeoDataFrame(windows, crs=self.working_crs)
            from utils.geo_utils import write_gdf_to_gpkg_safe
            write_gdf_to_gpkg_safe(wdf, os.path.join(out_dir, "crossing_window_index.gpkg"),
                                     "crossing_window_index")

        meta = {
            "status": status,
            "build_mode": build_mode,
            "node_count": len(all_nodes),
            "edge_count": len(edges),
            "crossing_window_count": len(windows),
        }
        # v0.4.3 审核问题 2-B: 把连通性诊断写进 metadata, 运维/第三方可直接读
        if connectivity is not None:
            meta["start_end_connectivity"] = connectivity
        if extra_meta:
            meta.update(extra_meta)

        save_json(meta, os.path.join(out_dir, "hybrid_nav_graph_metadata.json"))

    # ─── 辅助 ─────────────────────────────────────────────
    @staticmethod
    def _infer_geometry_family(feat: dict) -> str:
        l1 = feat.get("level1", "")
        l2 = feat.get("level2", "")
        if l1 in ("交通敏感点", "电力设施敏感点", "管廊敏感点"):
            if l2 in ("机场", "变电站", "换流站", "接地极"):
                return "polygon"
            return "line"
        if l1 == "河流":
            return "line"
        if l1 == "地形":
            return "raster"
        if feat.get("buffer_m", 0) > 0 and l1 == "重要设施与政府规划敏感点":
            if l2 in ("军事敏感点", "无线电设施", "导航台", "炸药库", "油气存储站",
                       "地震地磁台", "气象站", "采石场", "矿产资源"):
                return "point"
        return "polygon"

    @staticmethod
    def _determine_behavior(feat: dict) -> str:
        is_e = feat.get("is_enterable", True)
        is_l = feat.get("is_landable", True)
        cross = feat.get("cross_allow", True)
        min_angle = feat.get("min_cross_angle")
        cost_type = feat.get("cost_type", "fixed")

        if not is_e and not cross:
            return "FORBIDDEN_AREA"
        if not is_l:
            if min_angle:
                return "LINEAR_CROSS_CONTROL"
            return "NO_TOWER_AREA"
        if min_angle:
            return "LINEAR_CROSS_CONTROL"
        if feat.get("parallel_reward"):
            return "PREFERRED_CORRIDOR"
        if cost_type in ("angle_formula", "length_formula"):
            return "HIGH_COST_AREA"
        lc = feat.get("land_cost", 0)
        cc = feat.get("cross_cost", 0)
        if (isinstance(lc, (int, float)) and lc > 0) or (isinstance(cc, (int, float)) and cc > 0):
            return "HIGH_COST_AREA"
        return "NORMAL"

    # ─── 输出 ──────────────────────────────────────────────
    def _export(self, rule_config, solver_params, workspace,
                preprocessing_report, nav_graph_meta) -> dict:
        m3_dir = ensure_dir(str(self.output_dir / "m3"))
        save_json(rule_config, os.path.join(m3_dir, "rule_config.json"))
        save_json(solver_params, os.path.join(m3_dir, "solver_params.json"))
        save_json(workspace, os.path.join(m3_dir, "workspace.json"))
        save_json(preprocessing_report, os.path.join(m3_dir, "preprocessing_report.json"))

        return {
            "rule_config": rule_config,
            "solver_params": solver_params,
            "workspace": workspace,
            "preprocessing_report": preprocessing_report,
            "nav_graph_meta": nav_graph_meta,
        }
