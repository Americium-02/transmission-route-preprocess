"""
manifest.py — preprocess 包自描述清单 (v0.2 新增)

目的:
  预处理包完成后产出一份 <output_dir>/manifest.json, 它是算法端 (M4-M6) 消费本包的
  *唯一契约文件*。算法端拿到 output_dir 第一件事就是 load_manifest() + verify_manifest(),
  不依赖对目录结构的"默认知识", 从此减少"预处理看着跑完了下游却挂了"的情况。

manifest 的核心职责:
  1. 记录包版本 / 规则版本 / CRS / bbox / 分辨率
  2. 声明哪些文件是"必需产物"(required) 哪些是"可选产物"(optional)
  3. 列出实际写出的文件(存在 / 大小; 可选 md5)
  4. 把交付级别 (FORMAL_DELIVERY / PRELIMINARY_ROUTE_ONLY / SEVERE_DEGRADED) 挂在顶层
  5. 记录各项降级原因, 方便下游按原因有选择地容忍或拒绝

注: 下游 verify_manifest() 是一个"只读校验"函数, 不会改任何东西, 可以多次调用。
"""

from __future__ import annotations

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .geo_utils import save_json, load_json

logger = logging.getLogger("transmission_planning.manifest")

# 包格式版本 — 下游解析时据此判断兼容性
#   v0.1 -> v0.2: 首次引入 manifest 自描述 + 三档交付级别
#   v0.2 -> v0.3: nav_graph 块扩展 build_mode/nodes_available/edges_available
#                 /node_count/edge_count/crossing_windows_available;
#                 delivery_level 增加 runtime_degrades/runtime_degrades_summary;
#                 crossing_window_index.gpkg 从 OPTIONAL 移入 nav_graph 声明文件;
#                 wind_ice_line_multiplier_*.tif 从 OPTIONAL 下线
#                 (规则表有字段但 M2 未产出, 算法端未消费, 属于误导性条目)
#   v0.3 -> v0.4: 修 preprocess_v0.3.1 审核发现的 12 类问题 (CRS 未统一/bbox 优先级错
#                 /nodata 识别不完整/骨架退化不 FAIL 等)。新增字段:
#                   - manifest.crs_diagnostic (来自 m0/crs_diagnostic.json)
#                   - manifest.m1_diversity (来自 m1/semantic_mapping_report.json.diversity_score)
#                   - delivery_level.nav_graph_health (节点/边数 + 状态)
#                   - raster_inventory[*].bounds_in_working_crs / crs_fallback_used
#                 Solver 新增 bbox_start_end_buffer_km / bbox_enclave_filter_km。
# 向后兼容: 新字段都是加项, 旧消费端读 v0.4 只少看到新字段, 不报错;
#           v0.4 消费端读 v0.2/v0.3 老包 verify_manifest 只报 warning 不硬失败。
PACKAGE_FORMAT_VERSION = "0.4"
# 用来识别这份 preprocess 包的产品名, 方便未来做多包共存
PACKAGE_PRODUCT = "transmission_line_preprocess"


# ─── 必需 / 可选 产物声明 ────────────────────────────────────
# 约定:
#   - 必需产物缺失 => manifest.required_missing 非空 => delivery_level 至少 PRELIMINARY,
#     严重情况上升 SEVERE (见 determine_severity_from_manifest)
#   - 可选产物缺失 => 仅写入 optional_missing, 不触发降级
#
# 这些路径是相对 output_dir 的, 不带绝对前缀。
#
# 分组说明:
#   m0_required     : M0 的核心产物 — 缺了就意味着根本没扫到数据
#   m1_required     : M1 的核心产物
#   m2_required     : M2 的核心产物 (算法端 M4 会直接读的)
#   m3_required     : M3 的核心产物 (规则/求解器参数/工作区/栅格/导航图)
#   optional        : 各模块的可选产物 (数据降级时可能不存在)

M0_REQUIRED: List[str] = [
    "m0/gdb_inventory.json",
    "m0/raster_inventory.json",
    "m0/read_log.json",
    "m0/unified_vectors.gpkg",
    "m0/crs_diagnostic.json",  # v0.4: working_crs 合理性/轴序/起终点坐标校验
]

M1_REQUIRED: List[str] = [
    "m1/semantic_mapping_report.json",
    "m1/standardized_features.gpkg",
]

M2_REQUIRED: List[str] = [
    "m2/forbidden_polygons.gpkg",
    "m2/no_tower_polygons.gpkg",
    "m2/cost_polygons.gpkg",
]

M3_REQUIRED_CORE: List[str] = [
    "m3/rule_config.json",
    "m3/solver_params.json",
    "m3/workspace.json",
    "m3/preprocessing_report.json",
    # ★Round 4★ 文件名 suffix 从硬编码 _50m/_10m 改为分辨率角色 _coarse/_fine.
    # 具体数值在 manifest.coarse_resolution_m / fine_resolution_m 查.
    # 这样 fine_res 配 10/12.5/15/20m 时文件名都符合实际语义.
    "m3/forbidden_mask_coarse.tif",
    "m3/forbidden_mask_fine.tif",
    "m3/tower_mask_coarse.tif",
    "m3/tower_mask_fine.tif",
    "m3/lpcf_coarse.tif",
    "m3/lpcf_fine.tif",
    "m3/tscf_coarse.tif",
    "m3/tscf_fine.tif",
]

# ──────────────────────────────────────────────────────────────────
# ★P0 (v0.6)★ M3 必需清单动态化
# 背景: 算法端不消费任何预处理栅格(它自己重栅格化 GPKG), 故默认配置下
#   (emit_unconsumed_outputs=false) M3 不产 lpcf/tscf/mask, fine 档也默认关
#   (enable_fine_resolution=false)。若仍把这些 .tif 列为"必需", compile_manifest
#   会把它们记入 required_missing, 算法端启动闸门 validate_manifest_for_startup
#   读到非空 required_missing 就拒绝启动。
# 因此 required 必须随 project.json 的两个开关动态计算:
#   - 恒定必需: 4 个配置 JSON (rule_config/solver_params/workspace/preprocessing_report)
#   - emit_unconsumed_outputs=true: 追加 4 个 coarse 栅格
#   - 且 enable_fine_resolution=true: 再追加 4 个 fine 栅格
# 注: M3_REQUIRED_CORE (上面的全量静态清单) 保留, 供测试/文档引用与"全开"语义。
M3_REQUIRED_CONFIGS: List[str] = [
    "m3/rule_config.json",
    "m3/solver_params.json",
    "m3/workspace.json",
    "m3/preprocessing_report.json",
]
M3_COARSE_RASTERS: List[str] = [
    "m3/forbidden_mask_coarse.tif",
    "m3/tower_mask_coarse.tif",
    "m3/lpcf_coarse.tif",
    "m3/tscf_coarse.tif",
]
M3_FINE_RASTERS: List[str] = [
    "m3/forbidden_mask_fine.tif",
    "m3/tower_mask_fine.tif",
    "m3/lpcf_fine.tif",
    "m3/tscf_fine.tif",
]


def _output_flags(project_config: Optional[Dict]) -> Dict[str, bool]:
    """★P0★ 从 project.json 的 solver_params 读 v0.6 输出开关 (缺省安全)。"""
    sp = ((project_config or {}).get("solver_params", {}) or {})
    return {
        "enable_fine_resolution": bool(sp.get("enable_fine_resolution", False)),
        "emit_unconsumed_outputs": bool(sp.get("emit_unconsumed_outputs", False)),
    }


def compute_m3_required_core(project_config: Optional[Dict]) -> List[str]:
    """★P0★ 按输出开关动态计算 M3 必需清单 (替代静态 M3_REQUIRED_CORE)。"""
    flags = _output_flags(project_config)
    req = list(M3_REQUIRED_CONFIGS)
    if flags["emit_unconsumed_outputs"]:
        req += list(M3_COARSE_RASTERS)
        if flags["enable_fine_resolution"]:
            req += list(M3_FINE_RASTERS)
    return req

# 导航图及跨越窗口: v0.2 从"必需产物"里拆出来 — 走廊级算法(M4)才需要, 可以被懒构建
# 覆盖。manifest 里用单独的 nav_graph 字段描述。
# v0.3: crossing_window_index.gpkg 从 OPTIONAL 挪进来 - 它是算法端"跨越窗口搜索"
#       的契约输入, 不应该被当成一般"可选文件"; 但它的缺失本身不是 error
#       (没有跨越物的工程根本不需要)。manifest 的 nav_graph 块会用
#       crossing_windows_available 布尔字段明确语义。
M3_NAV_GRAPH_FILES: List[str] = [
    "m3/nav_graph_nodes.gpkg",
    "m3/nav_graph_edges.json",
    "m3/hybrid_nav_graph_metadata.json",
    "m3/crossing_window_index.gpkg",
]

OPTIONAL: List[str] = [
    # 线状交叉 / 优选走廊(无对应地物时可能没有)
    "m2/linear_cross_indexed.gpkg",
    "m2/preferred_corridors.gpkg",
    "m2/buffered_points.gpkg",
    "m2/building_clusters.gpkg",
    # 河流 / 风冰 / 地形 (数据降级时没有)
    "m2/river_crossing_windows.gpkg",
    "m2/wide_river_barriers.gpkg",
    "m2/wind_ice_path_adder.tif",
    "m2/wind_ice_max_turn.tif",
    "m2/wind_ice_tower_multiplier.tif",
    "m2/wind_ice_line_multiplier.tif",  # v0.3: M2 现在真产出这个
    "m2/terrain_slope.tif",
    "m2/terrain_tpi.tif",
    "m2/valley_mask.tif",
    "m2/peak_mask.tif",
    # M3 的一些派生栅格 (★Round 4★ 命名跟 fine/coarse 走)
    "m3/terrain_slope_fine.tif",
    "m3/wind_ice_max_turn_fine.tif",
    "m3/wind_ice_tower_multiplier_coarse.tif",
    "m3/wind_ice_tower_multiplier_fine.tif",
    "m3/wind_ice_line_multiplier_coarse.tif",  # v0.3: M2 产出后 M3 会派生
    "m3/wind_ice_line_multiplier_fine.tif",
    "m3/tower_difficulty_coarse.tif",
    # 调试产物
    "m2/processed_polygons.gpkg",
    "m1/unmapped_features_preview.json",
    # ★v0.5 新增审计产物★ (Phase A/B 产物, 算法端不消费, 仅供审计/排查)
    "m0/variant_inventory.json",          # Phase A: layer 变体 + 别名归并清单
    "m0/protection_coverage_report.json",  # Phase A: 规则要求保护范围但 GDB 缺变体的清单
    "m2/geometry_fixes_log.json",         # Phase B: make_valid 几何修复事件审计
]


def _file_entry(abs_path: Path, rel_path: str, compute_md5: bool) -> Dict:
    """单个文件的 manifest 条目。"""
    entry: Dict = {
        "path": rel_path,
        "exists": abs_path.is_file(),
    }
    if entry["exists"]:
        try:
            entry["size_bytes"] = abs_path.stat().st_size
        except OSError:
            entry["size_bytes"] = None
        if compute_md5 and entry.get("size_bytes", 0) and entry["size_bytes"] < 500_000_000:
            # 大于 500 MB 的文件 md5 成本太高, 跳过 (可在调用处选择不计算)
            try:
                h = hashlib.md5()
                with open(abs_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                entry["md5"] = h.hexdigest()
            except OSError:
                entry["md5"] = None
    return entry


def _split_existence(output_dir: Path, rels: List[str],
                     compute_md5: bool) -> Tuple[List[Dict], List[str]]:
    """返回 (entries, missing_rel_paths)。"""
    entries: List[Dict] = []
    missing: List[str] = []
    for rel in rels:
        abs_p = output_dir / rel
        entry = _file_entry(abs_p, rel, compute_md5)
        entries.append(entry)
        if not entry["exists"]:
            missing.append(rel)
    return entries, missing


def determine_severity_from_manifest(
    required_missing: List[str],
    m3_report: Dict,
    project_config: Optional[Dict] = None,
) -> Dict:
    """
    根据 manifest.required_missing 和 preprocessing_report 里的降级标记综合决定交付级别。

    规则 (优先级从高到低):
      SEVERE_DEGRADED — 连初步选线都不靠谱
        - DEM 完全缺失 (m3_report.delivery_level.reasons 命中 "未发现 DEM" 或 severe 标记)
        - bbox 未定 或 等效为 0 面积
        - M3 核心栅格产物缺失任一 (forbidden_mask/tower_mask/lpcf/tscf)
        - M2 核心必需产物缺失任一

      PRELIMINARY_ROUTE_ONLY — 数据有缺失但可以做初步方案
        - 河流面域 / 风冰栅格缺失
        - DEM 质量不达标但非缺失
        - 可选产物缺失(不直接触发, 看具体原因)

      FORMAL_DELIVERY — 无降级原因
    """
    severe_reasons: List[str] = []
    preliminary_reasons: List[str] = []
    upgrade_actions: List[str] = []

    # —— 1. M3 报告带过来的降级原因(由 _determine_delivery_level 塞入) ——
    # 注意: m3 已升级为 3 档, 它的 severe_reasons 是真正的严重降级(bbox 未定、核心矢量全空等),
    # 必须透传到 manifest 层的 severe, 不能被漏掉。
    m3_delivery = m3_report.get("delivery_level", {}) or {}
    for reason in m3_delivery.get("severe_reasons", []) or []:
        if reason not in severe_reasons:
            severe_reasons.append(reason)
    for reason in m3_delivery.get("reasons", []) or []:
        preliminary_reasons.append(reason)
    for act in m3_delivery.get("upgrade_actions", []) or []:
        if act not in upgrade_actions:
            upgrade_actions.append(act)

    # —— DEM 严重性双层闸门 (v0.5 C1 + C6) ——
    # 设计原则:
    #   v0.4.6 行为: DEM 完全缺失 → 一律 SEVERE_DEGRADED, 整个项目不能交付
    #   v0.5 改造: DEM 缺失改为 PRELIMINARY_ROUTE_ONLY (允许平面 2D 模式做初步选线),
    #              但只在 project_config.allow_planar_2d_mode=True 时启用降级;
    #              默认仍按 v0.4.6 严格模式 (SEVERE), 保持向后兼容.
    #
    # 闸门 1: project_config.allow_planar_2d_mode (C6 上层闸门, 由项目配置决定)
    # 闸门 2: M3 提供 dem_quality.severity 等细粒度信号 (下层闸门, 由 M3 判定)
    #
    # 此处只看上层闸门: allow_planar_2d_mode 是否启用.
    allow_planar_2d = bool((project_config or {}).get("allow_planar_2d_mode", False))
    
    # DEM severe 兜底识别 (以防老版 m3_report 把 DEM 缺失写进了 reasons 字段)
    dem_q = m3_report.get("dem_quality") or {}
    has_dem_severe_already = any("DEM" in r for r in severe_reasons)
    if not m3_report.get("raster_inventory_has_dem", True) and not has_dem_severe_already:
        # ★v0.5 C1+C6★ DEM 缺失分级处理:
        #   - allow_planar_2d_mode=True: 降为 preliminary (可平面 2D 选线),
        #                                算法端通过 manifest.operational_mode 知道走 2D 模式
        #   - 否则 (默认 + 向后兼容): 保持 v0.4.6 severe 行为
        if allow_planar_2d:
            preliminary_reasons.append("DEM 完全缺失 (allow_planar_2d_mode 启用, 降级为 2D 模式)")
            if "补齐 DEM 数据 (可选, 提升交付级别)" not in upgrade_actions:
                upgrade_actions.append("补齐 DEM 数据 (可选, 提升交付级别)")
        else:
            severe_reasons.append("DEM 完全缺失")
            if "补齐 DEM 数据" not in upgrade_actions:
                upgrade_actions.append("补齐 DEM 数据")
    # 字面兜底: preliminary_reasons 里如果混入"未发现 DEM"/"DEM 完全缺失", 看 allow_planar_2d_mode 决定升级与否
    for reason in list(preliminary_reasons):
        if ("未发现 DEM" in reason or "DEM 完全缺失" in reason) and "allow_planar_2d_mode" not in reason:
            # 没带 allow_planar_2d_mode 标签 → 来自 M3 / 旧版的 DEM 缺失信号
            if allow_planar_2d:
                # 改写为带标签的 preliminary
                preliminary_reasons.remove(reason)
                preliminary_reasons.append(f"{reason} (allow_planar_2d_mode 启用, 降级为 2D 模式)")
            else:
                # 严格模式: 升级为 severe
                if reason not in severe_reasons:
                    severe_reasons.append(reason)
                preliminary_reasons.remove(reason)
    
    # ★修复 #16★ 字面兜底也扫描 severe_reasons (M3 把 DEM 缺失写到 severe_reasons 时的场景):
    # M3._check_dem_quality 在 DEM 完全缺失时返回 severe=True, reason="未发现 DEM 数据..."
    # 然后 M3 把该 reason 直接写入 delivery_level.severe_reasons (m3:1002),
    # manifest 透传到本函数 severe_reasons. 上面的 preliminary 字面兜底扫不到这里.
    # 此处补一段扫描: 如果 allow_planar_2d=True 且 severe_reasons 含 DEM 缺失相关条目,
    # 把它从 severe_reasons 移到 preliminary_reasons (加 allow_planar 标签).
    # 这样 C1+C6 才能真正实现"上层闸门压制 M3 的 severe 判定"的设计意图.
    if allow_planar_2d:
        for reason in list(severe_reasons):
            # 只处理 "DEM 缺失" 相关 severe (不动 "DEM 单位错误"、"M2 标记 DEM severe (覆盖错位)" 等);
            # 因为 DEM 覆盖错位等场景下, 即使走 2D 模式也用不到 DEM 派生信号, 应保持 severe;
            # 而"完全缺失"才是 2D 模式可以兜的合法降级.
            is_dem_missing = (
                "未发现 DEM" in reason
                or "DEM 完全缺失" in reason
                or ("DEM" in reason and "缺失" in reason)
            )
            # 排除已带 allow_planar 标签的 (避免重复处理)
            if is_dem_missing and "allow_planar_2d_mode" not in reason:
                # 检查是否含"覆盖"/"单位"等不可降级关键词 (DEM 数据有问题, 不是单纯缺)
                non_downgradable = any(kw in reason for kw in ("覆盖", "单位", "投影", "bounds"))
                if not non_downgradable:
                    severe_reasons.remove(reason)
                    downgraded = f"{reason} (allow_planar_2d_mode 启用, 降级为 2D 模式)"
                    if downgraded not in preliminary_reasons:
                        preliminary_reasons.append(downgraded)
                    # 升级 upgrade_actions 也加标签 (如果已存在的话)
                    for i, act in enumerate(list(upgrade_actions)):
                        if "DEM" in act and "可选" not in act:
                            upgrade_actions[i] = f"{act} (可选, 提升交付级别)"

    # —— 2. manifest 文件缺失升级为 severe 的规则 ——
    # 任一 M3 核心栅格缺失 / M2 forbidden 或 tower 缺失 -> severe
    critical_missing = [m for m in required_missing
                        if m.startswith("m3/forbidden_mask_")
                        or m.startswith("m3/tower_mask_")
                        or m.startswith("m3/lpcf_")
                        or m.startswith("m3/tscf_")
                        or m in ("m2/forbidden_polygons.gpkg",)]
    for m in critical_missing:
        severe_reasons.append(f"核心产物缺失: {m}")

    # 其他 required_missing 归为 preliminary
    non_crit = [m for m in required_missing if m not in critical_missing]
    for m in non_crit:
        preliminary_reasons.append(f"必需产物缺失: {m}")
        upgrade_actions.append(f"补产 {m}")

    # —— 3. 最终级别 ——
    if severe_reasons:
        level = "SEVERE_DEGRADED"
    elif preliminary_reasons:
        level = "PRELIMINARY_ROUTE_ONLY"
    else:
        level = "FORMAL_DELIVERY"

    return {
        "level": level,
        "severe_reasons": severe_reasons,
        "preliminary_reasons": preliminary_reasons,
        "upgrade_actions": upgrade_actions,
    }


def compile_manifest(
    output_dir: str,
    project_config: Dict,
    m3_report: Dict,
    workspace: Dict,
    solver_param_keys: List[str],
    rule_count: int,
    timings: Dict,
    nav_graph_status: str = "completed",
    nav_graph_meta: Optional[Dict] = None,
    compute_md5: bool = True,
) -> Dict:
    """
    汇总整个预处理产出的清单。在 run_preprocess 最末调用。

    Args:
        output_dir: 预处理输出根目录
        project_config: 原始 project.json 内容 (不要存机密字段)
        m3_report: m3 的 preprocessing_report (已经包含 delivery_level)
        workspace: m3 的 workspace.json 内容
        solver_param_keys: m3 的 solver_params 的 key 列表
        rule_count: 编译后的规则数
        timings: 各阶段耗时 (dict: "M0"/"M1"/"M2"/"M3" -> seconds)
        nav_graph_status: "completed" | "skipped" | "deferred" | "failed"
        nav_graph_meta: v0.3 新增, M3._build_nav_graph_skeleton 的完整返回值,
                        含 build_mode / node_count / edge_count / crossing_window_count;
                        用于 manifest 的 nav_graph 契约字段 (让算法端
                        verify_manifest 后能直接判断要不要自己搭骨架)
        compute_md5: 是否对每个产出文件计算 md5 (大型 tif 可能耗时)

    Returns:
        写入 <output_dir>/manifest.json 的字典。
    """
    output_dir_p = Path(output_dir).resolve()

    # 1) 按分组收集文件条目 + missing
    m0_entries, m0_missing = _split_existence(output_dir_p, M0_REQUIRED, compute_md5)
    m1_entries, m1_missing = _split_existence(output_dir_p, M1_REQUIRED, compute_md5)
    m2_entries, m2_missing = _split_existence(output_dir_p, M2_REQUIRED, compute_md5)
    m3c_entries, m3c_missing = _split_existence(output_dir_p, compute_m3_required_core(project_config), compute_md5)
    m3n_entries, m3n_missing = _split_existence(output_dir_p, M3_NAV_GRAPH_FILES, compute_md5)
    opt_entries, opt_missing = _split_existence(output_dir_p, OPTIONAL, compute_md5)

    required_missing: List[str] = m0_missing + m1_missing + m2_missing + m3c_missing
    # 导航图缺失单独看 nav_graph_status, 不一定触发降级

    # 2) 把 raster_inventory_has_dem 塞回 m3_report, 方便 severity 判断
    raster_inv = m3_report.get("_raster_inventory_snapshot") or []
    has_dem = any(r.get("inferred_type") == "DEM" for r in raster_inv)
    m3_report_enriched = {**m3_report, "raster_inventory_has_dem": has_dem}

    # 3) 综合定级
    severity = determine_severity_from_manifest(required_missing, m3_report_enriched, project_config)

    # 4) 汇总导航图状态
    # v0.3: 这是下游"要不要自己搭骨架"的唯一契约字段。status="deferred" 时算法端应该
    # 调用预处理包提供的懒加载接口自行构建; status in ("completed", "ok") 时直接读文件。
    nav_graph_meta = nav_graph_meta or {}
    # `nodes_available`/`edges_available` 是最直观的布尔契约: 文件在且非空
    nodes_file = next((e for e in m3n_entries if e["path"].endswith("nav_graph_nodes.gpkg")), None)
    edges_file = next((e for e in m3n_entries if e["path"].endswith("nav_graph_edges.json")), None)
    # v0.3: 跨越窗口文件同样加入契约声明
    cwin_file = next((e for e in m3n_entries if e["path"].endswith("crossing_window_index.gpkg")), None)
    nodes_available = bool(nodes_file and nodes_file.get("exists") and nodes_file.get("size_bytes", 0) > 0)
    edges_available = bool(edges_file and edges_file.get("exists") and edges_file.get("size_bytes", 0) > 0)
    # 跨越窗口: 文件存在且>0 = 真有跨越物; 文件不存在 = 要么没跨越物 (crossing_window_count==0),
    # 要么 M3 骨架未构建 (deferred/skip 模式)。两种情况语义不同, 算法端据此分别处理。
    crossing_windows_available = bool(
        cwin_file and cwin_file.get("exists") and cwin_file.get("size_bytes", 0) > 0)
    nav_graph_block = {
        "status": nav_graph_status,                                   # completed/ok/deferred/skipped/failed
        "build_mode": nav_graph_meta.get("build_mode", "unknown"),    # immediate/deferred/skip
        "nodes_available": nodes_available,
        "edges_available": edges_available,
        "crossing_windows_available": crossing_windows_available,     # v0.3: 跨越窗口索引契约
        "node_count": nav_graph_meta.get("node_count"),
        "edge_count": nav_graph_meta.get("edge_count"),
        "crossing_window_count": nav_graph_meta.get("crossing_window_count"),
        "skip_reason": nav_graph_meta.get("skip_reason"),             # 若 status="skipped"
        "deferred_hint": nav_graph_meta.get("deferred_hint"),         # 若 status="deferred"
        "files_present": sum(1 for e in m3n_entries if e["exists"]),
        "files_expected": len(m3n_entries),
        "files": m3n_entries,
    }

    # 5) 其他数据可用性
    data_avail_effective = {
        "dem": has_dem,
        "river_polygon": bool(m3_report.get("river_polygon_available", False)),
        "wind_ice_raster": bool(m3_report.get("wind_ice_available", False)),
    }

    # ★v0.5 C4★ operational_mode: 算法端可以一眼看出该走 3D 还是 2D 模式 + 为什么
    # 设计:
    #   mode="3D"           : DEM 存在且质量合格, 走标准三维路径优化
    #   mode="2D_PLANAR"    : DEM 缺失/质量不够, 但 allow_planar_2d_mode=True, 走平面 2D 选线
    #   mode="UNAVAILABLE"  : DEM 缺失且 allow_planar_2d_mode=False, 不能走 (SEVERE)
    # reasons 字段列出导致非 3D 模式的原因, 供算法端日志或回退决策使用
    allow_planar_2d_for_mode = bool(project_config.get("allow_planar_2d_mode", False))
    # ★Bug 2 修复 (v0.5+, 二次修订)★ 字段路径修正:
    # M3 (m3_rule_compile_and_output.py:1117) 把 dem_quality 嵌套在
    #   report["delivery_level"]["dem_quality"]
    # 而不是顶层 report["dem_quality"]. 第一轮修复只改了 schema 兼容 (布尔派生 severity),
    # 但没改读取路径, 导致 m3_report.get("dem_quality") 永远是空 dict,
    # severe/degraded 都是 False, 最终 dem_severity 总是 "ok" 假象.
    # 二次修复: 先尝试嵌套位置, 再 fallback 顶层 (向前兼容未来 M3 schema 升级).
    dem_quality = (
        m3_report.get("delivery_level", {}).get("dem_quality")
        or m3_report.get("dem_quality")
        or {}
    )
    # schema 兼容: 优先用 M3 提供的 severity 字符串; 缺失则从 severe/degraded 布尔派生
    dem_severity = dem_quality.get("severity")
    if dem_severity is None:
        if dem_quality.get("severe"):
            dem_severity = "severe"
        elif dem_quality.get("degraded"):
            dem_severity = "degraded"
        else:
            dem_severity = "ok"
    operational_mode_reasons: List[str] = []
    if has_dem and dem_severity == "ok":
        operational_mode = "3D"
    elif has_dem and dem_severity in ("degraded",):
        # 有 DEM 但质量降级: 仍按 3D 跑, 但记录原因
        operational_mode = "3D"
        operational_mode_reasons.append(f"DEM 质量 degraded (severity={dem_severity})")
    elif allow_planar_2d_for_mode:
        operational_mode = "2D_PLANAR"
        if not has_dem:
            operational_mode_reasons.append("DEM 完全缺失")
        else:
            operational_mode_reasons.append(f"DEM 严重不达标 (severity={dem_severity})")
        operational_mode_reasons.append("allow_planar_2d_mode=True 启用 2D 模式")
    else:
        operational_mode = "UNAVAILABLE"
        if not has_dem:
            operational_mode_reasons.append("DEM 完全缺失")
        if dem_severity == "severe":
            operational_mode_reasons.append(f"DEM 严重不达标 (severity={dem_severity})")
        operational_mode_reasons.append("allow_planar_2d_mode=False, 不允许 2D 退化")
    operational_mode_block = {
        "mode": operational_mode,
        "reasons": operational_mode_reasons,
        "allow_planar_2d_mode": allow_planar_2d_for_mode,
        "dem_severity": dem_severity,
    }

    # ★Round 5 Bug C★ 透传 dem_coverage 字段, 让算法端能判断:
    #   - DEM 全覆盖 → 全程 3D 优化
    #   - DEM 部分覆盖 → 混合模式, 算法端按 covered_bbox 判断哪段路用 3D 哪段用 2D
    #   - DEM 不覆盖 / 无 DEM → 全程 2D
    # 路径与 dem_quality 相同 (delivery_level 内 → 顶层 fallback), 无 DEM 时给默认值.
    dem_coverage_raw = (
        m3_report.get("delivery_level", {}).get("dem_coverage")
        or m3_report.get("dem_coverage")
    )
    if dem_coverage_raw is None:
        # M3 没透传过来 (可能是无 DEM 时 M2 dem_coverage 没被 M3 拷到 delivery_level),
        # 兜底给默认值, 让算法端字段读取不报错
        dem_coverage_block = {
            "status": "no_dem" if not has_dem else "unknown",
            "ratio_to_workspace": 0.0,
            "covered_bbox_wcrs": None,
            "workspace_bbox_wcrs": workspace.get("bbox"),
            "recommended_mode": "2D_PLANAR" if not has_dem else operational_mode,
        }
    else:
        dem_coverage_block = dict(dem_coverage_raw)
        # 补 workspace bbox (M2 可能没填或填了旧值)
        if dem_coverage_block.get("workspace_bbox_wcrs") is None:
            dem_coverage_block["workspace_bbox_wcrs"] = workspace.get("bbox")

    manifest = {
        "package_product": PACKAGE_PRODUCT,
        "package_format_version": PACKAGE_FORMAT_VERSION,
        "project_name": project_config.get("project_name", ""),
        "voltage_kv": project_config.get("voltage_kv"),
        "ice_zone": project_config.get("ice_zone"),
        "wind_zone": project_config.get("wind_zone"),
        "source_crs": project_config.get("source_crs"),
        "working_crs": project_config.get("working_crs") or workspace.get("working_crs"),
        "bbox": workspace.get("bbox"),
        "bbox_clipped": bool(workspace.get("bbox_clipped", False)),  # ★P4★ bbox 是否被裁到数据范围
        "bbox_raw": workspace.get("bbox_raw"),                        # ★P4★ 裁剪前(外扩后)的 bbox; 未裁则为 None
        "coarse_resolution_m": workspace.get("coarse_resolution_m"),
        "fine_resolution_m": workspace.get("fine_resolution_m"),
        "output_flags": _output_flags(project_config),  # ★P0★ enable_fine_resolution / emit_unconsumed_outputs
        "start_point": workspace.get("start_point"),
        "end_point": workspace.get("end_point"),
        "delivery_level": severity,  # 含 level / severe_reasons / preliminary_reasons / upgrade_actions
        "operational_mode": operational_mode_block,  # ★v0.5 C4★ 算法端 3D/2D 模式判定
        "dem_coverage": dem_coverage_block,  # ★Round 5 Bug C★ DEM 覆盖率详情
        "data_availability_effective": data_avail_effective,
        "required_missing": required_missing,
        "optional_missing": opt_missing,
        "nav_graph": nav_graph_block,
        "rule_count": rule_count,
        "solver_param_keys": list(solver_param_keys),
        "timings_seconds": timings,
        "files": {
            "m0": m0_entries,
            "m1": m1_entries,
            "m2": m2_entries,
            "m3_core": m3c_entries,
            "m3_nav_graph": m3n_entries,
            "optional": opt_entries,
        },
    }

    # 6) 写盘
    manifest_path = os.path.join(output_dir, "manifest.json")
    save_json(manifest, manifest_path)
    logger.info(f"manifest 已写出: {manifest_path}")
    logger.info(f"  交付级别: {severity['level']}")
    if severity["severe_reasons"]:
        for r in severity["severe_reasons"]:
            logger.warning(f"  [SEVERE] {r}")
    for r in severity.get("preliminary_reasons", []):
        logger.info(f"  [PRELIMINARY] {r}")
    return manifest


def verify_manifest(output_dir: str) -> Dict:
    """
    *下游算法端调用点*。

    从 <output_dir>/manifest.json 读出清单, 再逐文件检查是否真的存在 + 大小一致。
    返回一个字典, 包含 ok (bool), mismatches (list), manifest (原清单), 以及便捷字段。

    这个函数是只读的, 对预处理包不做任何改动。
    """
    output_dir_p = Path(output_dir).resolve()
    manifest_path = output_dir_p / "manifest.json"
    if not manifest_path.exists():
        return {
            "ok": False,
            "mismatches": [f"manifest.json 不存在于 {output_dir}"],
            "manifest": None,
        }
    try:
        manifest = load_json(str(manifest_path))
    except Exception as e:
        return {
            "ok": False,
            "mismatches": [f"manifest.json 解析失败: {e}"],
            "manifest": None,
        }

    mismatches: List[str] = []
    warnings_: List[str] = []  # v0.3: 不硬失败的提示, 与 mismatches 区分

    # v0.3: 放宽包格式版本检查 -- 只要主版本号 (minor 以下) 一致就放行
    # 具体规则:
    #   - 期望 "0.x", 读到 "0.y" 且 y <= x -> 只 warning, 不 fail
    #     (例如 v0.3 消费端读 v0.2 老包: 新字段缺 = 降级, 不是错误)
    #   - 读到 "0.y" 且 y > x -> mismatch (老消费端无法解析新包)
    #   - 主版本号不同 (如 "1.0" vs "0.3") -> mismatch (不保证兼容)
    pver = manifest.get("package_format_version")

    def _parse_ver(v):
        try:
            parts = str(v).split(".")
            return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except Exception:
            return None

    expected = _parse_ver(PACKAGE_FORMAT_VERSION)
    actual = _parse_ver(pver)
    if expected is None or actual is None or expected[0] != actual[0]:
        mismatches.append(
            f"package_format_version 不兼容: 期望 {PACKAGE_FORMAT_VERSION}, 实际 {pver} "
            f"(主版本号不同, 可能无法解析)")
    elif actual[1] > expected[1]:
        mismatches.append(
            f"package_format_version 超前: 期望 {PACKAGE_FORMAT_VERSION}, 实际 {pver} "
            f"(消费端过旧, 请升级)")
    elif actual[1] < expected[1]:
        warnings_.append(
            f"package_format_version 较旧: 期望 {PACKAGE_FORMAT_VERSION}, 实际 {pver} "
            f"(向后兼容读取, 部分新字段可能缺失)")

    for group, entries in (manifest.get("files") or {}).items():
        for e in entries:
            rel = e.get("path", "")
            abs_p = output_dir_p / rel
            declared_exists = bool(e.get("exists"))
            real_exists = abs_p.is_file()
            if declared_exists != real_exists:
                mismatches.append(
                    f"[{group}] {rel}: 清单声明 exists={declared_exists}, 实际 {real_exists}")
                continue
            if real_exists and e.get("size_bytes") is not None:
                real_size = abs_p.stat().st_size
                if real_size != e["size_bytes"]:
                    mismatches.append(
                        f"[{group}] {rel}: 大小不一致 (声明 {e['size_bytes']}, 实际 {real_size})")

    required_missing = manifest.get("required_missing", [])
    return {
        "ok": (not mismatches) and (not required_missing),
        "mismatches": mismatches,
        "warnings": warnings_,              # v0.3: 非致命提示 (如版本较旧的后向兼容读取)
        "required_missing": required_missing,
        "delivery_level": (manifest.get("delivery_level") or {}).get("level"),
        "manifest": manifest,
    }
