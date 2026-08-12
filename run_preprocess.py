"""
输电线路路径规划 - 预处理流水线 (M0-M3 独立版)

本脚本从原 v5.7 完整流水线 (M0-M6) 中单独提取了预处理环节 (M0→M1→M2→M3),
产出一个完整的"标准化输出包", 可直接被下游算法端 (M4-M6) 通过文件路径挂载使用.

用法:
    # 使用真实项目数据 (默认输出到 output/run_YYYYMMDD_HHMMSS/):
    python run_preprocess.py --project_dir <项目数据目录> --output_dir ./output

    # 使用内置模拟测试数据:
    python run_preprocess.py --generate_test --output_dir ./output

    # 指定 project.json 路径 (默认 project_dir/project.json):
    python run_preprocess.py --project_dir ./my_proj --project_json ./my_proj/cfg.json --output_dir ./out

输出目录 (v0.4.5):
    每次运行自动在 --output_dir 下创建 run_YYYYMMDD_HHMMSS/ 子目录,
    历史运行保留, 不会被下次覆盖. 这样:
      - Windows 下 QGIS/资源管理器锁住上次输出的 tif 也不影响新运行
      - 多次对比调参结果时能看到每次的独立产物

产出目录结构 (示例):
    output/
        run_20260424_163541/                      <- 本次运行的独立目录
            m0/                   -- 原始输入适配器产出
            m1/                   -- 语义映射产出
            m2/                   -- 几何预处理产出
            m3/                   -- 规则编译+栅格层产出
            manifest.json         -- 包完整性契约
            preprocessing_summary.json  -- 阶段耗时 / 要素计数 / 交付级别
            preprocessing.log     -- 日志

算法端(M4-M6)使用方式:
    给 M4PathPlanner 传入本次运行的 output_dir (带时间戳那一层),
    M4/M5/M6 会自动从 <run_dir>/m2/ 和 <run_dir>/m3/ 读取所需文件.
    控制对象也已经以 control_*.gpkg 形式保存在 m2/ 下.

    如果算法端不做代码改动, 可按原来的方式把 control_objects(dict[str, GeoDataFrame])
    传入 M4PathPlanner.run(); 本包提供了 utils.load_control_objects() 帮助函数
    直接从 <run_dir>/m2/ 还原这个 dict.
"""
import os
import sys
import argparse
import time
import logging

# ★Tier1 优化 1.4★ GDAL/rasterio 性能调优 (必须在任何 import rasterio 之前设置)
# 默认 GDAL_CACHEMAX 只有 5MB, 大栅格 IO 时频繁刷盘. 调到 2GB 后, M3 写盘阶段
# (forbidden/tower/lpcf/tscf 共 8 个 .tif, 10m 分辨率每张 ~30-100MB) 性能提升明显.
# GDAL_NUM_THREADS=ALL_CPUS 让 rasterio/GDAL 自动用所有 CPU 核做 IO 并行.
# 这两个变量都是 GDAL 标准调优, 不影响数据正确性.
# 若用户已经在 shell 里设置了, 不覆盖.
os.environ.setdefault("GDAL_CACHEMAX", "2048")
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

from utils.geo_utils import setup_logging, load_json, save_json, ensure_dir


def run_preprocess(project_dir: str, output_dir: str,
                   project_json_path: str = None,
                   cli_solver_overrides: dict = None) -> dict:
    """
    执行 M0 → M1 → M2 → M3 预处理流水线。

    Args:
        project_dir: 项目数据根目录 (内含 gdb/shp/tif/control/project.json ...)
        output_dir: 预处理输出目录
        project_json_path: project.json 路径 (默认 project_dir/project.json)
        cli_solver_overrides: 由 CLI 传入的 solver_params 覆盖 (优先级最高)

    Returns:
        dict: 含 m0_result / m1_result / m2_result / m3_result / timings / summary
    """
    # 延迟导入,避免没跑到的环节也产生副作用
    from modules.m0_input_adapter import M0InputAdapter
    from modules.m1_semantic_mapping import M1SemanticMapper
    from modules.m2_geometry_preprocessing import M2GeometryPreprocessor
    from modules.m3_rule_compile_and_output import M3RuleCompiler

    logger = logging.getLogger("transmission_planning")
    total_start = time.time()

    # ─── 加载 project.json ───────────────────────────────
    if project_json_path is None:
        project_json_path = os.path.join(project_dir, "project.json")
    if os.path.exists(project_json_path):
        project_config = load_json(project_json_path)
        logger.info(f"加载项目配置: {project_json_path}")
    else:
        logger.warning(f"project.json 未找到: {project_json_path}, 使用默认配置")
        project_config = {
            "project_name": "默认项目",
            "voltage_kv": 500,
            "ice_zone": 10,
            "wind_zone": "B",
            "source_crs": "EPSG:4490",
            "working_crs": "EPSG:4547",
            "data_availability": {
                "river_polygon": False,
                "wind_ice_zone_raster": False,
            },
        }

    project_config["_project_dir"] = project_dir

    # v0.2: CLI 覆盖 > project.json > 默认。合并到 solver_params 子字典。
    if cli_solver_overrides:
        sp = dict(project_config.get("solver_params", {}) or {})
        sp.update(cli_solver_overrides)
        project_config["solver_params"] = sp
        logger.info(f"CLI 覆盖 solver_params: {cli_solver_overrides}")

    ensure_dir(output_dir)
    timings = {}

    # ─── M0: 原始输入适配器 ─────────────────────────────
    logger.info("=" * 60)
    t = time.time()
    m0 = M0InputAdapter(project_dir, project_config, output_dir)
    m0_result = m0.run()
    timings["M0"] = round(time.time() - t, 1)
    logger.info(f"M0 耗时: {timings['M0']}s")

    # ─── M1: 分级语义映射 ───────────────────────────────
    logger.info("=" * 60)
    t = time.time()
    m1 = M1SemanticMapper(project_config, output_dir)
    m1_result = m1.run(m0_result["unified_gdf"], m0_result["control_objects"])
    timings["M1"] = round(time.time() - t, 1)
    logger.info(f"M1 耗时: {timings['M1']}s")

    # ─── M2: 几何预处理与专题对象构建 ───────────────────
    logger.info("=" * 60)
    t = time.time()
    m2 = M2GeometryPreprocessor(
        project_config, output_dir, m0_result["raster_inventory"]
    )
    m2_result = m2.run(m1_result["standardized_gdf"], m1_result["control_objects"])
    timings["M2"] = round(time.time() - t, 1)
    logger.info(f"M2 耗时: {timings['M2']}s")

    # ─── M3: 规则编译 + 标准化输出包 + 导航图骨架 ───────
    logger.info("=" * 60)
    t = time.time()
    m3 = M3RuleCompiler(project_config, output_dir)
    m3_result = m3.run(m2_result)
    timings["M3"] = round(time.time() - t, 1)
    logger.info(f"M3 耗时: {timings['M3']}s")

    # ─── 总结 ───────────────────────────────────────────
    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"预处理完成! 总耗时: {total_time:.1f}s "
                f"(M0={timings['M0']}s M1={timings['M1']}s "
                f"M2={timings['M2']}s M3={timings['M3']}s)")

    delivery_level = m3_result.get("preprocessing_report", {}).get("delivery_level", {})
    if delivery_level:
        logger.info(f"交付级别: {delivery_level.get('level', 'UNKNOWN')}")
        for reason in delivery_level.get("reasons", []):
            logger.info(f"  降级原因: {reason}")
        for severe in delivery_level.get("severe_reasons", []):
            logger.warning(f"  严重降级: {severe}")

    # v0.4: 显式展示 CRS 诊断与数据多样性结论, 避免"看似完美实则偏单一"
    crs_diag = m0_result.get("crs_diagnostic", {}) or {}
    crs_val = crs_diag.get("working_crs_validation") or {}
    if crs_val.get("severity") == "warning":
        logger.warning(f"  CRS 校验: {crs_val.get('message', '')}")
    diversity = m1_result.get("diversity_score", {}) or {}
    if diversity.get("warning"):
        logger.warning(f"  数据多样性: {diversity['warning']}")
    nav_h = (delivery_level.get("nav_graph_health") or {}) if delivery_level else {}
    if nav_h:
        logger.info(
            f"  骨架健康: node={nav_h.get('node_count')}, "
            f"edge={nav_h.get('edge_count')}, status={nav_h.get('status')}"
        )

    summary = {
        "project_name": project_config.get("project_name", ""),
        "project_dir": project_dir,
        "output_dir": output_dir,
        "total_time_s": round(total_time, 1),
        "timings": timings,
        "delivery_level": delivery_level,
        "modules": {
            "m0": {
                "gdb_count": len(m0_result.get("gdb_inventory", [])),
                "raster_count": len(m0_result.get("raster_inventory", [])),
                "vector_feature_count": len(m0_result.get("unified_gdf", [])),
                "control_object_keys": list(m0_result.get("control_objects", {}).keys()),
                # v0.4: CRS 诊断 (working_crs 合理性 / 轴序 / 起终点坐标)
                "crs_diagnostic": m0_result.get("crs_diagnostic", {}),
            },
            "m1": {
                "stats": m1_result.get("stats", {}),
                # v0.4: 数据源多样性评分
                "diversity_score": m1_result.get("diversity_score", {}),
            },
            "m2": {
                "forbidden_polygon_count": len(m2_result.get("forbidden_polygons", [])),
                "no_tower_polygon_count": len(m2_result.get("no_tower_polygons", [])),
                "cost_polygon_count": len(m2_result.get("cost_polygons", [])),
                "linear_cross_segment_count": len(m2_result.get("linear_cross_segments", [])),
                "preferred_corridor_count": len(m2_result.get("preferred_corridors", [])),
            },
            "m3": {
                "nav_graph_meta": m3_result.get("nav_graph_meta", {}),
                "solver_param_keys": list(m3_result.get("solver_params", {}).keys()),
            },
        },
        "downstream_hint": {
            "m2_dir": os.path.join(output_dir, "m2"),
            "m3_dir": os.path.join(output_dir, "m3"),
            "algorithm_entrypoint": (
                "算法端 (M4PathPlanner) 应以 output_dir 初始化, 自动读取 m3_dir/*.json、"
                "m3_dir/*.tif、m2_dir/*.gpkg; 控制对象可通过 utils.control_io.load_control_objects "
                "从 m2_dir 还原为 dict[str, GeoDataFrame]."
            ),
        },
    }
    save_json(summary, os.path.join(output_dir, "preprocessing_summary.json"))
    logger.info(f"总结文件: {os.path.join(output_dir, 'preprocessing_summary.json')}")

    # ─── v0.2: 写出 manifest.json — 下游算法端唯一契约文件 ─────
    try:
        from utils.manifest import compile_manifest
        nav_meta = m3_result.get("nav_graph_meta") or {}
        nav_status = nav_meta.get("status", "completed")
        manifest = compile_manifest(
            output_dir=output_dir,
            project_config={k: v for k, v in project_config.items() if not k.startswith("_")},
            m3_report=m3_result.get("preprocessing_report", {}),
            workspace=m3_result.get("workspace", {}),
            solver_param_keys=list(m3_result.get("solver_params", {}).keys()),
            rule_count=len((m3_result.get("rule_config", {}) or {}).get("compiled_rules", [])),
            timings={k: float(v) for k, v in timings.items()},
            nav_graph_status=nav_status,
            nav_graph_meta=nav_meta,  # v0.3: 带上完整骨架元数据, 让 manifest 声明 skeleton 契约
            compute_md5=True,
        )
        # ★Bug 1 修复 (v0.5+)★ 把 manifest 计算的 delivery_level 完整回写到 summary,
        # 替换 M3 内部 _decide_delivery_level 给出的版本.
        #
        # 根因: M3 内部计算的 delivery_level 不知道 allow_planar_2d_mode (那是 project_config 字段,
        # 不属于 M3 输入), 所以 DEM 缺失时 M3 内部一律标 SEVERE_DEGRADED;
        # 而 manifest.compile_manifest 走的是 Phase C 双层闸门, 知道 allow_planar_2d_mode,
        # 同样场景会标 PRELIMINARY_ROUTE_ONLY. 两个字段同时存在但语义不同, 运维/算法端会困惑.
        #
        # 修复后: summary.delivery_level 完整等于 manifest.delivery_level, 不再矛盾.
        # 同时新增 operational_mode 字段, 让 summary 也能展示 3D/2D 模式判定.
        manifest_dl = manifest.get("delivery_level", {})
        if manifest_dl:
            # 保留 M3 内部的 runtime_degrades / nav_graph_health / dem_quality 等审计细节,
            # 但 level / reasons / severe_reasons / preliminary_reasons / upgrade_actions
            # 全部以 manifest 为准.
            m3_internal = summary.get("delivery_level", {}) or {}
            merged_dl = dict(manifest_dl)  # 浅拷贝 manifest 版本
            # 保留 M3 内部细节字段 (manifest 没生成的)
            for keep_key in ("runtime_degrades", "runtime_degrades_summary",
                             "nav_graph_health", "dem_quality"):
                if keep_key in m3_internal and keep_key not in merged_dl:
                    merged_dl[keep_key] = m3_internal[keep_key]
            summary["delivery_level"] = merged_dl
        summary["operational_mode"] = manifest.get("operational_mode", {})
        summary["manifest_delivery_level"] = manifest_dl.get("level")
        summary["manifest_required_missing"] = manifest.get("required_missing", [])
        save_json(summary, os.path.join(output_dir, "preprocessing_summary.json"))
        # 把对齐后的最终级别再 log 一次, 避免之前的 "12:53:08 SEVERE_DEGRADED" 这种误导
        logger.info(f"最终交付级别 (manifest 闸门): {manifest_dl.get('level', 'UNKNOWN')}")
    except Exception as e:
        logger.error(f"manifest.json 写出失败 (不中断流水线, 但算法端需要它!): {e}")

    return {
        "m0_result": m0_result,
        "m1_result": m1_result,
        "m2_result": m2_result,
        "m3_result": m3_result,
        "timings": timings,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="输电线路路径规划 - 预处理流水线 (M0-M3, 独立版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project_dir", type=str,
                        help="项目数据根目录 (内含 gdb/shp/tif/control/project.json)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="预处理输出基目录 (默认 ./output); 每次运行会在其下创建 "
                             "run_<YYYYMMDD>_<HHMMSS>/ 子目录存放结果")
    parser.add_argument("--project_json", type=str, default=None,
                        help="project.json 路径 (默认 project_dir/project.json)")
    parser.add_argument("--generate_test", action="store_true",
                        help="生成内置模拟测试数据并运行预处理")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="配合 --generate_test 使用: 指定模拟测试数据的生成目录 "
                             "(默认 <output_dir>/test_project_data)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别 (默认 INFO)")
    parser.add_argument("--skeleton_mode", type=str, default=None,
                        choices=["immediate", "deferred", "skip"],
                        help="骨架构建模式 (覆盖 project.json 里的 solver_params.skeleton_build_mode); "
                             "immediate=M3 立即构建; "
                             "deferred=M3 跳过, 下游按需构建; "
                             "skip=不构建且不标记可构建(v0.6 默认)")
    # ★P0 (v0.6)★ 新增输出开关 CLI (优先级高于 project.json)
    parser.add_argument("--enable_fine_resolution", action="store_true", default=False,
                        help="产出 *_fine.tif 精分辨率栅格 (默认关; 算法端当前用单一粗分辨率)")
    parser.add_argument("--emit_unconsumed_outputs", action="store_true", default=False,
                        help="产出算法端不消费的栅格/骨架 (lpcf/tscf/mask/difficulty/nav_graph 等; "
                             "默认关; 仅排查/可视化或后续启用时开)")
    parser.add_argument("--bbox_buffer_km", type=float, default=None,
                        help="起终点窗口外扩缓冲 (km, 覆盖 project.json; v0.6 默认 5)")
    parser.add_argument("--disable_river_real_barrier", action="store_true", default=False,
                        help="回退河流为 v0.5 圆盘禁区+窗口 (默认走 P7: 宽河真实多边形禁区+窄河角度线段)")
    args = parser.parse_args()

    # v0.4.5: 每次运行自动在 output_dir 下建一个时间戳子目录, 避免:
    #   1. 被 QGIS / 资源管理器锁住导致 rasterio 写 tif 报 Permission denied
    #   2. 新一次运行覆盖掉上一次的结果, 丢失历史对比数据
    base_output_dir = args.output_dir
    ensure_dir(base_output_dir)
    run_dir = os.path.join(base_output_dir, "run_" + time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(run_dir)

    setup_logging(log_dir=run_dir,
                  level=getattr(logging, args.log_level.upper()))

    logger = logging.getLogger("transmission_planning")
    logger.info(f"本次运行输出目录: {run_dir}")

    # CLI 覆盖: 优先于 project.json (合并到 solver_params)
    cli_overrides = {}
    if args.skeleton_mode:
        cli_overrides["skeleton_build_mode"] = args.skeleton_mode
    # ★P0 (v0.6)★ 输出开关 (store_true: 仅在显式传入时覆盖, 否则不动 project.json)
    if args.enable_fine_resolution:
        cli_overrides["enable_fine_resolution"] = True
    if args.emit_unconsumed_outputs:
        cli_overrides["emit_unconsumed_outputs"] = True
    if args.bbox_buffer_km is not None:
        cli_overrides["bbox_start_end_buffer_km"] = float(args.bbox_buffer_km)
    # ★P7 (v0.6)★ 河流真实障碍开关 (默认 True; 传 --disable_ 才回退 v0.5 圆盘)
    if args.disable_river_real_barrier:
        cli_overrides["enable_river_real_barrier"] = False

    if args.generate_test:
        from tests.generate_test_data import generate_test_project
        test_dir = args.test_dir or os.path.join(run_dir, "test_project_data")
        generate_test_project(test_dir)
        run_preprocess(test_dir, run_dir, cli_solver_overrides=cli_overrides)
    elif args.project_dir:
        run_preprocess(args.project_dir, run_dir, args.project_json,
                       cli_solver_overrides=cli_overrides)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
