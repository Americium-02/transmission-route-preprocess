#!/usr/bin/env python3
"""
骨架阈值校准脚本 (v0.2 新增)

目的:
    输入一份已经跑完 M2 的工程数据 (有 m3/forbidden_mask_50m.tif 和 m3/lpcf_50m.tif),
    扫描 500m×500m 块级别的三维直方图(forbid_ratio / min_dist_m / hc_ratio),
    帮你给新工程数据校准 skeleton_budget_nodes / 三档阈值。

背景:
    阶段 3 重写后, v0.2 默认阈值是:
      dense : forbid_ratio > 15% AND min_dist < 100m  或  hc_ratio > 50%
      complex: min_dist < 300m  或  hc_ratio > 20%
    这套阈值是在 "20km×20km 禁区占比 5% 病态栅格" 上验证好的。如果你的工程数据
    禁区分布差异很大(比如 30km×50km 极度稀疏 / 5km×5km 高密度城市路网),
    最好用这个脚本扫一遍真实数据, 微调阈值。

典型用法:
    # 已经跑过 preprocess v0.2, 得到了 output/m3/forbidden_mask_50m.tif 等
    python scripts/calibrate_skeleton_thresholds.py \
        --forbidden_mask output/m3/forbidden_mask_50m.tif \
        --lpcf           output/m3/lpcf_50m.tif \
        [--block_m 500] [--high_cost_lpcf 4.0]

输出:
    - 三维块特征直方图 (forbid_ratio / min_dist_m / hc_ratio 的分位数表)
    - 按当前 v0.2 阈值分类的 dense/complex/open 块数
    - 推荐阈值 (按 "让 dense 块占比落在 5%-15%" 的目标自动反推)
    - 推荐的 skeleton_budget_nodes (基于当前工程的面积 × 节点密度)

注: 本脚本只读, 不修改输入文件。
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# 只依赖 numpy + scipy + rasterio; 不依赖 shapely/geopandas
try:
    import rasterio
    from scipy.ndimage import distance_transform_edt
except ImportError as e:
    sys.stderr.write(
        f"缺少依赖: {e}\n"
        "本脚本需要 rasterio 和 scipy。请运行 pip install rasterio scipy\n"
    )
    sys.exit(2)


def _classify_block_v02(fr: float, md_m: float, hcr: float,
                        t_dense_fr=0.15, t_dense_md=100.0, t_dense_hcr=0.50,
                        t_cx_md=300.0, t_cx_hcr=0.20) -> str:
    """与 M3 里 _generate_skeleton_nodes 的 _classify_block 行为完全一致"""
    if (fr > t_dense_fr and md_m < t_dense_md) or hcr > t_dense_hcr:
        return "dense"
    if md_m < t_cx_md or hcr > t_cx_hcr:
        return "complex"
    return "open"


def scan_blocks(forbidden: np.ndarray, lpcf: np.ndarray, res_m: float,
                block_m: float = 500.0, high_cost_lpcf: float = 4.0) -> np.ndarray:
    """
    按 block_m 粒度扫描, 返回每块的特征数组 (shape = [n_blocks, 3]):
      列 0: forbid_ratio
      列 1: min_dist_m
      列 2: hc_ratio
    """
    h, w = forbidden.shape
    block_px = max(1, int(block_m / res_m))
    try:
        dist_to_forbid_px = distance_transform_edt(forbidden == 0)
    except Exception:
        dist_to_forbid_px = np.full((h, w), 1e6, dtype=np.float32)

    high_cost = (lpcf > high_cost_lpcf) & (lpcf < 999999)
    features = []
    for br in range(0, h, block_px):
        for bc in range(0, w, block_px):
            r_hi = min(br + block_px, h)
            c_hi = min(bc + block_px, w)
            if r_hi <= br or c_hi <= bc:
                continue
            blk_f = forbidden[br:r_hi, bc:c_hi]
            blk_hc = high_cost[br:r_hi, bc:c_hi]
            fr = float(blk_f.sum()) / max(blk_f.size, 1)
            hcr = float(blk_hc.sum()) / max(blk_hc.size, 1)
            md_m = float(dist_to_forbid_px[br:r_hi, bc:c_hi].min()) * res_m
            features.append((fr, md_m, hcr))
    return np.array(features, dtype=np.float64)


def recommend_thresholds(features: np.ndarray,
                         target_dense_ratio=(0.05, 0.15)) -> dict:
    """
    基于块特征分布, 反推一组让 dense 占比落在 [5%, 15%] 的阈值。
    策略: 固定 dense_hcr=50%, 搜 forbid_ratio 和 min_dist_m 的阈值对,
          让 dense 占比最接近 (0.05+0.15)/2 = 10%.
    """
    n = len(features)
    if n == 0:
        return {"note": "无块数据, 无法推荐"}

    target = sum(target_dense_ratio) / 2.0

    # 搜索 grid
    fr_candidates = [0.05, 0.10, 0.15, 0.20, 0.25]
    md_candidates = [50.0, 75.0, 100.0, 150.0, 200.0]
    best = None
    for t_fr in fr_candidates:
        for t_md in md_candidates:
            classes = np.array([
                _classify_block_v02(f, m, h,
                                    t_dense_fr=t_fr, t_dense_md=t_md)
                for (f, m, h) in features
            ])
            dense_cnt = int((classes == "dense").sum())
            dense_ratio = dense_cnt / n
            diff = abs(dense_ratio - target)
            if best is None or diff < best["diff"]:
                best = {
                    "t_dense_fr": t_fr,
                    "t_dense_md": t_md,
                    "dense_ratio": dense_ratio,
                    "dense_cnt": dense_cnt,
                    "diff": diff,
                }
    return best


def quantiles(arr: np.ndarray, qs=(0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)) -> dict:
    if len(arr) == 0:
        return {f"p{int(q*100)}": None for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(arr, q)) for q in qs}


def main():
    parser = argparse.ArgumentParser(description="骨架阈值校准 (v0.2)")
    parser.add_argument("--forbidden_mask", required=True,
                        help="M3 产出的 forbidden_mask_50m.tif 路径")
    parser.add_argument("--lpcf", required=True,
                        help="M3 产出的 lpcf_50m.tif 路径")
    parser.add_argument("--block_m", type=float, default=500.0,
                        help="块尺寸, 单位米 (与 M3 的 BLOCK_SIZE_M 对齐, 默认 500)")
    parser.add_argument("--high_cost_lpcf", type=float, default=4.0,
                        help="高代价 LPCF 阈值 (与 M3 的 HIGH_COST_LPCF 对齐, 默认 4.0)")
    parser.add_argument("--avg_nodes_per_km2", type=float, default=60.0,
                        help="用于反推 skeleton_budget_nodes 的每平方公里节点数 (默认 60)")
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    if not Path(args.forbidden_mask).exists():
        sys.exit(f"forbidden_mask 不存在: {args.forbidden_mask}")
    if not Path(args.lpcf).exists():
        sys.exit(f"lpcf 不存在: {args.lpcf}")

    # 1) 读栅格 + 元数据
    with rasterio.open(args.forbidden_mask) as ds:
        forbidden = ds.read(1)
        res_m = abs(ds.res[0])
        h, w = ds.shape
        bbox_km = ((ds.bounds.right - ds.bounds.left) / 1000.0,
                   (ds.bounds.top - ds.bounds.bottom) / 1000.0)
        bbox_area_km2 = bbox_km[0] * bbox_km[1]
    with rasterio.open(args.lpcf) as ds:
        lpcf = ds.read(1)

    # v0.3: 有效面积 = 非禁区 + 非 nodata 的像素 * res^2
    # 原先用 bbox_area_km2 算节点预算, 如果 bbox 里有大量海洋/境外/nodata 区域,
    # 预算会被虚高, 导致 skeleton 节点冗余。改用有效面积更贴真实工作量。
    nodata_val = -1
    is_valid = (forbidden != nodata_val) & (forbidden != 255)
    is_non_forbidden = (forbidden == 0) & is_valid
    effective_area_km2 = float(is_non_forbidden.sum()) * (res_m ** 2) / 1e6
    # 兜底: 如果 effective 面积接近 0 (数据异常), 回退到 bbox 面积
    area_km2 = effective_area_km2 if effective_area_km2 > 0.1 * bbox_area_km2 \
               else bbox_area_km2

    print(f"=" * 70)
    print(f"骨架阈值校准: {args.forbidden_mask}")
    print(f"  栅格尺寸  : {h} x {w} @ {res_m:.1f}m")
    print(f"  工作区 bbox: {bbox_km[0]:.2f}km × {bbox_km[1]:.2f}km "
          f"(矩形面积 {bbox_area_km2:.1f} km²)")
    print(f"  有效面积  : {effective_area_km2:.1f} km² "
          f"(非禁区非 nodata, 占 bbox {effective_area_km2/max(bbox_area_km2,1e-6):.0%})")
    print(f"  用于节点预算的面积: {area_km2:.1f} km²")
    print(f"  禁区占比  : {forbidden.sum() / max(forbidden.size, 1):.2%}")
    print(f"=" * 70)

    # 2) 扫块特征
    print(f"\n扫 {args.block_m:.0f}m 块特征...")
    features = scan_blocks(forbidden, lpcf, res_m,
                           block_m=args.block_m,
                           high_cost_lpcf=args.high_cost_lpcf)
    n_blocks = len(features)
    print(f"  总块数 = {n_blocks}")

    # 3) 分位数
    print(f"\n── 块特征分位数 ──")
    for col_idx, col_name in enumerate(["forbid_ratio", "min_dist_m", "hc_ratio"]):
        q = quantiles(features[:, col_idx])
        q_fmt = ", ".join(f"{k}={v:.3f}" if v is not None else f"{k}=N/A"
                          for k, v in q.items())
        print(f"  {col_name:13s}: {q_fmt}")

    # 4) 当前 v0.2 默认阈值下的分类
    print(f"\n── 按 v0.2 默认阈值 (dense fr>0.15 & md<100m 或 hc>0.50; "
          f"complex md<300m 或 hc>0.20) 分类 ──")
    classes = np.array([
        _classify_block_v02(f, m, h) for (f, m, h) in features
    ])
    for level in ("dense", "complex", "open"):
        cnt = int((classes == level).sum())
        ratio = cnt / max(n_blocks, 1)
        print(f"  {level:8s}: {cnt:5d} ({ratio:.1%})")

    # 5) 推荐阈值
    print(f"\n── 阈值推荐 (目标 dense 块占比 5%-15%) ──")
    rec = recommend_thresholds(features)
    if rec is None or "note" in rec:
        print(f"  {rec.get('note', '无推荐')}")
    else:
        print(f"  推荐 t_dense_fr = {rec['t_dense_fr']:.2f}")
        print(f"  推荐 t_dense_md = {rec['t_dense_md']:.0f} m")
        print(f"  此阈值下 dense = {rec['dense_cnt']} ({rec['dense_ratio']:.1%})")
        if abs(rec["dense_ratio"] - 0.10) > 0.08:
            print(f"  ⚠ 当前数据分布偏离目标较多, 可能需要手工调整 t_dense_hcr / t_cx_md")

    # 6) 推荐 skeleton_budget_nodes
    print(f"\n── skeleton_budget_nodes 推荐 ──")
    recommended_budget = int(area_km2 * args.avg_nodes_per_km2 * 1.3)  # 留 30% 冗余
    print(f"  工作区面积 {area_km2:.1f} km² × {args.avg_nodes_per_km2:.0f} 节点/km² × 1.3 冗余")
    print(f"  推荐 skeleton_budget_nodes = {recommended_budget}")
    print(f"  (v0.2 默认 12000, 适合 ~150 km² 工程; 你这个工程建议设为 {recommended_budget})")

    # 7) solver_params 片段
    print(f"\n── 建议写入 project.json.solver_params 的片段 ──")
    suggested = {
        "skeleton_budget_nodes": recommended_budget,
    }
    if rec and "t_dense_fr" in rec:
        suggested["_calibration_note"] = (
            f"推荐阈值(手工改代码使用): t_dense_fr={rec['t_dense_fr']}, "
            f"t_dense_md={rec['t_dense_md']}m; 当前这些阈值硬编码在 "
            f"m3_rule_compile_and_output.py 的 THRESH 字典里, "
            f"如需上线请同时改该字典。"
        )
    import json
    print(json.dumps({"solver_params": suggested}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
