"""
平行关系严格几何判定模块 (v5.3)

解决问题清单 2.1~2.4:
  - parallel_relation(): 严格判定路径段与参照线的平行关系
  - build_parallel_valid_zone(): 构造有效平行带（扣除保护范围与生态敏感区）
  - compute_parallel_reward(): 按 W/1000m × 实际平行长度 计算奖励
  - check_forbidden_parallel(): 800kV+ 严格禁止并行判定
  - 统一支持道路贴近、110kV+贴近、800kV+禁并行

用法:
  from utils.parallel_geometry import ParallelAnalyzer
  analyzer = ParallelAnalyzer(solver_params)
  result = analyzer.analyze_path_parallel(path_line, target_lines, ...)
"""
import math
import logging
from typing import List, Dict, Tuple, Optional
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import unary_union, substring
import numpy as np

logger = logging.getLogger("transmission_planning.parallel")


class ParallelAnalyzer:
    """平行关系严格几何判定器"""

    def __init__(self, solver_params: dict = None):
        params = solver_params or {}
        self.proximity_m = params.get("parallel_proximity_m", 200)
        self.max_angle_deg = params.get("parallel_min_angle_deg", 15)
        self.min_length_m = params.get("parallel_min_length_m", 500)
        self.hv_exclusion_m = params.get("high_voltage_parallel_exclusion_m", 600)
        self.hv_threshold_kv = params.get("high_voltage_parallel_threshold_kv", 800)
        # 分段长度：用于逐段判定角度
        self.check_seg_length_m = 50.0

    def parallel_relation(self, path_segment: LineString,
                          target_line: LineString,
                          dist_thresh_m: float = None,
                          angle_thresh_deg: float = None,
                          min_len_m: float = None) -> dict:
        """
        严格判定路径段与目标线的平行关系。

        条件（三条同时满足）：
          1. 路径与参照线距离 ≤ dist_thresh_m
          2. 路径与参照线夹角 ≤ angle_thresh_deg
          3. 满足条件的连续长度 ≥ min_len_m

        Args:
            path_segment: 路径线段
            target_line: 参照线（道路/输电线路等）
            dist_thresh_m: 距离阈值（默认200m）
            angle_thresh_deg: 角度阈值（默认15°）
            min_len_m: 最小连续长度（默认500m）

        Returns:
            {
                "is_parallel": bool,
                "parallel_segments": [(start_dist, end_dist, length), ...],
                "total_parallel_length_m": float,
                "max_continuous_length_m": float,
            }
        """
        dist_thresh = dist_thresh_m or self.proximity_m
        angle_thresh = angle_thresh_deg or self.max_angle_deg
        min_len = min_len_m or self.min_length_m

        if path_segment is None or path_segment.is_empty:
            return self._empty_result()
        if target_line is None or target_line.is_empty:
            return self._empty_result()

        path_len = path_segment.length
        if path_len < min_len:
            return self._empty_result()

        # 沿路径按check_seg_length逐段检查
        seg_len = self.check_seg_length_m
        qualifying_segments = []  # [(start_dist, end_dist)]
        cur_start = None

        dist = 0.0
        while dist < path_len:
            end = min(dist + seg_len, path_len)
            mid_dist = (dist + end) / 2.0

            # 条件1: 距离检查
            pt = path_segment.interpolate(mid_dist)
            nearest_dist = target_line.distance(pt)
            dist_ok = nearest_dist <= dist_thresh

            # 条件2: 角度检查
            angle_ok = False
            if dist_ok:
                p1 = path_segment.interpolate(dist)
                p2 = path_segment.interpolate(end)
                if p1.distance(p2) > 0.1:
                    path_az = _azimuth(p1.x, p1.y, p2.x, p2.y)
                    # 找target_line上最近点处的方位角
                    target_az = self._target_azimuth_at(target_line, pt)
                    if target_az is not None:
                        angle_diff = _crossing_angle_full(path_az, target_az)
                        angle_ok = angle_diff <= angle_thresh

            if dist_ok and angle_ok:
                if cur_start is None:
                    cur_start = dist
            else:
                if cur_start is not None:
                    qualifying_segments.append((cur_start, dist))
                    cur_start = None

            dist = end

        # 收尾
        if cur_start is not None:
            qualifying_segments.append((cur_start, path_len))

        # 过滤连续长度 >= min_len
        valid_segments = []
        for s, e in qualifying_segments:
            length = e - s
            if length >= min_len:
                valid_segments.append((s, e, length))

        total_length = sum(seg[2] for seg in valid_segments)
        max_length = max((seg[2] for seg in valid_segments), default=0)

        return {
            "is_parallel": len(valid_segments) > 0,
            "parallel_segments": valid_segments,
            "total_parallel_length_m": round(total_length, 1),
            "max_continuous_length_m": round(max_length, 1),
        }

    def build_parallel_valid_zone(self, target_line: LineString,
                                  parallel_range_m: float,
                                  protection_buffers: list = None,
                                  eco_sensitive_polygons: list = None):
        """
        构造有效平行带（问题清单2.4）：
            parallel_valid_zone = buffer(parallel_range_m)
                                - target_protection_buffer
                                - eco_sensitive_union

        Args:
            target_line: 参照线
            parallel_range_m: 平行范围（如200m）
            protection_buffers: [(geometry, buffer_m), ...] 保护范围
            eco_sensitive_polygons: [geometry, ...] 生态敏感区

        Returns:
            有效平行带 geometry（可能为空）
        """
        if target_line is None or target_line.is_empty:
            return None

        zone = target_line.buffer(parallel_range_m)

        # 扣除保护范围
        if protection_buffers:
            for geom, buf_m in protection_buffers:
                if geom and not geom.is_empty:
                    exclude = geom if buf_m <= 0 else geom.buffer(buf_m)
                    zone = zone.difference(exclude)
                    if zone.is_empty:
                        return None

        # 扣除生态敏感区
        if eco_sensitive_polygons:
            valid_polys = [g for g in eco_sensitive_polygons if g and not g.is_empty]
            if valid_polys:
                eco_union = unary_union(valid_polys)
                zone = zone.difference(eco_union)
                if zone.is_empty:
                    return None

        return zone

    def compute_parallel_reward(self, path_line: LineString,
                                target_line: LineString,
                                reward_per_1000m: float,
                                dist_thresh_m: float = None,
                                angle_thresh_deg: float = None,
                                min_len_m: float = None,
                                valid_zone=None) -> dict:
        """
        按 W/1000m × 实际平行长度 计算奖励（问题清单2.3）

        Args:
            path_line: 路径线
            target_line: 参照线
            reward_per_1000m: 每1000m奖励值（W/1000m）
            valid_zone: 有效平行带（可选，如果传入则只在带内计算）

        Returns:
            {
                "reward_total": float,
                "parallel_length_m": float,
                "parallel_segments": [...],
                "is_rewarded": bool,
            }
        """
        # 如果有有效平行带，先裁剪路径到带内
        effective_path = path_line
        if valid_zone and not valid_zone.is_empty:
            try:
                clipped = path_line.intersection(valid_zone)
                if clipped.is_empty:
                    return {"reward_total": 0, "parallel_length_m": 0,
                            "parallel_segments": [], "is_rewarded": False}
                # 可能返回MultiLineString
                if clipped.geom_type == "MultiLineString":
                    effective_path = clipped
                elif clipped.geom_type == "LineString":
                    effective_path = clipped
                else:
                    return {"reward_total": 0, "parallel_length_m": 0,
                            "parallel_segments": [], "is_rewarded": False}
            except Exception:
                effective_path = path_line

        # 对有效路径段逐个检查平行关系
        if effective_path.geom_type == "MultiLineString":
            lines = list(effective_path.geoms)
        else:
            lines = [effective_path]

        all_segments = []
        total_parallel_length = 0

        for line in lines:
            if line.length < 1:
                continue
            result = self.parallel_relation(
                line, target_line,
                dist_thresh_m=dist_thresh_m,
                angle_thresh_deg=angle_thresh_deg,
                min_len_m=min_len_m
            )
            all_segments.extend(result["parallel_segments"])
            total_parallel_length += result["total_parallel_length_m"]

        reward_total = reward_per_1000m * (total_parallel_length / 1000.0)

        return {
            "reward_total": round(reward_total, 2),
            "parallel_length_m": round(total_parallel_length, 1),
            "parallel_segments": all_segments,
            "is_rewarded": total_parallel_length > 0,
        }

    def check_forbidden_parallel(self, path_line: LineString,
                                 hv_line: LineString,
                                 voltage_kv: int,
                                 exclusion_m: float = None) -> dict:
        """
        800kV+ 严格禁止并行判定（问题清单2.2）

        真实语义：
          只有当路径与800kV+线路形成并行关系（距离<600m、角度<15°、长度>500m）时，
          才属于禁止并行。垂直穿越、短时靠近、非并行状态下的经过不算。

        Returns:
            {
                "is_forbidden": bool,
                "forbidden_segments": [...],
                "total_forbidden_length_m": float,
                "severity": str,
            }
        """
        if voltage_kv < self.hv_threshold_kv:
            return {"is_forbidden": False, "forbidden_segments": [],
                    "total_forbidden_length_m": 0, "severity": "NONE"}

        excl = exclusion_m or self.hv_exclusion_m
        result = self.parallel_relation(
            path_line, hv_line,
            dist_thresh_m=excl,
            angle_thresh_deg=self.max_angle_deg,
            min_len_m=self.min_length_m
        )

        return {
            "is_forbidden": result["is_parallel"],
            "forbidden_segments": result["parallel_segments"],
            "total_forbidden_length_m": result["total_parallel_length_m"],
            "severity": "CRITICAL" if result["is_parallel"] else "NONE",
        }

    def analyze_path_parallel(self, path_line: LineString,
                              corridors: list,
                              forbidden_polygons: list = None,
                              eco_sensitive_polygons: list = None) -> dict:
        """
        综合分析路径的所有平行关系。

        Args:
            path_line: 路径线
            corridors: [{
                "geometry": LineString, "level2": str,
                "parallel_reward": float, "parallel_range_m": float,
                "voltage_kv": int, "rule_id": int,
                "buffer_m": float,  # 保护范围
            }, ...]
            forbidden_polygons: 禁区面列表（用于扣除）
            eco_sensitive_polygons: 生态敏感区面列表（用于扣除）

        Returns:
            {
                "parallel_rewards": [...],
                "forbidden_parallels": [...],
                "total_reward": float,
                "total_reward_breakdown": {...},
            }
        """
        rewards = []
        forbidden_parallels = []
        total_reward = 0.0
        reward_breakdown = {}

        for corr in corridors:
            target_line = corr.get("geometry")
            if target_line is None or target_line.is_empty:
                continue

            level2 = corr.get("level2", "")
            voltage_kv = corr.get("voltage_kv", 0)
            reward_per_1000m = corr.get("parallel_reward", 0)
            range_m = corr.get("parallel_range_m", self.proximity_m)
            buf_m = corr.get("buffer_m", 0)

            # 构造保护范围
            protection = [(target_line, buf_m)] if buf_m > 0 else None

            # 构造有效平行带
            valid_zone = self.build_parallel_valid_zone(
                target_line, range_m,
                protection_buffers=protection,
                eco_sensitive_polygons=eco_sensitive_polygons
            )

            # 计算贴近奖励
            if reward_per_1000m and reward_per_1000m > 0:
                reward_result = self.compute_parallel_reward(
                    path_line, target_line, reward_per_1000m,
                    dist_thresh_m=range_m,
                    valid_zone=valid_zone
                )
                if reward_result["is_rewarded"]:
                    reward_result["level2"] = level2
                    reward_result["rule_id"] = corr.get("rule_id")
                    rewards.append(reward_result)
                    total_reward += reward_result["reward_total"]
                    reward_breakdown[level2] = reward_result["reward_total"]

            # 800kV+ 禁止并行检查
            if voltage_kv >= self.hv_threshold_kv:
                forbidden_result = self.check_forbidden_parallel(
                    path_line, target_line, voltage_kv
                )
                if forbidden_result["is_forbidden"]:
                    forbidden_result["level2"] = level2
                    forbidden_parallels.append(forbidden_result)

        return {
            "parallel_rewards": rewards,
            "forbidden_parallels": forbidden_parallels,
            "total_reward": round(total_reward, 2),
            "total_reward_breakdown": reward_breakdown,
        }

    # ─── 辅助方法 ─────────────────────────────────────────

    def _target_azimuth_at(self, target_line, point) -> Optional[float]:
        """获取目标线在最近点处的方位角"""
        try:
            proj_dist = target_line.project(point)
            total = target_line.length
            # 取前后各25m的段来计算方位角
            d1 = max(0, proj_dist - 25)
            d2 = min(total, proj_dist + 25)
            if d2 - d1 < 1:
                return None
            p1 = target_line.interpolate(d1)
            p2 = target_line.interpolate(d2)
            return _azimuth(p1.x, p1.y, p2.x, p2.y)
        except Exception:
            return None

    @staticmethod
    def _empty_result():
        return {
            "is_parallel": False,
            "parallel_segments": [],
            "total_parallel_length_m": 0,
            "max_continuous_length_m": 0,
        }


def _azimuth(x1, y1, x2, y2) -> float:
    """方位角（0-360°）"""
    dx = x2 - x1
    dy = y2 - y1
    return math.degrees(math.atan2(dx, dy)) % 360


def _crossing_angle_full(az1: float, az2: float) -> float:
    """两方位角之间的夹角（0-180° → 映射到0-90°锐角）"""
    diff = abs(az1 - az2) % 180
    return min(diff, 180 - diff)
