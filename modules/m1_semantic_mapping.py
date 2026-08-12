"""
M1: 分级语义映射与字段标准化
- 第一级：确定性映射（三区三线等标准数据源）
- 第二级：模糊匹配（字段值归一化匹配，置信度判定）
- 第三级：人工确认（输出待确认列表）
- 输出：semantic_mapping_report.json, 标准化矢量集
"""
import os
import re
import logging
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import geopandas as gpd
import pandas as pd

from utils.geo_utils import load_json, save_json, ensure_dir, get_config_dir

logger = logging.getLogger("transmission_planning.m1")


# ─── 第一级：确定性映射表 ──────────────────────────────────
# theme (来自M0的 _theme) → (level1, level2)
DETERMINISTIC_THEME_MAP = {
    "永久基本农田": ("生态敏感点", "基本农田"),
    "城镇开发边界": ("重要设施与政府规划敏感点", "城镇规划区"),
    "生态保护红线": ("生态敏感点", "生态保护红线（一般区域）"),
    "耕地保护目标": ("生态敏感点", "一般农田"),
}

# ─── 第二级：模糊匹配归一化表 ─────────────────────────────
# 原始字段值 → (level1, level2)
NORMALIZE_MAP = {
    # 生态敏感点
    "森林公园": ("生态敏感点", "森林公园"),
    "自然保护区核心区": ("生态敏感点", "自然保护区核心区"),
    "自然保护区-核心区": ("生态敏感点", "自然保护区核心区"),
    "自然保护区（核心）": ("生态敏感点", "自然保护区核心区"),
    "核心区": ("生态敏感点", "自然保护区核心区"),
    "自然保护区缓冲区": ("生态敏感点", "自然保护区缓冲区"),
    "自然保护区-缓冲区": ("生态敏感点", "自然保护区缓冲区"),
    "缓冲区": ("生态敏感点", "自然保护区缓冲区"),
    "自然保护区试验区": ("生态敏感点", "自然保护区试验区"),
    "自然保护区-试验区": ("生态敏感点", "自然保护区试验区"),
    "试验区": ("生态敏感点", "自然保护区试验区"),
    "一级水源保护区": ("生态敏感点", "一级水源保护区"),
    "二级水源保护区": ("生态敏感点", "二级水源保护区"),
    "水源准保护区": ("生态敏感点", "水源准保护区"),
    "生态保护红线": ("生态敏感点", "生态保护红线（一般区域）"),
    "生态红线": ("生态敏感点", "生态保护红线（一般区域）"),
    "生态保护红线（重要区域）": ("生态敏感点", "生态保护红线（重要区域）"),
    "生态保护红线（一般区域）": ("生态敏感点", "生态保护红线（一般区域）"),
    "国有林场": ("生态敏感点", "国有林场"),
    "一级林地": ("生态敏感点", "一级林地"),
    "二级林地": ("生态敏感点", "二级林地"),
    "基本农田": ("生态敏感点", "基本农田"),
    "永久基本农田": ("生态敏感点", "基本农田"),
    "一般农田": ("生态敏感点", "一般农田"),
    "湿地公园": ("生态敏感点", "湿地公园"),
    "地质公园": ("生态敏感点", "地质公园"),
    "风景名胜区": ("生态敏感点", "风景名胜区"),
    "海洋保护区": ("生态敏感点", "海洋保护区"),
    # 重要设施与政府规划敏感点
    "城镇规划区": ("重要设施与政府规划敏感点", "城镇规划区"),
    "城镇开发边界": ("重要设施与政府规划敏感点", "城镇规划区"),
    "禁止建设区": ("重要设施与政府规划敏感点", "禁止建设区"),
    "建筑物": ("重要设施与政府规划敏感点", "建筑物"),
    "房屋": ("重要设施与政府规划敏感点", "建筑物"),
    "其他规划用地": ("重要设施与政府规划敏感点", "其他规划用地"),
    "矿产资源": ("重要设施与政府规划敏感点", "矿产资源"),
    "矿区": ("重要设施与政府规划敏感点", "矿产资源"),
    "采石场": ("重要设施与政府规划敏感点", "采石场"),
    "军事敏感点": ("重要设施与政府规划敏感点", "军事敏感点"),
    "军事设施": ("重要设施与政府规划敏感点", "军事敏感点"),
    "无线电设施": ("重要设施与政府规划敏感点", "无线电设施"),
    "导航台": ("重要设施与政府规划敏感点", "导航台"),
    "炸药库": ("重要设施与政府规划敏感点", "炸药库"),
    "油气存储站": ("重要设施与政府规划敏感点", "油气存储站"),
    "地震地磁台": ("重要设施与政府规划敏感点", "地震地磁台"),
    "气象站": ("重要设施与政府规划敏感点", "气象站"),
    # 交通敏感点
    "机场": ("交通敏感点", "机场"),
    "铁路": ("交通敏感点", "普通铁路（标准轨距）"),
    "普通铁路": ("交通敏感点", "普通铁路（标准轨距）"),
    "普通铁路（标准轨距）": ("交通敏感点", "普通铁路（标准轨距）"),
    "普通铁路（窄轨）": ("交通敏感点", "普通铁路（窄轨）"),
    "高铁": ("交通敏感点", "高铁"),
    "高速铁路": ("交通敏感点", "高铁"),
    "铁路-高铁": ("交通敏感点", "高铁"),
    "高速公路": ("交通敏感点", "高速公路"),
    "高速": ("交通敏感点", "高速公路"),
    "国道": ("交通敏感点", "国道"),
    "省道": ("交通敏感点", "省道"),
    "一般公路": ("交通敏感点", "一般公路"),
    "县道": ("交通敏感点", "一般公路"),
    "乡道": ("交通敏感点", "一般公路"),
    "村道": ("交通敏感点", "一般公路"),
    "电车道": ("交通敏感点", "电车道（有轨及无轨）"),
    "有轨电车": ("交通敏感点", "电车道（有轨及无轨）"),
    # 电力设施
    "1000kV输电线路": ("电力设施敏感点", "1000kV输电线路"),
    "±800kV输电线路": ("电力设施敏感点", "±800kV输电线路"),
    "750kV输电线路": ("电力设施敏感点", "750kV输电线路"),
    "500kV输电线路": ("电力设施敏感点", "500kV输电线路"),
    "±500kV输电线路": ("电力设施敏感点", "±500kV输电线路"),
    "400kV输电线路": ("电力设施敏感点", "400kV输电线路"),
    "330kV输电线路": ("电力设施敏感点", "330kV输电线路"),
    "220kV输电线路": ("电力设施敏感点", "220kV输电线路"),
    "110kV输电线路": ("电力设施敏感点", "110kV输电线路"),
    "66kV输电线路": ("电力设施敏感点", "66kV输电线路"),
    "35kV输电线路": ("电力设施敏感点", "35kV输电线路"),
    "接地极线路": ("电力设施敏感点", "接地极线路"),
    "中低压输电线路": ("电力设施敏感点", "中低压输电线路"),
    "变电站": ("电力设施敏感点", "变电站"),
    "换流站": ("电力设施敏感点", "换流站"),
    "接地极": ("电力设施敏感点", "接地极"),
    "弱电线路": ("电力设施敏感点", "弱电线路（I、II、III级）"),
    # 管廊
    "输油管道": ("管廊敏感点", "输油管道"),
    "输气管道": ("管廊敏感点", "输气管道"),
    "光纤管道": ("管廊敏感点", "光纤管道"),
    # 河流
    "通航河流": ("河流", "通航河流"),
    "非通航河流": ("河流", "非通航河流"),
    "河流": ("河流", "非通航河流"),
    # 地形
    "山峰": ("地形", "山峰"),
    "山谷": ("地形", "山谷"),
}

# 电压等级正则 → 输电线路二级类别
VOLTAGE_RE = re.compile(r"[±]?\d+\s*[kK][vV]")

# 字段名候选列表（用于字段发现）
LEVEL1_FIELD_NAMES = ["一级分类", "LV1_CLASS", "level1", "一级类别", "LEVEL1"]
LEVEL2_FIELD_NAMES = ["二级分类", "LV2_CLASS", "level2", "二级类别", "LEVEL2", "name", "NAME", "类型", "type", "category", "TYPE"]


class M1SemanticMapper:
    """M1 分级语义映射"""

    def __init__(self, project_config: dict, output_dir: str):
        self.config = project_config
        self.output_dir = Path(output_dir)

        # 加载规则表
        rules_path = os.path.join(get_config_dir(), "default_feature_rules.json")
        rules_data = load_json(rules_path)
        self.valid_level2_names = set()
        self.level2_to_rule_id = {}
        for feat in rules_data["features"]:
            self.valid_level2_names.add(feat["level2"])
            self.level2_to_rule_id[feat["level2"]] = feat["id"]

        # 映射统计
        self.stats = {
            "deterministic": 0,
            "fuzzy_auto": 0,
            "fuzzy_review": 0,
            "unmapped": 0,
            "total": 0,
        }
        self.unmapped_previews: List[dict] = []
        self.mapping_details: List[dict] = []

    def run(self, unified_gdf: gpd.GeoDataFrame, control_objects: dict) -> dict:
        """执行M1全部流程"""
        logger.info("===== M1: 分级语义映射启动 =====")

        if len(unified_gdf) == 0:
            logger.warning("输入矢量为空，跳过M1")
            return {"standardized_gdf": unified_gdf, "control_objects": control_objects}

        # 初始化标准字段
        unified_gdf["std_level1"] = ""
        unified_gdf["std_level2"] = ""
        unified_gdf["std_rule_id"] = -1
        unified_gdf["mapping_method"] = ""
        unified_gdf["mapping_confidence"] = 0.0

        # 逐要素映射
        for idx in unified_gdf.index:
            row = unified_gdf.loc[idx]
            l1, l2, method, conf = self._map_single_feature(row)  # 函数签名不变 (4-tuple)
            unified_gdf.at[idx, "std_level1"] = l1
            unified_gdf.at[idx, "std_level2"] = l2                # ★Q4★ 原始名 (如 "其他型电站")
            unified_gdf.at[idx, "mapping_method"] = method
            unified_gdf.at[idx, "mapping_confidence"] = conf
            # ★Q4 决策 (Option A 4-tuple 极简) — 修复版★
            # 查表优先级: 优先用 M1 决策结果 l2 (因为 l2 已经是 _map_single_feature 选定的规范名,
            # 包括 theme/fuzzy 分支命中的规范名); l2 查不到才 fallback 到 _path_level2_canonical
            # (M0 写出的 LEVEL2_ALIAS 规范化结果,用于路径精确匹配场景下的 alias 归并参数继承).
            #
            # 覆盖场景:
            #   1. 路径精确匹配 + 别名 (其他型电站): l2="其他型电站" 查不到, canonical="变电站" 查得 → ✓
            #   2. 路径精确匹配 + 无别名 (变电站):   l2="变电站" 查得 → ✓
            #   3. theme 分支 (基本农田):           l2="基本农田" 查得 → ✓
            #   4. fuzzy 命中规范名:                l2 已是规范名,查得 → ✓
            #   5. 旧数据 (无 canonical):           l2 查得 → ✓
            std_rule_id = None
            if l2 in self.level2_to_rule_id:
                std_rule_id = self.level2_to_rule_id[l2]
            else:
                canonical = str(row.get("_path_level2_canonical", "") or "").strip()
                if canonical and canonical != l2 and canonical in self.level2_to_rule_id:
                    std_rule_id = self.level2_to_rule_id[canonical]
            if std_rule_id is not None:
                unified_gdf.at[idx, "std_rule_id"] = std_rule_id
            self.stats["total"] += 1

        # 输出
        result = self._export(unified_gdf, control_objects)

        logger.info(f"M1完成: 确定性(含路径)={self.stats['deterministic']}, "
                    f"模糊自动={self.stats['fuzzy_auto']}, "
                    f"待确认={self.stats['fuzzy_review']}, "
                    f"未映射={self.stats['unmapped']}")
        return result

    def _map_single_feature(self, row: pd.Series) -> Tuple[str, str, str, float]:
        """
        对单个要素执行多级映射。优先级：
          0. M0 已从文件夹路径推断出 _path_level1 / _path_level2（最高优先级）
          1. 确定性映射（基于 _theme，兼容三区三线）
          2. 表内字段模糊匹配
          3. 图层名推断
          4. 未映射 → 人工确认
        """
        # ── 第零级：路径推断（最高优先级）──────────────
        # ★Q4 决策 (Option A 4-tuple 极简)★
        # path_l2 是 M0 写出的"原始基名"(如 "其他型电站", 未经别名替换);
        # path_l2_canonical 是 M0 同时写出的"规则查询规范名"(如 "变电站", 经 LEVEL2_ALIAS 归并).
        # 这里用 canonical 查 valid_level2_names (因为 valid_level2_names 含规范名),
        # 但 return 仍以 path_l2 (原始名) 作 std_level2, 实现"名保留 + 参数继承".
        path_l1 = str(row.get("_path_level1", "") or "").strip()
        path_l2 = str(row.get("_path_level2", "") or "").strip()
        path_l2_canonical = str(row.get("_path_level2_canonical", "") or path_l2).strip()
        if path_l2_canonical and path_l2_canonical in self.valid_level2_names:
            self.stats["deterministic"] += 1
            # ↓ 写出 std_level2 = path_l2 (原始名); 若 path_l2 为空 fallback 到 canonical
            return path_l1, path_l2 or path_l2_canonical, "path_deterministic", 1.0
        if path_l2_canonical:
            # 路径推断出了二级名称但不在标准表中 → 做模糊匹配 (用 canonical 作为查询输入)
            l1, l2_match, conf = self._fuzzy_match(path_l1, path_l2_canonical)
            if conf >= 0.85:
                self.stats["deterministic"] += 1
                # ↓ 优先返回 path_l2 (原始名); 若空 fallback 到模糊匹配结果
                return l1, path_l2 or l2_match, "path_fuzzy_auto", conf

        # 如果路径只推断出了一级（如文件夹名是 "生态敏感点" 但没有二级子文件夹）
        # 继续往下尝试从表内字段获取二级

        # ── 第一级：确定性映射（基于theme，兼容三区三线）──
        theme = row.get("_theme", "")
        if theme and theme in DETERMINISTIC_THEME_MAP:
            l1, l2 = DETERMINISTIC_THEME_MAP[theme]
            self.stats["deterministic"] += 1
            return l1, l2, "deterministic_theme", 1.0

        # ── 第二级：查找表内字段值 ────────────────────
        raw_l1, raw_l2 = self._discover_fields(row)

        # 如果路径推断了一级，但表内有二级 → 组合使用
        if path_l1 and not path_l2 and raw_l2:
            l1_cand, l2_cand, conf = self._fuzzy_match(path_l1, raw_l2)
            if conf >= 0.85:
                self.stats["fuzzy_auto"] += 1
                return l1_cand, l2_cand, "path_l1_field_l2_auto", conf
            elif conf >= 0.6:
                self.stats["fuzzy_review"] += 1
                self._add_unmapped_preview(row, path_l1, raw_l2, l1_cand, l2_cand, conf)
                return l1_cand, l2_cand, "path_l1_field_l2_review", conf

        # 纯表内字段匹配
        if raw_l2:
            l1, l2, conf = self._fuzzy_match(raw_l1, raw_l2)
            if conf >= 0.85:
                self.stats["fuzzy_auto"] += 1
                return l1, l2, "fuzzy_auto", conf
            elif conf >= 0.6:
                self.stats["fuzzy_review"] += 1
                self._add_unmapped_preview(row, raw_l1, raw_l2, l1, l2, conf)
                return l1, l2, "fuzzy_review", conf

        # ── 第三级：从图层名推断 ──────────────────────
        layer_name = row.get("_source_layer", "")
        if layer_name:
            l1, l2, conf = self._match_from_layer_name(layer_name)
            if conf >= 0.85:
                self.stats["fuzzy_auto"] += 1
                return l1, l2, "layer_name_auto", conf
            elif conf >= 0.6:
                self.stats["fuzzy_review"] += 1
                self._add_unmapped_preview(row, "", layer_name, l1, l2, conf)
                return l1, l2, "layer_name_review", conf

        # ── 第四级：路径有一级但无法确定二级 ──────────
        if path_l1:
            self.stats["fuzzy_review"] += 1
            self._add_unmapped_preview(row, path_l1, "", path_l1, "", 0.5)
            return path_l1, "UNKNOWN", "path_l1_only", 0.5

        # ── 未映射 ───────────────────────────────────
        self.stats["unmapped"] += 1
        self._add_unmapped_preview(row, raw_l1, raw_l2, "", "", 0.0)
        return "UNKNOWN", "UNKNOWN", "unmapped", 0.0

    def _discover_fields(self, row: pd.Series) -> Tuple[str, str]:
        """按优先级发现一二级分类字段值"""
        raw_l1 = ""
        raw_l2 = ""
        for fname in LEVEL1_FIELD_NAMES:
            if fname in row.index and pd.notna(row[fname]) and str(row[fname]).strip():
                raw_l1 = str(row[fname]).strip()
                break
        for fname in LEVEL2_FIELD_NAMES:
            if fname in row.index and pd.notna(row[fname]) and str(row[fname]).strip():
                raw_l2 = str(row[fname]).strip()
                break
        return raw_l1, raw_l2

    def _fuzzy_match(self, raw_l1: str, raw_l2: str) -> Tuple[str, str, float]:
        """模糊匹配字段值到标准类别"""
        # 精确匹配
        if raw_l2 in NORMALIZE_MAP:
            l1, l2 = NORMALIZE_MAP[raw_l2]
            return l1, l2, 1.0

        # 去除空格、括号等归一化后匹配
        norm_raw = self._normalize_text(raw_l2)
        for key, (l1, l2) in NORMALIZE_MAP.items():
            if self._normalize_text(key) == norm_raw:
                return l1, l2, 0.95

        # 电压等级特殊匹配
        voltage_match = VOLTAGE_RE.search(raw_l2)
        if voltage_match:
            voltage_str = voltage_match.group()
            for key, (l1, l2) in NORMALIZE_MAP.items():
                if voltage_str.lower().replace(" ", "") in key.lower().replace(" ", ""):
                    return l1, l2, 0.90

        # 模糊字符串匹配
        best_score = 0.0
        best_l1, best_l2 = "", ""
        for key, (l1, l2) in NORMALIZE_MAP.items():
            score = SequenceMatcher(None, raw_l2, key).ratio()
            if score > best_score:
                best_score = score
                best_l1, best_l2 = l1, l2

        return best_l1, best_l2, best_score

    def _match_from_layer_name(self, layer_name: str) -> Tuple[str, str, float]:
        """从图层名推断地物类别"""
        best_score = 0.0
        best_l1, best_l2 = "", ""
        for key, (l1, l2) in NORMALIZE_MAP.items():
            score = SequenceMatcher(None, layer_name, key).ratio()
            if score > best_score:
                best_score = score
                best_l1, best_l2 = l1, l2
        return best_l1, best_l2, best_score

    @staticmethod
    def _normalize_text(text: str) -> str:
        """文本归一化：去空格、转小写、统一括号"""
        text = text.strip().lower()
        text = text.replace("（", "(").replace("）", ")").replace("－", "-")
        text = text.replace(" ", "").replace("\t", "")
        return text

    def _add_unmapped_preview(self, row, raw_l1, raw_l2, matched_l1, matched_l2, conf):
        """记录未自动确认的映射"""
        self.unmapped_previews.append({
            "source_layer": row.get("_source_layer", ""),
            "source_gdb": row.get("_source_gdb", ""),
            "raw_level1": raw_l1,
            "raw_level2": raw_l2,
            "suggested_level1": matched_l1,
            "suggested_level2": matched_l2,
            "confidence": round(conf, 3),
        })

    def _export(self, unified_gdf: gpd.GeoDataFrame, control_objects: dict) -> dict:
        """输出M1产物"""
        m1_dir = ensure_dir(str(self.output_dir / "m1"))

        # v0.4 问题 10: 数据源多样性评分
        # 真实工程常只有三区三线数据, 所有要素都被路径推断映射到 3-4 个类别上。
        # M1 报告"确定性=100%"看起来完美, 实际反映"数据源单一, 缺交通/电力/建筑物等"。
        # 在报告里显式暴露多样性, 让下游/运维能看到"只看到了什么"。
        diversity = self._compute_diversity_score(unified_gdf)

        # 语义映射报告
        save_json({
            "stats": self.stats,
            "mapping_details_count": len(self.mapping_details),
            "diversity_score": diversity,  # v0.4 新增
        }, os.path.join(m1_dir, "semantic_mapping_report.json"))

        # 按严重度记录到日志, 下游和运维能一眼看到
        if diversity.get("warning"):
            logger.warning(f"数据源多样性告警: {diversity['warning']}")
        else:
            logger.info(
                f"数据源多样性: {diversity['level1_count']}/7 个一级类别, "
                f"{diversity['level2_count']} 个二级类别命中"
            )

        # 待确认列表
        # 去重
        seen = set()
        deduped = []
        for item in self.unmapped_previews:
            key = (item["raw_level1"], item["raw_level2"], item["source_layer"])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        save_json({"unmapped_features": deduped},
                  os.path.join(m1_dir, "unmapped_features_preview.json"))

        # 标准化矢量
        # ★修复 #17 (v0.5+)★ 用公共安全写 GPKG 工具 (M0 unified_gdf 含混合 schema 字段,
        # 透传到 M1 后写盘仍会触发 FieldError, 用 fallback 把非元数据字段转 str)
        if len(unified_gdf) > 0:
            from utils.geo_utils import write_gdf_to_gpkg_safe
            write_gdf_to_gpkg_safe(
                unified_gdf,
                os.path.join(m1_dir, "standardized_features.gpkg"),
                "standardized_features",
            )

        return {
            "standardized_gdf": unified_gdf,
            "control_objects": control_objects,
            "stats": self.stats,
            "diversity_score": diversity,  # v0.4
        }

    def _compute_diversity_score(self, unified_gdf) -> Dict:
        """
        v0.4 问题 10: 数据源多样性评分。

        核心洞察: "确定性=100%, fuzzy_auto=0%" 看起来好, 实际常常意味着
                   只有三区三线这一类数据源, 没有交通/电力/建筑物图层。

        产出字段:
          level1_categories_seen: 命中的一级类别列表
          level1_count: 一级类别数 (满分 7: 7 个一级类别)
          level2_count: 二级类别数
          per_level1_counts: 每个一级类别下的要素数
          critical_categories_missing: 未命中的关键类别 (交通/电力/建筑物等)
          warning: 有则为告警文本, 无则为 None
        """
        # 标准 7 个一级类别 (来自 default_feature_rules.json)
        ALL_LEVEL1 = [
            "生态敏感点",
            "重要设施与政府规划敏感点",
            "交通敏感点",
            "电力设施敏感点",
            "管廊敏感点",
            "河流",
            "地形",
        ]
        # 路径规划必须要能看到的核心类别 (缺了路径质量会明显打折)
        CRITICAL_LEVEL1 = ["交通敏感点", "电力设施敏感点"]

        result = {
            "level1_categories_seen": [],
            "level1_count": 0,
            "level2_count": 0,
            "per_level1_counts": {},
            "critical_categories_missing": [],
            "warning": None,
        }

        if unified_gdf is None or len(unified_gdf) == 0:
            result["warning"] = "无任何要素命中标准分类"
            return result

        try:
            # 只统计有效映射 (非 UNKNOWN)
            valid = unified_gdf[
                (unified_gdf["std_level1"] != "")
                & (unified_gdf["std_level1"] != "UNKNOWN")
            ]
            if len(valid) == 0:
                result["warning"] = "无任何要素映射到标准一级类别"
                return result

            level1_counts = valid["std_level1"].value_counts().to_dict()
            result["per_level1_counts"] = {k: int(v) for k, v in level1_counts.items()}
            result["level1_categories_seen"] = sorted(level1_counts.keys())
            result["level1_count"] = len(level1_counts)

            valid_l2 = valid[
                (valid["std_level2"] != "")
                & (valid["std_level2"] != "UNKNOWN")
            ]
            result["level2_count"] = int(valid_l2["std_level2"].nunique())

            # 缺失关键类别
            missing_critical = [c for c in CRITICAL_LEVEL1 if c not in level1_counts]
            result["critical_categories_missing"] = missing_critical

            # 告警逻辑: 一级类别 < 3 或缺关键类别
            warnings = []
            if result["level1_count"] < 3:
                warnings.append(
                    f"只命中 {result['level1_count']}/{len(ALL_LEVEL1)} 个一级类别, "
                    f"数据源偏单一 ({', '.join(result['level1_categories_seen'])})"
                )
            if missing_critical:
                warnings.append(
                    f"关键类别缺失: {', '.join(missing_critical)} — "
                    f"路径规划可能无法避开相关地物"
                )
            if warnings:
                result["warning"] = "; ".join(warnings)
        except Exception as e:
            logger.debug(f"计算多样性评分失败: {e}")
            result["warning"] = f"多样性评分计算失败: {e}"

        return result
