# preprocess — 输电线路路径规划 预处理包 (M0-M3)

本包是输电线路路径规划项目的预处理环节,
用于接入真实工程数据 (gdb / shp / tif) 的独立测试与迭代。

预处理环节 (M0-M3) 与算法端 (M4-M6) 通过**磁盘文件**解耦 —— 本包跑完后产出的
`<output_dir>/m2/`、`<output_dir>/m3/` 目录, 就是算法端所需的全部输入。

---

> **v0.2 更新要点** (2026-04) — 详见 `CHANGELOG_v0_2.md`
>
> - **新增 `manifest.json`** 作为下游算法端的唯一契约文件, 自带完整性校验 (见 §7.1)
> - **交付级别升为三档**: `FORMAL_DELIVERY` / `PRELIMINARY_ROUTE_ONLY` / **`SEVERE_DEGRADED`**(新)
> - **骨架构建性能重构**: 节点数 -67% / 边候选对 -93% / 预算回退保护 (见 §7.4 和 `tests/benchmark_skeleton.py`)
> - **多个精确 bug 修复**: tscf 禁塔区、start_end 解析、DEM 缺失判定、栅格下采样精度等
> - **向后完全兼容**, 所有新参数有安全默认值

---


## 一、本包范围

| 模块 | 功能 | 状态 |
|-----|-----|-----|
| **M0** 原始输入适配器 | 递归扫描项目目录, 自动识别 `.gdb` / `.shp` / `.tif`; 统一 CRS; 读取控制对象 (起终点/必经点/必经路径/密集通道/可入区域); 基于文件夹路径链推断地物类别 | ✅ 完整保留 |
| **M1** 分级语义映射 | 四级映射策略 (路径推断 → 确定性主题 → 字段值模糊匹配 → 图层名推断); 输出未自动确认清单 | ✅ 完整保留 |
| **M2** 几何预处理 | 点→缓冲面; 面→禁区/禁立塔区/高代价区三分类; 线状交叉对象按 50m 分段并记录方位角; 机场/河流/建筑聚类/优选走廊 (贴近奖励带) 专题处理; DEM 坡度/TPI/山谷/山峰分析; 风冰区组合代价栅格 | ✅ 完整保留 (含 v5.7 R4-5 多层级保护环) |
| **M3** 规则编译 + 标准化输出包 | 50m 粗栅格 + 10m 精栅格双分辨率; forbidden_mask / tower_mask / lpcf / tscf / tower_difficulty_50m 等栅格产出; 混合导航图骨架构建; 规则配置/求解器参数/工作区/预处理报告 (含交付级别判定) | ✅ 完整保留 |

**不包含:** M4 路径规划 / M5 杆塔优化 / M6 方案评分。这些在另一个独立的算法包中。

**保留的降级与预留接口:**
- 河流宽度分析: 面域数据缺失时执行降级 (保守名单屏蔽), 功能接口保留 (见 `当前情况说明.md` 第 1 条)
- 风冰区组合代价: 栅格缺失时使用 `project.json` 的 `ice_zone`/`wind_zone` 查 `wind_ice_cost_table.json` 取全域统一参数, 栅格到位后自动切换像素级查表 (见 `当前情况说明.md` 第 2 条)
- 净空/弧垂复核 (M5-5): 不在本包范围内, 由算法端实现

---

## 二、目录结构

```
preprocess/
├── run_preprocess.py              # 主入口 (M0→M1→M2→M3)
├── modules/
│   ├── m0_input_adapter.py
│   ├── m1_semantic_mapping.py
│   ├── m2_geometry_preprocessing.py
│   └── m3_rule_compile_and_output.py
├── utils/
│   ├── geo_utils.py               # 通用几何/IO 工具
│   ├── parallel_geometry.py       # 严格平行关系判定
│   └── control_io.py              # ★下游算法端复用: 把磁盘上的 control_*.gpkg 还原成 dict
├── config/
│   ├── default_feature_rules.json # 65 条地物规则基础库 (对应《地物类别识别规则.md》)
│   └── wind_ice_cost_table.json   # 风/冰区组合代价表
├── tests/
│   ├── generate_test_data.py      # 内置模拟测试数据生成器 (20km×20km 模拟工程)
│   └── smoke_test.py              # 静态冒烟测试 (不依赖 GIS 运行时)
├── requirements.txt
├── README.md                      # 本文件
└── CHANGELOG.md                   # 从 v5.7 提取的变更记录
```

---

## 三、安装

```bash
cd preprocess
pip install -r requirements.txt
```

依赖核心: `numpy`, `pandas`, `shapely>=2.0`, `geopandas>=0.13`, `rasterio>=1.3`,
`fiona>=1.9`, `pyproj>=3.4`, `scipy>=1.9`。

> Windows 用户若直接 `pip install geopandas` 报错, 建议用 `conda install -c conda-forge geopandas rasterio fiona` 预先装好 GDAL 栈。

---

## 四、快速开始

### 4.1 用内置模拟数据一键验证 (不需要真实工程数据)

```bash
python run_preprocess.py --generate_test --output_dir ./output
```

这会:
1. 在 `./output/test_project_data/` 生成一个 20km×20km 的模拟工程数据包 (DEM / 矢量地物 / 河流面 / 风冰区栅格 / 控制对象 / 杆塔成本)
2. 跑完整个 M0→M1→M2→M3 流水线
3. 产出 `./output/m0/`、`./output/m1/`、`./output/m2/`、`./output/m3/` 四个子目录
4. 产出总结文件 `./output/preprocessing_summary.json` 和日志 `./output/preprocessing.log`

> 想把模拟数据和预处理产出分开存放, 加 `--test_dir <路径>`:
> ```bash
> python run_preprocess.py --generate_test --test_dir ./sim_data --output_dir ./output
> ```
>
> 注: 模拟数据包里生成的 `tower_cost/tower_cost.json` 是给下游 M5 算法端用的,
> 本预处理包 M0-M3 不读取它, 保留它仅为方便后续和算法端合跑时复用同一份数据。

### 4.2 接入真实项目数据

先按下面的目录约定准备数据, 然后:

```bash
python run_preprocess.py --project_dir /path/to/your/project --output_dir ./output
```

若 `project.json` 不在默认位置:

```bash
python run_preprocess.py --project_dir /path/to/your/project \
                         --project_json /path/to/cfg.json \
                         --output_dir ./output
```

### 4.3 只做静态校验 (不依赖完整 GIS 运行时)

```bash
python tests/smoke_test.py
```

用于 CI 或快速确认包结构未损坏。

---

## 五、项目数据目录约定 (输入)

本包支持真实甲方下发的 `.gdb` 目录(三区三线数据)、`.shp` 矢量、`.tif` 栅格混合输入。
**不强制使用固定的 kml/yaml 格式。**

### 5.1 推荐的目录结构

```
<project_dir>/
├── project.json                  # 工程参数配置 (见 5.2)
├── dem/
│   └── dem.tif                   # 数字高程模型 (必需, 用于坡度/山谷/山峰分析)
├── wind_ice/                     # 可选: 风冰区组合代价栅格
│   └── wind_ice_zone.tif
├── vectors/                      # SHP 方式(可选): 平铺的矢量文件
│   ├── buildings.shp
│   ├── roads.shp
│   └── ...
├── 441400梅州市-…/               # GDB 方式(可选): 三区三线标准数据
│   └── 441402梅江区/
│       ├── 城镇开发边界/
│       │   └── 441402CZKFBJ_ZW2_V2014.gdb/
│       ├── 永久基本农田/
│       │   └── …gdb/
│       ├── 生态保护红线陆域/
│       │   └── …gdb/
│       └── 耕地保护目标/
│           └── …gdb/
├── tower_cost/                   # 可选: 杆塔成本表 (供算法端使用, 本包不消费)
│   └── tower_cost.json
└── control/                      # 控制对象 (起终点必需)
    ├── start_end.geojson         # type=start / type=end
    ├── must_pass.geojson         # (可选) 必经点
    ├── must_path.geojson         # (可选) 必经路径
    ├── dense_corridor.geojson    # (可选) 密集通道
    └── accessible_area.geojson   # (可选) 可入区域
```

**重要: 地物分类识别靠文件夹路径链的关键词匹配。**
例如 `441402梅江区/城镇开发边界/xxx.gdb` 会被 M0 自动识别为
`(level1=重要设施与政府规划敏感点, level2=城镇规划区)`。
支持的关键词列表在 `modules/m0_input_adapter.py` 顶部 `PATH_KEYWORD_TO_CATEGORY` 字典中, 共 60+ 条。

### 5.2 `project.json` 示例

```json
{
  "project_name": "某220kV输电工程",
  "voltage_kv": 220,
  "ice_zone": 10,
  "wind_zone": "B",
  "source_crs": "EPSG:4490",
  "working_crs": "EPSG:4547",
  "start_point": [500000, 2610000],
  "end_point":   [520000, 2610000],
  "data_availability": {
    "river_polygon": false,
    "wind_ice_zone_raster": false
  },
  "building_cluster": {
    "buffer_m": 100, "merge_gap_m": 50,
    "dense_zone_penalty": 3000, "min_cluster_area_m2": 50000
  },
  "river_rule": {
    "major_river_threshold_m": 900,
    "conservative_wide_river_names": ["长江", "珠江"]
  },
  "solver_params": {
    "corridor_top_k": 3,
    "max_turn_angle_deg": 90
  }
}
```

`data_availability` 标记数据是否到位, 决定走完整分析还是降级:
- `river_polygon=false` → M2 对宽河做降级处理 (按保守名单扣为禁区), `preprocessing_report.river_impact` 会给出警告
- `wind_ice_zone_raster=false` → 全域使用 `ice_zone`/`wind_zone` 查表得到统一参数

`source_crs` 是原始数据 CRS (常见 `EPSG:4490`/`4326`), `working_crs` 是投影坐标 CRS (常见 `EPSG:4547` CGCS2000 3度带)。M0 会把所有输入统一到 `working_crs`。

控制对象的起终点有两种提供方式, 任选其一:
- `control/start_end.geojson` (优先)
- `project.json` 里的 `start_point`/`end_point` 字段 (兜底)

---

## 六、产出文件说明 (输出)

运行完毕后 `<output_dir>/` 结构:

```
<output_dir>/
├── preprocessing_summary.json    # ★总结: 每阶段耗时/要素计数/交付级别
├── preprocessing.log             # 日志
├── m0/
│   ├── gdb_inventory.json            # 扫描到的所有 gdb 清单
│   ├── raster_inventory.json         # 扫描到的所有栅格清单 (含 inferred_type)
│   ├── read_log.json                 # 每个图层的读取日志
│   ├── unified_vectors.gpkg          # 所有矢量统一 CRS 后的合并产物
│   └── control_*.gpkg                # 控制对象 (起终点/必经点/…)
├── m1/
│   ├── semantic_mapping_report.json  # 映射命中统计
│   ├── unmapped_features_preview.json # 需人工确认的待映射清单
│   └── standardized_features.gpkg    # 增加 std_level1/std_level2/std_rule_id 后的矢量
├── m2/
│   ├── forbidden_polygons.gpkg       # ★禁区 (算法绝对避让)
│   ├── no_tower_polygons.gpkg        # ★禁立塔区 (可跨越不可立塔)
│   ├── cost_polygons.gpkg            # ★高代价区 (含 land_cost/cross_cost/cost_type)
│   ├── linear_cross_indexed.gpkg     # ★线状交叉对象方向索引 (50m 分段, 带方位角)
│   ├── preferred_corridors.gpkg      # ★优选走廊 (贴近奖励带)
│   ├── buffered_points.gpkg          # 点状对象的缓冲面
│   ├── building_clusters.gpkg        # 建筑聚类 (可选)
│   ├── river_crossing_windows.gpkg   # 可跨河窗口 (河流面域可用时)
│   ├── wide_river_barriers.gpkg      # 宽河屏障 (>=900m 部分, 作为禁区)
│   ├── processed_polygons.gpkg       # 面对象总表 (调试用)
│   ├── control_*.gpkg                # 控制对象 (M2 原样透传, 方便 M4 直接读取)
│   ├── terrain_slope.tif             # 坡度 (度)
│   ├── terrain_tpi.tif               # 地形位置指数
│   ├── valley_mask.tif               # 山谷掩膜 (tpi < -30)
│   ├── peak_mask.tif                 # 山峰掩膜 (tpi > +30)
│   ├── wind_ice_*.tif                # 风冰区代价 (栅格可用时, 否则报告中标注降级)
│   └── (其他专题栅格)
└── m3/
    ├── rule_config.json              # ★编译后规则 (含 geometry_family / compiled_behavior)
    ├── solver_params.json            # ★求解器参数 (min_tower_spacing、平均档距、转角上限…)
    ├── workspace.json                # ★工作区 (bbox / 分辨率 / 起终点)
    ├── preprocessing_report.json     # ★预处理报告 (含 delivery_level 交付级别)
    ├── forbidden_mask_50m.tif        # ★粗分辨率禁区掩膜
    ├── forbidden_mask_10m.tif        # ★精分辨率禁区掩膜
    ├── tower_mask_50m.tif            # ★粗立塔可行性
    ├── tower_mask_10m.tif            # ★精立塔可行性
    ├── lpcf_50m.tif                  # ★线路路径代价场 (分级引导值, 不含一次性代价)
    ├── lpcf_10m.tif
    ├── tscf_50m.tif                  # 塔位连续代价场
    ├── tscf_10m.tif
    ├── tower_difficulty_50m.tif      # v5.4 塔位难度预查表
    ├── wind_ice_max_turn_10m.tif     # 按像素最大转角限制 (风冰区栅格可用时)
    ├── terrain_slope_10m.tif         # 对齐后的坡度栅格
    ├── nav_graph_nodes.gpkg          # 混合导航图节点
    ├── nav_graph_edges.json          # 导航图边
    ├── crossing_window_index.gpkg    # 跨越窗口索引
    └── hybrid_nav_graph_metadata.json
```

### 6.1 交付级别判定

`m3/preprocessing_report.json` 中 `delivery_level.level` 会取以下之一:

| 级别 | 含义 | 触发条件 |
|------|-----|---------|
| `FORMAL_DELIVERY` | 所有数据齐全, 可走正式交付 | 无任何降级原因 |
| `PRELIMINARY_ROUTE_ONLY` | 数据有缺失, 仅作初步选线参考 | 下列任一: 河流面域缺失 / 风冰栅格缺失 / DEM 空洞比例 > 5% / DEM 标称精度 > 30m |

`reasons` 字段列出具体降级原因清单, `upgrade_actions` 字段给出升回正式交付需要的补数据动作,
两者均可供项目报告附录直接引用。

> 未来若要细分"局部降级 (仅部分功能缺失)"与"严重降级 (DEM 不可用)"两档, 需要修改
> `modules/m3_rule_compile_and_output.py` 的 `_determine_delivery_level` 方法。
> 当前版本只区分 2 档, 与源码行为一致。

---

## 七、与算法端 (M4-M6) 的对接契约

### 7.1 v0.2 必读:`manifest.json` 是唯一契约文件

从 v0.2 起, 预处理包完成时会在 `<output_dir>/manifest.json` 写一份**自描述清单**。
算法端入口**第一件事**就是读它 + 校验, 不要依赖对目录结构的"默认知识"。

```python
from utils.manifest import verify_manifest

vr = verify_manifest("./output")
if not vr["ok"]:
    raise RuntimeError(
        f"预处理包不完整!\n"
        f"  大小/存在性不一致: {vr['mismatches']}\n"
        f"  必需产物缺失: {vr['required_missing']}"
    )

level = vr["delivery_level"]  # "FORMAL_DELIVERY" / "PRELIMINARY_ROUTE_ONLY" / "SEVERE_DEGRADED"
if level == "SEVERE_DEGRADED":
    raise RuntimeError(f"预处理严重降级 (原因见 manifest.json), 不建议用于选线")
elif level == "PRELIMINARY_ROUTE_ONLY":
    logger.warning(f"预处理有降级, 产出仅作初步方案参考")
# else: FORMAL_DELIVERY, 正式交付
```

**三档交付级别**详情:
- `FORMAL_DELIVERY`: 所有数据齐全, 可走正式交付
- `PRELIMINARY_ROUTE_ONLY`: 河流面域 / 风冰栅格缺失 / DEM 精度不达标等 (仍可初步选线)
- **`SEVERE_DEGRADED`**(v0.2 新增): DEM 完全缺失 / bbox 未定 / M3 核心栅格缺失 (**不建议用于选线**)

### 7.2 文件契约 (主要方式)

算法端的 `M4PathPlanner`/`M5TowerOptimizer`/`M6SchemeEvaluator` 以 `output_dir` 初始化后,
会自动从:
- `<output_dir>/m3/` 读取 `rule_config.json`、`solver_params.json`、`workspace.json`、各栅格、`nav_graph_*`
- `<output_dir>/m2/` 读取 `forbidden_polygons.gpkg`、`cost_polygons.gpkg`、`linear_cross_indexed.gpkg`、`river_crossing_windows.gpkg` 等

**预处理端只需保证这些文件齐全**, 无需算法端改任何代码。
`manifest.json` 的 `files.*` 字段是**权威文件清单**, 算法端想要完整列表直接读它。

### 7.3 控制对象契约 (唯一需要桥接的地方)

`M4PathPlanner.run(control_objects, ...)` 的第一个参数历史上是
`dict[str, GeoDataFrame]` (不是文件路径)。本包提供 `utils/control_io.py` 来还原:

```python
# 算法端入口示例 (v0.2)
from utils.manifest import verify_manifest
from utils.control_io import load_control_objects, validate_controls
from modules.m4_path_planning import M4PathPlanner

# 1) 校验包完整性
vr = verify_manifest("./output")
assert vr["ok"], f"预处理包不完整: {vr}"

# 2) 校验控制对象 (防御性)
cv = validate_controls("./output", required_keys=["start_end"])
assert cv["ok"], f"控制对象校验失败: {cv}"

# 3) 加载并运行算法 (与 v5.7 一致)
control_objects = load_control_objects("./output")
planner = M4PathPlanner("./output", project_config)
schemes = planner.run(control_objects)
```

`load_control_objects` 会查找 `<output_dir>/m2/control_*.gpkg` 还原出:
- `start_end` / `must_pass` / `must_path` / `dense_corridor` / `accessible_area`

只有实际存在的 gpkg 才会出现在返回字典里 (算法端已经按 `"xxx" in control_objects` 做了防御判断)。

### 7.4 骨架构建模式 (v0.2 新增)

`solver_params.skeleton_build_mode` 控制 M3 内是否同步构建导航图骨架:

| 模式 | 行为 | `manifest.nav_graph.status` | 适用场景 |
|:-:|:-:|:-:|:-:|
| `"immediate"` (默认) | M3 立即构建, 产 `nav_graph_nodes.gpkg` + `nav_graph_edges.json` | `"completed"` | 默认 — 与 v5.7 行为一致 |
| `"deferred"` | M3 跳过, 仅写 `hybrid_nav_graph_metadata.json` 占位 | `"deferred"` | 希望 M3 更快收口; 算法端按需构建 |
| `"skip"` | 完全跳过 | `"skip"` | 纯栅格级算法(只用 lpcf/tscf) |

可通过 CLI 覆盖 project.json:

```bash
python run_preprocess.py --project_dir ./proj --output_dir ./output --skeleton_mode deferred
```

### 7.5 本包与算法端合跑的建议目录布局

```
deployment/
├── preprocess/             # 本包
└── algorithm/              # M4-M6 算法包 (另行维护)
    ├── modules/
    │   ├── m4_path_planning.py
    │   ├── m5_tower_optimization.py
    │   └── m6_scheme_evaluation.py
    └── run_algorithm.py    # 调 verify_manifest + load_control_objects + M4→M5→M6
```

或者也可以把算法包作为子目录并共用同一套 `utils/`、`config/`。


---

## 八、字段/参数关系重点说明

### 8.1 `buffer_m` vs `parallel_range_m`

(对应《v5_2_完整版可改进的部分.md》问题 1)

- `buffer_m` = **保护范围 (禁立塔缓冲距离)**, 单位米, 在规则表中每条地物自有设定
- `parallel_range_m` = **贴近奖励范围**, 默认 200m, 与保护范围互相独立

M3 的 `rule_config.compiled_rules[*]` 里同时输出两个字段, 算法端贴近判定用 `parallel_range_m`,
禁立塔判定用 `buffer_dist_m`。

### 8.2 代价值一次性 vs 涉及长度/角度

- 规则表中没有 L/α 的 `cross_cost`/`land_cost` = **一次性事件代价**, 只在路径/塔位进入区域时收取一次
- 带 `cost_type=angle_formula`/`length_formula` + `cross_cost_formula` = 按实际角度/长度计算
- M3 的 `lpcf_*.tif` **不**包含一次性代价 (那部分由算法端在事件发生时按 `cost_polygons.gpkg` 单独扣); `lpcf` 只含分级引导值避免路径盲目穿越高代价区

### 8.3 密集通道 / 可入区域 / 必经路径

这些是由用户手绘/拾取得到的 `control/*.geojson`, 预处理阶段只做 CRS 统一和原样透传,
不作任何几何改造。算法端使用 `control_objects["dense_corridor"]` 等字段自己实现约束。

---

## 九、常见问题

**Q1: 只有 SHP 没有 GDB 能不能跑？**
A: 可以。把所有 shp 放到 `<project_dir>/vectors/` 或任意子目录, M0 会递归扫描。
为了让类别识别命中, 建议把 shp 放在带关键词的文件夹里 (如 `vectors/森林公园/`),
或者 shp 的属性字段里带有 `二级分类`/`类型`/`name` 等字段供 M1 模糊匹配。

**Q2: CRS 不对怎么办？**
A: `project.json` 里 `source_crs` 设为原始数据实际 CRS (不是目标 CRS)。
运行前可以用 `ogrinfo -al -so <shp>` 确认一下。

**Q3: DEM 分辨率低怎么办？**
A: 本包不重采样 DEM, 直接在原分辨率上做坡度/TPI。
但 M3 交付级别会检测 DEM 质量 (`dem_nodata_ratio>0.05` 或 `resolution>30m` 时给 warning),
严重时把交付级别降为 `PRELIMINARY_ROUTE_ONLY`。

**Q4: 中间跑挂了怎么断点续跑？**
A: 目前不支持增量。每次运行会覆盖整个 `output_dir`。建议每次用不同的 `output_dir` 或
先备份。M0→M1→M2→M3 顺序执行, 其中任一步失败的日志会在 `preprocessing.log` 里。

**Q5: 想看算法端到底读了哪些文件?**
A: 直接搜 `modules/m4_path_planning.py` 里的 `os.path.join(self.m3_dir, ...)` 和
`os.path.join(self.m2_dir, ...)` 就能看到全部文件清单。
本包的产出是这些文件清单的**超集** (多输出了一些仅供人工检查/可视化的文件, 如
`unified_vectors.gpkg`、`preprocessing_summary.json`)。

---

## 十、版本

- 基线: `transmission_line_planning v5.7`
- 提取版本: `preprocess v0.1.0` (2026-04)
- 变更详情见 `CHANGELOG.md`

## 十一、许可

同原 `transmission_line_planning` 项目。
