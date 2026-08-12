# preprocess — 输电线路路径规划 · 预处理端 (M0–M3)

> **版本：v0.6.5**（2026-08）
> 把原始 GIS 数据（GDB / SHP / DEM 栅格）转换为算法端可直接消费的标准化中间产物。
>

---

## 目录

1. [本包做什么](#一本包做什么)
2. [目录结构](#二目录结构)
3. [安装](#三安装)
4. [快速开始](#四快速开始)
5. [工程数据目录约定（输入）](#五工程数据目录约定输入)
6. [`project.json` 配置](#六projectjson-配置)
7. [命令行参数](#七命令行参数)
8. [产出文件说明（输出）](#八产出文件说明输出)
9. [与算法端的对接契约](#九与算法端的对接契约)
10. [核心算法要点](#十核心算法要点)
11. [测试](#十一测试)
12. [常见问题](#十二常见问题)
13. [版本历史](#十三版本历史)

---

## 一、本包做什么

```
原始 GIS 数据                    预处理端 (本包)                     算法端 (不在本包)
GDB / SHP / DEM  ──►  M0 ─► M1 ─► M2 ─► M3  ──►  GPKG + JSON  ──►  路径优化
```

| 模块 | 职责 | 主要产出 |
|---|---|---|
| **M0** 原始输入适配 | 扫描 GDB/SHP/栅格；识别图层对应的地物类别与变体；自动选择工作坐标系；统一 CRS 并合并 | `unified_vectors.gpkg`、`variant_inventory.json`、`protection_coverage_report.json` |
| **M1** 分级语义映射 | 为每个要素标注 `level1` / `level2` / `rule_id`，与 65 条规则表严格对应 | `standardized_features.gpkg`、`semantic_mapping_report.json` |
| **M2** 几何预处理 | 按规则分流为禁区 / 禁立塔 / 代价区 / 线状穿越；河流宽窄分治；贴近奖励走廊；DEM 地形分析；风冰矢量标准化 | `forbidden_polygons.gpkg`、`no_tower_polygons.gpkg`、`cost_polygons.gpkg`、`linear_cross_indexed.gpkg` 等 |
| **M3** 规则编译与输出包 | 编译规则配置；确定工作区范围；生成交付清单与交付级别判定 | `rule_config.json`、`workspace.json`、`manifest.json` |

**设计原则**

- 算法端**只通过文件消费**（`json.load()` / `gpd.read_file()`），不导入本包任何代码；
- 算法端**不读本包产生的任何栅格**（`.tif`），需要栅格时自行从 GPKG 栅格化；
- 因此本包可独立升级、独立验证 —— 只要文件名与字段不变，算法端无需改动。

---

## 二、目录结构

```
preprocess/
├── run_preprocess.py                  # 主入口 (M0→M1→M2→M3)
├── modules/
│   ├── m0_input_adapter.py            # 扫描/识别/统一 CRS/合并；working_crs 自动选带
│   ├── m1_semantic_mapping.py         # level1/level2/rule_id 映射
│   ├── m2_geometry_preprocessing.py   # 几何分流、河流分治、地形、风冰、走廊
│   └── m3_rule_compile_and_output.py  # 规则编译、bbox、manifest
├── utils/
│   ├── geo_utils.py                   # 几何/IO 工具；河流中心线与宽度算法
│   ├── bbox_infer.py                  # 工作区 bbox 推断（起终点 + 缓冲）
│   ├── crs_recommender.py             # CGCS2000 3°/6° 带推荐
│   ├── manifest.py                    # 交付清单与交付级别判定
│   ├── parallel_geometry.py           # 平行贴近关系判定
│   ├── control_io.py                  # ★算法端可复用：control_*.gpkg → dict
│   └── io_helpers.py
├── config/
│   ├── default_feature_rules.json     # 65 条地物规则基础库
│   └── wind_ice_cost_table.json       # 风区×冰区组合代价表
├── tests/                             # 见 §十一
├── docs/                              # 历史设计文档（部分已过时，以本 README 为准）
├── data/control/start_end.geojson     # 示例起终点
├── requirements.txt
├── README.md                          # 本文件
└── CHANGELOG*.md                      # 各版本变更记录
```

---

## 三、安装

需要完整的 Python GIS 运行时，建议 conda 独立环境：

```bash
conda create -n trans_route python=3.11
conda activate trans_route
conda install -c conda-forge geopandas rasterio fiona pyproj shapely gdal numpy scipy pandas
```

| 依赖 | 用途 |
|---|---|
| geopandas / fiona / shapely | 矢量读写与几何运算 |
| rasterio / GDAL | 栅格读写、重投影、裁剪 |
| pyproj | 坐标变换与投影带选择 |
| numpy / scipy / pandas | 数值计算与地形分析 |

> 纯函数单测（`tests/test_river_centerline_coverage.py` 等）只依赖 numpy，无 GIS 环境也能跑。

---

## 四、快速开始

```bash
# 1) 标准运行（生产用，喂算法端）
python run_preprocess.py --project_dir data --output_dir ./output

# 2) 调试运行（额外产出中间栅格供 QGIS 查看）
python run_preprocess.py --project_dir data --output_dir ./output \
                         --emit_unconsumed_outputs

```

每次运行生成带时间戳的独立目录 `output/run_YYYYMMDD_HHMMSS/`，历史结果不会被覆盖。

**用内置模拟数据验证（无需真实工程数据）**

```bash
python tests/generate_test_data.py          # 生成模拟工程
python tests/test_terrain_windice_e2e.py    # DEM/风冰端到端
python tests/smoke_test.py                  # 静态冒烟（不依赖 GIS 运行时）
```

---

## 五、工程数据目录约定（输入）

```
data/                              ← --project_dir 指向这里
├── project.json                            ← 工程配置（见 §六）
├── 大埔工程地物数据/
│   └── 梅江区梅县区大埔县敏感点数据.gdb/       ← 地物 GDB
├── 大埔工程地形数据/
│   └── ....gdb/地形地貌/山谷                  ← 地形 GDB（山谷等）
├── dem/                                    ← DEM 栅格
├── wind_ice/                               ← 风区/覆冰区矢量（GDB）
└── control/
    └── start_end.geojson                   ← 起终点（必需）
```

程序**递归扫描**该目录，自动识别 GDB、SHP 与栅格，不需要逐个登记。

### 5.1 图层名如何对应到规则

采用三级解析链，逐级尝试，命中即停：

| 层级 | 依据 | 示例 |
|---|---|---|
| **L1** 规则表反查 | 图层名（去掉变体后缀）直接匹配 65 条规则的二级类别 | `省道` → 规则 38，一级类别`交通敏感点` |
| **L2** GDB 数据集名 | 用 GDB 内部要素数据集名推断一级类别 | 数据集`交通敏感点` → 一级类别 |
| **L3** 别名字典 | 常见同义词兜底 | `高速路` → `高速公路` |

### 5.2 变体后缀

同一类地物常有多个图层，用后缀区分性质，程序自动识别并分流：

| 后缀 | `_variant` | 含义 |
|---|---|---|
| （无） | `primary` | 地物本体范围 |
| `_保护范围` | `protection` | 按规则表保护距离外扩的范围 |
| `_奖励范围` | `reward` | 平行贴近可获代价折减的范围 |

> **当前版本依赖甲方提供的变体图层**，不自动生成。`m0/protection_coverage_report.json` 会列出「规则要求保护范围但数据中无对应变体」的类别，供数据完整性核对。自动生成功能在后续版本规划中。

### 5.3 控制对象

`control/` 下可放 `start_end`（必需）、`must_pass`、`must_path`、`dense_corridor`、`accessible_area`，缺省则跳过并在日志中标注。

---

## 六、`project.json` 配置

```jsonc
{
  "project_name": "XX至YY 500kV输电工程",
  "voltage_kv": 500,
  "ice_zone": 10,                        // 全域覆冰区等级
  "wind_zone": "B",                      // 全域风区等级
  "working_crs": "auto",                 // "auto" / 省略 → 自动选带；也可写 "EPSG:4548"

  "data_availability": {
    "river_polygon": true,               // 有河流面要素 → 启用宽窄分治
    "wind_ice_zone_raster": false
  },

  "river_rule": {
    "major_river_threshold_m": 900,      // 宽/窄河分界
    "conservative_wide_river_names": []  // 强制按宽河处理的河名白名单
  },

  "wind_ice_vector": {                   // 风冰为矢量(GDB)时的图层名
    "enabled": true,
    "layers": ["wind50_clip", "ice50_clip"]
  },

  "allow_planar_2d_mode": true,
  "enable_airport_legacy_processing": false,
  "enable_building_cluster_legacy": false,
  "enable_corridor_legacy": false,

  "solver_params": {
    "coarse_resolution_m": 100,          // 必须与算法端一致
    "fine_resolution_m": 25,             // 平面 2D 阶段取值
    "terrain_proc_resolution_floor_m": 10
  }
}
```

### 6.1 `working_crs` 自动选带

所有几何计算（缓冲距离、面积、宽度）必须在以米为单位的平面投影下进行，选错带会引入长度畸变。

设为 `"auto"`（或省略）时，M0 读入数据后：

1. 按工程实际经纬度范围选择最优 CGCS2000 3° 带（跨度 > 3° 自动转 6° 带并告警）；
2. 估算该带下的最大长度畸变，超阈值时告警；
3. **回写配置**，M1/M2/M3 与 manifest 全程使用同一坐标系。

推断失败 → 回退 `EPSG:4547` 并告警。

> 大埔工程实测：中心经度 116.41° → 自动选 `EPSG:4548`（CM 117°），畸变 ≈ 0.39/10000。

### 6.2 关键参数速查

| 参数 | 含义 | 备注 |
|---|---|---|
| `coarse_resolution_m` | 粗规划栅格边长 | **必须 = 100**，算法端启动闸门会校验（±0.5） |
| `fine_resolution_m` | 精规划栅格边长 | 必须**非 null**（算法端 `float()` 转换）；默认 10，本工程覆盖为 25 |
| `major_river_threshold_m` | 河道宽窄分界 | 规则表 61/62 条规定为 900 |
| `terrain_proc_resolution_floor_m` | 地形处理分辨率下限 | 默认 10m；防止原生 ~1m DEM 重投影时内存溢出 |

---

## 七、命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--project_dir` | **必填** | 工程数据目录 |
| `--output_dir` | `./output` | 输出根目录，其下建带时间戳子目录 |
| `--enable_fine_resolution` | 关闭 | 是否产出细分辨率栅格。算法端当前只用粗分辨率，默认关闭以节省时间 |
| `--emit_unconsumed_outputs` | 关闭 | 是否写出中间栅格（地形栅格、lpcf/tscf、栅格层）。**算法端不读任何栅格**，此参数仅供调试与 QGIS 查看 |
| `--bbox_buffer_km` | `5.0` | 起终点窗口向外扩展距离 |
| `--disable_river_real_barrier` | 关闭 | 关闭河流宽窄分治，回退到 v0.5 简化处理（应急用） |
| `--log_level` | `INFO` | 日志详细程度 |

> **接入算法端不需要带 `--emit_unconsumed_outputs`**（默认关，更快）。

---

## 八、产出文件说明（输出）

```
output/run_20260812_205127/
├── manifest.json                       ★ 交付清单，算法端启动第一个读
├── preprocessing_summary.json          运行摘要
├── m0/
│   ├── unified_vectors.gpkg            统一坐标系后的全量要素（源属性加 attr_ 前缀保留）
│   ├── gdb_inventory.json              GDB 图层清点
│   ├── raster_inventory.json           栅格清点
│   ├── variant_inventory.json          变体分布清点
│   ├── protection_coverage_report.json 保护范围覆盖情况
│   ├── crs_diagnostic.json             坐标系诊断
│   └── read_log.json
├── m1/
│   ├── standardized_features.gpkg      带 rule_id 的标准化要素
│   ├── semantic_mapping_report.json    映射统计
│   └── unmapped_features_preview.json  未映射要素样本
├── m2/
│   ├── forbidden_polygons.gpkg         ★ 禁区
│   ├── no_tower_polygons.gpkg          ★ 禁立塔区
│   ├── cost_polygons.gpkg              ★ 代价区
│   ├── linear_cross_indexed.gpkg       ★ 线状穿越对象（50m 分段）
│   ├── preferred_corridors.gpkg        ★ 贴近奖励走廊
│   ├── wide_river_barriers.gpkg        宽河段（审计用）
│   ├── river_crossing_windows.gpkg     窄河跨越窗口（审计用）
│   ├── wind_ice_zones.gpkg             风冰区（已标准化，未套代价/未栅格化）
│   ├── processed_polygons.gpkg         面处理中间结果
│   └── geometry_fixes_log.json         几何修复日志
└── m3/
    ├── rule_config.json                ★ 编译后的规则配置
    ├── workspace.json                  工作区范围与分辨率
    ├── solver_params.json
    └── preprocessing_report.json       全流程分析报告
```

（★ = 算法端实际消费）

### 8.1 三类面产物的语义

严格区分，不可混用：

| 产物 | 线路能否从上方经过 | 能否立塔 | 典型内容 |
|---|---|---|---|
| `forbidden_polygons.gpkg` | 不能 | 不能 | 自然保护区核心区/缓冲区、禁止建设区、风景名胜区、**宽河段（≥900m）** |
| `no_tower_polygons.gpkg` | 能（跨越） | 不能 | 一级水源保护区、山谷、道路/线路的保护范围、通航河流水面 |
| `cost_polygons.gpkg` | 能 | 能，计代价 | 基本农田（立塔 200）、二级林地（300）、非通航河流（2000） |

**每个面要素的字段**

| 字段 | 含义 |
|---|---|
| `level2` | 二级类别名称 |
| `rule_id` | 规则表编号 1–65；**`-1` 表示预处理派生面**（如宽河屏障） |
| `area_m2` | 工作坐标系下的实际面积（几何重算，非源属性） |
| `_placeholder` | 空图层占位行标记，算法端跳过 |
| `attr_*` | 原始业务字段全部保留 |

> `area_m2` 用于解决「细碎地物污染整个粗栅格、导致粗走廊方向偏差」的问题：算法端可在粗规划阶段先排除小面确定走廊方向，精规划阶段再恢复全量。

### 8.2 `linear_cross_indexed.gpkg` 字段

道路、铁路、输电线路、管道、可跨越河道按 **50m** 切段，逐段携带：

| 字段 | 含义 |
|---|---|
| `segment_id` | 段唯一标识 |
| `parent_feature_id` | 所属原始线要素。算法端据此保证**同一条线路多次跨越只计一次代价** |
| `level2` / `rule_id` | 类别与规则编号 |
| `cross_cost` | 跨越代价（万元） |
| `min_cross_angle_deg` | 最小交叉角要求（铁路/高速 45°，220kV 30° 等） |
| `azimuth_deg` | 该段方位角（供人工核查；算法端从端点坐标现算，不读此字段） |

### 8.3 `manifest.json` 与交付级别

算法端启动时校验六项，任何一项不通过即拒绝启动：

| 校验项 | 要求 |
|---|---|
| `package_product` | `== "transmission_line_preprocess"` |
| `package_format_version` | 以 `"0."` 开头 |
| `delivery_level` | `!= SEVERE_DEGRADED` |
| `required_missing` | 为空 |
| `operational_mode.mode` | `!= UNAVAILABLE` |
| `coarse_resolution_m` | `== 100`（±0.5） |

| 交付级别 | 含义 |
|---|---|
| `FORMAL_DELIVERY` | 正式交付 |
| `PRELIMINARY_ROUTE_ONLY` | 初步路径：部分数据简化处理，可用于走廊方向判断 |
| `SEVERE_DEGRADED` | 严重降级：算法端拒绝启动 |

> 当前工程为 `PRELIMINARY_ROUTE_ONLY`，降级原因**仅**「风区/覆冰区组合栅格未提供，使用统一参数」—— 属设计安排（见 §10.4），非缺陷。

---

## 九、与算法端的对接契约

### 9.1 算法端实际读取什么

| 产物 | 消费的字段 |
|---|---|
| `manifest.json` | 全域 `wind_zone` / `ice_zone`；启动六关闸门 |
| `m3/rule_config.json` | `rule_id` → `level2` |
| `forbidden / no_tower / cost .gpkg` | `geometry` / `level2` / `rule_id` / `_placeholder` |
| `linear_cross_indexed.gpkg` | `geometry` / `level2` / `rule_id` / `parent_feature_id` / `cross_cost` / `min_cross_angle_deg` |
| `preferred_corridors.gpkg` | 走廊几何 |
| `control_*.gpkg` | 起终点/必经等（可用 `utils/control_io.py` 还原为 dict） |

**算法端不读**：任何 `.tif`、`wind_ice_zones.gpkg`、`area_m2`、`azimuth_deg`、`rule_config` 中的代价字段（当前走算法端内置表）。

### 9.2 硬约束

修改本包时，**文件名 / 字段 schema / 上述契约不可变更**。几何与数值可以改（算法端按值使用），但改名或增删字段会直接打断对接。

### 9.3 `rule_config.json` 的代价字段

已按算法端 `FeatureType` **同名**导出，对接时零改名：

| 字段 | 含义 |
|---|---|
| `cross_cost` | 跨越基础代价（数值） |
| `cross_cost_angle_coeff` | 角度项系数（× cosα） |
| `cross_cost_per_km` | 长度项系数（万元/km） |
| `tower_cost` | 立塔基础代价 |
| `cross_cost_expr` | 原始人读表达式（如 `30+30×cosα`），仅供审计，算法端无需解析 |

> 「禁止」类在编译时折算为哨兵值，导出到数值字段时统一写 **0**；「禁」这件事由 `is_landable` / `cross_allow` 布尔字段表达。**判"禁"请用布尔字段，不要用代价数值判。**

---

## 十、核心算法要点

### 10.1 河流宽窄分治（形态学开运算）

规则表 61/62 条：河宽 < 900m 可跨越，≥ 900m 禁止跨越。判定**不依赖中心线**：

```
腐蚀：河流面从两岸各向内收缩 450m  → 不足 900m 宽的河段自我抵消消失
膨胀：留下的"芯"再向外扩张 450m    → 长回原宽度（会溢出河岸）
裁剪：与原河流面求交               → 裁回河岸内

宽段 = 结果            → forbidden（rule_id = -1）
窄段 = 河流面 − 宽段   → linear_cross（通航 61 / 非通航 62）
```

该判据等价于「此处能否放进一个直径 900m 的圆」，是宽度的**严格几何定义**。性质：宽段 ⊆ 河流面（不会把河边陆地划成禁区）；宽段 ∪ 窄段 = 完整河流面（不重不漏）。两条性质均已写入单测。

**已知边界**：开运算各向同性，「横向很宽但顺流向很短」的水域（如小型水库口）会被判为窄段。当前工程数据无此类，如后续出现将增加横断面兜底判定。

### 10.2 跨河判定基准线

窄河段需要一条线供算法端做两件事：判断是否跨越、校验最小交叉角。

> **概念澄清**：这条线是「跨河判定基准线」，不是测绘意义的河道中心线。验收标准是**贯通、走向正确、不越岸**，不以精确居中为目标。河道占地范围由河流面要素本身表达，不经过这条线。

算法：**横断面中点法 + 局部走向精修**

```
第 0 步  取河段主轴（边长加权 PCA，与岸线顶点疏密无关）
第 1 步  沿主轴每 100m 作垂线截河面，取交弦中点 → 初始线
第 2 步  沿初始线重新布站，截线改为垂直于**局部走向**，取含当前点的那条弦的中点
第 3 步  重复第 2 步共 3 轮
```

不用中轴线（骨架）法的原因：真实岸线锯齿会让骨架长出大量伪分叉，在宽河段与汇流口退化成乱线，本项目实测无法收敛。弦中点法则因两岸锯齿方向相反、取中点相互抵消而天然稳健，且每个河段**只输出一条线**。

**端部处理**（v0.6.5）：布站两端各内缩半站避免站点贴边；端部覆盖由**沿拟合圆弧外推 + 岸距守卫**补回 —— 守卫用「到河岸的距离」（按走向排除端边），因此朝切口推进不受影响、朝河岸漂移立刻停止。

**分岔口**：一个河流面内有多条河道时，取弦可能在支流间切换，产生斜穿折返的假线。现已在跳变（> 2.5 站距）与折返（> 135°）处切开，各段内部走向一致；分段共用同一 `parent_feature_id`，算法端按 parent 去重，不重复计费。

**当前限制**：汇流口次要支流尚无独立基准线（该处不施加最小交叉角约束）。已消除假线，支流识别在后续版本。

### 10.3 河宽量测

沿基准线每 100m 一站，作垂直于局部走向的截线，弦长即该处河宽。

**口径**：S 弯/回折处一条截线可能穿过河道多段 —— 只取**基准线所在的那一段弦**（早期版本各段相加，会把 500m 宽的河算成 1600m 而误判为宽段），并对逐站宽度做滑动中位数消除孤立异常值。

> 注意 **100m 与 50m 是两个不同参数**：100m 是布站间距（算中心点与河宽），50m 是之后切成 linear_cross 段的长度。不是每 50m 算一遍中心线。

### 10.4 地形与风冰

**DEM**：原生分辨率约 1m 且覆盖整个行政区，直接重投影会内存溢出。处理策略为①按工作区 bbox 裁剪（实测节省 99%）②降采样到 `terrain_proc_resolution_floor_m`（默认 10m）。

> 10m 是面向粗规划的有损产物。将来做精细塔位核对（如「转角塔位塔腿方向坡度 ≤ 40°」）应直接对**原始高分辨率 DEM** 做点/小窗口采样，不用这份产物。

**风冰**：甲方数据为矢量（GDB，风速与覆冰两图层）。本包只做**标准化**——重投影到 working_crs、保留原始属性、输出 `wind_ice_zones.gpkg`，**不分档、不套代价、不栅格化**。原因是风冰二维代价模型尚未定稿（当前代价表是「风×冰」绑定的四项对角线，无法表达「风速变化而覆冰为零」）。在此之前规划使用 `project.json` 声明的全域参数 —— 这也是交付级别停在 `PRELIMINARY_ROUTE_ONLY` 的唯一原因。

### 10.5 工作区范围

`bbox = 起终点窗口 + 5km`，M3 再裁到数据范围。当前 **DEM 处理已按 bbox 裁剪**，但 M2 的矢量产物仍按全量产出（算法端只在自己的 grid 内栅格化，结果正确，但 M2 有无效计算）。M2 裁剪到工作区在后续版本规划中。

---

## 十一、常见问题

**Q：写 GPKG 时报 `FieldError: Error adding field 'SHAPE_Area'`？**
A：GeoPackage 字段名大小写不敏感，多源 SHP 合并常出现 `SHAPE_Area` / `Shape_Area` 并存。程序已内置去重兜底（加后缀保留源属性），日志会显示「解决 N 处字段名大小写冲突」，属正常。源头规避更稳妥：数据准备时避免仅大小写不同的重名字段。

**Q：交付级别为什么是 `PRELIMINARY_ROUTE_ONLY`？**
A：见 §10.4。降级原因应**只有**风冰一项；若出现其他原因，说明有非预期问题。

**Q：M2 为什么这么慢（约 200s，占总耗时 90%）？**
A：主要在河流宽度分析（约 130s）与贴近奖励走廊（约 45s）。当前 M2 处理全量地图，而规划实际只用到工作区范围内 —— 裁剪优化在后续版本规划中。

**Q：DEM 太大跑不动？**
A：确认 `solver_params.terrain_proc_resolution_floor_m` 已设（默认 10）。若能提供裁剪到工程周边的 DEM，可显著缩短处理时间。

**Q：改了规则参数要重跑哪些模块？**
A：当前规则表内置于 `config/`，改动后需重跑全流程（`rule_config.json` 是 M3 编译产物，但 M2 的几何分流也依赖规则）。规则库工程级可配置化在后续版本规划中。

---


## 后续规划

- M2 处理范围裁剪到工作区
- 保护范围 / 奖励范围自动生成
- 规则库工程级可配置化（平台基础库 + 工程库）
- 汇流口支流独立基准线

---


