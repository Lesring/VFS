# 视频取证推理系统 (VFS)

输入一段视频 + 一个问题，系统自动抽帧、判读画面、OCR 取字、联网核实，最后给出**带推理链和置信度**的答案。

> 例：输入一段飞机视频，问「这架飞机的目的地是哪里？」
> 系统会读出机型与涂装、识别航班信息屏上的航班号、查航线库反推起降机场、再用画面里的机场做交叉校验。

---

## 它不只是"把视频丢给大模型"

大多数同类方案就是"抽几帧 → 问多模态模型 → 复述模型的话"。问题是模型会**自信地编造**机型、航班号和机场。

本系统的做法是把它做成一条**取证推理流水线**：

```
视频 → 抽帧 → 视觉判读 ─┐
              OCR 取字 ─┼→ 证据池 → 实体归一 → 加权融合 → 联网核实 → 约束校验 → 答案 + 推理链
        本地图像统计 ─┘                                        ↓
                                                        冲突检测（扣分/告警）
```

关键设计：

| 机制 | 作用 |
|---|---|
| **证据池 + noisy-OR 加权** | 同一事实被多个独立来源确认才升高置信度，单一来源永远压不到"高" |
| **实体归一化** | `Airbus A320` / `A320-214` / `A320 214SL` 归到同一个 key，否则跨源比对全失效 |
| **约束推导** | 航班号 → 航司 + 起降机场；注册号 → 机型 + 机主；坐标 → 最近机场 |
| **交叉校验（最有价值）** | 画面看到的机场 vs 航班实际起降机场；画面涂装 vs 注册号登记机主；机型航程 vs 航线距离 |
| **循环推理防护** | 由航线库反查出的坐标不会反过来"证实"同一个机场，否则就是自证 |
| **诚实弃答** | 证据不足时明确说"信息不足"，而不是编一个答案 |
| **可插拔视觉模型** | OpenAI 兼容 / Ollama / Anthropic / mock，换个 `provider` 即可 |

---

## 快速开始

```bash
pip install -r requirements.txt

# 方式一：Web 界面（推荐）
python -m vfs.app
# 打开 http://127.0.0.1:8000 ，点"用内置样例试跑"即可看到完整效果

# 方式二：命令行
python -m vfs 你的视频.mp4 -q "这架飞机的目的地是哪里？" --trace
```

### 内置样例

没有视频也没有 API key 也能先跑通：

```bash
python tools/make_demo_video.py          # 生成合成机场视频 + 预录观察
python -m vfs data/demo/airport_plane.mp4 -q "这架飞机的目的地是哪里？"
```

样例刻意埋了一处矛盾（画面是浦东 PVG，但 MU5101 实际从虹桥 SHA 起飞），
用来演示系统的交叉校验会主动把这个问题指出来，而不是闷头给答案。

---

## 配置：页面里点几下就行

打开界面点右上角 **⚙ 配置**，可以直接设置模型、检索策略、Google key、OCR 等，**保存即生效，无需重启**。

- 按「模型 / 检索 / Google / 其他」分组
- API Key 输入框只写不读：保存后回传的是 `sk-t****7890` 这种打码值，留空表示不修改
- 面板顶部实时显示**各数据源可达性**，标出哪些被跳过了，头部标签也会显示「境外可用」还是「已降级国内」
- 配置保存在 `data/user_config.yaml`（与代码分离，升级不覆盖）

**不配也能用**，但只有 OCR 和本地图像统计，不含语义判读。要真正"看懂"画面，配一个多模态模型：

| 后端 | base_url 示例 | model 示例 |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-vl-max` |
| 智谱 | `https://open.bigmodel.cn/api/paas/v4` | `glm-4v-plus` |
| 本地 Ollama | `http://127.0.0.1:11434` | `qwen2.5vl:7b` |

也可以复制 `config.example.yaml` 为 `config.yaml`，或用环境变量（优先级：环境变量 > 页面配置 > config.yaml）。

没有 key 时 `provider=mock`，系统会**如实告知**"未配置视觉模型"，只用 OCR 和本地统计，绝不伪造语义结论。

---

## 联网数据源：境外优先，不可达自动降级

**不是固定顺序硬试**——被墙域名会让每个查询都卡 5 秒以上。系统先用带超时的 DNS 预检
判断可达性（结果缓存 10 分钟），不通的源直接跳过。

| 数据源 | 用途 | key |
|---|---|---|
| **Google (Programmable Search)** | 真谷歌搜索 | 需 key + cx |
| **Google News RSS** | 航班动态 / 近期事件 | 免 key |
| **Google Suggest** | 实体名纠错补全 | 免 key |
| **Google Maps** | 地理编码、静态地图、嵌入地图 | 需 key |
| **OpenStreetMap** | 地图兜底（服务端拼瓦片） | 免 key |
| **adsbdb** | 呼号 → 航线；注册号 → 机型+机主 | 免 key |
| **OpenFlights** | 机场库 7698 个、航司库 | 免 key |
| **api.wikimedia.org** | 机型/航司/机场百科 | 免 key |
| **Bing / 搜狗 / 百度** | 通用检索 | 免 key |
| **Tavily / Brave / Serper** | 境外检索备选 | 需 key |

`region_strategy` 控制区域策略：`overseas_first`（默认）→ `domestic_first` → 只用一个区域。

### 关于 Google 的两个实测结论

1. **`google.com/search` 不能直接爬**。它返回的是纯 JS 外壳（94KB、零个 `<h3>`），
   内容靠前端渲染。所以"真谷歌搜索"必须走 Programmable Search API。
   没配 key 时系统自动退到免 key 的 Google News RSS，不会静默失败。
2. **Bing 的外链全部包在 `/ck/a` 跳转里**，真实 URL 在 `u=a1<base64>` 参数中，
   必须解码否则结果全丢。另外无 cookie 爬 Bing 会混入与查询无关的结果，
   系统做了相关性过滤（宁可返回空，也不让噪声进证据池）。

`python tools/probe_google.py` / `tools/probe_sources.py` 可随时探测当前网络。

---

## 系统识别哪些信息

| 槽位 | 说明 | 典型来源 |
|---|---|---|
| `flight_number` | 航班号 | 信息屏、机身喷涂、OCR |
| `registration` | 注册号 | 机身小字 |
| `aircraft_type` | 机型 | 视觉判读 + 注册号反查 |
| `airline` | 航司 | 涂装 + 航班号反查 |
| `airport` / `origin` / `destination` | 机场与起降地 | 画面标识 + 航线库 |
| `city` / `country` / `coordinate` | 地点 | 机场库、画面角标 |
| `date` / `time_of_day` / `weather` | 时间与天气 | 画面时间戳、光照分析 |
| `landmark` / `object` / `activity` | 场景要素 | 视觉判读 |

问题问什么，系统就先答什么；主目标答不出时会明确说明，并退让到能答的其他信息。

---

## 需要配置什么？按需选择

**零配置就能跑**：OCR + 本地图像统计 + adsbdb 航线 + OpenFlights + Bing/搜狗 + OSM 地图。

想解锁更强能力，按价值排序：

| 配置项 | 解锁什么 | 必要性 |
|---|---|---|
| **视觉模型**（provider + key） | 机型/涂装/文字语义判读 —— 这是系统"看懂画面"的前提 | ★★★ 强烈建议 |
| **Google API key + cx** | 真谷歌搜索（News RSS 覆盖面窄） | ★★ 建议 |
| **Google Maps key** | 地理编码、卫星静态图、交互式嵌入地图（否则用 OSM） | ★ 可选 |
| **tesseract**（系统安装） | 传统 OCR 兜底，补模型易忽略的小字 | ★ 可选 |
| Tavily / Brave / Serper key | 境外搜索冗余 | 可选 |

> Programmable Search 与 Maps 是**两套独立 API**，同一把 Google key 需要在 Cloud Console
> 分别启用 Custom Search API / Geocoding API / Maps Static API。

## 输出长什么样

真实运行输出（内置样例）：

```
**结论：目的地 — Beijing Capital International Airport (PEK/ZBAA)**（置信度 74%，中（存在矛盾，需复核））

**需要注意的矛盾点：**
- 画面机场 Shanghai Pudong International Airport（PVG） 与航班实际起降机场
  Shanghai Hongqiao International Airport（SHA）、Beijing Capital International Airport（PEK） 均不匹配；
  可能是中转/经停、画面拍的并非该航班、或航班号被误读

**推理链：**
1. 画面中读到航班号 =「MU5101」（0.5s）：航班信息屏首行高亮 MU5101
2. 以该航班号为线索查外部数据源
3. 外部数据源返回 → 航班 MU5101 的目的地机场（adsbdb）
4. 但存在上述矛盾点，综合置信度 74%，结论应视为待核实
```

注意最后一步：它没有说"多源交叉验证一致"，而是承认矛盾。这正是本系统与"直接复述模型输出"的区别。

---

## 项目结构

```
vfs/
  frames.py       抽帧（场景切换 + 清晰度 + 均匀兜底），OpenCV 解码，无需 ffmpeg
  heuristics.py   本地 CV：天色/光照/植被/地平线倾角 → 时段、室内外、天气先验
  ocr.py          tesseract 封装，未安装则静默跳过
  entities.py     实体归一化（机型/航班号/注册号/日期/坐标）—— 跨源比对的基础
  reason.py       证据融合 + 约束推导 + 交叉校验（系统核心）
  retrieval.py    联网核实编排
  answer.py       答案与推理链生成
  pipeline.py     全流程编排
  app.py          FastAPI + SSE 实时进度
  static/         单页前端
  settings.py     页面配置：写入 user_config.yaml，热重载，密钥打码
  kb/
    health.py     数据源可达性探测（带超时 DNS）与降级决策
    google.py     Google 搜索 / News RSS / Suggest / Maps / OSM 拼图
    search.py     各搜索引擎适配 + 相关性过滤
    airports.py   OpenFlights 机场库
    flights.py    adsbdb 航线/航空器 + 机型归并
tools/
  make_demo_video.py   生成合成演示视频
  probe_sources.py     探测数据源可用性
  inspect_result.py    调试：查看槽位构成与证据归属
  test_web.py          Web 接口端到端自测
tests/            67 个测试（含端到端集成）
```

---

## 测试

```bash
python -m pytest tests -q
```

测试断言的是**行为契约**而非具体答案：有航班号必须能反查目的地；画面与航线冲突必须报出来；
没有视觉模型时必须弃答而不是编造；不同写法的同一机型必须归到同一个 key。

---

## 已知限制

- **机型判读依赖视觉模型质量**。细分型号（A320ceo vs neo）常需看清发动机和翼梢，模型容易给错，所以系统只把它当弱证据。
- **航班号是强线索但不可尽信**。同一航班号在不同日期可能飞不同航段，本系统查的是"固定航线"，不能替代当日航班动态。
- **画面里的机场要能读出来**。没有航班信息屏、没有角标时，机场槽位往往是空的，交叉校验也就无从触发。
- **OCR 需系统装 tesseract**，未装则自动跳过（不报错）。
- **视频解码依赖 OpenCV**。少数编码（如 HEVC）可能打不开，建议先转 H.264 MP4。

---

## 数据来源与许可

- 机场/航司数据：[OpenFlights](https://openflights.org/data.html)（ODbL）
- 航线/航空器：adsbdb（免费公共接口）
- 百科内容：Wikimedia（CC BY-SA）
