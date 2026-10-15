# T08d-2a Massive 数据源试验（免费档）

签发：产品经理 · 2026-10-14 · 状态：已签发 · 2026-10-15 补签第 2.4 节与第 7 节（续做）
依据：发起人 2026-10-14：考虑用 Massive（原 Polygon.io）替代 Norgate；已在 `.env` 设置免费版 `MASSIVE_API_KEY`。17 号评价协议第 1.1 节（选参集的数据要求）；03 计划第 2.9 节（Tiingo 交易所字段不是当时取值的问题）。

**目的**：在付费之前，用免费档回答三个问题，供发起人决定是否购买 Massive 付费档。

1. Massive 能否按**历史日期**给出当时在 NYSE、Nasdaq 交易的普通股与 ADR 清单，包括后来退市的证券？
2. 按日期查询时，**主交易所字段是不是当时的取值**？
3. 已知的退市证券、代码复用、改名，Massive 能否正确区分？

**本卡只做探查，不下载价格**（免费档只有最近 2 年的价格，也用不上），不改生命周期规则与评价协议。

## 必读

- `CLAUDE.md`、`docs/tasks/README.md`
- 02 号第 1 节（封存纪律）、第 11 节（选参集数据工具）
- T08d-1 的 `probe-tickers` 实现（封存处理的写法照此）
- Massive 文档：`https://massive.com/docs/rest/stocks/tickers/all-tickers.md`、`https://massive.com/docs/rest/stocks/tickers/ticker-overview.md`。认证方式以文档为准；如果密钥要放在 URL 里，任何日志、异常信息与输出文件中都不得出现完整 URL。

## 0. 前置修补（Gemini 第二阶段审核 R1，产品经理已接受）

先做这些修补并单独提交，再做第 1 节起的内容。

- **B-01 分层越界**：`interfaces/cli/main.py`、`selection.py` 不得直接导入 infrastructure，改为经 `bootstrap.py` 注入（或注入到相应的 application 服务）。`pyproject.toml` 增加 forbidden 契约：禁止 `stock_risk.interfaces` 直接导入 `stock_risk.infrastructure` 与 `stock_risk.domain`。现有其他越层导入如有，一并修正并在回执中列出。
- **B-02 MANIFEST**：`audit-raw` 在 MANIFEST.sha256 缺失或为空，或被审计证券的 CSV 不在清单中时，作为阻断问题返回 1；补测试。
- **B-03 参数与不变量**：
  - 不变量 2 补"confirm ≠ BOX 时 box\_high 必须为空"；
  - `ExplorationAssumptions` 的 h、w\_long，`V08Switches` 的 x₁、x₂ 改用 `_finite`、`_positive_int`，拒绝 bool 与非整数；补测试。
- **B-04 下载工具**：
  - 只有确实发出请求时，才把代码计入当月新代码额度；
  - `audit-raw --dir` 区分 `_fetch_status.json` 中已确认 no\_data 的代码与真正缺失的文件。
- **文档同步**：
  - 15 号第 3.3 节补齐 episodes.csv 的 5 列（entry\_path、delta\_start、prev\_episode\_id、days\_since\_prev\_release、start\_below\_prev\_l\_ep）；
  - 16 号第 5.4 节补 `long_path_off`，第 7 节改为 636 套；
  - 14 号第 8 节删去"期望待产品经理确认后写入 13 号"（已写入 13 号末尾）；
  - 02 号第 11 节开头加注"选参集正式数据源待定（Massive 试验中），本节 Tiingo 工具保留，用于探查与双源对照"；
  - T08d1 任务卡第 2 节的 `audit-raw --dir <out>` 补上 `--list <csv>`。
- V0 回归不需要重跑（以上都不改生命周期行为）。若 B-03 的不变量补充在现有夹具或回放上触发新的违反，停止并报告。

## 1. 适配器

- 新增 `infrastructure/massive/`：只读适配器，封装 All Tickers（`/v3/reference/tickers`，分页用 `next_url`）与 Ticker Overview（`/v3/reference/tickers/{ticker}`）。
- 密钥：`.env` 的 `MASSIVE_API_KEY`，使用 T08a 的 `Secret` 类型。
- **限频**：默认每分钟 5 次（免费档），可配置；遇 429 退避重试，连续 3 次 429 即保存进度并退出。
- **封存**：请求参数 `date` 不得晚于 2019-12-31，违反即拒绝执行（用 `domain/seal.py` 的常量）。

## 2. 命令：`stock-risk probe-massive`

### 2.1 历史日期的代码清单

- **日期**：2010-01-04、2012-06-01、2015-06-01、2019-06-03。
- **查询**：每个日期，对 `exchange` ∈ {XNYS, XNAS}、`type` ∈ {CS, ADRC}，分别用 `active=true` 与 `active=false` 查询，翻完全部分页。
- **报告**：
  - 每种组合的条数；
  - 同一 composite\_figi 在不同日期的主交易所是否变化，列出变化的例子；
  - 若历史日期被免费档拒绝或返回为空，**照实记录错误信息与 HTTP 状态，不算失败**，继续下一项。这本身就是要回答的问题。

### 2.2 已知案例（产品经理提供）

用 Ticker Overview 加 `date` 查询下表各案例，记录 `active`、`primary_exchange`、`type`、`cik`、`composite_figi`、`share_class_figi`、`delisted_utc`（按 2.3 节做封存处理），以及查不到的情况。

| 类别 | 代码 | 查询日期 | 说明（产品经理的已知信息，供对照） |
| --- | --- | --- | --- |
| 破产后转场外 | LEH | 2008-06-02 | 雷曼兄弟，NYSE，2008 年破产 |
| 破产后转场外 | GM | 2009-03-02 | 旧通用汽车，NYSE，2009 年破产；新通用汽车 2010 年用同一代码上市 |
| 破产后转场外 | EK | 2011-06-01 | 柯达，NYSE，2012 年初摘牌 |
| 破产后转场外 | RSH | 2014-06-02 | RadioShack，NYSE，2015 年摘牌 |
| 破产后转场外 | SHLD | 2015-06-01 | 西尔斯控股，Nasdaq，2018 年摘牌（探索集已有，见 `config/listing_periods.csv`） |
| 破产后转场外 | WFT | 2018-06-01 | Weatherford，NYSE，2019 年摘牌（同上） |
| 破产后转场外 | BTU | 2015-06-01 | 皮博迪能源，NYSE，2016 年破产 |
| 破产后转场外 | SUNE | 2015-06-01 | SunEdison，NYSE，2016 年破产 |
| 破产后转场外 | DNDN | 2014-01-02 | Dendreon，Nasdaq，2014 年破产 |
| 破产后转场外 | GTAT | 2014-06-02 | GT Advanced，Nasdaq，2014 年破产 |
| 被收购 | HNZ | 2013-01-02 | 亨氏，NYSE，2013 年被收购 |
| 被收购 | DELL | 2013-01-02 | 戴尔，Nasdaq，2013 年私有化；DELL 代码 2018 年起另属 Dell Technologies（NYSE） |
| 被收购 | MON | 2017-06-01 | 孟山都，NYSE，2018 年被收购 |
| 被收购 | LLTC | 2016-06-01 | 凌力尔特，Nasdaq，2017 年被收购 |
| 代码复用 | GM | 2011-06-01 | 新通用汽车（与上面的旧 GM 对照 FIGI 与 CIK） |
| 代码复用 | DELL | 2019-06-03 | Dell Technologies（与 2013 年的 DELL 对照） |
| 代码复用 | ADT | 2013-06-03 与 2019-06-03 | 2012 年分拆的 ADT 与 2018 年重新上市的 ADT |
| 改名 | FB | 2015-06-01 | Facebook，Nasdaq（探索集已有）。只核对是否查得到，**不记录名称字段** |
| 换代码 | VRX | 2016-06-01 | Valeant，NYSE（探索集已有，2018 年改为 BHC） |
| 对照 | MSFT | 2012-06-01 | 正常情形 |

查询日期都不晚于 2019-12-31。表中"说明"一列是产品经理的已知信息，用来对照，不作为测试期望；**结果与说明不符的，照实列出，不要调整**。

### 2.3 封存处理（照 `probe-tickers`）

- 响应只在内存中处理，原始响应不写盘。
- `delisted_utc` 晚于 2019-12-31 的，写为空，另加标记 `delisted_after_2019 = true`，不保留真实日期。
- `list_date` 晚于 2019-12-31 的行丢弃，不计数。
- **不保存** `name`、`last_updated_utc`、`description` 等可能带有 2020 年以后信息的字段。
- 输出到 `data/massive_probe/`（不入库）：`lists_<日期>.csv`、`cases.csv`、`summary.json`。

### 2.4 追加探查（2026-10-15 补签）

第一次真实运行发现：2019 年以前 XNAS 普通股几乎为 0（2010 年 0 条，2012 年 18 条，2015 年 67 条），各日期的 ADRC 在 2019 年以前都是 0。同时，案例表中所有 Nasdaq 证券（SHLD、DNDN、GTAT、DELL 2013、LLTC、FB、MSFT）的 `type` 都为空。产品经理的假设是：历史快照里许多证券没有 `type`，被 `type=CS` 过滤掉了，而不是 Massive 没有这些证券。本节要验证这个假设，**照实记录，不作结论**。

- **A. 去掉类型过滤**：4 个日期 × `exchange` ∈ {XNYS, XNAS} × `active` ∈ {true, false}，**不传 `type`**，翻完全部分页。每种组合报告：
  - 按 `type` 取值的条数，空值单列为"(空)"；
  - `cik`、`composite_figi` 非空的条数；
  - `active=false` 的组合另报 `delisted_utc` 非空的条数（封存后，含 `delisted_after_2019` 标记的条数）。
- **B. 去掉交易所与类型过滤**：4 个日期，`market=stocks`、`active=true`，不传 `exchange` 与 `type`，翻完全部分页。按 `primary_exchange × type` 交叉计数（空值单列）。这一项用来发现 XNAS 以外的 Nasdaq 历史代码（如 XNGS、XNMS、XNCM）或其他取值。
- **C. 补充案例**（Ticker Overview 加 `date`，字段同 2.2 节）：

  | 代码 | 查询日期 | 对照目的 |
  | --- | --- | --- |
  | MSFT | 2010-01-04 | 2010 年 Nasdaq 大盘股能否查到，主交易所与 type |
  | AAPL | 2010-01-04 | 同上 |
  | CME | 2010-01-04 | 第一次运行显示 2010 年为 XNYS；产品经理印象中 CME 2008 年已转 Nasdaq，需对照 |
  | BIDU | 2012-06-01 | Nasdaq 上市的 ADR，type 是否为 ADRC |
  | SNY | 2015-06-01 | NYSE 上市的 ADR，type 是否为 ADRC |
  | BABA | 2015-06-01 | NYSE 上市的 ADR（2014 年上市） |

- **D. 与 Tiingo 探查对照**（不发请求，用 T08d-1 的 `data/universe_probe/` 输出）：对每个日期，取 Tiingo 中 exchange ∈ {NYSE, NASDAQ}、资产类型 Stock、币种 USD、开始日 ≤ 该日期 ≤ 结束日（结束日只有"2019 年后结束"标记的，视为覆盖到该日期）的代码集合，与 A 中 `active=true` 的代码集合（两个交易所合并，不论 type）按代码比较。报告：
  - 两边都有、只在 Massive、只在 Tiingo 的条数；
  - 各列 10 个例子（代码与两边的交易所、type）。
  - 这一项同时作为第 4 节回执第 4 项的内容。
- **封存**：同 2.3 节。输出到 `data/massive_probe/followup/`：`lists_<日期>.csv`（A 的行）、`crosstab_<日期>.csv`（B 的计数，不保存 B 的逐行内容）、`cases.csv`、`coverage.csv`、`summary.json`。
- **请求量**：预计约 150 次，按每分钟 5 次约 30 分钟。B 的第一个日期翻页超过 40 页时，停止 B，报告实际页数与已得到的计数，其余照做。
- **实现**：在 `probe-massive` 上加选项（例如 `--followup`），或新增子命令，二选一，在回执中说明；`type`、`exchange` 改为可选参数。补测试：不传 `type`、`exchange` 时请求参数中不出现这两项；交叉计数；空值计为"(空)"；对照集合的计算。

## 3. 测试

- 用假传输层测试：限频与 429 退避；分页；`date` 晚于 2019-12-31 时拒绝执行；封存处理（`delisted_utc` 改写、字段丢弃、原始响应不落盘）；URL 中的密钥不出现在日志与异常信息中。
- 测试不访问网络、不用真实密钥。

## 4. 真实运行与回执

- 真实运行一次 `probe-massive`。按每分钟 5 次估计，约需 20–30 分钟；可以中断后续传。
- **回执列出**：
  1. 2.1 节每种组合的条数；历史日期是否被免费档拒绝（原文错误信息，去掉密钥）；
  2. 2.2 节每个案例的结果表；
  3. 同一 FIGI 主交易所随日期变化的例子；
  4. 与 T08d-1 的 Tiingo 探查对照：同一时期退市的证券，Massive 有、Tiingo 没有的条数，反之亦然（按代码与区间粗略对照即可）；
  5. 实际调用次数与耗时。
  6. 第 2.4 节 A–D 的结果（A、B 的计数表，C 的案例表，D 的对照条数与例子）。
- 不作结论性判断，结论由产品经理写。

## 5. 文档

- 02 号新增"Massive 试验"一小节：命令用法、限频、封存处理与输出目录。
- `docs/tasks/README.md`、00 号变更记录。

## 6. 验收

- [ ] 第 0 节前置修补完成，单独提交
- [ ] 适配器与 `probe-massive` 完成，各有测试
- [ ] 真实运行一次，回执给出第 4 节的六项内容
- [ ] 第 2.4 节追加探查完成，单独提交
- [ ] 输出文件中没有 2020 年以后的日期，也没有名称字段（用 grep 核对）
- [ ] pytest、lint-imports、ruff（含 format --check）全部通过
- [ ] 文档同步

## 停止条件

- 任何环节需要读取或保存 2020-01-01 以后的日期。
- 认证方式只能把密钥放在 URL 中，且无法保证日志与异常不泄露。这时停止并报告，不要绕过。

## 不做

- 不下载价格，不做付费档的任何操作。
- 不改 `domain/seal.py` 的常量，不改生命周期规则与评价协议。
- 需要暂存产品经理未提交的文档时，先在对话中告知。

## 7. 续做（2026-10-15）

上一个会话在第一次真实运行结束后中断，没有提交，也没有输出回执。现状：

- 第 0 节已提交（`cc1f1bc`）。
- 第 1–3 节的代码与测试、第 5 节的文档（02 号第 12 节、01 号、00 号条目、`docs/tasks/README.md`）都已在工作区，未提交。
- 第一次真实运行已完成，输出在 `data/massive_probe/`（`summary.json` 显示 66 次请求，781.9 秒）。

**按顺序续做：**

1. **不要重跑第一次探查。** 只读现有输出，补做第 6 节的 grep 核对（没有 2020 年以后的日期，没有名称字段），跑 pytest、lint-imports、ruff（含 format --check）。
2. **第一次提交**：第 1–3 节代码与测试，以及第 5 节文档中你写的部分。00 号与 `docs/tasks/README.md` 中有产品经理未提交的内容，照以往做法以 HEAD 为底，只暂存你自己的条目。产品经理的其他文件（03、04、06、10、13、17 号，reviews，analysis/T07，本任务卡）不要提交。
3. **做第 2.4 节**：实现、测试、真实运行，然后同步 02 号第 12 节与 00 号条目，**第二次提交**。
4. **输出完整回执**（第 4 节六项，加上两个提交号与偏离任务卡之处）。

真实运行如果要在后台进行，会话要保持到运行结束、回执输出之后，不要提前结束。
