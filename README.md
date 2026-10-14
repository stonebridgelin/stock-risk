# stock-risk

为单只美股提供风险预警通知；出现可观察的恢复证据后，发出恢复观察和预警解除通知，每次变化都说明依据。系统只报告证据，不给操作或仓位建议，不做预测。

当前阶段：**探索期**。只使用截至 2019-12-31 的数据，2020 年起的数据封存，留作验证。

## 快速开始

```
uv sync
cp .env.example .env              # Windows: copy .env.example .env，然后填写 TIINGO_API_KEY
uv run pytest
uv run stock-risk fetch-raw
```

## 文档

开发文档在 [docs/develop_docs](docs/develop_docs/README.md)，先读 01（项目结构与分层）和 02（数据获取）。
