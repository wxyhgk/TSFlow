# TSFlow AI 操作指南

本文件面向后续使用代码代理或其他 AI 工具维护 TSFlow 的开发者。它描述项目边界、常用入口和科学结果的判断规则；不能替代具体模块的测试和源码注释。

## 项目定位

TSFlow 是面向大分子反应的可验证过渡态搜索后端与算法库。公开项目名是 `TSFlow`，当前 Python 兼容包名为 `uma-workflows`，后端包名为 `uma-platform-api`。UMA 是当前主力计算引擎，但项目设计允许接入 xTB、Gaussian 和其他 provider。

核心链路是：

```text
反应物/产物/反应位点
  -> 端点预优化
  -> FB-ENM 或 Direct MaxFlux 路径
  -> 一阶鞍点精修
  -> Hessian 与虚频筛选
  -> 双向 IRC
  -> 端点拓扑和反应坐标验证
```

## 开始工作前

1. 先阅读根目录 `README.md`、`PROJECT_STATUS.md` 和相关模块 README。
2. 用 `git status --short` 查看工作区。工作区可能已有用户改动，不能使用 `git reset --hard`、`git checkout --` 或批量回滚。
3. 先确认任务属于源码、公开文档还是本地研究计算。没有用户明确要求时，不要把 `calculations/`、`mol/`、`data/`、`outputs/` 或 `archive/` 中的文件加入公开源码。
4. 修改前说明将修改的文件和验证范围；保持改动集中，不顺手重命名兼容包或重构无关代码。

## 目录边界

| 路径 | 用途 |
| --- | --- |
| `backend/app/api/routes/` | HTTP 路由 |
| `backend/app/application/` | 任务编排、Worker 和生命周期 |
| `backend/app/contracts/` | 稳定的数据契约 |
| `backend/app/adapters/` | UMA、xTB、Gaussian 等计算引擎适配 |
| `backend/app/calculators/` | 计算模型注册和能力选择 |
| `backend/app/formats/` | XYZ、SDF、Gaussian 等格式处理 |
| `packages/uma-workflows/src/uma_workflows/` | 不依赖 FastAPI 的通用算法 |
| `packages/uma-workflows/tests/` | 算法测试 |
| `backend/tests/` | 后端和 API 测试 |
| `docs/` | 可公开复用的架构、安装和接口文档 |
| `calculations/` | 本地分子案例、算力、日志、IRC 和报告，不进入公开源码 |
| `frontend/` | 本地 Web 工作台，当前不属于首个公开源码包 |

算法包不能导入 FastAPI、后端 Pydantic schema、数据库 repository 或项目私有路径。新的计算引擎应先实现 adapter/calculator 边界，再声明实际验证过的能力。

## 常用命令

从仓库根目录执行：

```bash
# 安装环境
uv venv --python 3.13 .venv-uma
scripts/install-backend-env.sh

# 启动 API
scripts/run-backend.sh

# 后端恢复烟测
scripts/test-backend-smoke.sh

# 算法包测试
.venv-uma/bin/python -m pytest -q packages/uma-workflows/tests

# 语法检查
.venv-uma/bin/python -m compileall -q backend packages/uma-workflows/src

# 查看模型和 provider 能力
curl -s http://127.0.0.1:8010/api/v1/models
```

完整测试可能收集到 `archive/source-recovery/` 中的历史副本；遇到这类失败时，区分历史归档问题和当前源码问题，不要为了让全量收集通过而改动科学代码。

## 科学结果规则

- `succeeded` 只表示任务执行完成，不表示过渡态已被科学验证。
- 合格 TS 至少需要驻点收敛、全体系 Hessian、一个显著虚频、虚频方向与反应坐标相符，以及双向 IRC 连接预期端点。
- 还要检查 `irc_validation_passed`、`scientific_outcome`、`screening_diagnostic_codes`、`irc_error` 和 `saddle_refinement_converged`。
- 路径最高点、部分 Hessian、动画或单个结构图不能单独证明 TS。
- 电子能垒不等于包含溶剂、温度和熵效应的实验活化自由能；不同计算引擎或不同参考态的能量不能直接相减。
- 未实际运行的模型、API、浏览器或计算不能写成“已验证”。明确区分静态检查、执行完成和科学验证。

## 修改原则

- 优先复用现有契约和 provider 注册表，不为单个案例复制一套后端逻辑。
- 保持原子映射、反应位点和坐标顺序稳定；输入校验失败时给出原子和帧的具体位置。
- 长任务必须传递取消标记并保留中间 artifact；不要用插值结构冒充已完成的路径搜索。
- 不在源码、日志、任务附件或文档中写入 API key、模型权重或机器私密信息。
- 只运行与改动风险匹配的测试；涉及跨模块契约时同时运行算法和后端相关测试。

## 发布边界

源码采用 MIT，见根目录 `LICENSE`。UMA 权重及其上游许可证单独管理，不因本项目采用 MIT 而改变。公开源码导出使用 `scripts/export-source.py` 的显式路径清单；发布前必须人工检查第三方来源、研究数据和许可证。

## 给 AI 的输出要求

完成任务时说明：修改了哪些文件、为什么修改、运行了哪些命令、哪些结果已验证、哪些仍未验证。不要把本地研究案例的成功直接推广为通用算法性能，也不要把测试通过写成科学结论成立。
