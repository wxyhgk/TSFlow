# 架构与扩展

```text
Web / HTTP 调用方
    ↓
backend/app/api/routes/          HTTP 路由注册与请求处理
    ↓
contracts/ + application/jobs/  请求契约、进程 Worker、任务生命周期
    ↓
services/ + calculators/        工作流编排、模型与 provider 选择
    ↓
adapters/{uma,xtb,gaussian}/    各引擎的 ASE 计算适配
    ↓
packages/uma-workflows/         与 HTTP 无关的几何、路径、鞍点、频率和 IRC 算法
```

## 路由替换

`backend/app/api/routes/registry.py` 提供 `RouteDefinition` 与 `build_router()`。
`routes/__init__.py` 中的 `create_router()` 组合默认 health、models、actions、jobs 路由。
新路由模块通过注册加入，避免把业务逻辑堆在 `api/router.py`。

## 计算引擎替换

模型定义在 `backend/app/calculators/models.py`，provider 解析在
`backend/app/calculators/registry.py`，实际引擎适配在 `backend/app/adapters/`。
算法库接受 ASE calculator；新增模型优先适配计算器，再声明已验证的任务能力。
解析 Hessian、有限差分、设备和外部程序调用的支持情况需分别记录。

后端当前仍直接依赖 fairchem-core；可替换引擎不意味着已经实现完全不安装 UMA 依赖的轻量后端。
ORCA 离线复核流程尚未变成通用 provider。

## 算法替换

`PathInitializer`、`PathOptimizer` 等契约在算法包内，FB-ENM 和 Direct MaxFlux 是现有实现。
宏工作流位于 `transition_states/workflow/`，端点、路径、鞍点、筛选、IRC 分阶段执行。
回调报告进度，取消标记向下传递；算法库不导入 FastAPI 契约。

## AI API 的后续位置

建议新增独立的规划层，负责将自然语言、结构和反应位点转换为明确的反应定义与计算任务。
规划层通过现有 HTTP 契约或算法 API 调用计算；API 密钥来自运行环境，不写入任务附件或仓库。
现阶段尚未实现这一层，不能仅凭填写 AI API 就自动获得已验证 TS。
