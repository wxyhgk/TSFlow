# 后端服务模块

## 过渡层说明

`services/` 目前是运行代码与恢复兼容代码的混合过渡层。不要继续在这里新增
大模块；新代码按 `backend/app/README.md` 的目标归属放置。

## 当前运行模块

- `platform_executor.py`：Worker 使用的统一计算分发器。
- `cli_service.py`：`uma` CLI 提交本机 API 任务的客户端。
- `model_registry.py`：旧兼容转发；模型资产发现和可用性实现位于 `../calculators/models.py`。
- `../calculators/`：统一计算器接口和引擎选择；当前包含 UMA、xTB、Gaussian 适配器。
- `geometric_optimizer.py`：旧兼容转发；实现位于 `../adapters/uma/geometric.py`。
- `dmf_executor.py`、`sella_refiner.py`：科学计算适配或兼容入口。
- `transition_state_search/`：TS 搜索的服务层适配。

TS 任务的新正式入口是 `app.application.transition_states`。其中
`TransitionStateExecutor` 负责 API 到共享工作流的转换；本目录下的
`dmf_executor.py` 和 `transition_state_search/*_stage.py` 只保留旧导入路径，
不再复制路径、鞍点或筛选实现。

格式实现已经移到 `app.formats`；`xyz_parser.py`、`sdf_distance_geometry.py` 和
`gaussian_input.py` 现在只是兼容转发，不要在这些旧文件中新增逻辑。

## Legacy 兼容模块

- `job_worker.py`：转发到 `application.jobs.local_worker.LocalJobWorker`，当前主服务不使用。
- `process_job_worker.py`：转发到 `application.jobs.process_worker.ProcessJobWorker`，保留给旧导入路径。
- `uma_executor.py`：旧 UMA Executor 导出兼容层。

当前主入口在 `backend/app/main.py`，直接创建 `application.jobs.ProcessJobWorker`。
修改 Worker 或执行器时优先改 `application/jobs/`、`platform_executor.py`、`calculators/` 和对应的 `adapters/`，
不要在兼容模块中复制实现。

## 剩余迁移范围

1. 将 `platform_executor.py` 的标准计算分派继续收敛到 `application/calculations/`。
2. 将 `xyz_parser.py`、`sdf_distance_geometry.py`、`gaussian_input.py` 的调用方迁移完，再删除兼容转发。
3. 完成调用方迁移后，再删除 `schemas/`、`repositories/`、`artifacts/` 和旧 Worker/UMA 转发模块。

每一步都应先更新调用方和测试，再删除旧路径；不要为了目录好看一次性改写全部 import。
