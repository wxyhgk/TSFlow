# TSFlow 后端

本后端包的发行标识仍为 `uma-platform-api`，这是当前 Python 环境和部署脚本使用的兼容名称；项目品牌统一为 **TSFlow**。

本目录仅负责 FastAPI、任务 Worker、存储与科学计算适配。Web UI 位于 `../frontend/`；
算法库位于 `../packages/uma-workflows/`，由 `pyproject.toml` 的本地依赖声明关联。
后端通过 HTTP 提供 `/api/v1`，不依赖前端开发服务器运行。

## 启动与测试

从仓库根目录运行：

```sh
scripts/run-backend.sh
scripts/test-backend-smoke.sh
uma --help
```

启动脚本使用根目录 `.venv-uma`，默认启用单个进程 Worker、CPU 计算，监听 `127.0.0.1:8010`。
Worker 的 UMA 子进程会在首个任务时加载模型，并在后续任务间复用同一进程内的 predictor；
默认空闲 1800 秒后退出，由操作系统释放 CPU/MPS 内存。设置
`UMA_WORKER_MAX_TASKS_PER_PROCESS` 为正数可额外启用任务数上限，默认 `0` 表示不按任务数退出。
数据库为 `../data/jobs.sqlite3`，附件为 `../data/artifacts`，模型为 `../model-assets/UMA`。
这些位置由 `app/core/config.py` 相对源码位置解析，不依赖启动终端目录。
环境变量可覆盖路径；请勿为现有 SQLite 队列启动多个 API 实例。

`uma` 是本后端的 CLI，入口为 `app.cli:main`，通过 `--service-url` 连接持久服务，
用于从单分子 2D SDF 生成距离几何、提交 UMA/TRIC 优化并写出 Gaussian GJF。
从源码环境中也可使用 `scripts/uma`，它显式加载根目录虚拟环境和共享工作流包。
如果 `command -v uma` 指向旧的 uv tool，请重新执行：

```sh
uv pip install --python .venv-uma/bin/python --no-deps -e backend
```

重建当前恢复环境（从根目录）：

```sh
uv venv --python 3.13 .venv-uma
scripts/install-backend-env.sh
```

`scripts/install-backend-env.sh` 会把本地 `tsflow-chem` 算法包和后端以 editable 方式安装，
并补齐 CLI 在 MPS/距离几何路径使用的 `pymatgen`。环境快照与项目依赖声明也已保留。

## 当前支持

- 结构 JSON 创建任务、单点能、几何优化、频率分析。
- 提供 TS 初猜的 geomeTRIC 精修、Hessian 筛选、双向 IRC 与端点匹配。
- SQLite 任务重载、结果附件及 Action 幂等提交。
- 计算引擎通过 `model_name` 选择。UMA checkpoint 已可用；xTB (`xtb-gfn2`) 和 Gaussian (`gaussian-b3lyp-def2svp`) 会在 `/api/v1/models` 中报告安装状态，xTB 需安装可选依赖，Gaussian 需设置 `GAUSSIAN_EXE`。

任务 `succeeded` 不等于科学验证通过，请读取 `irc_validation_passed` 与 `scientific_outcome`。
无初猜自动搜索和 XYZ/SDF 上传路由仍未恢复。完整历史测试仍有收集错误；
恢复回归入口仅覆盖已恢复的子集。

统一安装说明见 [快速开始](../docs/quickstart.md)，算法边界见 [项目状态](../PROJECT_STATUS.md)。
历史恢复证据仅保存在本地 `archive/source-recovery/inventory/`，不随源码导出。
其中旧路径与运行记录不作为当前版本的全量功能承诺。
