# 计算引擎接入

后端通过 `model_name` 选择计算引擎，不把 UMA 写死在任务 Worker 或 TS 工作流里。

## 当前模型

| 模型 | 引擎 | 用途 | 状态 |
| --- | --- | --- | --- |
| `uma-s-1p2p1` | UMA | 主力单点、优化、频率、TS/IRC | 已可用 |
| `xtb-gfn2` | xTB GFN2 | 快速预优化、路径筛选和初猜 | 需要安装 `xtb` Python 包 |
| `gaussian-b3lyp-def2svp` | Gaussian | 外部高层级复核 | 需要 `GAUSSIAN_EXE` |

`GET /api/v1/models` 会同时返回引擎、方法、能力、checkpoint/可执行文件和不可用原因。
不可用的引擎在提交时返回 `409`，不会在 Worker 中生成一个假的 checkpoint 路径。

## 安装 xTB

项目依赖声明提供了可选组：

```sh
uv pip install --python .venv-uma/bin/python -e 'backend[xtb]'
```

也可以用 conda-forge 安装带原生库的 xTB Python 包：

```sh
conda install -c conda-forge xtb-python
```

当前工作环境未能从 PyPI 完成下载，因此安装完成前 `xtb-gfn2` 会显示为不可用。

## 计算边界

- UMA 继续使用现有 `UmaCalculatorFactory` 和 Hessian 实现。
- xTB 使用 `xtb.ase.calculator.XTB(method="GFN2-xTB")`，仅 CPU，解析 Hessian 时回退到有限差分。
- Gaussian 使用 ASE Gaussian 外部计算器，命令由 `GAUSSIAN_EXE` 指定；不应把它作为大分子 DMF/IRC 的默认路径引擎。
- 不同引擎的能量不能直接相减。活化能必须在同一引擎上重新优化反应物、TS 和产物，并在结果中记录引擎与方法。

## 后续扩展

分阶段 engine 配置可在请求中扩展为 `path`、`saddle`、`hessian` 和 `validation` 四个选择；当前 `model_name` 兼容字段仍然代表整条任务所用的模型。
