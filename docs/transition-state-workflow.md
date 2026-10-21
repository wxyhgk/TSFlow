# 过渡态工作流

## 输入定义

输入必须包含完整的反应物、产物三维结构，总电荷、自旋多重度，以及成键/断键/配位原子对。
两端及可选 TS 初猜应具有一致的原子数、元素顺序、电荷和自旋，包含全部参与反应的碎片。
**算法与 HTTP 契约中的原子索引从 0 开始**，与从 1 开始的显示编号转换时需减 1。

目前默认前端使用已有 TS 初猜的入口。只有“分子 + 反应位点”尚不足以构成现有后端的完整 TS 输入；
产物构建、原子映射和 AI 规划仍需继续产品化。

## 两条算法路径

```text
已有 TS 初猜：R / P / TS guess → 鞍点精修 → 新 Hessian → 双向 IRC → 端点验证
无 TS 初猜： R / P / 位点 → 端点优化 → FB-ENM → Direct MaxFlux → 鞍点精修 → 同上
```

第二条的算法模块已存在于独立库，后端尚未开放其通用 HTTP 执行入口。
NEB 并非当前宏工作流的默认路径方法。

## 独立 Python API

以下是接入模板；`reactant`、`product`、`guess` 是调用方读取并完成原子映射的 ASE Atoms，
`calculator` 是调用方配置的 ASE 计算器。模板不包含研究分子和预先验证过的小反应数据。

```python
from dataclasses import dataclass
from threading import Event

from uma_workflows.transition_states.workflow import (
    IrcConfig, ScreeningConfig, TransitionSearchConfig,
    run_transition_state_search,
)

@dataclass(frozen=True)
class Coordinate:
    atom_i: int
    atom_j: int
    kind: str

# 示例编号必须替换成实际映射；不是任何特定反应的位点定义。
coordinates = (
    Coordinate(0, 1, "breaking"),
    Coordinate(0, 2, "forming"),
)
config = TransitionSearchConfig(
    screening=ScreeningConfig(force_full_hessian=True),
    irc=IrcConfig(enabled=True),
)
result = run_transition_state_search(
    reactant, product,
    transition_state_guess=guess,
    reaction_coordinates=coordinates,
    config=config, device="cpu",
    calculator=calculator, cpu_calculator=calculator,
    cancel_event=Event(),
)
print(result.outcome)
print(result.warnings)
```

无初猜时省略 `transition_state_guess`，并安装 DMF 等额外依赖。
实际全体系 Hessian 与 IRC 成本取决于分子、引擎和收敛要求；保存中间结果，避免反复从头计算。

## HTTP 入口

通过 `POST /api/v1/jobs` 提交 `calculation: "transition_state_search"`，包括
`structure`、`product_structure`、`transition_state_guess`、`reaction_coordinates`、`parameters`。
使用 `parameters.irc_follow: true` 请求 IRC，完整字段与参数范围以运行服务的 `/docs` 为准。
基础请求格式可参照 [水分子示例](../examples/water-single-point.json)，不要将它直接用作 TS 输入。

## 结果判定

- `succeeded`：任务执行完毕，不等于反应机理已验证。
- 全 Hessian 与驻点筛选通过：可作为一阶鞍点候选；部分 Hessian 仅作初筛。
- 双向 IRC 连接预期两端：才支持目标反应连接关系；步数耗尽或端点不符需继续处理。
- 算法库 `outcome` 与 API `scientific_outcome`/`irc_validation_passed`：用于区分候选、已筛选和已验证结果。

能垒报告需记录理论方法、反应物参考态和能量类型。电子能差不等于含溶剂、温度与熵的实验活化自由能。
