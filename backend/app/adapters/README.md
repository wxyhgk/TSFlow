# Adapters

这里连接后端数据契约与外部计算库，包括 UMA、ASE、geomeTRIC 和 `uma_workflows`。

适配器负责类型转换和调用边界；可复用数值算法应位于 `packages/uma-workflows`，而不是复制到后端。
UMA 专用的 geomeTRIC/ASE 执行桥位于 `adapters/uma/geometric.py`；旧 `services.geometric_optimizer` 仅保留兼容导出。

