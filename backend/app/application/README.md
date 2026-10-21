# Application

这里保存用例编排：任务创建、状态转换、Worker 调度以及计算流程分派。

本层负责“何时执行什么”，不实现具体量化化学算法，也不直接写 SQL 或文件。它通过协议调用基础设施，通过 adapters 调用 UMA/过渡态计算。

当前 `jobs/` 已包含本地线程 Worker、可回收子进程 Worker 及其消息协议。

`transition_states/` 是 TS 任务的正式用例入口。它只负责把 API 请求转换为
`uma_workflows.transition_states.workflow` 的调用，并把进度和结果转换回后端契约；
路径搜索、鞍点精修、Hessian 和 IRC 的科学实现仍属于共享算法包。

