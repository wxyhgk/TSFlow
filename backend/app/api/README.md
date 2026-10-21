# API

FastAPI 传输层。负责 HTTP 参数、状态码、响应模型和流式连接。

路由应调用 application 用例，不直接实现计算算法、SQLite 查询或 artifact 组装。大型资源按 commands、queries、events、uploads 和 downloads 拆分。

`actions.py` 是 AI/自动化客户端的兼容入口。它只公布已实现 capability，并把
首版 Action 映射到现有 Job；后续独立计算原语应进入 application 层，而不是在
HTTP 路由中编排科学算法。

路由由 `app.api.routes` 包通过 `create_router()` 组合。每个功能模块以
`RouteDefinition` 注册，应用入口只依赖组合结果；替换模块时传入新的定义集合，
不需要修改 `app/main.py`。`app.api.router.api_router` 仍保留为兼容导出。

