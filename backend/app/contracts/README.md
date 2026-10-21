# Contracts

这里保存后端各层共享的稳定数据契约，包括请求、结果、任务状态、事件和制品索引。

约束：

- 只依赖 Pydantic、标准库和同目录中的契约。
- 不导入 FastAPI、SQLite、文件系统或具体计算执行器。
- API、应用服务和基础设施可以依赖本目录，反向依赖不允许。
- `app.schemas` 暂时是本目录的兼容入口，新代码应直接使用 `app.contracts`。
- `actions.py` 使用 discriminated union 定义 AI 可调用的白名单 Action；搜索参数
  与服务器派生的科学验证结论必须分离。

