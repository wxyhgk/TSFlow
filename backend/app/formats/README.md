# Formats

这里集中处理外部化学文件格式：XYZ、SDF、Gaussian 输入及轨迹文本。

格式模块只负责解析、验证和序列化，不负责提交任务或运行模型。

当前实现：`xyz.py`、`sdf.py`、`gaussian.py`。旧的 `app.services.*` 路径只保留兼容转发。

