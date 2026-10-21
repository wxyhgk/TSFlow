# 验证工具

这里的脚本用于人工复跑和核对，不属于后端服务运行链：

- `check_uma_inference.py`：直接检查 UMA 模型推理。
- `benchmark_uma_cache.py`：连续提交两个小分子单点任务，比较首次加载和缓存复用的墙钟时间。

这些命令可能产生计算结果或新任务。测试后请检查 `data/` 和数据库变化。

直接推理检查的结果写入 `data/validation/`。
依赖历史 BH22 输入的 `check_uma_ts_irc.py` 和 `check_backend_ts_irc.py` 已归档到
本地 `archive/source-recovery/tools/`，不属于公开源码示例。

性能基准要求后端以 `UMA_WORKER_ENABLED=true` 运行：

```sh
.venv-uma/bin/python tools/benchmark_uma_cache.py
```
