# 目录管理

## 本次整理（2026-10-21）

| 原路径 | 新路径 |
| --- | --- |
| `inventory/` | `archive/source-recovery/inventory/` |
| `reconstructed/` | `archive/source-recovery/reconstructed/` |
| `full_file_reads/` | `archive/source-recovery/full_file_reads/` |
| `recovery/` | `archive/source-recovery/recovery/` |
| `tools/check_uma_ts_irc.py` | `archive/source-recovery/tools/check_uma_ts_irc.py` |
| `tools/check_backend_ts_irc.py` | `archive/source-recovery/tools/check_backend_ts_irc.py` |
| 根目录 `.chart-data-*` | `.ppt-build/chart-data/.chart-data-*` |
| `docs/<case-report>/` | `calculations/<case-report>/` |

两个旧 BH22 检查脚本依赖恢复目录内的本地输入，因此一同归档；已更新源码根目录与输入定位。
今后的人工计算检查输出写入 `data/validation/`，不再混入历史 inventory。
归档内的历史报告、来源路径和哈希保留原记录；其中旧目录导航可能只适用于当时快照。
GD451、MR 两个位点、Li/O 电子结构和 fast-mol 案例已按此规则从 `docs/` 移至 `calculations/`。

## 后续文件放置

- 通用算法只放 `packages/uma-workflows/src/uma_workflows/`。
- HTTP 路由放 `backend/app/api/routes/`；计算引擎放 `backend/app/adapters/` 和 `backend/app/calculators/`。
- 安装和启动入口放 `scripts/`，独立人工诊断放 `tools/`。
- `calculations/scripts/` 是本地案例研究、扫描、建模和动画脚本；不进入公开源码包。
- `backend/app/services/` 是兼容过渡层；新的任务编排放 `application/`，外部计算放 `adapters/`，格式处理放 `formats/`。
- 可分享的通用示例放 `examples/`，不得引用本地绝对路径或研究案例。
- 现有实验结构继续放 `mol/`；任务文件放 `data/`；外部量化计算放 `calculations/`。
- PPT、HTML 动画、图片等交付物放 `outputs/<主题-日期>/`。
- 公共使用文档放 `docs/`；具体算力、运行记录、分子案例和研究报告放 `calculations/<case-name>/`，不进入发布包。

不再把临时脚本、计算输出或演示文稿构建文件放到根目录。Git 忽略仅影响后续添加，不能删除已有提交里的历史文件；发布方式见 [发布文档](github-release.md)。
