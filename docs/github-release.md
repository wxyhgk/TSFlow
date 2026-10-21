# GitHub 发布准备

## Git 现状

本地此前已有 `.git` 与开发提交，本次 `git init` 保留了原历史。
当前没有配置远程仓库；本次未提交或推送。既有工作区改动保留。

原历史含恢复清单和旧材料。新增 `.gitignore` 不会移除历史提交中的文件。
对外首次发布建议从当前源码导出独立目录，在该目录建立干净的发布仓库；原项目继续保留研究历史。

## 导出当前源码

```sh
python3 scripts/export-source.py --list
python3 scripts/export-source.py --output outputs/github-source-20261021
```

导出包含当前工作区的后端与算法源码（包括未提交的新模块），不依赖最后一次提交是否完整。
使用明确路径清单，并再次应用 Git 忽略规则。不会携带 `.git`、权重、虚拟环境、node_modules、
研究结构、`calculations/` 算力记录、研究文档、计算结果、PPT、恢复档案或 `frontend/`；案例专用的 `calculations/scripts/` 也不进入发布包。
已有目标目录会拒绝覆盖，超过 5 MiB 的单文件与符号链接需先审查。
`SOURCE_MANIFEST.json` 记录文件清单、SHA-256、来源 HEAD 和是否含本地改动。
导出工具只控制文件范围，不代替人工内容和第三方来源审查。

## 当前明确排除的目录

| 目录 | 原因 |
| --- | --- |
| `frontend/` | 本地工作台尚未完成独立的干净环境构建和浏览器验证 |
| `calculations/scripts/` | 分子案例专用脚本、扫描脚本、动画脚本和一次性诊断工具 |
| `data/` | 任务数据库、运行附件和研究路径 |
| `calculations/` | 算力配置、日志、能量、Hessian、IRC 和研究报告 |
| `mol/` | 未公开分子结构与输入文件 |
| `outputs/` | PPT、HTML、图片和发布构建缓存 |
| `archive/` | 历史源码恢复材料和机器记录 |
| `.venv-uma/`、`model-assets/UMA/checkpoints/` | 本地运行环境和模型权重 |

## 发布仓库

PyPI 发布工作流位于 `.github/workflows/publish-pypi.yml`，使用 GitHub
Trusted Publishing，不把 API token 写入仓库。首次发布前，在 PyPI 账户中新增
Trusted Publisher：

| 字段 | 值 |
| --- | --- |
| Owner | `wxyhgk` |
| Repository | `TSFlow` |
| Workflow name | `publish-pypi.yml` |
| Environment | `pypi` |

配置完成后，创建版本标签即可触发发布：

```sh
git tag v0.1.4
git push origin v0.1.4
```

也可以在 GitHub Actions 页面手动运行 `Publish ts-flow to PyPI`。

对导出目录完成源码与许可证审查后，在导出目录执行：

```sh
git init -b main
git status --short
git add .
git commit -m "Initial transition-state workflow source"
```

随后创建用户选定名称、可见性和归属的 GitHub 仓库，再按 GitHub 给出的地址添加 `origin`、推送 `main`。
本地研究仓库不直接推送到公开仓库，避免恢复历史和未公开研究资料随历史发布。

## 首次发布仍需完成

1. 已选定源码许可证为 MIT；发布前仍需确认复制引入的查看器及其他第三方源码的授权与署名要求。
   上游 UMA 的许可证与使用政策仅覆盖相应模型资产，不是整个项目的许可证。
2. 在干净环境执行安装与一个完整小分子 TS 案例，公布输入、原子映射、参数、依赖版本、Hessian 与 IRC 结果。
3. 核验前端与后端实际交互，修复历史遗留的导入和契约不一致。
4. 单独完成前端构建、浏览器交互和 fast-mol 许可审查后，再决定是否加入后续发布；当前发布时明确标为研究开发版本，自动无初猜 HTTP 搜索和 AI API 规划层不能宣称已可用。

这些是发布质量的待办，不影响本地继续开发。此次目录整理未运行测试或重新开展计算。
