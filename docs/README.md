# 文档导航

- [安装与启动](quickstart.md)
- [过渡态工作流与验证](transition-state-workflow.md)
- [架构与扩展位置](architecture.md)
- [目录管理与本次搬移](project-layout.md)
- [GitHub 发布准备](github-release.md)
- [计算引擎接入](calculator-providers/README.md)

具体算力、运行日志、分子案例、能量、Hessian、IRC 和研究报告统一放在根目录 `calculations/`，该目录由 Git 忽略，不进入源码发布包。
`docs/` 只保留可公开复用的安装、接口、算法、架构和发布说明。
当前 GitHub 首个源码包暂不包含 `frontend/`；前端保留在本地，用于后续工作台和 fast-mol 验证。
