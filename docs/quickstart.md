# 安装与启动

从仓库根目录执行。需要 Python 3.11+、uv；前端另需支持 Vite 8 的 Node.js（20.19+ 或 22.12+）及 npm。
下面是依据当前入口整理的安装步骤，本次未在干净环境复跑。

## 后端

```sh
uv venv --python 3.13 .venv-uma
scripts/install-backend-env.sh
cp backend/.env.example backend/.env
```

按上游模型授权自行获取 UMA 权重，放入 `model-assets/UMA/checkpoints/`。
默认推荐文件名为 `uma-s-1p2p1.pt`；元素参考数据保留在 `model-assets/UMA/references/`。
其他路径、设备和队列配置参见 `backend/.env.example` 与 `backend/app/core/config.py`。
模型权重不随本仓库分发。

已有初猜的精修需要鞍点和频率相关依赖：

```sh
uv pip install --python .venv-uma/bin/python -e 'packages/uma-workflows[sella,frequencies]'
scripts/run-backend.sh
```

后端监听 `http://127.0.0.1:8010`，接口文档为 `/docs`；模型首次计算时加载。
`GET /api/v1/models` 返回各模型的可用状态和能力。未安装模型不妨碍浏览接口文档。

可选 xTB：

```sh
uv pip install --python .venv-uma/bin/python -e 'backend[xtb]'
```

xTB 原生库与 Python 包需匹配，具体安装说明见 [计算引擎文档](calculator-providers/README.md)。
Gaussian 需要用户自行安装并设置 `GAUSSIAN_EXE`；ORCA 尚无通用后端入口。

## 最小 API 输入

服务启动且 UMA 模型可用后，提交通用水分子单点示例：

```sh
curl --fail-with-body http://127.0.0.1:8010/api/v1/jobs \
  -H 'Content-Type: application/json' \
  --data-binary @examples/water-single-point.json
```

保存响应中的 `id`，通过 `GET /api/v1/jobs/<id>` 查询任务。
这一步只演示 API 和单点计算；TS 输入及结果判定见 [工作流文档](transition-state-workflow.md)。

## 本地前端（暂不随源码包发布）

另开终端：

```sh
cd frontend
npm ci
npm run build --workspace @retainmol/mol-viewer
npm run dev
```

访问 `http://127.0.0.1:5173`，`/api` 请求会代理到后端。前端当前只作为本地开发工作台保留，
不属于首个 GitHub 源码包；本次尚未完成干净环境构建与浏览器核验。
默认入口是 LiteWorkbench；完整旧工作台仍在源码中，本次未核验构建与浏览器交互。

## 自动路径的额外依赖

仅在使用算法库的 FB-ENM + Direct MaxFlux 分支时安装：

```sh
uv pip install --python .venv-uma/bin/python -e 'packages/uma-workflows[dmf,sella,uma,frequencies]'
```

`cyipopt` 需要系统 Ipopt 开发库和 pkg-config，需先按操作系统安装，不能假定纯 pip 即可完成。
HTTP 无初猜搜索仍受限；不要把安装 extras 当成 API 功能已经接通。
