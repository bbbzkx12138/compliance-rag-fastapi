# 规镜 PolicyLens · FastAPI RAG

合规与制度检索增强生成项目。检索、证据判定、版本过滤、评测和文档持久化全部由 Python / FastAPI 完成，浏览器端只负责界面展示和 API 调用。

> 演示制度为虚构数据，不构成正式法律意见。

## 技术栈

- Python 3.12+
- FastAPI
- Uvicorn
- Pydantic v2
- SQLite
- sqlite-vec
- jieba
- BGE-small-zh-v1.5
- BGE Reranker Base
- 原生 HTML / CSS / JavaScript

项目使用本地 BGE 模型生成语义向量和 Cross-Encoder 重排结果，不调用外部大模型或 Embedding API。当前生成层仍使用可验证的抽取式回答。

## 功能

- jieba 中文分词、领域词典和规则查询改写。
- BM25、BGE 语义向量、TF-IDF 特征和 Cross-Encoder 混合检索。
- 向量持久化到 SQLite `sqlite-vec` 虚拟索引。
- 单文档嵌入缓存和跨查询 Cross-Encoder 重排缓存。
- 现行制度与历史版本过滤。
- 部门元数据过滤。
- 条款、页码、制度和版本级引用。
- 证据不足时的明确拒答。
- 54 题固定评测集，其中包含 10 个无依据负例。
- Markdown / TXT 文档导入。
- SQLite 唯一数据源、基础制度幂等同步和自动迁移。
- 索引版本管理、单文档刷新、全量重建和维护 CLI。
- JSON 日志、请求 ID、耗时和服务端统一错误响应。
- 模型状态、显存/模型体积、检索延迟和本地成本统计。
- Swagger 文档：`/api/docs`。
- Windows 一键启动脚本。

## 工程基础

第一阶段已经完成以下数据处理规则：

- `data/policies.json` 只作为基础制度种子，不再直接进入检索。
- 所有内置制度和 API 导入制度统一存入 SQLite。
- 启动时根据内容哈希同步基础制度，未变化的制度不会重复写入。
- 当前数据库结构版本为 `2`，旧版数据库会自动补齐字段。
- 索引由 `IndexManager` 统一管理，读请求和重建操作通过锁串行化。
- 新增或删除文档时只重新切分目标文档，再重算 IDF 和向量。
- 全量重建通过 `/api/index/rebuild` 或维护脚本执行。

## 检索质量升级

- `bge-small-zh-v1.5` 生成 512 维归一化语义向量。
- SQLite 关系表保存向量 BLOB，`sqlite-vec` 保存可检索的向量索引。
- `bge-reranker-base` 对混合召回的前 20 个条款进行 Cross-Encoder 重排。
- 查询改写负责处理“酒店费→住宿标准”“资料传到海外→数据出境”等表达差异。
- 长条款按标题、句子和 Markdown 段落切片，并保留页码与条款标题。

当前本地 CPU 基准：

| 检索模式 | 通过 | Recall@5 | 引用准确率 | 拒答准确率 | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| BM25 | 45/54 | 95.5% | 81.8% | 90% | 0.6 ms |
| BGE 向量 | 45/54 | 100% | 79.5% | 100% | 110 ms |
| BM25 + BGE + Reranker | 54/54 | 100% | 100% | 100% | 约 1.04 秒 |

模型体积约 1.15 GB，当前使用 CPU，显存占用为 0，单次检索 API 成本为 0 元。

## 启动

```powershell
cd C:\Users\Lenovo\Documents\Codex\2026-11-11\wo-x\outputs\compliance-rag-fastapi
python -m pip install -r requirements.txt
.\scripts\download-models.ps1
.\scripts\start.ps1
```

打开：

```text
应用：http://127.0.0.1:8000
Swagger：http://127.0.0.1:8000/api/docs
```

也可以直接运行：

```powershell
python -m app.run
```

复制 `.env.example` 为 `.env` 可修改端口、数据库路径、日志级别和 API 地址。当前只允许单进程索引，启动脚本会固定 `WEB_CONCURRENCY=1`。

## 测试

```powershell
python -W error::ResourceWarning -m unittest discover -s tests -v
```

当前覆盖旧数据库迁移、基础制度幂等同步、重启持久化、索引刷新、真实 Uvicorn HTTP 接口和检索质量。测试使用隔离 SQLite 数据库，不依赖 `pytest` 或 `httpx`。

## 维护命令

查看数据库和索引状态：

```powershell
python -m app.cli status
```

重建运行中服务的索引：

```powershell
.\scripts\rebuild-index.ps1
```

如果服务未运行，命令会执行离线索引校验，并提示服务下次启动时会自动重建进程内索引。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 服务与索引状态 |
| `GET` | `/api/documents` | 制度列表 |
| `GET` | `/api/documents/{id}` | 制度详情与全部条款 |
| `POST` | `/api/chat` | 合规问答与证据检索 |
| `POST` | `/api/evaluate` | 检索评测 |
| `POST` | `/api/documents` | 导入并持久化制度 |
| `DELETE` | `/api/documents/{id}` | 删除用户导入文档 |
| `GET` | `/api/index/status` | 数据库、索引版本与重建状态 |
| `POST` | `/api/index/rebuild` | 全量同步种子并重建索引 |

问答示例：

```powershell
$body = @{
  question = "发现疑似数据泄露后多久要报告？"
  scope = "current"
  department = "all"
  mode = "hybrid"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

导入文档示例：

```powershell
$body = @{
  title = "客户投诉处理办法"
  version = "V1.0"
  department = "客户服务部"
  effective_date = "2026-11-12"
  content = "# 第一条 受理时限`n客户投诉应在2个工作日内完成首次受理。"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/documents" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## 项目结构

```text
compliance-rag-fastapi/
├── app/
│   ├── main.py                 # FastAPI 路由、静态页面和异常处理
│   ├── config.py               # 路径与版本配置
│   ├── logging_config.py       # 文本 / JSON 日志配置
│   ├── cli.py                  # 数据库与索引维护命令
│   ├── run.py                  # 单进程服务启动入口
│   ├── schemas.py              # Pydantic 请求模型
│   └── services/
│       ├── chunking.py         # 长条款与表格感知切片
│       ├── index_manager.py    # 索引锁、版本与增量刷新
│       ├── semantic.py         # BGE 向量与 Cross-Encoder
│       ├── retrieval.py        # BM25、向量、回答与评测
│       └── repository.py       # SQLite 数据源、迁移和种子同步
├── data/
│   └── policies.json           # 制度、同义词与评测集
├── static/
│   ├── index.html
│   ├── styles.css
│   └── app.js                  # 只调用 FastAPI，不包含检索算法
├── storage/
│   └── policy_rag.sqlite3      # 首次启动后自动创建
├── scripts/
│   ├── start.ps1
│   ├── download-models.ps1
│   └── rebuild-index.ps1
├── tests/
│   ├── test_retrieval.py
│   ├── test_repository.py
│   └── test_api.py
├── .env.example
└── requirements.txt
```

## RAG 链路

```text
Browser
  → POST /api/chat
  → FastAPI
  → IndexManager 读锁
  → 查询清洗与同义词扩展
  → 版本 / 部门元数据过滤
  → BM25 + jieba + BGE 语义召回
  → sqlite-vec 向量近邻检索
  → Cross-Encoder 重排
  → 字段加权与证据阈值判定
  → 条款引用或拒答
  → JSON 返回浏览器
```

## 当前边界

- 当前生成层是抽取式回答，不调用 LLM。
- 当前 SQLite + sqlite-vec 用于单机部署；更大规模可迁移 PostgreSQL / pgvector。
- 索引仍在单个 Python 进程内，禁止直接使用多 worker 部署。
- 尚未实现登录、RBAC、文档级 ACL、限流与审计日志。

这些边界可以在后续升级为 LLM 引用生成、共享向量索引和企业身份认证。
