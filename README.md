# 规镜 PolicyLens · FastAPI RAG

合规与制度检索增强生成项目。检索、证据判定、版本过滤、评测和文档持久化全部由 Python / FastAPI 完成，浏览器端只负责界面展示和 API 调用。

> 演示制度为虚构数据，不构成正式法律意见。

## 技术栈

- Python 3.12+
- FastAPI
- Uvicorn
- Pydantic v2
- SQLite
- 原生 HTML / CSS / JavaScript

项目没有调用外部大模型或 Embedding API。当前生成层使用可验证的抽取式回答，确保离线运行时每条结论都来自制度原文。

## 功能

- BM25、字词 TF-IDF 向量和字段重排的混合检索。
- 现行制度与历史版本过滤。
- 部门元数据过滤。
- 条款、页码、制度和版本级引用。
- 证据不足时的明确拒答。
- 12 题固定评测集，对比三种检索模式。
- Markdown / TXT 文档导入。
- SQLite 持久化与自动索引重建。
- Swagger 文档：`/api/docs`。
- Windows 页面、API 和检索单测。

## 启动

```powershell
cd C:\Users\Lenovo\Documents\Codex\2026-11-11\wo-x\outputs\compliance-rag-fastapi
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开：

```text
应用：http://127.0.0.1:8000
Swagger：http://127.0.0.1:8000/api/docs
```

开发模式：

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试使用真实 Uvicorn 临时端口和隔离 SQLite 数据库，不依赖 `pytest` 或 `httpx`。

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
│   ├── schemas.py              # Pydantic 请求模型
│   └── services/
│       ├── retrieval.py        # BM25、向量、回答与评测
│       └── repository.py       # SQLite 文档持久化
├── data/
│   └── policies.json           # 制度、同义词与评测集
├── static/
│   ├── index.html
│   ├── styles.css
│   └── app.js                  # 只调用 FastAPI，不包含检索算法
├── storage/
│   └── policy_rag.sqlite3      # 首次启动后自动创建
├── tests/
│   ├── test_retrieval.py
│   └── test_api.py
└── requirements.txt
```

## RAG 链路

```text
Browser
  → POST /api/chat
  → FastAPI
  → 查询清洗与同义词扩展
  → 版本 / 部门元数据过滤
  → BM25 + TF-IDF 向量召回
  → 字段重排与证据阈值判定
  → 条款引用或拒答
  → JSON 返回浏览器
```

## 当前边界

- 当前向量是字词 TF-IDF，不是神经网络 Embedding。
- 当前生成层是抽取式回答，不调用 LLM。
- 当前 SQLite 用于单机演示，未接入 PostgreSQL / pgvector。
- 尚未实现登录、RBAC、文档级 ACL、限流与审计日志。

这些边界可以在后续升级为 BGE Embedding、Cross-Encoder 重排、LLM 引用生成和企业身份认证。
