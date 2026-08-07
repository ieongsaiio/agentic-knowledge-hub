# Agentic Knowledge Hub

一个配置驱动、可插拔、可观测的 RAG Knowledge Hub。项目提供完整的文档摄取、Hybrid Search、Rerank、评估与 MCP Server，允许 Copilot、Claude Desktop 或其他 MCP Client 将本地知识库作为标准工具调用。

当前版本聚焦于可靠的 **RAG 检索基础设施**：MCP 工具返回带来源、页码和 Chunk 标识的检索证据，最终答案由上游 MCP Host/Agent 生成。Centralized Multi-Agent Orchestration 是下一阶段的演进方向，而不是当前版本已经完成的能力。

## 设计理念

项目架构被落实为可以运行、测试和替换的工程模块，通过真实数据摄取、查询追踪和基准评估来理解：

- PDF 解析、页码映射、Token/Character Chunking 与 Chunk Refinement
- Dense Retrieval、BM25、Weighted RRF 与 Metadata Filter
- Bi-Encoder Embedding、Cross-Encoder/LLM Rerank 与两阶段检索
- Provider、Factory、Settings 和配置驱动的可插拔架构
- MCP 生命周期、Tool Calling、结构化引用与多模态 Content Block
- FinanceBench、Ragas、自定义 Retrieval Metric 与 LLM Evidence Judge

项目保留清晰的抽象边界和完整 Trace，让学习不只停留在框架 API，也能追踪每个阶段的输入、输出、耗时与失败回退。

## 核心能力

- **可观测摄取链路**：PDF -> Markdown -> 页码区间映射 -> Recursive Split -> Rule/LLM Refine -> Metadata Enrich -> Embedding -> Chroma/BM25 Upsert。
- **Token 或字符切块**：Recursive Splitter 支持 `characters` 与 Hugging Face tokenizer 驱动的 `tokens` 长度单位，并保留源文本 offsets。
- **Hybrid Search**：Dense Embedding 与 BM25 双路召回，通过可配置权重的 RRF 融合候选结果。
- **可插拔 Reranker**：支持本地 Cross-Encoder、API Cross-Encoder、LLM Reranker，以及失败时回退到融合结果。
- **多模态 PDF**：Vision LLM 开启时提取 PDF 图片、生成 Caption，并将图像语义接入文本检索；关闭时跳过图片提取与索引。
- **配置驱动 Provider**：LLM、Vision LLM、Embedding、Splitter、Reranker、Vector Store、Evaluator 与 Benchmark 都通过 Settings 和 Factory 解耦。
- **MCP Server**：使用官方 Python MCP SDK 和 stdio transport，向任意兼容 MCP 的 Agent 暴露知识库工具。
- **Trace Dashboard**：通过 Streamlit 查看摄取、查询、召回、重排、评估指标及各阶段延迟。
- **评估与消融实验**：支持多组 Settings Override、索引 Fingerprint/复用、断点恢复、FinanceBench Retrieval Evaluation 与结果导出。
- **数据生命周期**：基于文件 Hash 与 Collection 的幂等摄取，支持文档列表、删除、强制重建和运行数据清理。

## 系统流程

```text
Ingestion
PDF
  -> Loader Factory
       -> default: MarkItDown 文本解析 + PyMuPDF 页码/图片处理
       -> paddle/docker: PaddleOCR-VL Transformers + 页面重组
       -> paddle/api: PaddleOCR Studio 异步 Job API + 逐页 Markdown
  -> Parsed Document Cache
  -> Recursive Splitter (characters | tokens)
  -> Chunk Refiner / Metadata Enricher / Image Captioner
  -> Batch Embedding
  -> Chroma Dense Index + BM25 Sparse Index + Ingestion History

Query
Question
  -> Query Processing
  -> Dense Retrieval + BM25 Retrieval
  -> Weighted RRF Fusion
  -> Optional Reranker
  -> Evidence Snippets + Citations + Optional Images
  -> CLI / Dashboard / MCP Client
```

## 项目结构

```text
.
|-- config/
|   |-- prompts/                  # Rerank、评估与 Evidence Judge Prompt
|   `-- settings.yaml.example     # 无凭据的完整配置结构
|-- scripts/
|   |-- ingest.py                 # 文件/目录摄取入口
|   |-- query.py                  # RAG 查询入口
|   |-- prepare_benchmark.py      # Benchmark 下载与样本准备
|   |-- evaluate.py               # 评估、消融实验、索引复用与断点恢复
|   |-- export_evaluation_history.py
|   |-- clear_data.py             # 清理 Storage、Parsed Cache、Evaluation 和 Logs
|   `-- start_dashboard.py        # Streamlit Dashboard 启动器
|-- docker/
|   `-- paddleocr-transformers/   # PaddleOCR-VL Transformers 隔离运行镜像
|-- src/
|   |-- core/                     # Settings、共享类型、Query Engine、Response、Trace
|   |-- ingestion/                # Pipeline、Chunking、Transform、Embedding、Storage
|   |-- libs/                     # Provider 接口、Factory 与具体实现
|   |-- mcp_server/               # MCP 协议处理、stdio Server 与 Tools
|   `-- observability/            # Dashboard、日志和评估系统
|-- tests/
|   |-- unit/                     # 组件与配置契约测试
|   |-- integration/              # 跨模块检索测试
|   |-- e2e/                      # MCP SDK 与原始 JSON-RPC 测试
|   `-- fixtures/                 # 小型测试文档与图片
|-- data/                         # 本地运行数据，Git 忽略
`-- pyproject.toml
```

## 可插拔组件

| 组件 | 当前实现 |
|---|---|
| LLM | OpenAI、Azure OpenAI、DeepSeek、Ollama |
| Vision LLM | OpenAI-compatible、Azure OpenAI |
| Embedding | OpenAI、Azure OpenAI、Ollama、SiliconFlow |
| Splitter | Recursive Splitter（characters/tokens） |
| Reranker | None、Local Cross-Encoder、Cross-Encoder API、LLM |
| Vector Store | ChromaDB |
| Sparse Retrieval | BM25 |
| Evaluator | Custom、Ragas、FinanceBench Benchmark |

新增 Provider 的基本方式是实现对应 `Base*` 接口，然后注册到 Factory；Pipeline 和调用方不需要依赖具体供应商。

## PDF Loader 与 Parsed Cache

`ingestion.loader.provider` 支持 `default` 与 `paddle`。`default` 保留原有
MarkItDown + PyMuPDF 流程；`paddle` 可选择本地 Docker Transformers 或
PaddleOCR Studio API，适合表格、标题层级和复杂版面较多的 PDF。

```yaml
ingestion:
  loader:
    provider: "paddle"
    parsed_dir: "./data/parsed"
    paddle:
      backend: "api"  # docker 或 api
      docker:
        engine: "transformers"
        pipeline_version: "v1.6"
        docker_image: "agentic-knowledge-hub/paddleocr-vl-transformers:latest"
        docker_cache_volume: "paddleocr-transformers-cache"
        paddlex_cache_volume: "paddleocr-paddlex-cache"
        device: "cpu"
        shm_size: "4g"
        merge_tables: false
        relevel_titles: true
        concatenate_pages: false
        use_queues: false
        timeout_seconds: 7200
      api:
        job_url: "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
        token_env: "PADDLEOCR_API_TOKEN"
        model: "PaddleOCR-VL-1.6"
        poll_interval_seconds: 5
        timeout_seconds: 1800
        request_timeout_seconds: 120
        optional_payload:
          useDocOrientationClassify: false
          useDocUnwarping: false
          useChartRecognition: false
          restructurePages: true
          mergeTables: false
          relevelTitles: true
          concatenatePages: false
          returnMarkdownImages: false
```

API token 只通过环境变量注入，不应写入 YAML：

```powershell
$env:PADDLEOCR_API_TOKEN="<your-token>"
```

切换 `backend: "docker"` 后，先构建本地镜像：

```powershell
docker build -t agentic-knowledge-hub/paddleocr-vl-transformers:latest `
  -f docker/paddleocr-transformers/Dockerfile .
```

解析结果保存在 `data/parsed/`。缓存文件名由 PDF SHA-256 和 Loader 配置
Hash 组成；改变 Backend、模型、输出相关参数或图片策略会使用另一份缓存。
Token、轮询间隔和网络 timeout 不进入缓存 key。缓存 JSON 同时保存完整
Markdown、物理页字符区间和 Paddle 原始/重组 JSON。Chunk 写入 Chroma 前
会移除这些大型中间字段，只保留来源、offset 与
`page_start/page_end/page_num`。

项目固定使用 `merge_tables=false` / `mergeTables=false`，不合并跨页表格，
也不推断续表关系或继承上一页表格信息。每个 API 结果页独立保留自己的
Markdown 和物理页字符区间，以页面顺序与引用准确性为优先。

`vision_llm.enabled=false` 时，两个 Loader 都不会提取或索引图片，Paddle 也会忽略 image label，不产生图片占位符。开启后，Paddle 导出图片并在 Markdown 中写入 `[IMAGE: image_id]`，后续 Captioner 再处理对应图片。

真实 Loader 验证命令：

```powershell
python scripts/validate_paddle_loader.py path/to/document.pdf
```

只清理 parsed cache：

```powershell
python scripts/clear_data.py --parsed --yes
```

真实 API 单页同步/异步测试：

```powershell
python -m pytest tests/integration/test_paddleocr_api_integration.py -q
```

## Table Summary 补充向量

结构化切分可以为每张完整 Table Parent 生成一条额外的 LLM Summary Vector。原始
Table Child 不会被摘要替换；Summary 命中时，Chroma 返回完整 Parent Table 内容，
摘要只作为 Dense Embedding 输入。Summary 记录不写入 BM25，避免同一表格被重复计算
稀疏词频。

```yaml
ingestion:
  structured_chunking:
    table_summary:
      enabled: false
      prompt_path: "./config/prompts/table_summary.txt"
      prompt_version: "v1"
      max_workers: 5
      fail_on_error: false
      llm:
        provider: "openai"
        model: "gpt-4o-mini"
        temperature: 0.0
        max_tokens: 512
        extra_chat_configs: {}
```

嵌套 `llm` 只覆盖 Summary 专用模型需要改变的字段；未提供的 API Key、Base URL、
Deployment 等字段会复用顶层 `llm` 配置。需要完全独立的服务时，可在该映射内填写
完整 LLM Provider 字段。

`enabled=true` 时，每张完整 Parent Table 调用一次 Summary LLM，而不是每个 Table
Child 调用一次。调用在单份文档内按 `max_workers` 并发执行。默认
`fail_on_error=false`，单张表摘要失败时只跳过该补充向量，原始 Table Child 仍正常
入库；设为 `true` 可让任何摘要失败终止该文档摄取。

Summary 记录的关键契约：

```text
Chroma document       = 完整原始 Parent Table（用于返回与引用）
Dense embedding input = LLM Table Summary
BM25                  = 不写入
chunk_role            = table_summary
parent_chunk_id       = 对应的完整 Table Parent ID
```

修改 Summary Model、Prompt Version 或相关配置会改变 Benchmark Index Fingerprint，
避免错误复用旧 Collection。

## 快速开始

### 1. 安装环境

```powershell
conda create -n mini-agent python=3.11 -y
conda activate mini-agent
pip install -e ".[dev]"
```

`mini-agent` 是 Conda 环境名，不绑定项目目录，因此可以在任意路径执行
`conda activate mini-agent` 或 `conda run -n mini-agent ...`。不过，本项目默认配置、
数据目录和日志目录使用相对路径；从其他目录启动时，应通过绝对路径运行脚本并显式
指定配置文件，或像 MCP Client 配置一样将 `cwd` 设置为本项目根目录：

```powershell
conda run -n mini-agent python C:\path\to\agentic-knowledge-hub\scripts\query.py `
  --config C:\path\to\agentic-knowledge-hub\config\settings.yaml `
  --query "What does the document say about revenue?" `
  --collection knowledge_hub
```

如果执行过 `pip install -e ".[dev]"`，Python Package 会以 Editable Mode 安装；
但为了让相对配置和持久化目录保持一致，启动 MCP Server 时仍建议保留 `cwd`。

### 2. 准备配置

```powershell
Copy-Item config/settings.yaml.example config/settings.yaml
```

填写 `config/settings.yaml` 中所选 Provider 的模型、维度、URL、布尔值和数值参数。真实配置和凭据不会被 Git 跟踪；API Key 也可以通过环境变量提供。

OpenAI 文本 Provider 可通过 `llm.api_mode` 选择 API：

```yaml
llm:
  provider: "openai"
  api_mode: "chat_completions"  # chat_completions 或 responses
```

`chat_completions` 调用 `/chat/completions`，也是默认值，适合现有 OpenAI-compatible 服务；`responses` 调用 `/responses`。第三方服务是否支持 Responses API 取决于该服务自身的实现。

注意：同一 Chroma Collection 的向量维度必须与当前 Embedding 配置一致。更换 Embedding 模型或维度时，应使用新 Collection 或重建旧索引。

### 3. 摄取文档

```powershell
python scripts/ingest.py --path sample_documents/report.pdf --collection knowledge_hub
```

摄取整个目录：

```powershell
python scripts/ingest.py --path sample_documents --collection knowledge_hub
```

使用 `--force` 可删除该文档在目标 Collection 中的旧记录并重新摄取。

### 4. 查询

```powershell
python scripts/query.py --query "What does the document say about revenue?" --collection knowledge_hub --verbose
```

临时关闭 Reranker：

```powershell
python scripts/query.py --query "What does the document say about revenue?" --collection knowledge_hub --no-rerank
```

### 5. Dashboard

```powershell
python scripts/start_dashboard.py --port 8501
```

浏览器访问 `http://localhost:8501`。

## MCP 接入

启动 MCP Server：

```powershell
python -m src.mcp_server.server
```

MCP Client 配置示例：

```json
{
  "mcpServers": {
    "agentic-knowledge-hub": {
      "command": "conda",
      "args": [
        "run",
        "-n",
        "mini-agent",
        "python",
        "-m",
        "src.mcp_server.server"
      ],
      "cwd": "C:\\path\\to\\agentic-knowledge-hub"
    }
  }
}
```

当前暴露四个工具：

| Tool | 作用 |
|---|---|
| `query_knowledge_hub` | 执行 Hybrid Search 和可选 Rerank，返回完整 Chunk Text、来源、分数与可选图片 |
| `list_collections` | 返回 Collection、文档数量和 Chunk 数量 |
| `list_documents` | 返回指定 Collection 内的文档目录和 `doc_id` |
| `get_document_summary` | 根据 `doc_id` 获取文档预览与元数据 |

MCP Server 只提供知识检索工具，不直接生成最终答案。Copilot、Claude Desktop 或上层 Agent 会把 Tool Result 作为上下文，再生成面向用户的回答。

`query_knowledge_hub` 默认使用 `response_format="xml"`。也可以选择 `json` 或
`markdown`；该参数只控制 `TextContent` 的渲染格式。无论选择哪种格式，MCP
`structuredContent` 都会返回同一份完整 JSON Payload，且所有格式都保留完整 Chunk
Text，不进行 Snippet 截断：

```json
{
  "query": "What was the reported revenue?",
  "collection": "financebench",
  "top_k": 5,
  "response_format": "xml"
}
```

### `query_knowledge_hub` 返回格式

MCP Client 始终接收完整的 `CallToolResult`。`response_format` 只改变
`content` 中的文本表现形式，不会关闭或改变 `structuredContent`：

```json
{
  "content": [
    {
      "type": "text",
      "text": "<retrieval_results>...</retrieval_results>"
    }
  ],
  "structuredContent": {
    "query": "What was the reported revenue?",
    "collection": "financebench",
    "result_count": 1,
    "results": [
      {
        "rank": 1,
        "chunk_id": "doc_hash_source_hash_0001_content_hash",
        "text": "Complete retrieved chunk text...",
        "source": {
          "doc_id": "doc_hash",
          "path": "data/documents/report.pdf",
          "page_start": 48,
          "page_end": 49
        },
        "scores": {
          "final": 0.91,
          "original": 0.03,
          "rerank": 0.91
        },
        "metadata": {
          "title": "Cash Flow Statement",
          "chunk_index": 84,
          "start_offset": 199192,
          "end_offset": 202457
        }
      }
    ],
    "has_images": false,
    "image_count": 0,
    "is_empty": false
  },
  "isError": false
}
```

| 字段 | 用途 |
|---|---|
| `content` | 面向 LLM 或通用 MCP Host 的 Content Blocks；默认第一个 Block 是完整 XML，存在关联图片时追加 `ImageContent` |
| `structuredContent` | 与 XML 来自同一 Canonical Payload 的完整 JSON，供 Agent State、Evidence Ledger、过滤和程序解析使用 |
| `results[].text` | 未截断、未压缩空白的完整 Chunk Text，是回答问题的主要 Evidence |
| `results[].source` | 文档 ID、来源路径以及 Chunk 覆盖的物理 PDF 页码范围 |
| `results[].scores` | 最终分数，以及当前检索路径能够提供的原始、融合或重排分数 |
| `results[].metadata` | 标题、Chunk Index、Offsets 和处理方式等辅助信息；不再重复返回 `text` |
| `isError` | Tool 执行是否失败；无命中结果使用 `is_empty=true`，不等同于执行错误 |

Chroma 内部将完整正文保存在 `documents` 字段，BM25 只保存词项统计、Chunk ID
和评分信息。Sparse Retrieval 根据 BM25 返回的 Chunk ID 调用 Chroma
`get_by_ids()`，再从 `documents` 取回正文。因此，MCP Response 删除重复的
`metadata.text` 不会影响 Dense Retrieval、Sparse Retrieval 或 Rerank。

对于自定义 Agent，推荐将 `content` 中的 XML 交给 LLM，将
`structuredContent` 保存到 Global State；第三方 MCP Host 如何组装最终模型上下文，
则由对应 Host 的实现决定。

## FinanceBench 评估

本项目的金融问答评估适配自 Patronus AI 发布的
[FinanceBench 官方 GitHub 仓库](https://github.com/patronus-ai/financebench)，
基准设计与数据说明可参考
[FinanceBench 论文](https://arxiv.org/abs/2311.11944)。本项目仅下载和转换其公开数据，
不将 Benchmark 原始 PDF、JSONL 或生成的索引提交到仓库。

Benchmark 数据保存在本地 `data/benchmarks/`，不会上传 GitHub。先根据配置准备数据：

```powershell
python scripts/prepare_benchmark.py --config config/settings.yaml
```

查看实验配置和索引复用计划：

```powershell
python scripts/evaluate.py --config config/settings.yaml --dry-run
```

运行指定实验：

```powershell
python scripts/evaluate.py --config config/settings.yaml --experiments baseline
```

评估系统支持 Document/Page/Evidence Hit Rate、MRR、Atomic-Fact Context Recall、Context Precision@K、答案指标、LLM Evidence Judge、每 Query 记录、Checkpoint Resume 和实验对比 CSV。索引相关配置会生成 Fingerprint，相同索引可以复用，查询与评估配置变化不必重复解析全部 PDF。

## 测试

```powershell
pytest tests/unit -q
pytest tests/integration -m "not llm" -q
pytest tests/e2e/test_mcp_client.py tests/e2e/test_mcp_sdk_client.py -q
```

涉及真实 Provider 的测试需要对应 API Key 和可用模型；其余测试使用本地 Fixture 或 Mock。

## 数据清理

先预览：

```powershell
python scripts/clear_data.py --all --dry-run
```

确认后清理 Chroma、BM25、摄取记录、评估结果和日志：

```powershell
python scripts/clear_data.py --all --yes
```

只重建知识库索引并保留耗时生成的 PDF parsed cache：

```powershell
python scripts/clear_data.py --storage --keep-parsed --yes
```

## Agentic RAG Roadmap

下一阶段计划在当前 Knowledge Hub 之上构建 **Centralized Multi-Agent System**：

- Root Orchestrator 独占全局状态、控制流与最终决策。
- Planner、Query Rewriter、Retriever、Evidence Curator、Context Sufficiency Checker 和 Synthesizer 使用独立 Prompt 与局部工作记忆。
- Sub-agent 不直接相互通信，也不直接修改全局状态，只向 Orchestrator 返回严格结构化结果。
- Orchestrator 支持多轮检索、证据去重、缺口分析、停止条件、预算控制与完整 Trace。
- 当前 MCP Tools 作为可复用的 Knowledge Retrieval Tools，保持与 Agent 编排层解耦。

这一边界让传统 RAG 基线仍可独立使用和评估，也让未来的 Agent Workflow 可以替换 Planner、Retriever 或 Judge，而无需重写摄取和索引系统。
