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
  -> MarkItDown 文本解析 + PyMuPDF 页码/图片处理
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
|   |-- clear_data.py             # 清理 Storage、Evaluation 和 Logs
|   `-- start_dashboard.py        # Streamlit Dashboard 启动器
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
| `query_knowledge_hub` | 执行 Hybrid Search 和可选 Rerank，返回 Evidence Snippet、Citation 与可选图片 |
| `list_collections` | 返回 Collection、文档数量和 Chunk 数量 |
| `list_documents` | 返回指定 Collection 内的文档目录和 `doc_id` |
| `get_document_summary` | 根据 `doc_id` 获取文档预览与元数据 |

MCP Server 只提供知识检索工具，不直接生成最终答案。Copilot、Claude Desktop 或上层 Agent 会把 Tool Result 作为上下文，再生成面向用户的回答。

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

评估系统支持 Document/Page/Evidence Hit Rate、MRR、答案指标、LLM Evidence Judge、每 Query 记录、Checkpoint Resume 和实验对比 CSV。索引相关配置会生成 Fingerprint，相同索引可以复用，查询与评估配置变化不必重复解析全部 PDF。

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

## Agentic RAG Roadmap

下一阶段计划在当前 Knowledge Hub 之上构建 **Centralized Multi-Agent System**：

- Root Orchestrator 独占全局状态、控制流与最终决策。
- Planner、Query Rewriter、Retriever、Evidence Curator、Context Sufficiency Checker 和 Synthesizer 使用独立 Prompt 与局部工作记忆。
- Sub-agent 不直接相互通信，也不直接修改全局状态，只向 Orchestrator 返回严格结构化结果。
- Orchestrator 支持多轮检索、证据去重、缺口分析、停止条件、预算控制与完整 Trace。
- 当前 MCP Tools 作为可复用的 Knowledge Retrieval Tools，保持与 Agent 编排层解耦。

这一边界让传统 RAG 基线仍可独立使用和评估，也让未来的 Agent Workflow 可以替换 Planner、Retriever 或 Judge，而无需重写摄取和索引系统。
