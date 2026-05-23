# 软件测试教学助理

一个面向软件测试教育的检索增强生成（RAG）代理，专注于软件测试教学，结合本地文档检索、题目生成、答案评估、讲解与测试用例设计。基于 **LangChain**、**ChromaDB**、**BM25** 和 **AIHubMix OpenAI API**（GPT-3.5-turbo）构建，并通过有状态图进行编排。

## 功能

- **PDF 预处理与分块** - 从软件测试教材和文档中提取文本与表格。将内容按语义切分为块（500 字符，带重叠）并序列化存储为 `all_docs.json`。
- **知识检索** - 对本地软件测试文档查询使用 BM25 与 ChromaDB（向量搜索）集成检索。
- **题目生成** - 生成有关软件测试概念与方法论的练习题和测试题。
- **答案评估** - 评估学生答案并提供建设性反馈。
- **讲解** - 提供关于软件测试概念、技术和最佳实践的详细解释。
- **测试用例设计** - 分析需求与代码以设计全面的测试用例。
- **上下文管理** - 构建并压缩对话历史以供 LLM 提示使用。记录时间戳并记录工具调用/响应。
- **编排有向图** - 有状态的查询路由：意图分类 → 工具调用 → 查询精化 → 最终回答。
- **可选 Chainlit UI** - 用于实时演示与测试的交互式前端。

## 安装与配置

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置 AIHubMix API**
   - 详见 [AIHubMix_Config.md](AIHubMix_Config.md) 中的设置说明
   - 在 `.env` 文件中填写您的 AIHubMix API Key

3. **准备文档**
   - 将软件测试教材与资料放入 `PDFs/` 文件夹
   - 运行预处理以创建知识库

4. **运行系统**
   ```bash
   python main.py
   # 或者
   chainlit run app.py
   ```

## 工作流
<img src="Output and Workflow/graph.png" width="600"/>


## 文件夹结构

```
/ (project root)
├── __pycache__/            # Python 缓存
├── .chainlit/              # Chainlit 状态/配置
├── .vscode/                # VS Code 设置
├── chromaDB/               # 持久化的 ChromaDB 存储
├── myenv/                  # 虚拟环境
├── PDFs/                   # 源 PDF（例如 软件测试教材、需求文档）
├── utils/                  # 辅助笔记本（draw.ipynb）
├── Workflows/              # 工作流图（graph.png）
├── .env                    # 环境变量
├── .gitignore              # 忽略规则
├── all_docs.json           # 序列化后的分块与表格
├── app.py                  # Chainlit 入口
├── chainlit.md             # Chainlit 文档
├── check_agent_log.json    # 工具调用日志
├── main.py                 # 编排与工具集合
├── preprocessing.py        # PDF 解析与分块
├── query.txt               # 示例查询
├── requirements.txt        # Python 依赖
├── utility.py              # 上下文与日志工具
└── README.md               # 项目概览
```

## 安装与部署

克隆仓库：
```bash
git clone <repo_url>
cd <repo_dir>
```

创建并激活虚拟环境：
```bash
python3 -m venv myenv
source myenv/bin/activate    # macOS/Linux
myenv\Scripts\activate     # Windows
```

安装依赖：
```bash
pip install -r requirements.txt
```

通过复制 `.env.example` 为 `.env` 并设置所需变量来配置环境：
```ini
MODEL_NAME=qwen/qwen3-32b
SERPER_API_KEY=<your_serper_key>
ALPHAVANTAGE_API_KEY=<your_alpha_vantage_key>
PDF_DIR=PDFs/
ALL_DOCS_JSON=all_docs.json
CHROMA_DB_PATH=chromaDB/saved/
COLLECTION_NAME=RAG_DOCS
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

## 运行代理

**预处理 PDFs** - 提取并分块 PDFs：
```bash
python preprocessing.py
```
这会生成 `all_docs.json` 并填充 ChromaDB 索引。

**通过 Chainlit UI 启动代理**：
```bash
chainlit run app.py
```

或在命令行模式运行：
```bash
python main.py
```

## 核心组件

**PDF 预处理（`preprocessing.py`）** - 包含 `recursive_split` 与 `semantic_chunker` 用于语义文本分块。通过 `pdfplumber` 做表格提取。序列化到 `all_docs.json`。

**上下文工具（`utility.py`）** - `get_context` 与 `compress_context` 用于构建/压缩聊天历史。`append_to_response` 记录带有 IST 时间戳的工具调用。`remove_think` 用于剥离 `<think>` 块。

**工具与检索（`main.py`）** - 基于 BM25 + ChromaDB 的混合 PDF 搜索。包含网页工具如 `google_search` 与 `wiki_lookup`。

**编排有向图** - 使用 `StateGraph`（langgraph）构建。节点：输入 → 意图路由 → 工具节点 → 检查 → 扩展/回答 → 结束。路由器根据 LLM/工具输出引导流程。

## 定制

- **添加 PDFs** - 将文件放入 `PDFs/` 并重新运行预处理
- **切换模型** - 在 `.env` 中更新 `MODEL_NAME`
- **调优检索** - 调整 BM25 的 `k`、MMR 的 `lambda_mult` 或集成权重
- **扩展工具** - 用 `@tool` 装饰新函数并在 `main.py` 中接入

## 贡献

欢迎贡献！可以打开 Issue 或提交 Pull Request。
