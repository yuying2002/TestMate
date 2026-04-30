# AIHubMix OpenAI 配置指南

## 1. 获取AIHubMix API密钥

1. 访问 [AIHubMix](https://aihubmix.com)
2. 注册账号并登录
3. 在控制台获取你的API Key

## 2. 配置环境变量

编辑 `.env` 文件，填入你的配置：

```env
# AIHubMix OpenAI Configuration
MODEL_NAME=gpt-3.5-turbo
OPENAI_API_KEY=你的AIHubMix_API_KEY
OPENAI_BASE_URL=https://aihubmix.com/v1

# 其他配置保持默认
PDF_DIR=PDFs/
ALL_DOCS_JSON=all_docs.json
CHROMA_DB_PATH=chromaDB/saved/
COLLECTION_NAME=LEARNING_DOCS
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 4. 运行系统

```bash
# 启动教学助手
python main.py

# 或使用Chainlit界面
chainlit run app.py
```

## 注意事项

- AIHubMix提供免费的OpenAI兼容API
- 默认使用GPT-3.5-turbo模型，性价比高
- 如果需要更强的模型，可以修改MODEL_NAME为gpt-4或gpt-4-turbo
- 确保网络连接正常，能够访问aihubmix.com