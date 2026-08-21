# 企业私有知识库RAG问答系统
> 端到端RAG应用
> 基于LangChain实现完整RAG链路，控制台交互，

## ✨项目功能
1. 文档解析：支持PDF/TXT/Markdown文档自动加载，异常文件容错跳过
2. 文本切片：RecursiveCharacterTextSplitter，自定义chunk_size、overlap分割文档
3. 向量化与向量库：sentence‑transformers Embedding，Chroma本地向量数据库持久化存储
4. RAG检索增强问答：基于私有知识库回答，约束大模型幻觉；采用ChatPromptTemplate区分System角色与用户上下文；知识库无信息时禁止编造输出
5. 可观测调试：输出检索原始文档片段，方便调优召回效果（top‑k、切片大小、重叠度）
6. 简单评测脚本，用于RAG链路自测：幻觉检测、答非所问、召回效果评估

## 🛠技术栈
- Python：3.12
- LangChain生态：langchain / langchain‑core / langchain‑community / langchain‑huggingface
- Chroma 本地向量数据库
- sentence‑transformers 开源Embedding嵌入模型
- LLM：字节豆包API（OpenAI兼容接口，远程调用，无需本地部署大模型）

## 📁项目目录结构
rag_fde_demo
├── docs                # 存放测试知识库文档 (PDF/TXT)
├── .gitignore          # git 忽略配置
├── config_example.py   # 配置模板，复制改名为 config.py 填入密钥
├── main.py             # 主程序，控制台 RAG 问答
├── test_evaluation.py  # RAG 简单评测脚本
├── requirements.txt
└── README.md
