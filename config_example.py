# config_example.py
# 使用说明：复制本文件，重命名为 config.py，填入自己的API KEY
# config.py
# 字节开放平台获取豆包API_KEY
DOUBAO_API_KEY = "XXX-XXXX-XXXX-XXXX-XXXX-XXXXXXX"

# 大模型名称
LLM_MODEL = "XXXXX"
# 向量化模型，中文开源embedding
EMBED_MODEL = "all-MiniLM-L6-v2"
# 向量数据库保存路径
PERSIST_PATH = "./db"

# RAG核心调参参数
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
TOP_K = 4