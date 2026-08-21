import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from config import *



def load_documents(folder_path: str):
    """加载docs文件夹下面所有pdf、txt文档，容错跳过损坏文件"""
    docs = []
    if not os.path.exists(folder_path):
        print(f"⚠️文件夹 {folder_path} 不存在")
        return docs

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf‑8")
            else:
                continue
            docs.extend(loader.load())
            print(f"✅成功读取文件：{file}")
        except Exception as e:
            print(f"❌读取文件失败 {file}，跳过，错误：{str(e)}")
    return docs




def build_vector_db():
    """构建向量数据库，文档加载→切片→向量化→存入Chroma"""
    documents = load_documents("./docs")
    if len(documents) == 0:
        raise RuntimeError("docs文件夹没有可读取文档，请放入pdf/txt")

    print(f"\n文档总页数：{len(documents)}")

    # 文档切片
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n','\n',' ','']
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"切片完成，切片总数：{len(split_docs)}")

    # 向量化
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 确保向量库目录存在
    os.makedirs(PERSIST_PATH, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=PERSIST_PATH
    )
    print("\n✅向量库构建完成！")
    return vectordb



def get_qa_chain():
    """加载向量库，组装RAG检索问答链"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=PERSIST_PATH,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    # 对接豆包，使用OpenAI兼容接口
    llm = ChatOpenAI(
        api_key=DOUBAO_API_KEY,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model=LLM_MODEL,
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是企业内部知识库问答助手，回答简洁专业。严格遵守规则：\n1.依据提供的参考资料提炼总结回答，不必逐字照搬原文。\n2.只有完全没有相关内容才回复【知识库未查询到相关信息】，禁止编造外部知识。\n3.回答结尾标注信息来源。"),
        ("human", "参考资料：\n{context}\n\n用户问题：{input}")
    ])



    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain



if __name__ == "__main__":
    # ==========第一次运行取消注释构建向量库，构建完注释掉==========
    # build_vector_db()

    try:
        qa_chain = get_qa_chain()
        print("\n🤖知识库问答机器人已启动，输入exit退出")
        while True:
            query = input("\n请输入你的问题：")
            if query.strip().lower() == "exit":
                print("👋程序退出")
                break
            if not query.strip():
                continue

            result = qa_chain.invoke({"input": query})

            print("\n===AI回答===")
            print(result["answer"])
            print("\n===检索到的原文片段===")
            for idx, doc in enumerate(result["context"]):
                print(f"【片段{idx+1}】{doc.page_content[:300]}...")

    except Exception as err:
        print(f"\n❌运行异常：{err}")
