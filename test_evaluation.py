# test_evaluation.py
"""
RAG召回自测脚本
测试用例：问题 + 预期应该召回的关键词，简单计算召回率
"""
from main import get_qa_chain

test_case = [
    {"question":"员工年假有多少天","expect_keyword":"年假"},
    {"question":"年假可以跨年累计吗","expect_keyword":"跨年累计"},
    {"question":"月度迟到扣款规则是什么","expect_keyword":"迟到"},
    {"question":"普通员工出差住宿标准","expect_keyword":"普通员工"},
    {"question":"出差报销最晚多久提交","expect_keyword":"7个工作日"},
    {"question":"试用期离职提前多少天申请","expect_keyword":"试用期"},
    {"question":"加班调休有效期多久","expect_keyword":"调休有效期"}
]

def evaluate():
    qa_chain = get_qa_chain()
    hit = 0
    total = len(test_case)
    for case in test_case:
        # 传入必须是 {"input":xxx}
        res = qa_chain.invoke({"input": case["question"]})
        # 字段是 context，不是 source_documents
        source_text = "".join([d.page_content for d in res["context"]])

        if case["expect_keyword"] in source_text:
            hit += 1
            print(f"✅问题: {case['question']} 召回命中")
        else:
            print(f"❌问题: {case['question']} 未召回目标内容")

    recall = hit / total
    print(f"\n✅简单评测召回率：{recall:.2f}")

if __name__ == "__main__":
    evaluate()
