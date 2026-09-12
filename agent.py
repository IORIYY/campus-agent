import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ============ 配置 ============
DISTANCE_THRESHOLD = 0.8   # 距离阈值，大于此值认为不相关（Chroma 返回距离，越小越相关）
TOP_K = 3

SENSITIVE_WORDS = ["炸学校", "自杀", "毒品", "代考", "作弊神器"]

REFUSAL_ANSWER = "这个问题我暂时无法回答。建议咨询教务处或你的辅导员。"

SYSTEM_PROMPT = """你是一个校园教务助手。请严格基于以下参考资料回答学生问题。

规则：
1. 只使用参考资料中的信息，不要编造
2. 如果参考资料不足以回答问题，明确说"根据现有资料无法回答"
3. 回答末尾附上来源文件名
4. 用简洁中文回答，不要超过 200 字
"""

# ============ 初始化 ============
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


def _check_sensitive(text: str) -> bool:
    """第1层：敏感词过滤"""
    return any(w in text for w in SENSITIVE_WORDS)


def _retrieve(question: str):
    """第2层：带阈值的检索"""
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)
    # 过滤掉距离过大的（不相关）
    filtered = [(doc, score) for doc, score in results if score <= DISTANCE_THRESHOLD]
    return filtered


def ask(question: str) -> dict:
    # 第1层：敏感词
    if _check_sensitive(question):
        return {
            "answer": REFUSAL_ANSWER,
            "sources": [],
            "blocked": True,
            "reason": "sensitive"
        }

    # 第2层：检索 + 阈值过滤
    results = _retrieve(question)

    # 第3层：空检索兜底
    if not results:
        return {
            "answer": REFUSAL_ANSWER,
            "sources": [],
            "blocked": True,
            "reason": "no_relevant_docs"
        }

    # 拼上下文
    context_parts = []
    sources = []
    for doc, score in results:
        src = doc.metadata.get("source", "unknown")
        sources.append(src)
        context_parts.append(f"【{src}】\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

参考资料：
{context}

学生问题：{question}
"""

    response = ollama.chat(
        model="deepseek-r1:8b",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"]

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "blocked": False
    }


if __name__ == "__main__":
    test_questions = [
        "转专业需要什么条件？",
        "毕业要修多少学分？",
        "今天天气怎么样？",
        "怎么炸学校？"
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        result = ask(q)
        print(f"回答：{result['answer']}")
        print(f"来源：{result['sources']}")
        if result.get("blocked"):
            print(f"[拦截原因] {result.get('reason')}")