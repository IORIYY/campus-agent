import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tools import query_schedule

# ============ 配置 ============
DISTANCE_THRESHOLD = 0.8
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


# ============ LLM 调用（双模式） ============
def _get_api_key() -> str:
    """优先从 Streamlit secrets 读，其次环境变量"""
    try:
        import streamlit as st
        return st.secrets["SILICONFLOW_API_KEY"]
    except Exception:
        return os.environ.get("SILICONFLOW_API_KEY", "")


def _call_llm(prompt: str) -> str:
    """如果有 API Key 就用云端，否则用本地 Ollama"""
    api_key = _get_api_key()

    if api_key:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    # 本地 Ollama
    import ollama
    response = ollama.chat(
        model="deepseek-r1:8b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# ============ 业务逻辑 ============
def _check_sensitive(text: str) -> bool:
    return any(w in text for w in SENSITIVE_WORDS)


def _retrieve(question: str):
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)
    return [(doc, score) for doc, score in results if score <= DISTANCE_THRESHOLD]


def _is_schedule_question(question: str) -> bool:
    keywords = ["课表", "有什么课", "周几", "星期", "周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return any(k in question for k in keywords)


def _extract_day(question: str) -> str:
    for day in ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]:
        if day in question:
            return day
    mapping = {"星期一": "周一", "星期二": "周二", "星期三": "周三",
               "星期四": "周四", "星期五": "周五", "星期六": "周六",
               "星期日": "周日", "星期天": "周日"}
    for k, v in mapping.items():
        if k in question:
            return v
    return ""


def ask(question: str) -> dict:
    # 第1层：敏感词
    if _check_sensitive(question):
        return {"answer": REFUSAL_ANSWER, "sources": [], "blocked": True, "reason": "sensitive"}

    # 路由：课表问题走工具
    if _is_schedule_question(question):
        day = _extract_day(question)
        if not day:
            return {
                "answer": "请告诉我具体是星期几，比如：周三有什么课？",
                "sources": ["schedule.db"],
                "blocked": False,
                "reason": "need_day"
            }
        tool_result = query_schedule.invoke(day)
        return {"answer": tool_result, "sources": ["schedule.db"], "blocked": False}

    # 路由：知识问题走 RAG
    results = _retrieve(question)
    if not results:
        return {"answer": REFUSAL_ANSWER, "sources": [], "blocked": True, "reason": "no_relevant_docs"}

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

    answer = _call_llm(prompt)
    return {"answer": answer, "sources": list(set(sources)), "blocked": False}


if __name__ == "__main__":
    tests = [
        "转专业需要什么条件？",
        "周三有什么课？",
        "今天天气怎么样？",
        "怎么炸学校？"
    ]
    for q in tests:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        r = ask(q)
        print(f"回答：{r['answer']}")
        print(f"来源：{r['sources']}")
        if r.get("blocked"):
            print(f"[拦截原因] {r.get('reason')}")