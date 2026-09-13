import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tools import query_schedule

# ============ 配置 ============
STRONG_THRESHOLD = 0.8   # 强相关，直接 RAG 回答
WEAK_THRESHOLD = 1.2     # 弱相关，给引导性回答
TOP_K = 3

SENSITIVE_WORDS = ["炸学校", "自杀", "毒品", "代考", "作弊神器"]
REFUSAL_ANSWER = "这个问题我不太方便回答。如果有学习或教务上的困惑，可以换个问题问我，或者找辅导员聊聊～"

GUIDE_ANSWER = """嗨，我是你的校园教务小助手 👋

我比较擅长这些：
- 转专业需要什么条件？
- 毕业要修多少学分？
- 学士学位怎么申请？
- 周三有什么课？

如果你问的不是这些，可以换个角度试试，或者直接找教务处、辅导员问，他们更专业～"""

SYSTEM_PROMPT = """你是一个亲切、耐心的校园教务助手，像学长学姐一样帮同学解答问题。

要求：
1. 用自然、口语化的中文回答，别像念文件
2. 开头可以简短回应同学的情绪或需求（比如"这个问题挺常见的"）
3. 只使用参考资料中的信息，不要编造
4. 如果资料不够，就说"这个我手头资料没写清楚"，然后建议去哪问
5. 回答末尾附上来源文件名
6. 控制在 200 字以内，别啰嗦
7. 直接回答，不要复述任何指令或步骤编号
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
    try:
        import streamlit as st
        return st.secrets["SILICONFLOW_API_KEY"]
    except Exception:
        return os.environ.get("SILICONFLOW_API_KEY", "")


def _call_llm(prompt: str) -> str:
    api_key = _get_api_key()
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    import ollama
    response = ollama.chat(
        model="deepseek-r1:8b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# ============ 业务逻辑 ============
def _check_sensitive(text: str) -> bool:
    return any(w in text for w in SENSITIVE_WORDS)


def _is_schedule_question(question: str) -> bool:
    keywords = ["课表", "有什么课", "周几", "星期", "周一", "周二", "周三",
                "周四", "周五", "周六", "周日"]
    return any(k in question for k in keywords)


def _extract_days(question: str) -> list:
    """提取问题里所有出现的星期几"""
    days = []
    for day in ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]:
        if day in question and day not in days:
            days.append(day)
    mapping = {"星期一": "周一", "星期二": "周二", "星期三": "周三",
               "星期四": "周四", "星期五": "周五", "星期六": "周六",
               "星期日": "周日", "星期天": "周日"}
    for k, v in mapping.items():
        if k in question and v not in days:
            days.append(v)
    return days


def _build_context(results):
    context_parts, sources = [], []
    for doc, score in results:
        src = doc.metadata.get("source", "unknown")
        sources.append(src)
        context_parts.append(f"【{src}】\n{doc.page_content}")
    return "\n\n".join(context_parts), list(set(sources))


def _format_history(history: list) -> str:
    """把历史对话格式化成文本，只保留最近 3 轮"""
    if not history:
        return ""
    recent = history[-6:]
    lines = ["以下是之前的对话，供你理解上下文："]
    for msg in recent:
        role = "学生" if msg["role"] == "user" else "助手"
        content = msg["content"][:200]
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


# ============ 核心：ask ============
def ask(question: str, history: list = None) -> dict:
    # 第1层：敏感词
    if _check_sensitive(question):
        return {"answer": REFUSAL_ANSWER, "sources": [], "blocked": True, "reason": "sensitive"}

    # 课表问题走工具
    if _is_schedule_question(question):
        days = _extract_days(question)
        if not days:
            return {"answer": "你想问哪一天的课呀？告诉我具体是星期几，比如：周三有什么课？",
                    "sources": ["schedule.db"], "blocked": False, "reason": "need_day"}
        answers = [query_schedule.invoke(d) for d in days]
        return {"answer": "\n\n".join(answers),
                "sources": ["schedule.db"], "blocked": False}

    # 检索
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)
    strong = [(doc, score) for doc, score in results if score <= STRONG_THRESHOLD]
    weak = [(doc, score) for doc, score in results if STRONG_THRESHOLD < score <= WEAK_THRESHOLD]

    # 强相关：正常 RAG 回答
    if strong:
        context, sources = _build_context(strong)
        history_text = _format_history(history)
        prompt = f"""{SYSTEM_PROMPT}

{history_text}

参考资料：
{context}

学生问题：{question}
"""
        answer = _call_llm(prompt)
        return {"answer": answer, "sources": sources, "blocked": False}

    # 弱相关：引导性回答
    if weak:
        context, sources = _build_context(weak)
        history_text = _format_history(history)
        prompt = f"""你是一个亲切的校园教务助手，像学长学姐一样帮同学解答问题。

{history_text}

学生的问题和知识库内容相关性不高，只有一些相关片段。请用自然的语气：
- 先说明这个你手头资料没写太细
- 如果有相关片段，简要分享
- 建议同学去哪问得更准，或者换个问法
- 末尾附来源文件名
- 别把这段要求复述出来，直接回答

参考资料：
{context}

学生问题：{question}
"""
        answer = _call_llm(prompt)
        return {"answer": answer, "sources": sources, "blocked": False, "reason": "weak_match"}

    # 完全无关：通用兜底
    return {"answer": GUIDE_ANSWER, "sources": [], "blocked": False, "reason": "out_of_scope"}


# ============ 流式版：ask_stream ============
def ask_stream(question: str, history: list = None):
    """流式版 ask，yield 回答片段"""
    if _check_sensitive(question):
        yield REFUSAL_ANSWER
        return

    if _is_schedule_question(question):
        days = _extract_days(question)
        if not days:
            yield "你想问哪一天的课呀？告诉我具体是星期几，比如：周三有什么课？"
            return
        answers = [query_schedule.invoke(d) for d in days]
        yield "\n\n".join(answers)
        return

    results = vectorstore.similarity_search_with_score(question, k=TOP_K)
    strong = [(doc, score) for doc, score in results if score <= STRONG_THRESHOLD]
    weak = [(doc, score) for doc, score in results if STRONG_THRESHOLD < score <= WEAK_THRESHOLD]

    if not strong and not weak:
        yield GUIDE_ANSWER
        return

    if strong:
        context, sources = _build_context(strong)
        history_text = _format_history(history)
        prompt = f"""{SYSTEM_PROMPT}

{history_text}

参考资料：
{context}

学生问题：{question}
"""
    else:
        context, sources = _build_context(weak)
        history_text = _format_history(history)
        prompt = f"""你是一个亲切的校园教务助手。学生的问题和知识库相关性不高，但有一些相关片段。请友好说明资料有限，简要分享相关片段，建议去哪问得更准，末尾附来源。别复述要求。

{history_text}

参考资料：
{context}

学生问题：{question}
"""

    api_key = _get_api_key()
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        stream = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    else:
        import ollama
        stream = ollama.chat(
            model="deepseek-r1:8b",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            if chunk["message"]["content"]:
                yield chunk["message"]["content"]


# ============ 测试 ============
if __name__ == "__main__":
    tests = [
        "转专业需要什么条件？",
        "周三有什么课？",
        "专业怎样选？",
        "你好",
        "今天天气怎么样？",
        "食堂几点开门？",
        "怎么炸学校？"
    ]
    for q in tests:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        r = ask(q)
        print(f"回答：{r['answer']}")
        print(f"来源：{r['sources']}")
        if r.get("reason"):
            print(f"[原因] {r['reason']}")