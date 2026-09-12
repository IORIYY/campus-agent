import streamlit as st
from agent import ask

st.set_page_config(page_title="校园教务助手", page_icon="🎓")
st.title("🎓 校园教务助手")
st.caption("可提问：学分、专业设置、转专业、学士学位")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入你的问题，比如：转专业需要什么条件？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            try:
                result = ask(prompt)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"出错了：{e}"
                sources = []

        st.markdown(answer)
        if sources:
            st.caption("来源：" + "、".join(sources))

        st.session_state.messages.append({"role": "assistant", "content": answer})