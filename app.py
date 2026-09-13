import streamlit as st
from agent import ask_stream, ask

st.set_page_config(page_title="校园教务助手", page_icon="🎓")
st.title("🎓 校园教务助手")
st.caption("可提问：学分、专业设置、转专业、学士学位、课表查询")

# 初始化历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入
if prompt := st.chat_input("输入你的问题，比如：转专业需要什么条件？"):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 流式生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成回答..."):
            # 用 ask 拿到 sources（非流式），再流式输出
            result = ask(prompt, history=st.session_state.messages[:-1])
            sources = result.get("sources", [])

        # 流式展示
        answer = st.write_stream(ask_stream(prompt, history=st.session_state.messages[:-1]))

        if sources:
            st.caption("来源：" + "、".join(sources))

    # 保存回答
    st.session_state.messages.append({"role": "assistant", "content": answer})