import streamlit as st
import ollama

st.set_page_config(page_title="校园AI助手", page_icon="🎓")
st.title("🎓 校园AI助手")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 聊天输入框
if prompt := st.chat_input("输入你的问题，比如：奖学金怎么申请？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                response = ollama.chat(
                    model="deepseek-r1:8b",
                    messages=[
                        {"role": "system", "content": "你是一个校园助手，用简洁中文回答学生问题。"},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response["message"]["content"]
            except Exception as e:
                answer = f"调用模型出错：{e}"
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})