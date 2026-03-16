import streamlit as st
from llm_model import generate_response

st.set_page_config(page_title="AI Mental Health Chatbot", page_icon="🧠")

st.title("🧠 AI Mental Health Chatbot")
st.write("Talk to me. I’m here to listen and respond kindly 💛")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.write(f"**You:** {msg['content']}")
    else:
        st.write(f"**AI:** {msg['content']}")

user_input = st.text_input("Your message")

if st.button("Send") and user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    reply = generate_response(st.session_state.messages)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

    st.rerun()
