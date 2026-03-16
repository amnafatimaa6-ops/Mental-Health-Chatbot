# app.py
import streamlit as st
from llm_model import generate_response

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Mental Health Chatbot",
    page_icon="🧠",
    layout="centered"
)

# ---------- STYLING ----------
st.markdown(
    """
    <style>
    .stTextInput>div>div>input {height: 3em; font-size: 18px;}
    .chat-message {padding: 10px; border-radius: 12px; margin-bottom: 8px; max-width: 80%;}
    .user {background-color: #DCF8C6; align-self: flex-end;}
    .assistant {background-color: #F1F0F0; align-self: flex-start;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 AI Mental Health Chatbot")
st.write("Talk to me. I’m here to listen and respond kindly 💛")

# ---------- INPUT ----------
with st.form(key="user_input_form", clear_on_submit=True):
    user_input = st.text_input("Your message:")
    submitted = st.form_submit_button("Send")

# ---------- FUNCTION FOR DISPLAY ----------
def display_chat():
    for msg in st.session_state.messages:
        role_class = "user" if msg["role"] == "user" else "assistant"
        st.markdown(f'<div class="chat-message {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

# ---------- PROCESS INPUT ----------
if submitted and user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate AI reply
    with st.spinner("Thinking..."):
        assistant_reply = generate_response(st.session_state.messages)

    # Add assistant reply
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

display_chat()
