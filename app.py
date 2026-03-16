import streamlit as st
from llm_model import generate_response


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Mental Health Chatbot",
    page_icon="🧠",
    layout="centered"
)


# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- STYLING ----------
st.markdown("""
<style>

.chat-container{
    display:flex;
    flex-direction:column;
}

.user{
    background:#DCF8C6;
    padding:10px;
    border-radius:10px;
    margin:5px 0;
    width:fit-content;
    max-width:80%;
    align-self:flex-end;
}

.assistant{
    background:#F1F0F0;
    padding:10px;
    border-radius:10px;
    margin:5px 0;
    width:fit-content;
    max-width:80%;
    align-self:flex-start;
}

</style>
""", unsafe_allow_html=True)


# ---------- TITLE ----------
st.title("🧠 AI Mental Health Chatbot")
st.write("Talk to me. I’m here to listen and respond kindly 💛")


# ---------- DISPLAY CHAT ----------
def display_chat():
    for msg in st.session_state.messages:

        if msg["role"] == "user":
            st.markdown(
                f'<div class="user">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f'<div class="assistant">{msg["content"]}</div>',
                unsafe_allow_html=True
            )


display_chat()


# ---------- USER INPUT ----------
with st.form("chat_form", clear_on_submit=True):

    user_input = st.text_input("Your message")

    submitted = st.form_submit_button("Send")


# ---------- PROCESS MESSAGE ----------
if submitted and user_input.strip():

    # add user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # generate response
    with st.spinner("Thinking..."):
        reply = generate_response(st.session_state.messages)

    # add assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

    # rerun to update chat
    st.rerun()
