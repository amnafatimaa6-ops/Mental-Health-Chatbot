import streamlit as st
from llm_model import generate_response

st.title("🧠 AI Mental Health Chatbot")

user_input = st.text_input("How are you feeling today?")

if user_input:

    if check_safety(user_input):  # optional
        st.error(
            "⚠️ I'm concerned about your safety. "
            "Please reach out to a trusted person or professional."
        )
    else:
        with st.spinner("Thinking..."):
            reply = generate_response(user_input)
        st.write("Assistant:")
        st.write(reply)
