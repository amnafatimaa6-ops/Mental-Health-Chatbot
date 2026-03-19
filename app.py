import streamlit as st
import requests

st.set_page_config(page_title="Mental Health Chatbot")

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def query_hf(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

st.session_state.setdefault("conversation_history", [])

def generate_response(user_input):
    # Keep only last 5 messages to reduce token count
    history = st.session_state["conversation_history"][-5:]

    # Build messages for Llama API
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
    messages.append({"role": "user", "content": user_input})

    payload = {"inputs": messages}
    res = query_hf(payload)

    # Hugging Face returns nested JSON
    try:
        ai_response = res[0]["generated_text"]
    except:
        ai_response = "⚠️ Model is loading or busy. Try again in a moment."

    st.session_state["conversation_history"].append({"role": "user", "content": user_input})
    st.session_state["conversation_history"].append({"role": "assistant", "content": ai_response})
    return ai_response

# UI
st.title("Mental Health Support Agent")
for msg in st.session_state["conversation_history"]:
    role = "You" if msg["role"] == "user" else "AI"
    st.markdown(f"**{role}:** {msg['content']}")

user_message = st.text_input("How can I help you today?")
if user_message:
    with st.spinner("Thinking..."):
        ai_response = generate_response(user_message)
        st.markdown(f"**AI:** {ai_response}")
