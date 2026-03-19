import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(page_title="Therapist Bot")

# Load sentiment model
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment()

# Session state for conversation
st.session_state.setdefault('conversation_history', [])

# Simple rule-based therapist
def generate_response(user_input):
    sentiment = sentiment_analyzer(user_input)[0]['label']
    response = ""

    # Keyword-based rules
    user_lower = user_input.lower()
    if any(word in user_lower for word in ["sad", "depressed", "unhappy"]):
        response = "I'm here for you. Can you tell me more about what’s making you feel this way?"
    elif any(word in user_lower for word in ["stress", "overwhelmed", "anxious", "worried"]):
        response = "It sounds stressful. Try taking a slow breath with me: inhale… exhale… How do you feel now?"
    elif any(word in user_lower for word in ["happy", "good", "great"]):
        response = "I’m glad to hear that! Keep that positive energy going. 🌟"
    else:
        # Default based on sentiment
        if sentiment == "NEGATIVE":
            response = "I see. That sounds tough. Want to talk more about it?"
        else:
            response = "I understand. Tell me more or take a deep breath and relax."

    # Save to conversation
    st.session_state['conversation_history'].append({"role": "user", "content": user_input})
    st.session_state['conversation_history'].append({"role": "assistant", "content": response})
    return response

# UI
st.title("Therapist Bot (Sentiment + Rules)")

for msg in st.session_state['conversation_history']:
    role = "You" if msg['role'] == "user" else "AI"
    st.markdown(f"**{role}:** {msg['content']}")

user_message = st.text_input("How are you feeling today?")
if user_message:
    with st.spinner("Thinking..."):
        ai_response = generate_response(user_message)
        st.markdown(f"**AI:** {ai_response}")
