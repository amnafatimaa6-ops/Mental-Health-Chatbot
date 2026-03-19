import streamlit as st
from transformers import pipeline

# Page setup
st.set_page_config(page_title="Therapist Bot", layout="centered")

# Load sentiment analysis model (DistilBERT fine-tuned on SST-2)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

# Session state for conversation
st.session_state.setdefault('conversation_history', [])

# Keywords for basic emotional understanding
NEGATIVE_KEYWORDS = [
    "sad", "depressed", "unhappy", "not good", "i don't feel good", "anxious",
    "worried", "stressed", "upset", "tired", "overwhelmed"
]
POSITIVE_KEYWORDS = [
    "happy", "good", "great", "excited", "joyful", "glad", "awesome", "fantastic"
]

# Therapist-like response generator
def generate_response(user_input):
    user_lower = user_input.lower()

    # Check keywords first
    if any(kw in user_lower for kw in NEGATIVE_KEYWORDS):
        response = "I'm here for you. Can you tell me more about what’s bothering you?"
    elif any(kw in user_lower for kw in POSITIVE_KEYWORDS):
        response = "That's wonderful! Keep that positive energy going. 🌟"
    else:
        # fallback: sentiment analysis
        sentiment = sentiment_analyzer(user_input)[0]['label']
        if sentiment == "NEGATIVE":
            response = "I see. That sounds tough. Want to talk more about it?"
        else:
            response = "I understand. Take a deep breath and relax. Tell me more if you want."

    # Save conversation
    st.session_state['conversation_history'].append({"role": "user", "content": user_input})
    st.session_state['conversation_history'].append({"role": "assistant", "content": response})
    return response

# Generate positive affirmation
def generate_affirmation():
    affirmations = [
        "You are stronger than you think. 💪",
        "Take a deep breath — you’ve got this. 🌟",
        "Every step forward, no matter how small, is progress.",
        "You deserve peace and happiness today. 💙",
        "It’s okay to feel how you feel. You are valid."
    ]
    import random
    return random.choice(affirmations)

# Generate guided meditation
def generate_meditation_guide():
    guides = [
        "Close your eyes. Take a deep breath in, hold for 3 seconds, then exhale slowly. Repeat 5 times while imagining a peaceful place.",
        "Focus on your breathing. Inhale deeply for 4 seconds, exhale for 6 seconds. Let go of tension with each breath.",
        "Picture yourself on a quiet beach. Feel the sun, the sand, and the waves. Take a slow breath in and out, letting your body relax."
    ]
    import random
    return random.choice(guides)

# Streamlit UI
st.title("🧠 Therapist Bot")

# Show conversation history
for msg in st.session_state['conversation_history']:
    role = "You" if msg['role'] == "user" else "AI"
    st.markdown(f"**{role}:** {msg['content']}")

# User input
user_message = st.text_input("How are you feeling today?")

if user_message:
    with st.spinner("Thinking..."):
        ai_response = generate_response(user_message)
        st.markdown(f"**AI:** {ai_response}")

# Buttons for affirmation and meditation
col1, col2 = st.columns(2)

with col1:
    if st.button("Give me a positive affirmation"):
        affirmation = generate_affirmation()
        st.markdown(f"**Affirmation:** {affirmation}")

with col2:
    if st.button("Give me a guided meditation"):
        meditation = generate_meditation_guide()
        st.markdown(f"**Guided Meditation:** {meditation}")
