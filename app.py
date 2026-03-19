import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Page config
st.set_page_config(page_title="Mental Health Chatbot")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# Load models (cached for speed)
@st.cache_resource(show_spinner=True)
def load_models():
    # Chatbot: DialoGPT-small
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Sentiment analysis: DistilBERT
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return chatbot, sentiment

chatbot, sentiment = load_models()

# Session state for conversation
st.session_state.setdefault('conversation_history', [])

# Generate response
def generate_response(user_input):
    # Sentiment check
    sentiment_result = sentiment(user_input)[0]
    sentiment_label = sentiment_result['label']

    # Optional: empathetic prefix if negative
    if sentiment_label == "NEGATIVE":
        prompt = f"I'm an empathetic assistant. Respond kindly: {user_input}"
    else:
        prompt = user_input

    # Generate response (short for cloud efficiency)
    response = chatbot(prompt, max_length=100, do_sample=True)
    ai_response = response[0]['generated_text']

    # Store conversation
    st.session_state['conversation_history'].append({"role": "user", "content": user_input})
    st.session_state['conversation_history'].append({"role": "assistant", "content": ai_response})

    return ai_response

# Generate positive affirmation
def generate_affirmation():
    prompt = "Give a short, kind, and uplifting affirmation for someone feeling stressed."
    response = chatbot(prompt, max_length=50, do_sample=True)
    return response[0]['generated_text']

# Generate guided meditation
def generate_meditation_guide():
    prompt = "Provide a short 5-minute guided meditation to help someone relax."
    response = chatbot(prompt, max_length=150, do_sample=True)
    return response[0]['generated_text']

# UI
st.title("Mental Health Support Agent")

# Show chat history
for msg in st.session_state['conversation_history']:
    role = "You" if msg['role'] == "user" else "AI"
    st.markdown(f"**{role}:** {msg['content']}")

# User input
user_message = st.text_input("How can I help you today?")

if user_message:
    with st.spinner("Thinking..."):
        ai_response = generate_response(user_message)
        st.markdown(f"**AI:** {ai_response}")

# Buttons for extra features
col1, col2 = st.columns(2)

with col1:
    if st.button("Give me a positive affirmation"):
        affirmation = generate_affirmation()
        st.markdown(f"**Affirmation:** {affirmation}")

with col2:
    if st.button("Give me a guided meditation"):
        meditation_guide = generate_meditation_guide()
        st.markdown(f"**Guided Meditation:** {meditation_guide}")
