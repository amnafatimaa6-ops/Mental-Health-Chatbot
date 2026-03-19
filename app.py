import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Mental Health Chatbot")

# Load models
@st.cache_resource(show_spinner=False)
def load_models():
    chatbot = pipeline("text2text-generation", model="google/flan-t5-large")
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return chatbot, sentiment

chatbot, sentiment = load_models()

# Session state
st.session_state.setdefault('conversation_history', [])

# Generate chatbot response
def generate_response(user_input):
    # Analyze sentiment
    sentiment_result = sentiment(user_input)[0]
    sentiment_label = sentiment_result['label']
    
    # Modify prompt based on sentiment
    if sentiment_label == "NEGATIVE":
        prompt = f"Respond empathetically and supportively to: {user_input}"
    else:
        prompt = user_input
    
    response = chatbot(prompt, max_length=150, do_sample=True)
    ai_response = response[0]['generated_text']
    
    # Save conversation
    st.session_state['conversation_history'].append({"role": "user", "content": user_input})
    st.session_state['conversation_history'].append({"role": "assistant", "content": ai_response})
    
    return ai_response

# UI
st.title("Mental Health Chatbot")

# Show history
for msg in st.session_state['conversation_history']:
    role = "You" if msg['role'] == "user" else "AI"
    st.markdown(f"**{role}:** {msg['content']}")

user_message = st.text_input("How can I help you today?")

if user_message:
    with st.spinner("Thinking..."):
        ai_response = generate_response(user_message)
        st.markdown(f"**AI:** {ai_response}")
