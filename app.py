import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Page config
st.set_page_config(page_title="Mental Health Chatbot")

# Optional clean dark UI
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

# Session state
st.session_state.setdefault('conversation_history', [])

# Generate chatbot response
def generate_response(user_input):
    st.session_state['conversation_history'].append(
        {"role": "user", "content": user_input}
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state['conversation_history']
    )

    ai_response = response.choices[0].message.content

    st.session_state['conversation_history'].append(
        {"role": "assistant", "content": ai_response}
    )

    return ai_response

# Generate affirmation
def generate_affirmation():
    prompt = "Give a short, kind, and uplifting affirmation for someone feeling stressed."
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Generate meditation guide
def generate_meditation_guide():
    prompt = "Provide a short 5-minute guided meditation to relax and reduce stress."
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# UI Title
st.title("Mental Health Support Agent")

# Show chat history
for msg in st.session_state['conversation_history']:
    role = "You" if msg['role'] == "user" else "AI"
    st.markdown(f"**{role}:** {msg['content']}")

# User input
user_message = st.text_input("How can I help you today?")

# Chat response
if user_message:
    with st.spinner("Thinking..."):
        ai_response = generate_response(user_message)
        st.markdown(f"**AI:** {ai_response}")

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Give me a positive affirmation"):
        affirmation = generate_affirmation()
        st.markdown(f"**Affirmation:** {affirmation}")

with col2:
    if st.button("Give me a guided meditation"):
        meditation_guide = generate_meditation_guide()
        st.markdown(f"**Guided Meditation:** {meditation_guide}")
