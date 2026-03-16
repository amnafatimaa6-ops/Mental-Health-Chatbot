import requests
import os

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

def generate_response(messages):

    user_message = messages[-1]["content"]

    prompt = f"""
You are a compassionate mental health support assistant.
Respond kindly and supportively.

User: {user_message}
Assistant:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 120,
            "temperature": 0.7
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return "I'm having trouble connecting to the AI right now."

    result = response.json()

    try:
        text = result[0]["generated_text"]
        return text.replace(prompt, "").strip()
    except:
        return "I'm here for you. Do you want to tell me what's going on?"
