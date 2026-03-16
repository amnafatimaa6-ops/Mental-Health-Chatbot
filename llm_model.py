import requests
import os

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def generate_response(messages):

    user_message = messages[-1]["content"]

    prompt = f"""
You are a compassionate mental health support assistant.
Respond kindly and empathetically.

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
        return "⚠️ The AI model is currently unavailable. Please try again."

    result = response.json()

    try:
        text = result[0]["generated_text"]
        return text.replace(prompt, "").strip()
    except:
        return "I'm here for you. Do you want to tell me what's been bothering you today?"
