# llm_model.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # load the .env file

HF_TOKEN = os.getenv("HF_TOKEN")  # read your API key

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate_response(user_message):
    prompt = f"""
You are a kind and supportive mental health assistant.
Respond empathetically.

User: {user_message}
Assistant:
"""
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 120, "temperature": 0.7}
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "The AI model is currently unavailable. Try again later."
