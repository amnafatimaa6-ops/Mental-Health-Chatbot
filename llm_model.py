import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()  # load the .env file

HF_TOKEN = os.getenv("HF_TOKEN")  # read your API key

# ✅ Updated to a working, small Hugging Face model
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate_response(user_message, max_retries=2):
    """
    Generates a response from the Hugging Face LLM with timeout and retries.
    """
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

    for attempt in range(max_retries + 1):
        try:
            # 20s timeout to prevent hanging
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            response.raise_for_status()  # raise exception if HTTP error

            result = response.json()
            # Some models return list with "generated_text"
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            else:
                return str(result)

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(2)  # wait a bit before retry
                continue
            return "⚠️ The AI model is taking too long to respond. Please try again later."

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return f"⚠️ Error contacting the AI model: {e}"
