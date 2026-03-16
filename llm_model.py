# llm_model.py
import os
import requests
from dotenv import load_dotenv
import time

load_dotenv()  # Load HF_TOKEN from .env

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen3.5-9B"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate_response(messages, max_retries=2):
    """
    messages: list of {"role": "user"/"assistant", "content": str}
    Generates AI response from Hugging Face Qwen 3.5-9B.
    Keeps it safe and retryable.
    """
    # Build instruction prompt
    prompt_lines = ["You are a kind and supportive mental health assistant. Respond empathetically.\n"]
    for m in messages[-4:]:  # last 4 messages for context
        role = "User" if m["role"] == "user" else "Assistant"
        prompt_lines.append(f"{role}: {m['content']}")
    prompt_lines.append("Assistant:")
    prompt_text = "\n".join(prompt_lines)

    payload = {
        "inputs": prompt_text,
        "parameters": {"max_new_tokens": 150, "temperature": 0.7}
    }

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=25)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                reply = result[0]["generated_text"]
                # Remove prompt echo if present
                if reply.startswith(prompt_text):
                    reply = reply[len(prompt_text):].strip()
                return reply
            else:
                return str(result)
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return "⚠️ The AI is taking too long to respond. Please try again later."
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return f"⚠️ Error contacting the AI model: {e}"
