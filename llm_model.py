from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_response(messages):

    # Use ONLY the latest user message
    user_message = messages[-1]["content"]

    prompt = f"""
You are a supportive and kind mental health assistant.
Respond with empathy and encouragement.

User: {user_message}
Assistant:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # If model echoes prompt, remove it
    response = response.replace(prompt, "").strip()

    # Fallback if empty
    if len(response) < 3:
        response = "I'm really sorry you're feeling this way. Do you want to tell me what happened today?"

    return response
