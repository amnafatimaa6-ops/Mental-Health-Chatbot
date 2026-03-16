# llm_model.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_response(messages):
    # get latest user message
    user_message = messages[-1]["content"]

    prompt = (
        "You are a compassionate mental health support assistant. "
        "Respond with empathy and encouragement.\n\n"
        f"User: {user_message}\nAssistant:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # remove prompt if echoed
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    # fallback if model returns empty
    if len(text.strip()) < 3:
        text = (
            "I'm really sorry you're feeling this way. "
            "If you want, you can tell me what’s been making today hard."
        )

    return text
