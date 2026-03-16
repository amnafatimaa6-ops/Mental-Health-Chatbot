# llm_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- Load model and tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# ---------- Generate response ----------
def generate_response(user_message):
    """
    Generates a response using the local Flan-T5 small model.
    """
    # Build prompt
    prompt = f"You are a kind and supportive mental health assistant. Respond empathetically.\n{user_message}\nAssistant:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )

    # Decode output
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply
