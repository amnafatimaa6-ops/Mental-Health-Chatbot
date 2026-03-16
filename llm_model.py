# llm_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_response(user_message):
    """
    Generates a response using Flan-T5 small model.
    Sends only latest user input for reliable responses.
    """
    prompt = f"You are a kind and supportive mental health assistant. Respond empathetically.\nUser: {user_message}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove repeating user input if the model echoes it
    if reply.lower().startswith(user_message.lower()):
        reply = reply[len(user_message):].strip()

    return reply
