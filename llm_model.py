from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_response(messages):

    # Build prompt from last few messages
    prompt = "You are a kind and supportive mental health assistant.\n\n"

    for m in messages[-4:]:
        role = "User" if m["role"] == "user" else "Assistant"
        prompt += f"{role}: {m['content']}\n"

    prompt += "Assistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # remove prompt echo
    reply = reply.replace(prompt, "").strip()

    return reply
