# llm_model.py
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import torch

# Load processor and model
processor = AutoProcessor.from_pretrained("Qwen/Qwen-3.5-9B")
model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen-3.5-9B")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_response(messages):
    """
    messages: list of {"role": "user"/"assistant", "content": str}
    Generates AI response considering last few messages.
    """
    # Build prompt
    prompt_lines = ["You are a kind and supportive mental health assistant. Respond empathetically.\n"]
    for m in messages[-4:]:  # last 4 messages for context
        role = "User" if m["role"] == "user" else "Assistant"
        prompt_lines.append(f"{role}: {m['content']}")
    prompt_lines.append("Assistant:")
    prompt_text = "\n".join(prompt_lines)

    # Prepare input tensors
    inputs = processor.apply_chat_template(
        [{"role":"user","content":[{"type":"text","text":prompt_text}]}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    # Generate response
    outputs = model.generate(**inputs, max_new_tokens=150)
    
    # Decode generated text
    reply = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    return reply.strip()
