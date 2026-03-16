import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

def generate_response(messages):
    """
    messages: list of dicts: [{"role":"user","content":"Hi"}, ...]
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a kind mental health assistant. Always respond supportively."},
            *messages
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content
