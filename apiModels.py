import os
import requests
import openai
import replicate

def setup_openai_client():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai.OpenAI(api_key=openai_api_key)

def query_openai(prompt, model="text-davinci-002"):
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 1.0
    }
    response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['text']
    else:
        return "Error in querying OpenAI API"

# Setup Replicate client
def get_replicate_client():
    return replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Query Llama-2-70b-chat model
def query_llama(prompt):
    client = get_replicate_client()
    input = {
        "top_p": 1,
        "prompt": prompt,
        "temperature": 0.5,
        "system_prompt": "You are a helpful, respectful, and honest assistant...",
        "max_new_tokens": 500
    }
    output = client.run(
        "meta/llama-2-70b-chat",
        input=input
    )
    return output

# Query Falcon-40b-instruct model
def query_falcon(prompt):
    client = get_replicate_client()
    input = {
        "prompt": prompt,
        "temperature": 1
    }
    output = client.run(
        "joehoover/falcon-40b-instruct:latest",
        input=input
    )
    return output
