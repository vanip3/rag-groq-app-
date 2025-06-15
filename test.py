import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key loaded: {api_key}")
print(f"API Key starts with 'gsk_': {api_key.startswith('gsk_') if api_key else False}")
print(f"API Key length: {len(api_key) if api_key else 0}")