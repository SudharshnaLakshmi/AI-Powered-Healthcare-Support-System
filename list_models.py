import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("API key is missing or invalid. Please check your .env file.")

# Configure Generative AI API
genai.configure(api_key=API_KEY)

try:
    models = genai.list_models()
    print("API key is valid! Models available:")
    for model in models:
        print(f"- {model.name}")
except Exception as e:
    print(f"Error: {e}")
