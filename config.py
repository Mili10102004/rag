import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing GOOGLE_API_KEY in .env file")
if not SERPER_API_KEY:
    raise ValueError("❌ Missing SERPER_API_KEY in .env file")
