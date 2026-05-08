from google import genai

API_KEY = "AIzaSyBYz2YhmQ9fSglIIXk5fowohNVyf5yKPnM"

client = genai.Client(api_key=API_KEY)
response = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents="hi, reply in one word"
)
print("OK:", response.text)