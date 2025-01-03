import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyB-i9YL2_PzzmEIMBZTNoOc-wTJrqwGjcA"
genai.configure(api_key=GOOGLE_API_KEY)

result = genai.embed_content(
        model="models/text-embedding-004",
        content="What is the meaning of life?")

print(str(result['embedding']))