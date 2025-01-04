import google.generativeai as genai

result = genai.embed_content(
        model="models/text-embedding-004",
        content="What is the meaning of life?")

print(str(result['embedding']))