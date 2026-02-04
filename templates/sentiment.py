import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("groq_api_key"))

def analyze_sentiment(text):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment analysis assistant."
            },
            {
                "role": "user",
                "content": f"""
Analyze the following paragraph sentence by sentence.

Classify EACH sentence strictly into:
1. Positive
2. Negative
3. Neutral

Rules:
- Do NOT summarize.
- Do NOT explain.
- Do NOT mix sentences.
- Keep original sentences exactly as written.
- Place each sentence under its correct category.

Return output in this format only:

Positive:
- sentence

Negative:
- sentence

Neutral:
- sentence

Paragraph:
{text}
"""
            }
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


# 🔹 User input
user_text = input("Enter your paragraph: ")

if user_text.strip() == "":
    print("No text entered!")
else:
    result = analyze_sentiment(user_text)
    print("\n Sentiment Analysis Result:\n")
    print(result)
