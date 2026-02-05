import os
import json
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


st.title("Sentiment Analysis")

user_text = st.text_area("Enter your paragraph:")

def analyze_sentiment(text):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """
You are a sentiment analysis assistant.

Analyze each sentence separately and return output strictly in JSON format like this:

{
  "Positive": ["sentence1", "sentence2"],
  "Negative": ["sentence1"],
  "Neutral": ["sentence1"]
}

Do not return anything except valid JSON.
"""
            },
            {
                "role": "user",
                "content": f"Analyze this paragraph:\n{text}"
            }
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = analyze_sentiment(user_text)

        try:
            json_result = json.loads(result)
            st.json(json_result)  # Proper JSON display
        except:
            st.error("Model did not return valid JSON.")
            st.write(result)
