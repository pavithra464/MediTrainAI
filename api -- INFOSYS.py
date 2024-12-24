import os
from dotenv import load_dotenv
from groq import Groq

# import requests
from flask import Flask, request, jsonify
# from flask_cors import CORS

load_dotenv()

app = Flask(__name__)

client = Groq(
    api_key=os.environ.get("API_KEY"),
)


def get_reponse(text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a Doctor also give some advice to regular check up on health.
            Provide response in consistent manner around 50 words.
            """,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content


@app.route("/response", methods=["POST"])
def response():
    try:
        data = request.getjson()
        query = data.get("query")
        print(query)
        response = get_reponse(query)
        return jsonify({"response": response})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
