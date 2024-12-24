import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/": {"origins": "*"}})

groq_api_key = os.environ.get("GROQ_API_KEY")
model = "llama3-8b-8192"
client = ChatGroq(groq_api_key=groq_api_key, model_name=model)
system_prompt = "your prompt"

memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

def get_response(text):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )
    conversation = LLMChain(
        llm=client,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    return conversation.predict(human_input=text)

@app.route("/response", methods=["POST"])
def response():
    try:
        data = request.get_json()
        query = data.get("query")
        if not query:
            return jsonify({"error": "Query parameter is missing."}), 400
        response = get_response(query)
        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"api_key_set": bool(groq_api_key), "model": model})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT variable if set
    app.run(host="0.0.0.0", port=port, debug=True)


