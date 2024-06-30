from flask import Flask, request, jsonify, session
import os
import google.generativeai as genai
import logging
from datetime import timedelta

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(minutes=30)  # Set session timeout to 30 minutes
logging.basicConfig(level=logging.DEBUG)

class ApiKeyNotSetError(Exception):
    pass

# Configure API key
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ApiKeyNotSetError("API key is not set. Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=API_KEY)

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

@app.route('/ask', methods=['GET'])
def ask_question():
    question = request.args.get('q')
    user_id = request.args.get('user_id')

    if not question or not user_id:
        return jsonify({"error": "Question and user_id are required"}), 400

    response = query_gemini_api(question, user_id)
    return jsonify(response)

def query_gemini_api(question, user_id):
    try:
        # Use user_id to manage chat history
        if user_id not in session:
            session[user_id] = []
        chat_history = session[user_id]

        # Start chat session with user's history
        chat_session = model.start_chat(history=chat_history)

        # Send the question to the model
        response = chat_session.send_message(question)

        # Update chat history in session
        session[user_id] = chat_session.history

        return {"response": response.text}
    except ApiKeyNotSetError as e:
        app.logger.error(f"API Key Error: {str(e)}")
        return jsonify({"error": "API key is not set or invalid"}), 500
    except Exception as e:
        app.logger.error(f"Error in query_gemini_api: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
