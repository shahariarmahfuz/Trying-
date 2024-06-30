from flask import Flask, request, jsonify, session
import os
import google.generativeai as genai
import logging
from datetime import timedelta

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(minutes=30)  # Session timeout

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ApiKeyNotSetError(Exception):
    pass

# Configure API key
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ApiKeyNotSetError("API key is not set. Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=API_KEY)

# Create the model (Gemini Pro is recommended if available)
generation_config = {
    "temperature": 0.8,  # Adjust for creativity (0 = deterministic, 1 = most creative)
    "top_p": 0.95,       # Nucleus sampling for better quality responses
    "top_k": 40,         # Consider top-k most likely tokens
    "max_output_tokens": 1024, # Limit response length
}

model = genai.GenerativeModel(
    model_name="models/chat-bison-001",  # Gemini Pro model
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
        # Manage chat history per user
        if user_id not in session:
            session[user_id] = []
        chat_history = session[user_id]
        
        app.logger.info(f"Current chat history for user {user_id}: {chat_history}")

        # Start/continue chat session
        chat_session = model.start_chat(history=chat_history, context="You are a helpful AI assistant.")

        # Get response from Gemini
        response = chat_session.send_message(question)
        
        app.logger.info(f"Response from Gemini: {response.text}")

        # Update chat history
        session[user_id] = chat_session.history
        
        app.logger.info(f"Updated chat history for user {user_id}: {session[user_id]}")

        return {"response": response.text}  # Ensure response is JSON serializable

    except ApiKeyNotSetError as e:
        app.logger.error(f"API Key Error: {str(e)}")
        return jsonify({"error": "API key is not set or invalid"}), 500
    except Exception as e:
        app.logger.error(f"Error in query_gemini_api: {str(e)}")
        return jsonify({"error": f"An error occurred while processing your request: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
