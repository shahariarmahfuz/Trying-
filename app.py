from flask import Flask, request, jsonify, session
import os
import google.generativeai as genai
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session management

# Configure API key (replace with your actual API key)
API_KEY = os.environ.get("GEMINI_API_KEY")
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

@app.route('/ask', methods=['GET', 'POST'])
def ask_question():
    if request.method == 'GET':
        question = request.args.get('q')
        image_file = None
    else:
        question = request.form.get('q')
        image_file = request.files.get('image')

    if not question and not image_file:
        return jsonify({"error": "No question or image provided"}), 400

    # Get or create user session
    if 'user_id' not in session:
        session['user_id'] = os.urandom(16).hex()  # Generate a unique user ID
    user_id = session['user_id']

    response = query_gemini_api(question, image_file, user_id)
    return jsonify(response)

def query_gemini_api(question=None, image_file=None, user_id=None):
    try:
        # Use user ID to manage chat history
        if user_id not in session:
            session[user_id] = []  # Initialize chat history for new user
        chat_history = session[user_id]

        # Start chat session with user's history
        chat_session = model.start_chat(history=chat_history)

        if question:
            chat_session.send_message(question)

        if image_file:
            image = Image.open(BytesIO(image_file.read()))
            response = chat_session.send_message(image=image)
        else:
            response = chat_session.send_message(question)

        # Update chat history in session
        session[user_id] = chat_session.history

        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
