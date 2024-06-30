from flask import Flask, request, jsonify
import os
import google.generativeai as genai
from PIL import Image
import io
import base64

app = Flask(__name__)

# Configure API key
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

@app.route('/ask', methods=['GET'])
def ask_question():
    question = request.args.get('q')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    response = query_gemini_api(question)
    return jsonify(response)

@app.route('/ask_image', methods=['POST'])
def ask_image_question():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        image = Image.open(file)
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        response = query_gemini_image_api(img_str)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def query_gemini_api(question):
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(question)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

def query_gemini_image_api(img_str):
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(img_str)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
