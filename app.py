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

def upload_to_gemini(image_data, mime_type="image/jpeg"):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(io.BytesIO(image_data), mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
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
    if 'file' not in request.files or 'q' not in request.form:
        return jsonify({"error": "File or question not provided"}), 400
    
    file = request.files['file']
    question = request.form['q']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        image = Image.open(file)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = buffered.getvalue()
        
        gemini_file = upload_to_gemini(img_str)
        response = query_gemini_image_api(gemini_file.uri, question)
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

def query_gemini_api(question):
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(question)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

def query_gemini_image_api(file_uri, question):
    try:
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        {"type": "file", "uri": file_uri},
                    ],
                },
            ]
        )
        response = chat_session.send_message(question)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
