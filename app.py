from flask import Flask, request, jsonify
import os
import google.generativeai as genai

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

@app.route('/ask_with_image', methods=['POST'])
def ask_question_with_image():
    question = request.form.get('q')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    image_path = os.path.join('/tmp', image.filename)
    image.save(image_path)

    response = query_gemini_api_with_image(question, image_path, image.mimetype)
    return jsonify(response)

def query_gemini_api(question):
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(question)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

def query_gemini_api_with_image(question, image_path, mime_type):
    try:
        file = genai.upload_file(image_path, mime_type=mime_type)
        chat_session = model.start_chat(history=[
            {
                "role": "user",
                "parts": [file.uri],
            },
            {
                "role": "user",
                "parts": [question],
            },
        ])
        response = chat_session.send_message(question)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
