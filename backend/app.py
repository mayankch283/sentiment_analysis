from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from transformers import pipeline
from deepface import DeepFace
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import logging

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up logging
logging.basicConfig(level=logging.DEBUG)

sentiment_pipeline = pipeline('sentiment-analysis')
emotion_pipeline = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

def analyze_text(text):
    sentiment = sentiment_pipeline(text)
    emotion = emotion_pipeline(text)
    return sentiment, emotion

def analyze_image(image_path):
    analysis = DeepFace.analyze(image_path, actions=['emotion'])
    return analysis

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    return text

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'filename': filename}), 200
    except Exception as e:
        app.logger.error(f"Error uploading file: {e}")
        return jsonify({'error': 'File upload failed'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_content():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            result = analyze_image(file_path)
        elif filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            sentiment, emotion = analyze_text(text)
            result = {'sentiment': sentiment, 'emotion': emotion}
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"Error analyzing content: {e}")
        return jsonify({'error': 'Content analysis failed'}), 500

@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    try:
        data = request.get_json()
        url = data.get('url')
        text = extract_text_from_url(url)
        sentiment, emotion = analyze_text(text)
        return jsonify({'sentiment': sentiment, 'emotion': emotion}), 200
    except Exception as e:
        app.logger.error(f"Error analyzing URL: {e}")
        return jsonify({'error': 'URL analysis failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
