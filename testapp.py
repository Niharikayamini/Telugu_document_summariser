from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator


app = Flask(__name__)
CORS(app)

@app.route('/summarize_translate_page')
def summarize_translate_page():
    return render_template("summarize_translate.html")



@app.route('/summarize-translate', methods=['POST'])
def summarize_translate():
    if summarizer is None:
        return jsonify({"error": "Summarization model not loaded"}), 500

    try:
        input_text = request.json.get("text", "")
        if not input_text:
            return jsonify({"error": "No input text provided"}), 400

        summary_output = summarizer(
            input_text,
            max_length=150,
            min_length=30,
            do_sample=False,
            early_stopping=True
        )
        telugu_summary = summary_output[0]["summary_text"]
        english_translation = GoogleTranslator(source='auto', target='en').translate(telugu_summary)

        return jsonify({
            "telugu_summary": telugu_summary,
            "english_summary": english_translation
        })

    except Exception as e:
        logger.exception("Summarize-translate failed.")
        return jsonify({"error": str(e)}), 500

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Akhilathirumalaraju/telugu_summary"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    logger.info("Telugu summarization model loaded successfully.")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    summarizer = None

@app.route('/')
def summarizer_home():
    return render_template("summarizer.html")

@app.route('/voice-summarizer')
def voice_summarizer():
    return render_template("voice.html")

@app.route('/summarize', methods=['POST'])
def summarize_text():
    if summarizer is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        input_text = request.json.get("text", "")
        if not input_text:
            return jsonify({"error": "No input text provided"}), 400

        summary_output = summarizer(
            input_text,
            max_length=150,
            min_length=30,
            do_sample=False,
            early_stopping=True
        )
        summary = summary_output[0]["summary_text"]
        return jsonify({"summary": summary})

    except Exception as e:
        logger.exception("Summarization failed.")
        return jsonify({"error": str(e)}), 500


@app.route('/summarize-voice', methods=['POST'])
def summarize_voice_text():
    if summarizer is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        input_text = request.json.get("text", "")
        if not input_text:
            return jsonify({"error": "No input text provided"}), 400

        summary_output = summarizer(
            input_text,
            max_length=150,
            min_length=30,
            do_sample=False,
            early_stopping=True
        )
        summary = summary_output[0]["summary_text"]
        return jsonify({"summary": summary})

    except Exception as e:
        logger.exception("Summarization failed.")
        return jsonify({"error": str(e)}), 500
    

@app.route('/career')
def career_home():
    return render_template("hometest.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        res = result.to_dict(flat=True)
        arr = [value for value in res.values()]
        data = np.array(arr).reshape(1, -1)

        loaded_model = pickle.load(open("careerlast.pkl", 'rb'))
        predictions = loaded_model.predict(data)
        pred = loaded_model.predict_proba(data)
        pred = pred > 0.05

        i = 0
        j = 0
        index = 0
        res = {}
        final_res = {}
        while j < 17:
            if pred[i, j]:
                res[index] = j
                index += 1
            j += 1

        index = 0
        for key, values in res.items():
            if values != predictions[0]:
                final_res[index] = values
                index += 1

        jobs_dict = {
            0: 'AI ML Specialist', 1: 'API Integration Specialist', 2: 'Application Support Engineer',
            3: 'Business Analyst', 4: 'Customer Service Executive', 5: 'Cyber Security Specialist',
            6: 'Data Scientist', 7: 'Database Administrator', 8: 'Graphics Designer', 9: 'Hardware Engineer',
            10: 'Helpdesk Engineer', 11: 'Information Security Specialist', 12: 'Networking Engineer',
            13: 'Project Manager', 14: 'Software Developer', 15: 'Software Tester', 16: 'Technical Writer'
        }

        return render_template("testafter.html", final_res=final_res, job_dict=jobs_dict, job0=predictions[0])

    return "Only POST allowed"

if __name__ == '__main__':
    app.run(debug=True)
