from flask import Flask, render_template, request, jsonify
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import re

app = Flask(__name__)

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# Set languages
src_lang, tgt_lang = "en_XX", "ta_IN"

# Define a set of technical terms (expand this list as needed)
technical_terms = {
    "machine translation", "natural language processing", "nlp", "transformer architecture",
    "machine learning", "deep learning", "artificial intelligence", "ai", "neural network",
    # ... (keep the rest of your technical terms)
}

def preprocess_text(text):
    for term in sorted(technical_terms, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        text = pattern.sub(lambda m: f"<keep>{m.group()}</keep>", text)
    return text

def postprocess_text(text):
    return re.sub(r'<keep>(.*?)</keep>', r'<strong>\1</strong>', text)

def translate(text, src_lang=src_lang, tgt_lang=tgt_lang):
    preprocessed_text = preprocess_text(text)
    inputs = tokenizer(preprocessed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=512,
        num_beams=5,
        length_penalty=1.0,
        early_stopping=True
    )
    
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return postprocess_text(translated_text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    text = request.json['text']
    translated = translate(text)
    return jsonify({'translation': translated})

if __name__ == '__main__':
    app.run(debug=True)