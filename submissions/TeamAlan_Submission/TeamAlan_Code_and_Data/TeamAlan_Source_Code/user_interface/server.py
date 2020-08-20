import os
import re
import random

import requests
from IPython import embed
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = os.path.join('..', '..', 'TeamAlan_Compiled_Models', 'distilbert-gsa-eula-opp')

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}

app = Flask(__name__)


def get_predictions(sequences, batch_size=4):
    encoded = tokenizer.batch_encode_plus(sequences, return_tensors="pt", pad_to_max_length=True)
    logits = model(**encoded)[0]
    results = torch.softmax(logits, dim=1).tolist()
    probs = [r[1] for r in results]
    return [{'score': prob, 'acceptable': prob < 0.5, 'text': sequence} for sequence, prob in zip(sequences, probs)]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def highlight_prediction(prediction):
    """
    prediction: dict(score=int, acceptable:bool)
    """
    score = prediction['score']
    text = prediction['text']
    if 0.0 <= score < 0.25:
        return "<p>{}</p>".format(text)
    elif 0.25 <= score < 0.50:
        return "<p style='background-color: peachpuff'>{}</p>".format(text)
    elif 0.50 <= score < 0.75:
        return "<p style='background-color: lightsalmon'>{}</p>".format(text)
    elif 0.75 <= score:
        return "<p style='background-color: tomato'>{}</p>".format(text)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        doc = request.form['doc']
        lines = doc.split('\n')
        lines = [line.strip().replace('\t', ' ').replace('\r', ' ').strip() for line in lines]
        lines = [line for line in lines if line != '']

        sentences = []
        for line in lines:
            sents = sent_tokenize(line)
            sentences = sentences + sents

        sentences = [s for s in sentences if len(s) > 12]
        sentences = [s[:512] for s in sentences]

        predictions = get_predictions(sentences)

        highlighted_predictions = [highlight_prediction(p) for p in predictions]

        return "\n".join(highlighted_predictions)

        # embed(using=False)
        # return '''
        # <pre>
        # {}
        # </pre>
        # '''.format()
    else:
        return '''
        <!doctype html>
        <title>EULA Clause Accaptence Checker</title>
        <h1>Copy and paste EULA text</h1>
        <form method=post action='/'>
          <textarea rows=50 cols=50 name=doc></textarea>
          <input type=submit value=submit>
        </form>
        '''

if __name__ == '__main__':
    app.run()
