from email import message
from re import template
from urllib import response
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import get_response
from chatbot import predict_class
import json
app = Flask(__name__)
CORS(app)

intents = json.loads(open('intents.json').read())

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    ints = predict_class(text)
    response = get_response(ints, intents)
    message = {"answer": response}
    return jsonify(message)


if __name__ =="__main__":
    app.run(debug=True)
