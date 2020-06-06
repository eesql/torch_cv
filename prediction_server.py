from flask import Flask, jsonify
from flask import request
import commons
app = Flask(__name__)


@app.route('/')
def app_info():
    return 'Yet Another Image Classification Predict Service.'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        result = commons.get_prediction(image_bytes=img_bytes)
        return jsonify(result)
