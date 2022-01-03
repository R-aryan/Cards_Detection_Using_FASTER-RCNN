from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from datetime import datetime

from services.card_detector.application.ai.inference.prediction import CardsDetector
from services.card_detector.application.ai.utils.utils import decodeImage

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    image_name = "input_image.jpg"
    image_path = "../images/input_images/"
    decodeImage(image, image_name, image_path)
    result = cards_detector.predict(cards_detector.settings.INPUT_IMAGE_PATH + image_name)
    return jsonify(result)


if __name__ == "__main__":
    cards_detector = CardsDetector()
    port = 9000
    app.run(host='127.0.0.1', port=port)
