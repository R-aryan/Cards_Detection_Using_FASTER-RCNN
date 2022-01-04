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
    try:
        image = request.json['image']
        image_name = "input_image_" + str(datetime.now()).split(':')[-1] + ".jpg"
        cards_detector.settings.logger.info("Received Post Request for inference--!!")
        decodeImage(image, image_name, cards_detector.settings.INPUT_IMAGE_PATH)
        cards_detector.settings.logger.info("Image stored in directory -- " + cards_detector.settings.INPUT_IMAGE_PATH,
                                            "with image name--" + str(image_name))
        result = cards_detector.predict(cards_detector.settings.INPUT_IMAGE_PATH + image_name)
        return jsonify(result)
    except BaseException as ex:
        cards_detector.settings.logger.error("Following Error occurred while inference---!!", str(ex))
        return jsonify(str(ex))


if __name__ == "__main__":
    cards_detector = CardsDetector()
    port = 9000
    app.run(host='127.0.0.1', port=port)
