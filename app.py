# from label import main

from omr.omr import main
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin

import PIL.Image as Image
import numpy as np
import io
import base64


app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route('/', methods=['POST', 'OPTIONS', 'GET'])
@cross_origin(supports_credentials=True)
def home():
    req = request.get_json()['image']

    base64_decoded = base64.b64decode(req)

    image = Image.open(io.BytesIO(base64_decoded))
    image_np = np.array(image)
    # image_np = np.dot(image_np[...,:3], [0.299, 0.587, 0.114])

    omr = main(image_np)

    res = np.array(omr[0])
    im = Image.fromarray(res.astype('uint8'))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    img = base64.b64encode(rawBytes.read())

    return img
    # return {"image": img, "txt": omr[1]}


if __name__ == '__main__':
    app.run(threaded=True, port=5000, debug=True)