import os
from flask import Flask, request, Response, jsonify, send_from_directory, abort, make_response
from flask_cors import CORS, cross_origin
from imagesimilarity import image_similarity


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/similarity', methods=['POST'])
def similarity():
    real_image = request.files["realimage"]
    real_image_name = real_image.filename
    real_image.save(os.path.join(os.getcwd(), real_image_name))


    similar_image = request.files["similarimage"]
    similar_image_name = similar_image.filename
    similar_image.save(os.path.join(os.getcwd(), similar_image_name))

    result = image_similarity(real_image_name, similar_image_name)

    return str(result)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)