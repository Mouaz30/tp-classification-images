from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB').resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0

    rouge = float(img_array[:, :, 0].mean())
    vert = float(img_array[:, :, 1].mean())
    bleu = float(img_array[:, :, 2].mean())

    score = rouge - bleu

    if score > 0.02:
        result = f"Chien 🐶 ({round(score*200, 1)}%)"
    else:
        result = f"Chat 🐱 ({round((1-score)*100, 1)}%)"

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
