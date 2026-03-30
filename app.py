from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Charge le vrai modèle CNN
print("Chargement du modèle CNN...")
model = tf.keras.models.load_model('cnn_model.h5')
print("Modèle chargé ✅")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB').resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = f"Chien 🐶 ({round(float(prediction)*100, 1)}%)"
    else:
        result = f"Chat 🐱 ({round((1-float(prediction))*100, 1)}%)"

    return jsonify({'result': result})


if __name__ == '__main__':
    # host='0.0.0.0' permet l'accès depuis ton téléphone
    app.run(debug=True, host='0.0.0.0', port=5000)
