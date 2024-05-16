from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/static')


app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG"]

# Load tokenizer and model
max_length = 32
tokenizer = load(open("tokenizer.p", "rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

def extract_features(filename, model):
    try:
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.array(image)
        if image.shape[2] == 4:
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension are correct.")
        return None

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text[6:-3] if 'end' in in_text else in_text

def allowed_image(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1]
    return ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/service', methods=['GET', 'POST'])
def service():
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return render_template('service.html', error='No file selected')

        if allowed_image(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('static', 'uploads', filename)
            file.save(file_path)

            # Process image and generate caption
            photo_features = extract_features(file_path, xception_model)
            if photo_features is not None:
                description = generate_desc(model, tokenizer, photo_features, max_length)
                return render_template('service1.html', image_file=f'uploads/{filename}', result=description)
            else:
                return render_template('service1.html', error='Error processing image')

        else:
            return render_template('service1.html', error='Invalid file format')

    return render_template('service1.html')

if __name__ == '__main__':
    app.run(debug=True)

