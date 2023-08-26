from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import pyttsx3

app = Flask(__name__)

max_length = 32
tokenizer = load(open("tokenizer.p", "rb"))
model = load_model('models/model_1.h5')
xception_model = Xception(include_top=False, pooling="avg")

#Function to help generate caption from image

def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text



def generate_caption(image_path):
    photo = extract_features(image_path, xception_model)
    description = generate_desc(model, tokenizer, photo, max_length)
    description = ' '.join(description.split(' ')[:-1])
    description = ' '.join(description.split(' ')[1:])
    # Read the generated caption aloud
    engine = pyttsx3.init()
    engine.say(description)
    print("reading")
    engine.runAndWait()
    return description

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        if image:
            image_path = f"static/images/{image.filename}"
            image.save(image_path)
            caption = generate_caption(image_path)
            return render_template("index.html", image_path=image_path, caption=caption)
    return render_template("index.html", caption=None)

if __name__ == "__main__":
    app.run(debug=True)
