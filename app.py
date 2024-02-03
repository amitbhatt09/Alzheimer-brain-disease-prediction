from pyexpat import model
from unicodedata import name
from flask import Flask, redirect, url_for, request, render_template
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
import numpy as np

app=Flask(__name__)

MODEL_PATH="model\\alzheimer_model.h5"
CLASS_NAMES=['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
model=tf.keras.models.load_model(MODEL_PATH)

def model_predict(file_path,model):
    image=tf.keras.preprocessing.image.load_img(file_path)
    image=tf.image.resize(image,(224,224))
    input_array=tf.keras.preprocessing.image.img_to_array(image)
    input_array=np.array([input_array])
    pred=model.predict(input_array)
    res=np.argmax(pred)
    return CLASS_NAMES[res]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['file']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            basepath,
            'uploads',
            secure_filename(f.filename)
        )
        f.save(file_path)
        preds=model_predict(file_path,model)
        return preds
    return None

if __name__=="__main__":
    app.run(debug=True)