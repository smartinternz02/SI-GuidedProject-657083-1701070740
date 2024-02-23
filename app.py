import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import os
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("rice.h5", custom_objects={'KerasLayer': hub.KerasLayer})

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/details")
def details():
    return render_template('details.html')

@app.route("/result", methods=['GET', 'POST'])
def result():
    if request.method == "POST":
      f = request.files['image']
      basepath = os.path.dirname(__file__)  # getting the current path i.e where app.py is present
    # print("current path", basepath)
      filepath = os.path.join(basepath, 'Data', 'val', f.filename)  # from anywhere in the system we #print("upload folder is", filepath)
      f.save(filepath)
      a2 = cv2.imread(filepath)
      a2 = cv2.resize(a2, (224, 224))
      a2 = np.array(a2)
      a2 = a2 / 255
      a2 = np.expand_dims(a2, 0)
      pred = model.predict(a2)
      pred = pred.argmax()
      df_labels = {
        'Arborio': 0,
        'Basmati': 1,
        'Ipsala': 2,
        'Jasmine': 3,
        'Karacadag': 4
    }
      for i, j in df_labels.items():
          if pred == j:
            prediction = i
            print(prediction)
    return render_template('result.html', prediction_text=prediction)

   

if __name__ == "__main__":
    app.run(debug=True)
