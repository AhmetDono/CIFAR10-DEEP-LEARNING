import re
#from flask_ngrok import run_with_ngrok
from flask import Flask, render_template ,request
from keras.models import load_model
import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



app = Flask(__name__)
#un_with_ngrok(app)


@app.route('/')
def text():
  return render_template("ana.html")

@app.route('/prediction',methods=["POST"])
def prediction():
  img = request.files['img']
  img.save("img.png")

  loaded_model=load_model("cifar10.h5")
  image=cv2.imread("img.png")
  r_image = cv2.resize(image, dsize=(32, 32))
  n_image=keras.utils.normalize(r_image,axis=1)
  r2_image = n_image.reshape((1,32,32,3))
  predict= loaded_model.predict(r2_image)
  sonuc=np.argmax(predict)
  return render_template("prediction.html",data=sonuc) 

app.run()
