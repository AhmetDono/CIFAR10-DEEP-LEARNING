from keras.models import load_model
import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

loaded_model=load_model("cifar10.h5")

img=cv2.imread("ucak2.png")
res = cv2.resize(img, dsize=(32, 32))
son=keras.utils.normalize(res,axis=1)


sample = son.reshape((1,32,32,3))
predict_x = loaded_model.predict(sample)
a=np.argmax(predict_x)
print(a)


#loaded_model=load_model("cifar10.h5")

#img=cv2.imread("deneme2.png")
#res = cv2.resize(img, dsize=(32, 32))
#son=keras.utils.normalize(res,axis=1)


#sample = son.reshape((1,32,32,3))
#predict_x = loaded_model.predict(sample)
#a=np.argmax(predict_x)
#print(a)
