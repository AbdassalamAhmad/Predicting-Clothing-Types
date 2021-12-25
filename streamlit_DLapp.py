import numpy as np
import streamlit as st
from PIL import Image, ImageOps# to read the image 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input
import os
from skimage.transform import resize
#Classes to predict
classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

#loading the model
model = keras.models.load_model("xception_v5_1_10_0.889.h5")


def main():
    uploaded_file = st.file_uploader("Choose a picture to predict its class", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Sunrise by the mountains')
        if st.button("Predict Class"):
            result=predict_class(image)
            st.write(result)
        #st.success(result)
    


def predict_class(image):
    #loading the test picture
    #path = f'./{uploaded_file.name}'
    #img = load_img(path, target_size=(299, 299))
 
    image = np.asarray(image)/255
    
    my_image= resize(image, (299,299)).reshape((1, 299*299*3)).T
    #preprocessing the test picture
    x = np.array(my_image)
    X = np.array([x])
    X = preprocess_input(X)

    #predicting
    pred = model.predict(X)
    pre_result = dict(zip(classes, pred[0]))
    max_key = max(pre_result, key=pre_result.get)
    result = "your picture contains {} ".format(max_key)
    return (result)




if __name__=='__main__':
    main()

    
