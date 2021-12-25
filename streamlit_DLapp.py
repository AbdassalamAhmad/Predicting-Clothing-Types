import numpy as np
import streamlit as st
from PIL import Image# to read & resize the image 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input

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
    st.title("Predicting Clothing Types Project")
    uploaded_file = st.file_uploader("Upload a picture to predict its class", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        st.title("Here is the image you've uploded")
        image = Image.open(uploaded_file)
        st.image(image)
        if st.button("Predict Class"):
            result=predict_class(image)
            st.write(result)
        #st.success(result)
    

def predict_class(image):
    #preprocessing the test picture
    #image sizing
    test_image = image.resize((299,299))
    image_array = np.asarray(test_image)
    X = np.array([image_array])#because the next function expects a list.
    X = preprocess_input(X)

    #predicting
    pred = model.predict(X)
    pre_result = dict(zip(classes, pred[0]))
    max_key = max(pre_result, key=pre_result.get)
    result = "your picture contains {} ".format(max_key)
    return (result)

if __name__=='__main__':
    main()

    
