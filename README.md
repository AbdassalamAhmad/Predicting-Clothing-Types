# Predicting-Clothing-Types
Predicting 10 different clothing types using Xception pre-trained model from Keras library.
It is reimplemented version from [lesson 8-deep learning](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/08-deep-learning) held by [DataTalksClub](https://datatalks.club/).

## About the Dataset
#### Data Source
I use the dataset from here:
https://github.com/alexeygrigorev/clothing-dataset-small

#### Dataset Information
This is a small dataset contains 10 different clothing types (dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt, t-shirt).

## Short Description of the Files
1. []()

2. [streamlit_DLapp.py](https://github.com/AbdassalamAhmad/Predicting-Clothing-Types/blob/main/streamlit_DLapp.py) It deploy the trained model to streamlit cloud 

3. [xception_v5_1_10_0.889.h5](https://github.com/AbdassalamAhmad/Predicting-Clothing-Types/blob/main/xception_v5_1_10_0.889.h5) - Best model from training saved in this binary format to load it easily.


4. [Pipfile](https://github.com/AbdassalamAhmad/Predicting-Clothing-Types/blob/main/Pipfile) and [Pipfile.lock](https://github.com/AbdassalamAhmad/Predicting-Clothing-Types/blob/main/Pipfile.lock) - Python package dependencies, in the pipfile you can find all necessary librares and packages to be able to run the scripts with no problem.
## How to run this model
1. open this [link](https://share.streamlit.io/abdassalamahmad/predicting-clothing-types/main/streamlit_DLapp.py)
2. Upload an image from test dataset or any image from your device that has one clothing type.
3. click on Predict Class button.

Note: watch this [video](https://vimeo.com/660125714/22a8430d76) to see the model in action


## How to reproduce this model
1. clone this repo to get all the code.
2. clone the dataset using this command
```py
!git clone git@github.com:alexeygrigorev/clothing-dataset-small.git
```
3. install pipenv -which is a packaging tool that will help installing all dependencies- , use this command on your terminal.
```py
pip install pipenv
```
4. install all dependencies using pipenv by typing this command in your terminal **inside your cloned repo folder** 
```py
pipenv install
```
5. Deploying the app locally or on the web
5.1. Locally: open the terminal and use this command
```py
streamlit run streamlit_DLapp.py
```
5.2. on the web: check the [documentation](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app) from official website.  

## Note
If you like my project, I appreciate you starring this repo. Please feel free to fork the content and contact me if you have any questions.

[my linkedIn account](https://www.linkedin.com/in/abdassalam-ahmad/)
